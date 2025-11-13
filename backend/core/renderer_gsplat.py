"""
GaussianRenderer2D_GSplat: gsplat-based high-performance renderer

Uses gsplat library for CUDA-accelerated Gaussian splatting
Achieves 50-100 FPS on RTX 3090 through:
- Native orthographic projection
- Optimized CUDA kernels
- Minimal CPU-GPU data transfer
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from .gaussian import Gaussian2D
from .debug_visualizer import DebugVisualizer
import cv2

try:
    from gsplat import rasterization

    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("[GSplat Renderer] Warning: gsplat not available, use pip install gsplat")


class GaussianRenderer2D_GSplat:
    """
    gsplat-based 2D Gaussian Splatting renderer

    High-performance renderer using native orthographic projection
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        background_color: np.ndarray = np.array([1.0, 1.0, 1.0]),
        device: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Args:
            width: Rendering width (pixels)
            height: Rendering height (pixels)
            background_color: Background RGB [0, 1]
            device: 'cuda' or 'cpu', auto-detect if None
            debug_mode: Enable debug visualization
        """
        if not GSPLAT_AVAILABLE:
            raise ImportError("gsplat not available. Install with: pip install gsplat")

        # Device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.width = width
        self.height = height

        print(f"[GSplat Renderer] Initialized on device: {self.device}")
        if self.device.type == "cuda":
            print(f"[GSplat Renderer] GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"[GSplat Renderer] CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        # Background color
        self.background_color = torch.tensor(
            background_color, dtype=torch.float32, device=self.device
        )

        # World space bounds
        self.world_min = torch.tensor(
            [-1.0, -1.0], dtype=torch.float32, device=self.device
        )
        self.world_max = torch.tensor(
            [1.0, 1.0], dtype=torch.float32, device=self.device
        )

        # Setup orthographic camera
        self._setup_camera()

        # Debug mode
        self.debug_mode = debug_mode
        self.debug_visualizer = None
        if self.debug_mode:
            # Convert torch tensors to numpy for DebugVisualizer
            world_min_np = self.world_min.cpu().numpy()
            world_max_np = self.world_max.cpu().numpy()
            self.debug_visualizer = DebugVisualizer(
                width=width, height=height,
                world_min=world_min_np, world_max=world_max_np
            )
            print(f"[GSplat Renderer] Debug mode enabled")

    def _setup_camera(self):
        """Setup orthographic camera for 2D rendering"""
        # World-to-Camera (w2c) matrix - gsplat uses OpenCV convention
        # Camera at (0, 0, 5) in world space, looking down -Z axis
        # Gaussians at z=0 are transformed to z=+5 in camera space
        # Y-axis flipped to match screen coordinates (Y-down)
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],  # Flip Y-axis for correct orientation
                [0.0, 0.0, 1.0, 5.0],  # w2c: Translate world +5 in Z to camera space
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.viewmat = self.viewmat.unsqueeze(0)  # [1, 4, 4] for batch

        # Orthographic projection matrix
        # Maps world space [-1, 1] x [-1, 1] to pixel space [0, width] x [0, height]
        self.K = self._compute_ortho_projection()

    def _compute_ortho_projection(self) -> torch.Tensor:
        """
        Compute orthographic projection matrix

        Returns:
            [1, 3, 3] projection matrix
        """
        # World space dimensions
        world_width = self.world_max[0] - self.world_min[0]
        world_height = self.world_max[1] - self.world_min[1]

        # Scale factors: world -> pixel
        fx = self.width / world_width
        fy = self.height / world_height

        # Principal point (center of image)
        cx = self.width / 2.0
        cy = self.height / 2.0

        # Intrinsic matrix for orthographic projection
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )

        return K.unsqueeze(0)  # [1, 3, 3]

    def set_world_bounds(self, world_min: np.ndarray, world_max: np.ndarray):
        """Set world space bounds"""
        self.world_min = torch.tensor(
            world_min, dtype=torch.float32, device=self.device
        )
        self.world_max = torch.tensor(
            world_max, dtype=torch.float32, device=self.device
        )

        # Recompute projection matrix
        self.K = self._compute_ortho_projection()

        # Update debug visualizer bounds if exists
        if self.debug_visualizer:
            self.debug_visualizer.world_min = world_min
            self.debug_visualizer.world_max = world_max

    def world_to_pixel(self, world_pos: np.ndarray) -> np.ndarray:
        """World coordinates to pixel coordinates (CPU)"""
        world_size = self.world_max.cpu().numpy() - self.world_min.cpu().numpy()
        normalized = (world_pos - self.world_min.cpu().numpy()) / world_size

        px = normalized[0] * self.width
        py = (1.0 - normalized[1]) * self.height

        return np.array([px, py])

    def pixel_to_world(self, pixel_pos: np.ndarray) -> np.ndarray:
        """Pixel coordinates to world coordinates (CPU)"""
        normalized = np.array(
            [pixel_pos[0] / self.width, 1.0 - pixel_pos[1] / self.height]
        )

        world_size = self.world_max.cpu().numpy() - self.world_min.cpu().numpy()
        world_pos = self.world_min.cpu().numpy() + normalized * world_size

        return world_pos

    def render(
        self, gaussians: List[Gaussian2D], spline_data: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Render Gaussians to 2D image (gsplat-accelerated)

        Args:
            gaussians: List of Gaussian2D objects
            spline_data: Optional spline visualization data for debug mode

        Returns:
            RGB image (height, width, 3) in range [0, 1]
        """
        if len(gaussians) == 0:
            # Return white background
            return self.background_color.cpu().numpy().reshape(1, 1, 3) * np.ones(
                (self.height, self.width, 3), dtype=np.float32
            )

        # Depth sort (painter's algorithm)
        sorted_gaussians = sorted(gaussians, key=lambda g: -g.position[2])
        n = len(sorted_gaussians)

        # Batch: Collect all Gaussian data on CPU
        positions = np.zeros((n, 3), dtype=np.float32)
        scales = np.zeros((n, 3), dtype=np.float32)
        rotations = np.zeros((n, 4), dtype=np.float32)
        opacities = np.zeros(n, dtype=np.float32)
        colors = np.zeros((n, 3), dtype=np.float32)

        for i, g in enumerate(sorted_gaussians):
            positions[i] = g.position
            scales[i] = g.scale
            scales[i][2] = 0.1  # Override z-scale: gsplat requires non-zero depth
            rotations[i] = g.rotation
            opacities[i] = g.opacity
            colors[i] = g.color

        # Transfer to GPU in single batch
        means = torch.from_numpy(positions).to(self.device, non_blocking=True)
        scales_t = torch.from_numpy(scales).to(self.device, non_blocking=True)
        quats_xyzw = torch.from_numpy(rotations).to(self.device, non_blocking=True)
        opacities_t = torch.from_numpy(opacities).to(self.device, non_blocking=True)
        colors_t = torch.from_numpy(colors).to(self.device, non_blocking=True)

        # Convert quaternions: (x,y,z,w) -> (w,x,y,z) for gsplat
        quats = torch.zeros_like(quats_xyzw)
        quats[:, 0] = quats_xyzw[:, 3]  # w
        quats[:, 1] = quats_xyzw[:, 0]  # x
        quats[:, 2] = quats_xyzw[:, 1]  # y
        quats[:, 3] = quats_xyzw[:, 2]  # z

        # Normalize quaternions
        quats = quats / (torch.norm(quats, dim=1, keepdim=True) + 1e-8)

        # Render using gsplat
        try:
            import time

            start_time = time.time()

            render_colors, render_alphas, meta = rasterization(
                means=means,
                quats=quats,
                scales=scales_t,
                opacities=opacities_t,
                colors=colors_t,
                viewmats=self.viewmat,
                Ks=self.K,
                width=self.width,
                height=self.height,
                packed=False,
                render_mode="RGB",
                sh_degree=None,
                camera_model="ortho",  # Orthographic projection for 2D rendering
                # Critical parameters to prevent culling
                near_plane=0.1,  # Adjusted near plane (default 0.01)
                far_plane=10.0,  # Adjusted far plane (default 1e10)
                radius_clip=0.0,  # Disable radius culling (default may cull small Gaussians)
                eps2d=0.3,  # 2D covariance regularization
            )

            # Extract RGB image
            # gsplat returns [B, H, W, C] format
            if len(render_colors.shape) == 4:
                # Format: [B, H, W, C] -> [H, W, C]
                image = render_colors[0]  # Remove batch dimension
                alpha = render_alphas[0]  # [H, W, 1]
            elif len(render_colors.shape) == 3:
                # Format: [C, H, W] -> [H, W, C]
                image = render_colors.permute(1, 2, 0)  # [H, W, 3]
                alpha = render_alphas.unsqueeze(-1)  # [H, W, 1]
            else:
                # Fallback
                image = render_colors[0]
                alpha = render_alphas[0]

            # Blend with background
            image = image * alpha + self.background_color * (1.0 - alpha)

            elapsed_ms = (time.time() - start_time) * 1000
            fps = 1000 / elapsed_ms if elapsed_ms > 0 else 0
            print(
                f"[GSplat Renderer] ✓ {n} Gaussians in {elapsed_ms:.2f}ms ({fps:.1f} FPS)"
            )

            # Return to CPU
            result = torch.clamp(image, 0.0, 1.0).cpu().numpy()

            # Add debug overlay if enabled
            if self.debug_mode and self.debug_visualizer:
                # Convert float image to uint8 for OpenCV
                image_uint8 = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)

                # Create debug overlay
                debug_overlay = self.debug_visualizer.create_debug_overlay(
                    gaussians, image=image_uint8, spline_data=spline_data
                )

                # Convert back to float
                result = debug_overlay.astype(np.float32) / 255.0

            return result

        except Exception as e:
            print(f"[GSplat Renderer] Error during rasterization: {e}")
            import traceback

            traceback.print_exc()

            # Fallback to background
            return self.background_color.cpu().numpy().reshape(1, 1, 3) * np.ones(
                (self.height, self.width, 3), dtype=np.float32
            )

    def set_debug_mode(self, enabled: bool):
        """
        Enable or disable debug mode

        Args:
            enabled: True to enable debug mode
        """
        self.debug_mode = enabled
        if self.debug_mode and self.debug_visualizer is None:
            # Convert torch tensors to numpy for DebugVisualizer
            world_min_np = self.world_min.cpu().numpy()
            world_max_np = self.world_max.cpu().numpy()
            self.debug_visualizer = DebugVisualizer(
                width=self.width, height=self.height,
                world_min=world_min_np, world_max=world_max_np
            )
            print(f"[GSplat Renderer] Debug mode enabled")
        elif not self.debug_mode:
            print(f"[GSplat Renderer] Debug mode disabled")

    def set_debug_options(self, options: Dict[str, Any]):
        """
        Configure debug visualization options

        Args:
            options: Dictionary of debug options
        """
        if self.debug_visualizer:
            self.debug_visualizer.set_debug_options(options)
            print(f"[GSplat Renderer] Debug options updated")

    def render_with_depth(self, gaussians: List[Gaussian2D]) -> tuple:
        """
        Render RGB + depth map

        Args:
            gaussians: List of Gaussian2D

        Returns:
            (rgb_image, depth_map)
        """
        # gsplat can provide depth information
        # For now, use simplified approach
        rgb = self.render(gaussians)

        # Depth map (simplified - could be extracted from gsplat meta)
        depth_map = np.zeros((self.height, self.width), dtype=np.float32)

        for gaussian in gaussians:
            center_pixel = self.world_to_pixel(gaussian.position[:2])
            cx, cy = int(center_pixel[0]), int(center_pixel[1])

            if 0 <= cx < self.width and 0 <= cy < self.height:
                depth_map[cy, cx] = max(depth_map[cy, cx], gaussian.position[2])

        return rgb, depth_map


def test_gsplat_renderer():
    """Test gsplat renderer functionality"""
    print("Testing GSplat Renderer...")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available, using CPU")

    # Check gsplat
    if not GSPLAT_AVAILABLE:
        print("✗ gsplat not available")
        print("Install with: pip install gsplat")
        return

    print("✓ gsplat available")

    # Create renderer
    renderer = GaussianRenderer2D_GSplat(width=512, height=512)

    # Create test Gaussians
    from .gaussian import Gaussian2D

    # Test with varying counts
    for count in [10, 100, 1000]:
        gaussians = []
        for i in range(count):
            pos = np.random.uniform(-0.5, 0.5, 2)
            g = Gaussian2D(
                position=np.array([pos[0], pos[1], 0.0]),
                scale=np.array([0.05, 0.05, 1e-4]),
                rotation=np.array([0.0, 0.0, 0.0, 1.0]),
                opacity=0.5,
                color=np.random.uniform(0, 1, 3),
            )
            gaussians.append(g)

        import time

        start = time.time()
        image = renderer.render(gaussians)
        elapsed = time.time() - start

        print(
            f"✓ Rendered {count:4d} Gaussians in {elapsed*1000:6.2f}ms ({1/elapsed:5.1f} FPS)"
        )

    print("GSplat Renderer test complete!")


if __name__ == "__main__":
    test_gsplat_renderer()
