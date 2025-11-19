"""
PyTorch-based Optimizer for Gaussian Parameters using gsplat

Uses gsplat's differentiable CUDA rasterization directly for 1000x speedup
over manual Python loops. This is the standard 3DGS optimization approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Callable, Optional
import logging

try:
    from gsplat import rasterization
except ImportError:
    raise ImportError("gsplat is required for PyTorch optimization. Please install it.")

from .gaussian import Gaussian2D

logger = logging.getLogger(__name__)


class TorchGaussianOptimizer:
    """
    PyTorch-based optimizer for 2D Gaussian Splatting parameters.

    Uses gsplat's differentiable CUDA rasterization for fast optimization
    of position, scale, rotation, and opacity.
    """

    def __init__(
        self,
        gaussians: List[Gaussian2D],
        target_image: np.ndarray,
        renderer,
        alpha_mask: Optional[np.ndarray] = None,
        device: Optional[str] = None
    ):
        """
        Initialize PyTorch optimizer.

        Args:
            gaussians: Initial Gaussians to optimize
            target_image: Target image (H, W, 3) in range [0, 1]
            renderer: Renderer instance (must have viewmat and K tensors)
            alpha_mask: Optional alpha mask (H, W) in range [0, 1] for masked loss
            device: 'cuda' or 'cpu', auto-detect if None
        """
        # Device selection
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)
        logger.info(f"[TorchOptimizer] Using device: {self.device}")

        # Store initial Gaussians for reference
        self.initial_gaussians = gaussians
        self.n_gaussians = len(gaussians)

        # Convert 2D positions to 3D (add z=0)
        positions_2d = np.array([[g.position[0], g.position[1]] for g in gaussians], dtype=np.float32)
        positions_3d = np.concatenate([
            positions_2d,
            np.zeros((self.n_gaussians, 1), dtype=np.float32)
        ], axis=1)

        # Convert scales to 3D (add small z scale)
        scales_2d = np.array([[g.scale[0], g.scale[1]] for g in gaussians], dtype=np.float32)
        scales_3d = np.concatenate([
            scales_2d,
            np.ones((self.n_gaussians, 1), dtype=np.float32) * 1e-4  # Small z scale
        ], axis=1)

        # Convert rotation angles to quaternions (2D rotation around Z axis)
        angles = np.array([self._quat_to_angle(g.rotation) for g in gaussians], dtype=np.float32)
        quaternions = self._angles_to_quaternions(angles)

        # Create PyTorch parameters using nn.Parameter (required for optimizer)
        self.means = nn.Parameter(
            torch.tensor(positions_3d, device=self.device, dtype=torch.float32)
        )
        self.scales = nn.Parameter(
            torch.tensor(scales_3d, device=self.device, dtype=torch.float32)
        )
        self.quats = nn.Parameter(
            torch.tensor(quaternions, device=self.device, dtype=torch.float32)
        )

        # Opacities
        opacities_np = np.array([g.opacity for g in gaussians], dtype=np.float32)
        self.opacities = nn.Parameter(
            torch.tensor(opacities_np, device=self.device, dtype=torch.float32)  # [N] shape for gsplat
        )

        # Colors (convert list to numpy first to avoid warning)
        colors_np = np.array([g.color for g in gaussians], dtype=np.float32)
        self.colors = nn.Parameter(
            torch.tensor(colors_np, device=self.device, dtype=torch.float32)
        )

        # Convert target image to tensor
        self.target = torch.tensor(target_image, device=self.device, dtype=torch.float32)
        self.height, self.width = self.target.shape[:2]

        # Convert alpha mask to tensor if provided
        if alpha_mask is not None:
            self.alpha_mask = torch.tensor(alpha_mask, device=self.device, dtype=torch.float32)
            # Ensure binary mask
            self.alpha_mask = (self.alpha_mask > 0.5).float()
            logger.info(f"[TorchOptimizer] Using alpha mask for loss computation (coverage: {self.alpha_mask.mean():.1%})")
        else:
            # Default: create mask from non-white pixels (works for both white and black backgrounds)
            # Compute luminance
            luminance = 0.299 * self.target[..., 0] + 0.587 * self.target[..., 1] + 0.114 * self.target[..., 2]
            # Pixels that are not close to white (1.0) or black (0.0)
            self.alpha_mask = ((luminance > 0.05) & (luminance < 0.95)).float()
            logger.info(f"[TorchOptimizer] Created alpha mask from non-background pixels (coverage: {self.alpha_mask.mean():.1%})")

        # Get camera matrices from renderer (already torch tensors on device)
        if hasattr(renderer, 'viewmat') and hasattr(renderer, 'K'):
            self.viewmat = renderer.viewmat  # [1, 4, 4]
            self.K = renderer.K  # [1, 3, 3]
        else:
            # For backward compatibility with other renderers
            logger.warning("[TorchOptimizer] Renderer doesn't have viewmat/K, creating default")
            self._create_default_camera_matrices()

        logger.info(f"[TorchOptimizer] Initialized with {self.n_gaussians} Gaussians")
        logger.info(f"[TorchOptimizer] Target shape: {self.target.shape}, Render size: {self.width}x{self.height}")

    def _quat_to_angle(self, quat: np.ndarray) -> float:
        """
        Convert quaternion to 2D rotation angle.

        Args:
            quat: Quaternion [x, y, z, w]

        Returns:
            Rotation angle in radians
        """
        # For 2D rotation around Z-axis
        # quat = [0, 0, sin(θ/2), cos(θ/2)]
        return 2.0 * np.arctan2(quat[2], quat[3])

    def _angles_to_quaternions(self, angles: np.ndarray) -> np.ndarray:
        """
        Convert 2D rotation angles to quaternions.

        Args:
            angles: Array of rotation angles in radians [N]

        Returns:
            Quaternions [N, 4] in (w, x, y, z) order for gsplat
        """
        half_angles = angles / 2.0
        quaternions = np.zeros((len(angles), 4), dtype=np.float32)
        quaternions[:, 0] = np.cos(half_angles)  # w
        quaternions[:, 1] = 0.0  # x
        quaternions[:, 2] = 0.0  # y
        quaternions[:, 3] = np.sin(half_angles)  # z
        return quaternions

    def _create_default_camera_matrices(self):
        """Create default orthographic camera matrices if renderer doesn't provide them"""
        # Default orthographic view matrix
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],  # Flip Y
                [0.0, 0.0, 1.0, 5.0],   # Camera at z=5
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, 4, 4]

        # Default orthographic projection matrix
        fx = fy = min(self.width, self.height) / 2.0
        cx = self.width / 2.0
        cy = self.height / 2.0

        self.K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, 3, 3]

    def _render_differentiable(self) -> torch.Tensor:
        """
        Render Gaussians using gsplat's differentiable CUDA rasterization.

        This is the standard 3DGS approach - let gsplat handle everything.

        Returns:
            Rendered image [H, W, 3] in range [0, 1]
        """
        # Call gsplat's differentiable rasterization directly
        # This uses optimized CUDA kernels with automatic gradient support
        render_colors, render_alphas, meta = rasterization(
            means=self.means,              # [N, 3] positions
            quats=self.quats,               # [N, 4] quaternions (w,x,y,z)
            scales=self.scales,             # [N, 3] scales
            opacities=self.opacities,       # [N] opacities
            colors=self.colors,             # [N, 3] RGB colors
            viewmats=self.viewmat,          # [1, 4, 4] view matrix
            Ks=self.K,                      # [1, 3, 3] projection matrix
            width=self.width,
            height=self.height,
            packed=False,
            camera_model="ortho",           # Orthographic for 2D
            near_plane=0.1,
            far_plane=10.0,
            eps2d=0.01,                     # Regularization for stability
            render_mode="RGB",
        )

        # Extract image from batch [1, H, W, 3] -> [H, W, 3]
        return render_colors[0]

    def _compute_ssim_torch(
        self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
    ) -> torch.Tensor:
        """
        Compute SSIM (Structural Similarity Index) in PyTorch (differentiable).

        Args:
            img1: Image 1 [H, W, 3]
            img2: Image 2 [H, W, 3]
            window_size: Gaussian window size

        Returns:
            SSIM value (0-1, higher is better)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Convert to grayscale
        gray1 = 0.299 * img1[..., 0] + 0.587 * img1[..., 1] + 0.114 * img1[..., 2]
        gray2 = 0.299 * img2[..., 0] + 0.587 * img2[..., 1] + 0.114 * img2[..., 2]

        # Add batch and channel dimensions: [1, 1, H, W]
        gray1 = gray1.unsqueeze(0).unsqueeze(0)
        gray2 = gray2.unsqueeze(0).unsqueeze(0)

        # Create Gaussian window
        sigma = 1.5
        gauss = torch.exp(
            -torch.arange(window_size, dtype=torch.float32, device=self.device) ** 2 / (2 * sigma ** 2)
        )
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]

        # Compute local means
        mu1 = F.conv2d(gray1, window, padding=window_size // 2)
        mu2 = F.conv2d(gray2, window, padding=window_size // 2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = F.conv2d(gray1 ** 2, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(gray2 ** 2, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(gray1 * gray2, window, padding=window_size // 2) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def optimize(
        self,
        iterations: int = 50,
        lr_position: float = 0.001,
        lr_scale: float = 0.005,
        lr_rotation: float = 0.001,
        lr_opacity: float = 0.01,
        progress_callback: Optional[Callable] = None
    ) -> List[Gaussian2D]:
        """
        Optimize Gaussian parameters using PyTorch Adam optimizer with gsplat.

        Args:
            iterations: Number of optimization iterations
            lr_position: Learning rate for position
            lr_scale: Learning rate for scale
            lr_rotation: Learning rate for rotation
            lr_opacity: Learning rate for opacity
            progress_callback: Optional callback(iter, total, loss, rendered_np)

        Returns:
            Optimized Gaussians as List[Gaussian2D]
        """
        logger.info(f"[TorchOptimizer] Starting optimization for {iterations} iterations")
        logger.info(f"[TorchOptimizer] Learning rates: pos={lr_position}, scale={lr_scale}, "
                   f"rot={lr_rotation}, opacity={lr_opacity}")

        # Create Adam optimizer with per-parameter learning rates
        optimizer = torch.optim.Adam([
            {'params': self.means, 'lr': lr_position},
            {'params': self.scales, 'lr': lr_scale},
            {'params': self.quats, 'lr': lr_rotation},
            {'params': self.opacities, 'lr': lr_opacity}
        ])

        best_loss = float('inf')
        best_state = None

        for iter_idx in range(iterations):
            optimizer.zero_grad()

            # Render with gsplat's differentiable CUDA kernels
            rendered = self._render_differentiable()

            # Apply alpha mask to compute loss only in stroke regions
            mask_3d = self.alpha_mask.unsqueeze(-1).expand(-1, -1, 3)  # Expand to RGB channels
            masked_rendered = rendered * mask_3d
            masked_target = self.target * mask_3d

            # Normalize by valid pixels to avoid bias
            valid_pixels = self.alpha_mask.sum()
            total_pixels = self.height * self.width

            # Compute losses only in masked regions
            if valid_pixels > 0:
                l1_loss = F.l1_loss(masked_rendered, masked_target) * (total_pixels / valid_pixels)
                # For SSIM, use full images but weight by mask coverage
                ssim_value = self._compute_ssim_torch(rendered, self.target)
                ssim_loss = (1.0 - ssim_value) * (valid_pixels / total_pixels)
            else:
                l1_loss = F.l1_loss(rendered, self.target)
                ssim_value = self._compute_ssim_torch(rendered, self.target)
                ssim_loss = 1.0 - ssim_value

            # Regularization terms
            scale_reg = torch.mean((self.scales[:, :2] - 0.02) ** 2) * 0.01  # Only x,y scales
            # L1 sparsity regularization for opacity (encourages low opacity in background)
            opacity_reg = torch.mean(self.opacities) * 0.02  # Increased weight for stronger sparsity

            # Total loss
            total_loss = 0.7 * l1_loss + 0.2 * ssim_loss + scale_reg + opacity_reg

            # Backpropagation (gradients flow through CUDA kernels automatically)
            total_loss.backward()

            # Gradient clipping (prevent instability)
            torch.nn.utils.clip_grad_norm_(
                [self.means, self.scales, self.quats, self.opacities],
                max_norm=1.0
            )

            # Update parameters
            optimizer.step()

            # Clamp to valid ranges (non-in-place to avoid autograd issues)
            with torch.no_grad():
                self.scales.data = self.scales.clamp(min=0.001, max=0.2)
                self.opacities.data = self.opacities.clamp(min=0.01, max=1.0)

                # Normalize quaternions
                self.quats.data = F.normalize(self.quats, dim=-1)

            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state = {
                    'means': self.means.clone(),
                    'scales': self.scales.clone(),
                    'quats': self.quats.clone(),
                    'opacities': self.opacities.clone()
                }

            # Progress callback
            if progress_callback and iter_idx % 1 == 0:
                with torch.no_grad():
                    rendered_np = rendered.cpu().numpy()
                    progress_callback(iter_idx, iterations, total_loss.item(), rendered_np)

            # Logging
            if iter_idx % 5 == 0:
                logger.info(
                    f"[TorchOptimizer] Iter {iter_idx}/{iterations}: "
                    f"Loss={total_loss.item():.4f} (L1={l1_loss.item():.4f}, "
                    f"SSIM={ssim_loss.item():.4f})"
                )

            # Early stopping
            if total_loss.item() < 0.01:
                logger.info(f"[TorchOptimizer] Early stopping at iteration {iter_idx}")
                break

        # Restore best state if current is worse
        if best_state and total_loss.item() > best_loss:
            self.means.data = best_state['means']
            self.scales.data = best_state['scales']
            self.quats.data = best_state['quats']
            self.opacities.data = best_state['opacities']
            logger.info(f"[TorchOptimizer] Restored best state (loss={best_loss:.4f})")

        # Convert back to Gaussian2D objects
        return self._tensors_to_gaussians()

    def _tensors_to_gaussians(self) -> List[Gaussian2D]:
        """
        Convert optimized tensors back to Gaussian2D objects.

        Returns:
            List of optimized Gaussian2D objects
        """
        with torch.no_grad():
            means_cpu = self.means.cpu().numpy()
            scales_cpu = self.scales.cpu().numpy()
            quats_cpu = self.quats.cpu().numpy()
            opacities_cpu = self.opacities.cpu().numpy()  # Already [N] shape
            colors_cpu = self.colors.cpu().numpy()

        optimized_gaussians = []
        for i in range(self.n_gaussians):
            # Convert quaternion back to standard form
            quat_wxyz = quats_cpu[i]
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)

            g = Gaussian2D(
                position=means_cpu[i, :2],  # Only x,y (ignore z)
                rotation=quat_xyzw,
                scale=scales_cpu[i, :2],    # Only x,y scales
                opacity=opacities_cpu[i],
                color=colors_cpu[i]
            )
            optimized_gaussians.append(g)

        return optimized_gaussians