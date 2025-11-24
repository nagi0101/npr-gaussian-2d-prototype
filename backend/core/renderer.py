"""
GaussianRenderer2D: 2D Gaussian Splatting 렌더러

초기 프로토타입: NumPy 기반 CPU 렌더러
향후 GPU 가속 버전으로 업그레이드 가능
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .gaussian import Gaussian2D
from .debug_visualizer import DebugVisualizer
import cv2


class GaussianRenderer2D:
    """
    2D Gaussian Splatting 렌더러

    z=0 평면의 Gaussian들을 2D 이미지로 렌더링
    Alpha blending을 사용하여 투명도 처리
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        background_color: np.ndarray = np.array([1.0, 1.0, 1.0]),
        debug_mode: bool = False,
    ):
        """
        Args:
            width: 렌더링 이미지 너비 (픽셀)
            height: 렌더링 이미지 높이 (픽셀)
            background_color: 배경색 RGB [0, 1]
            debug_mode: Enable debug visualization
        """
        self.width = width
        self.height = height
        self.background_color = np.array(background_color, dtype=np.float32)

        # World space 좌표계 설정
        # 기본: (-1, 1) x (-1, 1) world space → (width, height) pixel space
        self.world_min = np.array([-1.0, -1.0])
        self.world_max = np.array([1.0, 1.0])

        # Compute uniform scale and offset for aspect ratio preservation
        self._compute_scale_and_offset()

        # Debug mode
        self.debug_mode = debug_mode
        self.debug_visualizer = None
        if self.debug_mode:
            self.debug_visualizer = DebugVisualizer(
                width=width, height=height,
                world_min=self.world_min, world_max=self.world_max
            )

    def set_world_bounds(self, world_min: np.ndarray, world_max: np.ndarray):
        """
        World space 범위 설정

        Args:
            world_min: (x_min, y_min)
            world_max: (x_max, y_max)
        """
        self.world_min = np.array(world_min, dtype=np.float32)
        self.world_max = np.array(world_max, dtype=np.float32)

        # Recompute scale and offset with new bounds
        self._compute_scale_and_offset()

        # Update debug visualizer bounds if exists
        if self.debug_visualizer:
            self.debug_visualizer.world_min = self.world_min
            self.debug_visualizer.world_max = self.world_max

    def _compute_scale_and_offset(self):
        """
        Compute uniform scale and offset to preserve aspect ratio.
        Uses letterboxing if world and pixel aspect ratios don't match.
        """
        world_width = self.world_max[0] - self.world_min[0]
        world_height = self.world_max[1] - self.world_min[1]

        # Use uniform scale (smaller of the two) to preserve aspect ratio
        scale_x = self.width / world_width
        scale_y = self.height / world_height
        self.uniform_scale = min(scale_x, scale_y)

        # Compute offset to center content
        scaled_world_width = world_width * self.uniform_scale
        scaled_world_height = world_height * self.uniform_scale

        self.offset_x = (self.width - scaled_world_width) / 2.0
        self.offset_y = (self.height - scaled_world_height) / 2.0

    def world_to_pixel(self, world_pos: np.ndarray) -> np.ndarray:
        """
        World 좌표를 픽셀 좌표로 변환 (uniform scaling, aspect ratio preserved)

        Args:
            world_pos: (x, y) in world space

        Returns:
            (px, py) in pixel space
        """
        # Translate to origin, scale uniformly, then offset to center
        px = (world_pos[0] - self.world_min[0]) * self.uniform_scale + self.offset_x
        py = (self.world_max[1] - world_pos[1]) * self.uniform_scale + self.offset_y  # Y flip

        return np.array([px, py])

    def pixel_to_world(self, pixel_pos: np.ndarray) -> np.ndarray:
        """
        픽셀 좌표를 world 좌표로 변환

        Args:
            pixel_pos: (px, py) in pixel space

        Returns:
            (x, y) in world space
        """
        # Inverse of world_to_pixel transformation
        # 1. Remove offset
        adjusted_x = pixel_pos[0] - self.offset_x
        adjusted_y = pixel_pos[1] - self.offset_y

        # 2. Scale back to world space (inverse uniform scale)
        world_x = self.world_min[0] + adjusted_x / self.uniform_scale
        world_y = self.world_max[1] - adjusted_y / self.uniform_scale  # Y flip

        return np.array([world_x, world_y])

    def render(
        self, gaussians: List[Gaussian2D], spline_data: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Gaussian 리스트를 2D 이미지로 렌더링

        Args:
            gaussians: List of Gaussian2D objects
            spline_data: Optional spline visualization data for debug mode

        Returns:
            RGB 이미지 (height, width, 3) in range [0, 1]
        """
        # 초기화: 배경색
        image = (
            np.ones((self.height, self.width, 3), dtype=np.float32)
            * self.background_color
        )
        alpha_buffer = np.zeros((self.height, self.width), dtype=np.float32)

        # Debug: Print first 3 Gaussians
        if len(gaussians) > 0 and len(gaussians) <= 20:  # Only for small counts
            print(
                f"[Renderer] First Gaussian: pos={gaussians[0].position[:2]}, color={gaussians[0].color}, opacity={gaussians[0].opacity}"
            )

        # Depth sort (painter's algorithm)
        # z=0이지만 opacity에 따라 정렬
        sorted_gaussians = sorted(gaussians, key=lambda g: -g.position[2])

        # 각 Gaussian을 렌더링
        for i, gaussian in enumerate(sorted_gaussians):
            self._render_single_gaussian(gaussian, image, alpha_buffer, debug=(i < 3))

        # Add debug overlay if enabled
        if self.debug_mode and self.debug_visualizer:
            # Convert float image to uint8 for OpenCV
            image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

            # Create debug overlay
            debug_overlay = self.debug_visualizer.create_debug_overlay(
                gaussians, image=image_uint8, spline_data=spline_data
            )

            # Convert back to float
            image = debug_overlay.astype(np.float32) / 255.0

        return np.clip(image, 0.0, 1.0)

    def _render_single_gaussian(
        self,
        gaussian: Gaussian2D,
        image: np.ndarray,
        alpha_buffer: np.ndarray,
        debug: bool = False,
    ):
        """
        단일 Gaussian을 이미지에 렌더링 (alpha blending)

        Args:
            gaussian: Gaussian2D object
            image: 렌더링할 이미지 (in-place 수정)
            alpha_buffer: Alpha accumulation buffer
            debug: If True, print debug info
        """
        # World 좌표 → 픽셀 좌표
        center_pixel = self.world_to_pixel(gaussian.position[:2])
        cx, cy = int(center_pixel[0]), int(center_pixel[1])

        if debug:
            print(
                f"[Renderer] Gaussian: world=({gaussian.position[0]:.3f}, {gaussian.position[1]:.3f}), pixel=({cx}, {cy}), color={gaussian.color}, opacity={gaussian.opacity:.2f}"
            )

        # Covariance matrix 계산
        cov_world = gaussian.compute_covariance_2d()

        # World space covariance → Pixel space covariance
        # Use uniform scale to preserve aspect ratio
        scale_matrix = np.diag([self.uniform_scale, self.uniform_scale])

        cov_pixel = scale_matrix @ cov_world @ scale_matrix.T

        # Bounding box 계산 (3σ 규칙)
        try:
            # Eigenvalue로 최대 반경 계산
            eigenvalues = np.linalg.eigvalsh(cov_pixel)
            max_radius = 3.0 * np.sqrt(max(eigenvalues))
        except:
            max_radius = 50.0  # Fallback

        # Bounding box
        x_min = max(0, int(cx - max_radius))
        x_max = min(self.width, int(cx + max_radius) + 1)
        y_min = max(0, int(cy - max_radius))
        y_max = min(self.height, int(cy + max_radius) + 1)

        if debug:
            print(
                f"[Renderer] Bounding box: x=[{x_min}, {x_max}), y=[{y_min}, {y_max}), radius={max_radius:.1f}"
            )

        if x_min >= x_max or y_min >= y_max:
            if debug:
                print(f"[Renderer] ✗ Gaussian culled (out of bounds)")
            return  # Culled

        # Inverse covariance (for Gaussian evaluation)
        try:
            cov_inv = np.linalg.inv(cov_pixel)
        except np.linalg.LinAlgError:
            # Singular matrix, skip
            return

        # 픽셀별 Gaussian 평가
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                # 중심으로부터 offset
                dx = x - cx
                dy = y - cy
                delta = np.array([dx, dy])

                # Gaussian 값 계산: exp(-0.5 * delta^T * Σ^-1 * delta)
                exponent = -0.5 * (delta @ cov_inv @ delta)

                if exponent < -20:  # Negligible
                    continue

                gaussian_value = np.exp(exponent)

                # Alpha blending
                alpha = gaussian.opacity * gaussian_value

                # Front-to-back blending
                # 이미 누적된 alpha 고려
                accumulated_alpha = alpha_buffer[y, x]

                if accumulated_alpha > 0.99:
                    continue  # Already opaque

                # Blend (correct alpha blending formula)
                contribution = alpha * (1.0 - accumulated_alpha)
                image[y, x] = (
                    image[y, x] * (1.0 - contribution) + gaussian.color * contribution
                )
                alpha_buffer[y, x] += contribution

    def render_with_depth(
        self, gaussians: List[Gaussian2D]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RGB + Depth 맵 반환

        Args:
            gaussians: List of Gaussian2D

        Returns:
            (rgb_image, depth_map)
            - rgb_image: (H, W, 3) in [0, 1]
            - depth_map: (H, W) in [0, 1] (normalized)
        """
        rgb = self.render(gaussians)

        # Depth map (z 좌표 기반)
        depth_map = np.zeros((self.height, self.width), dtype=np.float32)

        for gaussian in gaussians:
            center_pixel = self.world_to_pixel(gaussian.position[:2])
            cx, cy = int(center_pixel[0]), int(center_pixel[1])

            cov_world = gaussian.compute_covariance_2d()
            scale_x = self.width / (self.world_max[0] - self.world_min[0])
            scale_y = self.height / (self.world_max[1] - self.world_min[1])
            scale_matrix = np.diag([scale_x, scale_y])
            cov_pixel = scale_matrix @ cov_world @ scale_matrix.T

            try:
                eigenvalues = np.linalg.eigvalsh(cov_pixel)
                max_radius = 3.0 * np.sqrt(max(eigenvalues))
            except:
                max_radius = 50.0

            x_min = max(0, int(cx - max_radius))
            x_max = min(self.width, int(cx + max_radius) + 1)
            y_min = max(0, int(cy - max_radius))
            y_max = min(self.height, int(cy + max_radius) + 1)

            if x_min >= x_max or y_min >= y_max:
                continue

            try:
                cov_inv = np.linalg.inv(cov_pixel)
            except:
                continue

            z_value = gaussian.position[2]

            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    dx = x - cx
                    dy = y - cy
                    delta = np.array([dx, dy])

                    exponent = -0.5 * (delta @ cov_inv @ delta)
                    if exponent < -20:
                        continue

                    gaussian_value = np.exp(exponent)
                    alpha = gaussian.opacity * gaussian_value

                    # 가장 가까운 depth 유지
                    if alpha > 0.1:
                        current_depth = depth_map[y, x]
                        if current_depth == 0 or z_value < current_depth:
                            depth_map[y, x] = z_value

        # Normalize depth
        if depth_map.max() > depth_map.min():
            depth_map = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min()
            )

        return rgb, depth_map

    def render_to_uint8(self, gaussians: List[Gaussian2D]) -> np.ndarray:
        """
        렌더링 결과를 uint8 이미지로 반환 (저장/전송용)

        Args:
            gaussians: List of Gaussian2D

        Returns:
            RGB 이미지 (H, W, 3) in range [0, 255] uint8
        """
        image_float = self.render(gaussians)
        image_uint8 = (image_float * 255).astype(np.uint8)
        return image_uint8

    def save_image(self, gaussians: List[Gaussian2D], filename: str):
        """
        렌더링 결과를 파일로 저장

        Args:
            gaussians: List of Gaussian2D
            filename: 저장할 파일 경로 (.png 권장)
        """
        image = self.render_to_uint8(gaussians)
        # OpenCV는 BGR 순서
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image_bgr)
        print(f"Image saved to {filename}")

    def set_debug_mode(self, enabled: bool):
        """
        Enable or disable debug mode

        Args:
            enabled: True to enable debug mode
        """
        self.debug_mode = enabled
        if self.debug_mode and self.debug_visualizer is None:
            self.debug_visualizer = DebugVisualizer(
                width=self.width, height=self.height,
                world_min=self.world_min, world_max=self.world_max
            )

    def set_debug_options(self, options: Dict[str, Any]):
        """
        Configure debug visualization options

        Args:
            options: Dictionary of debug options
        """
        if self.debug_visualizer:
            self.debug_visualizer.set_debug_options(options)


def create_renderer(
    width: int = 1024,
    height: int = 768,
    background_color: np.ndarray = np.array([1.0, 1.0, 1.0]),
    prefer_gpu: bool = True,
    debug_mode: bool = False,
):
    """
    Factory function to create renderer with automatic GPU/CPU selection

    Priority:
    1. GSplat renderer (fastest, native 2D support)
    2. PyTorch GPU renderer (fallback)
    3. CPU renderer (final fallback)

    Args:
        width: Rendering width
        height: Rendering height
        background_color: Background RGB [0, 1]
        prefer_gpu: If True, try GPU renderers first, fallback to CPU
        debug_mode: Enable debug visualization

    Returns:
        Best available renderer based on GPU availability and library availability
    """
    if prefer_gpu:
        # Try gsplat renderer first (best performance)
        try:
            import torch

            if torch.cuda.is_available():
                from .renderer_gsplat import GaussianRenderer2D_GSplat

                print(f"[Renderer] Using GSplat renderer (CUDA + gsplat available)")
                return GaussianRenderer2D_GSplat(
                    width=width,
                    height=height,
                    background_color=background_color,
                    debug_mode=debug_mode,
                )
        except ImportError:
            print(f"[Renderer] GSplat not available (pip install gsplat)")
        except Exception as e:
            print(f"[Renderer] GSplat renderer failed: {e}")

        # Fallback to PyTorch GPU renderer
        try:
            import torch

            if torch.cuda.is_available():
                from .renderer_gpu import GaussianRenderer2D_GPU

                print(f"[Renderer] Using PyTorch GPU renderer (CUDA available)")
                return GaussianRenderer2D_GPU(
                    width=width, height=height, background_color=background_color
                )
        except Exception as e:
            print(f"[Renderer] GPU renderer failed: {e}")
            print(f"[Renderer] Falling back to CPU renderer")

    # Final fallback to CPU
    print(f"[Renderer] Using CPU renderer")
    return GaussianRenderer2D(
        width=width,
        height=height,
        background_color=background_color,
        debug_mode=debug_mode,
    )


def test_renderer():
    """렌더러 테스트 함수"""
    from .gaussian import create_test_gaussian

    # 렌더러 생성
    renderer = GaussianRenderer2D(width=512, height=512)

    # 테스트 Gaussian 생성
    gaussians = []
    for i in range(5):
        x = -0.5 + i * 0.25
        g = create_test_gaussian(x=x, y=0.0)
        g.color = np.array([i / 5.0, 0.5, 1.0 - i / 5.0])
        gaussians.append(g)

    # 렌더링
    image = renderer.render(gaussians)

    print(f"Rendered image shape: {image.shape}")
    print(f"Image range: [{image.min()}, {image.max()}]")

    # 저장 (테스트용)
    renderer.save_image(gaussians, "test_render.png")


if __name__ == "__main__":
    test_renderer()
