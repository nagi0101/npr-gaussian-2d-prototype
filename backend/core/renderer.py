"""
GaussianRenderer2D: 2D Gaussian Splatting 렌더러

초기 프로토타입: NumPy 기반 CPU 렌더러
향후 GPU 가속 버전으로 업그레이드 가능
"""

import numpy as np
from typing import List, Tuple, Optional
from .gaussian import Gaussian2D
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
        background_color: np.ndarray = np.array([1.0, 1.0, 1.0])
    ):
        """
        Args:
            width: 렌더링 이미지 너비 (픽셀)
            height: 렌더링 이미지 높이 (픽셀)
            background_color: 배경색 RGB [0, 1]
        """
        self.width = width
        self.height = height
        self.background_color = np.array(background_color, dtype=np.float32)

        # World space 좌표계 설정
        # 기본: (-1, 1) x (-1, 1) world space → (width, height) pixel space
        self.world_min = np.array([-1.0, -1.0])
        self.world_max = np.array([1.0, 1.0])

    def set_world_bounds(self, world_min: np.ndarray, world_max: np.ndarray):
        """
        World space 범위 설정

        Args:
            world_min: (x_min, y_min)
            world_max: (x_max, y_max)
        """
        self.world_min = np.array(world_min, dtype=np.float32)
        self.world_max = np.array(world_max, dtype=np.float32)

    def world_to_pixel(self, world_pos: np.ndarray) -> np.ndarray:
        """
        World 좌표를 픽셀 좌표로 변환

        Args:
            world_pos: (x, y) in world space

        Returns:
            (px, py) in pixel space
        """
        world_size = self.world_max - self.world_min
        normalized = (world_pos - self.world_min) / world_size

        # Y축 뒤집기 (screen space는 top-left origin)
        px = normalized[0] * self.width
        py = (1.0 - normalized[1]) * self.height

        return np.array([px, py])

    def pixel_to_world(self, pixel_pos: np.ndarray) -> np.ndarray:
        """
        픽셀 좌표를 world 좌표로 변환

        Args:
            pixel_pos: (px, py) in pixel space

        Returns:
            (x, y) in world space
        """
        # Y축 뒤집기
        normalized = np.array([
            pixel_pos[0] / self.width,
            1.0 - pixel_pos[1] / self.height
        ])

        world_size = self.world_max - self.world_min
        world_pos = self.world_min + normalized * world_size

        return world_pos

    def render(self, gaussians: List[Gaussian2D]) -> np.ndarray:
        """
        Gaussian 리스트를 2D 이미지로 렌더링

        Args:
            gaussians: List of Gaussian2D objects

        Returns:
            RGB 이미지 (height, width, 3) in range [0, 1]
        """
        # 초기화: 배경색
        image = np.ones((self.height, self.width, 3), dtype=np.float32) * self.background_color
        alpha_buffer = np.zeros((self.height, self.width), dtype=np.float32)

        # Depth sort (painter's algorithm)
        # z=0이지만 opacity에 따라 정렬
        sorted_gaussians = sorted(gaussians, key=lambda g: -g.position[2])

        # 각 Gaussian을 렌더링
        for gaussian in sorted_gaussians:
            self._render_single_gaussian(gaussian, image, alpha_buffer)

        return np.clip(image, 0.0, 1.0)

    def _render_single_gaussian(
        self,
        gaussian: Gaussian2D,
        image: np.ndarray,
        alpha_buffer: np.ndarray
    ):
        """
        단일 Gaussian을 이미지에 렌더링 (alpha blending)

        Args:
            gaussian: Gaussian2D object
            image: 렌더링할 이미지 (in-place 수정)
            alpha_buffer: Alpha accumulation buffer
        """
        # World 좌표 → 픽셀 좌표
        center_pixel = self.world_to_pixel(gaussian.position[:2])
        cx, cy = int(center_pixel[0]), int(center_pixel[1])

        # Covariance matrix 계산
        cov_world = gaussian.compute_covariance_2d()

        # World space covariance → Pixel space covariance
        # 스케일 factor 계산
        scale_x = self.width / (self.world_max[0] - self.world_min[0])
        scale_y = self.height / (self.world_max[1] - self.world_min[1])
        scale_matrix = np.diag([scale_x, scale_y])

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

        if x_min >= x_max or y_min >= y_max:
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

                # Blend
                contribution = alpha * (1.0 - accumulated_alpha)
                image[y, x] += contribution * gaussian.color
                alpha_buffer[y, x] += contribution

    def render_with_depth(
        self,
        gaussians: List[Gaussian2D]
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
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

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
        g.color = np.array([i/5.0, 0.5, 1.0 - i/5.0])
        gaussians.append(g)

    # 렌더링
    image = renderer.render(gaussians)

    print(f"Rendered image shape: {image.shape}")
    print(f"Image range: [{image.min()}, {image.max()}]")

    # 저장 (테스트용)
    renderer.save_image(gaussians, "test_render.png")


if __name__ == "__main__":
    test_renderer()
