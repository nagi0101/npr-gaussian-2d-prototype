"""
Debug Visualizer for 2D Gaussian Splatting

Provides debug visualization overlays including:
- Gaussian origin points
- Local coordinate frames (X, Y, Z axes)
- Spline tangent/normal/binormal frames
- Deformation visualization
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .gaussian import Gaussian2D
from .quaternion_utils import quaternion_to_matrix
import cv2


class DebugVisualizer:
    """
    Debug visualization for Gaussian splatting painting system
    """

    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize debug visualizer

        Args:
            width: Canvas width
            height: Canvas height
        """
        self.width = width
        self.height = height

        # Debug settings
        self.show_gaussian_origins = True
        self.show_basis_vectors = True
        self.show_spline_frames = False
        self.show_deformation_vectors = False
        self.debug_opacity = 0.8

        # Visualization parameters
        self.origin_radius = 3
        self.basis_vector_length = 20
        self.axis_thickness = 2

        # Colors (BGR for OpenCV)
        self.colors = {
            'x_axis': (0, 0, 255),     # Red
            'y_axis': (0, 255, 0),     # Green
            'z_axis': (255, 0, 0),     # Blue
            'origin': (255, 255, 255), # White
            'spline': (255, 255, 0),   # Cyan
            'deformation': (255, 0, 255)  # Magenta
        }

    def create_debug_overlay(
        self,
        gaussians: List[Gaussian2D],
        image: Optional[np.ndarray] = None,
        spline_data: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Create debug overlay showing Gaussian frames

        Args:
            gaussians: List of Gaussians to visualize
            image: Base image to overlay on (if None, creates blank)
            spline_data: Optional spline visualization data

        Returns:
            Image with debug overlay
        """
        # Create base image if not provided
        if image is None:
            overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            overlay = image.copy()

        # Create transparent overlay for debug graphics
        debug_layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw Gaussian frames
        if self.show_gaussian_origins or self.show_basis_vectors:
            for g in gaussians:
                self._draw_gaussian_frame(debug_layer, g)

        # Draw spline frames if provided
        if self.show_spline_frames and spline_data is not None:
            self._draw_spline_frames(debug_layer, spline_data)

        # Blend debug layer with base image
        if image is not None:
            overlay = cv2.addWeighted(
                overlay, 1.0,
                debug_layer, self.debug_opacity,
                0
            )
        else:
            overlay = debug_layer

        return overlay

    def _draw_gaussian_frame(self, image: np.ndarray, gaussian: Gaussian2D):
        """
        Draw a single Gaussian's origin and basis vectors

        Args:
            image: Image to draw on
            gaussian: Gaussian to visualize
        """
        # Convert 3D position to 2D screen coordinates
        # Assuming orthographic projection for 2D painting
        origin_x = int(gaussian.position[0] * 100 + self.width / 2)
        origin_y = int(-gaussian.position[1] * 100 + self.height / 2)  # Flip Y for screen coords

        # Clamp to image bounds
        if not (0 <= origin_x < self.width and 0 <= origin_y < self.height):
            return

        # Draw origin point
        if self.show_gaussian_origins:
            cv2.circle(
                image,
                (origin_x, origin_y),
                self.origin_radius,
                self.colors['origin'],
                -1  # Filled circle
            )

        # Draw basis vectors
        if self.show_basis_vectors:
            # Get rotation matrix from quaternion
            R = quaternion_to_matrix(gaussian.rotation)

            # Scale basis vectors by Gaussian scale
            scale_factor = np.mean(gaussian.scale[:2]) * self.basis_vector_length

            # X axis (red)
            x_axis = R[:, 0]  # First column of rotation matrix
            x_end_x = int(origin_x + x_axis[0] * scale_factor)
            x_end_y = int(origin_y - x_axis[1] * scale_factor)  # Flip Y
            cv2.arrowedLine(
                image,
                (origin_x, origin_y),
                (x_end_x, x_end_y),
                self.colors['x_axis'],
                self.axis_thickness,
                tipLength=0.2
            )

            # Y axis (green)
            y_axis = R[:, 1]  # Second column of rotation matrix
            y_end_x = int(origin_x + y_axis[0] * scale_factor)
            y_end_y = int(origin_y - y_axis[1] * scale_factor)  # Flip Y
            cv2.arrowedLine(
                image,
                (origin_x, origin_y),
                (y_end_x, y_end_y),
                self.colors['y_axis'],
                self.axis_thickness,
                tipLength=0.2
            )

            # Z axis (blue) - minimal for 2D, just show as small point
            if not np.allclose(gaussian.position[2], 0):
                z_axis = R[:, 2]  # Third column of rotation matrix
                z_scale = scale_factor * 0.3  # Smaller for Z in 2D mode
                z_end_x = int(origin_x + z_axis[0] * z_scale)
                z_end_y = int(origin_y - z_axis[1] * z_scale)  # Flip Y
                cv2.arrowedLine(
                    image,
                    (origin_x, origin_y),
                    (z_end_x, z_end_y),
                    self.colors['z_axis'],
                    max(1, self.axis_thickness // 2),
                    tipLength=0.3
                )

    def _draw_spline_frames(self, image: np.ndarray, spline_data: Dict[str, Any]):
        """
        Draw spline tangent/normal/binormal frames

        Args:
            image: Image to draw on
            spline_data: Dictionary containing spline points and frames
        """
        if 'positions' not in spline_data or 'frames' not in spline_data:
            return

        positions = spline_data['positions']
        frames = spline_data['frames']

        for pos, (tangent, normal, binormal) in zip(positions, frames):
            # Convert to screen coordinates
            screen_x = int(pos[0] * 100 + self.width / 2)
            screen_y = int(-pos[1] * 100 + self.height / 2)

            if not (0 <= screen_x < self.width and 0 <= screen_y < self.height):
                continue

            # Draw frame vectors
            scale = 30  # Fixed scale for spline frames

            # Tangent (cyan)
            t_end_x = int(screen_x + tangent[0] * scale)
            t_end_y = int(screen_y - tangent[1] * scale)
            cv2.line(
                image,
                (screen_x, screen_y),
                (t_end_x, t_end_y),
                self.colors['spline'],
                1
            )

            # Normal (lighter cyan)
            n_end_x = int(screen_x + normal[0] * scale * 0.7)
            n_end_y = int(screen_y - normal[1] * scale * 0.7)
            cv2.line(
                image,
                (screen_x, screen_y),
                (n_end_x, n_end_y),
                (128, 255, 255),  # Lighter cyan
                1
            )

    def set_debug_options(self, options: Dict[str, Any]):
        """
        Update debug visualization options

        Args:
            options: Dictionary of debug options
        """
        if 'show_gaussian_origins' in options:
            self.show_gaussian_origins = options['show_gaussian_origins']
        if 'show_basis_vectors' in options:
            self.show_basis_vectors = options['show_basis_vectors']
        if 'show_spline_frames' in options:
            self.show_spline_frames = options['show_spline_frames']
        if 'show_deformation_vectors' in options:
            self.show_deformation_vectors = options['show_deformation_vectors']
        if 'debug_opacity' in options:
            self.debug_opacity = np.clip(options['debug_opacity'], 0.0, 1.0)
        if 'basis_vector_length' in options:
            self.basis_vector_length = max(5, options['basis_vector_length'])

    def create_deformation_comparison(
        self,
        original_gaussians: List[Gaussian2D],
        deformed_gaussians: List[Gaussian2D]
    ) -> np.ndarray:
        """
        Create side-by-side comparison of original vs deformed Gaussians

        Args:
            original_gaussians: Original Gaussians before deformation
            deformed_gaussians: Gaussians after deformation

        Returns:
            Side-by-side comparison image
        """
        # Create two panels
        left_panel = np.zeros((self.height, self.width // 2, 3), dtype=np.uint8)
        right_panel = np.zeros((self.height, self.width // 2, 3), dtype=np.uint8)

        # Draw original on left
        for g in original_gaussians:
            self._draw_gaussian_frame_panel(left_panel, g, offset_x=0)

        # Draw deformed on right
        for g in deformed_gaussians:
            self._draw_gaussian_frame_panel(right_panel, g, offset_x=self.width // 2)

        # Combine panels
        comparison = np.hstack([left_panel, right_panel])

        # Add labels
        cv2.putText(
            comparison, "Original", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            comparison, "Deformed", (self.width // 2 + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        return comparison

    def _draw_gaussian_frame_panel(
        self,
        panel: np.ndarray,
        gaussian: Gaussian2D,
        offset_x: int = 0
    ):
        """
        Draw Gaussian frame in a specific panel

        Args:
            panel: Panel image to draw on
            gaussian: Gaussian to visualize
            offset_x: X offset for positioning
        """
        # Adjust position for panel
        origin_x = int(gaussian.position[0] * 100 + panel.shape[1] / 2)
        origin_y = int(-gaussian.position[1] * 100 + panel.shape[0] / 2)

        # Clamp to panel bounds
        if not (0 <= origin_x < panel.shape[1] and 0 <= origin_y < panel.shape[0]):
            return

        # Draw origin
        cv2.circle(
            panel,
            (origin_x, origin_y),
            self.origin_radius,
            self.colors['origin'],
            -1
        )

        # Draw basis vectors (simplified for panel view)
        R = quaternion_to_matrix(gaussian.rotation)
        scale_factor = np.mean(gaussian.scale[:2]) * self.basis_vector_length

        # X axis
        x_axis = R[:, 0]
        x_end_x = int(origin_x + x_axis[0] * scale_factor)
        x_end_y = int(origin_y - x_axis[1] * scale_factor)
        cv2.line(
            panel,
            (origin_x, origin_y),
            (x_end_x, x_end_y),
            self.colors['x_axis'],
            self.axis_thickness
        )

        # Y axis
        y_axis = R[:, 1]
        y_end_x = int(origin_x + y_axis[0] * scale_factor)
        y_end_y = int(origin_y - y_axis[1] * scale_factor)
        cv2.line(
            panel,
            (origin_x, origin_y),
            (y_end_x, y_end_y),
            self.colors['y_axis'],
            self.axis_thickness
        )


def test_debug_visualizer():
    """Test debug visualization"""
    visualizer = DebugVisualizer()

    # Create test Gaussians
    gaussians = []
    for i in range(5):
        angle = i * np.pi / 4
        g = Gaussian2D(
            position=np.array([np.cos(angle) * 2, np.sin(angle) * 2, 0]),
            scale=np.array([0.1, 0.1, 0.001]),
            rotation=np.array([0, 0, np.sin(angle/2), np.cos(angle/2)]),  # Rotation around Z
            opacity=0.8,
            color=np.array([0.5, 0.5, 0.5])
        )
        gaussians.append(g)

    # Create debug overlay
    overlay = visualizer.create_debug_overlay(gaussians)

    # Save test image
    cv2.imwrite('debug_test.png', overlay)
    print("Debug visualization saved to debug_test.png")


if __name__ == "__main__":
    test_debug_visualizer()