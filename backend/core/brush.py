"""
BrushStamp and StrokePainter: Brush system for 3DGS painting

BrushStamp: Collection of Gaussians forming a brush pattern
StrokePainter: Applies brush along a stroke spline
"""

import numpy as np
from typing import List, Optional, Tuple
from .gaussian import Gaussian2D
from .spline import StrokeSpline
import copy


class BrushStamp:
    """
    Brush stamp: Collection of Gaussians with metadata

    브러시는 여러 Gaussian의 집합으로, 상대 위치를 유지하면서
    stroke를 따라 반복 배치됨
    """

    def __init__(self):
        """Initialize empty brush"""
        self.gaussians: List[Gaussian2D] = []
        self.center: np.ndarray = np.zeros(3, dtype=np.float32)

        # Orientation frame
        self.tangent: np.ndarray = np.array([1, 0, 0], dtype=np.float32)
        self.normal: np.ndarray = np.array([0, 0, 1], dtype=np.float32)
        self.binormal: np.ndarray = np.array([0, 1, 0], dtype=np.float32)

        # Brush parameters
        self.size: float = 1.0
        self.spacing: float = 0.2  # Arc length spacing between stamps

    def create_circular_pattern(
        self,
        num_gaussians: int = 20,
        radius: float = 0.5,
        gaussian_scale: float = 0.05,
        opacity: float = 0.8,
        color: Optional[np.ndarray] = None
    ):
        """
        Create circular brush pattern

        Args:
            num_gaussians: Number of Gaussians in circle
            radius: Circle radius
            gaussian_scale: Individual Gaussian scale
            opacity: Gaussian opacity
            color: RGB color (default: gray)
        """
        self.gaussians = []

        if color is None:
            color = np.array([0.5, 0.5, 0.5])

        for i in range(num_gaussians):
            angle = 2 * np.pi * i / num_gaussians
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            g = Gaussian2D(
                position=np.array([x, y, 0.0]),
                scale=np.array([gaussian_scale, gaussian_scale, 1e-4]),
                rotation=np.array([0, 0, 0, 1]),
                opacity=opacity,
                color=color.copy()
            )
            self.gaussians.append(g)

        self._update_center()

    def create_line_pattern(
        self,
        num_gaussians: int = 10,
        length: float = 1.0,
        thickness: float = 0.05,
        opacity: float = 0.8,
        color: Optional[np.ndarray] = None
    ):
        """
        Create line brush pattern (for stroke-like brushes)

        Args:
            num_gaussians: Number of Gaussians
            length: Line length
            thickness: Gaussian thickness
            opacity: Gaussian opacity
            color: RGB color
        """
        self.gaussians = []

        if color is None:
            color = np.array([0.5, 0.5, 0.5])

        for i in range(num_gaussians):
            # Distribute along line
            t = -0.5 + i / (num_gaussians - 1) if num_gaussians > 1 else 0
            x = t * length

            g = Gaussian2D(
                position=np.array([x, 0.0, 0.0]),
                scale=np.array([thickness, thickness, 1e-4]),
                rotation=np.array([0, 0, 0, 1]),
                opacity=opacity,
                color=color.copy()
            )
            self.gaussians.append(g)

        self._update_center()

    def create_grid_pattern(
        self,
        grid_size: int = 5,
        spacing: float = 0.1,
        gaussian_scale: float = 0.04,
        opacity: float = 0.8,
        color: Optional[np.ndarray] = None
    ):
        """
        Create grid brush pattern

        Args:
            grid_size: NxN grid size
            spacing: Spacing between Gaussians
            gaussian_scale: Individual Gaussian scale
            opacity: Gaussian opacity
            color: RGB color
        """
        self.gaussians = []

        if color is None:
            color = np.array([0.5, 0.5, 0.5])

        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - grid_size // 2) * spacing
                y = (j - grid_size // 2) * spacing

                g = Gaussian2D(
                    position=np.array([x, y, 0.0]),
                    scale=np.array([gaussian_scale, gaussian_scale, 1e-4]),
                    rotation=np.array([0, 0, 0, 1]),
                    opacity=opacity,
                    color=color.copy()
                )
                self.gaussians.append(g)

        self._update_center()

    def _update_center(self):
        """Update brush center as mean of Gaussian positions"""
        if len(self.gaussians) == 0:
            self.center = np.zeros(3, dtype=np.float32)
            return

        positions = np.array([g.position for g in self.gaussians])
        self.center = np.mean(positions, axis=0)

    def place_at(
        self,
        position: np.ndarray,
        tangent: np.ndarray,
        normal: np.ndarray
    ) -> List[Gaussian2D]:
        """
        Place stamp at given position and orientation (rigid transform)

        Args:
            position: 3D world position
            tangent: Tangent direction (t)
            normal: Normal direction (n)

        Returns:
            List of transformed Gaussians
        """
        # Compute binormal: b = n × t
        binormal = np.cross(normal, tangent)
        binormal_norm = np.linalg.norm(binormal)
        if binormal_norm > 1e-8:
            binormal = binormal / binormal_norm
        else:
            binormal = self.binormal

        # Build rotation matrix: brush frame → world frame
        # Brush frame: (tangent_B, normal_B, binormal_B)
        # World frame: (tangent, normal, binormal)

        # Source frame (brush)
        R_src = np.column_stack([self.tangent, self.binormal, self.normal])

        # Target frame (world)
        R_tgt = np.column_stack([tangent, binormal, normal])

        # Rotation: R = R_tgt @ R_src^T
        R = R_tgt @ R_src.T

        # Build 4x4 transform matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = position - (R @ self.center)

        # Transform all Gaussians
        placed_gaussians = []
        for g in self.gaussians:
            g_new = g.transform(T)
            placed_gaussians.append(g_new)

        return placed_gaussians

    def add_gaussian(self, gaussian: Gaussian2D):
        """Add a Gaussian to the brush"""
        self.gaussians.append(gaussian)
        self._update_center()

    def set_color(self, color: np.ndarray):
        """Set color for all Gaussians"""
        for g in self.gaussians:
            g.color = color.copy()

    def set_opacity(self, opacity: float):
        """Set opacity for all Gaussians"""
        for g in self.gaussians:
            g.opacity = opacity

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of brush

        Returns:
            (min_pos, max_pos) - 3D bounds
        """
        if len(self.gaussians) == 0:
            return np.zeros(3), np.zeros(3)

        positions = np.array([g.position for g in self.gaussians])
        return positions.min(axis=0), positions.max(axis=0)

    def copy(self) -> 'BrushStamp':
        """Deep copy of brush"""
        brush_copy = BrushStamp()
        brush_copy.gaussians = [g.copy() for g in self.gaussians]
        brush_copy.center = self.center.copy()
        brush_copy.tangent = self.tangent.copy()
        brush_copy.normal = self.normal.copy()
        brush_copy.binormal = self.binormal.copy()
        brush_copy.size = self.size
        brush_copy.spacing = self.spacing
        return brush_copy

    def __len__(self) -> int:
        return len(self.gaussians)

    def __repr__(self) -> str:
        return f"BrushStamp(gaussians={len(self.gaussians)}, spacing={self.spacing})"


class StrokePainter:
    """
    Stroke painting logic

    브러시를 사용하여 spline을 따라 stamp를 배치
    """

    def __init__(
        self,
        brush: BrushStamp,
        scene_gaussians: Optional[List[Gaussian2D]] = None
    ):
        """
        Initialize painter

        Args:
            brush: Brush stamp to use
            scene_gaussians: Existing scene Gaussians (in-place 수정)
        """
        self.brush = brush
        self.scene = scene_gaussians if scene_gaussians is not None else []
        self.current_stroke: Optional[StrokeSpline] = None
        self.placed_stamps: List[List[Gaussian2D]] = []
        self.last_stamp_arc_length: float = 0.0

    def start_stroke(self, position: np.ndarray, normal: np.ndarray):
        """
        Start a new stroke

        Args:
            position: 3D starting position
            normal: Surface normal
        """
        self.current_stroke = StrokeSpline()
        self.current_stroke.add_point(position, normal, threshold=0.0)
        self.placed_stamps = []
        self.last_stamp_arc_length = 0.0

    def update_stroke(self, position: np.ndarray, normal: np.ndarray):
        """
        Update stroke with new point

        Args:
            position: 3D position
            normal: Surface normal
        """
        if self.current_stroke is None:
            self.start_stroke(position, normal)
            return

        # Add point to spline
        added = self.current_stroke.add_point(position, normal, threshold=self.brush.spacing * 0.1)

        if not added:
            return

        # Place new stamps along spline
        self._place_new_stamps()

    def _place_new_stamps(self):
        """Place stamps from last_stamp_arc_length to current end"""
        if self.current_stroke is None:
            return

        total_length = self.current_stroke.total_arc_length
        spacing = self.brush.spacing

        # Calculate stamp positions
        arc_length = self.last_stamp_arc_length

        while arc_length <= total_length:
            # Get position, tangent, normal at this arc length
            position = self.current_stroke.evaluate_at_arc_length(arc_length)
            tangent = self.current_stroke.get_tangent_at_arc_length(arc_length)
            normal = self.current_stroke.get_normal_at_arc_length(arc_length)

            # Place stamp
            stamp_gaussians = self.brush.place_at(position, tangent, normal)

            # Store stamp
            self.placed_stamps.append(stamp_gaussians)

            # Add to scene
            self.scene.extend(stamp_gaussians)

            arc_length += spacing

        # Update last stamp position
        self.last_stamp_arc_length = arc_length - spacing

    def finish_stroke(self):
        """
        Finish current stroke

        여기서 deformation이나 inpainting 적용 가능 (Phase 2)
        """
        if self.current_stroke is None:
            return

        # TODO: Apply non-rigid deformation
        # TODO: Apply diffusion inpainting

        # Reset
        self.current_stroke = None
        self.placed_stamps = []
        self.last_stamp_arc_length = 0.0

    def get_stroke_gaussians(self) -> List[Gaussian2D]:
        """
        Get all Gaussians from current stroke

        Returns:
            List of Gaussians in current stroke
        """
        result = []
        for stamp in self.placed_stamps:
            result.extend(stamp)
        return result

    def clear_scene(self):
        """Clear all scene Gaussians"""
        self.scene.clear()
        self.placed_stamps = []


def test_brush():
    """Test brush functionality"""
    # Create circular brush
    brush = BrushStamp()
    brush.create_circular_pattern(num_gaussians=20, radius=0.3)
    print(f"Created brush: {brush}")

    # Test placement
    position = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    placed = brush.place_at(position, tangent, normal)
    print(f"Placed {len(placed)} Gaussians")

    # Test painter
    painter = StrokePainter(brush)
    painter.start_stroke(np.array([0.0, 0.0, 0.0]), np.array([0, 0, 1]))
    painter.update_stroke(np.array([0.5, 0.2, 0.0]), np.array([0, 0, 1]))
    painter.update_stroke(np.array([1.0, 0.1, 0.0]), np.array([0, 0, 1]))
    painter.finish_stroke()

    print(f"Total scene Gaussians: {len(painter.scene)}")


if __name__ == "__main__":
    test_brush()
