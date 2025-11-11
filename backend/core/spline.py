"""
StrokeSpline: 3D cubic spline with arc-length parameterization

마우스 입력 포인트를 smooth한 spline으로 변환하고,
브러시 스탬프를 등간격으로 배치하기 위한 arc-length 파라미터화 제공
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.integrate import cumtrapz


class StrokeSpline:
    """
    3D cubic spline for brush strokes

    Features:
    - Smooth interpolation of input points
    - Arc-length parameterization for uniform stamp spacing
    - Tangent and curvature computation for deformation
    """

    def __init__(self):
        """Initialize empty spline"""
        self.control_points: List[np.ndarray] = []  # 3D points
        self.normals: List[np.ndarray] = []  # Surface normals at each point
        self.spline: Optional[CubicSpline] = None
        self.spline_x: Optional[object] = None  # Parametric spline (t -> x)
        self.spline_y: Optional[object] = None  # Parametric spline (t -> y)
        self.spline_z: Optional[object] = None  # Parametric spline (t -> z)
        self.arc_lengths: Optional[np.ndarray] = None
        self.total_arc_length: float = 0.0

    def add_point(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        threshold: float = 0.01
    ) -> bool:
        """
        Add a point to the spline

        Args:
            point: 3D position (x, y, z)
            normal: Surface normal at this point
            threshold: Minimum distance from previous point

        Returns:
            True if point was added, False if too close to previous
        """
        if len(self.control_points) == 0:
            self.control_points.append(np.array(point, dtype=np.float32))
            self.normals.append(np.array(normal, dtype=np.float32))
            return True

        # Check distance threshold
        distance = np.linalg.norm(point - self.control_points[-1])
        if distance < threshold:
            return False

        self.control_points.append(np.array(point, dtype=np.float32))
        self.normals.append(np.array(normal, dtype=np.float32))

        # Refit spline
        self._refit_spline()

        return True

    def _refit_spline(self):
        """Recompute the spline from control points"""
        if len(self.control_points) < 2:
            self.spline = None
            return

        points = np.array(self.control_points)

        if len(points) == 2:
            # Linear interpolation for 2 points
            t = np.array([0, 1])
            self.spline_x = lambda s: np.interp(s, t, points[:, 0])
            self.spline_y = lambda s: np.interp(s, t, points[:, 1])
            self.spline_z = lambda s: np.interp(s, t, points[:, 2])
        elif len(points) == 3:
            # Quadratic for 3 points
            t = np.linspace(0, 1, len(points))
            self.spline_x = CubicSpline(t, points[:, 0], bc_type='natural')
            self.spline_y = CubicSpline(t, points[:, 1], bc_type='natural')
            self.spline_z = CubicSpline(t, points[:, 2], bc_type='natural')
        else:
            # Cubic spline for 4+ points
            t = np.linspace(0, 1, len(points))
            self.spline_x = CubicSpline(t, points[:, 0], bc_type='natural')
            self.spline_y = CubicSpline(t, points[:, 1], bc_type='natural')
            self.spline_z = CubicSpline(t, points[:, 2], bc_type='natural')

        # Compute arc-length parameterization
        self._compute_arc_length()

    def _compute_arc_length(self):
        """
        Compute arc-length parameterization

        Arc length: L(t) = ∫₀ᵗ ||ds/du|| du

        저장:
        - arc_lengths: t 파라미터에 대응하는 arc length 배열
        - total_arc_length: 전체 spline의 arc length
        """
        if self.spline_x is None:
            return

        # Sample points along spline
        t_samples = np.linspace(0, 1, 100)
        positions = np.array([
            [self.spline_x(t), self.spline_y(t), self.spline_z(t)]
            for t in t_samples
        ])

        # Compute segment lengths
        segment_vectors = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)

        # Cumulative arc length
        self.arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        self.total_arc_length = self.arc_lengths[-1]

        # Store t_samples for inverse lookup
        self.t_samples = t_samples

    def evaluate_at_t(self, t: float) -> np.ndarray:
        """
        Evaluate spline at parameter t

        Args:
            t: Parameter in [0, 1]

        Returns:
            3D position
        """
        if self.spline_x is None:
            if len(self.control_points) > 0:
                return self.control_points[0]
            return np.zeros(3)

        t = np.clip(t, 0.0, 1.0)

        return np.array([
            self.spline_x(t),
            self.spline_y(t),
            self.spline_z(t)
        ], dtype=np.float32)

    def evaluate_at_arc_length(self, arc_length: float) -> np.ndarray:
        """
        Evaluate spline at given arc length

        Args:
            arc_length: Arc length from start

        Returns:
            3D position
        """
        if self.arc_lengths is None or self.total_arc_length == 0:
            return self.evaluate_at_t(0.0)

        # Clamp arc length
        arc_length = np.clip(arc_length, 0.0, self.total_arc_length)

        # Find t corresponding to this arc length
        t = np.interp(arc_length, self.arc_lengths, self.t_samples)

        return self.evaluate_at_t(t)

    def get_tangent_at_t(self, t: float) -> np.ndarray:
        """
        Get tangent vector at parameter t

        Args:
            t: Parameter in [0, 1]

        Returns:
            Normalized tangent vector
        """
        if self.spline_x is None:
            return np.array([1, 0, 0], dtype=np.float32)

        t = np.clip(t, 0.0, 1.0)

        # Derivative
        if isinstance(self.spline_x, CubicSpline):
            dx = self.spline_x.derivative()(t)
            dy = self.spline_y.derivative()(t)
            dz = self.spline_z.derivative()(t)
        else:
            # Numerical derivative for simple interpolation
            eps = 0.001
            t_plus = min(t + eps, 1.0)
            t_minus = max(t - eps, 0.0)

            pos_plus = self.evaluate_at_t(t_plus)
            pos_minus = self.evaluate_at_t(t_minus)

            derivative = (pos_plus - pos_minus) / (t_plus - t_minus)
            return derivative / (np.linalg.norm(derivative) + 1e-8)

        tangent = np.array([dx, dy, dz], dtype=np.float32)
        norm = np.linalg.norm(tangent)

        if norm < 1e-8:
            return np.array([1, 0, 0], dtype=np.float32)

        return tangent / norm

    def get_tangent_at_arc_length(self, arc_length: float) -> np.ndarray:
        """
        Get tangent at given arc length

        Args:
            arc_length: Arc length from start

        Returns:
            Normalized tangent vector
        """
        if self.arc_lengths is None or self.total_arc_length == 0:
            return self.get_tangent_at_t(0.0)

        arc_length = np.clip(arc_length, 0.0, self.total_arc_length)
        t = np.interp(arc_length, self.arc_lengths, self.t_samples)

        return self.get_tangent_at_t(t)

    def get_normal_at_arc_length(self, arc_length: float) -> np.ndarray:
        """
        Get surface normal at given arc length

        Args:
            arc_length: Arc length from start

        Returns:
            Normalized normal vector (interpolated from control points)
        """
        if len(self.normals) == 0:
            return np.array([0, 0, 1], dtype=np.float32)

        if self.total_arc_length == 0:
            return self.normals[0]

        # Find nearest control point
        # 간단한 버전: 가장 가까운 control point의 normal 사용
        arc_length = np.clip(arc_length, 0.0, self.total_arc_length)
        ratio = arc_length / self.total_arc_length
        index = int(ratio * (len(self.normals) - 1))
        index = np.clip(index, 0, len(self.normals) - 1)

        return self.normals[index]

    def get_binormal_at_arc_length(self, arc_length: float) -> np.ndarray:
        """
        Get binormal vector at given arc length

        binormal = normal × tangent

        Args:
            arc_length: Arc length from start

        Returns:
            Normalized binormal vector
        """
        tangent = self.get_tangent_at_arc_length(arc_length)
        normal = self.get_normal_at_arc_length(arc_length)

        binormal = np.cross(normal, tangent)
        norm = np.linalg.norm(binormal)

        if norm < 1e-8:
            # Fallback: perpendicular to tangent in xy plane
            binormal = np.array([-tangent[1], tangent[0], 0], dtype=np.float32)
            norm = np.linalg.norm(binormal)
            if norm < 1e-8:
                binormal = np.array([0, 1, 0], dtype=np.float32)
                return binormal

        return binormal / norm

    def get_frame_at_arc_length(self, arc_length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get orientation frame (tangent, normal, binormal) at arc length

        Args:
            arc_length: Arc length from start

        Returns:
            (tangent, normal, binormal) - orthonormal frame
        """
        tangent = self.get_tangent_at_arc_length(arc_length)
        normal = self.get_normal_at_arc_length(arc_length)
        binormal = self.get_binormal_at_arc_length(arc_length)

        return tangent, normal, binormal

    def get_curvature_at_t(self, t: float) -> float:
        """
        Get curvature at parameter t

        κ = ||ds/dt × d²s/dt²|| / ||ds/dt||³

        Args:
            t: Parameter in [0, 1]

        Returns:
            Curvature value
        """
        if self.spline_x is None or not isinstance(self.spline_x, CubicSpline):
            return 0.0

        t = np.clip(t, 0.0, 1.0)

        # First derivative
        dx1 = self.spline_x.derivative(1)(t)
        dy1 = self.spline_y.derivative(1)(t)
        dz1 = self.spline_z.derivative(1)(t)
        first_deriv = np.array([dx1, dy1, dz1])

        # Second derivative
        dx2 = self.spline_x.derivative(2)(t)
        dy2 = self.spline_y.derivative(2)(t)
        dz2 = self.spline_z.derivative(2)(t)
        second_deriv = np.array([dx2, dy2, dz2])

        # Curvature formula
        cross = np.cross(first_deriv, second_deriv)
        numerator = np.linalg.norm(cross)
        denominator = np.linalg.norm(first_deriv) ** 3

        if denominator < 1e-8:
            return 0.0

        return numerator / denominator

    def sample_by_arc_length(
        self,
        spacing: float
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Sample points along spline at uniform arc-length intervals

        Args:
            spacing: Arc length spacing between samples

        Returns:
            List of (position, tangent, normal) tuples
        """
        if self.total_arc_length == 0 or spacing <= 0:
            return []

        samples = []
        arc_length = 0.0

        while arc_length <= self.total_arc_length:
            position = self.evaluate_at_arc_length(arc_length)
            tangent = self.get_tangent_at_arc_length(arc_length)
            normal = self.get_normal_at_arc_length(arc_length)

            samples.append((position, tangent, normal))

            arc_length += spacing

        return samples

    def get_num_points(self) -> int:
        """Get number of control points"""
        return len(self.control_points)

    def get_last_point(self) -> Optional[np.ndarray]:
        """Get last control point"""
        if len(self.control_points) == 0:
            return None
        return self.control_points[-1]

    def clear(self):
        """Clear all data"""
        self.control_points = []
        self.normals = []
        self.spline = None
        self.spline_x = None
        self.spline_y = None
        self.spline_z = None
        self.arc_lengths = None
        self.total_arc_length = 0.0

    def __repr__(self) -> str:
        return (f"StrokeSpline(points={len(self.control_points)}, "
                f"length={self.total_arc_length:.3f})")


def test_spline():
    """Test spline functionality"""
    spline = StrokeSpline()

    # Add some test points
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.3, 0.2, 0.0]),
        np.array([0.6, 0.1, 0.0]),
        np.array([1.0, 0.3, 0.0]),
    ]

    normal = np.array([0, 0, 1])

    for p in points:
        spline.add_point(p, normal, threshold=0.0)

    print(f"Spline: {spline}")
    print(f"Total arc length: {spline.total_arc_length:.4f}")

    # Sample
    samples = spline.sample_by_arc_length(spacing=0.1)
    print(f"Number of samples with spacing 0.1: {len(samples)}")

    # Test evaluation
    for i, (pos, tangent, normal) in enumerate(samples[:3]):
        print(f"Sample {i}: pos={pos}, tangent={tangent}")


if __name__ == "__main__":
    test_spline()
