"""
Quaternion utilities for smooth rotation interpolation
"""

import numpy as np


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions

    Args:
        q1: Starting quaternion [x, y, z, w]
        q2: Ending quaternion [x, y, z, w]
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated quaternion
    """
    # Ensure quaternions are normalized
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute dot product
    dot = np.dot(q1, q2)

    # If the quaternions are nearly identical, return q2
    if dot > 0.9995:
        return q1 + t * (q2 - q1)

    # If the dot product is negative, slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # Clamp dot product to avoid numerical errors
    dot = np.clip(dot, -1, 1)

    # Calculate the angle between quaternions
    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    # Calculate the interpolated quaternion
    q2_orthogonal = q2 - q1 * dot
    q2_orthogonal = q2_orthogonal / np.linalg.norm(q2_orthogonal)

    result = q1 * np.cos(theta) + q2_orthogonal * np.sin(theta)

    # Normalize result
    return result / np.linalg.norm(result)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions

    Args:
        q1: First quaternion [x, y, z, w]
        q2: Second quaternion [x, y, z, w]

    Returns:
        Product quaternion q1 * q2
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion

    Args:
        matrix: 3x3 rotation matrix

    Returns:
        Quaternion [x, y, z, w]
    """
    trace = np.trace(matrix)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s

    return np.array([x, y, z, w])


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix

    Args:
        q: Quaternion [x, y, z, w]

    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = q

    # Normalize quaternion
    norm = np.linalg.norm(q)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])