"""
Inpainting for overlapping brush stamps

Phase 3-3: Simple opacity-based blending for overlapping regions
"""

import numpy as np
from typing import List, Tuple
from .gaussian import Gaussian2D


def find_overlapping_gaussians(
    stamp1: List[Gaussian2D],
    stamp2: List[Gaussian2D],
    threshold: float = 0.1
) -> List[Tuple[int, int, float]]:
    """
    Find overlapping Gaussian pairs between two stamps

    Args:
        stamp1: First stamp Gaussians
        stamp2: Second stamp Gaussians
        threshold: Distance threshold for overlap detection

    Returns:
        List of (index1, index2, distance) tuples for overlapping pairs
    """
    overlaps = []

    for i, g1 in enumerate(stamp1):
        for j, g2 in enumerate(stamp2):
            # Compute 2D distance (ignore z)
            dist = np.linalg.norm(g1.position[:2] - g2.position[:2])

            if dist < threshold:
                overlaps.append((i, j, dist))

    return overlaps


def compute_overlap_factor(distance: float, threshold: float) -> float:
    """
    Compute overlap factor based on distance

    Args:
        distance: Distance between Gaussians
        threshold: Overlap threshold

    Returns:
        Overlap factor in [0, 1], where 1 = complete overlap
    """
    if distance >= threshold:
        return 0.0

    # Linear falloff: 1.0 at distance=0, 0.0 at distance=threshold
    return max(0.0, 1.0 - distance / threshold)


def blend_overlapping_stamps(
    stamps: List[List[Gaussian2D]],
    overlap_threshold: float = 0.1,
    blend_strength: float = 0.3
) -> None:
    """
    Blend overlapping regions between consecutive stamps

    Modifies Gaussians in-place by reducing opacity in overlap regions

    Args:
        stamps: List of stamp Gaussian lists
        overlap_threshold: Distance threshold for overlap detection
        blend_strength: Maximum opacity reduction factor (0.0-1.0)
    """
    if len(stamps) < 2:
        return

    print(f"[Inpainting] Blending {len(stamps)} stamps with threshold={overlap_threshold:.3f}")

    total_overlaps = 0

    # Process consecutive stamp pairs
    for i in range(len(stamps) - 1):
        stamp1 = stamps[i]
        stamp2 = stamps[i + 1]

        # Find overlapping Gaussians
        overlaps = find_overlapping_gaussians(stamp1, stamp2, overlap_threshold)

        if len(overlaps) == 0:
            continue

        total_overlaps += len(overlaps)

        # Reduce opacity for overlapping Gaussians
        for idx1, idx2, dist in overlaps:
            # Compute overlap factor
            overlap_factor = compute_overlap_factor(dist, overlap_threshold)

            # Reduce opacity: opacity *= (1 - blend_strength * overlap_factor)
            # At complete overlap (factor=1.0), reduce by blend_strength
            # At threshold (factor=0.0), no reduction
            reduction = 1.0 - blend_strength * overlap_factor

            # Apply to both Gaussians
            stamp1[idx1].opacity *= reduction
            stamp2[idx2].opacity *= reduction

    print(f"[Inpainting] ✓ Blending complete: {total_overlaps} overlapping pairs processed")


def inpaint_stroke(
    stamps: List[List[Gaussian2D]],
    overlap_threshold: float = 0.1
) -> List[List[Gaussian2D]]:
    """
    Apply inpainting to a complete stroke

    This is a convenience function that applies blending to stamps
    and returns them (though blending is done in-place)

    Args:
        stamps: List of stamp Gaussian lists
        overlap_threshold: Distance threshold for overlap detection

    Returns:
        Same stamps list (modified in-place)
    """
    blend_overlapping_stamps(stamps, overlap_threshold, blend_strength=0.3)
    return stamps


def test_inpainting():
    """Test inpainting functions"""
    print("Testing inpainting...")

    # Create two simple stamps
    stamp1 = [
        Gaussian2D(
            position=np.array([0.0, 0.0, 0.0]),
            scale=np.array([0.05, 0.05, 1e-4]),
            rotation=np.array([0, 0, 0, 1]),
            opacity=0.8,
            color=np.array([1.0, 0.0, 0.0])
        )
    ]

    stamp2 = [
        Gaussian2D(
            position=np.array([0.05, 0.0, 0.0]),  # 0.05 units away
            scale=np.array([0.05, 0.05, 1e-4]),
            rotation=np.array([0, 0, 0, 1]),
            opacity=0.8,
            color=np.array([0.0, 1.0, 0.0])
        )
    ]

    print(f"Stamp 1 opacity before: {stamp1[0].opacity}")
    print(f"Stamp 2 opacity before: {stamp2[0].opacity}")

    # Find overlaps
    overlaps = find_overlapping_gaussians(stamp1, stamp2, threshold=0.1)
    print(f"Found {len(overlaps)} overlaps")

    # Blend
    blend_overlapping_stamps([stamp1, stamp2], overlap_threshold=0.1)

    print(f"Stamp 1 opacity after: {stamp1[0].opacity}")
    print(f"Stamp 2 opacity after: {stamp2[0].opacity}")


if __name__ == "__main__":
    test_inpainting()
