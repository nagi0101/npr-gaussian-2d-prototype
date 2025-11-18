"""
Brush Converter: 2D Image to 3DGS Brush Conversion Pipeline

Main orchestrator for converting 2D brush stroke images into
3D Gaussian Splatting brush stamps.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from scipy.spatial import KDTree
from skimage import morphology, measure
from scipy import ndimage

from .gaussian import Gaussian2D
from .brush import BrushStamp
from .depth_estimator import create_depth_estimator, MiDaSDepthEstimator
from .quaternion_utils import quaternion_from_axis_angle

# Configure logging
logger = logging.getLogger(__name__)


class BrushConverter:
    """
    Converts 2D brush stroke images to 3D Gaussian Splatting brushes

    Pipeline stages:
    1. Depth estimation (MiDaS or heuristic)
    2. Point cloud generation
    3. Feature extraction (medial axis, gradients)
    4. Gaussian initialization
    5. Procedural refinement
    6. Appearance optimization (future)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        use_midas: bool = True,
        target_gaussian_count: int = 200,  # Increased from 50 for better quality with importance sampling
    ):
        """
        Initialize brush converter

        Args:
            device: Device for computation ('cuda', 'cpu', or None for auto)
            use_midas: Whether to use MiDaS for depth (fallback to heuristics)
            target_gaussian_count: Target number of Gaussians per brush
        """
        self.device = device or (
            "cuda" if np.random.rand() > 1.5 else "cpu"
        )  # For now use CPU
        self.target_gaussian_count = target_gaussian_count

        # Initialize depth estimator
        self.depth_estimator = create_depth_estimator(
            prefer_midas=use_midas, device=self.device
        )

        logger.info(
            f"[BrushConverter] Initialized with target {target_gaussian_count} Gaussians"
        )

    def convert_2d_to_3dgs(
        self,
        image: np.ndarray,
        brush_name: str = "converted_brush",
        depth_profile: str = "convex",
        depth_scale: float = 0.2,
        optimization_steps: int = 0,  # Disabled for initial implementation
    ) -> BrushStamp:
        """
        Main conversion pipeline: 2D image → 3DGS brush

        Args:
            image: Input brush stroke image (H, W, 3) or (H, W, 4) with alpha
            brush_name: Name for the converted brush
            depth_profile: Depth estimation profile ("flat", "convex", "concave", "wavy")
            depth_scale: Scale factor for depth (0.1-2.0)
            optimization_steps: Number of appearance optimization iterations (0 to skip)

        Returns:
            BrushStamp containing converted Gaussians
        """
        logger.info(
            f"[BrushConverter] Converting '{brush_name}' with {depth_profile} profile"
        )

        # Step 1: Depth estimation
        depth_map = self._estimate_depth(image, depth_profile, depth_scale)

        # Step 2: Extract alpha mask
        alpha_mask = self._extract_alpha_mask(image)

        # Step 3: Feature extraction (before point cloud for importance sampling)
        features = self._extract_features(image, alpha_mask)

        # Step 4: Generate point cloud with importance-based sampling
        points, colors, normals = self._generate_point_cloud(
            image, depth_map, alpha_mask, features=features
        )

        # Step 5: Initialize Gaussians
        gaussians = self._initialize_gaussians(points, colors, normals, features)

        # Step 6: Procedural refinement
        gaussians = self._refine_gaussians(gaussians, features)

        # Step 7: Appearance optimization (if requested)
        if optimization_steps > 0:
            gaussians = self._optimize_appearance(gaussians, image, optimization_steps)

        # Step 8: Create BrushStamp
        brush = self._create_brush_stamp(gaussians, brush_name)

        logger.info(f"[BrushConverter] ✓ Created brush with {len(gaussians)} Gaussians")

        return brush

    def _estimate_depth(
        self, image: np.ndarray, profile: str, scale: float
    ) -> np.ndarray:
        """
        Estimate depth map from image

        Args:
            image: Input image
            profile: Depth profile preset
            scale: Depth scale factor

        Returns:
            Depth map (H, W) in range [0, scale]
        """
        # Use depth estimator with specified profile
        depth_map = self.depth_estimator.estimate_with_profiles(
            image, profile=profile, strength=scale
        )

        # Scale to desired range
        depth_map = depth_map * scale

        return depth_map

    def _extract_alpha_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Extract alpha channel or create from intensity

        Args:
            image: Input image

        Returns:
            Alpha mask (H, W) in range [0, 1]
        """
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Use existing alpha channel
            alpha = image[:, :, 3].astype(np.float32) / 255.0
        else:
            # Create from grayscale intensity
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Threshold to create mask
            _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            alpha = alpha.astype(np.float32) / 255.0

        # Smooth edges
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0.5)

        return alpha

    def _generate_point_cloud(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        alpha_mask: np.ndarray,
        alpha_threshold: float = 0.2,  # Increased from 0.1 to filter more semi-transparent pixels
        features: Dict[str, np.ndarray] = None,  # Added for importance-based sampling
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3D point cloud from image and depth using importance-based adaptive sampling

        Args:
            image: Color image
            depth_map: Depth values
            alpha_mask: Opacity mask
            alpha_threshold: Minimum alpha to include point
            features: Pre-computed features (medial_axis, thickness_map, flow_field) for importance sampling

        Returns:
            (points, colors, normals) arrays
        """
        h, w = depth_map.shape

        # Preserve aspect ratio and normalize to fixed size
        aspect = w / h
        target_world_size = 0.5  # Maximum extent (adjustable)

        if w > h:
            # Wide image
            x_range = target_world_size
            y_range = target_world_size / aspect
        else:
            # Tall image (like vertical drip)
            x_range = target_world_size * aspect
            y_range = target_world_size

        # Create virtual camera (orthographic projection)
        xx, yy = np.meshgrid(
            np.linspace(-x_range, x_range, w),
            np.linspace(-y_range, y_range, h)
        )

        # Create point cloud
        points = []
        colors = []
        normals = []

        # Sample points based on alpha mask
        mask = alpha_mask > alpha_threshold

        # Importance-based adaptive sampling
        total_pixels = np.sum(mask)
        if total_pixels > self.target_gaussian_count:
            # Compute importance map for quality-aware sampling
            importance_map = self._compute_importance_map(
                image, depth_map, alpha_mask, features, mask
            )

            # Weighted sampling based on importance
            mask = self._importance_based_sampling(
                mask, importance_map, self.target_gaussian_count
            )

        # Compute luminance map with contrast enhancement
        if len(image.shape) == 3:
            color_bgr = image[:, :, :3] / 255.0
            luminance_map = 0.114 * color_bgr[:, :, 0] + 0.587 * color_bgr[:, :, 1] + 0.299 * color_bgr[:, :, 2]
        else:
            luminance_map = image / 255.0

        # Apply contrast stretching to enhance texture detail
        # Only stretch within valid mask region to avoid background affecting range
        valid_luminance = luminance_map[mask]
        if len(valid_luminance) > 0:
            lum_min = np.percentile(valid_luminance, 5)  # Use percentiles to avoid outliers
            lum_max = np.percentile(valid_luminance, 95)

            # Ensure minimum contrast range
            if lum_max - lum_min < 0.2:
                # Expand range to at least 0.2
                center = (lum_min + lum_max) / 2
                lum_min = max(0.0, center - 0.1)
                lum_max = min(1.0, center + 0.1)

            # Apply contrast stretching
            luminance_map = np.clip((luminance_map - lum_min) / (lum_max - lum_min + 1e-8), 0.0, 1.0)

        # Extract points where mask is valid
        valid_indices = np.where(mask)

        for i, j in zip(valid_indices[0], valid_indices[1]):
            # 3D position
            x = xx[i, j]
            y = yy[i, j]
            z = depth_map[i, j]
            points.append([x, y, z])

            # Get enhanced luminance from pre-computed map
            luminance = luminance_map[i, j]

            # Store luminance as grayscale color (preserves brightness variations = texture)
            # This will be tinted with runtime color while preserving luminance pattern
            color = [luminance, luminance, luminance]
            colors.append(color)

            # Normal (computed from depth gradient)
            normal = self._compute_normal_at_point(depth_map, i, j, xx, yy)
            normals.append(normal)

        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)

        logger.info(
            f"[BrushConverter] Generated {len(points)} points from {h}x{w} image"
        )

        return points, colors, normals

    def _compute_normal_at_point(
        self, depth_map: np.ndarray, i: int, j: int, xx: np.ndarray, yy: np.ndarray
    ) -> np.ndarray:
        """
        Compute surface normal from depth gradient

        Args:
            depth_map: Depth values
            i, j: Pixel coordinates
            xx, yy: World coordinate grids

        Returns:
            Normal vector (3,)
        """
        h, w = depth_map.shape

        # Compute gradients using finite differences
        if j > 0 and j < w - 1:
            dz_dx = (depth_map[i, j + 1] - depth_map[i, j - 1]) / (
                xx[i, j + 1] - xx[i, j - 1]
            )
        else:
            dz_dx = 0

        if i > 0 and i < h - 1:
            dz_dy = (depth_map[i + 1, j] - depth_map[i - 1, j]) / (
                yy[i + 1, j] - yy[i - 1, j]
            )
        else:
            dz_dy = 0

        # Normal = (-dz/dx, -dz/dy, 1) normalized
        normal = np.array([-dz_dx, -dz_dy, 1.0])
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        return normal

    def _compute_importance_map(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        alpha_mask: np.ndarray,
        features: Dict[str, np.ndarray],
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute importance map for adaptive sampling.
        Higher values indicate regions that should have more Gaussians.

        Strategy:
        - High gradient regions (edges, texture) → important
        - Thick regions of stroke → important
        - Near medial axis (stroke centerline) → important

        Args:
            image: Input image
            depth_map: Depth map
            alpha_mask: Alpha channel
            features: Pre-computed features dict
            mask: Valid pixel mask

        Returns:
            Importance map (same shape as image)
        """
        h, w = alpha_mask.shape

        # Initialize importance map
        importance = np.zeros((h, w), dtype=np.float32)

        # Component 1: Gradient magnitude (edges and texture)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (5, 5), 1.0)
        gradient_normalized = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)

        # Component 2: Thickness map (thick regions need more Gaussians for coverage)
        if features is not None and 'thickness_map' in features:
            thickness_map = features['thickness_map']
            thickness_normalized = thickness_map / (np.max(thickness_map) + 1e-8)
        else:
            # Fallback: use alpha as proxy for thickness
            thickness_normalized = alpha_mask

        # Component 3: Distance to skeleton (closer = more important for structure)
        if features is not None and 'medial_axis' in features:
            skeleton = features['medial_axis']
            # Distance transform from skeleton
            skeleton_dist = ndimage.distance_transform_edt(~skeleton.astype(bool))
            # Invert: closer to skeleton = higher importance
            max_dist = np.max(skeleton_dist[mask]) if np.any(mask) else 1.0
            skeleton_importance = 1.0 - (skeleton_dist / (max_dist + 1e-8))
            skeleton_importance = np.clip(skeleton_importance, 0, 1)
        else:
            # Fallback: uniform importance
            skeleton_importance = np.ones((h, w), dtype=np.float32)

        # Combine importance components with weights
        # Gradient: 20% - edges and details (reduced to avoid over-clustering at edges)
        # Thickness: 40% - coverage of thick areas (increased for better fill)
        # Skeleton: 40% - structural integrity (increased for even distribution)
        importance = (
            0.2 * gradient_normalized +
            0.4 * thickness_normalized +
            0.4 * skeleton_importance
        )

        # Apply alpha mask (only consider visible regions)
        importance = importance * (alpha_mask > 0.1)

        # Normalize to [0, 1]
        importance = importance / (np.max(importance) + 1e-8)

        logger.info(
            f"[ImportanceSampling] Importance map computed. "
            f"Mean: {np.mean(importance[mask]):.3f}, "
            f"Max: {np.max(importance[mask]):.3f}"
        )

        return importance

    def _importance_based_sampling(
        self,
        mask: np.ndarray,
        importance_map: np.ndarray,
        target_count: int,
    ) -> np.ndarray:
        """
        Perform weighted random sampling based on importance map.

        Args:
            mask: Binary mask of valid pixels
            importance_map: Importance values for each pixel
            target_count: Number of points to sample

        Returns:
            Updated mask with sampled points
        """
        h, w = mask.shape

        # Get indices of valid pixels
        valid_indices = np.where(mask)
        num_valid = len(valid_indices[0])

        if num_valid <= target_count:
            # No need to downsample
            return mask

        # Extract importance values for valid pixels
        importance_values = importance_map[valid_indices]

        # Normalize to probability distribution
        total_importance = np.sum(importance_values)
        if total_importance < 1e-8:
            # Fallback to uniform sampling
            probabilities = np.ones(num_valid) / num_valid
        else:
            probabilities = importance_values / total_importance

        # Weighted random sampling without replacement
        try:
            selected_indices = np.random.choice(
                num_valid,
                size=target_count,
                replace=False,
                p=probabilities
            )
        except ValueError:
            # If probabilities sum != 1.0 due to numerical issues, normalize again
            probabilities = probabilities / np.sum(probabilities)
            selected_indices = np.random.choice(
                num_valid,
                size=target_count,
                replace=False,
                p=probabilities
            )

        # Create new mask with only selected points
        new_mask = np.zeros((h, w), dtype=bool)
        selected_i = valid_indices[0][selected_indices]
        selected_j = valid_indices[1][selected_indices]
        new_mask[selected_i, selected_j] = True

        logger.info(
            f"[ImportanceSampling] Sampled {target_count} points from {num_valid} valid pixels"
        )

        return new_mask

    def _find_skeleton_tangent(
        self,
        position: np.ndarray,
        features: Dict[str, np.ndarray],
        image_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Find tangent vector from nearest skeleton point.

        Args:
            position: 3D world position [x, y, z]
            features: Features dict with 'medial_axis'
            image_shape: (height, width) of original image

        Returns:
            Tangent vector [dx, dy, 0] or None if skeleton not available
        """
        if 'medial_axis' not in features:
            return None

        skeleton = features['medial_axis']
        h, w = image_shape

        # Map world position to image space
        x_img = int((position[0] + 1) * w / 2)
        y_img = int((1 - position[1]) * h / 2)

        # Bounds check
        if not (0 <= x_img < w and 0 <= y_img < h):
            return None

        # Find nearest skeleton pixel
        skeleton_points = np.argwhere(skeleton > 0)
        if len(skeleton_points) == 0:
            return None

        # Distance to all skeleton points
        dists = np.sqrt(
            (skeleton_points[:, 0] - y_img)**2 +
            (skeleton_points[:, 1] - x_img)**2
        )
        nearest_idx = np.argmin(dists)
        nearest_y, nearest_x = skeleton_points[nearest_idx]

        # Compute tangent at nearest skeleton point using local neighbors
        # Find skeleton points within small radius
        radius = 5  # pixels
        nearby_mask = dists < radius
        nearby_points = skeleton_points[nearby_mask]

        if len(nearby_points) < 2:
            # Fallback to flow field
            return None

        # Fit line to nearby skeleton points (PCA for direction)
        points_centered = nearby_points - nearby_points.mean(axis=0)
        cov = np.cov(points_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Principal direction (largest eigenvalue)
        principal_idx = np.argmax(eigenvalues)
        tangent_img = eigenvectors[:, principal_idx]  # [dy, dx] in image space

        # Convert to world space (flip y, normalize)
        tangent_world = np.array([tangent_img[1], -tangent_img[0], 0.0])
        tangent_world = tangent_world / (np.linalg.norm(tangent_world) + 1e-8)

        return tangent_world

    def _extract_features(
        self, image: np.ndarray, alpha_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract image features for Gaussian initialization

        Args:
            image: Input image
            alpha_mask: Alpha channel

        Returns:
            Dictionary of features (medial_axis, thickness_map, flow_field)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Binary mask for morphology
        binary = (alpha_mask > 0.5).astype(np.uint8)

        # 1. Medial axis (skeleton)
        skeleton = morphology.skeletonize(binary > 0)

        # 2. Distance transform (thickness)
        thickness_map = cv2.distanceTransform(binary * 255, cv2.DIST_L2, 5).astype(
            np.float32
        )

        # Normalize thickness
        if thickness_map.max() > 0:
            thickness_map = thickness_map / thickness_map.max()

        # 3. Gradient flow field
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Normalize gradients to get flow direction
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
        flow_x = grad_x / grad_magnitude
        flow_y = grad_y / grad_magnitude

        features = {
            "skeleton": skeleton.astype(np.float32),
            "thickness": thickness_map,
            "flow_x": flow_x,
            "flow_y": flow_y,
            "grad_magnitude": grad_magnitude,
        }

        return features

    def _initialize_gaussians(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray,
        features: Dict[str, np.ndarray],
    ) -> List[Gaussian2D]:
        """
        Initialize Gaussians from point cloud

        Args:
            points: 3D positions
            colors: RGB colors
            normals: Surface normals
            features: Image features

        Returns:
            List of initialized Gaussians
        """
        gaussians = []

        if len(points) == 0:
            logger.warning("[BrushConverter] No points to create Gaussians")
            return gaussians

        # Build KD-tree for neighbor search
        kdtree = KDTree(points)

        for i, (pos, color, normal) in enumerate(zip(points, colors, normals)):
            # Collect all parameters first before creating Gaussian

            # Position (already provided)
            position = pos

            # Color (already provided)
            color_rgb = color

            # Opacity from alpha at this position
            # Map world position back to image space
            x_img = int((pos[0] + 1) * features["thickness"].shape[1] / 2)
            y_img = int((1 - pos[1]) * features["thickness"].shape[0] / 2)

            if (
                0 <= x_img < features["thickness"].shape[1]
                and 0 <= y_img < features["thickness"].shape[0]
            ):
                thickness = features["thickness"][y_img, x_img]
                opacity = min(1.0, thickness * 2.0)  # Scale opacity by thickness
            else:
                opacity = 0.8

            # Scale based on local density
            distances, _ = kdtree.query([pos], k=min(8, len(points)))
            if len(distances[0]) > 1:
                avg_distance = np.mean(distances[0][1:])  # Exclude self
            else:
                avg_distance = 0.05

            # Thickness-based anisotropic scale
            # Thicker regions get larger Gaussians for better coverage
            thickness_scale = np.clip(thickness * 2.0, 0.5, 2.0) if 0 <= x_img < features["thickness"].shape[1] and 0 <= y_img < features["thickness"].shape[0] else 1.0

            scale = np.array(
                [
                    avg_distance * 2.0 * thickness_scale,  # X scale (increased from 1.5, modulated by thickness)
                    avg_distance * 2.0 * thickness_scale,  # Y scale (increased from 1.5, modulated by thickness)
                    avg_distance * 0.3,  # Z scale (thin in depth)
                ]
            )

            # Rotation: Try skeleton tangent first, fallback to flow field
            h, w = features["thickness"].shape
            skeleton_tangent = self._find_skeleton_tangent(pos, features, (h, w))

            if skeleton_tangent is not None:
                # Use skeleton tangent for more accurate stroke direction
                # Compute angle from tangent vector
                angle = np.arctan2(skeleton_tangent[1], skeleton_tangent[0])

                # Create quaternion for Z-axis rotation
                rotation = quaternion_from_axis_angle(
                    axis=np.array([0, 0, 1]), angle=angle
                )
            elif (
                0 <= x_img < features["flow_x"].shape[1]
                and 0 <= y_img < features["flow_y"].shape[0]
            ):
                # Fallback to flow field
                flow_x = features["flow_x"][y_img, x_img]
                flow_y = features["flow_y"][y_img, x_img]

                # Compute angle from flow
                angle = np.arctan2(flow_y, flow_x)

                # Create quaternion for Z-axis rotation
                rotation = quaternion_from_axis_angle(
                    axis=np.array([0, 0, 1]), angle=angle
                )
            else:
                # Default quaternion (no rotation)
                rotation = np.array([0, 0, 0, 1])

            # Now create Gaussian with all required parameters
            g = Gaussian2D(
                position=position,
                scale=scale,
                rotation=rotation,
                opacity=opacity,
                color=color_rgb,
            )

            gaussians.append(g)

        return gaussians

    def _refine_gaussians(
        self, gaussians: List[Gaussian2D], features: Dict[str, np.ndarray]
    ) -> List[Gaussian2D]:
        """
        Apply procedural refinements to Gaussians

        Args:
            gaussians: Initial Gaussians
            features: Image features

        Returns:
            Refined Gaussians
        """
        for g in gaussians:
            # Map position to image space
            x_img = int((g.position[0] + 1) * features["thickness"].shape[1] / 2)
            y_img = int((1 - g.position[1]) * features["thickness"].shape[0] / 2)

            # 1. Scale by thickness
            if (
                0 <= x_img < features["thickness"].shape[1]
                and 0 <= y_img < features["thickness"].shape[0]
            ):
                thickness = features["thickness"][y_img, x_img]
                thickness_scale = 0.5 + thickness  # Range [0.5, 1.5]
                g.scale = g.scale * thickness_scale

            # 2. Add artistic jitter
            jitter_pos = np.random.normal(0, 0.005, 3)  # Small position noise
            g.position = g.position + jitter_pos

            # Small rotation jitter
            jitter_angle = np.random.normal(0, np.pi / 36)  # ±5 degrees
            jitter_quat = quaternion_from_axis_angle(
                axis=np.array([0, 0, 1]), angle=jitter_angle
            )
            # Combine rotations (simplified - just add small Z rotation)
            g.rotation = jitter_quat

            # 3. Adjust opacity based on gradient magnitude
            if (
                0 <= x_img < features["grad_magnitude"].shape[1]
                and 0 <= y_img < features["grad_magnitude"].shape[0]
            ):
                grad_mag = features["grad_magnitude"][y_img, x_img]
                # Higher gradient = edge = lower opacity
                opacity_factor = 1.0 - (grad_mag / (grad_mag + 50))
                g.opacity = g.opacity * opacity_factor

        return gaussians

    def _compute_ssim(
        self, img1: np.ndarray, img2: np.ndarray, C1: float = 0.01**2, C2: float = 0.03**2
    ) -> float:
        """
        Compute SSIM (Structural Similarity Index) between two images.

        Args:
            img1, img2: Images to compare (H, W) or (H, W, C)
            C1, C2: Stability constants

        Returns:
            SSIM value (0-1, higher is better)
        """
        # Convert to float
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1.astype(np.float32), cv2.COLOR_BGR2GRAY) / 255.0
        else:
            img1 = img1.astype(np.float32) / 255.0

        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2.astype(np.float32), cv2.COLOR_BGR2GRAY) / 255.0
        else:
            img2 = img2.astype(np.float32) / 255.0

        # Compute means
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    def _optimize_appearance(
        self, gaussians: List[Gaussian2D], target_image: np.ndarray, iterations: int
    ) -> List[Gaussian2D]:
        """
        Optimize Gaussian parameters to match target image using differentiable rendering.

        Uses simple gradient descent on opacity and position to match appearance.
        More sophisticated optimization with PyTorch can be added later.

        Args:
            gaussians: Initial Gaussians
            target_image: Target appearance (H, W, 3) or (H, W, 4)
            iterations: Optimization steps

        Returns:
            Optimized Gaussians
        """
        if iterations <= 0:
            return gaussians

        logger.info(
            f"[BrushConverter] Starting appearance optimization: {iterations} iterations"
        )

        # Import renderer (lazy import to avoid circular dependency)
        try:
            from .renderer_gsplat import GaussianRenderer2D_GSplat as Renderer
            logger.info("[Optimization] Using GSplat renderer")
        except ImportError:
            try:
                from .renderer_gpu import GaussianRenderer2D_GPU as Renderer
                logger.info("[Optimization] Using GPU renderer")
            except ImportError:
                from .renderer import GaussianRenderer2D as Renderer
                logger.info("[Optimization] Using CPU renderer")

        # Extract target dimensions and alpha
        if len(target_image.shape) == 4:  # (H, W, 4)
            h, w = target_image.shape[:2]
            target_rgb = target_image[:, :, :3]
            target_alpha = target_image[:, :, 3] / 255.0
        elif len(target_image.shape) == 3:  # (H, W, 3)
            h, w = target_image.shape[:2]
            target_rgb = target_image
            target_alpha = None
        else:
            h, w = target_image.shape
            target_rgb = np.stack([target_image] * 3, axis=-1)
            target_alpha = None

        # Initialize renderer
        renderer = Renderer(width=w, height=h)

        # Store initial parameters for regularization
        initial_opacities = [g.opacity for g in gaussians]

        # Optimization hyperparameters
        learning_rate = 0.01
        l1_weight = 0.7
        ssim_weight = 0.2
        reg_weight = 0.1

        best_loss = float('inf')
        best_gaussians = [g.copy() for g in gaussians]

        # Optimization loop
        for iter_idx in range(iterations):
            # Render current state
            try:
                rendered_img = renderer.render(gaussians)
            except Exception as e:
                logger.warning(f"[Optimization] Render failed at iteration {iter_idx}: {e}")
                break

            # Compute losses
            # L1 loss (pixel-wise difference)
            l1_loss = np.mean(np.abs(rendered_img.astype(np.float32) - target_rgb.astype(np.float32)))

            # SSIM loss
            try:
                ssim_value = self._compute_ssim(rendered_img, target_rgb)
                ssim_loss = 1.0 - ssim_value
            except:
                ssim_loss = 0.0
                ssim_value = 0.0

            # Regularization loss (prevent drastic opacity changes)
            reg_loss = np.mean([
                (g.opacity - init_opacity) ** 2
                for g, init_opacity in zip(gaussians, initial_opacities)
            ])

            # Total loss
            total_loss = l1_weight * l1_loss + ssim_weight * ssim_loss + reg_weight * reg_loss

            # Log progress
            if iter_idx % 20 == 0:
                logger.info(
                    f"[Optimization] Iter {iter_idx}/{iterations}: "
                    f"Loss={total_loss:.4f} (L1={l1_loss:.4f}, SSIM={ssim_value:.3f}, Reg={reg_loss:.4f})"
                )

            # Save best
            if total_loss < best_loss:
                best_loss = total_loss
                best_gaussians = [g.copy() for g in gaussians]

            # Simple opacity optimization using finite differences
            # (Full gradient descent would require differentiable renderer)
            for i, g in enumerate(gaussians):
                # Small perturbation to opacity
                delta = 0.001
                old_opacity = g.opacity

                # Try increasing opacity
                g.opacity = np.clip(old_opacity + delta, 0.01, 1.0)
                rendered_plus = renderer.render(gaussians)
                loss_plus = np.mean(np.abs(rendered_plus.astype(np.float32) - target_rgb.astype(np.float32)))

                # Try decreasing opacity
                g.opacity = np.clip(old_opacity - delta, 0.01, 1.0)
                rendered_minus = renderer.render(gaussians)
                loss_minus = np.mean(np.abs(rendered_minus.astype(np.float32) - target_rgb.astype(np.float32)))

                # Compute gradient
                gradient = (loss_plus - loss_minus) / (2 * delta)

                # Update opacity
                new_opacity = old_opacity - learning_rate * gradient
                g.opacity = np.clip(new_opacity, 0.01, 1.0)

            # Early stopping
            if total_loss < 0.05:  # Good enough
                logger.info(f"[Optimization] Early stopping at iteration {iter_idx}, loss={total_loss:.4f}")
                break

        logger.info(f"[Optimization] Completed. Best loss: {best_loss:.4f}")

        return best_gaussians

    def _create_brush_stamp(
        self, gaussians: List[Gaussian2D], brush_name: str
    ) -> BrushStamp:
        """
        Create BrushStamp from Gaussians

        Args:
            gaussians: List of Gaussians
            brush_name: Name for the brush

        Returns:
            BrushStamp object
        """
        if len(gaussians) == 0:
            logger.warning("[BrushConverter] Creating empty brush stamp")
            center = np.array([0, 0, 0])
            size = 0.1
        else:
            # Compute brush center (centroid)
            positions = np.array([g.position for g in gaussians])
            center = np.mean(positions, axis=0)

            # Compute brush size (bounding box diagonal)
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            size = np.linalg.norm(max_pos - min_pos)

        # Create orientation frame (identity for converted brushes)
        tangent = np.array([1, 0, 0])
        normal = np.array([0, 1, 0])
        binormal = np.array([0, 0, 1])

        # Create brush stamp (no parameters in constructor)
        brush = BrushStamp()

        # Set brush pattern (base gaussians)
        brush.base_gaussians = gaussians
        brush.center = center
        brush.size = size  # Bounding box diagonal
        brush.tangent = tangent
        brush.normal = normal
        brush.binormal = binormal
        brush.spacing = size * 0.5  # Default spacing

        # Apply default parameters to create working gaussians
        brush.apply_parameters()

        # Set metadata as custom attribute
        brush.metadata = {
            "name": brush_name,
            "gaussian_count": len(gaussians),
            "type": "converted",
            "source": "image",
        }

        return brush


def test_brush_converter():
    """Test brush conversion with synthetic image"""
    print("Testing brush converter...")

    # Create test image
    test_image = np.zeros((256, 256, 4), dtype=np.uint8)

    # Draw a gradient brush stroke
    cv2.ellipse(test_image, (128, 128), (80, 40), 0, 0, 360, (200, 150, 100, 255), -1)

    # Add some texture
    noise = np.random.normal(0, 10, test_image.shape[:2])
    test_image[:, :, :3] = np.clip(
        test_image[:, :, :3] + noise[:, :, np.newaxis], 0, 255
    )

    # Create converter
    converter = BrushConverter(use_midas=False, target_gaussian_count=500)

    # Test conversion
    brush = converter.convert_2d_to_3dgs(
        test_image, brush_name="test_brush", depth_profile="convex", depth_scale=0.2
    )

    print(f"✓ Converted brush: {len(brush.gaussians)} Gaussians")
    print(f"  Center: {brush.center}")
    print(f"  Size: {brush.size:.3f}")

    # Save test image
    cv2.imwrite("test_brush_input.png", test_image)
    print("Test complete! Input saved as test_brush_input.png")


if __name__ == "__main__":
    test_brush_converter()
