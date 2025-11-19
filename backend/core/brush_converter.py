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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from omegaconf import OmegaConf, DictConfig
import os
from datetime import datetime
import contextlib
import traceback

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
        config: Optional[DictConfig] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize brush converter

        Args:
            device: Device for computation ('cuda', 'cpu', or None for auto)
            use_midas: Whether to use MiDaS for depth (fallback to heuristics)
            target_gaussian_count: Target number of Gaussians per brush
            config: Optional Hydra config (if None, will load default)
            debug_mode: Enable debug visualization
        """
        # Load configuration
        if config is None:
            config_path = Path(__file__).parent.parent / "config" / "brush_converter_config.yaml"
            if config_path.exists():
                self.config = OmegaConf.load(config_path)
                logger.info(f"[BrushConverter] Loaded config from {config_path}")
            else:
                # Fallback to default config
                self.config = OmegaConf.create({
                    "debug": {"enabled": False, "visualize": False, "output_dir": "debug_output"},
                    "gaussian": {"target_count": 800, "scale_multiplier": 0.6, "z_scale": 0.3},
                    "contrast": {"enabled": True, "min_contrast": 0.3},
                })
                logger.warning(f"[BrushConverter] Config file not found, using defaults")
        else:
            self.config = config

        # Override debug mode if specified
        if debug_mode:
            self.config.debug.enabled = True

        self.device = device or (
            "cuda" if np.random.rand() > 1.5 else "cpu"
        )  # For now use CPU

        # Use config value or parameter
        self.target_gaussian_count = self.config.gaussian.target_count if hasattr(self.config, 'gaussian') else target_gaussian_count

        # Initialize depth estimator
        use_midas_config = self.config.depth.use_midas if hasattr(self.config, 'depth') else use_midas
        self.depth_estimator = create_depth_estimator(
            prefer_midas=use_midas_config, device=self.device
        )

        # Debug mode setup
        self.debug_enabled = self.config.debug.enabled
        if self.debug_enabled:
            self.debug_data = {}  # Store intermediate results for visualization
            output_dir = Path(self.config.debug.output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"[BrushConverter] Debug mode enabled, output: {output_dir}")

        logger.info(
            f"[BrushConverter] Initialized with target {self.target_gaussian_count} Gaussians"
        )

    def convert_2d_to_3dgs(
        self,
        image: np.ndarray,
        brush_name: str = "converted_brush",
        depth_profile: str = "convex",
        depth_scale: float = 0.2,
        optimization_steps: int = 0,  # Disabled for initial implementation
        progress_callback=None,  # Optional callback for progress updates
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

        # Initialize debug data storage
        if self.debug_enabled:
            self.debug_data = {
                'original_image': image.copy(),
                'brush_name': brush_name,
            }

        # Step 1: Depth estimation
        depth_map = self._estimate_depth(image, depth_profile, depth_scale)
        if self.debug_enabled:
            self.debug_data['depth_map'] = depth_map.copy()

        # Step 2: Extract alpha mask
        alpha_mask = self._extract_alpha_mask(image)
        if self.debug_enabled:
            self.debug_data['alpha_mask'] = alpha_mask.copy()

        # Step 3: Feature extraction (before point cloud for importance sampling)
        features = self._extract_features(image, alpha_mask)
        if self.debug_enabled:
            self.debug_data['features'] = {k: v.copy() for k, v in features.items()}

        # Step 4: Generate point cloud with importance-based sampling
        points, colors, normals = self._generate_point_cloud(
            image, depth_map, alpha_mask, features=features
        )
        if self.debug_enabled:
            self.debug_data['points'] = points.copy()
            self.debug_data['colors'] = colors.copy()
            self.debug_data['normals'] = normals.copy()

        # Step 5: Initialize Gaussians
        gaussians = self._initialize_gaussians(points, colors, normals, features)
        if self.debug_enabled:
            self.debug_data['gaussians_initial'] = [g.copy() for g in gaussians]

        # Step 6: Procedural refinement
        gaussians = self._refine_gaussians(gaussians, features)
        if self.debug_enabled:
            self.debug_data['gaussians_refined'] = [g.copy() for g in gaussians]

        # Step 7: Appearance optimization (if requested)
        if optimization_steps > 0:
            gaussians = self._optimize_appearance(gaussians, image, optimization_steps, progress_callback)

        # Step 8: Create BrushStamp
        brush = self._create_brush_stamp(gaussians, brush_name)

        logger.info(f"[BrushConverter] ✓ Created brush with {len(gaussians)} Gaussians")

        # Generate debug visualization
        if self.debug_enabled and self.config.debug.visualize:
            self._visualize_pipeline(brush_name)

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
        Extract alpha channel or create from intensity with adaptive background detection

        Args:
            image: Input image

        Returns:
            Alpha mask (H, W) in range [0, 1]
        """
        # Flag to determine if we need RGB-based extraction
        use_rgb_extraction = True

        if len(image.shape) == 3 and image.shape[2] == 4:
            # Check if alpha channel is meaningful (has variation)
            alpha_channel = image[:, :, 3].astype(np.float32) / 255.0
            alpha_variance = np.var(alpha_channel)

            if alpha_variance > 0.01:
                # Meaningful alpha with variation - use it but make it binary
                alpha = alpha_channel
                # Make alpha strictly binary (0 or 1) to avoid semi-transparent backgrounds
                alpha = (alpha > 0.5).astype(np.float32)
                use_rgb_extraction = False
                logger.info(f"[AlphaMask] Using existing alpha channel (variance={alpha_variance:.4f}), binarized at 0.5")
            else:
                # Alpha is uniform (e.g., all 255) - ignore and extract from RGB
                logger.info(f"[AlphaMask] Ignoring uniform alpha channel (variance={alpha_variance:.4f}), extracting from RGB")

        if use_rgb_extraction:
            # Create from grayscale intensity with adaptive thresholding
            # Handle both RGB and RGBA images
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Detect background color from corner pixels
            h, w = gray.shape
            margin = max(min(h, w) // 10, 5)  # Sample 10% margin from corners, minimum 5 pixels
            corners = [
                gray[0:margin, 0:margin],           # Top-left
                gray[0:margin, w-margin:w],         # Top-right
                gray[h-margin:h, 0:margin],         # Bottom-left
                gray[h-margin:h, w-margin:w],       # Bottom-right
            ]
            bg_luminance = np.mean([np.mean(c) for c in corners])

            logger.info(f"[AlphaMask] Detected background luminance: {bg_luminance:.1f}")

            # Adaptive threshold based on background using Otsu's method
            if bg_luminance > 128:
                # Bright background → dark strokes → keep dark pixels
                threshold_value, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                logger.info(f"[AlphaMask] Using THRESH_BINARY_INV + OTSU (bright background), threshold={threshold_value:.1f}")
            else:
                # Dark background → bright strokes → keep bright pixels
                threshold_value, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                logger.info(f"[AlphaMask] Using THRESH_BINARY + OTSU (dark background), threshold={threshold_value:.1f}")

            # Apply morphological operations to clean the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # Remove small noise pixels in background
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
            # Fill small holes in strokes
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

            # Make it strictly binary after morphological operations
            alpha = (alpha > 127).astype(np.uint8) * 255

            # Store alpha for debugging
            if self.debug_enabled:
                self.debug_data['alpha_before_binary'] = alpha.copy()
                logger.info(f"[AlphaMask] Alpha stats before final binarization - Min: {np.min(alpha):.3f}, Max: {np.max(alpha):.3f}, Mean: {np.mean(alpha):.3f}")

            # Final conversion to float and strict binarization
            alpha = alpha.astype(np.float32) / 255.0
            alpha = (alpha > 0.5).astype(np.float32)  # Ensure strictly 0 or 1

        # Apply morphological cleaning for both alpha channel and RGB extraction
        if use_rgb_extraction or True:  # Always apply cleaning
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # Convert to uint8 for morphological operations
            alpha_uint8 = (alpha * 255).astype(np.uint8)
            # Clean up the mask
            alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_OPEN, kernel_small)
            alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel_small)
            # Convert back to float and ensure binary
            alpha = (alpha_uint8 > 127).astype(np.float32)

        if self.debug_enabled:
            logger.info(f"[AlphaMask] Final alpha stats - Min: {np.min(alpha):.3f}, Max: {np.max(alpha):.3f}, Mean: {np.mean(alpha):.3f}")
            logger.info(f"[AlphaMask] Unique values: {np.unique(alpha)}")

        return alpha

    def _generate_point_cloud(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        alpha_mask: np.ndarray,
        alpha_threshold: float = 0.8,  # Increased to 0.8 to strictly filter background pixels
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
        # Use 'xy' indexing to ensure correct X/Y mapping: xx[i,j] corresponds to X at column j
        xx, yy = np.meshgrid(
            np.linspace(-x_range, x_range, w),
            np.linspace(-y_range, y_range, h),
            indexing='xy'  # Cartesian indexing: xx varies along columns, yy along rows
        )

        # Create point cloud
        points = []
        colors = []
        normals = []

        # Sample points based on alpha mask
        mask = alpha_mask > alpha_threshold

        # Importance-based adaptive sampling
        total_pixels = np.sum(mask)
        importance_map = None
        if total_pixels > self.target_gaussian_count:
            # Compute importance map for quality-aware sampling
            importance_map = self._compute_importance_map(
                image, depth_map, alpha_mask, features, mask
            )

            # Store for debug visualization
            if self.debug_enabled:
                self.debug_data['importance_map'] = importance_map.copy()
                self.debug_data['mask_before_sampling'] = mask.copy()

            # Weighted sampling based on importance
            mask = self._importance_based_sampling(
                mask, importance_map, self.target_gaussian_count
            )

            if self.debug_enabled:
                self.debug_data['mask_after_sampling'] = mask.copy()

        # Compute luminance map with contrast enhancement
        if len(image.shape) == 3:
            color_bgr = image[:, :, :3] / 255.0
            luminance_map = 0.114 * color_bgr[:, :, 0] + 0.587 * color_bgr[:, :, 1] + 0.299 * color_bgr[:, :, 2]
        else:
            luminance_map = image / 255.0

        # Apply contrast stretching to enhance texture detail
        # Only stretch within valid mask region to avoid background affecting range
        valid_luminance = luminance_map[mask]
        if len(valid_luminance) > 0 and self.config.contrast.enabled:
            percentile_low = self.config.contrast.percentile_low
            percentile_high = self.config.contrast.percentile_high
            min_contrast = self.config.contrast.min_contrast

            lum_min = np.percentile(valid_luminance, percentile_low)  # Use percentiles to avoid outliers
            lum_max = np.percentile(valid_luminance, percentile_high)

            # Ensure minimum contrast range (use config value)
            if lum_max - lum_min < min_contrast:
                # Expand range to at least min_contrast
                center = (lum_min + lum_max) / 2
                lum_min = max(0.0, center - min_contrast / 2)
                lum_max = min(1.0, center + min_contrast / 2)

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
        # DISABLED: Gradient is luminance-dependent and causes uneven sampling
        # Gaussians should be placed based on structure (skeleton + thickness) only
        # if len(image.shape) == 3:
        #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray = image
        #
        # grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        # grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        # gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (5, 5), 1.0)
        # gradient_normalized = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)

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
        # Gradient: REMOVED - luminance-dependent, causes uneven distribution
        # Thickness: 50% - coverage of thick areas
        # Skeleton: 50% - structural integrity and even distribution
        importance = (
            0.5 * thickness_normalized +
            0.5 * skeleton_importance
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
            # Use config values for scale parameters
            scale_mult = self.config.gaussian.scale_multiplier
            z_scale = self.config.gaussian.z_scale

            if self.config.gaussian.thickness.enabled:
                thickness_mult = self.config.gaussian.thickness.multiplier
                min_t_scale = self.config.gaussian.thickness.min_scale
                max_t_scale = self.config.gaussian.thickness.max_scale
                thickness_scale = np.clip(thickness * thickness_mult, min_t_scale, max_t_scale) if 0 <= x_img < features["thickness"].shape[1] and 0 <= y_img < features["thickness"].shape[0] else 1.0
            else:
                thickness_scale = 1.0

            # Rotation: Try skeleton tangent first, fallback to flow field
            # (rotation is calculated BEFORE scale to determine elongation direction)
            h, w = features["thickness"].shape
            skeleton_tangent = self._find_skeleton_tangent(pos, features, (h, w))
            has_direction = False  # Track if we have directional information

            if skeleton_tangent is not None:
                # Use skeleton tangent for more accurate stroke direction
                # Compute angle from tangent vector
                angle = np.arctan2(skeleton_tangent[1], skeleton_tangent[0])

                # Create quaternion for Z-axis rotation
                rotation = quaternion_from_axis_angle(
                    axis=np.array([0, 0, 1]), angle=angle
                )
                has_direction = True
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
                has_direction = True
            else:
                # Default quaternion (no rotation)
                rotation = np.array([0, 0, 0, 1])
                has_direction = False

            # Scale calculation with optional directional elongation
            # Apply elongation when we have directional information (skeleton or flow)
            if (
                has_direction
                and hasattr(self.config.gaussian, 'elongation')
                and self.config.gaussian.elongation.enabled
            ):
                # Get elongation parameters from config
                elongation_ratio = self.config.gaussian.elongation.ratio
                elongation_strength = self.config.gaussian.elongation.strength

                # Base scale (short axis)
                base_scale = avg_distance * scale_mult * thickness_scale

                # Long axis (along stroke direction)
                long_axis = base_scale * elongation_ratio

                # Blend based on strength (0.0 = circular, 1.0 = full elongation)
                long_axis = base_scale + (long_axis - base_scale) * elongation_strength

                # Create anisotropic scale: [long_axis, short_axis, depth]
                # X = along rotation direction, Y = perpendicular
                scale = np.array([long_axis, base_scale, avg_distance * z_scale])
            else:
                # No direction or elongation disabled: use isotropic (circular) scale
                base_scale = avg_distance * scale_mult * thickness_scale
                scale = np.array([base_scale, base_scale, avg_distance * z_scale])

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

            # Small rotation jitter (reduced to preserve directional elongation)
            jitter_angle = np.random.normal(0, np.pi / 90)  # ±2 degrees (reduced from ±5)
            jitter_quat = quaternion_from_axis_angle(
                axis=np.array([0, 0, 1]), angle=jitter_angle
            )
            # Combine rotations: multiply quaternions to preserve original direction
            from backend.core.quaternion_utils import quaternion_multiply
            g.rotation = quaternion_multiply(g.rotation, jitter_quat)

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

    def _align_to_target(
        self, gaussians: List[Gaussian2D], target_image: np.ndarray, renderer
    ) -> List[Gaussian2D]:
        """
        Align Gaussians to target image using center-of-mass and bounding box matching.
        This provides initial spatial alignment before fine-tuning with optimization.

        Args:
            gaussians: List of Gaussians to align
            target_image: Target image (H, W, 3) or (H, W, 4) in range [0, 255]
            renderer: Renderer instance for coordinate transformations

        Returns:
            Aligned Gaussians (modified in-place, also returned for convenience)
        """
        if len(gaussians) == 0:
            return gaussians

        # Extract RGB and alpha from target
        if len(target_image.shape) == 3 and target_image.shape[2] == 4:
            target_rgb = target_image[:, :, :3].astype(np.float32) / 255.0
            target_alpha = target_image[:, :, 3].astype(np.float32) / 255.0
        elif len(target_image.shape) == 3:
            target_rgb = target_image.astype(np.float32) / 255.0
            target_alpha = np.ones(target_image.shape[:2], dtype=np.float32)
        else:
            target_rgb = np.stack([target_image] * 3, axis=-1).astype(np.float32) / 255.0
            target_alpha = np.ones(target_image.shape, dtype=np.float32)

        h, w = target_rgb.shape[:2]

        # 1. Compute target center of mass (weighted by luminance * alpha)
        luminance = 0.299 * target_rgb[:, :, 0] + 0.587 * target_rgb[:, :, 1] + 0.114 * target_rgb[:, :, 2]
        weight = luminance * target_alpha
        weight = weight / (np.sum(weight) + 1e-8)

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        target_com_x = np.sum(x_coords * weight)
        target_com_y = np.sum(y_coords * weight)

        # Convert pixel coordinates to world coordinates using renderer
        target_com_world = renderer.pixel_to_world(np.array([target_com_x, target_com_y]))
        target_com_x_norm = target_com_world[0]
        target_com_y_norm = target_com_world[1]

        # 2. Compute Gaussian center of mass
        positions = np.array([g.position[:2] for g in gaussians])
        gaussian_com = np.mean(positions, axis=0)

        # 3. Compute translation offset
        offset = np.array([target_com_x_norm, target_com_y_norm]) - gaussian_com

        logger.info(
            f"[Alignment] Target CoM: ({target_com_x_norm:.3f}, {target_com_y_norm:.3f}), "
            f"Gaussian CoM: ({gaussian_com[0]:.3f}, {gaussian_com[1]:.3f}), "
            f"Offset: ({offset[0]:.3f}, {offset[1]:.3f})"
        )

        # Apply translation
        for g in gaussians:
            g.position[0] += offset[0]
            g.position[1] += offset[1]

        # 4. Compute bounding boxes for scale matching
        # Target bbox (in world coords using renderer)
        mask = (weight > np.percentile(weight[weight > 0], 10)).astype(np.float32)
        if np.sum(mask) > 0:
            y_indices, x_indices = np.where(mask > 0)

            # Convert pixel bbox to world coords
            target_min_pixel = np.array([np.min(x_indices), np.min(y_indices)])
            target_max_pixel = np.array([np.max(x_indices), np.max(y_indices)])

            target_min_world = renderer.pixel_to_world(target_min_pixel)
            target_max_world = renderer.pixel_to_world(target_max_pixel)

            target_width = target_max_world[0] - target_min_world[0]
            target_height = abs(target_max_world[1] - target_min_world[1])  # abs() because Y might be flipped

            # Gaussian bbox
            positions = np.array([g.position[:2] for g in gaussians])
            gaussian_min = np.min(positions, axis=0)
            gaussian_max = np.max(positions, axis=0)
            gaussian_width = gaussian_max[0] - gaussian_min[0]
            gaussian_height = gaussian_max[1] - gaussian_min[1]

            # Compute scale factor (use smaller dimension to avoid over-scaling)
            if gaussian_width > 0 and gaussian_height > 0:
                scale_x = target_width / gaussian_width
                scale_y = target_height / gaussian_height
                scale_factor = min(scale_x, scale_y)  # Use actual scale, no safety factor needed

                logger.info(
                    f"[Alignment] Target size: ({target_width:.3f}, {target_height:.3f}), "
                    f"Gaussian size: ({gaussian_width:.3f}, {gaussian_height:.3f}), "
                    f"Scale factor: {scale_factor:.3f}"
                )

                # Apply scale (relative to new center of mass)
                new_com = np.array([target_com_x_norm, target_com_y_norm])
                for g in gaussians:
                    # Translate to origin, scale, translate back
                    relative_pos = g.position[:2] - new_com
                    g.position[0] = new_com[0] + relative_pos[0] * scale_factor
                    g.position[1] = new_com[1] + relative_pos[1] * scale_factor

                    # Scale the Gaussian size as well
                    g.scale[0] *= scale_factor
                    g.scale[1] *= scale_factor
            else:
                logger.warning("[Alignment] Gaussian bbox has zero size, skipping scale adjustment")
        else:
            logger.warning("[Alignment] Target mask is empty, skipping scale adjustment")

        logger.info("[Alignment] Completed initial alignment")
        return gaussians

    def _save_optimization_debug(
        self,
        target_img: np.ndarray,
        rendered_img: np.ndarray,
        iteration: int,
        total_iterations: int,
        loss: float,
        learning_rate: float,
        gaussian_count: int,
        session_dir: Path
    ):
        """
        Save debug visualization of optimization progress with subplots.

        Args:
            target_img: Target image (H, W, 3), range [0, 1]
            rendered_img: Rendered image (H, W, 3), range [0, 1]
            iteration: Current iteration number
            total_iterations: Total iterations
            loss: Current loss value
            learning_rate: Current learning rate
            gaussian_count: Number of Gaussians
            session_dir: Directory to save debug images
        """
        try:
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Optimization Progress - Iteration {iteration}/{total_iterations}', fontsize=14, fontweight='bold')

            # Convert images to uint8 for display
            target_uint8 = (np.clip(target_img, 0, 1) * 255).astype(np.uint8)
            rendered_uint8 = (np.clip(rendered_img, 0, 1) * 255).astype(np.uint8)

            # Compute absolute difference
            diff = np.abs(target_img - rendered_img)
            diff_uint8 = (diff * 255).astype(np.uint8)

            # 1. Target image
            axes[0].imshow(target_uint8)
            axes[0].set_title('Target Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # 2. Rendered image
            axes[1].imshow(rendered_uint8)
            axes[1].set_title('Rendered Image', fontsize=12, fontweight='bold')
            axes[1].axis('off')

            # 3. Difference heatmap
            diff_gray = np.mean(diff_uint8, axis=2)
            im = axes[2].imshow(diff_gray, cmap='hot', vmin=0, vmax=255)
            axes[2].set_title('Error Heatmap', fontsize=12, fontweight='bold')
            axes[2].axis('off')

            # Add colorbar to heatmap
            cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label('Error Magnitude', rotation=270, labelpad=15)

            # Add info text below subplots
            info_text = (
                f'Loss: {loss:.6f}  |  '
                f'Learning Rate: {learning_rate:.4f}  |  '
                f'Gaussians: {gaussian_count}'
            )
            fig.text(0.5, 0.02, info_text, ha='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Save figure
            output_path = session_dir / f'iter_{iteration:04d}.png'
            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            logger.debug(f"[Debug] Saved optimization visualization: {output_path}")

        except Exception as e:
            logger.warning(f"[Debug] Failed to save optimization debug image: {e}")

    def _optimize_appearance(
        self, gaussians: List[Gaussian2D], target_image: np.ndarray, iterations: int, progress_callback=None
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

        # Setup debug output directory if enabled
        debug_session_dir = None
        if self.config.optimization.debug.enabled:
            debug_base_dir = Path(self.config.optimization.debug.output_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_session_dir = debug_base_dir / f"optimization_{timestamp}"
            debug_session_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Debug] Saving optimization visualizations to: {debug_session_dir}")

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
        if len(target_image.shape) == 3 and target_image.shape[2] == 4:  # RGBA (H, W, 4)
            h, w = target_image.shape[:2]

            # Extract alpha channel and normalize to [0, 1]
            target_alpha = target_image[:, :, 3]
            if target_alpha.max() > 1.0:
                target_alpha = target_alpha / 255.0

            # Extract raw RGB (NO compositing - alpha will be learned separately)
            target_rgb = target_image[:, :, :3]
            if target_rgb.max() > 1.0:
                target_rgb = target_rgb / 255.0

            logger.info(f"[Optimization] Using RGBA target (alpha coverage: {target_alpha.mean():.1%})")
        elif len(target_image.shape) == 3:  # RGB (H, W, 3)
            h, w = target_image.shape[:2]
            target_rgb = target_image
            target_alpha = None
        else:
            h, w = target_image.shape
            target_rgb = np.stack([target_image] * 3, axis=-1)
            target_alpha = None

        # Store original dimensions for pruning later
        original_h, original_w = h, w
        original_x_range, original_y_range = None, None

        # Apply padding if enabled (for better boundary optimization)
        if hasattr(self.config.optimization, 'padding') and self.config.optimization.padding.enabled:
            padding_percent = self.config.optimization.padding.percentage
            min_pad = self.config.optimization.padding.min_pixels
            max_pad = self.config.optimization.padding.max_pixels
            bg_fill = self.config.optimization.padding.background_fill

            # Compute padding in pixels
            pad_h = int(h * padding_percent)
            pad_w = int(w * padding_percent)

            # Clamp to min/max
            pad_h = max(min_pad, min(max_pad, pad_h))
            pad_w = max(min_pad, min(max_pad, pad_w))

            # Determine fill value
            if bg_fill == "white":
                fill_value = 1.0 if target_rgb.max() <= 1.0 else 255.0
            elif bg_fill == "black":
                fill_value = 0.0
            else:  # "extend"
                fill_value = None

            # Pad target RGB
            if fill_value is not None:
                target_rgb = np.pad(
                    target_rgb,
                    ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    mode='constant',
                    constant_values=fill_value
                )
            else:
                target_rgb = np.pad(
                    target_rgb,
                    ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    mode='edge'
                )

            # Pad alpha if exists (with zeros = transparent background)
            if target_alpha is not None:
                target_alpha = np.pad(
                    target_alpha,
                    ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant',
                    constant_values=0.0
                )

            # Update dimensions
            h, w = target_rgb.shape[:2]

            logger.info(
                f"[Optimization] Applied {padding_percent*100:.0f}% padding: "
                f"{original_h}x{original_w} → {h}x{w} "
                f"(+{pad_h}px vertical, +{pad_w}px horizontal)"
            )
        else:
            logger.info("[Optimization] Padding disabled")

        # Initialize renderer
        renderer = Renderer(width=w, height=h)

        # Compute world bounds from current (possibly padded) dimensions
        aspect = w / h
        target_world_size = 0.5  # Same as in point cloud generation

        if w > h:
            x_range = target_world_size
            y_range = target_world_size / aspect
        else:
            x_range = target_world_size * aspect
            y_range = target_world_size

        # Set world bounds to match current (possibly padded) target image
        world_min = np.array([-x_range, -y_range])
        world_max = np.array([x_range, y_range])

        # Store original (unpadded) world bounds for pruning later
        if hasattr(self.config.optimization, 'padding') and self.config.optimization.padding.enabled:
            original_aspect = original_w / original_h
            if original_w > original_h:
                original_x_range = target_world_size
                original_y_range = target_world_size / original_aspect
            else:
                original_x_range = target_world_size * original_aspect
                original_y_range = target_world_size
            logger.info(
                f"[Optimization] Original bounds: "
                f"[{-original_x_range:.3f}, {original_x_range:.3f}] x "
                f"[{-original_y_range:.3f}, {original_y_range:.3f}]"
            )

        # Filter Gaussians to only those within target bounds
        filtered_gaussians = []
        for g in gaussians:
            pos = g.position[:2]
            if -x_range <= pos[0] <= x_range and -y_range <= pos[1] <= y_range:
                filtered_gaussians.append(g)

        if len(filtered_gaussians) < len(gaussians):
            logger.info(f"[Optimization] Filtered {len(gaussians)} → {len(filtered_gaussians)} Gaussians to target bounds")
            gaussians = filtered_gaussians

        # Set renderer bounds to match target image
        renderer.set_world_bounds(world_min, world_max)

        logger.info(
            f"[Optimization] Set renderer world bounds: "
            f"[{world_min[0]:.3f}, {world_max[0]:.3f}] x [{world_min[1]:.3f}, {world_max[1]:.3f}]"
        )

        # Validate aspect ratio match between target and world space
        target_aspect = w / h
        world_width = renderer.world_max[0] - renderer.world_min[0]
        world_height = renderer.world_max[1] - renderer.world_min[1]
        world_aspect = world_width / world_height

        if abs(target_aspect - world_aspect) > 0.1:
            logger.warning(
                f"[Optimization] Aspect ratio mismatch detected! "
                f"Target: {target_aspect:.2f} ({w}x{h}), "
                f"World: {world_aspect:.2f}. "
                f"Using uniform scaling with letterboxing to preserve aspect ratio."
            )
        else:
            logger.info(
                f"[Optimization] Aspect ratios match. "
                f"Target: {target_aspect:.2f}, World: {world_aspect:.2f}"
            )

        # Store initial parameters for regularization
        initial_opacities = [g.opacity for g in gaussians]

        # Optimization hyperparameters
        learning_rate = 0.01
        l1_weight = 0.7
        ssim_weight = 0.2
        reg_weight = 0.1

        best_loss = float('inf')
        best_gaussians = [g.copy() for g in gaussians]

        # Perform initial alignment to match target position and scale
        logger.info("[Optimization] Performing initial alignment...")
        gaussians = self._align_to_target(gaussians, target_image, renderer)
        logger.info("[Optimization] Alignment complete, starting fine-tuning...")

        # Try PyTorch autograd optimizer first (GPU-accelerated, multi-parameter)
        try:
            from .optimizer_torch import TorchGaussianOptimizer

            logger.info("[Optimization] Attempting PyTorch autograd optimizer (GPU-accelerated)")

            # Extract alpha mask for loss computation (use padded alpha if available)
            if target_alpha is not None:
                # Use padded alpha channel (already processed above)
                target_alpha_mask = (target_alpha > 0.5).astype(np.float32)
            else:
                # Create mask from non-background pixels (works for both white and black backgrounds)
                target_gray = cv2.cvtColor(target_rgb.astype(np.uint8) if target_rgb.max() <= 1.0 else (target_rgb / 255.0 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                target_alpha_mask = cv2.threshold(target_gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1].astype(np.float32)

            # Create PyTorch optimizer with alpha mask and alpha channel
            torch_optimizer = TorchGaussianOptimizer(
                gaussians=gaussians,
                target_image=target_rgb if target_rgb.max() <= 1.0 else target_rgb / 255.0,
                renderer=renderer,
                alpha_mask=target_alpha_mask,
                target_alpha=target_alpha  # Pass alpha channel for alpha loss
            )

            # Progress callback wrapper for PyTorch optimizer
            def torch_progress_callback(iter_idx, total_iters, loss, rendered_np):
                if progress_callback:
                    # Calculate progress percentage
                    optimization_progress = 40 + (40 * iter_idx / total_iters)

                    # Encode rendered image to base64 for preview
                    import base64
                    import io
                    from PIL import Image

                    try:
                        pil_img = Image.fromarray((rendered_np * 255).astype(np.uint8))
                        buffer = io.BytesIO()
                        pil_img.save(buffer, format='PNG')
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        rendered_image = f"data:image/png;base64,{img_base64}"
                    except Exception as e:
                        logger.warning(f"[Optimization] Failed to encode image: {e}")
                        rendered_image = None

                    progress_callback({
                        'progress': optimization_progress,
                        'iteration': iter_idx,
                        'total_iterations': total_iters,
                        'loss': float(loss),
                        'rendered_image': rendered_image,
                        'status': f'Optimizing (PyTorch)... (iteration {iter_idx}/{total_iters})'
                    })

                # Save debug visualization
                if debug_session_dir is not None:
                    if (iter_idx + 1) % self.config.optimization.debug.save_interval == 0:
                        self._save_optimization_debug(
                            target_img=target_rgb if target_rgb.max() <= 1.0 else target_rgb / 255.0,
                            rendered_img=rendered_np,
                            iteration=iter_idx + 1,
                            total_iterations=total_iters,
                            loss=loss,
                            learning_rate=0.01,  # Placeholder
                            gaussian_count=len(gaussians),
                            session_dir=debug_session_dir
                        )

            # Run PyTorch optimization
            optimized_gaussians = torch_optimizer.optimize(
                iterations=iterations,
                progress_callback=torch_progress_callback
            )

            logger.info("[Optimization] PyTorch optimization completed successfully!")

            # Cull low-opacity Gaussians to remove edge artifacts
            opacity_threshold = 0.05
            culled_gaussians = [g for g in optimized_gaussians if g.opacity > opacity_threshold]

            if len(culled_gaussians) < len(optimized_gaussians):
                logger.info(f"[Optimization] Culled {len(optimized_gaussians) - len(culled_gaussians)} low-opacity Gaussians (< {opacity_threshold})")

            # Prune out-of-bounds Gaussians if padding was used
            final_gaussians = culled_gaussians
            if (hasattr(self.config.optimization, 'padding') and
                self.config.optimization.padding.enabled and
                original_x_range is not None):

                pruned_gaussians = []
                for g in culled_gaussians:
                    pos = g.position[:2]
                    # Keep only Gaussians within original (unpadded) bounds
                    if (-original_x_range <= pos[0] <= original_x_range and
                        -original_y_range <= pos[1] <= original_y_range):
                        pruned_gaussians.append(g)

                num_pruned = len(culled_gaussians) - len(pruned_gaussians)
                if num_pruned > 0:
                    logger.info(
                        f"[Optimization] Pruned {num_pruned} out-of-bounds Gaussians "
                        f"({num_pruned/len(culled_gaussians)*100:.1f}% of total)"
                    )

                final_gaussians = pruned_gaussians

            logger.info(f"[Optimization] Returning {len(final_gaussians)} optimized Gaussians")

            return final_gaussians

        except ImportError as e:
            logger.warning(f"[Optimization] PyTorch optimizer not available ({e}), falling back to finite differences")
        except Exception as e:
            logger.warning(f"[Optimization] PyTorch optimization failed: {str(e)}")
            logger.warning(f"[Optimization] Full traceback:\n{traceback.format_exc()}")
            logger.warning("[Optimization] Falling back to finite differences")

        # Fallback: Finite difference optimization (opacity only)
        logger.info("[Optimization] Using finite difference optimizer (CPU, opacity only)")

        # Optimization loop
        for iter_idx in range(iterations):
            # Render current state (suppress renderer stdout)
            try:
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
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

            # Save debug visualization if enabled
            if debug_session_dir is not None:
                if (iter_idx + 1) % self.config.optimization.debug.save_interval == 0:
                    # Normalize target_rgb to [0, 1] if needed
                    target_normalized = target_rgb.astype(np.float32)
                    if target_normalized.max() > 1.0:
                        target_normalized = target_normalized / 255.0

                    self._save_optimization_debug(
                        target_img=target_normalized,
                        rendered_img=rendered_img,
                        iteration=iter_idx + 1,
                        total_iterations=iterations,
                        loss=total_loss,
                        learning_rate=learning_rate,
                        gaussian_count=len(gaussians),
                        session_dir=debug_session_dir
                    )

            # Log progress
            if iter_idx % 20 == 0:
                logger.info(
                    f"[Optimization] Iter {iter_idx}/{iterations}: "
                    f"Loss={total_loss:.4f} (L1={l1_loss:.4f}, SSIM={ssim_value:.3f}, Reg={reg_loss:.4f})"
                )

            # Send progress update via callback (every iteration for real-time feedback)
            if progress_callback:
                # Calculate progress percentage (40% to 80% during optimization)
                optimization_progress = 40 + (40 * iter_idx / iterations)

                # Encode rendered image to base64 for preview
                import base64
                import io
                from PIL import Image

                try:
                    # Convert numpy array to PIL Image
                    # rendered_img is (H, W, 3) in RGB format, range [0, 1]
                    # Scale to [0, 255] before converting to uint8
                    pil_img = Image.fromarray((rendered_img * 255).astype(np.uint8))

                    # Encode to base64
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    rendered_image = f"data:image/png;base64,{img_base64}"
                except Exception as e:
                    logger.warning(f"[Optimization] Failed to encode rendered image: {e}")
                    rendered_image = None

                # Send progress update
                try:
                    progress_callback({
                        'progress': optimization_progress,
                        'iteration': iter_idx,
                        'total_iterations': iterations,
                        'loss': float(total_loss),
                        'rendered_image': rendered_image,  # Send actual rendered image
                        'status': f'Optimizing... (iteration {iter_idx}/{iterations})'
                    })
                except Exception as e:
                    logger.warning(f"[Optimization] Failed to send progress update: {e}")

            # Save best
            if total_loss < best_loss:
                best_loss = total_loss
                best_gaussians = [g.copy() for g in gaussians]

            # Simple opacity optimization using finite differences
            # (Full gradient descent would require differentiable renderer)
            # Use gradient accumulation: compute all gradients first, then apply together

            delta = 0.001
            gradients = []
            old_opacities = [g.opacity for g in gaussians]

            # Phase 1: Compute all gradients (read-only)
            for i, g in enumerate(gaussians):
                old_opacity = old_opacities[i]

                # Try increasing opacity (suppress renderer stdout)
                g.opacity = np.clip(old_opacity + delta, 0.01, 1.0)
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    rendered_plus = renderer.render(gaussians)
                loss_plus = np.mean(np.abs(rendered_plus.astype(np.float32) - target_rgb.astype(np.float32)))

                # Try decreasing opacity (suppress renderer stdout)
                g.opacity = np.clip(old_opacity - delta, 0.01, 1.0)
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    rendered_minus = renderer.render(gaussians)
                loss_minus = np.mean(np.abs(rendered_minus.astype(np.float32) - target_rgb.astype(np.float32)))

                # Restore original opacity
                g.opacity = old_opacity

                # Compute and store gradient
                gradient = (loss_plus - loss_minus) / (2 * delta)
                gradients.append(gradient)

            # Phase 2: Apply all gradients together
            for i, g in enumerate(gaussians):
                new_opacity = old_opacities[i] - learning_rate * gradients[i]
                g.opacity = np.clip(new_opacity, 0.01, 1.0)

            # Log gradient statistics periodically
            if iter_idx % 5 == 0:
                grad_mean = np.mean(np.abs(gradients))
                grad_max = np.max(np.abs(gradients))
                opacity_changes = [abs(gaussians[i].opacity - old_opacities[i]) for i in range(len(gaussians))]
                opacity_delta_mean = np.mean(opacity_changes)
                opacity_delta_max = np.max(opacity_changes)
                logger.info(
                    f"[Optimization] Iter {iter_idx}: "
                    f"Grad(μ={grad_mean:.6f}, max={grad_max:.6f}), "
                    f"ΔOpacity(μ={opacity_delta_mean:.6f}, max={opacity_delta_max:.6f})"
                )

            # Early stopping
            if total_loss < 0.05:  # Good enough
                logger.info(f"[Optimization] Early stopping at iteration {iter_idx}, loss={total_loss:.4f}")
                break

        logger.info(f"[Optimization] Completed. Final loss: {total_loss:.4f}, Best loss: {best_loss:.4f}")

        # Return the optimized gaussians (not the saved best from early iterations)
        # If current is better, use it; otherwise use best saved
        if total_loss < best_loss:
            final_gaussians = gaussians
        else:
            final_gaussians = best_gaussians

        # Cull low-opacity Gaussians to remove edge artifacts
        opacity_threshold = 0.05
        culled_gaussians = [g for g in final_gaussians if g.opacity > opacity_threshold]

        if len(culled_gaussians) < len(final_gaussians):
            logger.info(f"[Optimization] Culled {len(final_gaussians) - len(culled_gaussians)} low-opacity Gaussians (< {opacity_threshold})")

        logger.info(f"[Optimization] Returning {len(culled_gaussians)} optimized Gaussians")
        return culled_gaussians

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
        # Match brush.py default frame: tangent=X, normal=Z(up), binormal=Y
        tangent = np.array([1, 0, 0])  # X-axis (horizontal, brush direction)
        normal = np.array([0, 0, 1])   # Z-axis (up, surface normal)
        binormal = np.array([0, 1, 0]) # Y-axis (forward, perpendicular)

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

    def _visualize_pipeline(self, brush_name: str):
        """
        Visualize the entire conversion pipeline in a single comprehensive image

        Args:
            brush_name: Name of the brush being converted
        """
        try:
            logger.info("[Visualizer] Generating pipeline visualization...")

            # Create figure with subplots
            fig = plt.figure(figsize=self.config.visualization.figure_size, dpi=self.config.visualization.dpi)
            fig.suptitle(f'Brush Conversion Pipeline: {brush_name}', fontsize=16, fontweight='bold')

            # Define grid layout (4 rows x 4 columns)
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

            # Row 1: Original, Depth, Alpha, Features
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[0, 3])

            # Row 2: Skeleton, Thickness, Flow, Importance
            ax5 = fig.add_subplot(gs[1, 0])
            ax6 = fig.add_subplot(gs[1, 1])
            ax7 = fig.add_subplot(gs[1, 2])
            ax8 = fig.add_subplot(gs[1, 3])

            # Row 3: Point Cloud, Luminance, Sampling, Gaussians
            ax9 = fig.add_subplot(gs[2, 0])
            ax10 = fig.add_subplot(gs[2, 1])
            ax11 = fig.add_subplot(gs[2, 2])
            ax12 = fig.add_subplot(gs[2, 3])

            # Row 4: Gaussian Detail Views
            ax13 = fig.add_subplot(gs[3, 0])
            ax14 = fig.add_subplot(gs[3, 1])
            ax15 = fig.add_subplot(gs[3, 2])
            ax16 = fig.add_subplot(gs[3, 3])

            # 1. Original Image
            self._plot_image(ax1, self.debug_data['original_image'], "1. Original Image")

            # 2. Depth Map
            self._plot_heatmap(ax2, self.debug_data['depth_map'], "2. Depth Map", cmap='plasma')

            # 3. Alpha Mask
            self._plot_heatmap(ax3, self.debug_data['alpha_mask'], "3. Alpha Mask", cmap='gray')

            # 4. Combined Features Overlay
            self._plot_features_overlay(ax4, self.debug_data['original_image'], self.debug_data['features'])

            # 5. Skeleton
            self._plot_heatmap(ax5, self.debug_data['features']['skeleton'], "5. Skeleton", cmap='gray')

            # 6. Thickness Map
            self._plot_heatmap(ax6, self.debug_data['features']['thickness'], "6. Thickness Map", cmap='hot')

            # 7. Flow Field
            self._plot_flow_field(ax7, self.debug_data['features'])

            # 8. Importance Map
            if 'importance_map' in self.debug_data:
                self._plot_heatmap(ax8, self.debug_data['importance_map'], "8. Importance Map", cmap='viridis')
            else:
                ax8.text(0.5, 0.5, 'No importance\nmap generated', ha='center', va='center')
                ax8.axis('off')

            # 9. Point Cloud Distribution
            self._plot_point_cloud(ax9)

            # 10. Luminance Distribution
            self._plot_luminance_histogram(ax10)

            # 11. Sampling Comparison
            if 'mask_before_sampling' in self.debug_data and 'mask_after_sampling' in self.debug_data:
                self._plot_sampling_comparison(ax11)
            else:
                ax11.text(0.5, 0.5, 'No sampling\nperformed', ha='center', va='center')
                ax11.axis('off')

            # 12. Gaussian Positions
            self._plot_gaussian_positions(ax12)

            # 13. Gaussian Scale Distribution
            self._plot_gaussian_scales(ax13)

            # 14. Gaussian Orientation
            self._plot_gaussian_orientations(ax14)

            # 15. Gaussian Opacity Distribution
            self._plot_gaussian_opacities(ax15)

            # 16. Statistics Summary
            self._plot_statistics_summary(ax16)

            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.config.debug.output_dir) / f"{brush_name}_{timestamp}_pipeline.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=self.config.visualization.dpi)
            plt.close(fig)

            logger.info(f"[Visualizer] ✓ Saved visualization to {output_path}")

        except Exception as e:
            logger.error(f"[Visualizer] Failed to generate visualization: {e}")
            import traceback
            traceback.print_exc()

    def _plot_image(self, ax, image, title):
        """Plot RGB/RGBA image"""
        if len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA - show alpha blended on white
            alpha = image[:, :, 3:4] / 255.0
            rgb = image[:, :, :3] / 255.0
            composite = rgb * alpha + (1 - alpha)
            ax.imshow(composite)
        elif len(image.shape) == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    def _plot_heatmap(self, ax, data, title, cmap='viridis'):
        """Plot heatmap with colorbar"""
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_features_overlay(self, ax, image, features):
        """Plot skeleton and thickness overlaid on original"""
        # Create composite
        base = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB) / 255.0 * 0.5

        # Overlay skeleton in red
        skeleton_mask = features['skeleton'] > 0
        base[skeleton_mask] = [1, 0, 0]  # Red for skeleton

        # Overlay thickness as heat
        thickness_norm = features['thickness'] / (features['thickness'].max() + 1e-8)
        base[:, :, 1] += thickness_norm * 0.5  # Add green for thickness
        base = np.clip(base, 0, 1)

        ax.imshow(base)
        ax.set_title('4. Features Overlay\n(Red=Skeleton, Green=Thickness)', fontsize=10, fontweight='bold')
        ax.axis('off')

    def _plot_flow_field(self, ax, features):
        """Plot optical flow field"""
        flow_x = features['flow_x']
        flow_y = features['flow_y']
        grad_mag = features['grad_magnitude']

        # Downsample for visualization
        h, w = flow_x.shape
        step = max(h // 20, w // 20, 1)

        y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step]

        ax.imshow(grad_mag, cmap='gray', alpha=0.5)
        ax.quiver(
            x_coords, y_coords,
            flow_x[y_coords, x_coords],
            flow_y[y_coords, x_coords],
            color='cyan', scale=50, width=0.003
        )
        ax.set_title('7. Flow Field', fontsize=10, fontweight='bold')
        ax.axis('off')

    def _plot_point_cloud(self, ax):
        """Plot 3D point cloud distribution"""
        points = self.debug_data['points']
        colors = self.debug_data['colors']

        # Project to 2D (x, y)
        ax.scatter(points[:, 0], points[:, 1], c=colors[:, 0], cmap='gray', s=5, alpha=0.6)
        ax.set_title(f'9. Point Cloud\n({len(points)} points)', fontsize=10, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    def _plot_luminance_histogram(self, ax):
        """Plot luminance value distribution"""
        colors = self.debug_data['colors']
        luminance = colors[:, 0]  # Grayscale, so all channels are same

        ax.hist(luminance, bins=50, color='gray', alpha=0.7, edgecolor='black')
        ax.set_title('10. Luminance Distribution', fontsize=10, fontweight='bold')
        ax.set_xlabel('Luminance Value')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_lum = np.mean(luminance)
        std_lum = np.std(luminance)
        ax.axvline(mean_lum, color='red', linestyle='--', label=f'Mean: {mean_lum:.3f}')
        ax.legend()

    def _plot_sampling_comparison(self, ax):
        """Compare before/after sampling"""
        before = self.debug_data['mask_before_sampling']
        after = self.debug_data['mask_after_sampling']

        # Create RGB composite
        h, w = before.shape
        composite = np.zeros((h, w, 3))
        composite[before, 0] = 0.5  # Red for all valid pixels
        composite[after, 1] = 1.0  # Green for sampled pixels

        ax.imshow(composite)
        ax.set_title(f'11. Sampling\nBefore: {np.sum(before)} → After: {np.sum(after)}',
                     fontsize=10, fontweight='bold')
        ax.axis('off')

    def _plot_gaussian_positions(self, ax):
        """Plot Gaussian positions with size indication"""
        gaussians = self.debug_data['gaussians_initial']

        for g in gaussians:
            # Draw ellipse representing Gaussian
            angle = np.arctan2(2*(g.rotation[3]*g.rotation[2] + g.rotation[0]*g.rotation[1]),
                               1 - 2*(g.rotation[1]**2 + g.rotation[2]**2))
            angle_deg = np.degrees(angle)

            ellipse = Ellipse(
                (g.position[0], g.position[1]),
                width=g.scale[0]*2, height=g.scale[1]*2,
                angle=angle_deg,
                facecolor='none',
                edgecolor=plt.cm.gray(g.color[0]),
                linewidth=0.5,
                alpha=g.opacity
            )
            ax.add_patch(ellipse)

        # Set axis limits
        positions = np.array([g.position for g in gaussians])
        ax.set_xlim(positions[:, 0].min() - 0.1, positions[:, 0].max() + 0.1)
        ax.set_ylim(positions[:, 1].min() - 0.1, positions[:, 1].max() + 0.1)
        ax.set_title(f'12. Gaussian Positions\n({len(gaussians)} Gaussians)', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    def _plot_gaussian_scales(self, ax):
        """Plot distribution of Gaussian scales"""
        gaussians = self.debug_data['gaussians_initial']
        scales = np.array([g.scale for g in gaussians])

        ax.hist(scales[:, 0], bins=30, alpha=0.7, label='X scale', color='red')
        ax.hist(scales[:, 1], bins=30, alpha=0.7, label='Y scale', color='green')
        ax.hist(scales[:, 2], bins=30, alpha=0.7, label='Z scale', color='blue')
        ax.set_title('13. Gaussian Scale Distribution', fontsize=10, fontweight='bold')
        ax.set_xlabel('Scale Value')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_scale = np.mean(scales[:, :2])  # X, Y scales
        ax.axvline(mean_scale, color='black', linestyle='--', linewidth=2, label=f'Mean XY: {mean_scale:.4f}')

    def _plot_gaussian_orientations(self, ax):
        """Plot Gaussian orientation distribution"""
        gaussians = self.debug_data['gaussians_initial']

        # Extract rotation angles
        angles = []
        for g in gaussians:
            angle = np.arctan2(2*(g.rotation[3]*g.rotation[2] + g.rotation[0]*g.rotation[1]),
                               1 - 2*(g.rotation[1]**2 + g.rotation[2]**2))
            angles.append(np.degrees(angle))

        ax.hist(angles, bins=36, color='purple', alpha=0.7, edgecolor='black')
        ax.set_title('14. Gaussian Orientations', fontsize=10, fontweight='bold')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

    def _plot_gaussian_opacities(self, ax):
        """Plot Gaussian opacity distribution"""
        gaussians = self.debug_data['gaussians_initial']
        opacities = [g.opacity for g in gaussians]

        ax.hist(opacities, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax.set_title('15. Gaussian Opacity Distribution', fontsize=10, fontweight='bold')
        ax.set_xlabel('Opacity')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_opacity = np.mean(opacities)
        ax.axvline(mean_opacity, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_opacity:.3f}')
        ax.legend()

    def _plot_statistics_summary(self, ax):
        """Plot text summary of statistics"""
        ax.axis('off')

        gaussians = self.debug_data['gaussians_initial']
        scales = np.array([g.scale for g in gaussians])
        opacities = [g.opacity for g in gaussians]
        colors = self.debug_data['colors']

        stats_text = f"""
CONVERSION STATISTICS

Gaussians: {len(gaussians)}
Points: {len(self.debug_data['points'])}

Scale (XY):
  Mean: {np.mean(scales[:, :2]):.4f}
  Std: {np.std(scales[:, :2]):.4f}
  Min: {np.min(scales[:, :2]):.4f}
  Max: {np.max(scales[:, :2]):.4f}

Opacity:
  Mean: {np.mean(opacities):.3f}
  Std: {np.std(opacities):.3f}

Luminance:
  Mean: {np.mean(colors[:, 0]):.3f}
  Std: {np.std(colors[:, 0]):.3f}
  Contrast: {np.max(colors[:, 0]) - np.min(colors[:, 0]):.3f}

Config:
  Scale Mult: {self.config.gaussian.scale_multiplier}
  Min Contrast: {self.config.contrast.min_contrast}
  eps2d: {self.config.rendering.eps2d}
        """

        ax.text(0.1, 0.9, stats_text.strip(), fontsize=9, fontfamily='monospace',
                verticalalignment='top', transform=ax.transAxes)
        ax.set_title('16. Statistics Summary', fontsize=10, fontweight='bold')


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
