"""
Depth Estimation Module for 2D-to-3DGS Brush Conversion

Uses MiDaS (Monocular Depth Estimation) to extract depth information
from 2D brush stroke images. Provides fallback heuristics for abstract art.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
import cv2
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


class MiDaSDepthEstimator:
    """
    MiDaS-based depth estimation for brush stroke images

    Supports MiDaS v3.1 models with multiple size options:
    - DPT-Large: Best quality, slower (default)
    - DPT-Hybrid: Good balance of quality and speed
    - MiDaS-small: Fastest, lower quality
    """

    def __init__(
        self,
        model_type: str = "DPT_Large",
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize MiDaS depth estimator

        Args:
            model_type: Model variant ("DPT_Large", "DPT_Hybrid", "MiDaS_small")
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            model_path: Optional path to cached model weights
        """
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info(f"[DepthEstimator] Initializing on device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"[DepthEstimator] GPU: {torch.cuda.get_device_name(0)}")

        self.model_type = model_type
        self.model = None
        self.transform = None

        # Model paths
        self.model_dir = Path(model_path) if model_path else Path("backend/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load MiDaS model and preprocessing transform"""
        try:
            # Try to import MiDaS
            torch.hub._validate_not_a_forked_repo = (
                lambda a, b, c: True
            )  # Bypass fork check

            # Download and load model from torch hub
            if self.model_type == "DPT_Large":
                self.model = torch.hub.load(
                    "intel-isl/MiDaS",
                    "DPT_Large",
                    trust_repo=True,
                    skip_validation=True,
                )
            elif self.model_type == "DPT_Hybrid":
                self.model = torch.hub.load(
                    "intel-isl/MiDaS",
                    "DPT_Hybrid",
                    trust_repo=True,
                    skip_validation=True,
                )
            else:  # MiDaS_small
                self.model = torch.hub.load(
                    "intel-isl/MiDaS",
                    "MiDaS_small",
                    trust_repo=True,
                    skip_validation=True,
                )

            # Load transforms
            self.transform = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )

            # Select appropriate transform
            if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = self.transform.dpt_transform
            else:
                self.transform = self.transform.small_transform

            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"[DepthEstimator] ✓ Loaded {self.model_type} model")

        except Exception as e:
            logger.warning(f"[DepthEstimator] Failed to load MiDaS: {e}")
            logger.info("[DepthEstimator] Falling back to heuristic depth estimation")
            self.model = None

    def estimate_depth(
        self,
        image: np.ndarray,
        normalize: bool = True,
        depth_range: Tuple[float, float] = (0.0, 1.0),
        use_fallback: bool = True,
    ) -> np.ndarray:
        """
        Estimate depth map from image

        Args:
            image: Input image (H, W, 3) BGR or (H, W) grayscale
            normalize: Whether to normalize depth to specified range
            depth_range: (min, max) depth values for normalization
            use_fallback: Use intensity-based fallback if MiDaS fails

        Returns:
            Depth map (H, W) with values in depth_range
        """
        # Handle grayscale images
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            # BGRA to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image_rgb = image

        # Try MiDaS depth estimation
        if self.model is not None:
            try:
                depth_map = self._estimate_midas_depth(image_rgb)
            except Exception as e:
                logger.warning(f"[DepthEstimator] MiDaS failed: {e}")
                if use_fallback:
                    depth_map = self._estimate_heuristic_depth(image)
                else:
                    raise
        elif use_fallback:
            # Use fallback if model not available
            depth_map = self._estimate_heuristic_depth(image)
        else:
            raise RuntimeError("MiDaS model not available and fallback disabled")

        # Normalize depth to specified range
        if normalize:
            depth_map = self._normalize_depth(depth_map, depth_range)

        return depth_map

    def _estimate_midas_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Estimate depth using MiDaS model

        Args:
            image_rgb: RGB image (H, W, 3)

        Returns:
            Raw depth map (H, W)
        """
        original_shape = image_rgb.shape[:2]

        # Preprocess image
        input_batch = self.transform(image_rgb).to(self.device)

        # Run inference
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=original_shape,
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy
        depth_map = prediction.cpu().numpy()

        return depth_map

    def _estimate_heuristic_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback heuristic depth estimation based on intensity and gradients

        Assumptions:
        - Darker regions are closer (higher paint density)
        - Stroke centers are thicker than edges
        - Smooth transitions follow brush pressure

        Args:
            image: Input image (any format)

        Returns:
            Heuristic depth map (H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Normalize to [0, 1]
        gray = gray.astype(np.float32) / 255.0

        # 1. Base depth from inverted intensity (darker = closer)
        depth_intensity = 1.0 - gray

        # 2. Distance transform for stroke thickness
        # Threshold to get stroke mask
        _, binary = cv2.threshold(gray, 0.1, 1.0, cv2.THRESH_BINARY_INV)
        binary = (binary * 255).astype(np.uint8)

        # Distance from edges
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_normalized = dist_transform / (dist_transform.max() + 1e-8)

        # 3. Gradient magnitude for texture detail
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_normalized = grad_magnitude / (grad_magnitude.max() + 1e-8)

        # 4. Combine heuristics
        depth_map = (
            0.5 * depth_intensity  # Base depth from darkness
            + 0.3 * dist_normalized  # Thickness from distance
            + 0.2 * (1.0 - grad_normalized)  # Smoother areas are higher
        )

        # 5. Apply smoothing for coherence
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 1.0)

        return depth_map

    def _normalize_depth(
        self, depth_map: np.ndarray, depth_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Normalize depth map to specified range

        Args:
            depth_map: Raw depth map
            depth_range: (min, max) target range

        Returns:
            Normalized depth map
        """
        # Handle constant depth
        if depth_map.max() == depth_map.min():
            return np.full_like(depth_map, depth_range[0])

        # Normalize to [0, 1]
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        # Scale to target range
        depth_scaled = depth_norm * (depth_range[1] - depth_range[0]) + depth_range[0]

        return depth_scaled

    def estimate_with_profiles(
        self, image: np.ndarray, profile: str = "convex", strength: float = 1.0
    ) -> np.ndarray:
        """
        Estimate depth with artistic profile presets

        Args:
            image: Input brush stroke image
            profile: Depth profile ("flat", "convex", "concave", "wavy")
            strength: Profile strength multiplier (0.0-2.0)

        Returns:
            Depth map with applied profile
        """
        # Get base depth
        depth_base = self.estimate_depth(image, normalize=True, depth_range=(0, 1))

        h, w = depth_base.shape

        if profile == "flat":
            # Minimal depth variation
            depth_map = depth_base * 0.1 * strength

        elif profile == "convex":
            # Center bulges outward
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            radial_factor = 1.0 - (dist_from_center / max_dist)
            depth_map = depth_base * (0.5 + 0.5 * radial_factor * strength)

        elif profile == "concave":
            # Center depressed inward
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            radial_factor = dist_from_center / max_dist
            depth_map = depth_base * (0.5 + 0.5 * radial_factor * strength)

        elif profile == "wavy":
            # Sinusoidal variation
            x = np.linspace(0, 4 * np.pi, w)
            y = np.linspace(0, 4 * np.pi, h)
            xx, yy = np.meshgrid(x, y)
            wave = 0.5 + 0.5 * np.sin(xx) * np.cos(yy)
            depth_map = depth_base * (0.5 + 0.5 * wave * strength)

        else:
            # Default to base depth
            depth_map = depth_base * strength

        # Apply mask from alpha channel if available
        if len(image.shape) == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3].astype(np.float32) / 255.0
            depth_map = depth_map * alpha

        return depth_map


def create_depth_estimator(
    prefer_midas: bool = True, device: Optional[str] = None
) -> MiDaSDepthEstimator:
    """
    Factory function to create depth estimator

    Args:
        prefer_midas: Try to load MiDaS model (fallback to heuristics if fails)
        device: Device preference ('cuda', 'cpu', or None for auto)

    Returns:
        Configured depth estimator instance
    """
    if prefer_midas:
        # Try to create with MiDaS
        try:
            estimator = MiDaSDepthEstimator(model_type="DPT_Large", device=device)
            logger.info("[DepthEstimator] Using MiDaS depth estimation")
        except Exception as e:
            logger.warning(f"[DepthEstimator] MiDaS unavailable: {e}")
            # Create with fallback only
            estimator = MiDaSDepthEstimator(
                model_type="DPT_Large", device="cpu"  # Heuristics don't need GPU
            )
            estimator.model = None  # Ensure using fallback
            logger.info("[DepthEstimator] Using heuristic depth estimation")
    else:
        # Explicitly use heuristics only
        estimator = MiDaSDepthEstimator(device="cpu")
        estimator.model = None
        logger.info("[DepthEstimator] Using heuristic depth estimation (by request)")

    return estimator


def test_depth_estimation():
    """Test depth estimation with sample image"""
    print("Testing depth estimation...")

    # Create test image (gradient brush stroke)
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.ellipse(test_image, (128, 128), (80, 40), 0, 0, 360, (200, 200, 200), -1)
    cv2.GaussianBlur(test_image, (15, 15), 5.0, test_image)

    # Test depth estimation
    estimator = create_depth_estimator(prefer_midas=False)  # Use heuristics for testing

    # Test different profiles
    profiles = ["flat", "convex", "concave", "wavy"]
    for profile in profiles:
        depth_map = estimator.estimate_with_profiles(test_image, profile=profile)
        print(
            f"✓ {profile} profile: depth range [{depth_map.min():.3f}, {depth_map.max():.3f}]"
        )

        # Save depth visualization
        depth_viz = (depth_map * 255).astype(np.uint8)
        cv2.imwrite(f"depth_{profile}.png", depth_viz)

    print("Depth estimation test complete!")


if __name__ == "__main__":
    test_depth_estimation()
