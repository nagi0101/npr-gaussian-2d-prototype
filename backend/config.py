"""Configuration settings for the backend server"""

import os
from typing import Optional

class Config:
    """Application configuration"""

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Rendering settings
    RENDER_WIDTH: int = 1024
    RENDER_HEIGHT: int = 768
    BACKGROUND_COLOR: tuple = (1.0, 1.0, 1.0)  # White

    # World space bounds
    WORLD_MIN: tuple = (-2.0, -2.0)
    WORLD_MAX: tuple = (2.0, 2.0)

    # Brush settings
    DEFAULT_BRUSH_SPACING: float = 0.3  # Increased from 0.1 to reduce Gaussian density
    DEFAULT_BRUSH_SIZE: float = 1.0
    DEFAULT_GAUSSIAN_OPACITY: float = 0.8

    # Phase 3 Features
    ENABLE_DEFORMATION: bool = True
    ENABLE_INPAINTING: bool = False
    INPAINT_OVERLAP_THRESHOLD: float = 0.1  # Distance threshold for overlap detection
    INPAINT_BLEND_STRENGTH: float = 0.3  # Opacity reduction strength (0.0-1.0)
    INPAINT_GLOBAL_MODE: bool = False  # Blend all overlapping pairs (not just consecutive)
    INPAINT_BLEND_MODE: str = 'smoothstep'  # Blending falloff: 'linear', 'smoothstep', 'gaussian'
    INPAINT_COLOR_BLENDING: bool = True  # Enable color blending for smoother transitions
    INPAINT_ANISOTROPIC: bool = True  # Use anisotropic (elliptical) distance for overlap detection

    # Optimization debug settings
    DEBUG_OPTIMIZATION: bool = True  # Save debug visualizations during optimization
    DEBUG_SAVE_INTERVAL: int = 1  # Save debug image every N iterations (1 = every iteration)
    DEBUG_OUTPUT_DIR: str = "debug_output"  # Directory for debug output files

    # Spline settings
    SPLINE_POINT_THRESHOLD: float = 0.02  # Minimum distance between points

    # Performance settings
    MAX_GAUSSIANS: int = 100000  # Maximum number of Gaussians in scene

    # Debug visualization settings
    DEBUG_MODE_ENABLED: bool = False  # Enable debug visualization by default
    SHOW_GAUSSIAN_ORIGINS: bool = True  # Show Gaussian center points
    SHOW_BASIS_VECTORS: bool = True  # Show local X, Y, Z axes for each Gaussian
    SHOW_SPLINE_FRAMES: bool = False  # Show spline tangent/normal/binormal frames
    SHOW_DEFORMATION_VECTORS: bool = False  # Show deformation displacement vectors
    DEBUG_OVERLAY_OPACITY: float = 0.8  # Debug overlay opacity (0.0-1.0)
    BASIS_VECTOR_LENGTH: int = 50  # Length of basis vectors in pixels (adjustable)
    DEBUG_AUTO_ENABLE_ON_ERROR: bool = True  # Auto-enable debug when deformation issues detected

    # CORS settings
    CORS_ORIGINS: list = [
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
    ]

    @classmethod
    def get(cls, key: str, default: Optional[any] = None) -> any:
        """Get configuration value"""
        return getattr(cls, key, default)


config = Config()
