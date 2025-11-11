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
    DEFAULT_BRUSH_SPACING: float = 0.1
    DEFAULT_BRUSH_SIZE: float = 1.0
    DEFAULT_GAUSSIAN_OPACITY: float = 0.8

    # Spline settings
    SPLINE_POINT_THRESHOLD: float = 0.02  # Minimum distance between points

    # Performance settings
    MAX_GAUSSIANS: int = 100000  # Maximum number of Gaussians in scene

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
