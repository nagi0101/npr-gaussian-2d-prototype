"""Helper utilities for backend"""

import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import List
import json


def numpy_to_base64_png(image: np.ndarray) -> str:
    """
    Convert numpy array to base64-encoded PNG string

    Args:
        image: numpy array (H, W, 3) in range [0, 1] (float) or [0, 255] (uint8)

    Returns:
        Base64-encoded PNG string
    """
    # Convert to uint8 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    # Create PIL Image
    pil_image = Image.fromarray(image, mode='RGB')

    # Save to bytes buffer
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return img_base64


def base64_to_numpy(base64_string: str) -> np.ndarray:
    """
    Convert base64-encoded image to numpy array

    Args:
        base64_string: Base64-encoded image

    Returns:
        numpy array (H, W, 3)
    """
    # Decode base64
    img_bytes = base64.b64decode(base64_string)

    # Load image
    buffer = BytesIO(img_bytes)
    pil_image = Image.open(buffer)

    # Convert to numpy
    image = np.array(pil_image)

    return image


def gaussian_list_to_json(gaussians: List) -> list:
    """
    Convert list of Gaussian2D objects to JSON-serializable list

    Args:
        gaussians: List of Gaussian2D objects

    Returns:
        List of dictionaries
    """
    return [g.to_dict() for g in gaussians]


def json_to_gaussian_list(data: list):
    """
    Convert JSON list to Gaussian2D objects

    Args:
        data: List of dictionaries

    Returns:
        List of Gaussian2D objects
    """
    from backend.core.gaussian import Gaussian2D
    return [Gaussian2D.from_dict(d) for d in data]


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range"""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation"""
    return a + (b - a) * t


def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 3D points"""
    return np.linalg.norm(p2 - p1)
