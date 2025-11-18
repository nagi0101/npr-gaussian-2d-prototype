"""
File Upload API for Brush Images

Handles image uploads for 2D-to-3DGS brush conversion.
Validates images and stores them temporarily for processing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import uuid
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io
import logging
import asyncio
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["upload"])

# Upload settings
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_IMAGE_DIMENSION = 2048  # Max width or height
ALLOWED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
UPLOAD_DIR = Path("backend/temp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Temporary storage for upload metadata
upload_storage: Dict[str, Dict[str, Any]] = {}


class ImageValidator:
    """Validates uploaded images for brush conversion"""

    @staticmethod
    def validate_file_size(file: UploadFile) -> None:
        """Check file size is within limits"""
        # Read file to check size (will reset after)
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset to beginning

        if size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // 1024 // 1024}MB"
            )

    @staticmethod
    def validate_file_format(filename: str) -> None:
        """Check file extension is allowed"""
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_FORMATS)}"
            )

    @staticmethod
    def validate_image_content(image_data: bytes) -> np.ndarray:
        """
        Validate image can be loaded and meets requirements

        Returns:
            Loaded image as numpy array
        """
        try:
            # Try to load with PIL first (better format support)
            pil_image = Image.open(io.BytesIO(image_data))

            # Check dimensions
            width, height = pil_image.size
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                # Resize if too large
                ratio = min(MAX_IMAGE_DIMENSION / width, MAX_IMAGE_DIMENSION / height)
                new_size = (int(width * ratio), int(height * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image from ({width}, {height}) to {new_size}")

            # Convert to numpy array (RGB or RGBA)
            if pil_image.mode == "RGBA":
                image_np = np.array(pil_image)
            elif pil_image.mode == "RGB":
                image_np = np.array(pil_image)
            elif pil_image.mode == "L":  # Grayscale
                image_np = np.array(pil_image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            else:
                # Convert to RGB
                pil_image = pil_image.convert("RGB")
                image_np = np.array(pil_image)

            # Convert RGB to BGR for OpenCV compatibility
            if len(image_np.shape) == 3:
                if image_np.shape[2] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                elif image_np.shape[2] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGRA)

            return image_np

        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )


@router.post("/upload-brush-image")
async def upload_brush_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: Optional[str] = None,
    profile: Optional[str] = "convex"  # Depth profile preset
) -> JSONResponse:
    """
    Upload a brush image for conversion to 3DGS

    Args:
        file: Image file to upload
        name: Optional brush name (defaults to filename)
        profile: Depth profile ("flat", "convex", "concave", "wavy")

    Returns:
        Upload ID and metadata for tracking conversion progress
    """
    # Generate unique upload ID
    upload_id = str(uuid.uuid4())
    logger.info(f"[Upload] Processing upload {upload_id}: {file.filename}")

    try:
        # Validate file
        ImageValidator.validate_file_size(file)
        ImageValidator.validate_file_format(file.filename)

        # Read file content
        content = await file.read()

        # Validate and process image
        image_np = ImageValidator.validate_image_content(content)

        # Save processed image to temp directory
        temp_path = UPLOAD_DIR / f"{upload_id}.png"
        cv2.imwrite(str(temp_path), image_np)

        # Extract image metadata
        height, width = image_np.shape[:2]
        has_alpha = len(image_np.shape) == 3 and image_np.shape[2] == 4

        # Create metadata
        metadata = {
            "upload_id": upload_id,
            "filename": file.filename,
            "name": name or Path(file.filename).stem,
            "profile": profile,
            "width": width,
            "height": height,
            "has_alpha": has_alpha,
            "file_path": str(temp_path),
            "timestamp": datetime.now().isoformat(),
            "status": "uploaded",
            "conversion_progress": 0
        }

        # Store metadata
        upload_storage[upload_id] = metadata

        # Schedule cleanup after 1 hour
        background_tasks.add_task(cleanup_upload, upload_id, delay_hours=1)

        logger.info(f"[Upload] ✓ Saved {upload_id} ({width}x{height})")

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "upload_id": upload_id,
                "metadata": metadata
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Upload] Failed to process upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upload-status/{upload_id}")
async def get_upload_status(upload_id: str) -> JSONResponse:
    """
    Get status of an uploaded brush image

    Args:
        upload_id: Upload ID to check

    Returns:
        Upload metadata and conversion status
    """
    if upload_id not in upload_storage:
        raise HTTPException(status_code=404, detail="Upload not found")

    metadata = upload_storage[upload_id]
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "metadata": metadata
        }
    )


@router.delete("/upload/{upload_id}")
async def delete_upload(upload_id: str) -> JSONResponse:
    """
    Delete an uploaded brush image

    Args:
        upload_id: Upload ID to delete

    Returns:
        Deletion confirmation
    """
    if upload_id not in upload_storage:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Get file path
    metadata = upload_storage[upload_id]
    file_path = Path(metadata["file_path"])

    # Delete file if exists
    if file_path.exists():
        file_path.unlink()
        logger.info(f"[Upload] Deleted file: {file_path}")

    # Remove from storage
    del upload_storage[upload_id]

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": f"Upload {upload_id} deleted"
        }
    )


@router.get("/uploads")
async def list_uploads() -> JSONResponse:
    """
    List all pending uploads

    Returns:
        List of upload metadata
    """
    uploads = list(upload_storage.values())

    # Sort by timestamp (newest first)
    uploads.sort(key=lambda x: x["timestamp"], reverse=True)

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "count": len(uploads),
            "uploads": uploads
        }
    )


async def cleanup_upload(upload_id: str, delay_hours: int = 1):
    """
    Background task to cleanup old uploads

    Args:
        upload_id: Upload to cleanup
        delay_hours: Hours to wait before cleanup
    """
    # Wait for specified delay
    await asyncio.sleep(delay_hours * 3600)

    # Check if still exists
    if upload_id in upload_storage:
        metadata = upload_storage[upload_id]

        # Only cleanup if not being processed
        if metadata.get("status") == "uploaded":
            file_path = Path(metadata["file_path"])
            if file_path.exists():
                file_path.unlink()
                logger.info(f"[Upload] Auto-cleaned upload {upload_id}")

            del upload_storage[upload_id]


def get_upload_image(upload_id: str) -> Optional[np.ndarray]:
    """
    Get uploaded image by ID

    Args:
        upload_id: Upload ID

    Returns:
        Image as numpy array or None if not found
    """
    if upload_id not in upload_storage:
        return None

    metadata = upload_storage[upload_id]
    file_path = Path(metadata["file_path"])

    if not file_path.exists():
        return None

    # Load image
    image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    return image


def update_upload_status(upload_id: str, status: str, progress: int = 0):
    """
    Update upload conversion status

    Args:
        upload_id: Upload ID
        status: New status ("uploaded", "processing", "completed", "failed")
        progress: Conversion progress (0-100)
    """
    if upload_id in upload_storage:
        upload_storage[upload_id]["status"] = status
        upload_storage[upload_id]["conversion_progress"] = progress
        logger.info(f"[Upload] Status update: {upload_id} -> {status} ({progress}%)")