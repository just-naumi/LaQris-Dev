from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.core.dependencies import get_yolo_service
from app.models.detection import DetectionResponse
from app.services.yolo_service import YOLOService
from app.core.config import settings

router = APIRouter()


@router.post("/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(..., description="Image file to run YOLO detection on"),
    yolo: YOLOService = Depends(get_yolo_service),
):
    """
    Upload an image and receive YOLO object detection results.
    Supported formats: JPEG, PNG, WebP, BMP.
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, WebP, or BMP.",
        )

    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Max allowed: {settings.MAX_IMAGE_SIZE_MB} MB.",
        )

    # Run YOLO inference
    results = await yolo.detect_from_bytes(contents)
    return results
