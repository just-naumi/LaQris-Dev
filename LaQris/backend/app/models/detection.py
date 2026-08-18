from pydantic import BaseModel
from typing import List


class BoundingBox(BaseModel):
    """Bounding box in [x1, y1, x2, y2] format (pixel coordinates)."""
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedObject(BaseModel):
    """A single detected object from YOLO."""
    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int


class DetectionResponse(BaseModel):
    """Full detection response returned to the frontend."""
    success: bool
    image_width: int
    image_height: int
    objects_detected: int
    detections: List[DetectedObject]
    inference_time_ms: float
