from fastapi import Depends
from functools import lru_cache

from app.core.config import Settings, settings
from app.services.yolo_service import YOLOService


@lru_cache()
def get_settings() -> Settings:
    return settings


# Singleton YOLO service — model loaded once on first request
_yolo_service: YOLOService | None = None


def get_yolo_service() -> YOLOService:
    global _yolo_service
    if _yolo_service is None:
        _yolo_service = YOLOService(
            model_path=settings.YOLO_MODEL_PATH,
            confidence=settings.YOLO_CONFIDENCE_THRESHOLD,
            iou=settings.YOLO_IOU_THRESHOLD,
        )
    return _yolo_service
