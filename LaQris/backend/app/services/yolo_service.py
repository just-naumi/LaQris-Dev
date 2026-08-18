import asyncio
import time
from pathlib import Path

from ai.inference import YOLOInference
from app.models.detection import DetectionResponse, DetectedObject, BoundingBox


class YOLOService:
    """
    Service layer wrapping YOLO inference.
    Handles model lifecycle and result serialization.
    """

    def __init__(self, model_path: str, confidence: float, iou: float):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self._inference = YOLOInference(
            model_path=model_path,
            confidence=confidence,
            iou=iou,
        )

    async def detect_from_bytes(self, image_bytes: bytes) -> DetectionResponse:
        """
        Run YOLO detection on raw image bytes.
        Runs inference in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        start = time.perf_counter()

        raw_results = await loop.run_in_executor(
            None, self._inference.predict_from_bytes, image_bytes
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        detections = [
            DetectedObject(
                label=obj["label"],
                confidence=round(obj["confidence"], 4),
                class_id=obj["class_id"],
                bbox=BoundingBox(
                    x1=obj["bbox"][0],
                    y1=obj["bbox"][1],
                    x2=obj["bbox"][2],
                    y2=obj["bbox"][3],
                ),
            )
            for obj in raw_results["detections"]
        ]

        return DetectionResponse(
            success=True,
            image_width=raw_results["image_width"],
            image_height=raw_results["image_height"],
            objects_detected=len(detections),
            detections=detections,
            inference_time_ms=round(elapsed_ms, 2),
        )
