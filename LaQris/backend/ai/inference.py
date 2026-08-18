import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO


class YOLOInference:
    """
    Low-level YOLO inference wrapper using Ultralytics.
    Model is loaded once and reused across requests (singleton via YOLOService).
    """

    def __init__(self, model_path: str, confidence: float = 0.5, iou: float = 0.45):
        model_file = Path(model_path)

        # Auto-download pretrained weights if not present locally
        if not model_file.exists():
            print(f"⬇️  Model not found at {model_path}. Downloading pretrained weights...")
            model_file.parent.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(str(model_file) if model_file.exists() else model_path)
        self.confidence = confidence
        self.iou = iou
        print(f"✅ YOLO model loaded: {self.model.model_name}")

    def predict_from_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Run inference on raw image bytes.
        Returns a dict with detections and image metadata.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)
        width, height = image.size

        results = self.model.predict(
            source=img_array,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "label": self.model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return {
            "image_width": width,
            "image_height": height,
            "detections": detections,
        }
