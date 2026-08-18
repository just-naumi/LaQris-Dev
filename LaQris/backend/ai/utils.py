import io
import base64
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert raw bytes to a PIL Image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def pil_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """Convert PIL Image to base64-encoded string for sending over JSON."""
    raw = pil_to_bytes(image, fmt)
    return base64.b64encode(raw).decode("utf-8")


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> Image.Image:
    """
    Draw bounding boxes and labels on a PIL image.
    Useful for debug / preview endpoints.
    """
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        draw.text((x1 + 4, y1 + 4), label, fill=color)
    return image
