// Types matching the FastAPI Pydantic models

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectedObject {
  label: string;
  confidence: number;
  bbox: BoundingBox;
  class_id: number;
}

export interface DetectionResponse {
  success: boolean;
  image_width: number;
  image_height: number;
  objects_detected: number;
  detections: DetectedObject[];
  inference_time_ms: number;
}

export interface HealthResponse {
  status: string;
  timestamp: number;
  service: string;
}
