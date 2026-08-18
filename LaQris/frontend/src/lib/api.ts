import type { DetectionResponse, HealthResponse } from "@/types/detection";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Generic fetch wrapper with error handling.
 */
async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`API error ${res.status}: ${errorText}`);
  }

  return res.json() as Promise<T>;
}

// ─── Endpoints ────────────────────────────────────────────────────────────────

/**
 * Check backend health.
 */
export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/v1/health");
}

/**
 * Upload an image file and run YOLO object detection.
 * @param file - The image File object from an <input type="file"> or drag-and-drop.
 */
export async function detectImage(file: File): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return apiFetch<DetectionResponse>("/api/v1/detection/image", {
    method: "POST",
    body: formData,
  });
}
