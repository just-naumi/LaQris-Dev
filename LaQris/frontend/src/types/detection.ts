// Types matching the FastAPI Pydantic models

export interface ScanResponse {
  session_id: string;
  is_mismatch: boolean;
  risk_level: string;
  overall_risk_score: number;
  trust_score: number;
  name_similarity: number;
  match_level: string;
  explanation: string;
  physical_merchant: string;
  digital_merchant: string;
  digital_city: string;
  physical_nmid: string;
  digital_nmid: string;
  physical_acquirer: string;
  digital_acquirer: string;
  physical_tid: string;
  digital_tid: string;
  visualization_url: string;
  reputation: Record<string, unknown>;
  technical_info: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  service: string;
}
