from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ReportSchema(BaseModel):
    id: int
    category: str
    description: Optional[str] = None
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True

class MerchantSchema(BaseModel):
    id: int
    nmid: str
    merchant_name: str
    acquirer: str
    rating: float
    total_reports: int
    verified_reports: int
    reports: List[ReportSchema] = []

    class Config:
        from_attributes = True

class ReputationBreakdown(BaseModel):
    qris_replacement: int
    additional_fee: int
    merchant_mismatch: int

class MerchantReputationSummary(BaseModel):
    nmid: str
    merchant_name: str
    rating: float
    total_reports: int
    verified_reports: int
    breakdown_categories: ReputationBreakdown

class ScanResponse(BaseModel):
    session_id: str
    skenario: str
    verdict_status: str
    risk_level: str  # HIGH_RISK, MEDIUM_RISK, SAFE
    trust_score: float
    explanation: str
    digital_entity: Dict[str, Any]
    physical_entity: Dict[str, Any]
    scan_quality: Dict[str, Any]
    evidence_breakdown: Dict[str, Any]
    reputation: Optional[MerchantReputationSummary] = None
    visualization_image_url: Optional[str] = None
