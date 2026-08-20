from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# Report & Dispute Schemas
# ─────────────────────────────────────────────────────────────

class ReportSchema(BaseModel):
    id: int
    category: str
    severity: str
    description: Optional[str] = None
    evidence_level: int
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


class DisputeSchema(BaseModel):
    id: int
    description: Optional[str] = None
    severity: str
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────
# EMRS Component Breakdown (Refined Weights)
# ─────────────────────────────────────────────────────────────

class EMRSComponents(BaseModel):
    A: float                          # Authenticity / Identity Consistency (40%)
    C: float                          # Complaint Score (30%)
    D: float                          # Dispute Score (20%)
    L: float                          # Observed Longevity & History (10%)
    T_observed: Optional[float] = None # Observed Transaction Reliability (Optional, if in-app verified tx exist)


class ReputationScoreSchema(BaseModel):
    reputation_score: float            # 0–100 final EMRS
    grade: str                         # "Excellent" | "Very Good" | "Good" | "Fair" | "Poor"
    confidence_level: str              # "HIGH" | "MEDIUM" | "LOW"
    confidence_score: float            # 0–100%
    data_sufficiency_status: str       # "SUFFICIENT DATA" | "INSUFFICIENT HISTORY"
    components: EMRSComponents
    evidence_quality: str              # "HIGH" | "MEDIUM" | "LOW" | "INSUFFICIENT"
    total_evidence_count: int
    found_in_db: bool
    nmid: Optional[str] = None
    merchant_name: Optional[str] = None
    registered_at: Optional[datetime] = None
    first_seen_observed: Optional[str] = None
    last_seen_observed: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# QRIS Raw EMVCo Payload Analysis
# ─────────────────────────────────────────────────────────────

class QRISRawAnalysisSchema(BaseModel):
    point_of_initiation: str           # "Statis (Stiker Meja/Kasir)" | "Dinamis (EDC/Layar)"
    initiation_type_code: str          # "11" | "12"
    mcc_code: str                      # e.g. "5812"
    mcc_category: str                  # e.g. "Restoran / Rumah Makan"
    nmid_parsed: Dict[str, Any]        # {"country": "Indonesia (ID)", "estimated_reg_year": 2020, ...}
    currency: str                      # "360 (IDR)"
    crc_checksum: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Merchant Detail Schema (full)
# ─────────────────────────────────────────────────────────────

class MerchantSchema(BaseModel):
    id: int
    nmid: str
    merchant_name: str
    acquirer: str
    rating: float
    total_reports: int
    verified_reports: int
    verified_transactions: int
    successful_transactions: int
    identity_match_count: int
    identity_mismatch_count: int
    reputation_score: float
    registered_at: datetime
    reports: List[ReportSchema] = []
    disputes: List[DisputeSchema] = []

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────
# Current QR Risk Schema
# ─────────────────────────────────────────────────────────────

class CurrentQRRiskSchema(BaseModel):
    risk_level: str
    overall_risk_score: float
    trust_score: float
    is_mismatch: bool
    name_similarity: float
    match_level: str
    explanation: str
    physical_merchant: str
    digital_merchant: str
    digital_city: str
    physical_nmid: str
    digital_nmid: str
    physical_acquirer: str
    digital_acquirer: str
    physical_tid: str
    digital_tid: str
    technical_info: Dict[str, Any]
    qris_raw_analysis: Optional[QRISRawAnalysisSchema] = None


class ScanResponseSchema(BaseModel):
    session_id: str
    current_qr_risk: CurrentQRRiskSchema
    merchant_reputation: ReputationScoreSchema
    visualization_url: str


# ─────────────────────────────────────────────────────────────
# Feedback Submission Schema
# ─────────────────────────────────────────────────────────────

class FeedbackSubmitSchema(BaseModel):
    nmid: str
    category: str           # "QRIS Replacement" | "Additional Fee" | "Merchant Mismatch" | "General Complaint"
    severity: str           # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    description: Optional[str] = None
    transaction_ref: Optional[str] = None
    has_evidence: bool = False


class FeedbackResponseSchema(BaseModel):
    success: bool
    message: str
    evidence_level: int
    new_reputation_score: float


# ─────────────────────────────────────────────────────────────
# User Authentication Schemas
# ─────────────────────────────────────────────────────────────

class UserRegisterSchema(BaseModel):
    full_name: str
    email: str
    phone: Optional[str] = None
    role: str = "PENGGUNA"  # "PENGGUNA" | "MERCHANT"
    password: str


class UserLoginSchema(BaseModel):
    email: str
    password: str


class UserResponseSchema(BaseModel):
    id: int
    full_name: str
    email: str
    phone: Optional[str] = None
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

