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
# LaQris Observation History Schema
# (Data dikumpulkan dari verification_sessions — BUKAN transaksi)
# ─────────────────────────────────────────────────────────────

class ObservationHistorySchema(BaseModel):
    total_observations: int           # Total scan QRIS oleh pengguna LaQris
    unique_observers: int             # Jumlah user unik yang pernah scan merchant ini
    first_observed: Optional[str]     # Tanggal pertama kali di-scan ("dd MMM yyyy")
    last_observed: Optional[str]      # Tanggal terakhir di-scan
    identity_match: int               # Jumlah scan yang identitasnya MATCH
    identity_mismatch: int            # Jumlah scan yang identitasnya MISMATCH
    physical_anomaly: int             # Jumlah scan yang terindikasi anomali fisik
    identity_match_rate: float        # match / total (0.0 - 100.0 %)
    complaint_rate: Optional[float]   # verified_complaints / total_observations (%)
    verified_feedback: int            # Total feedback terverifikasi (evidence_level == 2)
    complaints: int                   # Total complaint/report
    disputes: int                     # Total dispute (sengketa terverifikasi)


# ─────────────────────────────────────────────────────────────
# EMRS Component Breakdown
# ─────────────────────────────────────────────────────────────

class EMRSComponents(BaseModel):
    A: float                          # Authenticity / Identity Consistency (40%)
    C: float                          # Complaint Score (30%)
    D: float                          # Dispute Score (20%)
    L: float                          # Observed Longevity & History (10%)
    T_observed: Optional[float] = None


class ReputationScoreSchema(BaseModel):
    reputation_score: Optional[float]          # 0–100 final EMRS (None jika belum terdaftar)
    grade: str                         # "Excellent" | "Very Good" | "Good" | "Fair" | "Poor" | "Belum Terdaftar"
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
    observation_history: Optional[ObservationHistorySchema] = None


# ─────────────────────────────────────────────────────────────
# QRIS Raw EMVCo Payload Analysis
# ─────────────────────────────────────────────────────────────

class QRISRawAnalysisSchema(BaseModel):
    point_of_initiation: str
    initiation_type_code: str
    mcc_code: str
    mcc_category: str
    nmid_parsed: Dict[str, Any]
    currency: str
    crc_checksum: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Current QR Risk Schema (4-Level: NORMAL/CAUTION/WARNING/DANGER)
# ─────────────────────────────────────────────────────────────

class CurrentQRRiskSchema(BaseModel):
    risk_level: str                    # "NORMAL" | "CAUTION" | "WARNING" | "DANGER"
    risk_label: str                    # Human-readable label
    risk_color: str                    # "green" | "yellow" | "orange" | "red"
    overall_risk_score: float
    trust_score: float
    is_mismatch: bool
    name_similarity: float
    match_level: str
    explanation: str                   # Pesan aman secara hukum untuk user
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
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    role: str = "PENGGUNA"  # "PENGGUNA" | "MERCHANT"


class UserLoginSchema(BaseModel):
    email: str
    password: str


class UserResponseSchema(BaseModel):
    id: int
    user_id: str
    username: str
    email: str
    status: str
    full_name: Optional[str] = None
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponseSchema(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponseSchema
