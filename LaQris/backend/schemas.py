from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class TransactionStatusEnum(str, Enum):
    """Status transaksi resmi dari payment provider (P1 Item 7 di Readme)."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    REVERSED = "REVERSED"


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
# Identity Evidence Sub-Schema
# ─────────────────────────────────────────────────────────────

class IdentityFieldEvidenceSchema(BaseModel):
    physical: Optional[str] = None
    digital: Optional[str] = None
    match: bool = False
    similarity: Optional[float] = None
    priority: str = "SUPPORTING_SIGNAL"


class IdentityEvidenceSchema(BaseModel):
    nmid: IdentityFieldEvidenceSchema
    merchant_name: IdentityFieldEvidenceSchema
    acquirer: IdentityFieldEvidenceSchema
    tid: IdentityFieldEvidenceSchema


# ─────────────────────────────────────────────────────────────
# Current QR Risk Schema (4-Level: NORMAL/CAUTION/WARNING/DANGER)
# ─────────────────────────────────────────────────────────────

class CurrentQRRiskSchema(BaseModel):
    risk_level: str                    # "NORMAL" | "CAUTION" | "WARNING" | "DANGER"
    risk_label: str                    # Human-readable label
    risk_color: str                    # "green" | "yellow" | "orange" | "red"
    overall_risk_score: float
    trust_score: float
    decision: str = "ALLOW"            # "ALLOW" | "WARN" | "BLOCK" (Security Decision Contract)
    reason_codes: List[str] = []       # e.g. ["NMID_MISMATCH"], ["CRC_INVALID"], ["NAME_COMPLETELY_DIFFERENT"]
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
    identity_evidence: Optional[Dict[str, Any]] = None


class ScanResponseSchema(BaseModel):
    session_id: str
    current_qr_risk: CurrentQRRiskSchema
    merchant_reputation: ReputationScoreSchema
    visualization_url: str


# ─────────────────────────────────────────────────────────────
# Pre-Payment Verification Schemas (Contract A: /api/v1/verify)
# ─────────────────────────────────────────────────────────────

class VerifyRequestSchema(BaseModel):
    session_id: Optional[str] = None
    verification_id: Optional[str] = None
    provider: str = "DemoPay"
    qr_payload: Optional[str] = None
    user_id: Optional[str] = None
    amount: Optional[float] = None


class VerifyMerchantInfoSchema(BaseModel):
    digital_name: Optional[str] = None
    physical_name: Optional[str] = None
    nmid: Optional[str] = None
    acquirer: Optional[str] = None
    terminal_id: Optional[str] = None


class VerifyResponseSchema(BaseModel):
    verification_id: str
    decision: str                      # "ALLOW" | "WARN" | "BLOCK"
    risk_level: str                    # "NORMAL" | "CAUTION" | "WARNING" | "DANGER"
    risk_score: float
    trust_score: float
    reason_codes: List[str] = []
    merchant_info: VerifyMerchantInfoSchema
    can_proceed_payment: bool          # True jika ALLOW atau WARN, False jika BLOCK
    is_bound: bool = False
    expires_at: Optional[datetime] = None
    message: str = "Verifikasi pre-payment berhasil."


# ─────────────────────────────────────────────────────────────
# Verification Session Detail Schema
# ─────────────────────────────────────────────────────────────

class VerificationSessionDetailSchema(BaseModel):
    id: int
    session_id: str
    user_id: Optional[str] = None
    nmid: Optional[str] = None
    digital_name: Optional[str] = None
    physical_name: Optional[str] = None
    scanned_at: datetime
    status: str
    trust_score: float
    risk_level: str
    reputation_score: float
    decision: str = "ALLOW"
    reason_codes: List[str] = []
    expires_at: Optional[datetime] = None
    is_bound: bool = False
    bound_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────
# Payment Transaction Schemas (Post-Payment Event Layer - Contract B)
# ─────────────────────────────────────────────────────────────

class PaymentTransactionSchema(BaseModel):
    id: int
    verification_session_id: Optional[str] = None
    provider: str
    provider_transaction_id: str
    merchant_id: Optional[int] = None
    nmid: Optional[str] = None
    user_id: Optional[str] = None
    amount: float
    status: str
    response_code: str
    invoice_number: Optional[str] = None
    terminal_id: Optional[str] = None
    transaction_time: Optional[datetime] = None
    latency_ms: int = 0
    retry_count: int = 0
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PaymentTransactionCreateSchema(BaseModel):
    verification_id: Optional[str] = None
    verification_session_id: Optional[str] = None
    provider: str = "DemoPay"
    provider_transaction_id: str
    amount: float
    status: TransactionStatusEnum        # Mandatory TransactionStatusEnum (P1 Item 7)
    response_code: str                   # Mandatory ISO 8583 / ASPI Response Code (e.g. "00")
    merchant_id: Optional[Any] = None
    merchant_name: Optional[str] = None
    nmid: Optional[str] = None
    user_id: Optional[str] = None
    terminal_id: Optional[str] = None
    invoice_number: Optional[str] = None
    transaction_time: Optional[str] = None
    latency_ms: int = 0
    retry_count: int = 0


class PaymentTransactionResponseSchema(BaseModel):
    success: bool
    message: str
    transaction_id: str
    verification_id: str
    status: str
    amount: float
    is_bound: bool = True
    merchant_reputation_impact: Optional[Dict[str, Any]] = None


# Contract B Aliases
TransactionEventCreateSchema = PaymentTransactionCreateSchema
TransactionEventResponseSchema = PaymentTransactionResponseSchema


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
# Feedback Submission Schema (Legacy + Contract C)
# ─────────────────────────────────────────────────────────────

class FeedbackSubmitSchema(BaseModel):
    nmid: str
    category: str           # "PENIPUAN_STIKER_QRIS_PALSU", "KETIDAKSESUAIAN_IDENTITAS_MERCHANT", dll
    severity: str = "MEDIUM"# "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    description: Optional[str] = None
    transaction_ref: Optional[str] = None
    has_evidence: bool = False


class FeedbackResponseSchema(BaseModel):
    success: bool
    message: str
    evidence_level: int
    new_reputation_score: float
    detected_category: Optional[str] = None
    detected_category_key: Optional[str] = None
    confidence: Optional[float] = None
    severity: Optional[str] = None
    action: Optional[str] = None


class FeedbackContractCSchema(BaseModel):
    """
    Contract C: Post-Payment User Feedback Payload
    Sesuai Bagian 32 & 37 Dokumen Roadmap LaQris.
    """
    verification_id: Optional[str] = None      # e.g. "LQ-V-001"
    transaction_id: Optional[str] = None       # e.g. "TX-001"
    comment: str                               # Cerita/ulasan pengalaman pengguna
    nmid: Optional[str] = None                 # Opsional jika verification_id disertakan
    manual_category: Optional[str] = None      # Opsional input manual pengguna
    has_evidence: bool = False                 # Apakah mengunggah file bukti fisik
    user_id: Optional[str] = None              # Identitas akun pengguna pengulas


class FeedbackContractCResponseSchema(BaseModel):
    """
    Contract C: Post-Payment Feedback Response
    Mengembalikan hasil inferensi NLP IndoBERT, tingkat bukti, dan dampak EMRS toko.
    """
    success: bool = True
    message: str
    event_type: str                            # e.g. "KETIDAKSESUAIAN_IDENTITAS_MERCHANT"
    category_title: str                        # Judul ramah pengguna
    severity: str                              # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    evidence_level: int                        # 0: comment only, 1: +verification, 2: +tx event
    confidence: float                          # Confidence model (0.0 - 1.0)
    merchant_reputation_updated: bool = True
    previous_reputation_score: Optional[float] = None
    new_reputation_score: float
    recommended_action: Optional[str] = None
    model_used: Optional[str] = "IndoBERT-FineTuned-EMRS"
    verification_id: Optional[str] = None
    transaction_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


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
    account_number: Optional[str] = "1858868768"
    account_type: Optional[str] = "TAPLUS"
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponseSchema(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponseSchema
