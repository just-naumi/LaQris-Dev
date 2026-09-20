from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

# Unified Constant Lifecycle Sesi Verifikasi LaQris (P0 Item 3 di Readme.md)
VERIFICATION_SESSION_TTL_MINUTES = 30


class Merchant(Base):
    __tablename__ = "merchants"

    id = Column(Integer, primary_key=True, index=True)
    nmid = Column(String, unique=True, index=True, nullable=False)
    merchant_name = Column(String, nullable=False)
    acquirer = Column(String, default="93600014")

    # ── Longevity (L) ──────────────────────────────────────────
    registered_at = Column(DateTime, default=datetime.utcnow)

    # ── Transaction Reliability (T) ────────────────────────────
    verified_transactions = Column(Integer, default=0)
    successful_transactions = Column(Integer, default=0)
    failed_transactions = Column(Integer, default=0)

    # ── Authenticity / Identity Consistency (A) ─────────────────
    identity_match_count = Column(Integer, default=0)
    identity_mismatch_count = Column(Integer, default=0)
    critical_mismatch_count = Column(Integer, default=0)  # severity = CRITICAL mismatch

    # ── Legacy / UI fields ─────────────────────────────────────
    rating = Column(Float, default=5.0)
    total_reports = Column(Integer, default=0)
    verified_reports = Column(Integer, default=0)

    # ── Cached EMRS Score (updated on each scan/feedback) ──────
    reputation_score = Column(Float, default=50.0)

    # ── Relationships ──────────────────────────────────────────
    reports = relationship("Report", back_populates="merchant", cascade="all, delete-orphan")
    disputes = relationship("Dispute", back_populates="merchant", cascade="all, delete-orphan")


class Report(Base):
    """
    Complaint / laporan dari pengguna.
    Bisa tanpa bukti (evidence_level=1) atau dengan bukti transaksi (evidence_level=2).
    """
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(Integer, ForeignKey("merchants.id"), nullable=False)

    # Kategori: "QRIS Replacement" | "Additional Fee" | "Merchant Mismatch" | "General Complaint"
    category = Column(String, nullable=False)

    # Severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    severity = Column(String, default="MEDIUM")

    description = Column(Text, nullable=True)

    # Evidence Level: 1 = tanpa bukti transaksi, 2 = ada bukti terverifikasi
    evidence_level = Column(Integer, default=1)

    # Referensi transaksi untuk cegah duplikat feedback per transaksi
    transaction_ref = Column(String, nullable=True)

    is_verified = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    merchant = relationship("Merchant", back_populates="reports")


class Dispute(Base):
    """
    Dispute = sengketa transaksi yang sudah TERVERIFIKASI.
    Berbeda dari Report (complaint): dispute lebih berat bobotnya dalam EMRS.
    """
    __tablename__ = "disputes"

    id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(Integer, ForeignKey("merchants.id"), nullable=False)
    description = Column(Text, nullable=True)
    evidence_ref = Column(String, nullable=True)     # referensi bukti transaksi

    # Severity: "MEDIUM" | "HIGH" | "CRITICAL"
    severity = Column(String, default="HIGH")

    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    merchant = relationship("Merchant", back_populates="disputes")


class VerificationSession(Base):
    __tablename__ = "verification_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    user_id = Column(String, index=True, nullable=True) # User ID penanda pemilik scan
    nmid = Column(String, nullable=True)
    digital_name = Column(String, nullable=True)
    physical_name = Column(String, nullable=True)
    scanned_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="PENDING")       # "MATCH" | "MISMATCH"
    trust_score = Column(Float, default=0.0)
    risk_level = Column(String, default="LOW")
    reputation_score = Column(Float, default=50.0)   # EMRS score saat scan

    # ── Session Hardening & Security Decision ─────────────────
    decision = Column(String, default="ALLOW")       # "ALLOW" | "WARN" | "BLOCK"
    reason_codes = Column(Text, default="[]")        # JSON string array e.g. '["NMID_MISMATCH"]'
    expires_at = Column(DateTime, nullable=True)     # Sesi kedaluwarsa setelah 30 menit (VERIFICATION_SESSION_TTL_MINUTES)
    is_bound = Column(Boolean, default=False)        # Anti-replay: True jika sudah diikat transaksi
    bound_at = Column(DateTime, nullable=True)       # Waktu sesi diikat ke transaksi
    amount = Column(Float, nullable=True)            # Nominal transaksi yang dikunci (Payment Intent anti-tamper)
    amount_locked = Column(Boolean, default=False)   # Status penguncian nominal di server

    # Relasi 0..1 ke transaksi pembayaran (Post-Payment Transaction Event)
    payment_transaction = relationship("PaymentTransaction", back_populates="verification_session", uselist=False)


class PaymentTransaction(Base):
    """
    Entity transaksi pembayaran resmi yang terhubung dengan VerificationSession.
    Menjadi jembatan antara fase Pre-Payment Verification dan Post-Payment Result.
    """
    __tablename__ = "payment_transactions"

    id = Column(Integer, primary_key=True, index=True)
    verification_session_id = Column(String, ForeignKey("verification_sessions.session_id"), nullable=True, index=True)
    provider = Column(String, default="DemoPay")                 # "DemoPay" | "DANA" | "GOPAY" | "BCA"
    provider_transaction_id = Column(String, unique=True, index=True, nullable=False) # e.g. "TX-001"
    merchant_id = Column(Integer, ForeignKey("merchants.id"), nullable=True)
    nmid = Column(String, nullable=True)
    user_id = Column(String, index=True, nullable=True)          # User ownership tracking
    amount = Column(Float, nullable=False, default=0.0)
    status = Column(String, default="SUCCESS")                   # "SUCCESS" | "FAILED" | "BLOCKED"
    response_code = Column(String, default="00")                 # ISO 8583 / ASPI Response Code
    invoice_number = Column(String, nullable=True)
    terminal_id = Column(String, nullable=True)
    transaction_time = Column(DateTime, default=datetime.utcnow)
    latency_ms = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    verification_session = relationship("VerificationSession", back_populates="payment_transaction")
    merchant = relationship("Merchant")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, nullable=True)
    role = Column(String, default="PENGGUNA")        # "PENGGUNA" | "MERCHANT"
    password_hash = Column(String, nullable=False)
    status = Column(String, default="ACTIVE")        # "ACTIVE" | "PENDING" | "SUSPENDED"
    account_number = Column(String, default="1858868768")
    account_type = Column(String, default="TAPLUS")
    pin = Column(String, default="123456")
    created_at = Column(DateTime, default=datetime.utcnow)




