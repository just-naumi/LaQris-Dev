from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


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
    nmid = Column(String, nullable=True)
    digital_name = Column(String, nullable=True)
    physical_name = Column(String, nullable=True)
    scanned_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="PENDING")       # "MATCH" | "MISMATCH"
    trust_score = Column(Float, default=0.0)
    risk_level = Column(String, default="LOW")
    reputation_score = Column(Float, default=50.0)   # EMRS score saat scan


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, nullable=True)
    role = Column(String, default="PENGGUNA")        # "PENGGUNA" | "MERCHANT"
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

