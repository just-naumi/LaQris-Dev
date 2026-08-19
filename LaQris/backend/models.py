from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Merchant(Base):
    __tablename__ = "merchants"

    id = Column(Integer, primary_key=True, index=True)
    nmid = Column(String, unique=True, index=True, nullable=False)
    merchant_name = Column(String, nullable=False)
    acquirer = Column(String, default="93600014")
    rating = Column(Float, default=5.0)
    total_reports = Column(Integer, default=0)
    verified_reports = Column(Integer, default=0)

    # Relasi ke tabel laporan
    reports = relationship("Report", back_populates="merchant", cascade="all, delete-orphan")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(Integer, ForeignKey("merchants.id"), nullable=False)
    category = Column(String, nullable=False)  # "QRIS Replacement", "Additional Fee", "Merchant Mismatch"
    description = Column(String, nullable=True)
    is_verified = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relasi balik ke merchant
    merchant = relationship("Merchant", back_populates="reports")

class VerificationSession(Base):
    __tablename__ = "verification_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    nmid = Column(String, nullable=True)
    digital_name = Column(String, nullable=True)
    physical_name = Column(String, nullable=True)
    scanned_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="PENDING")
    trust_score = Column(Float, default=0.0)
    risk_level = Column(String, default="LOW")
