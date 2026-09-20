# models.py - Struktur Tabel Database LaQris
# Berisi model data untuk toko, laporan masalah, sesi scan QRIS, transaksi, dan akun pengguna.

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

# Masa berlaku sesi verifikasi QRIS (30 menit)
VERIFICATION_SESSION_TTL_MINUTES = 30


class Merchant(Base):
    """
    Tabel Toko / Merchant QRIS:
    Menyimpan profil toko, nomor identitas resmi (NMID), bank penerbit (acquirer),
    serta riwayat skor reputasi toko berdasarkan kejujuran transaksi.
    """
    __tablename__ = "merchants"

    id = Column(Integer, primary_key=True, index=True)
    nmid = Column(String, unique=True, index=True, nullable=False)   # ID Nasional Toko QRIS
    merchant_name = Column(String, nullable=False)                  # Nama resmi toko
    acquirer = Column(String, default="93600014")                   # Bank / e-wallet pengelola toko

    # Data Usia & Waktu Pendaftaran Toko
    registered_at = Column(DateTime, default=datetime.utcnow)

    # Catatan Transaksi Toko
    verified_transactions = Column(Integer, default=0)              # Total transaksi yang diawasi sistem
    successful_transactions = Column(Integer, default=0)            # Transaksi yang berhasil
    failed_transactions = Column(Integer, default=0)                # Transaksi yang gagal

    # Catatan Keaslian QRIS Toko
    identity_match_count = Column(Integer, default=0)               # Jumlah scan yang cocok (nama asli sesuai barcode)
    identity_mismatch_count = Column(Integer, default=0)            # Jumlah scan tidak cocok (indikasi beda nama)
    critical_mismatch_count = Column(Integer, default=0)            # Jumlah scan palsu/stiker ditimpa penipu

    # Data Tambahan & Skor Penilaian
    rating = Column(Float, default=5.0)                             # Bintang penilaian toko (skala 1-5)
    total_reports = Column(Integer, default=0)                      # Total laporan keluhan
    verified_reports = Column(Integer, default=0)                   # Laporan yang terbukti valid

    # Skor Reputasi Toko Terkini (0 - 100)
    reputation_score = Column(Float, default=50.0)

    # Hubungan dengan data laporan dan sengketa
    reports = relationship("Report", back_populates="merchant", cascade="all, delete-orphan")
    disputes = relationship("Dispute", back_populates="merchant", cascade="all, delete-orphan")


class Report(Base):
    """
    Tabel Laporan Masalah dari Pembeli:
    Menampung ulasan atau laporan pembeli jika menemukan stiker QRIS mencurigakan,
    nama toko berbeda, atau kasir meminta biaya tambahan ilegal.
    """
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(Integer, ForeignKey("merchants.id"), nullable=False)

    # Jenis keluhan (misal: nama toko beda, pungutan biaya tambahan, atau stiker rusak)
    category = Column(String, nullable=False)

    # Tingkat keparahan keluhan: "LOW" (ringan), "MEDIUM" (sedang), "HIGH" (berat), "CRITICAL" (sangat bahaya)
    severity = Column(String, default="MEDIUM")

    # Ulasan / cerita keluhan dari pembeli
    description = Column(Text, nullable=True)

    # Bukti transaksi: 1 = ulasan umum, 2 = terhubung langsung dengan bukti bayar resmi
    evidence_level = Column(Integer, default=1)

    # Nomor transaksi referensi agar satu pembayaran tidak bisa dilaporkan berkali-kali
    transaction_ref = Column(String, nullable=True)

    is_verified = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    merchant = relationship("Merchant", back_populates="reports")


class Dispute(Base):
    """
    Tabel Sengketa Transaksi:
    Menampung klaim sengketa keuangan yang sudah terverifikasi valid oleh sistem.
    """
    __tablename__ = "disputes"

    id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(Integer, ForeignKey("merchants.id"), nullable=False)
    description = Column(Text, nullable=True)                       # Rincian sengketa
    evidence_ref = Column(String, nullable=True)                    # Bukti nomor transaksi

    # Tingkat dampak sengketa: "MEDIUM", "HIGH", "CRITICAL"
    severity = Column(String, default="HIGH")

    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    merchant = relationship("Merchant", back_populates="disputes")


class VerificationSession(Base):
    """
    Tabel Sesi Pemeriksaan Scan QRIS:
    Mencatat hasil pemeriksaan kamera saat pembeli memindai QRIS fisik.
    Memastikan barcode yang discan asli dan nama toko cocok sebelum uang ditransfer.
    """
    __tablename__ = "verification_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)            # Kode unik sesi scan
    user_id = Column(String, index=True, nullable=True)             # Akun pembeli yang melakukan scan
    nmid = Column(String, nullable=True)                            # Nomor ID toko hasil scan
    digital_name = Column(String, nullable=True)                    # Nama toko yang ada di dalam barcode digital
    physical_name = Column(String, nullable=True)                   # Nama toko yang tertulis di banner fisik
    scanned_at = Column(DateTime, default=datetime.utcnow)          # Waktu pemindaian
    status = Column(String, default="PENDING")                      # "MATCH" (cocok) atau "MISMATCH" (beda)
    trust_score = Column(Float, default=0.0)                        # Nilai kepercayaan hasil scan (0-100)
    risk_level = Column(String, default="LOW")                      # Tingkat risiko: NORMAL, CAUTION, WARNING, DANGER
    reputation_score = Column(Float, default=50.0)                  # Skor toko saat scan dilakukan

    # Keputusan Keamanan Sistem
    decision = Column(String, default="ALLOW")                      # "ALLOW" (boleh bayar), "WARN" (waspada), "BLOCK" (tolak)
    reason_codes = Column(Text, default="[]")                       # Alasan deteksi jika ada bahaya
    expires_at = Column(DateTime, nullable=True)                    # Waktu batas sesi aktif (30 menit)
    is_bound = Column(Boolean, default=False)                       # Menandai apakah sesi sudah dipakai membayar
    bound_at = Column(DateTime, nullable=True)                      # Waktu pembayaran dilakukan
    amount = Column(Float, nullable=True)                           # Jumlah uang yang disetujui untuk dibayar
    amount_locked = Column(Boolean, default=False)                  # Tanda nominal sudah dikunci di server

    # Hubungan langsung ke data pembayaran
    payment_transaction = relationship("PaymentTransaction", back_populates="verification_session", uselist=False)


class PaymentTransaction(Base):
    """
    Tabel Bukti Transaksi Pembayaran:
    Menyimpan riwayat pembayaran resmi yang diproses oleh server pembayaran.
    """
    __tablename__ = "payment_transactions"

    id = Column(Integer, primary_key=True, index=True)
    verification_session_id = Column(String, ForeignKey("verification_sessions.session_id"), nullable=True, index=True)
    provider = Column(String, default="DemoPay")                    # Nama aplikasi pembayaran (DemoPay, DANA, GoPay, BCA)
    provider_transaction_id = Column(String, unique=True, index=True, nullable=False) # Nomor ID resmi transaksi
    merchant_id = Column(Integer, ForeignKey("merchants.id"), nullable=True)
    nmid = Column(String, nullable=True)                            # ID toko tujuan pembayaran
    user_id = Column(String, index=True, nullable=True)             # Akun pembeli yang membayar
    amount = Column(Float, nullable=False, default=0.0)             # Jumlah uang yang dibayar (Rupiah)
    status = Column(String, default="SUCCESS")                      # Status: SUCCESS, FAILED, TIMEOUT, CANCELLED
    response_code = Column(String, default="00")                    # Kode status bank (00 = sukses)
    invoice_number = Column(String, nullable=True)                  # Nomor invoice digital
    terminal_id = Column(String, nullable=True)                     # Kode mesin kasir / terminal
    transaction_time = Column(DateTime, default=datetime.utcnow)    # Waktu pemrosesan bank
    latency_ms = Column(Integer, default=0)                         # Kecepatan proses bank (milidetik)
    retry_count = Column(Integer, default=0)                        # Percobaan ulang jika sinyal terganggu
    created_at = Column(DateTime, default=datetime.utcnow)

    verification_session = relationship("VerificationSession", back_populates="payment_transaction")
    merchant = relationship("Merchant")


class User(Base):
    """
    Tabel Pengguna Aplikasi:
    Menyimpan data akun pembeli maupun pemilik toko yang terdaftar di aplikasi LaQris.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False) # Kode unik pengguna (contoh: USR-123456)
    username = Column(String, nullable=False)                         # Nama pengguna untuk login
    full_name = Column(String, nullable=True)                         # Nama lengkap pengguna
    email = Column(String, unique=True, index=True, nullable=False)   # Alamat email aktif
    phone = Column(String, nullable=True)                             # Nomor telepon / WhatsApp
    role = Column(String, default="PENGGUNA")                         # Peran: "PENGGUNA" atau "MERCHANT"
    password_hash = Column(String, nullable=False)                    # Kata sandi yang sudah dienkripsi aman
    status = Column(String, default="ACTIVE")                         # Status akun: ACTIVE, PENDING, SUSPENDED
    account_number = Column(String, default="1858868768")             # Nomor rekening dompet digital
    account_type = Column(String, default="TAPLUS")                   # Tipe rekening tabungan
    pin = Column(String, default="123456")                            # 6-digit PIN keamanan pembayaran
    created_at = Column(DateTime, default=datetime.utcnow)           # Tanggal pembuatan akun





