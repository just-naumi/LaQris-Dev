import os
import shutil
import cv2
import numpy as np
import time
from collections import defaultdict
from threading import Lock
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

import hashlib
from database import init_db, get_db, reset_db
from models import Merchant, Report, Dispute, VerificationSession, User, PaymentTransaction
import schemas
import auth
from engine import (
    process_qris_verification,
    evaluasi_posisi_qris,
    submit_feedback_to_db,
    get_merchant_reputation_by_nmid,
    calculate_emrs,
    get_session_archive_data,
    list_all_scan_sessions
)
from nlp_classifier import classify_feedback

# Auto-initialize SQLite database on startup
init_db()

app = FastAPI(
    title="LaQris POC Tahap 2 — QRIS Fraud Detection & Evidence-Based Merchant Reputation",
    description=(
        "Backend API untuk deteksi stiker QRIS ditimpa, matching identitas fisik vs digital, "
        "dan kalkulasi EMRS (Evidence-Based Merchant Reputation Score) berbasis "
        "T·A·L·C·D dengan time decay dan evidence weighting."
    ),
    version="2.0.0"
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 30 di Readme2.md: CORS Restriction & Production Baseline
# ─────────────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "null"  # Mengizinkan file HTML lokal yang dibuka via protokol file://
]
custom_origins_env = os.getenv("LAQRIS_ALLOWED_ORIGINS")
if custom_origins_env:
    for o in custom_origins_env.split(","):
        if o.strip() and o.strip() not in ALLOWED_ORIGINS:
            ALLOWED_ORIGINS.append(o.strip())

ALLOW_ALL_CORS = os.getenv("LAQRIS_ALLOW_ALL_CORS", "false").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ALL_CORS else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 30 di Readme2.md: In-Memory Sliding Window Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────
class SlidingWindowRateLimiter:
    """
    In-memory thread-safe rate limiter untuk mencegah brute force dan flooding API.
    """
    def __init__(self):
        self._records = defaultdict(list)
        self._lock = Lock()

    def check(self, key: str, max_requests: int, window_seconds: int = 60) -> tuple:
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            timestamps = self._records[key]
            # Hapus timestamp di luar window waktu
            self._records[key] = [t for t in timestamps if t > cutoff]
            if len(self._records[key]) >= max_requests:
                oldest = self._records[key][0]
                retry_after = int(window_seconds - (now - oldest)) + 1
                return False, max(retry_after, 1)
            self._records[key].append(now)
            return True, max_requests - len(self._records[key])

rate_limiter = SlidingWindowRateLimiter()

def enforce_rate_limit(request: Request, max_requests: int, window_seconds: int = 60, endpoint_tag: str = "generic"):
    """Validasi pembatasan laju panggilan (Rate Limiter) per IP client."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    rate_key = f"{endpoint_tag}:{client_ip}"
    allowed, val = rate_limiter.check(rate_key, max_requests, window_seconds)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Terlalu banyak permintaan ({endpoint_tag}). Silakan coba lagi dalam {val} detik.",
            headers={"Retry-After": str(val)}
        )

FOLDER_BACKEND = os.path.dirname(os.path.abspath(__file__))
FOLDER_STATIC = os.path.join(FOLDER_BACKEND, "static")
FOLDER_FRONTEND = os.path.abspath(os.path.join(FOLDER_BACKEND, "..", "frontend"))

os.makedirs(FOLDER_STATIC, exist_ok=True)
os.makedirs(os.path.join(FOLDER_STATIC, "vis_output"), exist_ok=True)

# Mount static file routes
app.mount("/static", StaticFiles(directory=FOLDER_STATIC), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "app": "LaQris POC Tahap 2",
        "version": "2.0.0",
        "database": "SQLite",
        "reputation_engine": "EMRS v2 (T·A·L·C·D + Time Decay)"
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scan Endpoint (Main Pipeline)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/scan")
async def scan_qris_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Main pipeline endpoint.
    Menerima foto QRIS (upload atau nama sampel), menjalankan:
    1. Dual YOLO + TrOCR physical extraction
    2. EMVCo QR payload parsing
    3. Identity matching
    4. EMRS reputation scoring (terpisah dari QR risk)

    Returns: { session_id, current_qr_risk, merchant_reputation, visualization_url }
    """
    enforce_rate_limit(request, max_requests=30, window_seconds=60, endpoint_tag="scan-verify")
    folder_project_utama = os.path.abspath(os.path.join(FOLDER_BACKEND, "..", ".."))
    folder_physical_exp = os.path.join(folder_project_utama, "LaQris Physical Identity Extraction")

    gambar_input = None
    filename_base = "scan_upload"

    if sample_name:
        filename_base = sample_name.split(".")[0]
        path_sample = os.path.join(folder_physical_exp, f"{filename_base}.png")
        if not os.path.exists(path_sample):
            path_sample = os.path.join(folder_physical_exp, f"{filename_base}.jpeg")
        if os.path.exists(path_sample):
            gambar_input = cv2.imread(path_sample)

    elif file:
        filename_base = os.path.splitext(file.filename)[0]
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        gambar_input = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if gambar_input is None:
        raise HTTPException(status_code=400, detail="Upload foto QRIS atau pilih sampel gambar.")

    hasil = process_qris_verification(gambar_input, filename_base=filename_base, user_id=user_id)
    return hasil


@app.post("/api/scan/check-position")
async def check_position_endpoint(
    file: Optional[UploadFile] = File(None)
):
    """
    Lightweight endpoint untuk deteksi fisik QRIS dan evaluasi posisi/framing
    sebelum pemicu pemindaian otomatis (auto-capture).
    """
    if not file:
        raise HTTPException(status_code=400, detail="File frame kamera diperlukan.")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Gagal membaca frame gambar.")

    return evaluasi_posisi_qris(frame_bgr)


# ─────────────────────────────────────────────────────────────────────────────
# Merchant Directory
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/merchants", response_model=List[schemas.MerchantSchema])
def list_merchants(db: Session = Depends(get_db)):
    """Mengambil daftar seluruh merchant beserta laporan & dispute di SQLite."""
    return db.query(Merchant).all()


@app.get("/api/merchants/{nmid}", response_model=schemas.MerchantSchema)
def get_merchant_detail(nmid: str, db: Session = Depends(get_db)):
    """Mengambil detail lengkap satu merchant berdasarkan NMID."""
    merchant = db.query(Merchant).filter(Merchant.nmid == nmid).first()
    if not merchant:
        raise HTTPException(status_code=404, detail=f"Merchant NMID '{nmid}' tidak ditemukan.")
    return merchant


# ─────────────────────────────────────────────────────────────────────────────
# EMRS — Reputation Score Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/merchants/{nmid}/reputation", response_model=schemas.ReputationScoreSchema)
def get_merchant_reputation(nmid: str):
    """
    Kalkulasi dan return EMRS (Evidence-Based Merchant Reputation Score)
    untuk satu merchant.
    Components: T (Transaction Reliability), A (Authenticity),
                L (Longevity), C (Complaint), D (Dispute)
    """
    rep = get_merchant_reputation_by_nmid(nmid)
    if not rep.get("found_in_db"):
        raise HTTPException(status_code=404, detail=f"Merchant NMID '{nmid}' tidak ditemukan.")
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# Feedback Submission
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/feedback", response_model=schemas.FeedbackResponseSchema)
def submit_feedback(
    request: Request,
    payload: schemas.FeedbackSubmitSchema
):
    """
    Endpoint Feedback Legacy / Direct.
    Menerima feedback pengguna, menjalankan inferensi IndoBERT, dan memperbarui EMRS.
    """
    enforce_rate_limit(request, max_requests=20, window_seconds=60, endpoint_tag="feedback")
    nlp_res = classify_feedback(payload.description or "")
    cat_to_use = payload.category
    if not cat_to_use or cat_to_use in ["other", "General Complaint"]:
        cat_to_use = nlp_res["category_key"]

    result = submit_feedback_to_db(
        nmid=payload.nmid,
        category=cat_to_use,
        severity=payload.severity or nlp_res["severity"],
        description=payload.description or "",
        transaction_ref=payload.transaction_ref,
        has_evidence=payload.has_evidence
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    result["detected_category"] = nlp_res["category_title"]
    result["detected_category_key"] = nlp_res["category_key"]
    result["confidence"] = nlp_res["confidence"]
    result["severity"] = nlp_res["severity"]
    result["action"] = nlp_res["action"]

    return result


@app.post("/api/v1/feedback", response_model=schemas.FeedbackContractCResponseSchema)
def submit_feedback_contract_c(
    request: Request,
    payload: schemas.FeedbackContractCSchema,
    token: Optional[str] = Depends(auth.oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Contract C: Post-Payment User Feedback Ingestion Endpoint.
    Menerima ulasan naratif pengguna, menjalankan inferensi IndoBERT 7-kelas,
    menghitung Evidence Level secara otomatis (0, 1, 2), mengikat transaksi,
    memvalidasi user ownership (Step 29), dan memperbarui reputasi merchant (EMRS).
    """
    enforce_rate_limit(request, max_requests=20, window_seconds=60, endpoint_tag="feedback")

    if not payload.comment or not payload.comment.strip():
        raise HTTPException(status_code=400, detail="Komentar ulasan feedback tidak boleh kosong.")

    # 1. Jalankan Model IndoBERT 7-Kelas
    nlp_res = classify_feedback(payload.comment)
    event_type = nlp_res["category_key"]
    severity = nlp_res["severity"]
    confidence = nlp_res["confidence"]

    # 2. Tentukan Hubungan & Resolusi NMID serta Evidence Level
    target_nmid = payload.nmid
    evidence_level = 0
    resolved_session = None
    resolved_tx = None

    # A. Cek relasi Transaction
    if payload.transaction_id:
        resolved_tx = db.query(PaymentTransaction).filter(
            PaymentTransaction.provider_transaction_id == payload.transaction_id
        ).first()
        if resolved_tx:
            evidence_level = 2
            if not target_nmid:
                target_nmid = resolved_tx.nmid
            if resolved_tx.verification_session_id and not payload.verification_id:
                payload.verification_id = resolved_tx.verification_session_id

    # B. Cek relasi Verification Session
    if payload.verification_id:
        resolved_session = db.query(VerificationSession).filter(
            VerificationSession.session_id == payload.verification_id
        ).first()
        if resolved_session:
            if evidence_level < 1:
                evidence_level = 1
            if not target_nmid:
                target_nmid = resolved_session.nmid

    # 2.5 Enforcement User Ownership (Step 29 di Readme2.md)
    # User A tidak dapat mengirim feedback atas transaksi / sesi milik User B
    requester_user_id = None
    if token:
        user_claims = auth.decode_access_token(token)
        if user_claims:
            requester_user_id = user_claims.get("sub")
    if not requester_user_id and getattr(payload, "user_id", None):
        requester_user_id = payload.user_id

    if requester_user_id:
        if resolved_tx and resolved_tx.user_id:
            auth.verify_user_ownership(requester_user_id, resolved_tx.user_id, "transaksi pembayaran")
        if resolved_session and resolved_session.user_id:
            auth.verify_user_ownership(requester_user_id, resolved_session.user_id, "sesi verifikasi")

    # Jika bukti fisik diunggah dan sudah ada session
    if payload.has_evidence and evidence_level >= 1:
        evidence_level = 2

    if not target_nmid:
        raise HTTPException(
            status_code=400,
            detail="NMID toko tidak ditemukan. Mohon sertakan verification_id, transaction_id, atau nmid yang valid."
        )

    # 3. Transaksi DB & Kalkulasi EMRS
    tx_ref = payload.transaction_id or (f"VERIF-{payload.verification_id}" if payload.verification_id else None)

    result = submit_feedback_to_db(
        nmid=target_nmid,
        category=event_type,
        severity=severity,
        description=payload.comment,
        transaction_ref=tx_ref,
        has_evidence=payload.has_evidence,
        evidence_level=evidence_level
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    return schemas.FeedbackContractCResponseSchema(
        success=True,
        message="Feedback berhasil dianalisis oleh AI dan reputasi toko telah diperbarui.",
        event_type=event_type,
        category_title=nlp_res["category_title"],
        severity=severity,
        evidence_level=evidence_level,
        confidence=confidence,
        merchant_reputation_updated=True,
        previous_reputation_score=result.get("previous_reputation_score"),
        new_reputation_score=result["new_reputation_score"],
        recommended_action=nlp_res["action"],
        model_used=nlp_res.get("model_used", "IndoBERT-FineTuned-EMRS"),
        verification_id=payload.verification_id,
        transaction_id=payload.transaction_id
    )


# ─────────────────────────────────────────────────────────────────────────────
# Disputes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/merchants/{nmid}/disputes", response_model=List[schemas.DisputeSchema])
def list_merchant_disputes(nmid: str, db: Session = Depends(get_db)):
    """Mengambil daftar dispute terverifikasi untuk satu merchant."""
    merchant = db.query(Merchant).filter(Merchant.nmid == nmid).first()
    if not merchant:
        raise HTTPException(status_code=404, detail=f"Merchant NMID '{nmid}' tidak ditemukan.")
    return db.query(Dispute).filter(Dispute.merchant_id == merchant.id).all()


@app.post("/api/merchants/{nmid}/disputes")
def submit_dispute(nmid: str, description: str = Form(...), evidence_ref: str = Form(None),
                   severity: str = Form("HIGH"), db: Session = Depends(get_db)):
    """
    Submit sengketa transaksi yang terverifikasi.
    Dispute memiliki bobot lebih besar dari Complaint biasa di EMRS.
    """
    merchant = db.query(Merchant).filter(Merchant.nmid == nmid).first()
    if not merchant:
        raise HTTPException(status_code=404, detail=f"Merchant NMID '{nmid}' tidak ditemukan.")

    disp = Dispute(
        merchant_id=merchant.id,
        description=description,
        evidence_ref=evidence_ref,
        severity=severity,
        is_verified=False  # Admin yang verifikasi
    )
    db.add(disp)
    db.commit()
    return {"message": "Dispute berhasil disubmit. Akan diverifikasi oleh tim LaQris.", "dispute_id": disp.id}


# ─────────────────────────────────────────────────────────────────────────────
# Database Management
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/seed")
def seed_database_endpoint(
    request: Request,
    x_admin_secret: Optional[str] = Header(None, alias="X-Admin-Secret")
):
    """
    Reset & re-seed seluruh data reputasi SQLite (termasuk skema EMRS baru).
    Step 31 di Readme2.md: Dilindungi pada environment production / non-dev.
    """
    env_mode = os.getenv("LAQRIS_ENV", "development").lower()
    admin_secret = os.getenv("LAQRIS_ADMIN_SECRET", "admin-secret-laqris-2026")

    if env_mode == "production":
        if not x_admin_secret or x_admin_secret != admin_secret:
            raise HTTPException(
                status_code=403,
                detail="Akses ditolak: Endpoint seed dinonaktifkan di mode production atau memerlukan otentikasi admin (X-Admin-Secret)."
            )

    reset_db()
    return {"message": "Database SQLite berhasil di-reset dan di-seed ulang dengan skema EMRS v2!"}


@app.get("/api/scans/history")
def get_recent_scans_history(user_id: Optional[str] = None, limit: int = 10, db: Session = Depends(get_db)):
    """Mengambil riwayat scan & transaksi QRIS teraktual dari database SQLite (filter per user_id)."""
    query = db.query(VerificationSession)
    if user_id:
        query = query.filter((VerificationSession.user_id == user_id) | (VerificationSession.user_id.is_(None)))
    
    sessions = (
        query.order_by(VerificationSession.scanned_at.desc())
        .limit(limit)
        .all()
    )
    
    results = []
    now = datetime.utcnow()
    for s in sessions:
        merchant_name = s.digital_name or s.physical_name or "Merchant QRIS"
        
        # Ambil transaksi pembayaran yang terhubung
        tx = s.payment_transaction
        if not tx and s.session_id:
            tx = db.query(PaymentTransaction).filter(PaymentTransaction.verification_session_id == s.session_id).first()

        amount_val = tx.amount if (tx and tx.amount) else None
        is_paid = bool(tx and tx.status == "SUCCESS")

        # Format waktu relatif (Hari ini, Kemarin, dll.)
        time_str = "Hari ini"
        if s.scanned_at:
            delta_days = (now.date() - s.scanned_at.date()).days
            if delta_days == 0:
                time_str = f"Hari ini, {s.scanned_at.strftime('%H:%M')}"
            elif delta_days == 1:
                time_str = f"Kemarin, {s.scanned_at.strftime('%H:%M')}"
            else:
                time_str = s.scanned_at.strftime("%d %b, %H:%M")

        results.append({
            "session_id": s.session_id,
            "merchant_name": merchant_name,
            "nmid": s.nmid or "ID-PENDING",
            "status": s.status,
            "risk_level": s.risk_level,
            "trust_score": s.trust_score,
            "scanned_at": time_str,
            "raw_scanned_at": s.scanned_at.isoformat() if s.scanned_at else None,
            "amount": amount_val,
            "is_paid": is_paid,
            "payment_status": tx.status if tx else None,
            "transaction_id": tx.provider_transaction_id if tx else None,
            "invoice_number": tx.invoice_number if tx else None
        })
    return {"scans": results}


@app.get("/api/scan/session/{session_id}")
def get_scan_session_details_endpoint(session_id: str):
    """
    Mengambil seluruh detail arsip sesi scan tertentu (JSON data prediksi & list crops).
    """
    data = get_session_archive_data(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Sesi scan tidak ditemukan.")
    return data


@app.get("/api/scan/sessions")
def list_scan_sessions_endpoint(limit: int = 20):
    """
    Mengambil daftar seluruh folder sesi scan yang tersimpan di static/sessions.
    """
    return {"sessions": list_all_scan_sessions(limit=limit)}


# ─────────────────────────────────────────────────────────────────────────────
# Pre-Payment Verification API (Contract A: Provider API Ready)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/verify", response_model=schemas.VerifyResponseSchema)
def verify_pre_payment(
    request: Request,
    payload: schemas.VerifyRequestSchema,
    db: Session = Depends(get_db)
):
    """
    Contract A: Pre-payment Verification Endpoint untuk Payment Provider (DemoPay / Gateway).
    Memeriksa integritas QRIS dan mengembalikan keputusan ALLOW / WARN / BLOCK
    sebelum otorisasi transaksi dilakukan di payment server.
    """
    enforce_rate_limit(request, max_requests=30, window_seconds=60, endpoint_tag="scan-verify")
    v_id = payload.session_id or payload.verification_id
    if not v_id:
        raise HTTPException(status_code=400, detail="Parameter session_id atau verification_id diperlukan.")

    session = db.query(VerificationSession).filter(VerificationSession.session_id == v_id).first()
    if not session:
        raise HTTPException(status_code=404, detail=f"Sesi verifikasi '{v_id}' tidak ditemukan.")

    now = datetime.utcnow()
    if session.expires_at and now > session.expires_at:
        raise HTTPException(status_code=400, detail="Sesi verifikasi telah kedaluwarsa. Silakan scan ulang QRIS.")

    if session.is_bound:
        raise HTTPException(
            status_code=409,
            detail="Sesi verifikasi telah digunakan untuk pembayaran sebelumnya (anti-replay)."
        )

    # Parse reason codes dari JSON text
    parsed_reasons = []
    if session.reason_codes:
        try:
            import json
            parsed_reasons = json.loads(session.reason_codes)
        except Exception:
            parsed_reasons = []

    # Map decision policy: NORMAL -> ALLOW, CAUTION/WARNING -> WARN, DANGER -> BLOCK
    decision = session.decision or "ALLOW"
    can_proceed = (decision in ["ALLOW", "WARN"])

    merchant_info = schemas.VerifyMerchantInfoSchema(
        digital_name=session.digital_name,
        physical_name=session.physical_name,
        nmid=session.nmid
    )

    return schemas.VerifyResponseSchema(
        verification_id=session.session_id,
        decision=decision,
        risk_level=session.risk_level or "NORMAL",
        risk_score=round(100.0 - (session.trust_score or 100.0), 1),
        trust_score=session.trust_score or 100.0,
        reason_codes=parsed_reasons,
        merchant_info=merchant_info,
        can_proceed_payment=can_proceed,
        is_bound=session.is_bound or False,
        expires_at=session.expires_at,
        message="Verifikasi pre-payment berhasil. Transaksi dapat dilanjutkan." if can_proceed else "Transaksi diblokir demi keamanan nasabah."
    )


@app.get("/api/v1/verify/{session_id}", response_model=schemas.VerifyResponseSchema)
def get_verification_session_status(request: Request, session_id: str, db: Session = Depends(get_db)):
    """Mengecek status dan masa berlaku sesi verifikasi QRIS."""
    return verify_pre_payment(request, schemas.VerifyRequestSchema(session_id=session_id), db=db)


# ─────────────────────────────────────────────────────────────────────────────
# Post-Payment Transaction Events API (Contract B & DB Transaction Layer)
# ─────────────────────────────────────────────────────────────────────────────

# Blacklist field sensitif pembayaran (Step 15 di Readme: Sanitization)
SENSITIVE_PAYMENT_FIELDS = {
    "card_number", "pan", "full_pan", "cvv", "cvc", "pin", "password",
    "auth_token", "secret", "card_expiry"
}

def _sanitize_payment_payload(raw_dict: dict) -> dict:
    """Menghapus seluruh field sensitif pembayaran sesuai spesifikasi keamanan LaQris."""
    return {k: v for k, v in raw_dict.items() if k.lower() not in SENSITIVE_PAYMENT_FIELDS}


@app.post("/api/v1/transaction-events", response_model=schemas.PaymentTransactionResponseSchema)
@app.post("/api/v1/transactions/events", response_model=schemas.PaymentTransactionResponseSchema)
def record_transaction_event(
    request: Request,
    payload: schemas.PaymentTransactionCreateSchema,
    db: Session = Depends(get_db)
):
    """
    Contract B: Post-payment Transaction Event Ingestion Endpoint.
    Menerima notifikasi hasil transaksi dari Payment Provider (DemoPay),
    melakukan sanitasi data sensitif, mengikat (binding) sesi verifikasi
    secara atomik, dan memperbarui metrik reputasi merchant.
    """
    enforce_rate_limit(request, max_requests=60, window_seconds=60, endpoint_tag="transaction-events")
    v_id = payload.verification_id or payload.verification_session_id
    if not v_id:
        raise HTTPException(status_code=400, detail="verification_id diperlukan untuk binding transaksi.")

    # 1. Validasi Keberadaan Sesi Verifikasi
    session = db.query(VerificationSession).filter(VerificationSession.session_id == v_id).first()
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Verification session '{v_id}' tidak valid atau tidak ditemukan."
        )

    # 2. Enforcement Session Binding & Security Decision (Step 3 & 18 di Readme)
    if session.decision == "BLOCK" or session.risk_level == "DANGER":
        raise HTTPException(
            status_code=403,
            detail="Pembayaran ditolak: Sesi verifikasi ini memiliki status BLOCK / DANGER."
        )

    now = datetime.utcnow()
    if session.expires_at and now > session.expires_at:
        raise HTTPException(
            status_code=400,
            detail="Pembayaran ditolak: Sesi verifikasi telah kedaluwarsa (>15 menit)."
        )

    if session.is_bound:
        raise HTTPException(
            status_code=409,
            detail=f"Sesi verifikasi '{v_id}' sudah terikat pada transaksi pembayaran lain (anti-replay)."
        )

    # 3. Sanitasi Payload (Step 15 di Readme)
    sanitized_data = _sanitize_payment_payload(payload.model_dump())

    # User Ownership Enforcement (Step 29 di Readme2.md)
    if session.user_id and sanitized_data.get("user_id"):
        auth.verify_user_ownership(sanitized_data["user_id"], session.user_id, "sesi verifikasi")

    target_user_id = sanitized_data.get("user_id") or session.user_id or "USR-001928"
    if not session.user_id:
        session.user_id = target_user_id

    # Resolve Merchant
    lookup_nmid = sanitized_data.get("nmid") or session.nmid
    merchant_obj = None
    if sanitized_data.get("merchant_id"):
        try:
            m_id_int = int(sanitized_data["merchant_id"])
            merchant_obj = db.query(Merchant).filter(Merchant.id == m_id_int).first()
        except (ValueError, TypeError):
            pass
    if not merchant_obj and lookup_nmid:
        merchant_obj = db.query(Merchant).filter(Merchant.nmid == lookup_nmid).first()

    # 4. DB Transaction Layer (Atomic Session Binding & Metric Update)
    try:
        # A. Bind Sesi
        session.is_bound = True
        session.bound_at = now

        # B. Simpan Entitas PaymentTransaction
        tx = PaymentTransaction(
            verification_session_id=v_id,
            provider=sanitized_data.get("provider") or "DemoPay",
            provider_transaction_id=sanitized_data["provider_transaction_id"],
            merchant_id=merchant_obj.id if merchant_obj else None,
            nmid=lookup_nmid,
            user_id=target_user_id,
            amount=sanitized_data.get("amount", 0.0),
            status=sanitized_data.get("status", "SUCCESS"),
            response_code=sanitized_data.get("response_code", "00"),
            invoice_number=sanitized_data.get("invoice_number"),
            terminal_id=sanitized_data.get("terminal_id"),
            latency_ms=sanitized_data.get("latency_ms", 0),
            retry_count=sanitized_data.get("retry_count", 0),
            created_at=now
        )
        db.add(tx)

        # C. Update Metrik Merchant & EMRS
        rep_impact = None
        if merchant_obj:
            merchant_obj.verified_transactions = (merchant_obj.verified_transactions or 0) + 1
            if tx.status == "SUCCESS":
                merchant_obj.successful_transactions = (merchant_obj.successful_transactions or 0) + 1
            else:
                merchant_obj.failed_transactions = (merchant_obj.failed_transactions or 0) + 1

            # Hitung ulang EMRS score merchant
            reports = db.query(Report).filter(Report.merchant_id == merchant_obj.id).all()
            disputes = db.query(Dispute).filter(Dispute.merchant_id == merchant_obj.id).all()
            emrs_res = calculate_emrs(merchant_obj, reports, disputes)
            merchant_obj.reputation_score = emrs_res.get("reputation_score", merchant_obj.reputation_score)

            rep_impact = {
                "nmid": merchant_obj.nmid,
                "merchant_name": merchant_obj.merchant_name,
                "new_reputation_score": merchant_obj.reputation_score,
                "verified_transactions": merchant_obj.verified_transactions,
                "successful_transactions": merchant_obj.successful_transactions
            }

        db.commit()
        db.refresh(tx)

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mencatat transaksi pembayaran: {str(e)}"
        )

    return schemas.PaymentTransactionResponseSchema(
        success=True,
        message="Event transaksi pembayaran berhasil dicatat dan diikat ke sesi verifikasi.",
        transaction_id=tx.provider_transaction_id,
        verification_id=tx.verification_session_id or "",
        status=tx.status,
        amount=tx.amount,
        is_bound=True,
        merchant_reputation_impact=rep_impact
    )


@app.get("/api/v1/transactions/{verification_session_id}", response_model=schemas.PaymentTransactionResponseSchema)
def get_transaction_by_session(
    verification_session_id: str,
    db: Session = Depends(get_db)
):
    """Mengambil riwayat transaksi pembayaran berdasarkan verification_session_id."""
    tx = db.query(PaymentTransaction).filter(PaymentTransaction.verification_session_id == verification_session_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail=f"Transaksi untuk session '{verification_session_id}' tidak ditemukan.")
    return schemas.PaymentTransactionResponseSchema(
        success=True,
        message="Transaksi ditemukan.",
        transaction_id=tx.provider_transaction_id,
        verification_id=tx.verification_session_id or "",
        status=tx.status,
        amount=tx.amount,
        is_bound=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# User Authentication API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

import secrets
import random

def _hash_password(raw_password: str) -> str:
    """Argon2id hashing wrapper (Step 28 di Readme2.md)."""
    return auth.hash_password(raw_password)


def _generate_user_id() -> str:
    return f"USR-{random.randint(100000, 999999)}"


@app.post("/api/register")
def register_user(
    request: Request,
    payload: schemas.UserRegisterSchema,
    db: Session = Depends(get_db)
):
    """
    Registrasi pengguna / merchant baru ke database LaQris.
    Menggunakan hashing password Argon2id & access token JWT (Step 28).
    """
    enforce_rate_limit(request, max_requests=5, window_seconds=60, endpoint_tag="auth-register")

    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email ini sudah terdaftar. Silakan login.")

    user_id = _generate_user_id()
    while db.query(User).filter(User.user_id == user_id).first():
        user_id = _generate_user_id()

    acc_num = f"1858{random.randint(100000, 999999)}"
    user = User(
        user_id=user_id,
        username=payload.username,
        full_name=payload.full_name or payload.username,
        email=payload.email,
        phone=payload.phone,
        role=payload.role,
        status="ACTIVE",
        account_number=acc_num,
        account_type="TAPLUS",
        password_hash=auth.hash_password(payload.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Buat JWT Bearer Token standar RFC 7519
    token = auth.create_access_token({
        "sub": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": user.role
    })

    return {
        "message": "Registrasi berhasil!",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "status": user.status,
            "full_name": user.full_name,
            "role": user.role,
            "account_number": getattr(user, "account_number", acc_num),
            "account_type": getattr(user, "account_type", "TAPLUS"),
            "pin": getattr(user, "pin", "123456") or "123456"
        }
    }


@app.post("/api/login")
def login_user(
    request: Request,
    payload: schemas.UserLoginSchema,
    db: Session = Depends(get_db)
):
    """
    Autentikasi masuk pengguna dengan verifikasi password Argon2id & JWT token.
    Mendukung migrasi otomatis akun lama (SHA-256) ke Argon2id.
    """
    enforce_rate_limit(request, max_requests=10, window_seconds=60, endpoint_tag="auth-login")

    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not auth.verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email atau password yang Anda masukkan salah.")

    # Migrasi transparan: jika hash di DB masih legacy (SHA-256), rehash ke Argon2id
    if auth.is_legacy_hash(user.password_hash):
        user.password_hash = auth.hash_password(payload.password)
        db.commit()
        db.refresh(user)

    # Terbitkan JWT Access Token
    token = auth.create_access_token({
        "sub": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": user.role
    })

    return {
        "message": "Login berhasil!",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "status": user.status,
            "full_name": user.full_name,
            "phone": user.phone,
            "role": user.role,
            "account_number": getattr(user, "account_number", "1858868768") or "1858868768",
            "account_type": getattr(user, "account_type", "TAPLUS") or "TAPLUS",
            "pin": getattr(user, "pin", "123456") or "123456"
        }
    }


@app.get("/api/user/current")
def get_current_user_profile(
    user_id: Optional[str] = None,
    token: Optional[str] = Depends(auth.oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Mengambil data profil pengguna yang sedang aktif dari Bearer JWT atau query param."""
    user = None
    if token:
        claims = auth.decode_access_token(token)
        if claims and claims.get("sub"):
            user = db.query(User).filter(User.user_id == claims["sub"]).first()

    if not user and user_id:
        user = db.query(User).filter(User.user_id == user_id).first()

    if not user:
        user = db.query(User).order_by(User.id.desc()).first()

    if not user:
        return {
            "user_id": "USR-001928",
            "username": "yantoalim",
            "full_name": "Yanto Alim",
            "account_number": "1858868768",
            "account_type": "TAPLUS",
            "pin": "123456"
        }
    return {
        "id": user.id,
        "user_id": user.user_id,
        "username": user.username,
        "full_name": user.full_name,
        "email": user.email,
        "phone": user.phone,
        "role": user.role,
        "account_number": getattr(user, "account_number", "1858868768") or "1858868768",
        "account_type": getattr(user, "account_type", "TAPLUS") or "TAPLUS",
        "pin": getattr(user, "pin", "123456") or "123456"
    }


@app.post("/api/user/verify-pin")
def verify_user_pin(payload: dict, db: Session = Depends(get_db)):
    """Verifikasi PIN transaksi pengguna."""
    input_pin = str(payload.get("pin", "")).strip()
    user_id = payload.get("user_id")
    user = None
    if user_id:
        user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = db.query(User).order_by(User.id.desc()).first()
    correct_pin = getattr(user, "pin", "123456") if user else "123456"
    if not correct_pin:
        correct_pin = "123456"
    if input_pin == correct_pin:
        return {"valid": True, "message": "PIN benar."}
    return {"valid": False, "message": "PIN tidak sesuai. Silakan coba lagi."}


# ─────────────────────────────────────────────────────────────────────────────
# Frontend SPA Serving
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/index.html")
def serve_frontend_index():
    index_path = os.path.join(FOLDER_FRONTEND, "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache"
            }
        )
    return {"message": "LaQris API backend is running. Frontend index.html not found."}


@app.get("/dashboard")
@app.get("/dashboard.html")
def serve_frontend_dashboard():
    dash_path = os.path.join(FOLDER_FRONTEND, "dashboard.html")
    if os.path.exists(dash_path):
        return FileResponse(
            dash_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache"
            }
        )
    return {"message": "dashboard.html not found."}



@app.get("/scan")
@app.get("/scan.html")
def serve_frontend_scan():
    scan_path = os.path.join(FOLDER_FRONTEND, "scan.html")
    if os.path.exists(scan_path):
        return FileResponse(
            scan_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache"
            }
        )
    return {"message": "scan.html not found."}


@app.get("/select_scan")
@app.get("/select_scan.html")
def serve_frontend_select_scan():
    select_path = os.path.join(FOLDER_FRONTEND, "select_scan.html")
    if os.path.exists(select_path):
        return FileResponse(
            select_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache"
            }
        )
    return {"message": "select_scan.html not found."}


@app.get("/register")
@app.get("/register.html")
def serve_frontend_register():
    reg_path = os.path.join(FOLDER_FRONTEND, "register.html")
    if os.path.exists(reg_path):
        return FileResponse(
            reg_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache"
            }
        )
    return {"message": "register.html not found."}


@app.get("/login")
@app.get("/login.html")
def serve_frontend_login():
    login_path = os.path.join(FOLDER_FRONTEND, "login.html")
    if os.path.exists(login_path):
        return FileResponse(
            login_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache"
            }
        )
    return {"message": "login.html not found."}


@app.get("/payment")
@app.get("/payment.html")
def serve_frontend_payment():
    pay_path = os.path.join(FOLDER_FRONTEND, "payment.html")
    if os.path.exists(pay_path):
        return FileResponse(
            pay_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache"
            }
        )
    return {"message": "payment.html not found."}


@app.get("/feedback")
@app.get("/feedback.html")
def serve_frontend_feedback():
    fb_path = os.path.join(FOLDER_FRONTEND, "feedback.html")
    if os.path.exists(fb_path):
        return FileResponse(
            fb_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache"
            }
        )
    return {"message": "feedback.html not found."}


if os.path.exists(FOLDER_FRONTEND):
    app.mount("/app", StaticFiles(directory=FOLDER_FRONTEND, html=True), name="frontend")
    app.mount("/", StaticFiles(directory=FOLDER_FRONTEND, html=True), name="frontend_root")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
