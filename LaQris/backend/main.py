import os
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional

import hashlib
from database import init_db, get_db, reset_db
from models import Merchant, Report, Dispute, VerificationSession, User, PaymentTransaction
import schemas
from engine import (
    process_qris_verification,
    evaluasi_posisi_qris,
    submit_feedback_to_db,
    get_merchant_reputation_by_nmid,
    calculate_emrs,
    get_session_archive_data,
    list_all_scan_sessions
)

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

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
def submit_feedback(payload: schemas.FeedbackSubmitSchema):
    """
    Menerima feedback/complaint pengguna terhadap merchant.

    Evidence Level:
    - Level 1 (has_evidence=false): Laporan tanpa bukti → bobot 0.5x
    - Level 2 (has_evidence=true):  Laporan + bukti transaksi → bobot 1.0x

    Constraint: Satu transaction_ref hanya boleh submit satu feedback.
    Setelah submit, EMRS merchant otomatis di-recalculate.
    """
    result = submit_feedback_to_db(
        nmid=payload.nmid,
        category=payload.category,
        severity=payload.severity,
        description=payload.description,
        transaction_ref=payload.transaction_ref,
        has_evidence=payload.has_evidence
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


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
def seed_database_endpoint():
    """Reset & re-seed seluruh data reputasi SQLite (termasuk skema EMRS baru)."""
    reset_db()
    return {"message": "Database SQLite berhasil di-reset dan di-seed ulang dengan skema EMRS v2!"}


@app.get("/api/scans/history")
def get_recent_scans_history(user_id: Optional[str] = None, limit: int = 5, db: Session = Depends(get_db)):
    """Mengambil 5 riwayat scan QRIS teraktual dari database SQLite (filter per user_id)."""
    query = db.query(VerificationSession)
    if user_id:
        query = query.filter(VerificationSession.user_id == user_id)
    
    sessions = (
        query.order_by(VerificationSession.scanned_at.desc())
        .limit(limit)
        .all()
    )
    
    results = []
    for s in sessions:
        merchant_name = s.digital_name or s.physical_name or "Merchant QRIS"
        results.append({
            "session_id": s.session_id,
            "merchant_name": merchant_name,
            "nmid": s.nmid or "ID-PENDING",
            "status": s.status,
            "risk_level": s.risk_level,
            "trust_score": s.trust_score,
            "scanned_at": s.scanned_at.strftime("%d %b %Y %H:%M") if s.scanned_at else "Baru saja"
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
# Transaction Events API Endpoints (Lifecycle Tracking)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/transactions/events", response_model=schemas.PaymentTransactionResponseSchema)
def record_transaction_event(
    payload: schemas.PaymentTransactionCreateSchema,
    db: Session = Depends(get_db)
):
    """
    Mencatat event transaksi pembayaran yang terikat pada verification_session_id.
    Digunakan untuk tracing siklus hidup verifikasi QRIS -> pembayaran gateway.
    """
    session = db.query(VerificationSession).filter(VerificationSession.session_id == payload.verification_session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail=f"Verification session '{payload.verification_session_id}' tidak ditemukan.")

    tx = PaymentTransaction(
        verification_session_id=payload.verification_session_id,
        provider=payload.provider,
        provider_transaction_id=payload.provider_transaction_id,
        merchant_id=payload.merchant_id or session.merchant_id,
        nmid=payload.nmid or session.nmid,
        amount=payload.amount,
        status=payload.status,
        response_code=payload.response_code,
        invoice_number=payload.invoice_number,
        terminal_id=payload.terminal_id,
        latency_ms=payload.latency_ms,
        retry_count=payload.retry_count or 0
    )
    db.add(tx)
    db.commit()
    db.refresh(tx)
    return tx


@app.get("/api/v1/transactions/{verification_session_id}", response_model=schemas.PaymentTransactionResponseSchema)
def get_transaction_by_session(
    verification_session_id: str,
    db: Session = Depends(get_db)
):
    """Mengambil riwayat transaksi pembayaran berdasarkan verification_session_id."""
    tx = db.query(PaymentTransaction).filter(PaymentTransaction.verification_session_id == verification_session_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail=f"Transaksi untuk session '{verification_session_id}' tidak ditemukan.")
    return tx


# ─────────────────────────────────────────────────────────────────────────────
# User Authentication API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

import secrets
import random

def _hash_password(raw_password: str) -> str:
    return hashlib.sha256(raw_password.encode('utf-8')).hexdigest()


def _generate_user_id() -> str:
    return f"USR-{random.randint(100000, 999999)}"


@app.post("/api/register")
def register_user(payload: schemas.UserRegisterSchema, db: Session = Depends(get_db)):
    """Registrasi pengguna / merchant baru ke database LaQris."""
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email ini sudah terdaftar. Silakan login.")

    user_id = _generate_user_id()
    while db.query(User).filter(User.user_id == user_id).first():
        user_id = _generate_user_id()

    user = User(
        user_id=user_id,
        username=payload.username,
        full_name=payload.full_name or payload.username,
        email=payload.email,
        phone=payload.phone,
        role=payload.role,
        status="ACTIVE",
        password_hash=_hash_password(payload.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = f"laqris_token_{secrets.token_hex(16)}"

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
            "role": user.role
        }
    }


@app.post("/api/login")
def login_user(payload: schemas.UserLoginSchema, db: Session = Depends(get_db)):
    """Autentikasi masuk pengguna dengan email dan password."""
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or user.password_hash != _hash_password(payload.password):
        raise HTTPException(status_code=401, detail="Email atau password yang Anda masukkan salah.")

    token = f"laqris_token_{secrets.token_hex(16)}"

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
            "role": user.role
        }
    }


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


if os.path.exists(FOLDER_FRONTEND):
    app.mount("/app", StaticFiles(directory=FOLDER_FRONTEND, html=True), name="frontend")
    app.mount("/", StaticFiles(directory=FOLDER_FRONTEND, html=True), name="frontend_root")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
