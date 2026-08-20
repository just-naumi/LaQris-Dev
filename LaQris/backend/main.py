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

from database import init_db, get_db, reset_db
from models import Merchant, Report, Dispute, VerificationSession
import schemas
from engine import (
    process_qris_verification,
    submit_feedback_to_db,
    get_merchant_reputation_by_nmid,
    calculate_emrs
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

    hasil = process_qris_verification(gambar_input, filename_base=filename_base)
    return hasil


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


# ─────────────────────────────────────────────────────────────────────────────
# Frontend SPA Serving
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
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


if os.path.exists(FOLDER_FRONTEND):
    app.mount("/app", StaticFiles(directory=FOLDER_FRONTEND, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
