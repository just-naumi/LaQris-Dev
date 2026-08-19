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

from database import init_db, get_db
from models import Merchant, Report, VerificationSession
import schemas
from engine import process_qris_verification

# Auto-initialize SQLite database on startup
init_db()

app = FastAPI(
    title="LaQris POC Tahap 1 — QRIS Fraud Detection & Merchant Reputation API",
    description="Backend API untuk deteksi stiker QRIS ditimpa, matching identitas fisik vs digital, dan cek reputasi SQLite",
    version="1.0.0"
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

@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "LaQris POC Tahap 1", "database": "SQLite"}

@app.post("/api/scan")
async def scan_qris_endpoint(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
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

    # Eksekusi AI Pipeline (Dual YOLO + TrOCR + SQLite Reputation)
    hasil = process_qris_verification(gambar_input, filename_base=filename_base)

    # Simpan histori verifikasi ke SQLite
    session_rec = VerificationSession(
        session_id=hasil["session_id"],
        nmid=hasil["digital_nmid"],
        digital_name=hasil["digital_merchant"],
        physical_name=hasil["physical_merchant"],
        status="MISMATCH" if hasil["is_mismatch"] else "MATCH",
        risk_level=hasil["risk_level"]
    )
    db.add(session_rec)
    db.commit()

    return hasil


@app.get("/api/merchants", response_model=List[schemas.MerchantSchema])
def list_merchants(db: Session = Depends(get_db)):
    """
    Mengambil daftar seluruh merchant dan histori reputasinya di database SQLite.
    """
    merchants = db.query(Merchant).all()
    return merchants


@app.get("/api/merchants/{nmid}", response_model=schemas.MerchantSchema)
def get_merchant_detail(nmid: str, db: Session = Depends(get_db)):
    """
    Mengambil detail reputasi merchant berdasarkan NMID dari SQLite database.
    """
    merchant = db.query(Merchant).filter(Merchant.nmid == nmid).first()
    if not merchant:
        raise HTTPException(status_code=404, detail=f"Merchant dengan NMID '{nmid}' tidak ditemukan di database SQLite.")
    return merchant


@app.post("/api/seed")
def seed_database_endpoint():
    """
    Endpoint untuk reset & re-seed ulang data reputasi SQLite.
    """
    init_db()
    return {"message": "Database SQLite berhasil di-seed ulang!"}


# Mount Frontend Single-Page App
@app.get("/")
def serve_frontend_index():
    index_path = os.path.join(FOLDER_FRONTEND, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "LaQris API backend is running. Frontend index.html not found."}

if os.path.exists(FOLDER_FRONTEND):
    app.mount("/app", StaticFiles(directory=FOLDER_FRONTEND, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=False)
