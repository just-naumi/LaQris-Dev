import os
import cv2
import re
import math
import difflib
import uuid
import numpy as np
import torch
from PIL import Image
from pyzbar import pyzbar
import warnings
from datetime import datetime
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, RobertaTokenizer

from database import SessionLocal
from models import Merchant, Report, Dispute, VerificationSession

warnings.filterwarnings("ignore")

PERANGKAT = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_YOLO_BARCODE = None
MODEL_YOLO_OCR = None
PROCESSOR_TROCR = None
MODEL_TROCR = None

DAFTAR_NAMA_BANK = {
    "93600014": "BCA",
    "93600009": "BNI",
    "93600008": "MANDIRI",
    "93600002": "BRI",
    "93600114": "LINKAJA",
    "93600153": "SHOPEEPAY",
    "93600914": "GOPAY",
    "93600915": "DANA",
    "93600811": "OVO"
}

DAFTAR_MCC = {
    "5812": "Restoran / Rumah Makan",
    "5814": "Makanan Cepat Saji (Fast Food)",
    "5411": "Supermarket / Toko Kelontong",
    "5311": "Department Store / Toko Serba Ada",
    "5912": "Apotek / Farmasi",
    "5999": "Toko Retail / Perdagangan Umum",
    "4111": "Transportasi & Tiket",
    "5541": "SPBU / Bahan Bakar",
    "7299": "Jasa Layanan Konsumen",
    "8299": "Pendidikan & Kursus",
    "8699": "Organisasi Sosial / Komunitas",
    "7999": "Hiburan & Rekreasi"
}

PEMETAAN_LABEL_ROBOFLOW = {
    "nama merchant": "nama_merchant",
    "national merchant id": "nmid",
    "dicetak oleh": "acquirer",
    "terminal id": "tid",
    "qr code": "qrcode",
    "nama_merchant": "nama_merchant",
    "national_merchant_id": "nmid",
    "dicetak_oleh": "acquirer",
    "terminal_id": "tid",
    "qr_code": "qrcode"
}

DAFTAR_WARNA_LABEL = [
    (255, 99, 71),    # 0: Cara Pakai QRIS
    (255, 165, 0),    # 1: Cek Aplikasi Penyelenggara
    (30, 144, 255),   # 2: Dicetak Oleh
    (147, 112, 219),  # 3: Logo GPN
    (50, 205, 50),    # 4: Logo dan deskripsi QRIS
    (0, 215, 255),    # 5: Nama Merchant
    (238, 130, 238),  # 6: National Merchant ID
    (0, 0, 255),      # 7: QR Code
    (255, 105, 180),  # 8: Slogan
    (128, 128, 0),    # 9: Terminal ID
    (0, 255, 255)     # 10: Versi Cetak
]

# ─── Severity penalty mapping untuk Complaint Score (C) ──────────────────────
SEVERITY_PENALTY = {
    "LOW": 2,
    "MEDIUM": 5,
    "HIGH": 10,
    "CRITICAL": 20
}

# ─── Dispute penalty mapping untuk Dispute Score (D) ─────────────────────────
DISPUTE_PENALTY = {
    "verified": 30,
    "unverified": 10
}


# ═════════════════════════════════════════════════════════════════════════════
# AI Model Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_ai_models():
    global MODEL_YOLO_BARCODE, MODEL_YOLO_OCR, PROCESSOR_TROCR, MODEL_TROCR

    folder_backend = os.path.dirname(os.path.abspath(__file__))
    path_barcode = os.path.join(folder_backend, "weights", "yolo_barcode.pt")
    path_ocr = os.path.join(folder_backend, "weights", "yolo_ocr.pt")

    if MODEL_YOLO_BARCODE is None:
        print("[LOG] Memuat Model 1: YOLO Barcode dari", path_barcode)
        MODEL_YOLO_BARCODE = YOLO(path_barcode)

    if MODEL_YOLO_OCR is None:
        print("[LOG] Memuat Model 2: YOLO OCR dari", path_ocr)
        MODEL_YOLO_OCR = YOLO(path_ocr)

    if PROCESSOR_TROCR is None or MODEL_TROCR is None:
        nama_trocr = "microsoft/trocr-base-printed"
        print(f"[LOG] Memuat Model TrOCR ({nama_trocr})...")
        try:
            tokenizer = RobertaTokenizer.from_pretrained(nama_trocr)
            image_processor = ViTImageProcessor.from_pretrained(nama_trocr)
            PROCESSOR_TROCR = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            MODEL_TROCR = VisionEncoderDecoderModel.from_pretrained(nama_trocr).to(PERANGKAT)
            MODEL_TROCR.eval()
        except Exception:
            nama_trocr = "microsoft/trocr-base-stage1"
            tokenizer = RobertaTokenizer.from_pretrained(nama_trocr)
            image_processor = ViTImageProcessor.from_pretrained(nama_trocr)
            PROCESSOR_TROCR = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            MODEL_TROCR = VisionEncoderDecoderModel.from_pretrained(nama_trocr).to(PERANGKAT)
            MODEL_TROCR.eval()

    return MODEL_YOLO_BARCODE, MODEL_YOLO_OCR, PROCESSOR_TROCR, MODEL_TROCR


# ═════════════════════════════════════════════════════════════════════════════
# Text Normalization
# ═════════════════════════════════════════════════════════════════════════════

def normalisasi_teks(teks_mentah):
    if not teks_mentah or teks_mentah in ["Tidak terbaca", "Tidak ditemukan"]:
        return ""
    teks_kapital = teks_mentah.upper()
    teks_bersih = re.sub(r'[^A-Z0-9\s]', ' ', teks_kapital)
    daftar_kata = teks_bersih.split()
    sebutan = ["PT", "CV", "UD", "TB", "PD", "PERSERO"]
    daftar_kata_murni = [k for k in daftar_kata if k not in sebutan]
    hasil = " ".join(daftar_kata_murni).strip()
    return hasil if hasil != "" else " ".join(daftar_kata).strip()


# ═════════════════════════════════════════════════════════════════════════════
# QR Code Decoding & Deep EMVCo Analysis
# ═════════════════════════════════════════════════════════════════════════════

def scan_qr_code_digital(gambar_input):
    hasil_scan = pyzbar.decode(gambar_input)
    if not hasil_scan:
        return None
    return hasil_scan[0].data.decode('utf-8')


def parse_emvco_qr_deep_analysis(teks_qr_mentah):
    """
    Analisis Mendalam Struktur Payload EMVCo QRIS:
    Mengekstrak informasi spesifikasi teknis dari tag EMVCo resmi:
    - Tag 01: Point of Initiation Method ("11" = Statis/Stiker, "12" = Dinamis/EDC)
    - Tag 52: Merchant Category Code (MCC)
    - Tag 53: Transaction Currency ("360" = IDR)
    - Tag 51/26: Merchant Account & NMID Structure
    - Tag 63: CRC16 Checksum
    - Estimasi Tahun Registrasi dari NMID
    """
    if not teks_qr_mentah:
        return None

    initiation_code = "11"
    initiation_label = "Statis (Stiker Meja/Kasir)"
    mcc_code = "5999"
    mcc_category = "Perdagangan Umum / Retail"
    currency = "360 (IDR)"
    crc_checksum = None

    indeks, total = 0, len(teks_qr_mentah)
    while indeks < total:
        kode_tag = teks_qr_mentah[indeks: indeks + 2]
        panjang_str = teks_qr_mentah[indeks + 2: indeks + 4]
        if not panjang_str.isdigit():
            break
        size = int(panjang_str)
        isi = teks_qr_mentah[indeks + 4: indeks + 4 + size]

        if kode_tag == "01":
            initiation_code = isi
            initiation_label = "Dinamis (EDC/Layar Digital)" if isi == "12" else "Statis (Stiker Meja/Kasir)"
        elif kode_tag == "52":
            mcc_code = isi
            mcc_category = DAFTAR_MCC.get(isi, f"Kategori MCC ({isi})")
        elif kode_tag == "53":
            currency = f"{isi} (IDR)" if isi == "360" else isi
        elif kode_tag == "63":
            crc_checksum = isi
        indeks += 4 + size

    # Parse NMID structure for estimated year & country
    nmid_match = re.search(r'ID(\d{2})(\d{2})?\d+', teks_qr_mentah)
    est_year = None
    if nmid_match:
        # Coba ekstrak digit indikator tahun registrasi jika sesuai rentang 2019-2026
        d1 = nmid_match.group(1)
        d2 = nmid_match.group(2)
        if d2 and 19 <= int(d2) <= 26:
            est_year = 2000 + int(d2)
        elif d1 and 19 <= int(d1) <= 26:
            est_year = 2000 + int(d1)

    return {
        "point_of_initiation": initiation_label,
        "initiation_type_code": initiation_code,
        "mcc_code": mcc_code,
        "mcc_category": mcc_category,
        "currency": currency,
        "crc_checksum": crc_checksum,
        "nmid_parsed": {
            "country": "Indonesia (ID)",
            "estimated_reg_year": est_year or 2023,
            "specification": "ASPI National QRIS Specification"
        }
    }


def validate_and_parse_emvco_qr(teks_qr_mentah):
    if not teks_qr_mentah:
        return {"status": "INVALID_QR", "is_valid": False}, "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan", None

    nama_dig = "Tidak ditemukan"
    kota_dig = "Tidak ditemukan"
    nmid_dig = "Tidak ditemukan"
    acq_dig = "Tidak ditemukan"
    tid_dig = "Tidak ditemukan"

    indeks, total = 0, len(teks_qr_mentah)
    has_format_indicator = False

    while indeks < total:
        kode_tag = teks_qr_mentah[indeks: indeks + 2]
        panjang_str = teks_qr_mentah[indeks + 2: indeks + 4]
        if not panjang_str.isdigit():
            break
        size = int(panjang_str)
        isi = teks_qr_mentah[indeks + 4: indeks + 4 + size]

        if kode_tag == "00":
            has_format_indicator = True
        elif kode_tag == "59":
            nama_dig = isi
        elif kode_tag == "60":
            kota_dig = isi
        elif kode_tag == "51":
            sub_idx = 0
            while sub_idx < len(isi):
                sub_tag = isi[sub_idx: sub_idx + 2]
                sub_len_str = isi[sub_idx + 2: sub_idx + 4]
                if not sub_len_str.isdigit():
                    break
                sub_len = int(sub_len_str)
                sub_isi = isi[sub_idx + 4: sub_idx + 4 + sub_len]
                if sub_tag == "02" and sub_isi.startswith("ID"):
                    nmid_dig = sub_isi
                sub_idx += 4 + sub_len
        elif kode_tag == "62":
            sub_idx = 0
            while sub_idx < len(isi):
                sub_tag = isi[sub_idx: sub_idx + 2]
                sub_len_str = isi[sub_idx + 2: sub_idx + 4]
                if not sub_len_str.isdigit():
                    break
                sub_len = int(sub_len_str)
                sub_isi = isi[sub_idx + 4: sub_idx + 4 + sub_len]
                if sub_tag == "07":
                    tid_dig = sub_isi
                    break
                sub_idx += 4 + sub_len
        indeks += 4 + size

    if nmid_dig == "Tidak ditemukan":
        m = re.search(r'ID\d{13}', teks_qr_mentah)
        if m:
            nmid_dig = m.group()

    m_acq = re.search(r'9360\d{4}', teks_qr_mentah)
    if m_acq:
        kode_bank = m_acq.group()
        nama_bank = DAFTAR_NAMA_BANK.get(kode_bank, "BANK LAIN")
        acq_dig = f"{kode_bank} ({nama_bank})"

    payload_status = "VALID_QR_PAYLOAD" if has_format_indicator and nama_dig != "Tidak ditemukan" else "INVALID_STRUCTURE"
    tech_info = {"status": payload_status, "is_valid": (payload_status == "VALID_QR_PAYLOAD")}

    qris_analysis = parse_emvco_qr_deep_analysis(teks_qr_mentah)

    return tech_info, nama_dig, kota_dig, nmid_dig, acq_dig, tid_dig, qris_analysis


# ═════════════════════════════════════════════════════════════════════════════
# TrOCR Inference (Optimized Speed)
# ═════════════════════════════════════════════════════════════════════════════

def ocr_trocr(gambar_potongan, processor, model):
    if gambar_potongan is None or gambar_potongan.size == 0:
        return ""
    gambar_rgb = cv2.cvtColor(gambar_potongan, cv2.COLOR_BGR2RGB)
    gambar_pil = Image.fromarray(gambar_rgb)
    piksel = processor(gambar_pil, return_tensors="pt").pixel_values.to(PERANGKAT)
    with torch.no_grad():
        tokens = model.generate(piksel, max_new_tokens=64)
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


# ═════════════════════════════════════════════════════════════════════════════
# Identity Matching (Current QR Risk Component)
# ═════════════════════════════════════════════════════════════════════════════

def calculate_identity_similarity(phys_name, dig_name):
    if phys_name in ["Tidak terbaca", ""] or dig_name in ["Tidak ditemukan", ""]:
        return 0.0, "COMPLETELY_DIFFERENT", 100.0

    p_raw = phys_name.lower().strip()
    d_raw = dig_name.lower().strip()
    p_norm = normalisasi_teks(phys_name).lower()
    d_norm = normalisasi_teks(dig_name).lower()

    if p_raw == d_raw:
        sim = 100.0
        level = "EXACT_MATCH"
    elif p_norm == d_norm and p_norm != "":
        sim = 100.0
        level = "NORMALIZED_MATCH"
    else:
        r1 = difflib.SequenceMatcher(None, p_raw, d_raw).ratio() * 100
        r2 = difflib.SequenceMatcher(None, p_norm, d_norm).ratio() * 100
        sim = round(max(r1, r2), 1)
        if sim >= 70.0:
            level = "PROBABLE_MATCH"
        elif sim >= 40.0:
            level = "UNCERTAIN"
        else:
            level = "COMPLETELY_DIFFERENT"

    if sim >= 90.0:
        identity_risk = 0.0
    elif sim >= 70.0:
        identity_risk = 30.0
    elif sim >= 40.0:
        identity_risk = 60.0
    else:
        identity_risk = 95.0

    return sim, level, identity_risk


# ═════════════════════════════════════════════════════════════════════════════
# EMRS — Evidence-Based Merchant Reputation Score Engine (REVISED FORMULA)
# ═════════════════════════════════════════════════════════════════════════════

def time_decay_weight(created_at: datetime) -> float:
    now = datetime.utcnow()
    months_ago = max(0, (now - created_at).days / 30.0)
    return math.exp(-0.1 * months_ago)


def calculate_emrs(merchant: Merchant, reports: list, disputes: list) -> dict:
    """
    REVISED EMRS FORMULA & CONFIDENCE MODEL:
    R = 0.40·A + 0.30·C + 0.20·D + 0.10·L

    A — Authenticity / Identity Consistency (40%)
    C — Complaint Score (30%)
    D — Dispute Score (20%)
    L — LaQris Observed Longevity & History (10%)
    T — LaQris Observed Transaction Reliability (Secondary/Optional Metric)

    Confidence Level:
    - LOW (< 5 evidence items / Insufficient history)
    - MEDIUM (5–19 evidence items)
    - HIGH (>= 20 evidence items)
    """
    now = datetime.utcnow()

    # ── A: Authenticity / Identity Consistency (40%) ──────────────────────────
    total_identity = merchant.identity_match_count + merchant.identity_mismatch_count
    if total_identity > 0:
        A_raw = (merchant.identity_match_count / total_identity) * 100.0
        A = max(0.0, A_raw - (merchant.critical_mismatch_count * 15.0))
    else:
        A = 50.0  # Netral

    # ── C: Complaint Score (30%) ──────────────────────────────────────────────
    C = 100.0
    for rpt in reports:
        sev_penalty = SEVERITY_PENALTY.get(rpt.severity, 5)
        ev_weight = 1.0 if rpt.evidence_level >= 2 else 0.5
        td = time_decay_weight(rpt.created_at)
        C -= sev_penalty * ev_weight * td
    C = max(0.0, min(100.0, C))

    # ── D: Dispute Score (20%) ────────────────────────────────────────────────
    D = 100.0
    for disp in disputes:
        penalty = DISPUTE_PENALTY["verified"] if disp.is_verified else DISPUTE_PENALTY["unverified"]
        td = time_decay_weight(disp.created_at)
        D -= penalty * td
    D = max(0.0, min(100.0, D))

    # ── L: LaQris Observed Longevity & History (10%) ──────────────────────────
    reg_at = merchant.registered_at or now
    days_active = (now - reg_at).days
    months_active = days_active / 30.0

    if months_active < 1:
        L = 40.0
    elif months_active < 6:
        L = 60.0 + (months_active / 6.0) * 15.0
    elif months_active < 24:
        L = 75.0 + ((months_active - 6) / 18.0) * 15.0
    else:
        L = min(100.0, 90.0 + ((months_active - 24) / 24.0) * 10.0)

    # ── Optional: LaQris Observed Transaction Reliability ─────────────────────
    T_observed = None
    if merchant.verified_transactions > 0:
        T_observed = round((merchant.successful_transactions / merchant.verified_transactions) * 100.0, 1)

    # ── Final EMRS Revised Formula ────────────────────────────────────────────
    R = round(
        (0.40 * A) +
        (0.30 * C) +
        (0.20 * D) +
        (0.10 * L),
        1
    )
    R = max(0.0, min(100.0, R))

    # ── Evidence Count & Confidence Level ─────────────────────────────────────
    total_evidence = total_identity + merchant.verified_transactions + len(reports) + len(disputes)
    confidence_score = min(100.0, round((total_evidence / 20.0) * 100.0, 1))

    if total_evidence >= 20:
        confidence_level = "HIGH"
        evidence_quality = "HIGH"
    elif total_evidence >= 5:
        confidence_level = "MEDIUM"
        evidence_quality = "MEDIUM"
    else:
        confidence_level = "LOW"
        evidence_quality = "LOW"

    data_sufficiency = "SUFFICIENT DATA" if total_evidence >= 3 else "INSUFFICIENT HISTORY"

    # ── Grade Assignment ──────────────────────────────────────────────────────
    if R >= 85:
        grade = "Excellent"
    elif R >= 70:
        grade = "Very Good"
    elif R >= 55:
        grade = "Good"
    elif R >= 40:
        grade = "Fair"
    else:
        grade = "Poor"

    return {
        "reputation_score": R,
        "grade": grade,
        "confidence_level": confidence_level,
        "confidence_score": confidence_score,
        "data_sufficiency_status": data_sufficiency,
        "components": {
            "A": round(A, 1),
            "C": round(C, 1),
            "D": round(D, 1),
            "L": round(L, 1),
            "T_observed": T_observed
        },
        "evidence_quality": evidence_quality,
        "total_evidence_count": total_evidence,
        "found_in_db": True,
        "nmid": merchant.nmid,
        "merchant_name": merchant.merchant_name,
        "registered_at": merchant.registered_at.isoformat() if merchant.registered_at else None,
        "first_seen_observed": merchant.registered_at.strftime("%B %Y") if merchant.registered_at else "Agustus 2026",
        "last_seen_observed": datetime.utcnow().strftime("%B %Y")
    }


def query_merchant_reputation(nmid_digital, nmid_physical, merchant_name_dig, merchant_name_phys) -> dict:
    db = SessionLocal()
    try:
        m = None
        if nmid_digital and nmid_digital != "Tidak ditemukan":
            m = db.query(Merchant).filter(Merchant.nmid == nmid_digital).first()
        if not m and nmid_physical and nmid_physical != "Tidak terbaca":
            m = db.query(Merchant).filter(Merchant.nmid == nmid_physical).first()
        if not m and merchant_name_dig:
            m = db.query(Merchant).filter(Merchant.merchant_name.ilike(f"%{merchant_name_dig}%")).first()
        if not m and merchant_name_phys:
            m = db.query(Merchant).filter(Merchant.merchant_name.ilike(f"%{merchant_name_phys}%")).first()

        if m:
            reports = db.query(Report).filter(Report.merchant_id == m.id).all()
            disputes = db.query(Dispute).filter(Dispute.merchant_id == m.id).all()
            return calculate_emrs(m, reports, disputes)
        else:
            return {
                "reputation_score": None,
                "grade": "Belum Terdaftar",
                "confidence_level": "LOW",
                "confidence_score": 0.0,
                "data_sufficiency_status": "INSUFFICIENT HISTORY",
                "components": {"A": 0.0, "C": 0.0, "D": 0.0, "L": 0.0, "T_observed": None},
                "evidence_quality": "INSUFFICIENT",
                "total_evidence_count": 0,
                "found_in_db": False,
                "nmid": nmid_digital if nmid_digital != "Tidak ditemukan" else nmid_physical,
                "merchant_name": merchant_name_dig if merchant_name_dig != "Tidak ditemukan" else merchant_name_phys,
                "registered_at": None,
                "first_seen_observed": None,
                "last_seen_observed": None
            }
    finally:
        db.close()


# ═════════════════════════════════════════════════════════════════════════════
# Full Integrated Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def process_qris_verification(gambar_input, filename_base="scan"):
    model_barcode, model_ocr, proc_trocr, model_trocr = load_ai_models()
    session_id = str(uuid.uuid4())[:8]

    res_barcode = model_barcode.predict(gambar_input, conf=0.25, verbose=False)[0]
    res_ocr = model_ocr.predict(gambar_input, conf=0.10, verbose=False)[0]

    gambar_vis = gambar_input.copy()
    tinggi_foto, lebar_foto = gambar_input.shape[:2]

    # Barcode extraction
    teks_qr_mentah = scan_qr_code_digital(gambar_input)
    if not teks_qr_mentah:
        for box in res_barcode.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                potongan = gambar_input[max(0, y1):min(tinggi_foto, y2), max(0, x1):min(lebar_foto, x2)]
                teks_qr_mentah = scan_qr_code_digital(potongan)
                if teks_qr_mentah:
                    break

    # Parse EMVCo + deep analysis
    tech_info, dig_name, dig_city, dig_nmid, dig_acq, dig_tid, qris_analysis = validate_and_parse_emvco_qr(teks_qr_mentah)

    # OCR extraction
    phys_name, phys_nmid, phys_acq, phys_tid = "", "", "", ""
    for box in res_ocr.boxes:
        cls_id = int(box.cls[0].item())
        nama_kelas = model_ocr.names[cls_id]
        label_std = PEMETAAN_LABEL_ROBOFLOW.get(nama_kelas.lower().strip(), nama_kelas.lower().strip())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        potongan = gambar_input[max(0, y1):min(tinggi_foto, y2), max(0, x1):min(lebar_foto, x2)]
        teks_ocr = ocr_trocr(potongan, proc_trocr, model_trocr)

        if label_std == "nama_merchant" and not phys_name:
            phys_name = teks_ocr
        elif label_std == "nmid" and not phys_nmid:
            phys_nmid = re.sub(r'^(NMID\s*:?\s*)', '', teks_ocr, flags=re.IGNORECASE).strip()
        elif label_std == "acquirer" and not phys_acq:
            phys_acq = teks_ocr
        elif label_std == "tid" and not phys_tid:
            phys_tid = teks_ocr

        warna = DAFTAR_WARNA_LABEL[cls_id % len(DAFTAR_WARNA_LABEL)]
        cv2.rectangle(gambar_vis, (x1, y1), (x2, y2), warna, 2)
        cv2.putText(gambar_vis, f"{label_std}: {teks_ocr}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, warna, 1)

    # Save visualization
    folder_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "vis_output")
    os.makedirs(folder_static, exist_ok=True)
    path_vis = os.path.join(folder_static, f"vis_{filename_base}.jpg")
    cv2.imwrite(path_vis, gambar_vis)

    # Identity matching & QR risk
    name_similarity, match_level, identity_risk = calculate_identity_similarity(phys_name, dig_name)
    is_mismatch = (match_level == "COMPLETELY_DIFFERENT" or name_similarity < 40.0)

    technical_risk = 0.0 if tech_info.get("is_valid") else 80.0
    current_qr_risk_score = round((0.70 * identity_risk) + (0.30 * technical_risk), 1)
    current_trust_score = round(100.0 - current_qr_risk_score, 1)

    if is_mismatch:
        risk_level = "HIGH_RISK"
        explanation = f"PERINGATAN: Identitas stiker fisik ('{phys_name}') TIDAK COCOK dengan penerima QRIS digital ('{dig_name}'). Terindikasi stiker ditimpa/palsu!"
    elif current_qr_risk_score >= 50.0:
        risk_level = "ELEVATED_RISK"
        explanation = f"HATI-HATI: Kemiripan nama '{name_similarity}%'. Periksa kembali nama toko sebelum melakukan transaksi."
    elif current_qr_risk_score >= 25.0:
        risk_level = "MODERATE_RISK"
        explanation = "Indikasi minor pergeseran identitas. Disarankan verifikasi ulang nominal pembayaran."
    else:
        risk_level = "SAFE"
        explanation = "Stiker QRIS terverifikasi aman. Identitas fisik dan digital cocok."

    # EMRS Merchant Reputation
    merchant_reputation = query_merchant_reputation(dig_nmid, phys_nmid, dig_name, phys_name)

    _update_merchant_identity_counter(dig_nmid, phys_nmid, dig_name, phys_name, is_mismatch)

    # DB session log
    db = SessionLocal()
    try:
        session_rec = VerificationSession(
            session_id=session_id,
            nmid=dig_nmid,
            digital_name=dig_name,
            physical_name=phys_name,
            status="MISMATCH" if is_mismatch else "MATCH",
            trust_score=current_trust_score,
            risk_level=risk_level,
            reputation_score=merchant_reputation.get("reputation_score", 50.0)
        )
        db.add(session_rec)
        db.commit()
    finally:
        db.close()

    return {
        "session_id": session_id,
        "current_qr_risk": {
            "risk_level": risk_level,
            "overall_risk_score": current_qr_risk_score,
            "trust_score": current_trust_score,
            "is_mismatch": is_mismatch,
            "name_similarity": name_similarity,
            "match_level": match_level,
            "explanation": explanation,
            "physical_merchant": phys_name if phys_name != "" else "Tidak terbaca",
            "digital_merchant": dig_name,
            "digital_city": dig_city,
            "physical_nmid": phys_nmid,
            "digital_nmid": dig_nmid,
            "physical_acquirer": phys_acq if phys_acq != "" else "Tidak terbaca",
            "digital_acquirer": dig_acq,
            "physical_tid": phys_tid if phys_tid != "" else "Tidak terbaca",
            "digital_tid": dig_tid,
            "technical_info": tech_info,
            "qris_raw_analysis": qris_analysis
        },
        "merchant_reputation": merchant_reputation,
        "visualization_url": f"/static/vis_output/vis_{filename_base}.jpg"
    }


def _update_merchant_identity_counter(dig_nmid, phys_nmid, dig_name, phys_name, is_mismatch):
    db = SessionLocal()
    try:
        m = None
        if dig_nmid and dig_nmid != "Tidak ditemukan":
            m = db.query(Merchant).filter(Merchant.nmid == dig_nmid).first()
        if not m and phys_nmid and phys_nmid != "Tidak terbaca":
            m = db.query(Merchant).filter(Merchant.nmid == phys_nmid).first()

        if m:
            if is_mismatch:
                m.identity_mismatch_count = (m.identity_mismatch_count or 0) + 1
            else:
                m.identity_match_count = (m.identity_match_count or 0) + 1
            db.commit()
    finally:
        db.close()


def submit_feedback_to_db(nmid: str, category: str, severity: str,
                          description: str, transaction_ref: str, has_evidence: bool) -> dict:
    db = SessionLocal()
    try:
        m = db.query(Merchant).filter(Merchant.nmid == nmid).first()
        if not m:
            return {"success": False, "message": f"Merchant NMID '{nmid}' tidak ditemukan.", "evidence_level": 0, "new_reputation_score": 0.0}

        evidence_level = 2 if has_evidence else 1

        if transaction_ref:
            existing = db.query(Report).filter(
                Report.merchant_id == m.id,
                Report.transaction_ref == transaction_ref
            ).first()
            if existing:
                return {"success": False, "message": "Feedback untuk transaksi ini sudah pernah disubmit.", "evidence_level": evidence_level, "new_reputation_score": m.reputation_score}

        rpt = Report(
            merchant_id=m.id,
            category=category,
            severity=severity,
            description=description,
            evidence_level=evidence_level,
            transaction_ref=transaction_ref,
            is_verified=(evidence_level == 2),
            created_at=datetime.utcnow()
        )
        db.add(rpt)

        m.total_reports = (m.total_reports or 0) + 1
        if evidence_level == 2:
            m.verified_reports = (m.verified_reports or 0) + 1
        if category == "QRIS Replacement" and severity == "CRITICAL":
            m.critical_mismatch_count = (m.critical_mismatch_count or 0) + 1

        db.commit()

        reports = db.query(Report).filter(Report.merchant_id == m.id).all()
        disputes = db.query(Dispute).filter(Dispute.merchant_id == m.id).all()
        emrs = calculate_emrs(m, reports, disputes)

        m.reputation_score = emrs["reputation_score"]
        db.commit()

        return {
            "success": True,
            "message": "Feedback berhasil disimpan. Terima kasih atas laporan Anda!",
            "evidence_level": evidence_level,
            "new_reputation_score": emrs["reputation_score"]
        }
    finally:
        db.close()


def get_merchant_reputation_by_nmid(nmid: str) -> dict:
    db = SessionLocal()
    try:
        m = db.query(Merchant).filter(Merchant.nmid == nmid).first()
        if not m:
            return {"found_in_db": False, "reputation_score": 0.0}
        reports = db.query(Report).filter(Report.merchant_id == m.id).all()
        disputes = db.query(Dispute).filter(Dispute.merchant_id == m.id).all()
        return calculate_emrs(m, reports, disputes)
    finally:
        db.close()
