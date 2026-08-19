import os
import cv2
import re
import difflib
import uuid
import numpy as np
import torch
from PIL import Image
from pyzbar import pyzbar
import warnings
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, RobertaTokenizer

from database import SessionLocal
from models import Merchant, Report, VerificationSession

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
    (255, 99, 71),   # 0: Cara Pakai QRIS
    (255, 165, 0),  # 1: Cek Aplikasi Penyelenggara
    (30, 144, 255), # 2: Dicetak Oleh
    (147, 112, 219),# 3: Logo GPN
    (50, 205, 50),  # 4: Logo dan deskripsi QRIS
    (0, 215, 255),  # 5: Nama Merchant
    (238, 130, 238),# 6: National Merchant ID
    (0, 0, 255),    # 7: QR Code
    (255, 105, 180),# 8: Slogan
    (128, 128, 0),  # 9: Terminal ID
    (0, 255, 255)   # 10: Versi Cetak
]

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
        except Exception:
            nama_trocr = "microsoft/trocr-base-stage1"
            tokenizer = RobertaTokenizer.from_pretrained(nama_trocr)
            image_processor = ViTImageProcessor.from_pretrained(nama_trocr)
            PROCESSOR_TROCR = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            MODEL_TROCR = VisionEncoderDecoderModel.from_pretrained(nama_trocr).to(PERANGKAT)

    return MODEL_YOLO_BARCODE, MODEL_YOLO_OCR, PROCESSOR_TROCR, MODEL_TROCR


def normalisasi_teks(teks_mentah):
    """
    Core Identity Normalization (M3):
    Membersihkan prefiks formal tanpa menghapus identitas nama utama toko.
    """
    if not teks_mentah or teks_mentah in ["Tidak terbaca", "Tidak ditemukan"]:
        return ""
    teks_kapital = teks_mentah.upper()
    teks_bersih = re.sub(r'[^A-Z0-9\s]', ' ', teks_kapital)
    daftar_kata = teks_bersih.split()
    
    # Prefiks legal / formal yang di-normalize
    sebutan = ["PT", "CV", "UD", "TB", "PD", "PERSERO"]
    daftar_kata_murni = [k for k in daftar_kata if k not in sebutan]
    hasil = " ".join(daftar_kata_murni).strip()
    return hasil if hasil != "" else " ".join(daftar_kata).strip()


def scan_qr_code_digital(gambar_input):
    hasil_scan = pyzbar.decode(gambar_input)
    if not hasil_scan:
        return None
    return hasil_scan[0].data.decode('utf-8')


def validate_and_parse_emvco_qr(teks_qr_mentah):
    """
    Tahap A — QR Payload Validation & EMVCo Structure Parsing (M1)
    Field 59: Merchant Name, Field 60: Merchant City, Field 51/26: Merchant Account, Field 54: Amount
    """
    if not teks_qr_mentah:
        return {"status": "INVALID_QR", "is_valid": False}, "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan"

    nama_dig = "Tidak ditemukan"
    kota_dig = "Tidak ditemukan"
    nmid_dig = "Tidak ditemukan"
    acq_dig = "Tidak ditemukan"
    tid_dig = "Tidak ditemukan"
    
    indeks, total = 0, len(teks_qr_mentah)
    has_format_indicator = False
    
    while indeks < total:
        kode_tag = teks_qr_mentah[indeks : indeks + 2]
        panjang_str = teks_qr_mentah[indeks + 2 : indeks + 4]
        if not panjang_str.isdigit():
            break
        size = int(panjang_str)
        isi = teks_qr_mentah[indeks + 4 : indeks + 4 + size]
        
        if kode_tag == "00":
            has_format_indicator = True
        elif kode_tag == "59":
            nama_dig = isi
        elif kode_tag == "60":
            kota_dig = isi
        elif kode_tag == "51":
            sub_idx = 0
            while sub_idx < len(isi):
                sub_tag = isi[sub_idx : sub_idx + 2]
                sub_len_str = isi[sub_idx + 2 : sub_idx + 4]
                if not sub_len_str.isdigit():
                    break
                sub_len = int(sub_len_str)
                sub_isi = isi[sub_idx + 4 : sub_idx + 4 + sub_len]
                if sub_tag == "02" and sub_isi.startswith("ID"):
                    nmid_dig = sub_isi
                sub_idx += 4 + sub_len
        elif kode_tag == "62":
            sub_idx = 0
            while sub_idx < len(isi):
                sub_tag = isi[sub_idx : sub_idx + 2]
                sub_len_str = isi[sub_idx + 2 : sub_idx + 4]
                if not sub_len_str.isdigit():
                    break
                sub_len = int(sub_len_str)
                sub_isi = isi[sub_idx + 4 : sub_idx + 4 + sub_len]
                if sub_tag == "07":
                    tid_dig = sub_isi
                    break
                sub_idx += 4 + sub_len
        indeks += 4 + size

    if nmid_dig == "Tidak ditemukan":
        m = re.search(r'ID\d{13}', teks_qr_mentah)
        if m: nmid_dig = m.group()

    m_acq = re.search(r'9360\d{4}', teks_qr_mentah)
    if m_acq: 
        kode_bank = m_acq.group()
        nama_bank = DAFTAR_NAMA_BANK.get(kode_bank, "BANK LAIN")
        acq_dig = f"{kode_bank} ({nama_bank})"

    payload_status = "VALID_QR_PAYLOAD" if has_format_indicator and nama_dig != "Tidak ditemukan" else "INVALID_STRUCTURE"
    tech_info = {"status": payload_status, "is_valid": (payload_status == "VALID_QR_PAYLOAD")}

    return tech_info, nama_dig, kota_dig, nmid_dig, acq_dig, tid_dig


def ocr_trocr(gambar_potongan, processor, model):
    if gambar_potongan is None or gambar_potongan.size == 0:
        return ""
    gambar_rgb = cv2.cvtColor(gambar_potongan, cv2.COLOR_BGR2RGB)
    gambar_pil = Image.fromarray(gambar_rgb)
    piksel = processor(gambar_pil, return_tensors="pt").pixel_values.to(PERANGKAT)
    with torch.no_grad():
        tokens = model.generate(piksel, max_new_tokens=64)
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def calculate_identity_similarity(phys_name, dig_name):
    """
    Multi-Evidence Identity Matching Model (M4 & M5):
    Level 1 — Exact Match (100)
    Level 2 — Normalized Match (100)
    Level 3 — Fuzzy Match (40-89)
    Level 4 — Completely Different (0-39)
    """
    if phys_name in ["Tidak terbaca", ""] or dig_name in ["Tidak ditemukan", ""]:
        return 0.0, "COMPLETELY_DIFFERENT", 100.0  # High Identity Risk

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

    # Identity Risk Score (Invers dari Similarity)
    # Thresholding: 90-100 -> Risk: 0-10, 70-89 -> Risk: 20-40, 40-69 -> Risk: 50-70, 0-39 -> Risk: 90-100
    if sim >= 90.0:
        identity_risk = 0.0
    elif sim >= 70.0:
        identity_risk = 30.0
    elif sim >= 40.0:
        identity_risk = 60.0
    else:
        identity_risk = 95.0

    return sim, level, identity_risk


def query_sqlite_reputation(nmid_digital, nmid_physical, merchant_name_dig, merchant_name_phys):
    """
    Reputation Risk Engine & Category Severity Weighting (M6):
    Categories & Severity:
    - QR Replacement: 80
    - Identity Mismatch: 60
    - Additional Fee: 40
    - General Complaint / Service: 10
    """
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
            rep_qris = db.query(Report).filter(Report.merchant_id == m.id, Report.category == "QRIS Replacement").count()
            rep_fee = db.query(Report).filter(Report.merchant_id == m.id, Report.category == "Additional Fee").count()
            rep_mismatch = db.query(Report).filter(Report.merchant_id == m.id, Report.category == "Merchant Mismatch").count()

            # Calculate Weighted Reputation Risk & Severity
            total = max(m.total_reports, 1)
            severity_risk = round(((rep_qris * 80) + (rep_mismatch * 60) + (rep_fee * 40)) / total, 1)
            
            # Reputation Risk (berdasarkan Rating & Jumlah Laporan Terverifikasi)
            rating_risk = (5.0 - m.rating) * 20.0  # Rating 2.1 -> (5-2.1)*20 = 58.0
            reputation_risk = min(100.0, round((rating_risk * 0.6) + (severity_risk * 0.4), 1))

            return {
                "status": "KNOWN",
                "found_in_db": True,
                "nmid": m.nmid,
                "merchant_name": m.merchant_name,
                "rating": m.rating,
                "total_reports": m.total_reports,
                "verified_reports": m.verified_reports,
                "reputation_risk_score": reputation_risk,
                "severity_risk_score": severity_risk,
                "breakdown_categories": {
                    "qris_replacement": rep_qris,
                    "additional_fee": rep_fee,
                    "merchant_mismatch": rep_mismatch
                }
            }
        else:
            return {
                "status": "UNKNOWN",
                "found_in_db": False,
                "nmid": nmid_digital if nmid_digital != "Tidak ditemukan" else nmid_physical,
                "merchant_name": merchant_name_dig if merchant_name_dig != "Tidak ditemukan" else merchant_name_phys,
                "rating": 5.0,
                "total_reports": 0,
                "verified_reports": 0,
                "reputation_risk_score": 0.0,
                "severity_risk_score": 0.0,
                "breakdown_categories": {"qris_replacement": 0, "additional_fee": 0, "merchant_mismatch": 0}
            }
    finally:
        db.close()


def process_qris_verification(gambar_input, filename_base="scan"):
    """
    Full Integrated Pipeline (M1-M8):
    1. QR Payload Validation (EMVCo)
    2. Physical Identity Extraction (YOLO26s + TrOCR)
    3. Multi-Evidence Identity Matching
    4. SQLite Reputation Risk & Report Severity Calculation
    5. Overall Risk Scoring (0.40 Identity + 0.30 Reputation + 0.20 Severity + 0.10 Technical)
    """
    model_barcode, model_ocr, proc_trocr, model_trocr = load_ai_models()
    session_id = str(uuid.uuid4())[:8]

    # 1. Deteksi Barcode pake Model 1
    res_barcode = model_barcode.predict(gambar_input, conf=0.25, verbose=False)[0]

    # 2. Deteksi Bounding Box Label OCR pake Model 2
    res_ocr = model_ocr.predict(gambar_input, conf=0.10, verbose=False)[0]

    gambar_vis = gambar_input.copy()
    tinggi_foto, lebar_foto = gambar_input.shape[:2]

    # Bounding box Barcode (Hijau)
    for box in res_barcode.boxes:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        cv2.rectangle(gambar_vis, (x1, y1), (x2, y2), (0, 255, 127), 4)
        cv2.putText(gambar_vis, "QR Code", (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)

    # Bounding box OCR Label
    daftar_nama_label = res_ocr.names
    kotak_terbaik = {}

    for box in res_ocr.boxes:
        id_label = int(box.cls[0].item())
        nama_resmi = daftar_nama_label.get(id_label, f"Label_{id_label}")
        conf = float(box.conf[0].item())
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        warna = DAFTAR_WARNA_LABEL[id_label % len(DAFTAR_WARNA_LABEL)]
        cv2.rectangle(gambar_vis, (x1, y1), (x2, y2), warna, 2)
        cv2.putText(gambar_vis, f"{nama_resmi}", (x1, max(y1-6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)

        kunci = PEMETAAN_LABEL_ROBOFLOW.get(nama_resmi.lower(), nama_resmi.lower())
        if conf >= 0.10:
            if kunci not in kotak_terbaik or conf > kotak_terbaik[kunci]['conf']:
                kotak_terbaik[kunci] = {'box': box, 'conf': conf}

    folder_backend = os.path.dirname(os.path.abspath(__file__))
    folder_vis = os.path.join(folder_backend, "static", "vis_output")
    os.makedirs(folder_vis, exist_ok=True)
    path_vis = os.path.join(folder_vis, f"vis_{filename_base}.jpg")
    cv2.imwrite(path_vis, gambar_vis)

    # 3. Decode & Validasi QR Payload Digital (M1)
    qr_text = scan_qr_code_digital(gambar_input)
    tech_info, dig_name, dig_city, dig_nmid, dig_acq, dig_tid = validate_and_parse_emvco_qr(qr_text)

    # 4. Extract Physical Text pake TrOCR (M2 & M3)
    potongan = {}
    for kunci, data in kotak_terbaik.items():
        coords = data['box'].xyxy[0].tolist()
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        pot = gambar_input[max(0, y1):min(tinggi_foto, y2), max(0, x1):min(lebar_foto, x2)]
        if pot.size > 0 and kunci != "qrcode":
            pot = cv2.resize(pot, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        potongan[kunci] = pot

    phys_name = ocr_trocr(potongan.get("nama_merchant"), proc_trocr, model_trocr) if "nama_merchant" in potongan else "Tidak terbaca"
    
    phys_nmid = "Tidak terbaca"
    if "nmid" in potongan:
        t = ocr_trocr(potongan["nmid"], proc_trocr, model_trocr).upper().replace(" ", "")
        m = re.search(r'[I1L][D0O][A-Z0-9]{13}', t)
        if m:
            tbl = str.maketrans("ILODSZBGT", "110052867")
            phys_nmid = "ID" + m.group()[-13:].translate(tbl)
        elif len(t) >= 10:
            phys_nmid = re.sub(r'^.*NMID[:\-\s]*', '', t)

    phys_acq = ocr_trocr(potongan.get("acquirer"), proc_trocr, model_trocr) if "acquirer" in potongan else "Tidak terbaca"
    phys_tid = ocr_trocr(potongan.get("tid"), proc_trocr, model_trocr) if "tid" in potongan else "Tidak terbaca"

    # 5. Multi-Evidence Identity Matching & Similarity (M4 & M5)
    name_similarity, match_level, identity_risk = calculate_identity_similarity(phys_name, dig_name)
    
    # NMID Matching check
    is_nmid_mismatch = (dig_nmid != "Tidak ditemukan" and phys_nmid != "Tidak terbaca" and dig_nmid != phys_nmid)
    if is_nmid_mismatch:
        identity_risk = max(identity_risk, 90.0)

    # 6. SQLite Reputation Check & Severity Risk (M6)
    reputation = query_sqlite_reputation(dig_nmid, phys_nmid, dig_name, phys_name)
    reputation_risk = reputation.get("reputation_risk_score", 0.0)
    severity_risk = reputation.get("severity_risk_score", 0.0)

    # Technical QR Risk (0 = Valid Payload, 100 = Invalid Payload/Structure)
    tech_risk = 0.0 if tech_info["is_valid"] else 100.0

    # 7. Combined Risk Scoring Model Formula (M7)
    # Risk Score = 0.40*IdentityRisk + 0.30*ReputationRisk + 0.20*SeverityRisk + 0.10*TechnicalRisk
    overall_risk_score = round(
        (0.40 * identity_risk) +
        (0.30 * reputation_risk) +
        (0.20 * severity_risk) +
        (0.10 * tech_risk),
        1
    )

    trust_score = round(max(0.0, 100.0 - overall_risk_score), 1)

    # Risk Levels:
    # 0-20 -> LOW, 21-40 -> MODERATE, 41-60 -> ELEVATED, 61-80 -> HIGH, 81-100 -> CRITICAL
    if overall_risk_score >= 61.0:
        risk_level = "HIGH_RISK"
    elif overall_risk_score >= 41.0:
        risk_level = "ELEVATED_RISK"
    elif overall_risk_score >= 21.0:
        risk_level = "MODERATE_RISK"
    else:
        risk_level = "SAFE"

    is_mismatch = (identity_risk >= 60.0)

    # 8. User Warning Explanation Text (M8)
    if is_mismatch:
        explanation = f"⚠️ INDIKASI KETIDAKSESUAIAN IDENTITAS: Stiker fisik toko ('{phys_name}') tidak cocok dengan rekening penerima QRIS digital ('{dig_name}'). Uang Anda berpotensi masuk ke rekening yang salah!"
    elif reputation.get("found_in_db") and reputation.get("rating", 5.0) <= 3.0:
        explanation = f"⚠️ PERINGATAN REPUTASI: Merchant digital '{dig_name}' memiliki rating {reputation['rating']}/5.0 dengan {reputation['total_reports']} laporan pengguna di SQLite!"
    else:
        explanation = f"✅ TERVERIFIKASI AMAN: Identitas fisik toko ('{phys_name}') 100% cocok dengan identitas penerima digital QR Code ('{dig_name}')."

    return {
        "session_id": session_id,
        "is_mismatch": is_mismatch,
        "risk_level": risk_level,
        "overall_risk_score": overall_risk_score,
        "trust_score": trust_score,
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
        "visualization_url": f"/static/vis_output/vis_{filename_base}.jpg",
        "reputation": reputation,
        "technical_info": tech_info
    }
