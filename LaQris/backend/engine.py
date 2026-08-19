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
    if not teks_mentah or teks_mentah in ["Tidak terbaca", "Tidak ditemukan"]:
        return ""
    teks_kapital = teks_mentah.upper()
    teks_bersih = re.sub(r'[^A-Z0-9\s]', ' ', teks_kapital)
    daftar_kata = teks_bersih.split()
    sebutan = ["PT", "CV", "TOKO", "UD", "WARUNG", "KIOS", "TB", "PD", "RESTORAN", "RM", "STORE"]
    daftar_kata_murni = [k for k in daftar_kata if k not in sebutan]
    hasil = " ".join(daftar_kata_murni).strip()
    return hasil if hasil != "" else " ".join(daftar_kata).strip()


def scan_qr_code_digital(gambar_input):
    hasil_scan = pyzbar.decode(gambar_input)
    if not hasil_scan:
        return None
    return hasil_scan[0].data.decode('utf-8')


def parse_emvco_qr(teks_qr_mentah):
    nama_dig, nmid_dig, acq_dig, tid_dig = "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan"
    
    indeks, total = 0, len(teks_qr_mentah)
    while indeks < total:
        kode_tag = teks_qr_mentah[indeks : indeks + 2]
        panjang_str = teks_qr_mentah[indeks + 2 : indeks + 4]
        if not panjang_str.isdigit():
            break
        size = int(panjang_str)
        isi = teks_qr_mentah[indeks + 4 : indeks + 4 + size]
        
        if kode_tag == "59":
            nama_dig = isi
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

    return nama_dig, nmid_dig, acq_dig, tid_dig


def ocr_trocr(gambar_potongan, processor, model):
    if gambar_potongan is None or gambar_potongan.size == 0:
        return ""
    gambar_rgb = cv2.cvtColor(gambar_potongan, cv2.COLOR_BGR2RGB)
    gambar_pil = Image.fromarray(gambar_rgb)
    piksel = processor(gambar_pil, return_tensors="pt").pixel_values.to(PERANGKAT)
    with torch.no_grad():
        tokens = model.generate(piksel, max_new_tokens=64)
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def query_sqlite_reputation(nmid_digital, nmid_physical, merchant_name_dig, merchant_name_phys):
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

            laporan_list = db.query(Report).filter(Report.merchant_id == m.id).order_by(Report.id.desc()).limit(5).all()
            rekap_laporan = [{"category": r.category, "description": r.description, "is_verified": r.is_verified} for r in laporan_list]

            return {
                "found_in_db": True,
                "nmid": m.nmid,
                "merchant_name": m.merchant_name,
                "rating": m.rating,
                "total_reports": m.total_reports,
                "verified_reports": m.verified_reports,
                "breakdown_categories": {
                    "qris_replacement": rep_qris,
                    "additional_fee": rep_fee,
                    "merchant_mismatch": rep_mismatch
                },
                "recent_reports": rekap_laporan
            }
        return {
            "found_in_db": False,
            "nmid": nmid_digital if nmid_digital != "Tidak ditemukan" else nmid_physical,
            "merchant_name": merchant_name_dig if merchant_name_dig != "Tidak ditemukan" else merchant_name_phys,
            "rating": 5.0,
            "total_reports": 0,
            "verified_reports": 0,
            "breakdown_categories": {"qris_replacement": 0, "additional_fee": 0, "merchant_mismatch": 0},
            "recent_reports": []
        }
    finally:
        db.close()


def process_qris_verification(gambar_input, filename_base="scan"):
    model_barcode, model_ocr, proc_trocr, model_trocr = load_ai_models()
    session_id = str(uuid.uuid4())[:8]

    # 1. Deteksi Barcode pake Model 1
    res_barcode = model_barcode.predict(gambar_input, conf=0.25, verbose=False)[0]

    # 2. Deteksi Bounding Box Label OCR pake Model 2
    res_ocr = model_ocr.predict(gambar_input, conf=0.10, verbose=False)[0]

    gambar_vis = gambar_input.copy()
    tinggi_foto, lebar_foto = gambar_input.shape[:2]

    # Gambar box barcode (Warna Hijau Lime)
    for box in res_barcode.boxes:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        cv2.rectangle(gambar_vis, (x1, y1), (x2, y2), (0, 255, 127), 4)
        cv2.putText(gambar_vis, "QR Code", (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)

    # Gambar box OCR label
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

    # 3. Decode Digital QR String
    qr_text = scan_qr_code_digital(gambar_input)
    dig_name, dig_nmid, dig_acq, dig_tid = parse_emvco_qr(qr_text) if qr_text else ("Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan")

    # 4. Extract Physical Text pake TrOCR
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

    # 5. Detail Matching & Risk Analysis
    norm_dig = normalisasi_teks(dig_name)
    norm_phys = normalisasi_teks(phys_name)

    is_name_match = (dig_name.lower().strip() == phys_name.lower().strip() and dig_name != "") or (norm_dig == norm_phys and norm_dig != "")
    if not is_name_match and dig_name != "Tidak ditemukan" and phys_name != "Tidak terbaca":
        sim = difflib.SequenceMatcher(None, norm_dig, norm_phys).ratio()
        is_name_match = (sim >= 0.70)

    is_nmid_match = (dig_nmid != "Tidak ditemukan" and phys_nmid != "Tidak terbaca" and dig_nmid == phys_nmid)
    is_mismatch = (not is_name_match) or (dig_nmid != "Tidak ditemukan" and phys_nmid != "Tidak terbaca" and not is_nmid_match)

    reputation = query_sqlite_reputation(dig_nmid, phys_nmid, norm_dig, norm_phys)

    risk_level = "HIGH_RISK" if is_mismatch or (reputation and reputation.get("rating", 5.0) <= 3.0) else "SAFE"

    # Penjelasan Detail
    if is_mismatch:
        explanation = f"⚠️ PERINGATAN BAHAYA: Stiker fisik toko ('{phys_name}') TIDAK COCOK dengan penerima QR Code digital ('{dig_name}')! Uang Anda akan terkirim ke rekening lain jika pembayaran dilanjutkan."
    elif risk_level == "HIGH_RISK":
        explanation = f"⚠️ PERINGATAN BAHAYA: Merchant digital '{dig_name}' memiliki {reputation['total_reports']} laporan penipuan di database SQLite!"
    else:
        explanation = f"✅ TERVERIFIKASI AMAN: Identitas fisik toko ('{phys_name}') 100% cocok dengan identitas penerima digital QR Code ('{dig_name}')."

    return {
        "session_id": session_id,
        "is_mismatch": is_mismatch,
        "risk_level": risk_level,
        "explanation": explanation,
        "physical_merchant": phys_name if phys_name != "" else "Tidak terbaca",
        "digital_merchant": dig_name,
        "physical_nmid": phys_nmid,
        "digital_nmid": dig_nmid,
        "physical_acquirer": phys_acq if phys_acq != "" else "Tidak terbaca",
        "digital_acquirer": dig_acq,
        "physical_tid": phys_tid if phys_tid != "" else "Tidak terbaca",
        "digital_tid": dig_tid,
        "visualization_url": f"/static/vis_output/vis_{filename_base}.jpg",
        "reputation": reputation
    }
