# =============================================================================
# LAQRIS BACKEND ENGINE (engine.py)
# sistem Deteksi Kecurangan QRIS & Merchant Reputation System (EMRS)
# =============================================================================
# File ini berisi seluruh logika utama aplikasi LaQris:
# 1. Memuat 3 Model AI (YOLO Barcode, YOLO OCR, dan TrOCR)
# 2. Membaca & mendekode data digital QRIS (Struktur Standar EMVCo ASPI)
# 3. Membaca teks fisik pada stiker QRIS menggunakan OCR (TrOCR)
# 4. Membandingkan identitas toko fisik vs digital (Identity Matching)
# 5. Menghitung Skor Reputasi Merchant (EMRS v2: Authenticity, Complaint, Dispute, Longevity)
# =============================================================================

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

# Impor koneksi database dan tabel dari file lokal
from database import SessionLocal
from models import Merchant, Report, Dispute, VerificationSession

# Abaikan pesan warning yang tidak kritis agar terminal tetap bersih
warnings.filterwarnings("ignore")

# Cek apakah komputer memiliki kartu grafis NVIDIA (GPU CUDA)
# Jika ada GPU maka gunakan "cuda", jika tidak gunakan prosesor "cpu"
PERANGKAT = "cuda" if torch.cuda.is_available() else "cpu"

# Variable global untuk menyimpan model AI agar hanya dimuat 1 kali ke memori RAM
MODEL_YOLO_BARCODE = None
MODEL_YOLO_OCR = None
PROCESSOR_TROCR = None
MODEL_TROCR = None

# Kamus (Dictionary) Kode Bank / Acquirer QRIS di Indonesia (EMVCo Tag 51/26)
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

# Kamus Kode Kategori Usaha (Merchant Category Code / MCC - EMVCo Tag 52)
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

# Pemetaan nama kelas label dari dataset YOLO OCR
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

# Daftar warna BGR untuk menggambar kotak deteksi (Bounding Box) pada foto
DAFTAR_WARNA_LABEL = [
    (255, 99, 71),    # Coral
    (255, 165, 0),    # Orange
    (30, 144, 255),   # Dodger Blue
    (147, 112, 219),  # Medium Purple
    (50, 205, 50),    # Lime Green
    (0, 215, 255),    # Gold
    (238, 130, 238),  # Violet
    (0, 0, 255),      # Red
    (255, 105, 180),  # Hot Pink
    (128, 128, 0),    # Olive
    (0, 255, 255)     # Cyan
]

# Pengurangan poin berdasarkan keparahan laporan keluhan (Complaint Score)
SEVERITY_PENALTY = {
    "LOW": 2,        # Keluhan ringan: minus 2 poin
    "MEDIUM": 5,     # Keluhan sedang: minus 5 poin
    "HIGH": 10,      # Keluhan berat: minus 10 poin
    "CRITICAL": 20   # Keluhan sangat kritis (misal QR ditimpa): minus 20 poin
}

# Pengurangan poin berdasarkan sengketa resmi (Dispute Score)
DISPUTE_PENALTY = {
    "verified": 30,    # Sengketa terbukti valid: minus 30 poin
    "unverified": 10   # Sengketa dalam proses: minus 10 poin
}


# =============================================================================
# FUNGSI 1: Memuat Model AI ke RAM (load_ai_models)
# =============================================================================
def load_ai_models():
    """
    Fungsi ini bertugas memuat 3 model AI ke dalam memori komputer:
    1. Model 1: YOLO Barcode (Menemukan letak QR Code)
    2. Model 2: YOLO OCR (Menemukan letak teks Nama Merchant, NMID, Bank)
    3. Model 3: TrOCR Microsoft (Membaca tulisan karakter dari potongan gambar)
    """
    global MODEL_YOLO_BARCODE, MODEL_YOLO_OCR, PROCESSOR_TROCR, MODEL_TROCR

    folder_backend = os.path.dirname(os.path.abspath(__file__))
    path_barcode = os.path.join(folder_backend, "weights", "yolo_barcode.pt")
    path_ocr = os.path.join(folder_backend, "weights", "yolo_ocr.pt")

    # 1. Memuat Model YOLO Barcode
    if MODEL_YOLO_BARCODE is None:
        print("[LOG] Memuat Model 1: YOLO Barcode dari", path_barcode)
        MODEL_YOLO_BARCODE = YOLO(path_barcode)

    # 2. Memuat Model YOLO OCR
    if MODEL_YOLO_OCR is None:
        print("[LOG] Memuat Model 2: YOLO OCR dari", path_ocr)
        MODEL_YOLO_OCR = YOLO(path_ocr)

    # 3. Memuat Model TrOCR (Transformer OCR)
    if PROCESSOR_TROCR is None or MODEL_TROCR is None:
        nama_trocr = "microsoft/trocr-base-printed"
        print(f"[LOG] Memuat Model TrOCR ({nama_trocr})...")
        try:
            tokenizer = RobertaTokenizer.from_pretrained(nama_trocr)
            image_processor = ViTImageProcessor.from_pretrained(nama_trocr)
            PROCESSOR_TROCR = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            MODEL_TROCR = VisionEncoderDecoderModel.from_pretrained(nama_trocr).to(PERANGKAT)
            MODEL_TROCR.eval()  # Set model ke mode evaluasi agar cepat
        except Exception:
            # Fallback jika model utama gagal diunduh
            nama_trocr = "microsoft/trocr-base-stage1"
            tokenizer = RobertaTokenizer.from_pretrained(nama_trocr)
            image_processor = ViTImageProcessor.from_pretrained(nama_trocr)
            PROCESSOR_TROCR = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            MODEL_TROCR = VisionEncoderDecoderModel.from_pretrained(nama_trocr).to(PERANGKAT)
            MODEL_TROCR.eval()

    return MODEL_YOLO_BARCODE, MODEL_YOLO_OCR, PROCESSOR_TROCR, MODEL_TROCR


# =============================================================================
# FUNGSI 2: Normalisasi Teks (normalisasi_teks)
# =============================================================================
def normalisasi_teks(teks_mentah):
    """
    Fungsi ini merapikan teks nama toko:
    - Mengubah menjadi huruf kapital semua.
    - Menghapus simbol/tanda baca.
    - Menghapus kata sebutan formal seperti PT, CV, UD, TB agar pencocokan nama lebih akurat.
    Contoh: "PT. Toko Maju Jaya!" -> "TOKO MAJU JAYA"
    """
    if not teks_mentah or teks_mentah in ["Tidak terbaca", "Tidak ditemukan"]:
        return ""
    
    # Ubah ke huruf besar
    teks_kapital = teks_mentah.upper()
    
    # Hapus semua karakter selain huruf A-Z, angka 0-9, dan spasi
    teks_bersih = re.sub(r'[^A-Z0-9\s]', ' ', teks_kapital)
    
    # Pecah kalimat menjadi kata-kata
    daftar_kata = teks_bersih.split()
    
    # Daftar kata sebutan badan usaha yang ingin diabaikan
    sebutan = ["PT", "CV", "UD", "TB", "PD", "PERSERO"]
    
    # Ambil kata-kata yang bukan sebutan badan usaha
    daftar_kata_murni = [k for k in daftar_kata if k not in sebutan]
    
    # Gabungkan kembali menjadi satu string teks
    hasil = " ".join(daftar_kata_murni).strip()
    return hasil if hasil != "" else " ".join(daftar_kata).strip()


# =============================================================================
# FUNGSI 3: Membaca QR Code Digital (scan_qr_code_digital)
# =============================================================================
def scan_qr_code_digital(gambar_input):
    """
    Fungsi ini membaca mentah teks QR Code dari gambar menggunakan library PyZBar.
    Mengembalikan string teks mentah QRIS (misal: "0002010102115144...")
    """
    hasil_scan = pyzbar.decode(gambar_input)
    if not hasil_scan:
        return None
    # Dekode bytes menjadi string UTF-8
    return hasil_scan[0].data.decode('utf-8')


# =============================================================================
# FUNGSI 4: Membedah Spesifikasi QRIS EMVCo (parse_emvco_qr_deep_analysis)
# =============================================================================
def parse_emvco_qr_deep_analysis(teks_qr_mentah):
    """
    Fungsi ini membedah (parse) tag-tag spesifikasi QRIS EMVCo resmi ASPI / Bank Indonesia:
    - Tag 01: Tipe Inisiasi ("11" = Statis/Stiker, "12" = Dinamis/EDC)
    - Tag 52: Merchant Category Code (MCC)
    - Tag 53: Mata Uang ("360" = Rupiah IDR)
    - Tag 63: CRC Checksum
    - Estimasi Tahun Registrasi dari digit NMID
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
    
    # Loop membaca tag TLV (Tag-Length-Value) pada standar EMVCo
    while indeks < total:
        kode_tag = teks_qr_mentah[indeks: indeks + 2]
        panjang_str = teks_qr_mentah[indeks + 2: indeks + 4]
        
        if not panjang_str.isdigit():
            break
            
        size = int(panjang_str)
        isi = teks_qr_mentah[indeks + 4: indeks + 4 + size]

        # Tag 01: Tipe Inisiasi QR
        if kode_tag == "01":
            initiation_code = isi
            initiation_label = "Dinamis (EDC/Layar Digital)" if isi == "12" else "Statis (Stiker Meja/Kasir)"
        # Tag 52: Kategori Usaha (MCC)
        elif kode_tag == "52":
            mcc_code = isi
            mcc_category = DAFTAR_MCC.get(isi, f"Kategori MCC ({isi})")
        # Tag 53: Mata Uang
        elif kode_tag == "53":
            currency = f"{isi} (IDR)" if isi == "360" else isi
        # Tag 63: CRC Checksum
        elif kode_tag == "63":
            crc_checksum = isi
            
        indeks += 4 + size

    # Estimasi tahun pendaftaran dari struktur NMID (ID1020... -> tahun 2020)
    nmid_match = re.search(r'ID(\d{2})(\d{2})?\d+', teks_qr_mentah)
    est_year = None
    if nmid_match:
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


# =============================================================================
# FUNGSI 5: Validasi & Ekstraksi Data Digital QRIS (validate_and_parse_emvco_qr)
# =============================================================================
def validate_and_parse_emvco_qr(teks_qr_mentah):
    """
    Fungsi ini mengekstrak data penting penerima digital dari string QRIS:
    - Tag 59: Nama Merchant Digital
    - Tag 60: Kota Merchant
    - Tag 51: National Merchant ID (NMID)
    - RegEx 9360: Bank Acquirer (BCA, Mandiri, BRI, BNI, DANA, dll)
    """
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
            # Parsing sub-tag di dalam Tag 51 untuk mengambil NMID
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
            # Parsing sub-tag di dalam Tag 62 untuk mengambil Terminal ID (TID)
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

    # Fallback pencarian NMID jika tidak ada di sub-tag 51
    if nmid_dig == "Tidak ditemukan":
        m = re.search(r'ID\d{13}', teks_qr_mentah)
        if m:
            nmid_dig = m.group()

    # Pencarian Kode Bank Acquirer (Standard Indonesia 9360xxxx)
    m_acq = re.search(r'9360\d{4}', teks_qr_mentah)
    if m_acq:
        kode_bank = m_acq.group()
        nama_bank = DAFTAR_NAMA_BANK.get(kode_bank, "BANK LAIN")
        acq_dig = f"{kode_bank} ({nama_bank})"

    payload_status = "VALID_QR_PAYLOAD" if has_format_indicator and nama_dig != "Tidak ditemukan" else "INVALID_STRUCTURE"
    tech_info = {"status": payload_status, "is_valid": (payload_status == "VALID_QR_PAYLOAD")}

    qris_analysis = parse_emvco_qr_deep_analysis(teks_qr_mentah)

    return tech_info, nama_dig, kota_dig, nmid_dig, acq_dig, tid_dig, qris_analysis


# =============================================================================
# FUNGSI 6: Membaca Karakter Teks Gambar Menggunakan TrOCR (ocr_trocr)
# =============================================================================
def ocr_trocr(gambar_potongan, processor, model):
    """
    Fungsi ini menerima potongan gambar (crop) dari stiker QRIS,
    lalu meminta AI TrOCR untuk menerjemahkannya menjadi teks string.
    """
    if gambar_potongan is None or gambar_potongan.size == 0:
        return ""
        
    # Konversi format warna gambar dari BGR (OpenCV) ke RGB (PIL Image)
    gambar_rgb = cv2.cvtColor(gambar_potongan, cv2.COLOR_BGR2RGB)
    gambar_pil = Image.fromarray(gambar_rgb)
    
    # Ubah gambar menjadi tensor piksel untuk model AI TrOCR
    piksel = processor(gambar_pil, return_tensors="pt").pixel_values.to(PERANGKAT)
    
    # Jalankan prediksi TrOCR tanpa menghitung gradient (agar cepat)
    with torch.inference_mode():
        tokens = model.generate(piksel, max_new_tokens=24)
        
    # Ubah hasil token angka menjadi string teks Latin
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


# =============================================================================
# FUNGSI 7: Pencocokan Identitas Fisik vs Digital (calculate_identity_similarity)
# =============================================================================
def calculate_identity_similarity(phys_name, dig_name):
    """
    Fungsi ini membandingkan nama toko yang tertulis di stiker fisik (OCR)
    dengan nama penerima uang yang ada di dalam barcode digital (QRIS).
    
    Tingkat Kemiripan (Similarity):
    - 90% - 100%: EXACT / NORMALIZED MATCH (Aman, Identitas Cocok)
    - 70% - 89%: PROBABLE MATCH (Kemiripan tinggi)
    - 40% - 69%: UNCERTAIN (Hati-hati)
    - < 40%: COMPLETELY DIFFERENT / MISMATCH (BAHAYA: Terindikasi QR ditimpa stiker lain!)
    """
    if phys_name in ["Tidak terbaca", ""] or dig_name in ["Tidak ditemukan", ""]:
        return 0.0, "COMPLETELY_DIFFERENT", 100.0

    p_raw = phys_name.lower().strip()
    d_raw = dig_name.lower().strip()
    p_norm = normalisasi_teks(phys_name).lower()
    d_norm = normalisasi_teks(dig_name).lower()

    # Cek pencocokan persis
    if p_raw == d_raw:
        sim = 100.0
        level = "EXACT_MATCH"
    elif p_norm == d_norm and p_norm != "":
        sim = 100.0
        level = "NORMALIZED_MATCH"
    else:
        # Hitung persentase kemiripan string menggunakan SequenceMatcher
        r1 = difflib.SequenceMatcher(None, p_raw, d_raw).ratio() * 100
        r2 = difflib.SequenceMatcher(None, p_norm, d_norm).ratio() * 100
        sim = round(max(r1, r2), 1)
        
        if sim >= 70.0:
            level = "PROBABLE_MATCH"
        elif sim >= 40.0:
            level = "UNCERTAIN"
        else:
            level = "COMPLETELY_DIFFERENT"

    # Tentukan poin risiko identitas (semakin tidak cocok, semakin tinggi risikonya)
    if sim >= 90.0:
        identity_risk = 0.0
    elif sim >= 70.0:
        identity_risk = 30.0
    elif sim >= 40.0:
        identity_risk = 60.0
    else:
        identity_risk = 95.0

    return sim, level, identity_risk


# =============================================================================
# FUNGSI 8: Menghitung Time Decay Weight (time_decay_weight)
# =============================================================================
def time_decay_weight(created_at: datetime) -> float:
    """
    Fungsi ini menghitung bobot penyusutan waktu (Time Decay) untuk laporan lama.
    Laporan masalah baru memiliki pengaruh 100%, sedangkan laporan yang sudah 1 tahun
    lalu pengaruhnya akan menyusut secara eksponensial.
    """
    now = datetime.utcnow()
    months_ago = max(0, (now - created_at).days / 30.0)
    return math.exp(-0.1 * months_ago)


# =============================================================================
# FUNGSI 9: Kalkulasi EMRS Merchant Reputation Score (calculate_emrs)
# =============================================================================
def calculate_emrs(merchant: Merchant, reports: list, disputes: list) -> dict:
    """
    FORMULA EMRS REVISI v2.0 (Evidence-Based Merchant Reputation Score):
    Skor Reputasi (R) = 0.40 * A + 0.30 * C + 0.20 * D + 0.10 * L
    
    Di mana:
    - A (Authenticity / Kesesuaian Identitas - 40%): Mengukur rasio match identitas stiker
    - C (Complaint / Rekam Laporan - 30%): Dimulai 100, dikurangi jika ada laporan pengguna
    - D (Dispute / Rekam Bebas Sengketa - 20%): Dimulai 100, dikurangi jika ada sengketa valid
    - L (Longevity / Keaktifan Toko - 10%): Lama teramatinya merchant di sistem LaQris
    """
    now = datetime.utcnow()

    # 1. Komponen A: Authenticity (40%)
    total_identity = merchant.identity_match_count + merchant.identity_mismatch_count
    if total_identity > 0:
        A_raw = (merchant.identity_match_count / total_identity) * 100.0
        # Hukuman ekstra jika pernah terjadi insiden QR replacement (stiker ditimpa)
        A = max(0.0, A_raw - (merchant.critical_mismatch_count * 15.0))
    else:
        A = 50.0  # Netral jika belum pernah di-scan

    # 2. Komponen C: Complaint Score (30%)
    C = 100.0
    for rpt in reports:
        sev_penalty = SEVERITY_PENALTY.get(rpt.severity, 5)
        ev_weight = 1.0 if rpt.evidence_level >= 2 else 0.5
        td = time_decay_weight(rpt.created_at)
        C -= sev_penalty * ev_weight * td
    C = max(0.0, min(100.0, C))

    # 3. Komponen D: Dispute Score (20%)
    D = 100.0
    for disp in disputes:
        penalty = DISPUTE_PENALTY["verified"] if disp.is_verified else DISPUTE_PENALTY["unverified"]
        td = time_decay_weight(disp.created_at)
        D -= penalty * td
    D = max(0.0, min(100.0, D))

    # 4. Komponen L: Keaktifan Toko / Longevity (10%)
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

    # Indikator Opsional: Keandalan Transaksi Terverifikasi di LaQris
    T_observed = None
    if merchant.verified_transactions > 0:
        T_observed = round((merchant.successful_transactions / merchant.verified_transactions) * 100.0, 1)

    # hitung Skor Akhir EMRS (0 - 100)
    R = round(
        (0.40 * A) +
        (0.30 * C) +
        (0.20 * D) +
        (0.10 * L),
        1
    )
    R = max(0.0, min(100.0, R))

    # Hitung Jumlah Bukti & Tingkat Kepercayaan Data (Confidence Level)
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

    # Penentuan Predikat Kategori (Grade)
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


# =============================================================================
# FUNGSI 10: Query Reputasi Merchant dari SQLite (query_merchant_reputation)
# =============================================================================
def query_merchant_reputation(nmid_digital, nmid_physical, merchant_name_dig, merchant_name_phys) -> dict:
    """
    Fungsi ini mencari data merchant di database SQLite berdasarkan NMID / Nama Toko.
    Jika merchant BELUM TERDAFTAR, fungsi mengembalikan nilai 'Belum Terdaftar' (found_in_db: False)
    tanpa memberikan skor palsu 50/100.
    """
    db = SessionLocal()
    try:
        m = None
        # Cari berdasarkan NMID digital
        if nmid_digital and nmid_digital != "Tidak ditemukan":
            m = db.query(Merchant).filter(Merchant.nmid == nmid_digital).first()
        # Cari berdasarkan NMID fisik
        if not m and nmid_physical and nmid_physical != "Tidak terbaca":
            m = db.query(Merchant).filter(Merchant.nmid == nmid_physical).first()
        # Cari berdasarkan nama merchant
        if not m and merchant_name_dig:
            m = db.query(Merchant).filter(Merchant.merchant_name.ilike(f"%{merchant_name_dig}%")).first()
        if not m and merchant_name_phys:
            m = db.query(Merchant).filter(Merchant.merchant_name.ilike(f"%{merchant_name_phys}%")).first()

        if m:
            # Merchant terdaftar -> hitung EMRS berdasarkan laporan dan sengketa di DB
            reports = db.query(Report).filter(Report.merchant_id == m.id).all()
            disputes = db.query(Dispute).filter(Dispute.merchant_id == m.id).all()
            return calculate_emrs(m, reports, disputes)
        else:
            # Merchant BELUM TERDAFTAR di database LaQris
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


# =============================================================================
# FUNGSI 11: Pipeline Utama Verifikasi QRIS (process_qris_verification)
# =============================================================================
def process_qris_verification(gambar_input, filename_base="scan"):
    """
    Ini adalah FUNGSI UTAMA yang memproses foto stiker QRIS dari pengguna:
    1. Menjalankan deteksi QR Code & OCR dengan AI
    2. Mendekode data digital QRIS EMVCo
    3. Membaca teks fisik toko
    4. Membandingkan kesesuaian fisik vs digital
    5. Mengambil skor reputasi merchant (EMRS)
    6. Menyimpan riwayat sesi ke database SQLite
    7. Mengembalikan hasil verifikasi lengkap ke Frontend
    """
    model_barcode, model_ocr, proc_trocr, model_trocr = load_ai_models()
    session_id = str(uuid.uuid4())[:8]

    # Predict dengan YOLO
    res_barcode = model_barcode.predict(gambar_input, conf=0.20, verbose=False)[0]
    res_ocr = model_ocr.predict(gambar_input, conf=0.15, verbose=False)[0]

    gambar_vis = gambar_input.copy()
    tinggi_foto, lebar_foto = gambar_input.shape[:2]

    # 1. Dekode QR Code Digital
    teks_qr_mentah = scan_qr_code_digital(gambar_input)
    if not teks_qr_mentah:
        # Jika scan full gagal, coba potong lokasi kotak QR hasil deteksi YOLO Barcode
        for box in res_barcode.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                potongan = gambar_input[max(0, y1):min(tinggi_foto, y2), max(0, x1):min(lebar_foto, x2)]
                teks_qr_mentah = scan_qr_code_digital(potongan)
                if teks_qr_mentah:
                    break

    # 2. Parse payload EMVCo QRIS digital
    tech_info, dig_name, dig_city, dig_nmid, dig_acq, dig_tid, qris_analysis = validate_and_parse_emvco_qr(teks_qr_mentah)

    # 3. Ekstraksi teks fisik dari stiker menggunakan YOLO OCR + TrOCR
    phys_name, phys_nmid, phys_acq, phys_tid = "", "", "", ""
    target_nmid_box = None
    target_qr_box = None

    for box in res_ocr.boxes:
        cls_id = int(box.cls[0].item())
        nama_kelas = model_ocr.names[cls_id]
        label_std = PEMETAAN_LABEL_ROBOFLOW.get(nama_kelas.lower().strip(), nama_kelas.lower().strip())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Potong area gambar teks fisik
        potongan = gambar_input[max(0, y1):min(tinggi_foto, y2), max(0, x1):min(lebar_foto, x2)]
        
        # Jalankan pembacaan karakter AI TrOCR
        teks_ocr = ocr_trocr(potongan, proc_trocr, model_trocr)

        if label_std == "nama_merchant" and not phys_name:
            phys_name = teks_ocr
        elif label_std == "nmid":
            target_nmid_box = (x1, y1, x2, y2)
            if not phys_nmid:
                # Bersihkan artefak awalan teks "NMID :"
                phys_nmid = re.sub(r'^(NMID\s*:?\s*)', '', teks_ocr, flags=re.IGNORECASE).strip()
        elif label_std == "qrcode":
            target_qr_box = (x1, y1, x2, y2)
        elif label_std == "acquirer" and not phys_acq:
            phys_acq = teks_ocr
        elif label_std == "tid" and not phys_tid:
            phys_tid = teks_ocr

        # Gambar kotak warna-warni pada foto visualisasi
        warna = DAFTAR_WARNA_LABEL[cls_id % len(DAFTAR_WARNA_LABEL)]
        cv2.rectangle(gambar_vis, (x1, y1), (x2, y2), warna, 2)
        cv2.putText(gambar_vis, f"{label_std}: {teks_ocr}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, warna, 1)

    # ── Fallback Cerdas: Jika Nama Merchant Fisik Belum Terdeteksi ───────────
    if not phys_name:
        ref_box = target_nmid_box or target_qr_box
        if ref_box is not None:
            rx1, ry1, rx2, ry2 = ref_box
            # Potong area persis di atas NMID / QR Code (tempat Nama Merchant berada)
            h_crop = max(45, int((ry2 - ry1) * 0.8))
            top_y1 = max(0, ry1 - h_crop)
            top_y2 = max(10, ry1 + 5)
            crop_nama = gambar_input[top_y1:top_y2, max(0, rx1 - 30):min(lebar_foto, rx2 + 30)]
            
            if crop_nama is not None and crop_nama.size > 0:
                teks_fallback = ocr_trocr(crop_nama, proc_trocr, model_trocr)
                if teks_fallback and len(teks_fallback) >= 3:
                    phys_name = teks_fallback
                    # Visualisasikan kotak fallback
                    cv2.rectangle(gambar_vis, (max(0, rx1 - 30), top_y1), (min(lebar_foto, rx2 + 30), top_y2), (0, 215, 255), 2)
                    cv2.putText(gambar_vis, f"nama_merchant (fallback): {phys_name}", (max(0, rx1 - 30), max(15, top_y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

    # 4. Pencocokan Identitas & Kalkulasi Risiko QR Saat Ini
    name_similarity, match_level, identity_risk = calculate_identity_similarity(phys_name, dig_name)
    is_mismatch = (match_level == "COMPLETELY_DIFFERENT" or name_similarity < 40.0)

    technical_risk = 0.0 if tech_info.get("is_valid") else 80.0
    current_qr_risk_score = round((0.70 * identity_risk) + (0.30 * technical_risk), 1)
    current_trust_score = round(100.0 - current_qr_risk_score, 1)

    # ── Gambar Banner Visual Status Fraud/Safe pada Foto ──────────────────────
    banner_color = (0, 0, 238) if is_mismatch else (34, 139, 34)  # Merah jika Mismatch, Hijau jika Aman
    banner_text = " [!] PERINGATAN: STIKER DITIMPA / FRAUD " if is_mismatch else " [v] VERIFIKASI BERHASIL: QRIS AMAN "

    # Draw top banner strip
    cv2.rectangle(gambar_vis, (0, 0), (lebar_foto, 42), banner_color, -1)
    cv2.putText(gambar_vis, banner_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

    # Sorotan khusus pada kotak QR Code (Highlight Box)
    if target_qr_box is not None:
        qx1, qy1, qx2, qy2 = target_qr_box
        border_col = (0, 0, 255) if is_mismatch else (0, 200, 0)
        tag_label = "TERINDIKASI DITIMPA / PALSU" if is_mismatch else "QR CODES MATCHED (AMAN)"
        cv2.rectangle(gambar_vis, (qx1 - 4, qy1 - 4), (qx2 + 4, qy2 + 4), border_col, 4)
        cv2.rectangle(gambar_vis, (qx1 - 4, max(42, qy1 - 28)), (qx2 + 4, qy1 - 4), border_col, -1)
        cv2.putText(gambar_vis, tag_label, (qx1 + 4, max(58, qy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    # Simpan gambar visualisasi hasil deteksi ke folder static
    folder_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "vis_output")
    os.makedirs(folder_static, exist_ok=True)
    path_vis = os.path.join(folder_static, f"vis_{filename_base}.jpg")
    cv2.imwrite(path_vis, gambar_vis)

    # Penentuan Tingkat Risiko & Kalimat Penjelasan untuk Pengguna
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

    # 5. Query Reputasi Merchant EMRS
    merchant_reputation = query_merchant_reputation(dig_nmid, phys_nmid, dig_name, phys_name)

    # Update counter identitas match/mismatch pada merchant jika terdaftar
    _update_merchant_identity_counter(dig_nmid, phys_nmid, dig_name, phys_name, is_mismatch)

    # 6. Simpan Catatan Sesi Verifikasi ke Database SQLite
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

    # 7. Kembalikan data lengkap dalam bentuk Dictionary JSON
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


# =============================================================================
# FUNGSI 12: Update Counter Identitas (_update_merchant_identity_counter)
# =============================================================================
def _update_merchant_identity_counter(dig_nmid, phys_nmid, dig_name, phys_name, is_mismatch):
    """
    Fungsi internal untuk menambah statistik match/mismatch pada merchant setiap kali di-scan.
    """
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


# =============================================================================
# FUNGSI 13: Menyimpan Laporan/Feedback Pengguna (submit_feedback_to_db)
# =============================================================================
def submit_feedback_to_db(nmid: str, category: str, severity: str,
                          description: str, transaction_ref: str, has_evidence: bool) -> dict:
    """
    Fungsi ini menyimpan laporan masalah pengguna ke database dan memperbarui skor EMRS toko.
    """
    db = SessionLocal()
    try:
        m = db.query(Merchant).filter(Merchant.nmid == nmid).first()
        if not m:
            return {"success": False, "message": f"Merchant NMID '{nmid}' tidak ditemukan.", "evidence_level": 0, "new_reputation_score": 0.0}

        # Level 2 jika melampirkan bukti transaksi, Level 1 jika laporan tanpa bukti
        evidence_level = 2 if has_evidence else 1

        # Cek pencegahan laporan duplikat untuk transaksi yang sama
        if transaction_ref:
            existing = db.query(Report).filter(
                Report.merchant_id == m.id,
                Report.transaction_ref == transaction_ref
            ).first()
            if existing:
                return {"success": False, "message": "Feedback untuk transaksi ini sudah pernah disubmit.", "evidence_level": evidence_level, "new_reputation_score": m.reputation_score}

        # Buat objek laporan baru
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

        # Update counter total laporan pada merchant
        m.total_reports = (m.total_reports or 0) + 1
        if evidence_level == 2:
            m.verified_reports = (m.verified_reports or 0) + 1
        if category == "QRIS Replacement" and severity == "CRITICAL":
            m.critical_mismatch_count = (m.critical_mismatch_count or 0) + 1

        db.commit()

        # Hitung ulang skor EMRS merchant setelah laporan baru masuk
        reports = db.query(Report).filter(Report.merchant_id == m.id).all()
        disputes = db.query(Dispute).filter(Dispute.merchant_id == m.id).all()
        emrs = calculate_emrs(m, reports, disputes)

        # Update skor EMRS terbaru ke database
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


# =============================================================================
# FUNGSI 14: Mengambil Reputasi Merchant Berdasarkan NMID (get_merchant_reputation_by_nmid)
# =============================================================================
def get_merchant_reputation_by_nmid(nmid: str) -> dict:
    """
    Fungsi bantuan API untuk mengambil skor EMRS merchant berdasarkan NMID.
    """
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
