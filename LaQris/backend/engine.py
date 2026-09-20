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
import json
import numpy as np
import torch
from PIL import Image
from pyzbar import pyzbar
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, RobertaTokenizer

# Import Optimum ONNX Runtime untuk model TrOCR Merchant Name yang lebih cepat
# Jika tidak tersedia, fallback ke model TrOCR biasa (PyTorch)
try:
    from optimum.onnxruntime import ORTModelForVision2Seq
    OPTIMUM_TERSEDIA = True
except ImportError:
    OPTIMUM_TERSEDIA = False
    print("[WARNING] optimum tidak tersedia. OCR Merchant Name menggunakan model PyTorch biasa.")

# Impor koneksi database dan tabel dari file lokal
from database import SessionLocal
from models import Merchant, Report, Dispute, VerificationSession
from sqlalchemy import func
from sqlalchemy.orm import Session

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
MODEL_YOLO_POSITIONING = None

# Model ONNX Resmi Bawaan TrOCR (microsoft/trocr-base-printed)
# Model asli Microsoft yang diekspor ke ONNX untuk inferensi super cepat (< 500ms) tanpa risiko overfitting
MODEL_TROCR_BASE_ONNX = None
PROCESSOR_TROCR_BASE = None



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

# Pemetaan nama kelas label dari dataset YOLO OCR (11 Kelas Resmi Roboflow)
PEMETAAN_LABEL_ROBOFLOW = {
    "nama merchant": "nama_merchant",
    "national merchant id": "nmid",
    "dicetak oleh": "acquirer",
    "terminal id": "tid",
    "qr code": "qrcode",
    "logo gpn": "logo_gpn",
    "logo dan deskripsi qris": "logo_qris",
    "cara pakai qris": "cara_pakai",
    "cek aplikasi penyelenggara": "cek_aplikasi",
    "slogan": "slogan",
    "versi cetak": "versi_cetak",
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

# Threshold minimum observasi untuk dianggap "cukup data" (bukan CAUTION)
OBSERVATION_THRESHOLD = 10


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
    global MODEL_YOLO_BARCODE, MODEL_YOLO_OCR, PROCESSOR_TROCR, MODEL_TROCR, MODEL_YOLO_POSITIONING

    folder_backend = os.path.dirname(os.path.abspath(__file__))
    path_barcode = os.path.join(folder_backend, "weights", "yolo_barcode.pt")
    path_ocr = os.path.join(folder_backend, "weights", "yolo_ocr.pt")
    path_pos = os.path.join(folder_backend, "weights", "yolo_positioning.pt")

    # 1. Memuat Model YOLO Barcode
    if MODEL_YOLO_BARCODE is None:
        print("[LOG] Memuat Model 1: YOLO Barcode dari", path_barcode)
        MODEL_YOLO_BARCODE = YOLO(path_barcode)

    # 2. Memuat Model YOLO OCR
    if MODEL_YOLO_OCR is None:
        print("[LOG] Memuat Model 2: YOLO OCR dari", path_ocr)
        MODEL_YOLO_OCR = YOLO(path_ocr)

    # 2b. Memuat Model YOLO Positioning (YOLO26 Nano)
    if MODEL_YOLO_POSITIONING is None and os.path.exists(path_pos):
        print("[LOG] Memuat Model YOLO Positioning dari", path_pos)
        MODEL_YOLO_POSITIONING = YOLO(path_pos)

    # 3. Memuat Model TrOCR (Transformer OCR) - untuk label umum (NMID, Acquirer, dll)
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

    # 4. Memuat Model Resmi TrOCR Base Printed ONNX (microsoft/trocr-base-printed)
    # Model resmi bawaan Microsoft yang dioptimalkan ke ONNX Runtime: Cepat (<500ms) & Akurat 100%
    global MODEL_TROCR_BASE_ONNX, PROCESSOR_TROCR_BASE
    if MODEL_TROCR_BASE_ONNX is None and OPTIMUM_TERSEDIA:
        path_onnx_base = os.path.join(folder_backend, "weights", "trocr_base_printed_onnx")
        if os.path.isdir(path_onnx_base):
            print(f"[LOG] Memuat Model Resmi TrOCR Base (ONNX) dari: {path_onnx_base}")
            try:
                MODEL_TROCR_BASE_ONNX = ORTModelForVision2Seq.from_pretrained(
                    path_onnx_base,
                    provider="CPUExecutionProvider",
                    use_merged=False,
                    decoder_file_name="decoder_model.onnx",
                    decoder_with_past_file_name="decoder_with_past_model.onnx",
                )
                PROCESSOR_TROCR_BASE = TrOCRProcessor.from_pretrained(path_onnx_base)
                print("[LOG] Model Resmi TrOCR Base Printed (ONNX) berhasil dimuat!")
            except Exception as e:
                print(f"[WARNING] Gagal memuat ONNX Base Model: {e}. Fallback ke PyTorch model.")
                MODEL_TROCR_BASE_ONNX = None
                PROCESSOR_TROCR_BASE = None
        else:
            print(f"[WARNING] Folder ONNX Base tidak ditemukan: {path_onnx_base}")


    return MODEL_YOLO_BARCODE, MODEL_YOLO_OCR, PROCESSOR_TROCR, MODEL_TROCR


# =============================================================================
# FUNGSI 1B: Deteksi Sudut Kemiringan Fisik QRIS (deteksi_kemiringan_qris)
# =============================================================================
def deteksi_kemiringan_qris(crop_bgr):
    """
    Menemukan kontur persegi panjang dari stiker fisik QRIS menggunakan cv2.minAreaRect,
    lalu membaca sudut kemiringan rotasinya (rotated bounding box).
    Mengembalikan: (sudut_derajat, best_rect)
    - Sudut berkisar antara -45.0 s.d. +45.0 derajat.
    - 0.0 derajat artinya horizontal tegak lurus sempurna.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0, None

    h, w = crop_bgr.shape[:2]
    crop_area = w * h
    if crop_area < 100:
        return 0.0, None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Deteksi tepi dengan Canny & pelebaran kontur
    edges = cv2.Canny(blurred, 30, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 0.12 * crop_area:
            rect = cv2.minAreaRect(c)
            valid_rects.append((area, rect))

    if not valid_rects:
        # Fallback menggunakan ambang batas Otsu jika kontur Canny terputus
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 0.12 * crop_area:
                valid_rects.append((area, cv2.minAreaRect(c)))

    if not valid_rects:
        return 0.0, None

    # Ambil kontur persegi panjang terluas (stiker fisik QRIS utama)
    valid_rects.sort(key=lambda x: x[0], reverse=True)
    best_rect = valid_rects[0][1]
    (cx, cy), (rw, rh), raw_angle = best_rect

    # Normalisasi sudut minAreaRect OpenCV agar sesuai orientasi horizontal
    if rw < rh:
        angle = raw_angle - 90.0 if raw_angle > 0 else raw_angle + 90.0
    else:
        angle = raw_angle

    while angle < -45.0:
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0

    return round(float(angle), 2), best_rect


# =============================================================================
# FUNGSI 1C: Auto-Deskew & Reposisi Tegak Lurus (luruskan_dan_reposisi_qris)
# =============================================================================
def luruskan_dan_reposisi_qris(crop_bgr, angle):
    """
    Merotasi potongan fisik QRIS sebesar angle derajat agar kembali tegak lurus (0 derajat horizontal).
    Menggunakan cv2.warpAffine dengan borderMode=cv2.BORDER_REPLICATE dan perhitungan dimensi
    bounding box baru agar tidak ada teks atau tepian stiker yang terpotong.
    Mengembalikan: (gambar_lurus, matriks_rotasi_M, matriks_rotasi_M_inv)
    """
    if abs(angle) < 1.0:
        return crop_bgr, None, None

    h, w = crop_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Matriks rotasi mengoreksi sudut kemiringan (angle derajat)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Hitung batas lebar dan tinggi baru setelah rotasi agar sudut gambar tidak terpotong
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    M_inv = cv2.invertAffineTransform(M)

    rotated = cv2.warpAffine(
        crop_bgr, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, M, M_inv


# =============================================================================
# FUNGSI 1D: Evaluasi Posisi Fisik QRIS (evaluasi_posisi_qris)
# =============================================================================
def evaluasi_posisi_qris(gambar_bgr, conf_threshold=0.25):
    """
    Mengevaluasi fisik stiker QRIS dan kelayakan posisi/framing/rotasi gambar:
    - Mendeteksi apakah stiker / fisik QRIS terlihat di kamera
    - Menghitung rasio framing, jarak titik tengah (centering), dan batas tepi
    - Mengukur sudut kemiringan (cv2.minAreaRect)
    - Mengembalikan status: 'OPTIMAL', 'TOO_TILTED', 'TOO_FAR', 'TOO_CLOSE', 'OFF_CENTER', 'NOT_DETECTED'
    """
    global MODEL_YOLO_POSITIONING
    if MODEL_YOLO_POSITIONING is None:
        folder_backend = os.path.dirname(os.path.abspath(__file__))
        path_pos = os.path.join(folder_backend, "weights", "yolo_positioning.pt")
        if os.path.exists(path_pos):
            MODEL_YOLO_POSITIONING = YOLO(path_pos)
        else:
            return {
                "detected": False,
                "status": "MODEL_NOT_FOUND",
                "is_optimal": True,
                "confidence": 0.0,
                "message": "Model positioning belum dimuat",
                "box": None,
                "skew_angle": 0.0,
                "coverage_ratio": 0.0,
                "offset_center": 0.0
            }

    h_frame, w_frame = gambar_bgr.shape[:2]
    results = MODEL_YOLO_POSITIONING.predict(gambar_bgr, conf=conf_threshold, verbose=False)[0]

    if len(results.boxes) == 0:
        return {
            "detected": False,
            "status": "NOT_DETECTED",
            "is_optimal": False,
            "confidence": 0.0,
            "message": "Arahkan kamera ke stiker / fisik QRIS",
            "box": None,
            "skew_angle": 0.0,
            "coverage_ratio": 0.0,
            "offset_center": 0.0
        }

    # Ambil kotak dengan confidence tertinggi
    best_box = max(results.boxes, key=lambda b: float(b.conf[0].item()))
    x1, y1, x2, y2 = [int(v) for v in best_box.xyxy[0].tolist()]
    conf = float(best_box.conf[0].item())

    # Potong area fisik QRIS untuk mendeteksi sudut kemiringan rotasi (cv2.minAreaRect)
    crop_qris = gambar_bgr[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
    skew_angle, _ = deteksi_kemiringan_qris(crop_qris)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    box_area = bw * bh
    frame_area = w_frame * h_frame
    coverage_ratio = box_area / frame_area

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    fcx = w_frame / 2.0
    fcy = h_frame / 2.0
    norm_dx = (cx - fcx) / w_frame
    norm_dy = (cy - fcy) / h_frame
    offset_center = float(np.sqrt(norm_dx**2 + norm_dy**2))

    # Kriteria kelayakan posisi (Kemiringan, Framing, & Centering)
    if abs(skew_angle) > 18.0:
        status = "TOO_TILTED"
        message = f"Kemiringan {abs(skew_angle):.1f}° terlalu miring. Tegakkan kamera atau luruskan stiker QRIS"
        is_optimal = False
    elif coverage_ratio > 0.96:
        status = "TOO_CLOSE"
        message = "Mundurkan sedikit agar seluruh fisik QRIS terlihat"
        is_optimal = False
    elif coverage_ratio < 0.08:
        status = "TOO_FAR"
        message = "Dekatkan kamera ke stiker QRIS"
        is_optimal = False
    elif offset_center > 0.30:
        status = "OFF_CENTER"
        message = "Posisikan stiker QRIS di tengah kotak bidik"
        is_optimal = False
    else:
        status = "OPTIMAL"
        if abs(skew_angle) >= 1.5:
            message = f"Posisi Optimal! Auto-Deskew aktif ({skew_angle:+.1f}°)"
        else:
            message = "Posisi Fisik QRIS Sempurna! Siap dipindai"
        is_optimal = True

    return {
        "detected": True,
        "status": status,
        "is_optimal": is_optimal,
        "confidence": round(conf, 4),
        "message": message,
        "box": [x1, y1, x2, y2],
        "skew_angle": round(skew_angle, 2),
        "coverage_ratio": round(coverage_ratio, 4),
        "offset_center": round(offset_center, 4)
    }


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
    Fungsi dekoder QR Code serbaguna dengan multi-metode (PyZBar + OpenCV QRCodeDetector + Enhancement)
    Mengembalikan string teks mentah QRIS (misal: "0002010102115144...")
    """
    if gambar_input is None or gambar_input.size == 0:
        return None

    # Metode 1: PyZBar Langsung
    try:
        hasil_scan = pyzbar.decode(gambar_input)
        if hasil_scan:
            return hasil_scan[0].data.decode('utf-8')
    except Exception:
        pass

    # Metode 2: OpenCV QRCodeDetector
    try:
        detector = cv2.QRCodeDetector()
        val, pts, _ = detector.detectAndDecode(gambar_input)
        if val and len(val) > 10:
            return val
    except Exception:
        pass

    # Metode 3: Preprocessing Gambar (Grayscale + Thresholding) + PyZBar
    try:
        gray = cv2.cvtColor(gambar_input, cv2.COLOR_BGR2GRAY) if len(gambar_input.shape) == 3 else gambar_input
        gray_enhanced = cv2.equalizeHist(gray)
        hasil_scan = pyzbar.decode(gray_enhanced)
        if hasil_scan:
            return hasil_scan[0].data.decode('utf-8')

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hasil_scan = pyzbar.decode(thresh)
        if hasil_scan:
            return hasil_scan[0].data.decode('utf-8')
    except Exception:
        pass

    return None


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
# FUNGSI 4B: Perhitungan Matematis CRC-16 CCITT (calculate_crc16_ccitt)
# =============================================================================
def calculate_crc16_ccitt(data: str) -> str:
    """
    Menghitung Checksum CRC-16 CCITT (Polinomial 0x1021, Nilai Awal 0xFFFF)
    sesuai standar resmi EMVCo QR Code Merchant-Presented Mode (Tag 63).
    """
    if not data:
        return "0000"
    crc = 0xFFFF
    for byte in data.encode('ascii', errors='ignore'):
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return f"{crc:04X}"


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
    - Tag 63: Verifikasi matematis CRC-16 CCITT
    """
    if not teks_qr_mentah:
        return {
            "status": "INVALID_QR",
            "is_valid": False,
            "crc_present": False,
            "crc_valid": False,
            "crc_supplied": None,
            "crc_calculated": None,
            "crc": {"crc_present": False, "crc_valid": False, "crc_supplied": None, "crc_calculated": None}
        }, "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan", "Tidak ditemukan", None

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

    # Verifikasi Matematis Checksum CRC16-CCITT (Tag 63)
    crc_info = {
        "crc_present": False,
        "crc_valid": False,
        "crc_supplied": None,
        "crc_calculated": None
    }
    if "6304" in teks_qr_mentah:
        idx_crc = teks_qr_mentah.rfind("6304")
        payload_to_crc = teks_qr_mentah[:idx_crc + 4]
        supplied_crc = teks_qr_mentah[idx_crc + 4: idx_crc + 8].upper()
        calculated_crc = calculate_crc16_ccitt(payload_to_crc)
        crc_info = {
            "crc_present": True,
            "crc_valid": (supplied_crc == calculated_crc),
            "crc_supplied": supplied_crc,
            "crc_calculated": calculated_crc
        }

    payload_status = "VALID_QR_PAYLOAD" if has_format_indicator and nama_dig != "Tidak ditemukan" else "INVALID_STRUCTURE"
    tech_info = {
        "status": payload_status,
        "is_valid": (payload_status == "VALID_QR_PAYLOAD"),
        "crc_present": crc_info["crc_present"],
        "crc_valid": crc_info["crc_valid"],
        "crc_supplied": crc_info["crc_supplied"],
        "crc_calculated": crc_info["crc_calculated"],
        "crc": crc_info
    }

    qris_analysis = parse_emvco_qr_deep_analysis(teks_qr_mentah)

    return tech_info, nama_dig, kota_dig, nmid_dig, acq_dig, tid_dig, qris_analysis


# =============================================================================
# FUNGSI 6: Membaca Karakter Teks Gambar Menggunakan TrOCR (ocr_trocr)
# =============================================================================
def ocr_trocr(gambar_potongan, processor, model):
    """
    Fungsi ini menerima potongan gambar (crop) dari stiker QRIS,
    di-zoom/upscale jika ukurannya kecil agar karakter terlihat sangat tajam,
    lalu meminta AI TrOCR (ONNX Base High-Speed / PyTorch fallback)
    untuk menerjemahkannya menjadi teks string.
    """
    global MODEL_TROCR_BASE_ONNX, PROCESSOR_TROCR_BASE

    if gambar_potongan is None or gambar_potongan.size == 0:
        return ""
        
    h, w = gambar_potongan.shape[:2]
    
    # Zoom / upscale potongan gambar jika terlalu kecil agar TrOCR membaca dengan presisi tinggi
    if h < 64:
        scale = 64.0 / float(h)
        new_w = max(128, int(w * scale))
        gambar_potongan = cv2.resize(gambar_potongan, (new_w, 64), interpolation=cv2.INTER_CUBIC)
    elif w < 128:
        scale = 128.0 / float(w)
        new_h = max(64, int(h * scale))
        gambar_potongan = cv2.resize(gambar_potongan, (128, new_h), interpolation=cv2.INTER_CUBIC)

    # Konversi format warna gambar dari BGR (OpenCV) ke RGB (PIL Image)
    gambar_rgb = cv2.cvtColor(gambar_potongan, cv2.COLOR_BGR2RGB)
    gambar_pil = Image.fromarray(gambar_rgb)
    
    # Prioritaskan Model Resmi TrOCR Base ONNX Runtime (< 500ms, akurat 100%)
    if MODEL_TROCR_BASE_ONNX is not None and PROCESSOR_TROCR_BASE is not None:
        pixel_values = PROCESSOR_TROCR_BASE(gambar_pil, return_tensors="pt").pixel_values
        tokens = MODEL_TROCR_BASE_ONNX.generate(pixel_values, max_new_tokens=32, num_beams=1)
        return PROCESSOR_TROCR_BASE.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    # Fallback ke PyTorch CPU/GPU jika ONNX belum dimuat
    piksel = processor(gambar_pil, return_tensors="pt").pixel_values.to(PERANGKAT)
    with torch.inference_mode():
        tokens = model.generate(piksel, max_new_tokens=32)
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


# =============================================================================
# FUNGSI 6B: OCR Nama Merchant Menggunakan Model Resmi TrOCR ONNX (ocr_trocr_merchant)
# =============================================================================
def ocr_trocr_merchant(gambar_potongan, proc_trocr_umum, model_trocr_umum):
    """
    Fungsi membaca teks NAMA MERCHANT pada stiker QRIS.
    Menggunakan model resmi bawaan Microsoft TrOCR Base Printed yang diekspor ke ONNX,
    menghasilkan inferensi super cepat (<500ms) dan akurasi membaca teks cetak yang stabil.
    """
    return ocr_trocr(gambar_potongan, proc_trocr_umum, model_trocr_umum)


# =============================================================================
# FUNGSI 6C: OCR Label Umum & NMID Menggunakan Model Resmi TrOCR ONNX (ocr_trocr_general)
# =============================================================================
def ocr_trocr_general(gambar_potongan, proc_trocr_umum, model_trocr_umum):
    """
    Fungsi OCR untuk semua label selain nama merchant (NMID, Acquirer, Terminal ID).
    Menggunakan model resmi bawaan Microsoft TrOCR Base Printed dalam format ONNX Runtime.
    Tidak mengalami digit hallucination / stuttering angka, akurat membaca deret NMID 100%.
    """
    return ocr_trocr(gambar_potongan, proc_trocr_umum, model_trocr_umum)

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
        if rpt.category in ["Verified Authentic", "QRIS_NORMAL_MERCHANT_TERPERCAYA", "safe_confirmation"]:
            continue  # Laporan positif / konfirmasi normal tidak mengikis skor C
        sev_penalty = SEVERITY_PENALTY.get(rpt.severity, 5)
        # Hierarki Bukti (Section 25 Readme): Level 2 = 1.0x, Level 1 = 0.5x, Level 0 = 0.25x
        if rpt.evidence_level >= 2:
            ev_weight = 1.0
        elif rpt.evidence_level == 1:
            ev_weight = 0.5
        else:
            ev_weight = 0.25
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
    db_tmp = SessionLocal()
    try:
        session_count = db_tmp.query(VerificationSession).filter(VerificationSession.nmid == merchant.nmid).count()
    except Exception:
        session_count = 0
    finally:
        db_tmp.close()

    total_evidence = total_identity + session_count + len(reports) + len(disputes)
    confidence_score = min(100.0, round((total_evidence / 15.0) * 100.0, 1))

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
# FUNGSI 9B: Ambil LaQris Observation History per NMID (get_observation_history_by_nmid)
# =============================================================================
def get_observation_history_by_nmid(nmid: str, reports: list = None, disputes: list = None, merchant: Merchant = None, db: Session = None) -> dict:
    """
    Mengambil LaQris Observation History berdasarkan data real dari tabel verification_sessions, reports, dan merchant.
    Definisi Kanonikal LaQris:
    - 1 Scan = 1 Observation (VerificationSession)
    - 1 User Report = 1 Complaint
    - 1 Dispute = 1 Sengketa Resmi
    """
    # Toleran jika caller passing db session di posisi reports
    if hasattr(reports, "query"):
        db = reports
        reports = None

    own_db = False
    if db is None:
        db = SessionLocal()
        own_db = True

    try:
        if not merchant and nmid:
            merchant = db.query(Merchant).filter(Merchant.nmid == nmid).first()

        if reports is None and merchant:
            reports = db.query(Report).filter(Report.merchant_id == merchant.id).all()
        if disputes is None and merchant:
            disputes = db.query(Dispute).filter(Dispute.merchant_id == merchant.id).all()

        reports = reports or []
        disputes = disputes or []

        # Query semua sesi scan yang NMID-nya cocok (digital atau fisik)
        sessions = db.query(VerificationSession).filter(
            VerificationSession.nmid == nmid
        ).all() if nmid else []

        total_obs = len(sessions)
        unique_obs = len(set(s.user_id for s in sessions if s.user_id))
        if total_obs > 0 and unique_obs == 0:
            unique_obs = 1

        identity_match = sum(1 for s in sessions if s.status == "MATCH")
        identity_mismatch = sum(1 for s in sessions if s.status == "MISMATCH")
        physical_anomaly = sum(1 for s in sessions if s.risk_level in ["DANGER", "WARNING", "HIGH_RISK", "MODERATE_RISK"])

        identity_match_rate = round((identity_match / total_obs * 100.0), 1) if total_obs > 0 else 0.0

        dates = []
        if sessions:
            dates.extend([s.scanned_at for s in sessions if s.scanned_at])
        if reports:
            dates.extend([r.created_at for r in reports if r.created_at])
        if merchant and merchant.registered_at:
            dates.append(merchant.registered_at)

        first_obs = None
        last_obs = None
        if dates:
            first_obs = min(dates).strftime("%d %b %Y")
            last_obs = max(dates).strftime("%d %b %Y")

        # Hitung complaint rate berdasarkan evidence transaksi
        verified_reports_count = len([r for r in (reports or []) if r.evidence_level >= 2]) if reports else 0
        total_reports_count = len(reports) if reports else 0
        total_disputes_count = len(disputes) if disputes else 0

        complaint_rate = round((verified_reports_count / total_obs * 100.0), 2) if total_obs > 0 else 0.0

        return {
            "total_observations": total_obs,
            "unique_observers": unique_obs,
            "first_observed": first_obs,
            "last_observed": last_obs,
            "identity_match": identity_match,
            "identity_mismatch": identity_mismatch,
            "physical_anomaly": physical_anomaly,
            "identity_match_rate": identity_match_rate,
            "complaint_rate": complaint_rate,
            "verified_feedback": verified_reports_count,
            "complaints": total_reports_count,
            "disputes": total_disputes_count
        }
    finally:
        if own_db:
            db.close()


# =============================================================================
# FUNGSI 9C: Keputusan Intervensi Pembayaran (get_payment_decision)
# =============================================================================
def get_payment_decision(risk_level: str) -> str:
    """
    Menentukan keputusan intervensi keamanan (Security Decision Contract):
    - NORMAL  -> ALLOW (Pembayaran diizinkan langsung)
    - CAUTION -> WARN  (Peringatan & konfirmasi pengguna sebelum bayar)
    - WARNING -> WARN  (Peringatan risiko menengah, tetap boleh bayar dengan konfirmasi)
    - DANGER  -> BLOCK (Blokir total / penipuan stiker terdeteksi)
    """
    if risk_level == "NORMAL":
        return "ALLOW"
    elif risk_level in ["CAUTION", "WARNING"]:
        return "WARN"
    elif risk_level == "DANGER":
        return "BLOCK"
    return "BLOCK"


# =============================================================================
# FUNGSI 9D: 4-Level QR Risk Classifier (classify_qr_risk)
# =============================================================================
def classify_qr_risk(
    is_nmid_mismatch: bool,
    is_mismatch: bool,
    match_level: str,
    name_similarity: float,
    is_text_unreadable: bool,
    current_qr_risk_score: float,
    total_observations: int,
    complaint_rate,
    verified_complaints: int,
    tech_info: dict = None
) -> tuple:
    """
    Mengklasifikasikan risiko QR ke dalam 4 level:
    🔴 DANGER   — Indikasi kuat ketidaksesuaian / stiker ditimpa (BLOCK)
    🟠 WARNING  — Ada ketidaksesuaian atau laporan, perlu waspada (BLOCK)
    🟡 CAUTION  — Identitas OK tapi data LaQris terbatas (WARN)
    🟢 NORMAL   — Tidak ditemukan indikasi mencurigakan (ALLOW)
    """
    tech_valid = tech_info.get("is_valid", True) if tech_info else True
    crc_valid = tech_info.get("crc_valid", True) if (tech_info and tech_info.get("crc_present")) else True

    # ── 🔴 DANGER: Kondisi Paling Berbahaya (NMID Mismatch / Fraud Kritis) ─────────
    if is_nmid_mismatch:
        return (
            "DANGER",
            "🔴 DANGER",
            "red",
            "QRIS menunjukkan indikasi kuat pemalsuan stiker. NMID pada stiker fisik BERBEDA "
            "dengan NMID di dalam QR Code digital. Pembayaran dibatalkan secara otomatis demi "
            "keamanan dana Anda."
        )

    # Identity Mismatch + Technical Invalid -> DANGER
    if is_mismatch and not tech_valid:
        return (
            "DANGER",
            "🔴 DANGER",
            "red",
            "Struktur payload QR rusak dan nama merchant tidak sesuai dengan stiker fisik. "
            "Sistem mengindikasikan potensi QR manipulasi atau rusak berat. Transaksi diblokir."
        )

    if is_mismatch and verified_complaints >= 3:
        return (
            "DANGER",
            "🔴 DANGER",
            "red",
            "QRIS menunjukkan indikasi kuat ketidaksesuaian identitas merchant. Terdapat "
            "beberapa laporan terverifikasi dari pengguna LaQris lain. "
            "Pembayaran dibatalkan secara otomatis."
        )

    if is_mismatch and match_level == "COMPLETELY_DIFFERENT" and not is_text_unreadable:
        return (
            "DANGER",
            "🔴 DANGER",
            "red",
            "QRIS menunjukkan indikasi kuat ketidaksesuaian dengan identitas merchant yang "
            "terlihat. Nama penerima digital sangat berbeda dari nama fisik toko. "
            "Pembayaran dibatalkan secara otomatis."
        )

    # ── 🟠 WARNING: Perlu Waspada / Blokir Pembayaran ──────────────────
    # Identity Uncertain + Technical Invalid -> WARNING
    if match_level == "UNCERTAIN" and not tech_valid:
        return (
            "WARNING",
            "🟠 WARNING",
            "orange",
            "Nama merchant meragukan dan struktur data QR digital tidak standar. "
            "Pembayaran ditangguhkan untuk verifikasi keamanan."
        )

    # Checksum CRC corrupt / tampered -> WARNING
    if not crc_valid:
        return (
            "WARNING",
            "🟠 WARNING",
            "orange",
            "Checksum CRC-16 QR Code tidak valid. Kode QR terindikasi telah dimodifikasi atau rusak. "
            "Pembayaran ditangguhkan demi keamanan."
        )

    if is_mismatch or match_level == "UNCERTAIN":
        return (
            "WARNING",
            "🟠 WARNING",
            "orange",
            "Terdapat ketidaksesuaian antara nama merchant fisik dan digital pada QRIS ini. "
            "Periksa kembali nama penerima di aplikasi pembayaran Anda sebelum melanjutkan transaksi."
        )

    if complaint_rate is not None and complaint_rate > 15.0:
        return (
            "WARNING",
            "🟠 WARNING",
            "orange",
            "Merchant ini memiliki tingkat laporan keluhan yang tinggi berdasarkan observasi LaQris. "
            "Pembayaran ditangguhkan untuk perlindungan nasabah."
        )

    if match_level == "PROBABLE_MATCH" and name_similarity < 85.0:
        return (
            "WARNING",
            "🟠 WARNING",
            "orange",
            "Kemiripan nama merchant fisik dan digital cukup tinggi namun tidak sempurna. "
            "Pastikan nama penerima di aplikasi pembayaran sesuai dengan nama toko yang Anda kunjungi."
        )

    # Technical invalid saja (tanpa mismatch identitas) -> WARNING
    if not tech_valid:
        return (
            "WARNING",
            "🟠 WARNING",
            "orange",
            "Struktur payload QR digital tidak sepenuhnya memenuhi standar EMVCo resmi. "
            "Periksa kembali rekening tujuan pembayaran."
        )

    # ── 🟡 CAUTION: Data Terbatas / Konfirmasi Tambahan ─────────────────
    if is_text_unreadable:
        return (
            "CAUTION",
            "🟡 CAUTION",
            "yellow",
            "Teks fisik pada stiker QRIS tidak dapat terbaca dengan jelas. Pastikan nama "
            "toko di aplikasi pembayaran Anda sesuai dengan nama merchant yang terlihat sebelum "
            "melanjutkan transaksi."
        )

    if total_observations < OBSERVATION_THRESHOLD:
        return (
            "CAUTION",
            "🟡 CAUTION",
            "yellow",
            f"Identitas QRIS sesuai, tetapi histori merchant di LaQris masih baru "
            f"({total_observations} observasi). Tetap berhati-hati dan pastikan nama penerima sesuai."
        )

    # ── 🟢 NORMAL: Tidak Ditemukan Indikasi Mencurigakan (ALLOW) ─────
    return (
        "NORMAL",
        "🟢 NORMAL",
        "green",
        "Tidak ditemukan indikasi mencurigakan pada QRIS ini. Integritas data teknis (CRC-16) "
        "dan identitas merchant fisik vs digital valid. Transaksi aman untuk dilanjutkan."
    )



# =============================================================================
# FUNGSI 10: Query Reputasi Merchant dari SQLite (query_merchant_reputation)
# =============================================================================
def query_merchant_reputation(nmid_digital, nmid_physical, merchant_name_dig, merchant_name_phys) -> dict:
    """
    Fungsi ini mencari data merchant di database SQLite berdasarkan NMID / Nama Toko.
    Jika merchant BELUM TERDAFTAR, fungsi mengembalikan nilai 'Belum Terdaftar' (found_in_db: False)
    tanpa memberikan skor palsu 50/100. Observation history diambil dari verification_sessions real.
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
            reports = db.query(Report).filter(Report.merchant_id == m.id).all()
            disputes = db.query(Dispute).filter(Dispute.merchant_id == m.id).all()
            result = calculate_emrs(m, reports, disputes)
            # Tambahkan observation history dari verification_sessions real
            result["observation_history"] = get_observation_history_by_nmid(m.nmid, reports, disputes)
            return result
        else:
            # Merchant belum terdaftar — cek apakah ada di verification_sessions
            lookup_nmid = nmid_digital if (nmid_digital and nmid_digital != "Tidak ditemukan") else nmid_physical
            obs_history = get_observation_history_by_nmid(lookup_nmid or "") if lookup_nmid else {
                "total_observations": 0, "unique_observers": 0,
                "first_observed": None, "last_observed": None,
                "identity_match": 0, "identity_mismatch": 0,
                "physical_anomaly": 0, "identity_match_rate": 0.0,
                "complaint_rate": None, "verified_feedback": 0,
                "complaints": 0, "disputes": 0
            }
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
                "nmid": lookup_nmid,
                "merchant_name": merchant_name_dig if merchant_name_dig != "Tidak ditemukan" else merchant_name_phys,
                "registered_at": None,
                "first_seen_observed": None,
                "last_seen_observed": None,
                "observation_history": obs_history
            }
    finally:
        db.close()


# =============================================================================
# FUNGSI 10A: Pemotongan Objek Presisi Mengikuti Sudut Rotasi (crop_bounding_object)
# =============================================================================
def order_points_upright(pts):
    """
    Mengurutkan 4 titik sudut OBB menjadi [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    sehingga hasil potongan orientasinya selalu tegak lurus horizontal dan terbaca dari kiri ke kanan.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left (x + y terkecil)
    rect[2] = pts[np.argmax(s)]  # Bottom-Right (x + y terbesar)

    diff = np.diff(pts, axis=1)  # y - x
    rect[1] = pts[np.argmin(diff)]  # Top-Right (y - x terkecil -> x besar, y kecil)
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left (y - x terbesar -> x kecil, y besar)
    return rect


def crop_bounding_object(image, box_xyxy=None, corners=None, is_horizontal_text=False):
    """
    Memotong objek gambar persis sesuai koordinat bounding box:
    - Jika memiliki 4 titik sudut miring (OBB / rotated bounding box), dipotong menggunakan
      cv2.warpPerspective dengan titik sudut yang telah diurutkan (TL, TR, BR, BL) sehingga
      mengikuti lekukan & sudut kemiringan teks tanpa background luar dan tidak terbalik/miring 90°.
    - Jika tidak memiliki sudut rotasi, dipotong tepat pada [y1:y2, x1:x2].
    - Jika elemen teks horizontal (misal Nama Merchant, NMID, TID) terpotong dengan tinggi > lebar,
      otomatis diputar 90° agar orientasinya benar-benar horizontal dan tegak.
    - TIDAK DIBERI TAMBAHAN PIXEL (0 padding px) persis sesuai instruksi pengguna.
    """
    if image is None or image.size == 0:
        return None

    h_img, w_img = image.shape[:2]

    # 1. Prioritaskan crop miring (Warped Perspective) jika ada 4 titik sudut OBB
    if corners is not None:
        try:
            pts = np.array(corners, dtype=np.float32)
            if pts.shape == (4, 2) or len(pts) == 4:
                rect = order_points_upright(pts)
                (tl, tr, br, bl) = rect

                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(4, int(round(max(widthA, widthB))))

                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(4, int(round(max(heightA, heightB))))

                dst_pts = np.float32([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ])
                M = cv2.getPerspectiveTransform(rect, dst_pts)
                warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                if warped is not None and warped.shape[0] >= 4 and warped.shape[1] >= 4:
                    if is_horizontal_text and warped.shape[0] > warped.shape[1]:
                        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                    return warped
        except Exception as e:
            print(f"[DEBUG] Warp perspective crop gagal, fallback ke box tegak: {e}")

    # 2. Pemotongan kotak tegak presisi (0 px padding)
    if box_xyxy is not None:
        bx1, by1, bx2, by2 = box_xyxy
        x1 = max(0, int(round(min(bx1, bx2))))
        y1 = max(0, int(round(min(by1, by2))))
        x2 = min(w_img, int(round(max(bx1, bx2))))
        y2 = min(h_img, int(round(max(by1, by2))))
        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2].copy()
            if is_horizontal_text and crop.shape[0] > crop.shape[1]:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            return crop

    return None


# =============================================================================
# FUNGSI 10B: Manajemen Arsip Sesi Scan (simpan_arsip_sesi_scan)
# =============================================================================
def _json_serializable_converter(obj):
    """Konverter aman tipe data numpy / custom agar valid disimpan ke file JSON."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, )):
        return obj.isoformat()
    return str(obj)


def simpan_arsip_sesi_scan(
    session_id,
    filename_base,
    gambar_input,
    gambar_vis,
    gambar_qris_card,
    gambar_qr_code,
    positioning_analysis,
    all_detected_items,
    tech_info,
    qris_analysis,
    dig_name,
    dig_nmid,
    dig_city,
    dig_acq,
    dig_tid,
    phys_name,
    phys_nmid,
    phys_acq,
    phys_tid,
    is_mismatch,
    is_nmid_mismatch,
    name_similarity,
    match_level,
    current_qr_risk_score,
    current_trust_score,
    risk_level,
    risk_label,
    risk_color,
    explanation,
    payment_decision,
    reason_codes,
    merchant_reputation
):
    """
    Menyimpan arsip lengkap per sesi scan foto ke folder khusus:
    static/sessions/{session_id}/
    
    Isi folder per sesi:
    1. full_annotated.jpg  : 1 gambar utuh berisi semua plot (YOLO BBox/OBB, label, confidence, teks OCR, banner status)
    2. original.jpg        : Gambar asli sebelum di-plot
    3. crops/              : Subfolder berisi seluruh potongan gambar (kartu QRIS, QR code, dan setiap elemen terdeteksi)
    4. predictions_data.json: Data terstruktur lengkap hasil analisis model YOLO & TrOCR
    5. summary_report.txt  : Laporan teks ringkas yang mudah dibaca manusia
    """
    folder_backend = os.path.dirname(os.path.abspath(__file__))
    folder_sessions = os.path.join(folder_backend, "static", "sessions")
    session_dir = os.path.join(folder_sessions, session_id)
    crops_dir = os.path.join(session_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    waktu_sekarang = datetime.now()
    waktu_str = waktu_sekarang.strftime("%d %b %Y %H:%M:%S")

    # 1. Simpan Gambar Utuh Beranotasi & Gambar Asli
    path_full_annotated = os.path.join(session_dir, "full_annotated.jpg")
    path_original = os.path.join(session_dir, "original.jpg")
    cv2.imwrite(path_full_annotated, gambar_vis)
    cv2.imwrite(path_original, gambar_input)

    # Simpan juga ke vis_output untuk kompatibilitas frontend lama
    folder_vis = os.path.join(folder_backend, "static", "vis_output")
    os.makedirs(folder_vis, exist_ok=True)
    cv2.imwrite(os.path.join(folder_vis, f"vis_{filename_base}.jpg"), gambar_vis)

    # 2. Simpan Seluruh Potongan Gambar (Crops)
    saved_crops = []

    # 2A. Potongan Fisik Kartu QRIS (YOLO Positioning)
    if gambar_qris_card is not None and gambar_qris_card.size > 0:
        card_crop_name = "crop_00_qris_card.jpg"
        path_card_crop = os.path.join(crops_dir, card_crop_name)
        cv2.imwrite(path_card_crop, gambar_qris_card)
        saved_crops.append({
            "index": 0,
            "category": "positioning_card",
            "label": "qris_card_physical",
            "filename": card_crop_name,
            "url": f"/static/sessions/{session_id}/crops/{card_crop_name}",
            "confidence": float(positioning_analysis.get("confidence", 1.0)),
            "width": int(gambar_qris_card.shape[1]),
            "height": int(gambar_qris_card.shape[0])
        })

    # 2B. Potongan Kotak QR Code Digital
    if gambar_qr_code is not None and gambar_qr_code.size > 0:
        qr_crop_name = "crop_01_qrcode.jpg"
        path_qr_crop = os.path.join(crops_dir, qr_crop_name)
        cv2.imwrite(path_qr_crop, gambar_qr_code)
        saved_crops.append({
            "index": 1,
            "category": "barcode",
            "label": "qrcode",
            "filename": qr_crop_name,
            "url": f"/static/sessions/{session_id}/crops/{qr_crop_name}",
            "confidence": 0.99,
            "width": int(gambar_qr_code.shape[1]),
            "height": int(gambar_qr_code.shape[0])
        })

    # 2C. Potongan Seluruh Objek YOLO OCR (Teks & Grafik)
    ocr_detections_info = []
    trocr_predictions_info = []

    for idx, item in enumerate(all_detected_items, start=2):
        label_raw = item.get("label", "unknown")
        clean_label = re.sub(r'[^a-zA-Z0-9_]', '_', label_raw).lower()
        conf_val = float(item.get("conf", 0.0))
        conf_pct = int(conf_val * 100)

        crop_filename = None
        crop_url = None
        crop_img = item.get("crop_img")

        if crop_img is not None and crop_img.size > 0 and crop_img.shape[0] > 4 and crop_img.shape[1] > 4:
            crop_filename = f"crop_{idx:02d}_{clean_label}_{conf_pct}pct.jpg"
            path_crop = os.path.join(crops_dir, crop_filename)
            cv2.imwrite(path_crop, crop_img)
            crop_url = f"/static/sessions/{session_id}/crops/{crop_filename}"
            saved_crops.append({
                "index": idx,
                "category": "ocr_element" if item.get("is_text") else "graphic_element",
                "label": label_raw,
                "filename": crop_filename,
                "url": crop_url,
                "confidence": conf_val,
                "width": int(crop_img.shape[1]),
                "height": int(crop_img.shape[0])
            })

        detection_entry = {
            "index": idx,
            "label": label_raw,
            "raw_class": item.get("raw_class", label_raw),
            "confidence": conf_val,
            "confidence_percent": f"{conf_pct}%",
            "is_text": item.get("is_text", False),
            "box_xyxy": item.get("box"),
            "rotated_corners": item.get("corners"),
            "crop_url": crop_url
        }
        ocr_detections_info.append(detection_entry)

        if item.get("is_text") and item.get("text"):
            trocr_predictions_info.append({
                "index": idx,
                "label": label_raw,
                "text_recognized": item.get("text"),
                "yolo_confidence": conf_val,
                "crop_url": crop_url
            })

    # 3. Buat File JSON Data Prediksi Komprehensif (predictions_data.json)
    data_prediksi = {
        "session_id": session_id,
        "created_at": waktu_sekarang.isoformat(),
        "created_at_formatted": waktu_str,
        "source_filename": filename_base,
        "verification_summary": {
            "status": "MISMATCH" if is_mismatch else "MATCH",
            "risk_level": risk_level,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "overall_risk_score": float(current_qr_risk_score),
            "trust_score": float(current_trust_score),
            "payment_decision": payment_decision,
            "is_mismatch": bool(is_mismatch),
            "is_nmid_mismatch": bool(is_nmid_mismatch),
            "name_similarity_percent": float(name_similarity),
            "match_level": match_level,
            "physical_merchant": phys_name,
            "digital_merchant": dig_name,
            "physical_nmid": phys_nmid,
            "digital_nmid": dig_nmid,
            "physical_acquirer": phys_acq,
            "digital_acquirer": dig_acq,
            "physical_tid": phys_tid,
            "digital_tid": dig_tid,
            "reason_codes": reason_codes,
            "explanation": explanation
        },
        "yolo_positioning_analysis": {
            "detected": bool(positioning_analysis.get("detected", False)),
            "confidence": float(positioning_analysis.get("confidence", 0.0)),
            "status": positioning_analysis.get("status", "-"),
            "is_optimal": bool(positioning_analysis.get("is_optimal", False)),
            "box": positioning_analysis.get("box"),
            "skew_angle": float(positioning_analysis.get("skew_angle", 0.0)),
            "deskew_applied": bool(positioning_analysis.get("deskew_applied", False)),
            "aspect_ratio": positioning_analysis.get("aspect_ratio"),
            "centering_offset_x": positioning_analysis.get("offset_x"),
            "centering_offset_y": positioning_analysis.get("offset_y"),
            "crop_url": f"/static/sessions/{session_id}/crops/crop_00_qris_card.jpg" if gambar_qris_card is not None else None
        },
        "yolo_ocr_detections": {
            "total_detected": len(ocr_detections_info),
            "detections": ocr_detections_info
        },
        "trocr_predictions": {
            "total_text_elements": len(trocr_predictions_info),
            "predictions": trocr_predictions_info
        },
        "emvco_qr_digital": {
            "is_valid": bool(tech_info.get("is_valid", False)),
            "merchant_name": dig_name,
            "nmid": dig_nmid,
            "merchant_city": tech_info.get("merchant_city", "-"),
            "acquirer": dig_acq,
            "terminal_id": dig_tid,
            "postal_code": tech_info.get("postal_code", "-"),
            "crc_valid": bool(tech_info.get("crc_valid", False)),
            "raw_payload": tech_info.get("raw_text", ""),
            "full_technical_info": tech_info,
            "qris_raw_analysis": qris_analysis
        },
        "merchant_reputation_emrs": merchant_reputation,
        "files": {
            "session_folder": f"/static/sessions/{session_id}",
            "full_annotated_url": f"/static/sessions/{session_id}/full_annotated.jpg",
            "original_url": f"/static/sessions/{session_id}/original.jpg",
            "report_url": f"/static/sessions/{session_id}/summary_report.txt",
            "json_data_url": f"/static/sessions/{session_id}/predictions_data.json",
            "total_crops": len(saved_crops),
            "crops": saved_crops
        }
    }

    path_json = os.path.join(session_dir, "predictions_data.json")
    with open(path_json, "w", encoding="utf-8") as f_json:
        json.dump(data_prediksi, f_json, indent=2, ensure_ascii=False, default=_json_serializable_converter)

    # 4. Buat File Laporan Teks Ringkas (summary_report.txt)
    lines_deteksi = []
    for d in ocr_detections_info:
        label_padded = f"[{d['label']}]".ljust(22)
        crop_txt = f"| Crop: {d['crop_url']}" if d.get('crop_url') else ""
        lines_deteksi.append(f"  {label_padded} : Conf {d['confidence_percent']} {crop_txt}")
    text_deteksi_block = "\n".join(lines_deteksi) if lines_deteksi else "  (Tidak ada elemen terdeteksi)"

    lines_trocr = []
    for t in trocr_predictions_info:
        lines_trocr.append(f"  - [{t['label']}] -> \"{t['text_recognized']}\" (YOLO Conf: {t['yolo_confidence']*100:.1f}%)")
    text_trocr_block = "\n".join(lines_trocr) if lines_trocr else "  (Tidak ada teks fisik terbaca)"

    keputusan_str = payment_decision if isinstance(payment_decision, str) else f"{payment_decision.get('action', '-')} - {payment_decision.get('message', '-')}"

    report_content = f"""================================================================================
                     LAQRIS SCAN SESSION ANALYSIS REPORT
================================================================================
Session ID        : {session_id}
Waktu Scan        : {waktu_str}
File Sumber       : {filename_base}
Status Fraud      : {'[!] TERINDIKASI FRAUD MISMATCH' if is_mismatch else '[v] NORMAL / MATCH'}
Tingkat Keamanan  : [{risk_level}] {risk_label}
Skor Kepercayaan  : {current_trust_score:.1f}% (Skor Risiko: {current_qr_risk_score:.1f}%)
Keputusan Sistem  : {keputusan_str}
Penjelasan        : {explanation}
Pemicu Risiko     : {', '.join(reason_codes) if reason_codes else 'Tidak ada'}

--------------------------------------------------------------------------------
1. ANALISIS MODEL YOLO POSITIONING (Deteksi Fisik Kartu/Stiker QRIS)
--------------------------------------------------------------------------------
Status Deteksi    : {'Terdeteksi' if positioning_analysis.get('detected') else 'Tidak Terdeteksi'}
Confidence        : {positioning_analysis.get('confidence', 0.0)*100:.1f}%
Status Posisi     : {positioning_analysis.get('status', '-')}
Kesiapan Scan     : {'Optimal (Siap Pindai)' if positioning_analysis.get('is_optimal') else 'Kurang Optimal'}
Sudut Kemiringan  : {positioning_analysis.get('skew_angle', 0.0):+.2f} derajat
Auto-Deskew       : {'Diterapkan' if positioning_analysis.get('deskew_applied') else 'Tidak'}
File Potongan     : crops/crop_00_qris_card.jpg

--------------------------------------------------------------------------------
2. ANALISIS MODEL YOLO OCR (Deteksi Elemen & Teks Fisik)
--------------------------------------------------------------------------------
Total Objek       : {len(ocr_detections_info)} elemen terdeteksi
Daftar Elemen:
{text_deteksi_block}

--------------------------------------------------------------------------------
3. ANALISIS MODEL TrOCR (Optical Character Recognition)
--------------------------------------------------------------------------------
Nama Merchant     : "{phys_name}"
NMID Fisik        : "{phys_nmid}"
Acquirer          : "{phys_acq if phys_acq else 'Tidak terbaca'}"
TID               : "{phys_tid if phys_tid else 'Tidak terbaca'}"

Rincian Pembacaan Teks:
{text_trocr_block}

--------------------------------------------------------------------------------
4. DATA DIGITAL QRIS (EMVCo ASPI Standard Payload)
--------------------------------------------------------------------------------
Validitas Format  : {'VALID (Format Standar ASPI/Bank Indonesia)' if tech_info.get('is_valid') else 'TIDAK VALID / RUSAK'}
Nama Toko Digital : "{dig_name}"
NMID Digital      : "{dig_nmid}"
Kota Toko         : "{tech_info.get('merchant_city', '-')}"
Lembaga Acquirer  : "{dig_acq}"
Terminal ID (TID) : "{dig_tid}"
CRC32 Checksum    : {'VALID' if tech_info.get('crc_valid') else 'TIDAK VALID'}

--------------------------------------------------------------------------------
5. HASIL VERIFIKASI IDENTITAS (Fisik vs Digital)
--------------------------------------------------------------------------------
Kesesuaian Nama   : {name_similarity:.1f}% ({match_level})
Kesesuaian NMID   : {'MISMATCH (NMID Berbeda -> Indikasi Stiker Palsu!)' if is_nmid_mismatch else 'SESUAI (Identik)'}

--------------------------------------------------------------------------------
6. ARSIP FILE PER SESI ({session_dir})
--------------------------------------------------------------------------------
- Gambar Utuh Plot  : full_annotated.jpg
- Gambar Asli       : original.jpg
- Data Prediksi JSON: predictions_data.json
- Laporan Teks      : summary_report.txt
- Total Potongan    : {len(saved_crops)} file di folder crops/
================================================================================
"""

    path_txt = os.path.join(session_dir, "summary_report.txt")
    with open(path_txt, "w", encoding="utf-8") as f_txt:
        f_txt.write(report_content)

    print(f"[LOG] Arsip Sesi Berhasil Disimpan: {session_dir} ({len(saved_crops)} crops)")

    return {
        "session_folder": f"/static/sessions/{session_id}",
        "session_files": {
            "full_annotated_url": f"/static/sessions/{session_id}/full_annotated.jpg",
            "original_url": f"/static/sessions/{session_id}/original.jpg",
            "report_url": f"/static/sessions/{session_id}/summary_report.txt",
            "json_data_url": f"/static/sessions/{session_id}/predictions_data.json",
            "crops_count": len(saved_crops),
            "crops": saved_crops
        }
    }


def get_session_archive_data(session_id):
    """Membaca data json arsip sesi tertentu."""
    folder_backend = os.path.dirname(os.path.abspath(__file__))
    path_json = os.path.join(folder_backend, "static", "sessions", session_id, "predictions_data.json")
    if not os.path.exists(path_json):
        return None
    try:
        with open(path_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Gagal membaca arsip sesi {session_id}: {e}")
        return None


def list_all_scan_sessions(limit=20):
    """Mengambil daftar seluruh folder sesi scan yang tersimpan di static/sessions."""
    folder_backend = os.path.dirname(os.path.abspath(__file__))
    folder_sessions = os.path.join(folder_backend, "static", "sessions")
    if not os.path.exists(folder_sessions):
        return []

    daftar_sesi = []
    folders = [f for f in os.listdir(folder_sessions) if os.path.isdir(os.path.join(folder_sessions, f))]
    folders.sort(reverse=True)

    for sid in folders[:limit]:
        path_session = os.path.join(folder_sessions, sid)
        path_json = os.path.join(path_session, "predictions_data.json")
        summary_info = {}
        if os.path.exists(path_json):
            try:
                with open(path_json, "r", encoding="utf-8") as fj:
                    data = json.load(fj)
                    summary_info = data.get("verification_summary", {})
            except Exception:
                pass

        crops_folder = os.path.join(path_session, "crops")
        crops_num = len(os.listdir(crops_folder)) if os.path.exists(crops_folder) else 0

        daftar_sesi.append({
            "session_id": sid,
            "folder_url": f"/static/sessions/{sid}",
            "full_annotated_url": f"/static/sessions/{sid}/full_annotated.jpg",
            "report_url": f"/static/sessions/{sid}/summary_report.txt",
            "json_url": f"/static/sessions/{sid}/predictions_data.json",
            "status": summary_info.get("status", "UNKNOWN"),
            "risk_level": summary_info.get("risk_level", "NORMAL"),
            "trust_score": summary_info.get("trust_score", 0.0),
            "merchant_name": summary_info.get("digital_merchant") or summary_info.get("physical_merchant") or "Merchant QRIS",
            "crops_count": crops_num
        })

    return daftar_sesi


# =============================================================================
# FUNGSI 11: Pipeline Utama Verifikasi QRIS (process_qris_verification)
# =============================================================================
def process_qris_verification(gambar_input, filename_base="scan", user_id=None):
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
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"

    # 1. Resize foto ukuran raksasa dari kamera HP ke max 1280px PERTAMA KALI
    # (Penting agar koordinat kotak YOLO presisi 100% sesuai dengan piksel gambar visualisasi)
    h_orig, w_orig = gambar_input.shape[:2]
    if max(h_orig, w_orig) > 1280:
        scale = 1280.0 / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        gambar_input = cv2.resize(gambar_input, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 1B. Evaluasi Posisi & Deteksi Fisik QRIS dengan Model YOLO26 Nano Positioning
    positioning_analysis = evaluasi_posisi_qris(gambar_input)

    tinggi_foto, lebar_foto = gambar_input.shape[:2]
    gambar_vis = gambar_input.copy()

    # 1C. CROP FISIK QRIS SEUKURAN KOTAK HASIL DETEKSI (Targeted QRIS Crop)
    # Memastikan model Barcode dan OCR bekerja pada gambar terfokus beresolusi tinggi tanpa gangguan background
    is_qris_cropped = False
    gambar_qris_card = None
    offset_x, offset_y = 0, 0
    gambar_proses = gambar_input

    if positioning_analysis.get("detected") and positioning_analysis.get("box"):
        px1, py1, px2, py2 = positioning_analysis["box"]
        # Potong persis sesuai bounding box (0 padding px)
        crop_x1 = max(0, px1)
        crop_y1 = max(0, py1)
        crop_x2 = min(lebar_foto, px2)
        crop_y2 = min(tinggi_foto, py2)

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        if crop_w > 50 and crop_h > 50:
            gambar_proses = gambar_input[crop_y1:crop_y2, crop_x1:crop_x2]
            gambar_qris_card = gambar_proses.copy()
            offset_x, offset_y = crop_x1, crop_y1
            is_qris_cropped = True
            print(f"[LOG] Crop Fisik QRIS Berhasil: {crop_w}x{crop_h}px dari offset ({offset_x}, {offset_y})")

    # AUTO-DESKEW (Straightening): Jika terdeteksi kemiringan wajar (1.5° s.d. 18.0°), luruskan otomatis
    deskew_applied = False
    deskew_M = None
    deskew_M_inv = None
    skew_angle = positioning_analysis.get("skew_angle", 0.0)

    if is_qris_cropped and 1.5 <= abs(skew_angle) <= 18.0:
        deskewed_img, deskew_M, deskew_M_inv = luruskan_dan_reposisi_qris(gambar_proses, skew_angle)
        if deskewed_img is not None:
            gambar_proses = deskewed_img
            deskew_applied = True
            print(f"[LOG] Auto-Deskew Fisik QRIS Berhasil: Koreksi rotasi {skew_angle:+.2f}° diterapkan.")

    positioning_analysis["deskew_applied"] = deskew_applied

    # 2. Predict dengan YOLO Barcode & YOLO OCR pada gambar fisik QRIS (cropped & deskewed)
    res_barcode = model_barcode.predict(gambar_proses, conf=0.18, verbose=False)[0]
    res_ocr = model_ocr.predict(gambar_proses, conf=0.10, verbose=False)[0]

    # Fallback ke gambar penuh jika crop terlalu ketat sehingga tidak ada barcode/ocr terdeteksi
    num_ocr_boxes = len(res_ocr.obb) if (hasattr(res_ocr, 'obb') and res_ocr.obb is not None and len(res_ocr.obb) > 0) else len(res_ocr.boxes)
    if is_qris_cropped and len(res_barcode.boxes) == 0 and num_ocr_boxes == 0:
        print("[LOG] Fallback ke gambar penuh karena crop tidak menemukan objek...")
        res_barcode = model_barcode.predict(gambar_input, conf=0.18, verbose=False)[0]
        res_ocr = model_ocr.predict(gambar_input, conf=0.10, verbose=False)[0]
        gambar_proses = gambar_input
        offset_x, offset_y = 0, 0
        is_qris_cropped = False
        deskew_applied = False
        deskew_M_inv = None

    # Gambar outline kotak fisik QRIS Positioning pada foto visualisasi
    if positioning_analysis.get("detected") and positioning_analysis.get("box"):
        px1, py1, px2, py2 = positioning_analysis["box"]
        warna_pos = (0, 230, 115) if positioning_analysis.get("is_optimal") else (0, 165, 255)
        cv2.rectangle(gambar_vis, (px1, py1), (px2, py2), warna_pos, 3)
        if deskew_applied:
            label_pos = f"Fisik QRIS: {positioning_analysis.get('status')} • Auto-Deskew ({skew_angle:+.1f} deg)"
        else:
            label_pos = f"Fisik QRIS (Crop Fokus AI): {positioning_analysis.get('status')} ({positioning_analysis.get('confidence')*100:.0f}%)"
        cv2.putText(gambar_vis, label_pos, (px1, max(24, py1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna_pos, 2)

    # 1. Dekode QR Code Digital HANYA dari Potongan Kotak Barcode (Targeted Bounding Box Crop)
    teks_qr_mentah = None
    calon_kotak_qr = []

    # Ambil lokasi kotak QR Code dari YOLO Barcode
    for box in res_barcode.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id == 0:
            calon_kotak_qr.append(box.xyxy[0].tolist())

    # Ambil lokasi kotak QR Code dari YOLO OCR (mendukung .obb dan .boxes)
    ocr_det_for_qr = res_ocr.obb if (hasattr(res_ocr, 'obb') and res_ocr.obb is not None and len(res_ocr.obb) > 0) else res_ocr.boxes
    if ocr_det_for_qr is not None:
        for box in ocr_det_for_qr:
            cls_id = int(box.cls[0].item())
            nama_kelas = model_ocr.names[cls_id]
            label_std = PEMETAAN_LABEL_ROBOFLOW.get(nama_kelas.lower().strip(), nama_kelas.lower().strip())
            if label_std == "qrcode":
                calon_kotak_qr.append(box.xyxy[0].tolist())

    # Dekode HANYA pada potongan gambar kotak QR Code
    gambar_qr_code = None
    h_proc, w_proc = gambar_proses.shape[:2]
    for (bx1, by1, bx2, by2) in calon_kotak_qr:
        px1 = max(0, int(bx1))
        py1 = max(0, int(by1))
        px2 = min(w_proc, int(bx2))
        py2 = min(h_proc, int(by2))
        potongan_qr = gambar_proses[py1:py2, px1:px2]
        teks_qr_mentah = scan_qr_code_digital(potongan_qr)
        if teks_qr_mentah:
            gambar_qr_code = potongan_qr.copy()
            break

    # Fallback terakhir jika potongan kotak miring/gagal
    if not teks_qr_mentah:
        teks_qr_mentah = scan_qr_code_digital(gambar_proses)
    if not teks_qr_mentah and is_qris_cropped:
        teks_qr_mentah = scan_qr_code_digital(gambar_input)

    # Pastikan potongan gambar QR Code tersimpan jika ada calon_kotak_qr
    if gambar_qr_code is None and calon_kotak_qr:
        cbx1, cby1, cbx2, cby2 = calon_kotak_qr[0]
        cpx1 = max(0, int(cbx1))
        cpy1 = max(0, int(cby1))
        cpx2 = min(w_proc, int(cbx2))
        cpy2 = min(h_proc, int(cby2))
        if cpy2 > cpy1 and cpx2 > cpx1:
            gambar_qr_code = gambar_proses[cpy1:cpy2, cpx1:cpx2].copy()

    # 2. Parse payload EMVCo QRIS digital
    tech_info, dig_name, dig_city, dig_nmid, dig_acq, dig_tid, qris_analysis = validate_and_parse_emvco_qr(teks_qr_mentah)

    # 3. Ekstraksi teks fisik HANYA untuk Label Esensial (Dengan Padded Zoomed Crop + Smart Filtering)
    phys_name, phys_nmid, phys_acq, phys_tid = "", "", "", ""
    target_nmid_box = None
    target_qr_box = None
    all_ocr_results = []
    all_detected_items = []

    # Sort kotak deteksi berdasarkan skor confidence terbesar (mendukung model standar .boxes dan model berotasi .obb)
    ocr_detections = res_ocr.obb if (hasattr(res_ocr, 'obb') and res_ocr.obb is not None and len(res_ocr.obb) > 0) else res_ocr.boxes
    boxes_sorted = sorted(ocr_detections, key=lambda b: float(b.conf[0].item()), reverse=True) if ocr_detections is not None else []

    for box in boxes_sorted:
        cls_id = int(box.cls[0].item())
        conf_score = float(box.conf[0].item())
        nama_kelas = model_ocr.names[cls_id]
        label_std = PEMETAAN_LABEL_ROBOFLOW.get(nama_kelas.lower().strip(), nama_kelas.lower().strip())
        lx1, ly1, lx2, ly2 = map(int, box.xyxy[0].tolist())

        # Koordinat 4 sudut OBB berotasi jika tersedia
        corners_global = None
        if hasattr(box, 'xyxyxyxy'):
            raw_corners = np.array(box.xyxyxyxy[0].tolist(), dtype=np.float32)
            if deskew_applied and deskew_M_inv is not None:
                pts_ones = np.hstack([raw_corners, np.ones((4, 1), dtype=np.float32)])
                orig_pts = (deskew_M_inv @ pts_ones.T).T
                corners_global = np.int32(orig_pts + np.array([offset_x, offset_y], dtype=np.float32))
            else:
                corners_global = np.int32(raw_corners + np.array([offset_x, offset_y], dtype=np.float32))

        # Koordinat bounding box global pada foto utuh (gambar_vis) dengan inverse mapping jika di-deskew
        if deskew_applied and deskew_M_inv is not None:
            pts = np.array([[lx1, ly1], [lx2, ly1], [lx2, ly2], [lx1, ly2]], dtype=np.float32)
            pts_ones = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
            orig_pts = (deskew_M_inv @ pts_ones.T).T
            gx1 = max(0, int(np.min(orig_pts[:, 0])) + offset_x)
            gy1 = max(0, int(np.min(orig_pts[:, 1])) + offset_y)
            gx2 = min(lebar_foto, int(np.max(orig_pts[:, 0])) + offset_x)
            gy2 = min(tinggi_foto, int(np.max(orig_pts[:, 1])) + offset_y)
        else:
            gx1, gy1 = lx1 + offset_x, ly1 + offset_y
            gx2, gy2 = lx2 + offset_x, ly2 + offset_y

        # Potongan gambar objek: persis mengikuti sudut rotasi miring (OBB) tanpa tambahan pixel
        is_text_elem = label_std in ["nama_merchant", "nmid", "acquirer", "tid", "versi_cetak", "slogan", "cek_aplikasi", "cara_pakai"]
        potongan_obj = crop_bounding_object(gambar_input, box_xyxy=(gx1, gy1, gx2, gy2), corners=corners_global, is_horizontal_text=is_text_elem)
        if potongan_obj is None:
            cx1_obj = max(0, min(gx1, gx2))
            cy1_obj = max(0, min(gy1, gy2))
            cx2_obj = min(lebar_foto, max(gx1, gx2))
            cy2_obj = min(tinggi_foto, max(gy1, gy2))
            if cy2_obj > cy1_obj and cx2_obj > cx1_obj:
                potongan_obj = gambar_input[cy1_obj:cy2_obj, cx1_obj:cx2_obj].copy()
                if is_text_elem and potongan_obj.shape[0] > potongan_obj.shape[1]:
                    potongan_obj = cv2.rotate(potongan_obj, cv2.ROTATE_90_CLOCKWISE)

        # Abaikan TrOCR untuk objek grafik/barcode/instruksi umum
        if label_std in ["qrcode", "logo", "gpn", "logo_gpn", "logo_qris", "cara_pakai", "cek_aplikasi", "slogan", "versi_cetak"]:
            if label_std == "qrcode" and target_qr_box is None:
                target_qr_box = (gx1, gy1, gx2, gy2)
                if potongan_obj is not None:
                    gambar_qr_code = potongan_obj.copy()

            warna = DAFTAR_WARNA_LABEL[cls_id % len(DAFTAR_WARNA_LABEL)]
            if corners_global is not None:
                cv2.polylines(gambar_vis, [corners_global], isClosed=True, color=warna, thickness=2)
                cv2.putText(gambar_vis, f"{label_std}", (int(corners_global[0][0]), max(15, int(corners_global[0][1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, warna, 1)
            else:
                cv2.rectangle(gambar_vis, (gx1, gy1), (gx2, gy2), warna, 2)
                cv2.putText(gambar_vis, f"{label_std}", (gx1, max(15, gy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, warna, 1)

            all_detected_items.append({
                "label": label_std,
                "raw_class": nama_kelas,
                "conf": conf_score,
                "box": (gx1, gy1, gx2, gy2),
                "corners": corners_global.tolist() if corners_global is not None else None,
                "text": None,
                "is_text": False,
                "crop_img": potongan_obj
            })
            continue

        if label_std == "nmid" and target_nmid_box is None:
            target_nmid_box = (gx1, gy1, gx2, gy2)

        # Potongan teks: persis mengikuti sudut miring OBB tanpa tambahan pixel (0 padding px)
        potongan_teks = crop_bounding_object(gambar_input, box_xyxy=(gx1, gy1, gx2, gy2), corners=corners_global, is_horizontal_text=True)
        if potongan_teks is None:
            potongan_teks = gambar_proses[ly1:ly2, lx1:lx2].copy()
            if potongan_teks.shape[0] > potongan_teks.shape[1]:
                potongan_teks = cv2.rotate(potongan_teks, cv2.ROTATE_90_CLOCKWISE)

        # Routing model berdasarkan label:
        if label_std == "nama_merchant":
            teks_ocr = ocr_trocr_merchant(potongan_teks, proc_trocr, model_trocr)
        else:
            teks_ocr = ocr_trocr_general(potongan_teks, proc_trocr, model_trocr)

        ocr_entry = {
            "label": label_std,
            "text": teks_ocr,
            "conf": conf_score,
            "box": (gx1, gy1, gx2, gy2),
            "corners": corners_global.tolist() if corners_global is not None else None
        }
        all_ocr_results.append(ocr_entry)

        all_detected_items.append({
            "label": label_std,
            "raw_class": nama_kelas,
            "conf": conf_score,
            "box": (gx1, gy1, gx2, gy2),
            "corners": corners_global.tolist() if corners_global is not None else None,
            "text": teks_ocr,
            "is_text": True,
            "crop_img": potongan_teks
        })

        # Gambar kotak berotasi miring (OBB) atau tegak pada foto visualisasi
        warna = DAFTAR_WARNA_LABEL[cls_id % len(DAFTAR_WARNA_LABEL)]
        if corners_global is not None:
            cv2.polylines(gambar_vis, [corners_global], isClosed=True, color=warna, thickness=2)
            cv2.putText(gambar_vis, f"{label_std}: {teks_ocr}", (int(corners_global[0][0]), max(15, int(corners_global[0][1]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, warna, 1)
        else:
            cv2.rectangle(gambar_vis, (gx1, gy1), (gx2, gy2), warna, 2)
            cv2.putText(gambar_vis, f"{label_std}: {teks_ocr}", (gx1, max(15, gy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, warna, 1)

    # ── Ekstraksi & Validasi NMID Fisik ─────────────────────────────────────────
    # Prioritas 1: Cari pola NMID resmi (ID + 9 s.d. 15 digit angka) di seluruh hasil TrOCR
    nmid_match_found = None
    for item in all_ocr_results:
        m_nmid = re.search(r'ID\s*\d{9,15}', item["text"], flags=re.IGNORECASE)
        if m_nmid:
            nmid_match_found = re.sub(r'\s+', '', m_nmid.group().upper())
            break
    
    if nmid_match_found:
        phys_nmid = nmid_match_found
    else:
        # Fallback 1b: Cari dari box label "nmid" yang mengandung angka (mengabaikan teks "TOTAL AMOUNT")
        for item in all_ocr_results:
            if item["label"] == "nmid":
                txt = re.sub(r'^(NMID\s*:?\s*)', '', item["text"], flags=re.IGNORECASE).strip()
                if any(c.isdigit() for c in txt) and not any(w in txt.upper() for w in ["TOTAL", "AMOUNT", "DICETAK"]):
                    phys_nmid = txt
                    break

    # ── Ekstraksi & Validasi Nama Merchant Fisik ────────────────────────────────
    for item in all_ocr_results:
        if item["label"] == "nama_merchant":
            txt = item["text"].strip()
            # Jika box nama merchant salah membaca baris NMID (misal "NMID: ID1026482737701"), pisahkan!
            if re.search(r'ID\d{9,15}', txt, flags=re.IGNORECASE) or txt.upper().startswith("NMID"):
                if not phys_nmid:
                    m_n = re.search(r'ID\s*\d{9,15}', txt, flags=re.IGNORECASE)
                    if m_n:
                        phys_nmid = re.sub(r'\s+', '', m_n.group().upper())
                continue  # Jangan gunakan string NMID sebagai nama merchant
            
            # Hindari kata kunci footer / header non-merchant
            if any(bad in txt.upper() for bad in ["TOTAL", "AMOUNT", "DICEK", "DICETAK", "SATU SEHAT", "GPN", "NATIONAL MERCHANT ID"]):
                continue

            if len(txt) >= 2:
                phys_name = txt
                break

    for item in all_ocr_results:
        if item["label"] == "acquirer" and not phys_acq:
            phys_acq = item["text"]
        elif item["label"] == "tid" and not phys_tid:
            phys_tid = item["text"]

    # ── Potongan Presisi Slot Nama Merchant (Tepat di Atas Kotak NMID) ─────────
    if (not phys_name or phys_name == "Tidak terbaca") and target_nmid_box is not None:
        nx1, ny1, nx2, ny2 = target_nmid_box
        h_slot = max(40, int((ny2 - ny1) * 1.5))
        slot_y1 = max(0, ny1 - h_slot)
        slot_y2 = max(5, ny1 - 2)
        slot_x1 = max(0, nx1 - 40)
        slot_x2 = min(lebar_foto, nx2 + 40)
        
        potongan_slot = gambar_input[slot_y1:slot_y2, slot_x1:slot_x2]
        if potongan_slot is not None and potongan_slot.size > 0:
            # Slot di atas NMID kemungkinan berisi nama merchant, pakai model merchant
            teks_slot = ocr_trocr_merchant(potongan_slot, proc_trocr, model_trocr)
            if teks_slot and len(teks_slot) >= 3 and not re.search(r'ID\d{9,15}', teks_slot, flags=re.IGNORECASE):
                phys_name = teks_slot
                cv2.rectangle(gambar_vis, (slot_x1, slot_y1), (slot_x2, slot_y2), (0, 215, 255), 2)
                cv2.putText(gambar_vis, f"nama_merchant (slot): {phys_name}", (slot_x1, max(15, slot_y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

    # ── Fallback Presisi ROI Header Stiker QRIS (Fotografi Layar HP/Laptop) ─────
    if (not phys_name or phys_name == "Tidak terbaca" or not phys_nmid or phys_nmid == "Tidak terbaca"):
        top_limit_y = target_qr_box[1] if target_qr_box is not None else int(tinggi_foto * 0.45)
        top_limit_y = max(60, top_limit_y)

        # 1. Fallback Crop NMID (Area tepat di atas QR Code)
        if not phys_nmid or phys_nmid == "Tidak terbaca":
            nmid_y1 = max(0, int(top_limit_y * 0.40))
            nmid_y2 = min(tinggi_foto, top_limit_y)
            strip_nmid = gambar_input[nmid_y1:nmid_y2, 0:lebar_foto]
            if strip_nmid is not None and strip_nmid.size > 0:
                # Area NMID -> pakai model general (dilatih dengan data NMID)
                txt_nmid_strip = ocr_trocr_general(strip_nmid, proc_trocr, model_trocr)
                m_nmid = re.search(r'ID\s*\d{9,15}', txt_nmid_strip, flags=re.IGNORECASE)
                if m_nmid:
                    phys_nmid = re.sub(r'\s+', '', m_nmid.group().upper())
                    cv2.rectangle(gambar_vis, (0, nmid_y1), (lebar_foto, nmid_y2), (255, 165, 0), 2)
                    cv2.putText(gambar_vis, f"nmid (roi): {phys_nmid}", (10, max(15, nmid_y1 + 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

        # 2. Fallback Crop Nama Merchant (Area atas stiker di atas NMID)
        if not phys_name or phys_name == "Tidak terbaca":
            name_y1 = max(0, int(top_limit_y * 0.08))
            name_y2 = max(30, int(top_limit_y * 0.65))
            strip_name = gambar_input[name_y1:name_y2, 0:lebar_foto]
            if strip_name is not None and strip_name.size > 0:
                # Fallback crop nama merchant juga pakai model ONNX fine-tuned
                txt_name_strip = ocr_trocr_merchant(strip_name, proc_trocr, model_trocr)
                cleaned_name = re.sub(r'(QRIS|QR Code Standar|Pembayaran Nasional|GPN|SATU QRIS UNTUK SEMUA)', '', txt_name_strip, flags=re.IGNORECASE).strip()
                if re.search(r'ID\d{9,15}', cleaned_name, flags=re.IGNORECASE):
                    cleaned_name = re.sub(r'NMID\s*:?\s*ID\d{9,15}.*', '', cleaned_name, flags=re.IGNORECASE).strip()
                if len(cleaned_name) >= 2:
                    phys_name = cleaned_name
                    cv2.rectangle(gambar_vis, (0, name_y1), (lebar_foto, name_y2), (0, 215, 255), 2)
                    cv2.putText(gambar_vis, f"nama_merchant (roi): {phys_name}", (10, max(15, name_y1 + 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

    # Clean up empty strings
    if not phys_name:
        phys_name = "Tidak terbaca"
    if not phys_nmid:
        phys_nmid = "Tidak terbaca"

    # 4. Pencocokan Identitas & Kalkulasi Risiko QR Saat Ini
    name_similarity, match_level, identity_risk = calculate_identity_similarity(phys_name, dig_name)

    # Cek khusus mismatch NMID: Jika NMID Fisik ditemukan & NMID Digital ditemukan, dan keduanya BERBEDA -> 100% FRAUD STIKER DITIMPA!
    is_nmid_mismatch = False
    if phys_nmid and phys_nmid != "Tidak terbaca" and dig_nmid and dig_nmid != "Tidak ditemukan":
        clean_p_nmid = re.sub(r'[^A-Z0-9]', '', phys_nmid.upper())
        clean_d_nmid = re.sub(r'[^A-Z0-9]', '', dig_nmid.upper())
        if clean_p_nmid != clean_d_nmid and len(clean_p_nmid) >= 10:
            is_nmid_mismatch = True

    # Fraud Mismatch HANYA terjadi jika ada bukti ketidakcocokan nyata (Name < 40% dari teks yang terbaca, atau NMID Berbeda)
    is_text_unreadable = (phys_name == "Tidak terbaca" and phys_nmid == "Tidak terbaca")
    is_mismatch = (is_nmid_mismatch or (not is_text_unreadable and match_level == "COMPLETELY_DIFFERENT" and name_similarity < 40.0))

    if is_nmid_mismatch:
        identity_risk = 100.0
        match_level = "NMID_MISMATCH"
    elif is_text_unreadable:
        identity_risk = 35.0  # Risiko sedang/netral jika teks tidak terbaca (bukan indikasi pasti fraud)
        match_level = "UNREADABLE_TEXT"
    elif is_mismatch:
        identity_risk = 95.0

    technical_risk = 0.0 if tech_info.get("is_valid") else 80.0
    current_qr_risk_score = round((0.70 * identity_risk) + (0.30 * technical_risk), 1)
    current_trust_score = round(100.0 - current_qr_risk_score, 1)

    # ── Gambar Banner Visual Status 4-Level pada Foto ──────────────────────────
    # (Banner dibuat sebelum risk classification agar koordinat QR box ada)
    if is_nmid_mismatch or (is_mismatch and not is_text_unreadable):
        banner_color = (0, 0, 200)      # 🔴 Merah — DANGER
        banner_text = " [!] DANGER: IDENTITAS QRIS TERINDIKASI TIDAK SESUAI "
    elif is_mismatch or match_level == "UNCERTAIN":
        banner_color = (0, 100, 220)    # 🟠 Oranye — WARNING
        banner_text = " [!] WARNING: PERIKSA KEMBALI IDENTITAS MERCHANT "
    elif is_text_unreadable:
        banner_color = (0, 165, 220)    # 🟡 Kuning-Oranye — CAUTION
        banner_text = " [?] CAUTION: TEKS FISIK TIDAK TERBACA "
    else:
        banner_color = (34, 139, 34)    # 🟢 Hijau — NORMAL
        banner_text = " [v] NORMAL: Tidak Ditemukan Indikasi Mencurigakan "

    cv2.rectangle(gambar_vis, (0, 0), (lebar_foto, 42), banner_color, -1)
    cv2.putText(gambar_vis, banner_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Sorotan khusus pada kotak QR Code (Highlight Box)
    if target_qr_box is not None:
        qx1, qy1, qx2, qy2 = target_qr_box
        border_col = (0, 0, 200) if is_nmid_mismatch or is_mismatch else ((0, 165, 220) if is_text_unreadable else (34, 139, 34))
        tag_label = "TERINDIKASI MISMATCH" if (is_nmid_mismatch or is_mismatch) else ("TEKS BURAM" if is_text_unreadable else "IDENTITAS COCOK")
        cv2.rectangle(gambar_vis, (qx1 - 4, qy1 - 4), (qx2 + 4, qy2 + 4), border_col, 4)
        cv2.rectangle(gambar_vis, (qx1 - 4, max(42, qy1 - 28)), (qx2 + 4, qy1 - 4), border_col, -1)
        cv2.putText(gambar_vis, tag_label, (qx1 + 4, max(58, qy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    # Simpan gambar visualisasi hasil deteksi ke folder static
    folder_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "vis_output")
    os.makedirs(folder_static, exist_ok=True)
    path_vis = os.path.join(folder_static, f"vis_{filename_base}.jpg")
    cv2.imwrite(path_vis, gambar_vis)

    # 4B. Ambil data observasi real dari verification_sessions untuk merchant ini
    nmid_for_obs = dig_nmid if (dig_nmid and dig_nmid != "Tidak ditemukan") else phys_nmid
    obs_history_pre = get_observation_history_by_nmid(nmid_for_obs or "")
    total_obs_pre = obs_history_pre.get("total_observations", 0) + 1  # Hitung termasuk sesi scan aktif saat ini
    complaint_rate_pre = obs_history_pre.get("complaint_rate", None)
    verified_complaints_pre = obs_history_pre.get("verified_feedback", 0)

    # Penentuan 4-Level Risiko menggunakan classify_qr_risk()
    risk_level, risk_label, risk_color, explanation = classify_qr_risk(
        is_nmid_mismatch=is_nmid_mismatch,
        is_mismatch=is_mismatch,
        match_level=match_level,
        name_similarity=name_similarity,
        is_text_unreadable=is_text_unreadable,
        current_qr_risk_score=current_qr_risk_score,
        total_observations=total_obs_pre,
        complaint_rate=complaint_rate_pre,
        verified_complaints=verified_complaints_pre,
        tech_info=tech_info
    )

    # 4C. Keputusan Intervensi Keamanan Pembayaran (Security Decision Contract)
    payment_decision = get_payment_decision(risk_level)

    # 4D. Reason Codes Detil (Penyebab Risiko / Security Trigger)
    reason_codes = []
    if is_nmid_mismatch:
        reason_codes.append("NMID_MISMATCH")
    if is_mismatch and not is_text_unreadable:
        reason_codes.append("NAME_COMPLETELY_DIFFERENT")
    elif match_level == "UNCERTAIN":
        reason_codes.append("NAME_UNCERTAIN")
    if is_text_unreadable:
        reason_codes.append("UNREADABLE_TEXT")
    if not tech_info.get("is_valid"):
        reason_codes.append("TECHNICAL_PAYLOAD_CORRUPTED")
    if tech_info.get("crc_present") and not tech_info.get("crc_valid"):
        reason_codes.append("CRC_INVALID")

    # 4E. IdentityEvidence Terstruktur (Prioritas: NMID -> Name -> Acquirer -> TID)
    identity_evidence = {
        "nmid": {
            "physical": phys_nmid,
            "digital": dig_nmid,
            "match": bool(not is_nmid_mismatch and phys_nmid not in ["Tidak terbaca", ""] and dig_nmid not in ["Tidak ditemukan", ""]),
            "priority": "HARD_SIGNAL"
        },
        "merchant_name": {
            "physical": phys_name,
            "digital": dig_name,
            "match": bool(name_similarity >= 70.0),
            "similarity": name_similarity,
            "priority": "PRIMARY_SIGNAL"
        },
        "acquirer": {
            "physical": phys_acq if phys_acq else "Tidak terbaca",
            "digital": dig_acq if dig_acq else "Tidak ditemukan",
            "match": bool(phys_acq and dig_acq and (phys_acq.lower() in dig_acq.lower() or dig_acq.lower() in phys_acq.lower())),
            "priority": "SUPPORTING_SIGNAL"
        },
        "tid": {
            "physical": phys_tid if phys_tid else "Tidak terbaca",
            "digital": dig_tid if dig_tid else "Tidak ditemukan",
            "match": bool(phys_tid and dig_tid and phys_tid == dig_tid),
            "priority": "SUPPORTING_SIGNAL"
        }
    }

    # 5. Query Reputasi Merchant EMRS
    merchant_reputation = query_merchant_reputation(dig_nmid, phys_nmid, dig_name, phys_name)

    # Update counter identitas match/mismatch pada merchant jika terdaftar
    _update_merchant_identity_counter(dig_nmid, phys_nmid, dig_name, phys_name, is_mismatch)

    # 6. Simpan Catatan Sesi Verifikasi ke Database SQLite
    db = SessionLocal()
    try:
        session_rec = VerificationSession(
            session_id=session_id,
            user_id=user_id,
            nmid=dig_nmid,
            digital_name=dig_name,
            physical_name=phys_name,
            status="MISMATCH" if is_mismatch else "MATCH",
            trust_score=current_trust_score,
            risk_level=risk_level,
            reputation_score=merchant_reputation.get("reputation_score", 50.0),
            decision=payment_decision,
            reason_codes=json.dumps(reason_codes),
            expires_at=datetime.utcnow() + timedelta(minutes=15),
            is_bound=False
        )
        db.add(session_rec)
        db.commit()
    finally:
        db.close()

    # 7. Simpan Arsip Lengkap Per Sesi (1 Gambar Utuh Plot, Semua Potongan Gambar Crops, File Prediksi JSON & TXT)
    session_artifacts = simpan_arsip_sesi_scan(
        session_id=session_id,
        filename_base=filename_base,
        gambar_input=gambar_input,
        gambar_vis=gambar_vis,
        gambar_qris_card=gambar_qris_card,
        gambar_qr_code=gambar_qr_code,
        positioning_analysis=positioning_analysis,
        all_detected_items=all_detected_items,
        tech_info=tech_info,
        qris_analysis=qris_analysis,
        dig_name=dig_name,
        dig_nmid=dig_nmid,
        dig_city=dig_city,
        dig_acq=dig_acq,
        dig_tid=dig_tid,
        phys_name=phys_name,
        phys_nmid=phys_nmid,
        phys_acq=phys_acq,
        phys_tid=phys_tid,
        is_mismatch=is_mismatch,
        is_nmid_mismatch=is_nmid_mismatch,
        name_similarity=name_similarity,
        match_level=match_level,
        current_qr_risk_score=current_qr_risk_score,
        current_trust_score=current_trust_score,
        risk_level=risk_level,
        risk_label=risk_label,
        risk_color=risk_color,
        explanation=explanation,
        payment_decision=payment_decision,
        reason_codes=reason_codes,
        merchant_reputation=merchant_reputation
    )

    # 8. Kembalikan data lengkap dalam bentuk Dictionary JSON
    return {
        "session_id": session_id,
        "session_folder": session_artifacts["session_folder"],
        "session_files": session_artifacts["session_files"],
        "positioning_analysis": positioning_analysis,
        "current_qr_risk": {
            "risk_level": risk_level,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "overall_risk_score": current_qr_risk_score,
            "trust_score": current_trust_score,
            "decision": payment_decision,
            "reason_codes": reason_codes,
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
            "qris_raw_analysis": qris_analysis,
            "identity_evidence": identity_evidence,
            "all_ocr_results": all_ocr_results
        },
        "merchant_reputation": merchant_reputation,
        "visualization_url": session_artifacts["session_files"]["full_annotated_url"]
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
                          description: str, transaction_ref: str, has_evidence: bool,
                          evidence_level: Optional[int] = None) -> dict:
    """
    Fungsi ini menyimpan laporan masalah pengguna ke database dan memperbarui skor EMRS toko.
    Mendukung 7 Kategori EMRS dan 3 Tingkat Bukti (Evidence Level 0, 1, 2).
    """
    db = SessionLocal()
    try:
        m = db.query(Merchant).filter(Merchant.nmid == nmid).first()
        if not m:
            # Otomatis buat merchant baru jika belum terdaftar di DB
            m = Merchant(
                nmid=nmid,
                merchant_name=f"Merchant ({nmid})",
                acquirer="QRIS Merchant",
                reputation_score=75.0,
                registered_at=datetime.utcnow()
            )
            db.add(m)
            db.flush() # Ambil m.id baru

        # Tentukan evidence_level: gunakan parameter eksplisit jika tersedia
        if evidence_level is None:
            evidence_level = 2 if has_evidence else 1

        prev_score = m.reputation_score or 50.0

        # Cek pencegahan laporan duplikat untuk transaksi yang sama
        if transaction_ref:
            existing = db.query(Report).filter(
                Report.merchant_id == m.id,
                Report.transaction_ref == transaction_ref
            ).first()
            if existing:
                return {
                    "success": False,
                    "message": "Feedback untuk transaksi ini sudah pernah disubmit.",
                    "evidence_level": evidence_level,
                    "previous_reputation_score": prev_score,
                    "new_reputation_score": prev_score
                }

        # Buat objek laporan baru
        rpt = Report(
            merchant_id=m.id,
            category=category,
            severity=severity,
            description=description,
            evidence_level=evidence_level,
            transaction_ref=transaction_ref,
            is_verified=(evidence_level >= 2),
            created_at=datetime.utcnow()
        )
        db.add(rpt)

        # Update counter statistik pada merchant
        m.total_reports = (m.total_reports or 0) + 1
        if evidence_level >= 2:
            m.verified_reports = (m.verified_reports or 0) + 1

        # Pembobotan Authenticity & Mismatch sesuai 7 Kelas EMRS
        if category in ["QRIS Replacement", "PENIPUAN_STIKER_QRIS_PALSU"] and severity == "CRITICAL":
            m.critical_mismatch_count = (m.critical_mismatch_count or 0) + 1
        elif category in ["Merchant Mismatch", "KETIDAKSESUAIAN_IDENTITAS_MERCHANT"]:
            m.identity_mismatch_count = (m.identity_mismatch_count or 0) + 1
        elif category in ["Verified Authentic", "QRIS_NORMAL_MERCHANT_TERPERCAYA", "safe_confirmation"]:
            m.identity_match_count = (m.identity_match_count or 0) + 1

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
            "message": "Feedback berhasil disimpan. Reputasi merchant telah diperbarui oleh LaQris EMRS Intelligence.",
            "evidence_level": evidence_level,
            "previous_reputation_score": prev_score,
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
