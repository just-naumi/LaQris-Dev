# ==============================================================================
# FILE: file_test_version1.py
# FUNGSI: TrustQR Final Identity Extraction Engine (Roboflow 12-Label Compatible)
# ARSITEKTUR SIKLUS:
# 1. EMVCo Digital QR Decoder (Tag 59 Merchant, Tag 51 NMID, Tag 26/9360 Acquirer, Tag 62 TID)
# 2. YOLO26 Bounding Box Detection (Menggunakan 12 Label Roboflow Resmi)
# 3. TrOCR Physical Text Extraction (HuggingFace TrOCR model)
# 4. Gambar Full Image dengan Bounding Box & Label Roboflow Resmi (Tanpa Crop)
# 5. Layer 1-4 Multi-Attribute Evidence Fusion & Trust Score Calculation
# ==============================================================================

import os
import cv2
import numpy as np
import torch
from PIL import Image
from pyzbar import pyzbar
import re
import difflib
import json
import warnings
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, RobertaTokenizer

# Mematikan pesan peringatan agar tampilan log console tetap bersih
warnings.filterwarnings("ignore")

# Cek ketersediaan GPU Nvidia (CUDA)
PERANGKAT = "cuda" if torch.cuda.is_available() else "cpu"

# Variabel memori model AI (dibuat None dulu)
MODEL_YOLO_SAYA = None
PROCESSOR_TROCR_SAYA = None
MODEL_TROCR_SAYA = None

# Kamus kode NNS Acquirer ke nama institusi bank/fintech
DAFTAR_NAMA_BANK = {
    "93600014": "BCA",
    "93600009": "BNI",
    "93600008": "MANDIRI",
    "93600002": "BRI",
    "93600114": "LINKAJA",
    "93600153": "SHOPEEPAY",
    "93600914": "GOPAY",
    "93600811": "OVO"
}

# Daftar kata sebutan badan usaha / toko untuk normalisasi
SEBUTAN_TOKO = [
    "PT", "CV", "TOKO", "UD", "WARUNG", "KIOS", "TB", "PD", 
    "DISTRIBUTOR", "RESTORAN", "RM", "SHOP", "STORE", "AGEN", "DEPOT"
]

# Pemetaan alias Roboflow ke kunci internal
PEMETAAN_LABEL_ROBOFLOW = {
    "nama merchant": "nama_merchant",
    "national merchant id": "nmid",
    "dicetak oleh": "acquirer",
    "terminal id": "tid",
    "qr code": "qrcode",
    # Kasus penulisan variasi Roboflow
    "nama_merchant": "nama_merchant",
    "national_merchant_id": "nmid",
    "dicetak_oleh": "acquirer",
    "terminal_id": "tid",
    "qr_code": "qrcode"
}

# Warna visualisasi berdasar indeks label (BGR)
DAFTAR_WARNA_LABEL = [
    (255, 99, 71),   # 0: Cara Pakai QRIS - Tomato
    (255, 165, 0),  # 1: Cek Aplikasi Penyelenggara - Orange
    (30, 144, 255), # 2: Dicetak Oleh - DodgerBlue
    (147, 112, 219),# 3: Logo dan deskripsi QRIS - MediumPurple
    (50, 205, 50),  # 4: Logo GPN - LimeGreen
    (0, 215, 255),  # 5: Nama Merchant - Gold
    (238, 130, 238),# 6: National Merchant ID - Violet
    (0, 0, 255),    # 7: QR Code - Red
    (255, 105, 180),# 8: QrisOCR - HotPink
    (128, 128, 0),  # 9: Slogan - Olive
    (0, 255, 255),  # 10: Terminal ID - Cyan
    (128, 0, 128)   # 11: Versi Cetak - Purple
]


def siapkan_model_yolo_dan_trocr():
    """
    Memuat model AI (YOLO26 & HuggingFace TrOCR) ke dalam memori.
    """
    global MODEL_YOLO_SAYA, PROCESSOR_TROCR_SAYA, MODEL_TROCR_SAYA

    folder_script = os.path.dirname(os.path.abspath(__file__))

    # 1. Cari file bobot YOLO26 hasil pelatihan
    lokasi_model_yolo = os.path.join(folder_script, "Train OCR Model", "runs", "detect", "train", "weights", "best.pt")
    if not os.path.exists(lokasi_model_yolo):
        lokasi_model_yolo = os.path.join(folder_script, "Train OCR Model", "yolo26s.pt")

    if MODEL_YOLO_SAYA is None:
        print("[LOG] Memuat model YOLO26 dari:", os.path.basename(lokasi_model_yolo))
        MODEL_YOLO_SAYA = YOLO(lokasi_model_yolo)
        print("[OK] Model YOLO26 berhasil dimuat ke memori.")

    # 2. Muat model HuggingFace TrOCR
    if PROCESSOR_TROCR_SAYA is None or MODEL_TROCR_SAYA is None:
        nama_trocr = "microsoft/trocr-base-printed"
        print(f"[LOG] Memuat model TrOCR ({nama_trocr}) pada perangkat {PERANGKAT}...")
        
        try:
            tokenizer = RobertaTokenizer.from_pretrained(nama_trocr)
            image_processor = ViTImageProcessor.from_pretrained(nama_trocr)
            PROCESSOR_TROCR_SAYA = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            MODEL_TROCR_SAYA = VisionEncoderDecoderModel.from_pretrained(nama_trocr).to(PERANGKAT)
            print("[OK] Model TrOCR berhasil dimuat ke memori.")
        except Exception as error:
            print(f"[WARNING] Gagal muat {nama_trocr}, mencoba model alternatif: {error}")
            nama_trocr = "microsoft/trocr-base-stage1"
            tokenizer = RobertaTokenizer.from_pretrained(nama_trocr)
            image_processor = ViTImageProcessor.from_pretrained(nama_trocr)
            PROCESSOR_TROCR_SAYA = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            MODEL_TROCR_SAYA = VisionEncoderDecoderModel.from_pretrained(nama_trocr).to(PERANGKAT)
            print("[OK] Model TrOCR alternatif berhasil dimuat ke memori.")

    return MODEL_YOLO_SAYA, PROCESSOR_TROCR_SAYA, MODEL_TROCR_SAYA


# ==============================================================================
# LAYER 1: ENTITY CANONICALIZATION & NORMALIZATION LAYER
# ==============================================================================
def normalisasi_teks_identitas(teks_mentah):
    if teks_mentah is None or teks_mentah == "" or teks_mentah == "Tidak terbaca":
        return ""

    teks_kapital = teks_mentah.upper()
    teks_bersih = re.sub(r'[^A-Z0-9\s]', ' ', teks_kapital)
    daftar_kata = teks_bersih.split()

    daftar_kata_murni = [kata for kata in daftar_kata if kata not in SEBUTAN_TOKO]
    hasil_normalisasi = " ".join(daftar_kata_murni).strip()
    
    if hasil_normalisasi == "":
        return " ".join(daftar_kata).strip()
        
    return hasil_normalisasi


# ==============================================================================
# LAYER 2: MULTI-ATTRIBUTE FEATURE BUILDER
# ==============================================================================
def hitung_skor_fitur_identitas(digital_entity, physical_entity):
    raw_dig = digital_entity.get("merchant_name", "")
    raw_phys = physical_entity.get("merchant_name", "")
    
    norm_dig = normalisasi_teks_identitas(raw_dig)
    norm_phys = normalisasi_teks_identitas(raw_phys)

    if raw_dig.lower().strip() == raw_phys.lower().strip() and raw_dig != "":
        s_name = 100.0
        level_name = "LEVEL_1_EXACT_MATCH"
    elif norm_dig == norm_phys and norm_dig != "":
        s_name = 100.0
        level_name = "LEVEL_2_NORMALIZED_MATCH"
    else:
        rasio_mentah = difflib.SequenceMatcher(None, raw_dig.lower(), raw_phys.lower()).ratio() * 100
        rasio_norm = difflib.SequenceMatcher(None, norm_dig, norm_phys).ratio() * 100
        s_name = max(rasio_mentah, rasio_norm)
        level_name = "LEVEL_3_FUZZY_MATCH" if s_name >= 65.0 else "MISMATCH"

    nmid_dig = digital_entity.get("nmid", "")
    nmid_phys = physical_entity.get("nmid", "")
    
    if nmid_dig != "" and nmid_dig != "Tidak ditemukan" and nmid_dig == nmid_phys:
        s_nmid = 100.0
    else:
        s_nmid = 0.0

    acq_dig = digital_entity.get("acquirer", "")
    acq_phys = physical_entity.get("acquirer", "")
    bank_dig = DAFTAR_NAMA_BANK.get(acq_dig, "").lower()
    acq_phys_low = acq_phys.lower()

    if acq_dig != "" and acq_dig != "Tidak ditemukan" and acq_phys != "Tidak terbaca":
        if (acq_dig in acq_phys) or (acq_phys in acq_dig):
            s_acq = 100.0
        elif bank_dig != "" and ((bank_dig in acq_phys_low) or (acq_phys_low in bank_dig)):
            s_acq = 100.0
        else:
            s_acq = 0.0
    else:
        s_acq = 100.0  # Netral jika tidak terbaca

    tid_dig = digital_entity.get("tid", "")
    tid_phys = physical_entity.get("tid", "")

    if tid_dig != "Tidak ditemukan" and tid_phys != "Tidak terbaca":
        if tid_dig.upper() in tid_phys.upper() or tid_phys.upper() in tid_dig.upper():
            s_tid = 100.0
        else:
            s_tid = 0.0
    else:
        s_tid = 100.0  # Netral

    return {
        "S_name": round(s_name, 1),
        "S_nmid": round(s_nmid, 1),
        "S_acq": round(s_acq, 1),
        "S_tid": round(s_tid, 1),
        "level_name": level_name,
        "norm_digital": norm_dig,
        "norm_physical": norm_phys
    }


# ==============================================================================
# LAYER 3 & 4: EVIDENCE FUSION & SCAN QUALITY
# ==============================================================================
def hitung_pure_evidence_trust_score(scores, ocr_conf_dict):
    w1, w2, w3, w4 = 0.25, 0.50, 0.15, 0.10

    s_name = scores["S_name"]
    s_nmid = scores["S_nmid"]
    s_acq = scores["S_acq"]
    s_tid = scores["S_tid"]

    trust_score = (w1 * s_name) + (w2 * s_nmid) + (w3 * s_acq) + (w4 * s_tid)

    conf_values = [val for val in ocr_conf_dict.values() if val > 0]
    avg_conf_percent = sum(conf_values) / len(conf_values) if len(conf_values) > 0 else 70.0

    if avg_conf_percent >= 70.0:
        status_kualitas = "BAGUS (HIGH_QUALITY)"
        saran_kualitas = "Kualitas visual foto sangat jelas dan terdeteksi sempurna."
    elif avg_conf_percent >= 40.0:
        status_kualitas = "SEDANG (MEDIUM_QUALITY)"
        saran_kualitas = "Kualitas foto cukup baik, tulisan berhasil diekstrak."
    else:
        status_kualitas = "KURANG OPTIMAL (LOW_QUALITY)"
        saran_kualitas = "Kualitas visual deteksi kurang optimal. Disarankan memfoto lebih dekat."

    return {
        "trust_score": round(trust_score, 1),
        "scan_quality": {
            "avg_visual_confidence_percent": round(avg_conf_percent, 1),
            "status": status_kualitas,
            "recommendation": saran_kualitas
        },
        "weights": {"w1_name": w1, "w2_nmid": w2, "w3_acq": w3, "w4_tid": w4}
    }


def evaluasi_skenario_entitas(digital_entity, physical_entity, ocr_conf_dict, nama_skenario=""):
    scores = hitung_skor_fitur_identitas(digital_entity, physical_entity)
    fusion_result = hitung_pure_evidence_trust_score(scores, ocr_conf_dict)
    
    trust_score = fusion_result["trust_score"]
    scan_qual = fusion_result["scan_quality"]

    if scores["S_nmid"] == 100.0 and trust_score >= 85.0 and scores["S_name"] >= 70.0:
        status_verdict = "SANGAT AMAN (100% TERVERIFIKASI ASLI)"
        penjelasan_ringkas = f"Skor keaslian identitas tinggi ({trust_score:.1f}%). Identitas fisik dan digital terverifikasi asli."
    elif scores["S_nmid"] == 100.0 and trust_score >= 65.0 and scores["S_name"] >= 50.0:
        status_verdict = "AMAN DENGAN CATATAN (PERHATIAN)"
        penjelasan_ringkas = f"NMID valid, namun skor keaslian berada di tingkat sedang ({trust_score:.1f}%). Ada perbedaan minor."
    else:
        status_verdict = "MENCURIGAKAN / BAHAYA (SUSPICIOUS / QRIS PALSU)"
        if scores["S_nmid"] == 0.0:
            penjelasan_ringkas = "NMID Digital dan NMID Fisik tidak cocok! Terindikasi stiker QRIS ditimpa penipu."
        elif scores["S_name"] < 50.0:
            penjelasan_ringkas = f"Nama merchant fisik dan digital sangat jauh berbeda ({scores['S_name']}%)! Terindikasi pencurian identitas."
        else:
            penjelasan_ringkas = f"Skor keaslian identitas sangat rendah ({trust_score:.1f}%)."

    return {
        "skenario": nama_skenario,
        "verdict_status": status_verdict,
        "trust_score": trust_score,
        "probabilitas_sameness_P_SameEntity": round(trust_score / 100.0, 2),
        "explanation": penjelasan_ringkas,
        "scan_quality": scan_qual,
        "evidence_breakdown": {
            "S_name_merchant": {
                "score": scores["S_name"],
                "weight": 0.25,
                "matching_level": scores["level_name"],
                "digital_canonical": scores["norm_digital"],
                "physical_canonical": scores["norm_physical"]
            },
            "S_nmid_identifier": {
                "score": scores["S_nmid"],
                "weight": 0.50,
                "digital": digital_entity.get("nmid", ""),
                "physical": physical_entity.get("nmid", ""),
                "is_matched": (scores["S_nmid"] == 100.0)
            },
            "S_acquirer_bank": {
                "score": scores["S_acq"],
                "weight": 0.15,
                "digital_code": digital_entity.get("acquirer", ""),
                "physical_text": physical_entity.get("acquirer", "")
            },
            "S_terminal_id": {
                "score": scores["S_tid"],
                "weight": 0.10,
                "digital": digital_entity.get("tid", ""),
                "physical": physical_entity.get("tid", "")
            }
        }
    }


def scan_qr_code_digital(gambar_input):
    hasil_scan = pyzbar.decode(gambar_input)
    if len(hasil_scan) == 0:
        return None
    return hasil_scan[0].data.decode('utf-8')


def ambil_data_dari_qr_code(teks_qr_mentah):
    print("\n--- [LANGKAH 1] MEMBEDAH ISI TEKS DIGITAL QR CODE ---")
    print(f"  -> RAW STRING QR CODE : '{teks_qr_mentah}'")
    
    nama_toko_digital = "Tidak ditemukan"
    tid_digital = "Tidak ditemukan"
    nmid_digital = "Tidak ditemukan"
    acquirer_digital = "Tidak ditemukan"
    
    indeks = 0
    total_karakter = len(teks_qr_mentah)
    
    while indeks < total_karakter:
        kode_tag = teks_qr_mentah[indeks : indeks + 2]
        panjang_teks = teks_qr_mentah[indeks + 2 : indeks + 4]
        
        if not panjang_teks.isdigit():
            break
            
        ukuran_isi = int(panjang_teks)
        isi_teks = teks_qr_mentah[indeks + 4 : indeks + 4 + ukuran_isi]
        
        if kode_tag == "59":
            nama_toko_digital = isi_teks
            print(f"     * Tag 59 (Nama Merchant) : '{nama_toko_digital}'")
        elif kode_tag == "60":
            print(f"     * Tag 60 (Kota Merchant) : '{isi_teks}'")
        elif kode_tag == "51":
            sub_indeks = 0
            while sub_indeks < len(isi_teks):
                sub_tag = isi_teks[sub_indeks : sub_indeks + 2]
                sub_panjang_str = isi_teks[sub_indeks + 2 : sub_indeks + 4]
                if not sub_panjang_str.isdigit():
                    break
                sub_panjang = int(sub_panjang_str)
                sub_isi = isi_teks[sub_indeks + 4 : sub_indeks + 4 + sub_panjang]
                
                if sub_tag == "02" and sub_isi.startswith("ID"):
                    nmid_digital = sub_isi
                    print(f"     * Tag 51-02 (NMID Resmi) : '{nmid_digital}'")
                sub_indeks = sub_indeks + 4 + sub_panjang
        elif kode_tag == "62":
            sub_indeks = 0
            while sub_indeks < len(isi_teks):
                sub_tag = isi_teks[sub_indeks : sub_indeks + 2]
                sub_panjang_str = isi_teks[sub_indeks + 2 : sub_indeks + 4]
                if not sub_panjang_str.isdigit():
                    break
                sub_panjang = int(sub_panjang_str)
                sub_isi = isi_teks[sub_indeks + 4 : sub_indeks + 4 + sub_panjang]
                
                if sub_tag == "07":
                    tid_digital = sub_isi
                    print(f"     * Tag 62-07 (Terminal ID) : '{tid_digital}'")
                    break
                sub_indeks = sub_indeks + 4 + sub_panjang

        indeks = indeks + 4 + ukuran_isi

    if nmid_digital == "Tidak ditemukan":
        pencari_nmid = re.search(r'ID\d{13}', teks_qr_mentah) 
        if pencari_nmid:
            nmid_digital = pencari_nmid.group()
            print(f"     * Cadangan NMID Digital  : '{nmid_digital}'")

    pencari_acquirer = re.search(r'9360\d{4}', teks_qr_mentah)
    if pencari_acquirer:
        acquirer_digital = pencari_acquirer.group()
        nama_bank = DAFTAR_NAMA_BANK.get(acquirer_digital, "")
        if nama_bank != "":
            print(f"     * Subtag Acquirer ID     : '{acquirer_digital}' (Penyedia: {nama_bank})")
        else:
            print(f"     * Subtag Acquirer ID     : '{acquirer_digital}'")
        
    print("---------------------------------------------------------")
    return nama_toko_digital, nmid_digital, acquirer_digital, tid_digital


def gambar_dan_simpan_visualisasi_penuh(gambar_input, hasil_deteksi, folder_output_vis, nama_file_asli):
    """
    Menggambar seluruh Bounding Box & Label resmi Roboflow langsung pada FOTO ASLI UTUH (tanpa crop).
    """
    if not os.path.exists(folder_output_vis):
        os.makedirs(folder_output_vis, exist_ok=True)

    gambar_visual = gambar_input.copy()
    daftar_nama_label = hasil_deteksi.names

    for box in hasil_deteksi.boxes:
        id_label = int(box.cls[0].item())
        nama_label_resmi = daftar_nama_label.get(id_label, f"Label_{id_label}")
        conf_score = float(box.conf[0].item())

        if conf_score < 0.10:
            continue

        koordinat = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(koordinat[0]), int(koordinat[1]), int(koordinat[2]), int(koordinat[3])

        # Pilih warna berdasarkan ID label
        warna_bgr = DAFTAR_WARNA_LABEL[id_label % len(DAFTAR_WARNA_LABEL)]

        # 1. Gambar Bounding Box
        cv2.rectangle(gambar_visual, (x1, y1), (x2, y2), warna_bgr, 3)

        # 2. Siapkan Teks Label Roboflow & Confidence
        teks_label = f"{nama_label_resmi}: {conf_score * 100:.1f}%"
        skala_font = 0.55
        ketebalan_font = 2

        (lebar_teks, tinggi_teks), baseline = cv2.getTextSize(teks_label, cv2.FONT_HERSHEY_SIMPLEX, skala_font, ketebalan_font)

        # Kotak Latar Belakang Label Teks
        y_label_top = max(y1 - tinggi_teks - 8, 0)
        cv2.rectangle(gambar_visual, (x1, y_label_top), (x1 + lebar_teks + 8, y_label_top + tinggi_teks + baseline + 6), warna_bgr, -1)

        # Teks Putih
        cv2.putText(gambar_visual, teks_label, (x1 + 4, y_label_top + tinggi_teks + 2), cv2.FONT_HERSHEY_SIMPLEX, skala_font, (255, 255, 255), ketebalan_font)

    path_simpan_vis = os.path.join(folder_output_vis, f"visualisasi_LABEL_{nama_file_asli}")
    cv2.imwrite(path_simpan_vis, gambar_visual)
    print(f"  [OK] Foto asli berlabel Roboflow disimpan ke: '{path_simpan_vis}'")


def potong_gambar_pake_yolo(gambar_input, model_yolo, folder_output, folder_output_vis, nama_file_asli):
    """
    Mendeteksi objek memakai 12 Label Roboflow, menyimpan visualisasi gambar utuh, dan memotong area identitas.
    """
    print("\n--- [LANGKAH 2] DETEKSI 12 LABEL ROBOFLOW PAKAI YOLO26 ---")
    tinggi_foto, lebar_foto = gambar_input.shape[:2]
    
    os.makedirs(folder_output, exist_ok=True)
    
    hasil_deteksi = model_yolo.predict(gambar_input, conf=0.10, verbose=False)[0]
    daftar_nama_label = hasil_deteksi.names
    
    # Simpan Visualisasi Gambar UTUH dengan Label Roboflow
    gambar_dan_simpan_visualisasi_penuh(gambar_input, hasil_deteksi, folder_output_vis, nama_file_asli)

    kotak_terbaik = {}
    for box in hasil_deteksi.boxes:
        id_label = int(box.cls[0].item())
        nama_label_resmi = daftar_nama_label.get(id_label, f"Label_{id_label}")
        nilai_yakin = float(box.conf[0].item())
        
        # Petakan ke nama kunci internal jika cocok
        kunci_internal = PEMETAAN_LABEL_ROBOFLOW.get(nama_label_resmi.lower(), nama_label_resmi.lower())
        
        if nilai_yakin >= 0.10:
            if kunci_internal not in kotak_terbaik or nilai_yakin > kotak_terbaik[kunci_internal]['conf']:
                kotak_terbaik[kunci_internal] = {
                    'box': box,
                    'conf': nilai_yakin,
                    'label_resmi': nama_label_resmi
                }

    hasil_potongan_foto = {}
    skor_confidence_visual = {}

    for kunci_internal, data_kotak in kotak_terbaik.items():
        box = data_kotak['box']
        nama_label_resmi = data_kotak['label_resmi']
        koordinat = box.xyxy[0].tolist()
        
        x1, y1, x2, y2 = int(koordinat[0]), int(koordinat[1]), int(koordinat[2]), int(koordinat[3])
        
        margin_x = int((x2 - x1) * 0.05)
        margin_y = int((y2 - y1) * 0.05)
        
        posisi_x1 = max(0, x1 - margin_x)
        posisi_y1 = max(0, y1 - margin_y)
        posisi_x2 = min(lebar_foto, x2 + margin_x)
        posisi_y2 = min(tinggi_foto, y2 + margin_y)

        foto_potongan_mentah = gambar_input[posisi_y1:posisi_y2, posisi_x1:posisi_x2]
        
        if kunci_internal == "qrcode":
            foto_final = foto_potongan_mentah.copy()
            print(f"  -> Ditemukan Label Roboflow [{nama_label_resmi.upper()}] : {foto_final.shape[1]}x{foto_final.shape[0]} px - Conf: {data_kotak['conf']*100:.1f}%")
        else:
            foto_final = cv2.resize(foto_potongan_mentah, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            print(f"  -> Ditemukan Label Roboflow [{nama_label_resmi.upper()}] : Zoom 1.5x ({foto_final.shape[1]}x{foto_final.shape[0]} px) - Conf: {data_kotak['conf']*100:.1f}%")

        nama_file_hasil = f"crop_zoom_{kunci_internal.upper()}.jpg"
        path_simpan_file = os.path.join(folder_output, nama_file_hasil)
        cv2.imwrite(path_simpan_file, foto_final)
        
        hasil_potongan_foto[kunci_internal] = foto_final
        skor_confidence_visual[kunci_internal] = round(data_kotak['conf'] * 100, 1)

    # Fallback jika Nama Merchant tidak terdeteksi
    if "nama_merchant" not in hasil_potongan_foto:
        y_mulai = int(tinggi_foto * 0.10)
        y_selesai = int(tinggi_foto * 0.35)
        potongan_merchant = gambar_input[y_mulai:y_selesai, 0:lebar_foto]
        potongan_merchant_zoom = cv2.resize(potongan_merchant, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        hasil_potongan_foto["nama_merchant"] = potongan_merchant_zoom
        skor_confidence_visual["nama_merchant"] = 70.0
        
        path_simpan_merchant = os.path.join(folder_output, "crop_zoom_NAMA_MERCHANT.jpg")
        cv2.imwrite(path_simpan_merchant, potongan_merchant_zoom)
        print(f"  -> Cadangan [NAMA_MERCHANT] : Potong Area Header Merchant ({potongan_merchant_zoom.shape[1]}x{potongan_merchant_zoom.shape[0]} px)")

    # Fallback jika Acquirer (Dicetak Oleh) tidak terdeteksi
    if "acquirer" not in hasil_potongan_foto:
        tinggi_potong_atas = int(tinggi_foto * 0.18)
        potongan_atas = gambar_input[0:tinggi_potong_atas, 0:lebar_foto]
        potongan_atas_zoom = cv2.resize(potongan_atas, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        hasil_potongan_foto["acquirer"] = potongan_atas_zoom
        skor_confidence_visual["acquirer"] = 70.0
        
        path_simpan_acquirer = os.path.join(folder_output, "crop_zoom_ACQUIRER.jpg")
        cv2.imwrite(path_simpan_acquirer, potongan_atas_zoom)
        print(f"  -> Cadangan [DICETAK OLEH / ACQUIRER] : Potong Area Atas Foto ({potongan_atas_zoom.shape[1]}x{potongan_atas_zoom.shape[0]} px)")

    return hasil_potongan_foto, skor_confidence_visual


def baca_tulisan_pake_trocr(gambar_potongan, processor, model):
    if gambar_potongan is None or gambar_potongan.size == 0:
        return ""

    gambar_rgb = cv2.cvtColor(gambar_potongan, cv2.COLOR_BGR2RGB)
    gambar_pil = Image.fromarray(gambar_rgb)

    data_piksel = processor(gambar_pil, return_tensors="pt").pixel_values.to(PERANGKAT)

    with torch.no_grad():
        hasil_tebakan_token = model.generate(data_piksel, max_new_tokens=64)

    teks_bacaan = processor.batch_decode(hasil_tebakan_token, skip_special_tokens=True)[0]
    return teks_bacaan.strip()


def periksa_keaslian_qris_v4(nama_file_gambar):
    """
    Fungsi utama TrustQR Engine untuk mengecek foto fisik QRIS.
    """
    folder_script = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(nama_file_gambar):
        path_foto = os.path.join(folder_script, nama_file_gambar)
    else:
        path_foto = nama_file_gambar

    if not os.path.exists(path_foto):
        nama_tanpa_ext = os.path.splitext(path_foto)[0]
        for ekstensi in ['.jpeg', '.jpg', '.png']:
            path_coba = nama_tanpa_ext + ekstensi
            if os.path.exists(path_coba):
                path_foto = path_coba
                break

    nama_basemame = os.path.basename(path_foto)
    nama_tanpa_ekstensi = os.path.splitext(nama_basemame)[0]
    folder_output_crop = os.path.join(folder_script, f"hasil_crop_{nama_tanpa_ekstensi}")
    folder_output_vis = os.path.join(folder_script, "hasil_visualisasi_label")

    print("==========================================================================")
    print("TRUSTQR IDENTITY ENGINE V4 (ROBOFLOW 12-LABEL ROBUST ENGINE)")
    print("File Foto Yang Dicek:", path_foto)
    print("==========================================================================")

    gambar_asli = cv2.imread(path_foto)
    if gambar_asli is None:
        print("[ERROR] File gambar tidak bisa dibuka atau tidak ditemukan!")
        return

    # 1. Siapkan model AI
    model_yolo, processor_trocr, model_trocr = siapkan_model_yolo_dan_trocr()

    # 2. Scan & Bedah Data Digital QR Code
    isi_qr_digital = scan_qr_code_digital(gambar_asli)
    if isi_qr_digital is None:
        print("[ERROR] QR Code tidak terbaca di gambar ini!")
        return
        
    nama_dig, nmid_dig, acq_dig, tid_dig = ambil_data_dari_qr_code(isi_qr_digital)

    digital_entity = {
        "merchant_name": nama_dig,
        "nmid": nmid_dig,
        "acquirer": acq_dig,
        "tid": tid_dig
    }

    # 3. Deteksi Objek dengan 12 Label Roboflow & Visualisasi Gambar Utuh
    kumpulan_potongan, conf_visual = potong_gambar_pake_yolo(gambar_asli, model_yolo, folder_output_crop, folder_output_vis, nama_basemame)

    print("\n--- [LANGKAH 3] BACA TULISAN FISIK PAKAI HUGGINGFACE TrOCR ---")

    # Baca Nama Merchant Fisik
    nama_fisik = "Tidak terbaca"
    if "nama_merchant" in kumpulan_potongan:
        nama_fisik = baca_tulisan_pake_trocr(kumpulan_potongan["nama_merchant"], processor_trocr, model_trocr)
        print(f"  -> Nama Merchant Fisik ('Nama Merchant') : '{nama_fisik}'")

    # Baca NMID Fisik
    nmid_fisik = "Tidak terbaca"
    if "nmid" in kumpulan_potongan:
        teks_nmid_mentah = baca_tulisan_pake_trocr(kumpulan_potongan["nmid"], processor_trocr, model_trocr)
        teks_nmid_kapital = teks_nmid_mentah.upper().replace(" ", "")
        
        pencari_nmid_fisik = re.search(r'[I1L][D0O][A-Z0-9]{13}', teks_nmid_kapital)
        if not pencari_nmid_fisik:
            if len(teks_nmid_kapital) >= 15:
                pencari_nmid_fisik = re.search(r'[A-Z0-9]{15}', teks_nmid_kapital)
            
        if pencari_nmid_fisik:
            teks_ditemukan = pencari_nmid_fisik.group()
            tabel_ubah = str.maketrans("ILODSZBGT", "110052867")
            angka_bersih = teks_ditemukan[-13:].translate(tabel_ubah)
            nmid_fisik = "ID" + angka_bersih
        else:
            if len(teks_nmid_kapital) >= 10:
                nmid_fisik = re.sub(r'^.*NMID[:\-\s]*', '', teks_nmid_kapital)

        print(f"  -> NMID Fisik ('National Merchant ID')  : '{nmid_fisik}'")

    # Baca Acquirer Fisik (Dicetak Oleh)
    acquirer_fisik = "Tidak terbaca"
    if "acquirer" in kumpulan_potongan:
        acq_mentah = baca_tulisan_pake_trocr(kumpulan_potongan["acquirer"], processor_trocr, model_trocr)
        acq_bersih = re.sub(r'^.*DICETAK\s*OLEH[:\-\s]*', '', acq_mentah, flags=re.IGNORECASE).strip()
        pencari_acq_angka = re.search(r'9360\d{4}', acq_mentah.upper().replace(" ", ""))
        
        if pencari_acq_angka:
            acquirer_fisik = pencari_acq_angka.group()
        else:
            if acq_bersih != "":
                acquirer_fisik = acq_bersih
            else:
                acquirer_fisik = acq_mentah
            
        print(f"  -> Acquirer Fisik ('Dicetak Oleh')      : '{acquirer_fisik}'")

    # Baca Terminal ID Fisik
    tid_fisik = "Tidak terbaca"
    if "tid" in kumpulan_potongan:
        tid_mentah = baca_tulisan_pake_trocr(kumpulan_potongan["tid"], processor_trocr, model_trocr)
        tid_fisik = tid_mentah.upper().replace(" ", "")
        print(f"  -> Terminal ID Fisik ('Terminal ID')   : '{tid_fisik}'")

    physical_entity = {
        "merchant_name": nama_fisik,
        "nmid": nmid_fisik,
        "acquirer": acquirer_fisik,
        "tid": tid_fisik
    }

    # Evaluasi hasil ekstraksi foto asli
    hasil_eval = evaluasi_skenario_entitas(
        digital_entity,
        physical_entity,
        conf_visual,
        nama_skenario=f"Foto Aktual ({nama_basemame})"
    )

    print("\n==========================================================================")
    print("OUTPUT JSON EXPLAINABLE (HASIL EVALUASI FOTO AKTUAL):")
    print(json.dumps(hasil_eval, indent=2))
    print("==========================================================================")


# ==============================================================================
# EKSEKUSI UTAMA DENGAN ARRAY DAFTAR FOTO TEST
# ==============================================================================
if __name__ == "__main__":
    # Array daftar gambar yang bisa Anda tambah / kurangi dengan mudah
    daftar_foto_rill = [
        "qris_safe1.png",
        "qris_danger1.png",
        "qris_danger2.png"
    ]
    
    print("\n>>> MENJALANKAN PENGUJIAN FOTO AKTUAL DENGAN 12 LABEL ROBOFLOW <<<")
    for foto_uji in daftar_foto_rill:
        periksa_keaslian_qris_v4(foto_uji)
        print("\n")
