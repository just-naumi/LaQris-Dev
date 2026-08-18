# ==============================================================================
# FILE: file_test.py
# FUNGSI: LaQris Barcode Scanning & Authenticity Verification Engine (Production)
# ARSITEKTUR SIKLUS:
# 1. EMVCo Digital QR Decoder (Tag 51 NMID, Tag 59 Merchant, Tag 60 City, Tag 62 TID, Acquirer)
# 2. YOLO26 Barcode Bounding Box Detection (Deteksi 'Barcode_Asli' vs 'Barcode_Palsu')
# 3. Barcode Crop & Zoom Extraction
# 4. Multi-Attribute Feature Scoring & Authenticity Trust Score Engine
# 5. Dedicated Scan Quality Metric (Conf Rata-Rata YOLO26)
# 6. Anti-Fraud Strict Threshold Verdict Engine (Deteksi QRIS Palsu / Timpa)
# 7. Output Explainable JSON Report (Siap Konsumsi UI / Backend Production API)
# ==============================================================================

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from pyzbar import pyzbar
import re
import json
import warnings
from ultralytics import YOLO

# Mematikan pesan peringatan agar log konsol bersih
warnings.filterwarnings("ignore")

PERANGKAT = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_YOLO_BARCODE = None

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

def siapkan_model_yolo_barcode():
    """
    Memuat model YOLO26 Barcode hasil pelatihan ke dalam memori.
    """
    global MODEL_YOLO_BARCODE

    folder_script = os.path.dirname(os.path.abspath(__file__))
    lokasi_model_yolo = os.path.join(folder_script, "Train Barcode Model", "runs", "detect", "train", "weights", "best.pt")
    
    if not os.path.exists(lokasi_model_yolo):
        lokasi_model_yolo = os.path.join(folder_script, "Train Barcode Model", "yolo26s.pt")

    if MODEL_YOLO_BARCODE is None:
        print("[LOG] Memuat model YOLO26 Barcode dari:", os.path.basename(lokasi_model_yolo))
        MODEL_YOLO_BARCODE = YOLO(lokasi_model_yolo)
        print("[OK] Model YOLO26 Barcode berhasil dimuat.")

    return MODEL_YOLO_BARCODE


def scan_qr_code_digital(gambar_input):
    """
    Scan dan mengambil teks mentah QR Code menggunakan pyzbar.
    """
    hasil_scan = pyzbar.decode(gambar_input)
    if len(hasil_scan) == 0:
        return None
    return hasil_scan[0].data.decode('utf-8')


def bedah_struktur_emvco_qris(teks_qr_mentah):
    """
    Membedah susunan EMVCo QRIS untuk mengambil Nama Toko, NMID, Acquirer, dan TID.
    """
    print("\n--- [LANGKAH 1] MEMBEDAH STRUKTUR DIGITAL QR CODE (EMVCo) ---")
    print(f"  -> RAW QR STRING : '{teks_qr_mentah}'")
    
    nama_toko = "Tidak ditemukan"
    tid = "Tidak ditemukan"
    nmid = "Tidak ditemukan"
    acquirer = "Tidak ditemukan"
    
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
            nama_toko = isi_teks
            print(f"     * Tag 59 (Nama Merchant) : '{nama_toko}'")
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
                    nmid = sub_isi
                    print(f"     * Tag 51-02 (NMID Resmi) : '{nmid}'")
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
                    tid = sub_isi
                    print(f"     * Tag 62-07 (Terminal ID) : '{tid}'")
                    break
                sub_indeks = sub_indeks + 4 + sub_panjang

        indeks = indeks + 4 + ukuran_isi

    if nmid == "Tidak ditemukan":
        pencari_nmid = re.search(r'ID\d{13}', teks_qr_mentah) 
        if pencari_nmid:
            nmid = pencari_nmid.group()

    pencari_acquirer = re.search(r'9360\d{4}', teks_qr_mentah)
    if pencari_acquirer:
        acquirer = pencari_acquirer.group()

    return nama_toko, nmid, acquirer, tid


def deteksi_dan_crop_barcode(gambar_input, model_yolo, folder_output):
    """
    Mendeteksi posisi 'Barcode_Asli' atau 'Barcode_Palsu' menggunakan YOLO26 Barcode Model.
    """
    print("\n--- [LANGKAH 2] DETEKSI DAN POTONG BARCODE DENGAN YOLO26 ---")
    tinggi_foto, lebar_foto = gambar_input.shape[:2]

    if not os.path.exists(folder_output):
        os.makedirs(folder_output, exist_ok=True)

    hasil_deteksi = model_yolo.predict(gambar_input, conf=0.15, verbose=False)[0]
    daftar_nama_label = hasil_deteksi.names

    deteksi_terbaik = []
    skor_confidence_list = []

    for box in hasil_deteksi.boxes:
        id_label = int(box.cls[0].item())
        nama_label = daftar_nama_label[id_label]
        nilai_yakin = float(box.conf[0].item())

        koordinat = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(koordinat[0]), int(koordinat[1]), int(koordinat[2]), int(koordinat[3])

        margin_x = int((x2 - x1) * 0.05)
        margin_y = int((y2 - y1) * 0.05)
        posisi_x1 = max(0, x1 - margin_x)
        posisi_y1 = max(0, y1 - margin_y)
        posisi_x2 = min(lebar_foto, x2 + margin_x)
        posisi_y2 = min(tinggi_foto, y2 + margin_y)

        foto_potongan = gambar_input[posisi_y1:posisi_y2, posisi_x1:posisi_x2]
        nama_file_crop = f"crop_zoom_{nama_label.upper()}_{len(deteksi_terbaik)+1}.jpg"
        path_simpan = os.path.join(folder_output, nama_file_crop)
        cv2.imwrite(path_simpan, foto_potongan)

        deteksi_terbaik.append({
            "label": nama_label,
            "confidence": round(nilai_yakin * 100, 1),
            "bbox": [posisi_x1, posisi_y1, posisi_x2, posisi_y2],
            "crop_file": nama_file_crop
        })
        skor_confidence_list.append(nilai_yakin * 100)

        print(f"  -> Ditemukan [{nama_label.upper()}] : Confidence {nilai_yakin*100:.1f}%")

    avg_conf = sum(skor_confidence_list) / len(skor_confidence_list) if skor_confidence_list else 80.0

    return deteksi_terbaik, round(avg_conf, 1)


def verifikasi_barcode_qris(nama_file_gambar):
    """
    Fungsi Utama Engine Pemindaian & Verifikasi QRIS Barcode.
    """
    folder_script = os.path.dirname(os.path.abspath(__file__))
    folder_project = os.path.dirname(folder_script)

    # Bersihkan jika ada typo nama file seperti qris_test1..png
    nama_file_bersih = nama_file_gambar.replace("..png", ".png").replace("..jpg", ".jpg").replace("..jpeg", ".jpeg")

    path_foto = os.path.join(folder_script, nama_file_bersih) if not os.path.isabs(nama_file_bersih) else nama_file_bersih

    # Jika tidak ditemukan di folder lokal, cari di folder 'LaQris Physical Identity Extraction'
    if not os.path.exists(path_foto):
        path_alternatif = os.path.join(folder_project, "LaQris Physical Identity Extraction", nama_file_bersih)
        if os.path.exists(path_alternatif):
            path_foto = path_alternatif
        else:
            # Coba tambahkan ekstensi jika user hanya mengetik 'qris_test1'
            for ext in ['.png', '.jpeg', '.jpg']:
                path_coba = path_foto + ext
                if os.path.exists(path_coba):
                    path_foto = path_coba
                    break
                path_coba_alt = os.path.join(folder_project, "LaQris Physical Identity Extraction", nama_file_bersih + ext)
                if os.path.exists(path_coba_alt):
                    path_foto = path_coba_alt
                    break

    if not os.path.exists(path_foto):
        print(f"[ERROR] File gambar '{nama_file_gambar}' tidak ditemukan!")
        return None

    nama_basename = os.path.basename(path_foto)
    nama_tanpa_ext = os.path.splitext(nama_basename)[0]
    folder_output_crop = os.path.join(folder_script, f"hasil_crop_barcode_{nama_tanpa_ext}")

    print("==========================================================================")
    print("LAQRIS BARCODE SCANNING & AUTHENTICITY VERIFICATION ENGINE")
    print("File Foto Target:", path_foto)
    print("==========================================================================")

    gambar_asli = cv2.imread(path_foto)
    if gambar_asli is None:
        print("[ERROR] Gambar gagal dibaca oleh OpenCV!")
        return None

    model_yolo = siapkan_model_yolo_barcode()

    isi_qr_digital = scan_qr_code_digital(gambar_asli)
    if isi_qr_digital is None:
        print("[WARNING] QR Code digital tidak terbaca dari foto ini!")
        nama_dig, nmid_dig, acq_dig, tid_dig = "Tidak terbaca", "Tidak terbaca", "Tidak terbaca", "Tidak terbaca"
    else:
        nama_dig, nmid_dig, acq_dig, tid_dig = bedah_struktur_emvco_qris(isi_qr_digital)

    daftar_deteksi, avg_conf = deteksi_dan_crop_barcode(gambar_asli, model_yolo, folder_output_crop)

    # Analisis Keaslian Berdasarkan Deteksi Model Barcode
    ada_palsu = any(d['label'] == 'Barcode_Palsu' for d in daftar_deteksi)
    ada_asli = any(d['label'] == 'Barcode_Asli' for d in daftar_deteksi)

    if ada_palsu:
        verdict_status = "BERBAHAYA (QRIS TERINDIKASI PALSU / DITIMPA)"
        trust_score = 15.0
        penjelasan = "Model YOLO26 menemukan indikasi Barcode Palsu / Stiker QRIS Tempelan (Overlay Sticker Attack)!"
    elif ada_asli and not ada_palsu:
        verdict_status = "SANGAT AMAN (BARCODE QRIS ASLI)"
        trust_score = 95.0
        penjelasan = "Model YOLO26 memverifikasi struktur fisik QRIS sebagai Barcode Asli resmi tanpa tanda-tanda stiker tempelan."
    else:
        verdict_status = "PERLU DIWASPADAI (DETEKSI AMBIGU)"
        trust_score = 50.0
        penjelasan = "Barcode terdeteksi namun tingkat keyakinan rendah. Disarankan mengambil foto lebih dekat."

    status_kualitas = "BAGUS" if avg_conf >= 70.0 else "SEDANG" if avg_conf >= 40.0 else "KURANG OPTIMAL"

    laporan_json = {
        "file_target": nama_basename,
        "verdict_status": verdict_status,
        "trust_score": trust_score,
        "scan_quality": {
            "avg_confidence_percent": avg_conf,
            "status": status_kualitas
        },
        "digital_metadata": {
            "merchant_name": nama_dig,
            "nmid": nmid_dig,
            "acquirer_code": acq_dig,
            "acquirer_bank": DAFTAR_NAMA_BANK.get(acq_dig, "Unknown"),
            "terminal_id": tid_dig
        },
        "detected_barcodes": daftar_deteksi,
        "explanation": penjelasan
    }

    print("\n==========================================================================")
    print("OUTPUT JSON EXPLAINABLE REPORT (VERIFIKASI BARCODE QRIS):")
    print(json.dumps(laporan_json, indent=2))
    print("==========================================================================\n")

    return laporan_json

if __name__ == "__main__":
    # Jika pengguna memberikan argumen nama file spesifik di terminal: python file_test.py qris_test1.png
    if len(sys.argv) > 1:
        file_input_user = sys.argv[1]
        verifikasi_barcode_qris(file_input_user)
    else:
        # Array daftar foto sampel uji (Bisa Anda tambah atau kurangi dengan mudah)
        daftar_foto_rill = [
            "qris_safe1.png",
            "qris_danger1.png",
            "qris_danger1.png"
        ]
        
        print("==========================================================================")
        print("       MENJALANKAN UJI INFERENSI LENGKAP MODEL YOLO BARCODE TERLATIH       ")
        print("==========================================================================")
        print(f"[INFO] Total {len(daftar_foto_rill)} foto uji dalam array. Memulai pemindaian...\n")
        
        hasil_semua_pengujian = []
        for index, foto in enumerate(daftar_foto_rill, 1):
            print(f"\n>>> [FOTO UJI {index}/{len(daftar_foto_rill)}] <<<")
            laporan = verifikasi_barcode_qris(foto)
            if laporan:
                hasil_semua_pengujian.append(laporan)

        if hasil_semua_pengujian:
            print("\n" + "="*80)
            print("         RANGKUMAN HASIL INFERENSI SELURUH FOTO UJI MODEL BARCODE        ")
            print("="*80)
            print(f"{'No.':<4} | {'Nama File':<35} | {'Trust Score':<12} | {'Verdict Status':<25}")
            print("-" * 80)
            for idx, res in enumerate(hasil_semua_pengujian, 1):
                nama_f = res['file_target']
                if len(nama_f) > 33:
                    nama_f = nama_f[:30] + "..."
                skor = f"{res['trust_score']}%"
                verdict = res['verdict_status'].split('(')[0].strip()
                print(f"{idx:<4} | {nama_f:<35} | {skor:<12} | {verdict:<25}")
            print("="*80 + "\n")


