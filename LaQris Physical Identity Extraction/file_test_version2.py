# ==============================================================================
# FILE: file_test_version2.py
# FUNGSI: TrustQR Identity Engine V2 (Sistem Verifikasi Keaslian QRIS Bertingkat)
# ARSITEKTUR:
# 1. Identity Aggregator (Menggabungkan Teks Fisik TrOCR + Metadata QR Digital)
# 2. Normalization Engine (Pembersihan PT, CV, Toko, Warung, Kios, & Simbol)
# 3. Tiered Entity Matching (Level 1: Exact, Level 2: Normalized, Level 3: Fuzzy)
# 4. Skor Kepercayaan OCR & Visual Confidence Integrator
# 5. Output Engine Explainable (Format JSON Lengkap Siap Presentasi Juri)
# GAYA KODE: Dasar Pemrograman / Pemula (Menggunakan Fungsi & Loop Jelas)
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

# Cek apakah komputer memiliki GPU Nvidia (CUDA) atau CPU biasa
PERANGKAT = "cuda" if torch.cuda.is_available() else "cpu"

# Variabel penyimpan model AI (dibuat None dulu, diisi saat pertama kali dijalankan)
MODEL_YOLO_SAYA = None
PROCESSOR_TROCR_SAYA = None
MODEL_TROCR_SAYA = None

# Kamus sederhana untuk mengubah kode NNS Acquirer menjadi nama institusi bank
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

# Daftar kata sebutan badan usaha / toko yang akan dibersihkan di Normalization Layer
SEBUTAN_TOKO = [
    "PT", "CV", "TOKO", "UD", "WARUNG", "KIOS", "TB", "PD", 
    "DISTRIBUTOR", "RESTORAN", "RM", "SHOP", "STORE", "AGEN", "DEPOT"
]


def siapkan_model_yolo_dan_trocr():
    """
    Fungsi ini bertugas memuat model AI (YOLO26 & HuggingFace TrOCR) ke dalam memori.
    Model hanya di-load 1 kali agar eksekusi cepat.
    """
    global MODEL_YOLO_SAYA, PROCESSOR_TROCR_SAYA, MODEL_TROCR_SAYA

    folder_script = os.path.dirname(os.path.abspath(__file__))

    # 1. Cari file bobot YOLO26 hasil pelatihan
    lokasi_model_yolo = os.path.join(folder_script, "Train OCR Model", "runs", "detect", "train", "weights", "best.pt")
    if not os.path.exists(lokasi_model_yolo):
        lokasi_model_yolo = os.path.join(folder_script, "Train OCR Model", "yolo26n.pt")

    if MODEL_YOLO_SAYA is None:
        print("[LOG] Lagi muat model YOLO26 dari:", os.path.basename(lokasi_model_yolo))
        MODEL_YOLO_SAYA = YOLO(lokasi_model_yolo)
        print("[OK] Model YOLO26 berhasil masuk ke memori.")

    # 2. Muat model HuggingFace TrOCR
    if PROCESSOR_TROCR_SAYA is None or MODEL_TROCR_SAYA is None:
        nama_trocr = "microsoft/trocr-base-printed"
        print(f"[LOG] Lagi muat model TrOCR ({nama_trocr}) di {PERANGKAT}...")
        
        try:
            tokenizer = RobertaTokenizer.from_pretrained(nama_trocr)
            image_processor = ViTImageProcessor.from_pretrained(nama_trocr)
            PROCESSOR_TROCR_SAYA = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            MODEL_TROCR_SAYA = VisionEncoderDecoderModel.from_pretrained(nama_trocr).to(PERANGKAT)
            print("[OK] Model TrOCR berhasil masuk ke memori.")
        except Exception as error:
            print(f"[WARNING] Gagal muat {nama_trocr}, nyoba model alternatif: {error}")
            nama_trocr = "microsoft/trocr-base-stage1"
            tokenizer = RobertaTokenizer.from_pretrained(nama_trocr)
            image_processor = ViTImageProcessor.from_pretrained(nama_trocr)
            PROCESSOR_TROCR_SAYA = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            MODEL_TROCR_SAYA = VisionEncoderDecoderModel.from_pretrained(nama_trocr).to(PERANGKAT)
            print("[OK] Model TrOCR alternatif berhasil masuk ke memori.")

    return MODEL_YOLO_SAYA, PROCESSOR_TROCR_SAYA, MODEL_TROCR_SAYA


# ==============================================================================
# MODUL 1: NORMALIZATION ENGINE
# ==============================================================================
def normalisasi_teks_identitas(teks_mentah):
    """
    Fungsi ini bertugas membersihkan string nama toko dari sebutan badan usaha (PT, CV, Toko, Warung),
    tanda baca, dan spasi berlebih sehingga didapatkan nama entitas murni.
    Contoh: 'TOKO BERKAH JAYA.' -> 'BERKAH JAYA' | 'CV BERKAH JAYA' -> 'BERKAH JAYA'
    """
    if teks_mentah is None or teks_mentah == "" or teks_mentah == "Tidak terbaca":
        return ""

    # 1. Ubah ke huruf kapital (UPPERCASE)
    teks_kapital = teks_mentah.upper()

    # 2. Hapus tanda baca dan simbol (hanya sisakan huruf dan angka)
    teks_bersih = re.sub(r'[^A-Z0-9\s]', ' ', teks_kapital)

    # 3. Pecah menjadi kata-kata (tokenization)
    daftar_kata = teks_bersih.split()

    # 4. Hapus sebutan toko (stopwords)
    daftar_kata_murni = []
    for kata in daftar_kata:
        if kata not in SEBUTAN_TOKO:
            daftar_kata_murni.append(kata)

    # 5. Gabungkan kembali kata yang tersisa
    hasil_normalisasi = " ".join(daftar_kata_murni).strip()
    
    # Jika semua kata terhapus (misal hanya berisi 'TOKO'), kembalikan teks bersih asli
    if hasil_normalisasi == "":
        return " ".join(daftar_kata).strip()
        
    return hasil_normalisasi


# ==============================================================================
# MODUL 2: TIERED ENTITY MATCHING ENGINE (FUZZY & EXACT MATCHING)
# ==============================================================================
def hitung_kemiripan_bertingkat(teks_digital, teks_fisik):
    """
    Fungsi ini melakukan pencocokan nama entitas secara bertingkat:
    - Level 1: Exact Match (Sama Persis Mentah) -> Skor 100%
    - Level 2: Normalized Match (Sama Persis Setelah Sebutan Toko Dihapus) -> Skor 100%
    - Level 3: Fuzzy Match (Toleransi Typo Karakter OCR) -> Skor 50-99%
    """
    if teks_digital == "Tidak ditemukan" or teks_fisik == "Tidak terbaca":
        return {
            "level": "UNREAD",
            "rasio": 0.0,
            "cocok": False,
            "keterangan": "Data digital atau fisik tidak terbaca"
        }

    # Level 1: Exact Match (Sama persis mentah)
    if teks_digital.lower().strip() == teks_fisik.lower().strip():
        return {
            "level": "LEVEL_1_EXACT_MATCH",
            "rasio": 100.0,
            "cocok": True,
            "keterangan": "String mentah digital dan fisik 100% sama persis"
        }

    # Normalisasi kedua teks
    norm_digital = normalisasi_teks_identitas(teks_digital)
    norm_fisik = normalisasi_teks_identitas(teks_fisik)

    # Level 2: Normalized Match (Sama persis setelah sebutan PT/CV/Toko dibersihkan)
    if norm_digital == norm_fisik and norm_digital != "":
        return {
            "level": "LEVEL_2_NORMALIZED_MATCH",
            "rasio": 100.0,
            "cocok": True,
            "keterangan": f"Entitas cocok sempurna setelah sebutan toko dibersihkan ('{norm_digital}')"
        }

    # Level 3: Fuzzy Match (RapidFuzz / Levenshtein Ratio) untuk toleransi typo OCR
    rasio_mentah = difflib.SequenceMatcher(None, teks_digital.lower(), teks_fisik.lower()).ratio() * 100
    rasio_norm = difflib.SequenceMatcher(None, norm_digital, norm_fisik).ratio() * 100
    
    # Ambil skor tertinggi antara mentah dan ternormalisasi
    skor_fuzzy_tertinggi = max(rasio_mentah, rasio_norm)

    if (norm_digital in norm_fisik) or (norm_fisik in norm_digital) or (skor_fuzzy_tertinggi >= 65.0):
        return {
            "level": "LEVEL_3_FUZZY_MATCH",
            "rasio": round(skor_fuzzy_tertinggi, 1),
            "cocok": True,
            "keterangan": f"Kemiripan fuzzy tingkatan teks cocok ({skor_fuzzy_tertinggi:.1f}%)"
        }

    return {
        "level": "MISMATCH",
        "rasio": round(skor_fuzzy_tertinggi, 1),
        "cocok": False,
        "keterangan": f"Nama entitas berbeda signifikan ({skor_fuzzy_tertinggi:.1f}%)"
    }


def scan_qr_code_digital(gambar_input):
    """
    Scan dan mengambil teks mentah QR Code menggunakan pyzbar.
    """
    hasil_scan = pyzbar.decode(gambar_input)
    if len(hasil_scan) == 0:
        return None
    return hasil_scan[0].data.decode('utf-8')


def ambil_data_dari_qr_code(teks_qr_mentah):
    """
    Membedah susunan EMVCo QRIS untuk mengambil Nama Toko, NMID (Tag 51-02), Acquirer (Tag 26/9360), dan TID (Tag 62-07).
    """
    print("\n--- [LANGKAH 1] MEMBEDAH ISI TEKS DIGITAL QR CODE ---")
    print(f"  -> RAW STRING QR CODE : '{teks_qr_mentah}'")
    print("  -> RINCIAN STRUKTUR TAG EMVCo:")
    
    nama_toko_digital = "Tidak ditemukan"
    tid_digital = "Tidak ditemukan"
    nmid_digital = "Tidak ditemukan"
    acquirer_digital = "Tidak ditemukan"
    
    indeks = 0
    total_karakter = len(teks_qr_mentah)
    
    # Perulangan pembacaan TLV (Tag-Length-Value)
    while indeks < total_karakter:
        kode_tag = teks_qr_mentah[indeks : indeks + 2]
        panjang_teks = teks_qr_mentah[indeks + 2 : indeks + 4]
        
        if not panjang_teks.isdigit():
            break
            
        ukuran_isi = int(panjang_teks)
        isi_teks = teks_qr_mentah[indeks + 4 : indeks + 4 + ukuran_isi]
        
        # Tag 59: Nama Merchant
        if kode_tag == "59":
            nama_toko_digital = isi_teks
            print(f"     * Tag 59 (Nama Merchant) : '{nama_toko_digital}'")
            
        # Tag 60: Kota Merchant
        elif kode_tag == "60":
            print(f"     * Tag 60 (Kota Merchant) : '{isi_teks}'")

        # Tag 51: Container Resmi QRIS Nasional (ID.CO.QRIS.WWW)
        elif kode_tag == "51":
            sub_indeks = 0
            while sub_indeks < len(isi_teks):
                sub_tag = isi_teks[sub_indeks : sub_indeks + 2]
                sub_panjang_str = isi_teks[sub_indeks + 2 : sub_indeks + 4]
                if not sub_panjang_str.isdigit():
                    break
                sub_panjang = int(sub_panjang_str)
                sub_isi = isi_teks[sub_indeks + 4 : sub_indeks + 4 + sub_panjang]
                
                # Subtag 02 adalah NMID Resmi QRIS
                if sub_tag == "02" and sub_isi.startswith("ID"):
                    nmid_digital = sub_isi
                    print(f"     * Tag 51-02 (NMID Resmi) : '{nmid_digital}'")
                sub_indeks = sub_indeks + 4 + sub_panjang

        # Tag 62: Info Tambahan (Terminal ID di sub-tag 07)
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

    # Fallback pencarian NMID jika Tag 51-02 tidak ditemukan
    if nmid_digital == "Tidak ditemukan":
        pencari_nmid = re.search(r'ID\d{13}', teks_qr_mentah) 
        if pencari_nmid:
            nmid_digital = pencari_nmid.group()
            print(f"     * Cadangan NMID Digital  : '{nmid_digital}'")

    # Cari kode Acquirer (NNS 8 angka)
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


def potong_gambar_pake_yolo(gambar_input, model_yolo, folder_output):
    """
    Memotong posisi komponen fisik stiker QRIS memakai YOLO26 dan menyimpan hasil crop ke folder terpisah.
    Menghitung visual confidence rata-rata dari deteksi bounding box.
    """
    print("\n--- [LANGKAH 2] CARI DAN POTONG KOTAK TULISAN PAKAI YOLO26 ---")
    tinggi_foto, lebar_foto = gambar_input.shape[:2]
    
    if not os.path.exists(folder_output):
        os.makedirs(folder_output, exist_ok=True)
    print(f"  -> Folder Penyimpanan Crop : '{folder_output}'")
    
    hasil_deteksi = model_yolo.predict(gambar_input, conf=0.10, verbose=False)[0]
    daftar_nama_label = hasil_deteksi.names
    
    kotak_terbaik = {}
    for box in hasil_deteksi.boxes:
        id_label = int(box.cls[0].item())
        nama_label = daftar_nama_label[id_label]
        nilai_yakin = float(box.conf[0].item())
        
        minimal_yakin = 0.10
        if nilai_yakin >= minimal_yakin:
            if nama_label not in kotak_terbaik or nilai_yakin > kotak_terbaik[nama_label]['conf']:
                kotak_terbaik[nama_label] = {
                    'box': box,
                    'conf': nilai_yakin
                }

    hasil_potongan_foto = {}
    skor_confidence_visual = {}

    for nama_label, data_kotak in kotak_terbaik.items():
        box = data_kotak['box']
        koordinat = box.xyxy[0].tolist()
        
        x1 = int(koordinat[0])
        y1 = int(koordinat[1])
        x2 = int(koordinat[2])
        y2 = int(koordinat[3])
        
        margin_x = int((x2 - x1) * 0.05)
        margin_y = int((y2 - y1) * 0.05)
        
        posisi_x1 = max(0, x1 - margin_x)
        posisi_y1 = max(0, y1 - margin_y)
        posisi_x2 = min(lebar_foto, x2 + margin_x)
        posisi_y2 = min(tinggi_foto, y2 + margin_y)

        foto_potongan_mentah = gambar_input[posisi_y1:posisi_y2, posisi_x1:posisi_x2]
        
        if nama_label == "qrcode":
            foto_final = foto_potongan_mentah.copy()
            print(f"  -> Ditemukan [{nama_label.upper()}] : Ukuran Asli ({foto_final.shape[1]}x{foto_final.shape[0]} px) - Conf: {data_kotak['conf']*100:.1f}%")
        else:
            foto_final = cv2.resize(foto_potongan_mentah, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            print(f"  -> Ditemukan [{nama_label.upper()}] : Crop & Zoom 1.5x ({foto_final.shape[1]}x{foto_final.shape[0]} px) - Conf: {data_kotak['conf']*100:.1f}%")

        nama_file_hasil = f"crop_zoom_{nama_label.upper()}.jpg"
        path_simpan_file = os.path.join(folder_output, nama_file_hasil)
        cv2.imwrite(path_simpan_file, foto_final)
        
        hasil_potongan_foto[nama_label] = foto_final
        skor_confidence_visual[nama_label] = round(data_kotak['conf'] * 100, 1)

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

    # Fallback jika Acquirer tidak terdeteksi
    if "acquirer" not in hasil_potongan_foto:
        tinggi_potong_atas = int(tinggi_foto * 0.18)
        potongan_atas = gambar_input[0:tinggi_potong_atas, 0:lebar_foto]
        potongan_atas_zoom = cv2.resize(potongan_atas, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        hasil_potongan_foto["acquirer"] = potongan_atas_zoom
        skor_confidence_visual["acquirer"] = 70.0
        
        path_simpan_acquirer = os.path.join(folder_output, "crop_zoom_ACQUIRER.jpg")
        cv2.imwrite(path_simpan_acquirer, potongan_atas_zoom)
        print(f"  -> Cadangan [ACQUIRER]   : Potong Area Atas Foto ({potongan_atas_zoom.shape[1]}x{potongan_atas_zoom.shape[0]} px)")

    return hasil_potongan_foto, skor_confidence_visual


def baca_tulisan_pake_trocr(gambar_potongan, processor, model):
    """
    Membaca tulisan fisik memakai HuggingFace TrOCR.
    """
    if gambar_potongan is None or gambar_potongan.size == 0:
        return ""

    gambar_rgb = cv2.cvtColor(gambar_potongan, cv2.COLOR_BGR2RGB)
    gambar_pil = Image.fromarray(gambar_rgb)

    data_piksel = processor(gambar_pil, return_tensors="pt").pixel_values.to(PERANGKAT)

    with torch.no_grad():
        hasil_tebakan_token = model.generate(data_piksel, max_new_tokens=64)

    teks_bacaan = processor.batch_decode(hasil_tebakan_token, skip_special_tokens=True)[0]
    return teks_bacaan.strip()


def periksa_keaslian_qris_v2(nama_file_gambar):
    """
    Fungsi utama TrustQR Engine V2 yang menghasilkan analisis bertingkat dan output JSON Explainable.
    """
    folder_script = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(nama_file_gambar):
        path_foto = os.path.join(folder_script, nama_file_gambar)
    else:
        path_foto = nama_file_gambar

    if not os.path.exists(path_foto):
        nama_tanpa_ext = os.path.splitext(path_foto)[0]
        daftar_ekstensi = ['.jpeg', '.jpg', '.png']
        for ekstensi in daftar_ekstensi:
            path_coba = nama_tanpa_ext + ekstensi
            if os.path.exists(path_coba):
                path_foto = path_coba
                break

    nama_basemame = os.path.basename(path_foto)
    nama_tanpa_ekstensi = os.path.splitext(nama_basemame)[0]
    folder_output_crop = os.path.join(folder_script, f"hasil_crop_{nama_tanpa_ekstensi}")

    print("==========================================================================")
    print("TRUSTQR IDENTITY ENGINE V2 (VERIFIKASI QRIS BERTINGKAT + EXPLAINABLE JSON)")
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

    # 3. Potong Objek Fisik dengan YOLO26 & Hitung Confidence Visual
    kumpulan_potongan, conf_visual = potong_gambar_pake_yolo(gambar_asli, model_yolo, folder_output_crop)

    print("\n--- [LANGKAH 3] BACA TULISAN FISIK PAKAI HUGGINGFACE TrOCR ---")

    # Baca Nama Merchant Fisik
    nama_fisik = "Tidak terbaca"
    if "nama_merchant" in kumpulan_potongan:
        nama_fisik = baca_tulisan_pake_trocr(kumpulan_potongan["nama_merchant"], processor_trocr, model_trocr)
        print(f"  -> Nama Merchant Fisik : '{nama_fisik}'")

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

        print(f"  -> NMID Fisik          : '{nmid_fisik}'")

    # Baca Acquirer Fisik
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
            
        print(f"  -> Acquirer Fisik      : '{acquirer_fisik}'")

    # Baca Terminal ID Fisik
    tid_fisik = "Tidak terbaca"
    if "tid" in kumpulan_potongan:
        tid_mentah = baca_tulisan_pake_trocr(kumpulan_potongan["tid"], processor_trocr, model_trocr)
        tid_fisik = tid_mentah.upper().replace(" ", "")
        print(f"  -> Terminal ID Fisik  : '{tid_fisik}'")

    # ==============================================================================
    # 4. PENCOCOKAN IDENTITY ENGINE V2 (NORMALISASI + TIERED MATCHING)
    # ==============================================================================
    print("\n--- [LANGKAH 4] EVALUASI ENTITY MATCHING BERTINGKAT ---")

    # Normalisasi Nama Merchant
    norm_merchant_digital = normalisasi_teks_identitas(nama_dig)
    norm_merchant_fisik = normalisasi_teks_identitas(nama_fisik)
    
    print(f"  -> Normalisasi Merchant Digital : '{norm_merchant_digital}'")
    print(f"  -> Normalisasi Merchant Fisik   : '{norm_merchant_fisik}'")

    # Match Merchant Name
    hasil_match_merchant = hitung_kemiripan_bertingkat(nama_dig, nama_fisik)
    print(f"  -> Match Merchant Result      : {hasil_match_merchant['level']} ({hasil_match_merchant['rasio']}%) - {hasil_match_merchant['keterangan']}")

    # Match NMID (Syarat Mutlak Verifikasi Keaslian)
    cocok_nmid = (nmid_dig == nmid_fisik) and (nmid_dig != "Tidak ditemukan")
    print(f"  -> Match NMID Status           : [{'COCOK' if cocok_nmid else 'TIDAK COCOK'}] (Digital: {nmid_dig} | Fisik: {nmid_fisik})")

    # Match Acquirer
    nama_bank_digital = DAFTAR_NAMA_BANK.get(acq_dig, "").lower()
    acq_fisik_kecil = acquirer_fisik.lower()
    cocok_acquirer = False
    if acq_dig != "Tidak ditemukan" and acquirer_fisik != "Tidak terbaca":
        if (acq_dig in acquirer_fisik) or (acquirer_fisik in acq_dig):
            cocok_acquirer = True
        elif nama_bank_digital != "" and ((nama_bank_digital in acq_fisik_kecil) or (acq_fisik_kecil in nama_bank_digital)):
            cocok_acquirer = True

    # Match TID
    cocok_tid = False
    if tid_dig != "Tidak ditemukan" and tid_fisik != "Tidak terbaca":
        rasio_tid = difflib.SequenceMatcher(None, tid_dig.upper(), tid_fisik.upper()).ratio()
        if (tid_dig.upper() in tid_fisik.upper()) or (tid_fisik.upper() in tid_dig.upper()) or (rasio_tid > 0.5):
            cocok_tid = True

    # ==============================================================================
    # 5. KALKULASI SKOR KEPERCAYAAN (TRUST SCORE) & OUTPUT EXPLAINABLE JSON
    # ==============================================================================
    skor_confidence_ocr = conf_visual.get("nama_merchant", 85.0)
    skor_kemiripan_merchant = hasil_match_merchant["rasio"]
    level_pencocokan = hasil_match_merchant["level"]
    
    # Formula Trust Score: (Kemiripan Teks x 70%) + (Skor Confidence Visual OCR x 30%)
    skor_trust_akhir = (skor_kemiripan_merchant * 0.70) + (skor_confidence_ocr * 0.30)
    
    # Penentuan Keputusan Akhir Keaslian secara Dinamis
    if cocok_nmid:
        if level_pencocokan == "LEVEL_1_EXACT_MATCH":
            status_verdict = "SANGAT AMAN (100% TERVERIFIKASI ASLI)"
            penjelasan_ringkas = "NMID cocok sempurna dan nama merchant fisik 100% sama persis dengan digital."
        elif level_pencocokan == "LEVEL_2_NORMALIZED_MATCH":
            status_verdict = "SANGAT AMAN (100% TERVERIFIKASI ASLI)"
            penjelasan_ringkas = f"NMID cocok sempurna dan entitas nama merchant terverifikasi sama ('{norm_merchant_digital}')."
        elif level_pencocokan == "LEVEL_3_FUZZY_MATCH":
            status_verdict = "AMAN DENGAN CATATAN"
            penjelasan_ringkas = f"NMID valid, namun terdapat perbedaan karakter/typo sebesar {skor_kemiripan_merchant:.1f}% pada cetakan fisik."
        else:
            status_verdict = "AMAN DENGAN CATATAN"
            penjelasan_ringkas = "NMID valid, namun tulisan nama merchant pada cetakan fisik kurang terbaca."
    else:
        status_verdict = "BAHAYA (PENIPUAN TERDETEKSI / QRIS PALSU)"
        penjelasan_ringkas = "NMID Digital dan NMID Fisik berbeda atau tidak cocok!"

    # Objek JSON Explainable Siap Presentasi / Konsumsi Frontend UI
    laporan_json_explainable = {
        "file_target": nama_basemame,
        "verdict_status": status_verdict,
        "trust_score": round(skor_trust_akhir, 1),
        "explanation": penjelasan_ringkas,
        "decision_level": hasil_match_merchant["level"],
        "identity_details": {
            "nmid": {
                "digital": nmid_dig,
                "physical": nmid_fisik,
                "is_matched": cocok_nmid
            },
            "merchant_name": {
                "raw_digital": nama_dig,
                "raw_physical": nama_fisik,
                "normalized_digital": norm_merchant_digital,
                "normalized_physical": norm_merchant_fisik,
                "similarity_score_percent": skor_kemiripan_merchant,
                "ocr_confidence_percent": skor_confidence_ocr,
                "is_matched": hasil_match_merchant["cocok"]
            },
            "acquirer": {
                "digital_code": acq_dig,
                "digital_bank_name": DAFTAR_NAMA_BANK.get(acq_dig, "Unknown"),
                "physical_text": acquirer_fisik,
                "is_matched": cocok_acquirer
            },
            "terminal_id": {
                "digital": tid_dig,
                "physical": tid_fisik,
                "is_matched": cocok_tid
            }
        }
    }

    print("\n==========================================================================")
    print("OUTPUT JSON EXPLAINABLE (RINGKASAN KEPUTUSAN UNTUK JURI / UI):")
    print(json.dumps(laporan_json_explainable, indent=2))
    print("==========================================================================")


# ==============================================================================
# EKSEKUSI UTAMA UNTUK 5 GAMBAR TEST
# ==============================================================================
if __name__ == "__main__":
    daftar_file_test = [
        "qris_test1.png",
        "qris_test2.jpeg",
        "qris_test3.jpeg",
        "qris_test4.png",
        "qris_test5.png"
    ]
    
    for foto_uji in daftar_file_test:
        periksa_keaslian_qris_v2(foto_uji)
        print("\n")
