# ==============================================================================
# FILE: file_test.py
# FUNGSI: Mengecek Keaslian Stiker QRIS (Data Digital vs Tulisan Fisik)
# TEKNOLOGI: YOLO26 (Cari Kotak Tulisan) + HuggingFace TrOCR (Baca Tulisan)
# GAYA KODE: Dasar Pemrograman / Pemula (Menggunakan Loop Biasa Tanpa Syntax Kompleks)
# ==============================================================================

import os
import sys
import cv2  # type: ignore
import numpy as np
import torch  # type: ignore
from PIL import Image
import re
import difflib
import json
import warnings
from ultralytics import YOLO  # type: ignore
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, RobertaTokenizer  # type: ignore

# Penanganan tangguh untuk pyzbar jika DLL C++ runtime (msvcr120.dll/libzbar-64.dll) tidak ditemukan di Windows
try:
    if sys.platform == "win32":
        import site
        for sp in site.getsitepackages():
            pyzbar_dir = os.path.join(sp, "pyzbar")
            if os.path.exists(pyzbar_dir):
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(pyzbar_dir)  # type: ignore
                    except Exception:
                        pass
                os.environ["PATH"] = pyzbar_dir + os.pathsep + os.environ.get("PATH", "")
    from pyzbar import pyzbar  # type: ignore
    PYZBAR_AVAILABLE = True
except Exception:
    PYZBAR_AVAILABLE = False

# Mematikan pesan peringatan yang tidak perlu agar tampilan log tetap bersih
warnings.filterwarnings("ignore")

# Cek apakah komputer menggunakan GPU Nvidia (CUDA) atau CPU biasa
PERANGKAT = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore

# Variabel penyimpan model (dibuat None dulu, diisi saat pertama kali dijalankan)
MODEL_YOLO_SAYA = None
PROCESSOR_TROCR_SAYA = None
MODEL_TROCR_SAYA = None

# Kamus sederhana untuk mengubah kode angka bank/acquirer jadi nama banknya
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


def siapkan_model_yolo_dan_trocr():
    """
    Fungsi ini dipakai untuk mendownload dan menyiapkan model AI ke dalam memori komputer.
    Model hanya di-load 1 kali saja supaya program tidak lambat saat dipanggil berulang kali.
    """
    global MODEL_YOLO_SAYA, PROCESSOR_TROCR_SAYA, MODEL_TROCR_SAYA

    # Ambil lokasi folder tempat file script ini disimpan
    folder_script = os.path.dirname(os.path.abspath(__file__))

    # 1. Cari file model YOLO26 yang sudah kita latih kemarin
    lokasi_model_yolo = os.path.join(folder_script, "Train OCR Model", "runs", "detect", "train", "weights", "best.pt")
    if not os.path.exists(lokasi_model_yolo):
        lokasi_model_yolo = os.path.join(folder_script, "Train OCR Model", "yolo26n.pt")

    if MODEL_YOLO_SAYA is None:
        print("[LOG] Lagi muat model YOLO26 dari:", os.path.basename(lokasi_model_yolo))
        MODEL_YOLO_SAYA = YOLO(lokasi_model_yolo)
        print("[OK] Model YOLO26 berhasil masuk ke memori.")

    # 2. Muat model HuggingFace TrOCR buat pembaca tulisan gambar
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


def scan_qr_code_digital(gambar_input):
    """
    Scan dan mengambil teks mentah QR Code menggunakan pyzbar atau OpenCV QRCodeDetector sebagai fallback.
    """
    if PYZBAR_AVAILABLE:
        try:
            hasil_scan = pyzbar.decode(gambar_input)
            if len(hasil_scan) > 0:
                return hasil_scan[0].data.decode('utf-8')
        except Exception:
            pass

    # Fallback menggunakan OpenCV QRCodeDetector
    try:
        detector = cv2.QRCodeDetector()
        teks_qr, _, _ = detector.detectAndDecode(gambar_input)
        if teks_qr and len(teks_qr) > 0:
            return teks_qr
    except Exception as e:
        print(f"[WARNING] Gagal membaca QR Code: {e}")

    return None


def ambil_data_dari_qr_code(teks_qr_mentah):
    """
    Fungsi ini bertugas membedah susunan teks rahasia QRIS (standar EMVCo)
    untuk mencari: Nama Toko, Terminal ID (TID), NMID (Tag 51-02), dan Kode Bank (Acquirer).
    Menampilkan raw string mentah dan rincian breakdown tag EMVCo.
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
    
    # Perulangan (loop) sederhana untuk membaca format TLV (Tag-Length-Value)
    while indeks < total_karakter:
        kode_tag = teks_qr_mentah[indeks : indeks + 2]
        panjang_teks = teks_qr_mentah[indeks + 2 : indeks + 4]
        
        if not panjang_teks.isdigit():
            break
            
        ukuran_isi = int(panjang_teks)
        isi_teks = teks_qr_mentah[indeks + 4 : indeks + 4 + ukuran_isi]
        
        # Tag 59: Nama Toko / Merchant
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
                
                # Subtag 02 di Tag 51 adalah NMID Resmi QRIS
                if sub_tag == "02" and sub_isi.startswith("ID"):
                    nmid_digital = sub_isi
                    print(f"     * Tag 51-02 (NMID Resmi) : '{nmid_digital}'")
                sub_indeks = sub_indeks + 4 + sub_panjang

        # Tag 62: Info Tambahan (seperti Terminal ID di sub-tag 07)
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

    # Fallback pencarian NMID jika Tag 51-02 tidak ketemu
    if nmid_digital == "Tidak ditemukan":
        pencari_nmid = re.search(r'ID\d{13}', teks_qr_mentah) 
        if pencari_nmid:
            nmid_digital = pencari_nmid.group()
            print(f"     * Cadangan NMID Digital  : '{nmid_digital}'")

    # Cari kode Acquirer (NNS 8 angka) pakai pola regex (9360 + 4 angka)
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
    Fungsi ini bertugas menyuruh YOLO26 mencari kotak bidang tulisan di gambar stiker QRIS.
    Lalu memotong gambar per-bagian dan memperbesarnya 1.5x (zooming) supaya tulisannya jelas.
    Hasil potongan disimpan ke folder terpisah masing-masing.
    """
    print("\n--- [LANGKAH 2] CARI DAN POTONG KOTAK TULISAN PAKAI YOLO26 ---")
    tinggi_foto, lebar_foto = gambar_input.shape[:2]
    
    # Buat folder tersendiri untuk menyimpan hasil crop jika belum ada
    if not os.path.exists(folder_output):
        os.makedirs(folder_output, exist_ok=True)
    print(f"  -> Folder Penyimpanan Crop : '{folder_output}'")
    
    # Jalankan deteksi objek dengan model YOLO26
    hasil_deteksi = model_yolo.predict(gambar_input, conf=0.10, verbose=False)[0]
    daftar_nama_label = hasil_deteksi.names
    
    # Pilih hasil kotak paling yakin (confidence paling tinggi) buat tiap-tiap label
    kotak_terbaik = {}
    for box in hasil_deteksi.boxes:
        id_label = int(box.cls[0].item())
        nama_label = daftar_nama_label[id_label]
        nilai_yakin = float(box.conf[0].item())
        
        # Ambang batas minimal yakin: 0.10 untuk semua kelas agar kandidat terbaik selalu ditangkap
        minimal_yakin = 0.10
        
        if nilai_yakin >= minimal_yakin:
            if nama_label not in kotak_terbaik:
                kotak_terbaik[nama_label] = {
                    'box': box,
                    'conf': nilai_yakin
                }
            else:
                if nilai_yakin > kotak_terbaik[nama_label]['conf']:
                    kotak_terbaik[nama_label] = {
                        'box': box,
                        'conf': nilai_yakin
                    }

    hasil_potongan_foto = {}

    # Loop biasa tanpa sintaks rumit untuk memotong gambar sesuai koordinat kotak YOLO
    for nama_label, data_kotak in kotak_terbaik.items():
        box = data_kotak['box']
        koordinat = box.xyxy[0].tolist()
        
        x1 = int(koordinat[0])
        y1 = int(koordinat[1])
        x2 = int(koordinat[2])
        y2 = int(koordinat[3])
        
        # Tambah margin sedikit (padding 5%) di pinggir kotak biar tulisan tidak terpotong
        margin_x = int((x2 - x1) * 0.05)
        margin_y = int((y2 - y1) * 0.05)
        
        posisi_x1 = max(0, x1 - margin_x)
        posisi_y1 = max(0, y1 - margin_y)
        posisi_x2 = min(lebar_foto, x2 + margin_x)
        posisi_y2 = min(tinggi_foto, y2 + margin_y)

        foto_potongan_mentah = gambar_input[posisi_y1:posisi_y2, posisi_x1:posisi_x2]
        
        # Khusus QR Code: Gunakan ukuran asli (murni) tanpa di-zoom
        if nama_label == "qrcode":
            foto_final = foto_potongan_mentah.copy()
            print(f"  -> Ditemukan [{nama_label.upper()}] : Ukuran Asli ({foto_final.shape[1]}x{foto_final.shape[0]} px)")
        else:
            # Khusus Tulisan Teks: Perbesar 1.5x (Zooming pas, tidak kebesaran)
            foto_final = cv2.resize(foto_potongan_mentah, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            print(f"  -> Ditemukan [{nama_label.upper()}] : Crop & Zoom 1.5x ({foto_final.shape[1]}x{foto_final.shape[0]} px)")

        # Simpan file potongan ke folder tersendiri untuk file gambar ini
        nama_file_hasil = f"crop_zoom_{nama_label.upper()}.jpg"
        path_simpan_file = os.path.join(folder_output, nama_file_hasil)
        cv2.imwrite(path_simpan_file, foto_final)
        
        hasil_potongan_foto[nama_label] = foto_final

    # Jika label Nama Merchant tidak ditemukan YOLO, potong area tengah-atas (antara 10% s/d 35% tinggi foto)
    if "nama_merchant" not in hasil_potongan_foto:
        y_mulai = int(tinggi_foto * 0.10)
        y_selesai = int(tinggi_foto * 0.35)
        potongan_merchant = gambar_input[y_mulai:y_selesai, 0:lebar_foto]
        potongan_merchant_zoom = cv2.resize(potongan_merchant, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        hasil_potongan_foto["nama_merchant"] = potongan_merchant_zoom
        
        path_simpan_merchant = os.path.join(folder_output, "crop_zoom_NAMA_MERCHANT.jpg")
        cv2.imwrite(path_simpan_merchant, potongan_merchant_zoom)
        print(f"  -> Cadangan [NAMA_MERCHANT] : Potong Area Header Merchant ({potongan_merchant_zoom.shape[1]}x{potongan_merchant_zoom.shape[0]} px)")

    # Jika label Acquirer (bank) tidak ditemukan YOLO, potong area paling atas gambar (18% tinggi foto)
    if "acquirer" not in hasil_potongan_foto:
        tinggi_potong_atas = int(tinggi_foto * 0.18)
        potongan_atas = gambar_input[0:tinggi_potong_atas, 0:lebar_foto]
        potongan_atas_zoom = cv2.resize(potongan_atas, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        hasil_potongan_foto["acquirer"] = potongan_atas_zoom
        
        path_simpan_acquirer = os.path.join(folder_output, "crop_zoom_ACQUIRER.jpg")
        cv2.imwrite(path_simpan_acquirer, potongan_atas_zoom)
        print(f"  -> Cadangan [ACQUIRER]   : Potong Area Atas Foto ({potongan_atas_zoom.shape[1]}x{potongan_atas_zoom.shape[0]} px)")

    return hasil_potongan_foto

    return hasil_potongan_foto


def baca_tulisan_pake_trocr(gambar_potongan, processor, model):
    """
    Fungsi ini bertugas menyuruh TrOCR membaca tulisan huruf/angka yang ada di potongan gambar.
    """
    if gambar_potongan is None:
        return ""
    if gambar_potongan.size == 0:
        return ""

    # Ubah format warna OpenCV (BGR) jadi format warna PIL Image (RGB)
    gambar_rgb = cv2.cvtColor(gambar_potongan, cv2.COLOR_BGR2RGB)
    gambar_pil = Image.fromarray(gambar_rgb)

    # Proses gambar jadi data angka piksel buat dimasukkan ke TrOCR
    data_piksel = processor(gambar_pil, return_tensors="pt").pixel_values.to(PERANGKAT)

    # Minta model TrOCR menebak tulisan yang ada di gambar
    with torch.no_grad():
        hasil_tebakan_token = model.generate(data_piksel, max_new_tokens=64)

    # Ubah tebakan angka token jadi huruf/teks biasa
    teks_bacaan = processor.batch_decode(hasil_tebakan_token, skip_special_tokens=True)[0]
    return teks_bacaan.strip()


def periksa_keaslian_qris(nama_file_gambar):
    """
    Fungsi utama untuk mengecek dan mencocokkan data digital vs data fisik stiker QRIS.
    Masing-masing gambar akan disimpan hasil crop-nya ke folder terpisah.
    """
    folder_script = os.path.dirname(os.path.abspath(__file__))
    
    # Tentukan path file gambar yang mau dicek
    if not os.path.isabs(nama_file_gambar):
        path_foto = os.path.join(folder_script, nama_file_gambar)
    else:
        path_foto = nama_file_gambar

    # Jika file .png tidak ketemu, coba otomatis cari file ber-ekstensi .jpeg atau .jpg
    if not os.path.exists(path_foto):
        nama_tanpa_ext = os.path.splitext(path_foto)[0]
        daftar_ekstensi = ['.jpeg', '.jpg', '.png']
        for ekstensi in daftar_ekstensi:
            path_coba = nama_tanpa_ext + ekstensi
            if os.path.exists(path_coba):
                path_foto = path_coba
                break

    # Jika file gambar masih belum ketemu, cari file gambar apa saja di folder test
    if not os.path.exists(path_foto):
        folder_uji = os.path.join(folder_script, "Train OCR Model", "test", "images")
        
        daftar_foto_test = []
        if os.path.exists(folder_uji):
            for file_nama in os.listdir(folder_uji):
                file_nama_kecil = file_nama.lower()
                if file_nama_kecil.endswith('.jpg') or file_nama_kecil.endswith('.png') or file_nama_kecil.endswith('.jpeg'):
                    path_lengkap_file = os.path.join(folder_uji, file_nama)
                    daftar_foto_test.append(path_lengkap_file)
            
            if len(daftar_foto_test) > 0:
                path_foto = daftar_foto_test[0]

    # Ambil nama file tanpa ekstensi untuk penamaan folder hasil crop terpisah
    nama_basemame = os.path.basename(path_foto)
    nama_tanpa_ekstensi = os.path.splitext(nama_basemame)[0]
    folder_output_crop = os.path.join(folder_script, f"hasil_crop_{nama_tanpa_ekstensi}")

    print("==========================================================================")
    print("SISTEM VERIFIKASI KEASLIAN QRIS (YOLO26 DETEKSI + HUGGINGFACE TrOCR)")
    print("File Foto Yang Dicek:", path_foto)
    print("==========================================================================")

    # Baca file gambar dengan cv2 (OpenCV)
    gambar_asli = cv2.imread(path_foto)
    if gambar_asli is None:
        print("[ERROR] File gambar tidak bisa dibuka atau tidak ditemukan!")
        return

    # 1. Siapkan model YOLO26 & TrOCR
    model_yolo, processor_trocr, model_trocr = siapkan_model_yolo_dan_trocr()

    # 2. Scan dan bedah isi QR Code digital
    isi_qr_digital = scan_qr_code_digital(gambar_asli)
    if isi_qr_digital is None:
        print("[ERROR] QR Code tidak terbaca di gambar ini!")
        return
        
    nama_dig, nmid_dig, acq_dig, tid_dig = ambil_data_dari_qr_code(isi_qr_digital)

    # 3. Potong area tulisan di foto fisik memakai YOLO26 (Simpan ke folder terpisah)
    kumpulan_potongan = potong_gambar_pake_yolo(gambar_asli, model_yolo, folder_output_crop)

    print("\n--- [LANGKAH 3] BACA TULISAN FISIK PAKAI HUGGINGFACE TrOCR ---")

    # Baca Tulisan Nama Merchant Fisik
    nama_fisik = "Tidak terbaca"
    if "nama_merchant" in kumpulan_potongan:
        nama_fisik = baca_tulisan_pake_trocr(kumpulan_potongan["nama_merchant"], processor_trocr, model_trocr)
        print(f"  -> Nama Merchant Fisik : '{nama_fisik}'")

    # Baca Tulisan NMID Fisik
    nmid_fisik = "Tidak terbaca"
    if "nmid" in kumpulan_potongan:
        teks_nmid_mentah = baca_tulisan_pake_trocr(kumpulan_potongan["nmid"], processor_trocr, model_trocr)
        teks_nmid_kapital = teks_nmid_mentah.upper().replace(" ", "")
        
        # Bersihkan typo huruf OCR (seperti huruf I, L, O diubah jadi angka 1, 0)
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

    # Baca Tulisan Acquirer (Bank) Fisik
    acquirer_fisik = "Tidak terbaca"
    if "acquirer" in kumpulan_potongan:
        acq_mentah = baca_tulisan_pake_trocr(kumpulan_potongan["acquirer"], processor_trocr, model_trocr)
        
        # Hapus kata prefiks seperti "DICETAK OLEH :" jika ada
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

    # Baca Tulisan Terminal ID (TID) Fisik
    tid_fisik = "Tidak terbaca"
    if "tid" in kumpulan_potongan:
        tid_mentah = baca_tulisan_pake_trocr(kumpulan_potongan["tid"], processor_trocr, model_trocr)
        tid_fisik = tid_mentah.upper().replace(" ", "")
        print(f"  -> Terminal ID Fisik  : '{tid_fisik}'")

    # 4. Pencocokan & Hasil Ujian Akhir
    print("\n--- [LANGKAH 4] PENCOCOKAN DATA DIGITAL VS TULISAN FISIK ---")

    # Cocokkan Nama Merchant
    rasio_nama = difflib.SequenceMatcher(None, nama_dig.lower(), nama_fisik.lower()).ratio()
    cocok_nama = False
    if (nama_dig.lower() in nama_fisik.lower()) or (nama_fisik.lower() in nama_dig.lower()) or (rasio_nama > 0.6):
        cocok_nama = True

    tampilan_nama_fisik = "Tidak terbaca"
    if nama_fisik != "Tidak terbaca":
        persen_nama = int(rasio_nama * 100)
        tampilan_nama_fisik = f"{nama_fisik[:20]} ({persen_nama}%)"

    status_nama = "TIDAK COCOK"
    if cocok_nama:
        status_nama = "COCOK"
    print(f"1. NAMA MERCHANT  | Digital: {nama_dig[:20]:<20} | Fisik: {tampilan_nama_fisik:<20} | Hasil: [{status_nama}]")

    # Cocokkan NMID
    cocok_nmid = False
    if (nmid_dig == nmid_fisik) and (nmid_dig != "Tidak ditemukan"):
        cocok_nmid = True

    status_nmid = "TIDAK COCOK"
    if cocok_nmid:
        status_nmid = "COCOK"
    print(f"2. NMID           | Digital: {nmid_dig:<20} | Fisik: {nmid_fisik:<20} | Hasil: [{status_nmid}]")

    # Cocokkan Acquirer (Bank)
    nama_bank_digital = DAFTAR_NAMA_BANK.get(acq_dig, "").lower()
    acq_fisik_kecil = acquirer_fisik.lower()
    
    cocok_acquirer = False
    if acq_dig != "Tidak ditemukan" and acquirer_fisik != "Tidak terbaca":
        if (acq_dig in acquirer_fisik) or (acquirer_fisik in acq_dig):
            cocok_acquirer = True
        else:
            if nama_bank_digital != "":
                if (nama_bank_digital in acq_fisik_kecil) or (acq_fisik_kecil in nama_bank_digital):
                    cocok_acquirer = True

    status_acquirer = "TIDAK TERBACA"
    if cocok_acquirer:
        status_acquirer = "COCOK"
    print(f"3. ACQUIRER (NNS) | Digital: {acq_dig:<20} | Fisik: {acquirer_fisik:<20} | Hasil: [{status_acquirer}]")

    # Cocokkan Terminal ID (TID)
    cocok_tid = False
    status_tid = "TIDAK TERBACA"
    
    if tid_dig != "Tidak ditemukan" and tid_fisik != "Tidak terbaca":
        rasio_tid = difflib.SequenceMatcher(None, tid_dig.upper(), tid_fisik.upper()).ratio()
        if (tid_dig.upper() in tid_fisik.upper()) or (tid_fisik.upper() in tid_dig.upper()) or (rasio_tid > 0.5):
            cocok_tid = True
            status_tid = "COCOK"
    elif tid_dig == "Tidak ditemukan" and tid_fisik != "Tidak terbaca":
        status_tid = "INFO: HANYA DI FISIK"

    print(f"4. TERMINAL ID    | Digital: {tid_dig:<20} | Fisik: {tid_fisik:<20} | Hasil: [{status_tid}]")

    print("\n==========================================================================")
    
    # Hitung jumlah total skor pendukung
    total_skor_pendukung = 0
    if cocok_nama:
        total_skor_pendukung = total_skor_pendukung + 1
    if cocok_acquirer:
        total_skor_pendukung = total_skor_pendukung + 1
    if cocok_tid:
        total_skor_pendukung = total_skor_pendukung + 1

    # Keputusan Akhir: NMID Digital Harus Sama Persis dengan NMID Fisik
    if cocok_nmid:
        if total_skor_pendukung >= 2:
            print("[STATUS KEPUTUSAN] SANGAT AMAN (100% TERVERIFIKASI ASLI)")
            print("Penjelasan: NMID cocok sempurna dan data pendukung fisik terverifikasi.")
        else:
            print("[STATUS KEPUTUSAN] AMAN DENGAN CATATAN")
            print("Penjelasan: NMID valid, tapi ada beberapa tulisan fisik pendukung yang buram.")
    else:
        print("[STATUS KEPUTUSAN] BAHAYA (PENIPUAN TERDETEKSI / QRIS PALSU)")
        print("Penjelasan: NMID Digital dan NMID Fisik berbeda atau tidak cocok!")

    print("==========================================================================")


# ==============================================================================
# EKSEKUSI PENGECEKAN UTAMA UNTUK 5 FOTO GAMBAR TEST
# ==============================================================================
if __name__ == "__main__":
    daftar_file_test = [
        "qris_test1.png",
        "qris_test2.jpeg",
        "qris_test3.jpeg",
        "qris_test4.png",
        "qris_test5.png"
    ]
    
    for nama_foto in daftar_file_test:
        periksa_keaslian_qris(nama_foto)
        print("\n")