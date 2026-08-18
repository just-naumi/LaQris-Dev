# ==============================================================================
# SCRIPT 2: TRAIN YOLO26 MODEL (Melatih Model YOLO26 Terbaru untuk OCR QRIS)
# ==============================================================================
# Tujuan script ini:
# 1. Menjalankan skrip penyiapan dataset (memastikan folder train, valid, test siap).
# 2. Menggunakan model TERBARU: YOLO26 Nano ('yolo26n.pt').
#    Keunggulan Utama YOLO26 untuk OCR QRIS:
#    a. Native End-to-End (NMS-Free): Deteksi bidang teks langsung tanpa NMS,
#       sehingga lokasi bidang teks yang berdekatan tidak saling tertutup/terhapus.
#    b. STAL (Small-Target-Aware Label Assignment): Sangat optimal menemukan teks
#       berukuran kecil seperti angka NMID dan TID pada fisik sticker QRIS.
#    c. DFL-Free Box Regression: Head deteksi lebih ringan dan lebih presisi.
#    d. Speed UP 43% di CPU: Inferensi ONNX di CPU jauh lebih cepat untuk server/backend.
# 3. Menerapkan teknik ANTI-OVERFITTING untuk dataset terbatas (~37 gambar).
# 4. Setelah pelatihan selesai, otomatis memanggil skrip 'evaluate_and_plot.py'
#    untuk membuat grafik dan visualisasi hasil deteksi uji coba!
# ==============================================================================

import os
import torch
from pathlib import Path
from ultralytics import YOLO

# Import 2 fungsi buatan dari script sebelah
from prepare_dataset import pisahkan_dan_siapkan_dataset
from evaluate_and_plot import buat_laporan_dan_plot_lengkap

def jalankan_pelatihan_yolo():
    """
    Fungsi utama untuk melatih model YOLO26s OCR dengan parameter anti-overfitting.
    """
    # 1. Tentukan lokasi folder kerja
    folder_saat_ini = Path(os.path.dirname(os.path.abspath(__file__)))
    path_data_yaml = folder_saat_ini / "data.yaml"

    print("=================================================================")
    print("      MULAI PELATIHAN MODEL YOLO26 SMALL (yolo26s.pt) DETEKSI QRIS OCR")
    print("=================================================================")

    # 2. LANGKAH PERTAMA: Pastikan dataset terbagi dengan rapi (train, valid, test)
    print("\n--- [Langkah 1/3] Memeriksa & Menyiapkan Pembagian Dataset ---")
    pisahkan_dan_siapkan_dataset()

    # 3. LANGKAH KEDUA: Memuat Model Terbaru Ultralytics YOLO26 Small
    nama_model_dasar = "yolo26s.pt"
    print(f"\n--- [Langkah 2/3] Memuat Pretrained Model Terbaru: {nama_model_dasar} ---")
    
    try:
        model = YOLO(nama_model_dasar)
        print(f"[OK] Berhasil memuat model {nama_model_dasar}")
    except Exception as error:
        print(f"[Warning] Gagal memuat {nama_model_dasar}: {error}")
        print("Mencoba memuat model cadangan yolo26n.pt...")
        try:
            model = YOLO("yolo26n.pt")
            print("[OK] Menggunakan model cadangan yolo26n.pt")
        except Exception as err2:
            print(f"[ERROR] Gagal memuat model YOLO: {err2}")
            return

    # 4. LANGKAH KETIGA: Memulai Proses Latihan (Training YOLO26s)
    print("\n--- [Langkah 3/3] Memulai Proses Latihan YOLO26s dengan Pengaturan Anti-Overfitting ---")
    print("Keunggulan Pengaturan YOLO26s untuk OCR QRIS:")
    print(" - Model Backbone   : YOLO26 Small (yolo26s.pt)")
    print(" - End-to-End       : Tanpa NMS post-processing, deteksi bidang teks lebih presisi")
    print(" - STAL Support     : Mengoptimalkan deteksi teks kecil (NMID & TID)")
    print(" - Epochs Max       : 100 ronde")
    print(" - Early Stopping   : 30 ronde (Stop otomatis jika val loss mandek)")
    print("-----------------------------------------------------------------\n")

    # Jalankan pelatihan model YOLO26s
    hasil_training = model.train(
        # File petunjuk lokasi data
        data=str(path_data_yaml),
        
        # Pengaturan Ronde & Ukuran
        epochs=100,           # Maksimal ronde latihan (100)
        patience=30,          # Stop otomatis jika 30 ronde berturut-turut nilai ujian validasi tidak naik (Anti-Overfitting)
        batch=8,              # Jumlah foto diproses per-langkah (8 foto)
        imgsz=640,            # Ukuran gambar diubah ke standar 640x640 piksel
        
        # Pengaturan Bobot & Regularisasi (Anti-Overfitting)
        weight_decay=0.001,   # Mencegah bobot angka model terlalu ekstrem besar
        dropout=0.1,          # Mematikan 10% neuron acak tiap langkah agar jaringan lebih mandiri
        warmup_epochs=3.0,    # 3 ronde pertama untuk pemanasan penyesuaian kecepatan belajar
        
        # Augmentasi Data
        degrees=5.0,          # Putar miring foto secara acak (+/- 5 derajat)
        translate=0.05,       # Geser posisi foto ringan (5%)
        scale=0.1,            # Zoom in / Zoom out ringan (+/- 10%)
        shear=2.0,            # Kemiringan sudut lensa kamera (2 derajat)
        perspective=0.0001,   # Variasi sudut pandang perspektif foto kamera HP
        hsv_h=0.015,          # Variasi warna (Hue) ringan
        hsv_s=0.4,            # Variasi pekat warna (Saturation)
        hsv_v=0.4,            # Variasi terang-gelap pencahayaan (Value)
        
        # Aturan Penting Khusus OCR Teks:
        fliplr=0.0,           # DILARANG flip horizontal
        mixup=0.0,            # Mematikan campur dua foto
        mosaic=0.5,           # Penggabungan sub-foto ringan
        
        # Tempat Penyimpanan Hasil Latihan & Hardware Acceleration (GPU RTX)
        device=0 if torch.cuda.is_available() else 'cpu',
        project=str(folder_saat_ini / "runs" / "detect"),
        name="train",
        exist_ok=True,        # Jika folder train sudah ada, timpa/perbarui secara rapi
        save=True
    )

    print("\n=================================================================")
    print("     PELATIHAN MODEL YOLO26 TELAH SELESAI DENGAN SUKSES!         ")
    print("=================================================================")

    # 5. Otomatis Membuat Laporan & Plot Grafik Hasil Pelatihan
    folder_hasil_run = hasil_training.save_dir
    print("\nSekarang membuat grafik hasil latihan & foto pengujian akhir...")
    buat_laporan_dan_plot_lengkap(run_dir=folder_hasil_run, path_data_yaml=str(path_data_yaml))

# Kode ini berjalan jika script ini dieksekusi langsung oleh Anda
if __name__ == "__main__":
    jalankan_pelatihan_yolo()
