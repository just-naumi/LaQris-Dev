# ==============================================================================
# SCRIPT 2: TRAIN YOLO26 NANO MODEL (Melatih Model YOLO26 Nano untuk QRIS Positioning)
# ==============================================================================
# Tujuan script ini:
# 1. Menjalankan skrip penyiapan dataset (memastikan folder train, valid, test siap).
# 2. Menggunakan model TERBARU: YOLO26 Nano ('yolo26n.pt').
#    Keunggulan Utama YOLO26 Nano untuk QRIS Positioning:
#    a. Sangat ringan (~5.5MB) dan super cepat (<15ms inferensi CPU/Mobile).
#    b. Native End-to-End (NMS-Free): Langsung menentukan bounding box fisik QRIS.
#    c. STAL Support: Optimal mengenali tepian stiker dan frame kartu QRIS.
#    d. Presisi tinggi untuk mengukur framing, margin, dan posisi tengah (centering).
# 3. Menerapkan teknik ANTI-OVERFITTING untuk dataset.
# 4. Setelah pelatihan selesai, otomatis memanggil skrip 'evaluate_and_plot.py'
#    untuk membuat grafik dan visualisasi hasil deteksi uji coba!
# ==============================================================================

import os
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

# Import fungsi pembantu dari script sebelah
from prepare_dataset import pisahkan_dan_siapkan_dataset
from evaluate_and_plot import buat_laporan_dan_plot_lengkap

def jalankan_pelatihan_yolo():
    """
    Fungsi utama untuk melatih model YOLO26 Nano QRIS Positioning dengan parameter anti-overfitting.
    """
    folder_saat_ini = Path(os.path.dirname(os.path.abspath(__file__)))
    path_data_yaml = folder_saat_ini / "data.yaml"
    folder_project_root = folder_saat_ini.parent

    print("=================================================================")
    print("   MULAI PELATIHAN MODEL YOLO26 NANO (yolo26n.pt) QRIS POSITIONING")
    print("=================================================================")

    # 1. LANGKAH PERTAMA: Pastikan dataset terbagi dengan rapi (train, valid, test)
    print("\n--- [Langkah 1/3] Memeriksa & Menyiapkan Pembagian Dataset ---")
    pisahkan_dan_siapkan_dataset()

    # 2. LANGKAH KEDUA: Memuat Model YOLO26 Nano
    nama_model_dasar = "yolo26n.pt"
    path_model_root = folder_project_root / nama_model_dasar
    path_model_lokal = folder_saat_ini / nama_model_dasar

    if not path_model_lokal.exists() and path_model_root.exists():
        print(f"[INFO] Menyalin {nama_model_dasar} dari root project ke folder kerja...")
        shutil.copy(str(path_model_root), str(path_model_lokal))

    target_weights = str(path_model_lokal) if path_model_lokal.exists() else str(path_model_root)
    print(f"\n--- [Langkah 2/3] Memuat Pretrained Model: {target_weights} ---")

    try:
        model = YOLO(target_weights)
        print(f"[OK] Berhasil memuat model {nama_model_dasar}")
    except Exception as error:
        print(f"[Warning] Gagal memuat dari path spesifik: {error}")
        print("Mencoba memuat langsung via identifier yolo26n.pt...")
        try:
            model = YOLO("yolo26n.pt")
            print("[OK] Berhasil memuat model yolo26n.pt")
        except Exception as err2:
            print(f"[ERROR] Gagal memuat model YOLO: {err2}")
            return

    # 3. LANGKAH KETIGA: Memulai Proses Latihan
    print("\n--- [Langkah 3/3] Memulai Proses Latihan YOLO26 Nano dengan Pengaturan Anti-Overfitting ---")
    print("Keunggulan Pengaturan YOLO26 Nano untuk QRIS Positioning:")
    print(" - Model Backbone   : YOLO26 Nano (yolo26n.pt)")
    print(" - End-to-End       : Tanpa NMS post-processing, deteksi fisik & orientasi sangat cepat")
    print(" - Epochs Max       : 50 ronde")
    print(" - Early Stopping   : 15 ronde (Stop otomatis jika val loss mandek)")
    print(" - Hardware         : GPU jika tersedia, CPU hemat daya")
    print("-----------------------------------------------------------------\n")

    hasil_training = model.train(
        # File petunjuk lokasi data
        data=str(path_data_yaml),
        
        # Pengaturan Ronde & Ukuran
        epochs=50,            # 50 ronde latihan optimal untuk 1 kelas positioning
        patience=15,          # Stop otomatis jika 15 ronde berturut-turut val loss tidak membaik
        batch=8,              # Jumlah foto diproses per-batch
        imgsz=640,            # Ukuran gambar standar 640x640 piksel
        
        # Pengaturan Bobot & Regularisasi (Anti-Overfitting)
        weight_decay=0.001,   # Mencegah bobot angka model terlalu ekstrem besar
        dropout=0.1,          # Dropout 10% neuron acak tiap langkah
        warmup_epochs=3.0,    # Pemanasan penyesuaian learning rate
        
        # Augmentasi Data (Rotasi, Perspektif, Pencahayaan untuk simulasi pegang HP)
        degrees=5.0,          # Kemiringan acak (+/- 5 derajat)
        translate=0.05,       # Geser posisi (5%)
        scale=0.1,            # Zoom in / Zoom out (+/- 10%)
        shear=2.0,            # Distorsi sudut lensa
        perspective=0.0001,   # Perspektif kamera
        hsv_h=0.015,          # Variasi warna (Hue)
        hsv_s=0.4,            # Variasi saturasi warna
        hsv_v=0.4,            # Variasi terang-gelap
        
        # Augmentasi Khusus Deteksi Posisi:
        fliplr=0.0,           # Hindari flip horizontal untuk menjaga orientasi fisik QRIS
        mixup=0.0,            # Mematikan campur dua gambar
        mosaic=0.5,           # Penggabungan mosaik ringan
        
        # Tempat Penyimpanan Hasil & Hardware
        device=0 if torch.cuda.is_available() else 'cpu',
        project=str(folder_saat_ini / "runs" / "detect"),
        name="train",
        exist_ok=True,
        save=True
    )

    print("\n=================================================================")
    print("     PELATIHAN MODEL YOLO26 NANO TELAH SELESAI DENGAN SUKSES!    ")
    print("=================================================================")

    # 4. Otomatis Membuat Laporan & Plot Grafik Hasil Pelatihan
    folder_hasil_run = hasil_training.save_dir
    print("\nSekarang membuat grafik hasil latihan & foto pengujian akhir...")
    buat_laporan_dan_plot_lengkap(run_dir=folder_hasil_run, path_data_yaml=str(path_data_yaml))

if __name__ == "__main__":
    jalankan_pelatihan_yolo()
