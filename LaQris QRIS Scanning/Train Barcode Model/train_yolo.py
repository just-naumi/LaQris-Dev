# ==============================================================================
# SCRIPT 2: TRAIN YOLO26 SMALL MODEL (Melatih Model YOLO26 Small untuk Barcode QRIS)
# ==============================================================================
# Tujuan script ini:
# 1. Menjalankan skrip penyiapan dataset (memastikan folder train, valid, test siap).
# 2. Menggunakan model YOLO26 Small ('yolo26s.pt') untuk akurasi ekstra tajam.
# 3. Menerapkan teknik ANTI-OVERFITTING untuk dataset.
# 4. Setelah pelatihan selesai, otomatis memanggil skrip 'evaluate_and_plot.py'
#    untuk membuat grafik dan visualisasi hasil deteksi uji coba!
# ==============================================================================

import os
import torch
from pathlib import Path
from ultralytics import YOLO

from prepare_dataset import pisahkan_dan_siapkan_dataset
from evaluate_and_plot import buat_laporan_dan_plot_lengkap

def jalankan_pelatihan_yolo():
    """
    Fungsi utama untuk melatih model YOLO26s Barcode dengan parameter anti-overfitting.
    """
    folder_saat_ini = Path(os.path.dirname(os.path.abspath(__file__)))
    path_data_yaml = folder_saat_ini / "data.yaml"

    print("=================================================================")
    print("     MULAI PELATIHAN MODEL YOLO26 SMALL (yolo26s.pt) BARCODE QRIS")
    print("=================================================================")

    # 1. Pastikan dataset terbagi rapi
    print("\n--- [Langkah 1/3] Memeriksa & Menyiapkan Pembagian Dataset ---")
    pisahkan_dan_siapkan_dataset()

    # 2. Memuat Pretrained Model YOLO26 Small
    nama_model_dasar = "yolo26s.pt"
    print(f"\n--- [Langkah 2/3] Memuat Pretrained Model: {nama_model_dasar} ---")

    try:
        model = YOLO(nama_model_dasar)
        print(f"[OK] Berhasil memuat model {nama_model_dasar}")
    except Exception as error:
        print(f"[Warning] Gagal memuat {nama_model_dasar}: {error}")
        print("Mencoba memuat model cadangan yolo11s.pt / yolo26n.pt...")
        try:
            model = YOLO("yolo26n.pt")
            print("[OK] Menggunakan model cadangan yolo26n.pt")
        except Exception as err2:
            print(f"[ERROR] Gagal memuat model YOLO: {err2}")
            return

    # 3. Memulai Proses Pelatihan
    print("\n--- [Langkah 3/3] Memulai Proses Latihan YOLO26s dengan Pengaturan Anti-Overfitting ---")
    print("Keunggulan Pengaturan YOLO26s untuk Deteksi Barcode QRIS:")
    print(" - Model Backbone   : YOLO26 Small (yolo26s.pt)")
    print(" - End-to-End       : Tanpa NMS post-processing, deteksi bidang barcode lebih presisi")
    print(" - Epochs Max       : 100 ronde")
    print(" - Early Stopping   : 30 ronde (Stop otomatis jika val loss mandek)")
    print("-----------------------------------------------------------------\n")

    hasil_training = model.train(
        data=str(path_data_yaml),
        epochs=100,
        patience=30,
        batch=8,
        imgsz=640,
        
        weight_decay=0.001,
        dropout=0.1,
        warmup_epochs=3.0,
        
        degrees=5.0,
        translate=0.05,
        scale=0.1,
        shear=2.0,
        perspective=0.0001,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        
        fliplr=0.0,
        mixup=0.0,
        mosaic=0.5,
        
        # Tempat Penyimpanan & Hardware Acceleration (GPU Nvidia RTX)
        device=0 if torch.cuda.is_available() else 'cpu',
        project=str(folder_saat_ini / "runs" / "detect"),
        name="train",
        exist_ok=True,
        save=True
    )

    print("\n=================================================================")
    print("    PELATIHAN MODEL BARCODE YOLO26S TELAH SELESAI DENGAN SUKSES!  ")
    print("=================================================================")

    folder_hasil_run = hasil_training.save_dir
    print("\nSekarang membuat grafik hasil latihan & foto pengujian akhir...")
    buat_laporan_dan_plot_lengkap(run_dir=folder_hasil_run, path_data_yaml=str(path_data_yaml))

if __name__ == "__main__":
    jalankan_pelatihan_yolo()
