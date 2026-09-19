# ==============================================================================
# SCRIPT 2: TRAIN YOLO26 MODEL (Melatih Model YOLO26 Small untuk OCR QRIS)
# ==============================================================================
# Tujuan script ini:
# 1. Menjalankan skrip penyiapan dataset (memastikan folder train, valid, test siap).
# 2. Menggunakan model KUAT & KAPASITAS TINGGI: YOLO26 Small ('yolo26s.pt').
#    Keunggulan Utama YOLO26 Small untuk Deteksi 11 Komponen Fisik QRIS:
#    a. Kapasitas Representasi Lebih Kaya (~20M params): Sangat optimal untuk
#       membedakan 11 kelas berbeda secara presisi (Logo GPN, Logo QRIS,
#       Cara Pakai, Slogan, Nama Merchant, NMID, TID, Acquirer, dll).
#    b. Native End-to-End (NMS-Free): Deteksi bidang teks langsung tanpa NMS,
#       sehingga lokasi bidang teks yang berdekatan tidak saling tertutup/terhapus.
#    c. STAL (Small-Target-Aware Label Assignment): Sangat optimal menemukan teks
#       berukuran kecil seperti angka NMID, TID, dan Acquirer pada fisik sticker QRIS.
#    d. Cosine LR Scheduler & Close Mosaic: Konvergensi optimal di epoch akhir
#       sehingga model belajar kontur asli stiker tanpa noise augmentasi.
#    e. Auto-Deploy: Otomatis menyalin bobot 'best.pt' ke folder backend LaQris!
# ==============================================================================

import os
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

# Import fungsi penyiapan dataset dan evaluasi
from prepare_dataset import pisahkan_dan_siapkan_dataset
from evaluate_and_plot import buat_laporan_dan_plot_lengkap

def jalankan_pelatihan_yolo():
    """
    Fungsi utama untuk melatih model YOLO26s OCR dengan parameter anti-overfitting
    dan optimasi akurasi 11 kelas komponen QRIS.
    """
    # 1. Tentukan lokasi folder kerja
    folder_saat_ini = Path(os.path.dirname(os.path.abspath(__file__)))
    path_data_yaml = folder_saat_ini / "data.yaml"

    print("=================================================================")
    print("      MULAI PELATIHAN MODEL YOLO26 SMALL (yolo26s.pt) DETEKSI QRIS OCR")
    print("=================================================================")

    # 2. LANGKAH PERTAMA: Pastikan dataset terbagi dengan rapi (train, valid, test)
    print("\n--- [Langkah 1/4] Memeriksa & Menyiapkan Pembagian Dataset ---")
    pisahkan_dan_siapkan_dataset()

    # 3. LANGKAH KEDUA: Memuat Model Ultralytics YOLO26 Small
    path_model_local = folder_saat_ini / "yolo26s.pt"
    nama_model_dasar = str(path_model_local) if path_model_local.exists() else "yolo26s.pt"
    print(f"\n--- [Langkah 2/4] Memuat Pretrained Model: {nama_model_dasar} ---")
    
    try:
        model = YOLO(nama_model_dasar)
        print(f"[OK] Berhasil memuat model {nama_model_dasar}")
    except Exception as error:
        print(f"[Warning] Gagal memuat {nama_model_dasar}: {error}")
        print("Mencoba memuat model cadangan yolo26s.pt dari nama file...")
        try:
            model = YOLO("yolo26s.pt")
            print("[OK] Menggunakan model yolo26s.pt")
        except Exception as err2:
            print(f"[ERROR] Gagal memuat model YOLO: {err2}")
            return

    # 4. LANGKAH KEDUA: Cek Hardware (GPU CUDA vs CPU)
    has_cuda = torch.cuda.is_available()
    device_to_use = 0 if has_cuda else 'cpu'
    print(f"\n--- [Hardware] Menggunakan: {'GPU CUDA (NVIDIA)' if has_cuda else 'CPU'} ---")
    if has_cuda:
        print(f" -> GPU Terdeteksi: {torch.cuda.get_device_name(0)}")
    else:
        print(" -> Info: PyTorch terpasang mode CPU. Latihan akan berjalan menggunakan CPU.")

    # 5. LANGKAH KETIGA: Memulai Proses Latihan (Training YOLO26s)
    print("\n--- [Langkah 3/4] Memulai Proses Latihan YOLO26s (11 Kelas OCR) ---")
    print("Keunggulan Pengaturan YOLO26s untuk OCR QRIS:")
    print(" - Model Backbone   : YOLO26 Small (yolo26s.pt) - Kapasitas Kaya untuk 11 Kelas")
    print(" - End-to-End       : NMS-Free Head, deteksi bidang teks lebih presisi")
    print(" - Cosine LR        : cos_lr=True untuk konvergensi halus")
    print(" - Close Mosaic     : Nonaktifkan mosaic di 10 epoch terakhir agar batas teks presisi")
    print(" - STAL Support     : Mengoptimalkan deteksi teks kecil (NMID & TID)")
    print(" - Epochs Max       : 100 ronde (Patience 30 ronde)")
    print(" - Batch Size       : 8 (Stabil & Ramah VRAM/RAM)")
    print("-----------------------------------------------------------------\n")

    # Jalankan pelatihan model YOLO26s
    hasil_training = model.train(
        # File petunjuk lokasi data
        data=str(path_data_yaml),
        
        # Pengaturan Ronde & Ukuran
        epochs=100,           # Maksimal ronde latihan
        patience=30,          # Stop otomatis jika 30 ronde berturut-turut val loss mandek
        batch=8,              # Batch 8 sangat pas untuk model small
        imgsz=640,            # Ukuran input standar 640x640 piksel
        
        # Pengaturan Learning Rate & Konvergensi
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate
        cos_lr=True,          # Cosine annealing schedule
        close_mosaic=10,      # Matikan mosaic 10 epoch terakhir agar model melihat bentuk asli
        weight_decay=0.001,   # Regularisasi bobot L2
        dropout=0.05,         # Sedikit dropout agar mencegah overfitting pada model small
        warmup_epochs=3.0,    # Pemanasan penyesuaian kecepatan belajar di awal
        optimizer='auto',     # Otomatis memilih AdamW / SGD terbaik
        
        # Augmentasi Data yang Realistis untuk Stiker Fisik
        degrees=3.0,          # Putar miring foto ringan (+/- 3 derajat)
        translate=0.05,       # Geser posisi foto ringan (5%)
        scale=0.15,           # Variasi jarak/zoom (+/- 15%)
        shear=1.5,            # Kemiringan sudut kamera ringan
        perspective=0.0002,   # Variasi sudut pandang foto kamera HP
        hsv_h=0.015,          # Variasi warna (Hue) ringan
        hsv_s=0.3,            # Variasi saturasi warna
        hsv_v=0.3,            # Variasi pencahayaan
        
        # Aturan Khusus Teks OCR:
        fliplr=0.0,           # DILARANG flip horizontal (teks tidak boleh terbalik)
        flipud=0.0,           # DILARANG flip vertikal
        mixup=0.0,            # Matikan mixup
        mosaic=0.5,           # Mosaic moderat di awal latihan
        
        # Tempat Penyimpanan Hasil Latihan & Hardware
        device=device_to_use,
        project=str(folder_saat_ini / "runs" / "detect"),
        name="train",
        exist_ok=True,        # Perbarui folder train secara rapi
        save=True,
        plots=True            # Buat grafik metrik, PR curve, dan F1 curve otomatis
    )

    print("\n=================================================================")
    print("     PELATIHAN MODEL YOLO26 TELAH SELESAI DENGAN SUKSES!         ")
    print("=================================================================")

    # 6. Otomatis Membuat Laporan & Plot Grafik Hasil Pelatihan
    print("\n--- [Langkah 4/4] Evaluasi Model & Deployment ke Backend ---")
    folder_hasil_run = hasil_training.save_dir
    print(f"Hasil training tersimpan di: {folder_hasil_run}")
    print("Membuat grafik kurva latihan & visualisasi foto pengujian test set...")
    buat_laporan_dan_plot_lengkap(run_dir=folder_hasil_run, path_data_yaml=str(path_data_yaml))

    # 7. Salin Bobot Terbaik (best.pt) ke Backend LaQris Otomatis
    path_best_weight = Path(folder_hasil_run) / "weights" / "best.pt"
    path_backend_target = folder_saat_ini.parent.parent / "LaQris" / "backend" / "weights" / "yolo_ocr.pt"

    if path_best_weight.exists():
        try:
            path_backend_target.parent.mkdir(parents=True, exist_ok=True)
            # Buat backup bobot lama jika ada
            if path_backend_target.exists():
                path_backup = path_backend_target.parent / "yolo_ocr_backup_old.pt"
                shutil.copy(str(path_backend_target), str(path_backup))
                print(f"[BACKUP] Bobot lama dicadangkan ke: {path_backup.name}")
            
            shutil.copy(str(path_best_weight), str(path_backend_target))
            print(f"[DEPLOYED] Model terbaik (best.pt) BERHASIL dipasang ke backend:")
            print(f"           -> {path_backend_target}")
        except Exception as err_deploy:
            print(f"[Warning] Gagal menyalin bobot otomatis ke backend: {err_deploy}")
            print(f"          Silakan salin manual: '{path_best_weight}' ke '{path_backend_target}'")

    print("\n=================================================================")
    print("     SEMUA PROSES SELESAI! MODEL SIAP DIGUNAKAN DI LAQRIS       ")
    print("=================================================================\n")

# Kode ini berjalan jika script ini dieksekusi langsung
if __name__ == "__main__":
    jalankan_pelatihan_yolo()
