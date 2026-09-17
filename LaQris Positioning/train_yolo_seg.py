# ==============================================================================
# SCRIPT PELATIHAN: YOLO26s INSTANCE SEGMENTATION (QRIS POSITIONING)
# ==============================================================================
# Keunggulan YOLO26s-Seg untuk QRIS Positioning:
# 1. Output berupa Masker Poligon Akurat (Pixel-level instance segmentation)
#    mengikuti lekukan fisik, kemiringan stiker, dan batas akurat kartu QRIS.
# 2. Backbone YOLO26s: Keseimbangan sempurna antara kecepatan tinggi dan presisi detail.
# 3. Dilengkapi Anti-Overfitting & Cosine Learning Rate Schedule.
# 4. Otomatis menghasilkan grafik loss/mAP masker dan visualisasi pada Data Test.
# ==============================================================================

import os
import sys
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

# Import script evaluasi & plot segmentasi
from evaluate_and_plot_seg import buat_laporan_dan_plot_lengkap_seg

def main():
    folder_saat_ini = Path(__file__).resolve().parent
    path_data_yaml = folder_saat_ini / "data.yaml"

    print("=================================================================")
    print("   PELATIHAN YOLO26s INSTANCE SEGMENTATION (yolo26s-seg.pt)      ")
    print("                   QRIS POSITIONING                              ")
    print("=================================================================")

    # 1. Verifikasi Hardware GPU
    cuda_tersedia = torch.cuda.is_available()
    device = 0 if cuda_tersedia else "cpu"
    print(f"\n[1/4] Pengecekan Perangkat:")
    if cuda_tersedia:
        nama_gpu = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  - Hardware : GPU ({nama_gpu})")
        print(f"  - Total VRAM: {vram_gb:.2f} GB")
        print(f"  - Target Device: CUDA 0")
    else:
        print(f"  - Hardware : CPU (Peringatan: Latihan akan lebih lambat tanpa GPU)")
        print(f"  - Target Device: cpu")

    # 2. Verifikasi Data Split (Train, Val, Test)
    train_count = len(list((folder_saat_ini / 'train' / 'images').glob('*.*')))
    val_count = len(list((folder_saat_ini / 'valid' / 'images').glob('*.*')))
    test_count = len(list((folder_saat_ini / 'test' / 'images').glob('*.*')))
    print(f"\n[2/4] Verifikasi Dataset:")
    print(f"  - Data Train : {train_count} gambar")
    print(f"  - Data Valid : {val_count} gambar")
    print(f"  - Data Test  : {test_count} gambar")
    print(f"  - File YAML  : {path_data_yaml.name}")

    if train_count == 0 or val_count == 0:
        print("[ERROR] Folder train/images atau valid/images kosong! Mohon periksa dataset.")
        sys.exit(1)

    # 3. Memuat Model YOLO26s-Seg
    weights_lokal = folder_saat_ini / "yolo26s-seg.pt"
    weights_root = folder_saat_ini.parent / "yolo26s-seg.pt"
    
    if weights_lokal.exists():
        path_weights = str(weights_lokal)
    elif weights_root.exists():
        shutil.copy(str(weights_root), str(weights_lokal))
        path_weights = str(weights_lokal)
    else:
        path_weights = "yolo26s-seg.pt"

    print(f"\n[3/4] Memuat Model Backbone: {path_weights}")
    model = YOLO(path_weights)
    print(f"[OK] Model loaded (Task: {model.task})")

    # 4. Memulai Training
    print("\n[4/4] Memulai Pelatihan YOLO26s Instance Segmentation...")
    print("-----------------------------------------------------------------")
    print(" - Model        : YOLO26s-Seg (Instance Segmentation)")
    print(" - Task         : segment")
    print(" - Epochs Max   : 100")
    print(" - Patience     : 20 (Early stopping otomatis jika validasi mandek)")
    print(" - Batch Size   : 8")
    print(" - Image Size   : 640x640")
    print(" - Cosine LR    : Aktif")
    print(" - Output Runs  : runs/segment/train")
    print("-----------------------------------------------------------------\n")

    hasil_training = model.train(
        data=str(path_data_yaml),
        task="segment",
        epochs=100,
        patience=20,
        batch=8,
        imgsz=640,
        device=device,
        workers=2,
        
        # Regularisasi & Optimizer
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        close_mosaic=10,
        
        # Augmentasi untuk variasi sudut foto kartu QRIS
        degrees=10.0,
        translate=0.1,
        scale=0.15,
        shear=2.0,
        perspective=0.0005,
        mosaic=0.5,
        fliplr=0.0,
        
        # Direktori Hasil
        project=str(folder_saat_ini / "runs" / "segment"),
        name="train",
        exist_ok=True,
        save=True
    )

    print("\n=================================================================")
    print("      PELATIHAN SELESAI! MEMULAI EVALUASI & PLOT GRAFIK...       ")
    print("=================================================================")

    # 5. Otomatis Membuat Grafik & Ujian Data Test
    save_dir = getattr(hasil_training, 'save_dir', str(folder_saat_ini / "runs" / "segment" / "train"))
    buat_laporan_dan_plot_lengkap_seg(run_dir=save_dir, path_data_yaml=str(path_data_yaml))

    print("\n[SELESAI] Hasil model terbaik disimpan di:")
    print(f"  -> {os.path.join(save_dir, 'weights', 'best.pt')}")

if __name__ == "__main__":
    main()
