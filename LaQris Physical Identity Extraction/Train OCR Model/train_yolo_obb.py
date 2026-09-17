# ==============================================================================
# SCRIPT: TRAIN YOLO26 SMALL OBB (Oriented Bounding Box untuk OCR QRIS)
# ==============================================================================
# Keunggulan YOLO26-OBB untuk Deteksi Teks QRIS Fisik:
# 1. Rotated Bounding Box: Kotak deteksi tidak lagi tegak datar, melainkan ikut
#    miring (berotasi theta derajat) persis sejajar dengan orientasi teks aslinya.
# 2. Snug-Fit (Tanpa Padding Kosong): Menghilangkan efek "zoom out/mengembang"
#    karena kotak bounding box memeluk teks miring secara rapat dan presisi.
# 3. 4 Titik Sudut Koordinat: Menghasilkan (x1, y1), (x2, y2), (x3, y3), (x4, y4)
#    yang bisa langsung di-warp perspective secara individual jika diperlukan.
# ==============================================================================

import os
import glob
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO

def jalankan_pelatihan_yolo_obb():
    """
    Fungsi utama untuk melatih model YOLO26s-OBB (Oriented Bounding Box)
    menggunakan GPU NVIDIA RTX 3050.
    """
    folder_saat_ini = Path(os.path.dirname(os.path.abspath(__file__)))
    path_data_yaml = folder_saat_ini / "data.yaml"

    print("=================================================================")
    print("      MULAI PELATIHAN MODEL YOLO26 SMALL OBB (yolo26s-obb.pt)    ")
    print("      Oriented Bounding Box (Kotak Miring Sejajar Teks QRIS)    ")
    print("=================================================================")

    # 1. Cek Model Pretrained YOLO26-OBB
    path_model_local = folder_saat_ini / "yolo26s-obb.pt"
    nama_model_dasar = str(path_model_local) if path_model_local.exists() else "yolo26s-obb.pt"
    print(f"\n--- [1/3] Memuat Model Pretrained: {nama_model_dasar} ---")
    
    try:
        model = YOLO(nama_model_dasar)
        print(f"[OK] Berhasil memuat model {nama_model_dasar} (Task: {model.task})")
    except Exception as error:
        print(f"[ERROR] Gagal memuat {nama_model_dasar}: {error}")
        return

    # 2. Cek Hardware Akselerasi GPU
    has_cuda = torch.cuda.is_available()
    device_to_use = 0 if has_cuda else 'cpu'
    print(f"\n--- [Hardware] Status: {'GPU CUDA Aktif' if has_cuda else 'CPU'} ---")
    if has_cuda:
        print(f" -> Device: {torch.cuda.get_device_name(0)}")

    # 3. Memulai Pelatihan YOLO26s-OBB
    print("\n--- [2/3] Memulai Proses Latihan YOLO26s-OBB (100 Epochs) ---")
    print("Fitur Khusus OBB:")
    print(" - Task             : OBB (Oriented Bounding Box / Kotak Miring)")
    print(" - Output Box       : 4 Titik Sudut Rotasi [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]")
    print(" - Cosine LR        : cos_lr=True")
    print(" - Close Mosaic     : Matikan mosaic di 10 epoch terakhir")
    print(" - Batch Size       : 8 (GPU RTX 3050)")
    print("-----------------------------------------------------------------\n")

    hasil_training = model.train(
        data=str(path_data_yaml),
        epochs=100,
        patience=25,
        batch=8,
        imgsz=640,
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        close_mosaic=10,
        weight_decay=0.0005,
        optimizer='auto',
        
        # Augmentasi Rotasi: OBB sangat bagus dilatih dengan variasi sudut derajat
        degrees=15.0,         # Variasi rotasi sudut agar model belajar sudut kemiringan teks
        translate=0.05,
        scale=0.15,
        shear=2.0,
        perspective=0.0002,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        
        # Aturan teks:
        fliplr=0.0,
        flipud=0.0,
        mixup=0.0,
        mosaic=0.5,
        
        device=device_to_use,
        project=str(folder_saat_ini / "runs" / "obb"),
        name="train",
        exist_ok=True,
        save=True,
        plots=True
    )

    print("\n=================================================================")
    print("     PELATIHAN MODEL YOLO26-OBB SELESAI DENGAN SUKSES!           ")
    print("=================================================================")

    # 4. Uji Coba Model OBB pada Foto Test Set (Visualisasi Kotak Miring)
    folder_hasil_run = Path(hasil_training.save_dir)
    path_best_weight = folder_hasil_run / "weights" / "best.pt"

    print("\n--- [3/3] Menjalankan Ujian Akhir pada Data Test Set ---")
    if path_best_weight.exists():
        model_best = YOLO(str(path_best_weight))
        
        # 1. Evaluasi metrik mAP OBB
        print("Menghitung skor rapor mAP50 OBB pada data test...")
        model_best.val(data=str(path_data_yaml), split='test', project=str(folder_hasil_run), name='test_eval', exist_ok=True)
        
        # 2. Prediksi dan gambar kotak miring pada 12 foto uji
        folder_test_img = folder_saat_ini / "test" / "images"
        list_test = list(folder_test_img.glob("*.*"))
        if list_test:
            print(f"Menggambar prediksi kotak miring (OBB) pada {len(list_test)} foto test...")
            model_best.predict(
                source=list_test,
                conf=0.25,
                save=True,
                project=str(folder_hasil_run),
                name="test_predictions",
                exist_ok=True
            )
            print(f"[OK] Hasil visualisasi kotak miring tersimpan di:")
            print(f"     -> {folder_hasil_run / 'test_predictions'}")

    print("\n=================================================================")
    print("     SEMUA PROSES OBB SELESAI! SILAKAN CEK FOTO HASIL PREDIKSI   ")
    print("=================================================================\n")

if __name__ == "__main__":
    jalankan_pelatihan_yolo_obb()
