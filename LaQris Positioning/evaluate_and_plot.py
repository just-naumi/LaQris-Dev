# ==============================================================================
# SCRIPT 3: EVALUATE & PLOT (Visualisasi & Evaluasi Hasil Pelatihan Model Positioning)
# ==============================================================================
# Tujuan script ini:
# 1. Membaca file 'results.csv' yang berisi catatan statistik latihan model.
# 2. Menggambar 4 Grafik Utama (Loss Curves & Metrik Keberhasilan):
#    - Box Loss (Kesalahan posisi bounding box QRIS Positioning)
#    - Class Loss (Kesalahan klasifikasi model)
#    - Precision & Recall
#    - mAP Score (Mean Average Precision: mAP@50 & mAP@50-95)
# 3. Menghasilkan Diagram Batang Per-Kelas (Performa untuk kelas Qris-Positioning).
# 4. Menguji model jadi pada Data Test (Ujian Akhir) dan menggambar kotak deteksi di atas foto!
# ==============================================================================

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from ultralytics import YOLO

# Atur tampilan grafik agar bersih dan estetis
sns.set_theme(style="darkgrid")
plt.rcParams['font.sans-serif'] = 'Arial'

def cari_folder_training_terbaru(folder_runs):
    """
    Mencari folder hasil latihan YOLO yang paling baru dibuat.
    """
    daftar_folder = glob.glob(os.path.join(folder_runs, "train*"))
    if not daftar_folder:
        return None
    daftar_folder.sort(key=os.path.getmtime, reverse=True)
    return daftar_folder[0]

def buat_grafik_loss_dan_metrik(path_csv_results, folder_tujuan):
    """
    Membaca data statistik latihan (csv) dan menggambar 4 grafik performa.
    """
    if not os.path.exists(path_csv_results):
        print(f"[Warning] File catatan statistik {path_csv_results} belum ada.")
        return

    tabel_metrik = pd.read_csv(path_csv_results)
    tabel_metrik.columns = [nama_kolom.strip() for nama_kolom in tabel_metrik.columns]

    fig, kumpulan_grafik = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Grafik Performa Latihan Model YOLO26 Nano QRIS Positioning', fontsize=16, fontweight='bold', y=0.98)

    daftar_epoch = tabel_metrik['epoch'] if 'epoch' in tabel_metrik.columns else range(1, len(tabel_metrik) + 1)

    # 1. Bounding Box Loss
    g1 = kumpulan_grafik[0, 0]
    if 'train/box_loss' in tabel_metrik.columns:
        g1.plot(daftar_epoch, tabel_metrik['train/box_loss'], label='Loss Kotak (Train)', color='#2b5c8f', linewidth=2)
    if 'val/box_loss' in tabel_metrik.columns:
        g1.plot(daftar_epoch, tabel_metrik['val/box_loss'], label='Loss Kotak (Valid)', color='#e74c3c', linewidth=2, linestyle='--')
    g1.set_title('1. Error Posisi Kotak (Box Loss)', fontsize=12, fontweight='bold')
    g1.set_xlabel('Epoch (Ronde Latihan)')
    g1.set_ylabel('Nilai Loss (Makin Kecil Makin Bagus)')
    g1.legend()

    # 2. Classification Loss
    g2 = kumpulan_grafik[0, 1]
    if 'train/cls_loss' in tabel_metrik.columns:
        g2.plot(daftar_epoch, tabel_metrik['train/cls_loss'], label='Loss Label (Train)', color='#2b5c8f', linewidth=2)
    if 'val/cls_loss' in tabel_metrik.columns:
        g2.plot(daftar_epoch, tabel_metrik['val/cls_loss'], label='Loss Label (Valid)', color='#e74c3c', linewidth=2, linestyle='--')
    g2.set_title('2. Error Klasifikasi (Class Loss)', fontsize=12, fontweight='bold')
    g2.set_xlabel('Epoch (Ronde Latihan)')
    g2.set_ylabel('Nilai Loss (Makin Kecil Makin Bagus)')
    g2.legend()

    # 3. Precision & Recall
    g3 = kumpulan_grafik[1, 0]
    if 'metrics/precision(B)' in tabel_metrik.columns:
        g3.plot(daftar_epoch, tabel_metrik['metrics/precision(B)'], label='Precision (Ketepatan Tebakan)', color='#2ecc71', linewidth=2)
    if 'metrics/recall(B)' in tabel_metrik.columns:
        g3.plot(daftar_epoch, tabel_metrik['metrics/recall(B)'], label='Recall (Kelengkapan Objek)', color='#9b59b6', linewidth=2)
    g3.set_title('3. Presisi & Recall (Precision & Recall)', fontsize=12, fontweight='bold')
    g3.set_xlabel('Epoch (Ronde Latihan)')
    g3.set_ylabel('Skor (0.0 sampai 1.0)')
    g3.set_ylim([0, 1.05])
    g3.legend()

    # 4. mAP Score
    g4 = kumpulan_grafik[1, 1]
    if 'metrics/mAP50(B)' in tabel_metrik.columns:
        g4.plot(daftar_epoch, tabel_metrik['metrics/mAP50(B)'], label='mAP@50 (Standar)', color='#f39c12', linewidth=2.5)
    if 'metrics/mAP50-95(B)' in tabel_metrik.columns:
        g4.plot(daftar_epoch, tabel_metrik['metrics/mAP50-95(B)'], label='mAP@50-95 (Ketat)', color='#16a085', linewidth=2, linestyle='--')
    g4.set_title('4. Nilai Rapor Akhir Model (mAP Score)', fontsize=12, fontweight='bold')
    g4.set_xlabel('Epoch (Ronde Latihan)')
    g4.set_ylabel('Skor mAP (Makin Dekat ke 1.0 = Makin Bagus)')
    g4.set_ylim([0, 1.05])
    g4.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_simpan_grafik = os.path.join(folder_tujuan, "training_loss_metrics_curves.png")
    plt.savefig(path_simpan_grafik, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Gambar 4 Kurva Hasil Latihan berhasil disimpan di:")
    print(f"     {path_simpan_grafik}")

def buat_diagram_batang_performa_kelas(model_jadi, path_data_yaml, folder_tujuan):
    """
    Membuat diagram batang performa kelas (Precision, Recall, mAP50, mAP50-95).
    """
    try:
        val_results = model_jadi.val(data=path_data_yaml, split='val', verbose=False)
        
        # Ekstraksi metrik
        precision = float(val_results.results_dict.get('metrics/precision(B)', 0.0))
        recall = float(val_results.results_dict.get('metrics/recall(B)', 0.0))
        map50 = float(val_results.results_dict.get('metrics/mAP50(B)', 0.0))
        map95 = float(val_results.results_dict.get('metrics/mAP50-95(B)', 0.0))

        metrik_nama = ['Precision', 'Recall', 'mAP@50', 'mAP@50-95']
        skor_nilai = [precision, recall, map50, map95]
        warna = ['#2ecc71', '#9b59b6', '#f39c12', '#16a085']

        plt.figure(figsize=(9, 5))
        batang = plt.bar(metrik_nama, skor_nilai, color=warna, width=0.5)
        plt.title('Evaluasi Model YOLO26 Nano: Kelas Qris-Positioning', fontsize=14, fontweight='bold')
        plt.ylabel('Nilai Metrik (0.0 - 1.0)', fontsize=11)
        plt.ylim([0, 1.15])

        for b in batang:
            yval = b.get_height()
            plt.text(b.get_x() + b.get_width()/2.0, yval + 0.03, f"{yval:.3f} ({yval*100:.1f}%)", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        path_diagram = os.path.join(folder_tujuan, "diagram_performa_per_kelas.png")
        plt.savefig(path_diagram, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Diagram Performa Kelas berhasil disimpan di:")
        print(f"     {path_diagram}")
    except Exception as e:
        print(f"[Warning] Gagal membuat diagram performa kelas: {e}")

def uji_model_pada_data_test(path_weights_best, path_data_yaml, folder_tujuan):
    """
    Melakukan UJIAN AKHIR pada gambar di folder 'test' dan memplot hasil prediksi.
    """
    print("\n--- Menjalankan Ujian Akhir Model Positioning pada Data Test ---")
    if not os.path.exists(path_weights_best):
        print(f"[Warning] File bobot model ({path_weights_best}) tidak ditemukan.")
        return

    model_jadi = YOLO(path_weights_best)
    
    # Simpan visualisasi prediksi data test
    folder_test_eval = os.path.join(folder_tujuan, "test_eval")
    os.makedirs(folder_test_eval, exist_ok=True)

    folder_saat_ini = Path(folder_tujuan).parent.parent.parent
    test_img_dir = folder_saat_ini / "test" / "images"

    gambar_test = list(test_img_dir.glob("*.*"))
    if not gambar_test:
        print(f"[Info] Tidak ada gambar di {test_img_dir} untuk visualisasi prediksi.")
        return

    print(f"[INFO] Menjalankan prediksi visual pada {len(gambar_test)} foto data test...")
    for idx, img_path in enumerate(gambar_test[:8]):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        
        results = model_jadi.predict(img_bgr, conf=0.25, verbose=False)[0]
        annotated_img = results.plot()

        out_name = f"uji_prediksi_test_{idx+1}_{img_path.name}"
        cv2.imwrite(os.path.join(folder_test_eval, out_name), annotated_img)

    print(f"[OK] Foto hasil ujian prediksi data test berhasil disimpan di:")
    print(f"     {folder_test_eval}")

def buat_laporan_dan_plot_lengkap(run_dir=None, path_data_yaml=None):
    """
    Fungsi gabungan utama untuk menghasilkan seluruh grafik & uji akhir.
    """
    folder_saat_ini = Path(os.path.dirname(os.path.abspath(__file__)))

    if run_dir is None:
        folder_runs = folder_saat_ini / "runs" / "detect"
        run_dir = cari_folder_training_terbaru(str(folder_runs))
        if not run_dir:
            print("[Error] Folder latihan tidak ditemukan.")
            return

    if path_data_yaml is None:
        path_data_yaml = str(folder_saat_ini / "data.yaml")

    run_dir = str(run_dir)
    print("=================================================================")
    print("       PEMBUATAN LAPORAN GRAFIK & EVALUASI PREDIKSI MODEL        ")
    print("=================================================================")
    print(f"Folder latihan yang dievaluasi : {run_dir}")

    # 1. Gambar Kurva Loss dan Metrik
    path_csv = os.path.join(run_dir, "results.csv")
    buat_grafik_loss_dan_metrik(path_csv, run_dir)

    # 2. Gambar Diagram Batang Performa Kelas & Uji Prediksi Test
    path_best_pt = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(path_best_pt):
        model_jadi = YOLO(path_best_pt)
        buat_diagram_batang_performa_kelas(model_jadi, path_data_yaml, run_dir)
        uji_model_pada_data_test(path_best_pt, path_data_yaml, run_dir)
    else:
        print(f"[Warning] File weights/best.pt belum ditemukan di {run_dir}")

    print("=================================================================")
    print("       SELURUH GRAFIK EVALUASI TELAH BERHASIL DICETAK!          ")
    print("=================================================================\n")

if __name__ == "__main__":
    buat_laporan_dan_plot_lengkap()
