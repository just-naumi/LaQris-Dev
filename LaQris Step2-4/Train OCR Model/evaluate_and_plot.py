# ==============================================================================
# SCRIPT 3: EVALUATE & PLOT (Visualisasi & Evaluasi Hasil Pelatihan Model)
# ==============================================================================
# Tujuan script ini:
# 1. Membaca file 'results.csv' yang berisi catatan statistik latihan model dari epoch 1 sampai akhir.
# 2. Menggambar 4 Grafik Utama (Loss Curves & Metrik Keberhasilan):
#    - Box Loss: Seberapa akurat model menggambar kotak deteksi (Makin kecil nilainya = Makin Bagus).
#    - Class Loss: Seberapa tepat model menebak jenis label (Makin kecil nilainya = Makin Bagus).
#    - Precision & Recall: Seberapa tepat tebakan model & berapa persen objek yang berhasil ditemukan.
#    - mAP Score (Mean Average Precision): Nilai rapor kelulusan utama model (Makin dekat ke 1.0 / 100% = Makin Hebat).
# 3. Menghasilkan Diagram Batang Per-Kelas (Performa untuk masing-masing: acquirer, nama_merchant, nmid, qrcode, tid).
# 4. Menguji model jadi pada Data Test (Ujian Akhir) dan menggambar kotak deteksi di atas foto secara langsung!
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

# Atur tampilan grafik agar bersih dan menarik dipandang
sns.set_theme(style="darkgrid")
plt.rcParams['font.sans-serif'] = 'Arial'

def cari_folder_training_terbaru(folder_runs):
    """
    Fungsi sederhana untuk mencari folder hasil latihan YOLO yang paling baru dibuat.
    """
    daftar_folder = glob.glob(os.path.join(folder_runs, "train*"))
    if not daftar_folder:
        return None
    # Urutkan berdasarkan waktu pembuatan terbaru
    daftar_folder.sort(key=os.path.getmtime, reverse=True)
    return daftar_folder[0]

def buat_grafik_loss_dan_metrik(path_csv_results, folder_tujuan):
    """
    Fungsi untuk membaca data statistik latihan (csv) dan menggambar 4 grafik penting.
    """
    if not os.path.exists(path_csv_results):
        print(f"[Warning] File catatan statistik {path_csv_results} belum ada.")
        return

    # Baca file tabel CSV
    tabel_metrik = pd.read_csv(path_csv_results)
    # Hapus spasi tidak sengaja pada nama kolom tabel
    tabel_metrik.columns = [nama_kolom.strip() for nama_kolom in tabel_metrik.columns]

    # Buat kanvas gambar 2x2 (4 sub-grafik)
    fig, kumpulan_grafik = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Grafik Performa Latihan Model YOLO OCR QRIS', fontsize=16, fontweight='bold', y=0.98)

    daftar_epoch = tabel_metrik['epoch'] if 'epoch' in tabel_metrik.columns else range(1, len(tabel_metrik) + 1)

    # --------------------------------------------------------------------------
    # GRAFIK 1: Bounding Box Loss (Kesalahan Posisi Kotak)
    # Penjelasan: Menunjukkan seberapa presisi posisi kotak deteksi. Makin turun = Makin Bagus.
    # --------------------------------------------------------------------------
    g1 = kumpulan_grafik[0, 0]
    if 'train/box_loss' in tabel_metrik.columns:
        g1.plot(daftar_epoch, tabel_metrik['train/box_loss'], label='Loss Kotak (Train - Latihan)', color='#2b5c8f', linewidth=2)
    if 'val/box_loss' in tabel_metrik.columns:
        g1.plot(daftar_epoch, tabel_metrik['val/box_loss'], label='Loss Kotak (Valid - Ujian)', color='#e74c3c', linewidth=2, linestyle='--')
    g1.set_title('1. Error Posisi Kotak (Box Loss)', fontsize=12, fontweight='bold')
    g1.set_xlabel('Epoch (Ronde Latihan)')
    g1.set_ylabel('Nilai Loss (Makin Kecil Makin Bagus)')
    g1.legend()

    # --------------------------------------------------------------------------
    # GRAFIK 2: Classification Loss (Kesalahan Tebakan Label)
    # Penjelasan: Menunjukkan kesalahan model dalam membedakan misal nama_merchant vs nmid. Makin turun = Makin Bagus.
    # --------------------------------------------------------------------------
    g2 = kumpulan_grafik[0, 1]
    if 'train/cls_loss' in tabel_metrik.columns:
        g2.plot(daftar_epoch, tabel_metrik['train/cls_loss'], label='Loss Label (Train - Latihan)', color='#2b5c8f', linewidth=2)
    if 'val/cls_loss' in tabel_metrik.columns:
        g2.plot(daftar_epoch, tabel_metrik['val/cls_loss'], label='Loss Label (Valid - Ujian)', color='#e74c3c', linewidth=2, linestyle='--')
    g2.set_title('2. Error Tebakan Jenis Label (Class Loss)', fontsize=12, fontweight='bold')
    g2.set_xlabel('Epoch (Ronde Latihan)')
    g2.set_ylabel('Nilai Loss (Makin Kecil Makin Bagus)')
    g2.legend()

    # --------------------------------------------------------------------------
    # GRAFIK 3: Precision & Recall
    # Penjelasan:
    # - Precision: Jika model menebak "Ini QR Code", seberapa persen tebakannya BENAR.
    # - Recall: Dari seluruh QR Code yang ada di foto, berapa persen yang BERSAHIL DITEMUKAN.
    # --------------------------------------------------------------------------
    g3 = kumpulan_grafik[1, 0]
    if 'metrics/precision(B)' in tabel_metrik.columns:
        g3.plot(daftar_epoch, tabel_metrik['metrics/precision(B)'], label='Precision (Ketepatan Tebakan)', color='#2ecc71', linewidth=2)
    if 'metrics/recall(B)' in tabel_metrik.columns:
        g3.plot(daftar_epoch, tabel_metrik['metrics/recall(B)'], label='Recall (Kelengkapan Objek Ditemukan)', color='#9b59b6', linewidth=2)
    g3.set_title('3. Presisi Tebakan (Precision & Recall)', fontsize=12, fontweight='bold')
    g3.set_xlabel('Epoch (Ronde Latihan)')
    g3.set_ylabel('Skor (0.0 sampai 1.0)')
    g3.set_ylim([0, 1.05])
    g3.legend()

    # --------------------------------------------------------------------------
    # GRAFIK 4: mAP (Mean Average Precision)
    # Penjelasan: Nilai IPK / Rapor kelulusan keseluruhan model.
    # - mAP@50   : Nilai kelulusan standar (ambang ketepatan 50%).
    # - mAP@50-95: Nilai kelulusan sangat ketat (ambang ketepatan 50% sampai 95%).
    # --------------------------------------------------------------------------
    g4 = kumpulan_grafik[1, 1]
    if 'metrics/mAP50(B)' in tabel_metrik.columns:
        g4.plot(daftar_epoch, tabel_metrik['metrics/mAP50(B)'], label='mAP@50 (Standar)', color='#f39c12', linewidth=2.5)
    if 'metrics/mAP50-95(B)' in tabel_metrik.columns:
        g4.plot(daftar_epoch, tabel_metrik['metrics/mAP50-95(B)'], label='mAP@50-95 (Sangat Ketat)', color='#16a085', linewidth=2, linestyle='--')
    g4.set_title('4. Nilai Rapor Akhir Model (mAP Score)', fontsize=12, fontweight='bold')
    g4.set_xlabel('Epoch (Ronde Latihan)')
    g4.set_ylabel('Skor mAP (Makin Mendekati 1.0 / 100% = Makin Hebat)')
    g4.set_ylim([0, 1.05])
    g4.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_simpan_grafik = os.path.join(folder_tujuan, "training_loss_metrics_curves.png")
    plt.savefig(path_simpan_grafik, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Gambar 4 Kurva Hasil Latihan berhasil disimpan di:")
    print(f"     {path_simpan_grafik}")

def uji_model_pada_data_test(path_weights_best, path_data_yaml, folder_tujuan):
    """
    Fungsi untuk melakukan UJIAN AKHIR pada 7 gambar di folder 'test'
    dan menggambar hasil tebakan kotak bounding box di atas gambar.
    """
    print("\n--- Menjalankan Ujian Akhir Model pada Data Test ---")
    if not os.path.exists(path_weights_best):
        print(f"[Warning] File bobot pintar model ({path_weights_best}) tidak ditemukan.")
        return

    # Muat model jadi yang sudah dilatih (bobot terbaik 'best.pt')
    model_jadi = YOLO(path_weights_best)

    # 1. Hitung skor ujian akhir pada data test
    hasil_ujian = model_jadi.val(data=path_data_yaml, split='test', project=folder_tujuan, name='test_eval', exist_ok=True)
    
    skor_precision = hasil_ujian.box.mp
    skor_recall = hasil_ujian.box.mr
    skor_map50 = hasil_ujian.box.map50
    skor_map50_95 = hasil_ujian.box.map

    print("\n" + "="*60)
    print("        RANGKUMAN SKOR UJIAN AKHIR (TEST SET EVALUATION)        ")
    print("="*60)
    print(f" Presisi Tebakan (Precision) : {skor_precision:.4f} ({skor_precision*100:.1f}%)")
    print(f" Kelengkapan Objek (Recall)  : {skor_recall:.4f} ({skor_recall*100:.1f}%)")
    print(f" Nilai Rapor Standar (mAP50) : {skor_map50:.4f} ({skor_map50*100:.1f}%)")
    print(f" Nilai Rapor Ketat (mAP50-95): {skor_map50_95:.4f} ({skor_map50_95*100:.1f}%)")
    print("="*60)

    # 2. Gambar Diagram Batang Performa Per-Kelas (5 Komponen QRIS)
    daftar_nama_kelas = list(hasil_ujian.names.values())
    skor_map_per_kelas = hasil_ujian.box.maps # Skor mAP50-95 per kelas

    if len(skor_map_per_kelas) == len(daftar_nama_kelas):
        plt.figure(figsize=(10, 5))
        sns.barplot(x=daftar_nama_kelas, y=skor_map_per_kelas, palette="Blues_d")
        plt.title("Performa Akurasi Deteksi per-Komponen QRIS (Data Test)", fontsize=14, fontweight='bold')
        plt.xlabel("Komponen QRIS", fontsize=12)
        plt.ylabel("Skor Akurasi (mAP@50-95)", fontsize=12)
        plt.ylim([0, 1.05])

        # Tampilkan angka di atas setiap batang diagram
        for idx, nilai in enumerate(skor_map_per_kelas):
            plt.text(idx, nilai + 0.02, f"{nilai:.3f}", ha='center', fontweight='bold')

        path_chart_kelas = os.path.join(folder_tujuan, "per_class_performance.png")
        plt.savefig(path_chart_kelas, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Diagram Batang Per-Kelas disimpan di: {path_chart_kelas}")

    # 3. Prediksi dan Visualisasi Kotak Bounding Box pada Foto Uji
    folder_gambar_test = os.path.join(os.path.dirname(path_data_yaml), "test", "images")
    daftar_foto_test = glob.glob(os.path.join(folder_gambar_test, "*.*"))

    if daftar_foto_test:
        print("Menggambar hasil prediksi kotak deteksi pada foto uji...")
        hasil_prediksi = model_jadi.predict(source=daftar_foto_test, conf=0.25, save=False)

        total_foto = len(daftar_foto_test)
        kolom = min(4, total_foto)
        baris = (total_foto + kolom - 1) // kolom

        fig, kumpulan_sub = plt.subplots(baris, kolom, figsize=(5 * kolom, 5 * baris))
        if baris == 1 and kolom == 1:
            kumpulan_sub = np.array([kumpulan_sub])
        kumpulan_sub = kumpulan_sub.flatten()

        for i, (pred, path_foto) in enumerate(zip(hasil_prediksi, daftar_foto_test)):
            # Gambar kotak bounding box hasil tebakan model
            gambar_hasil_bgr = pred.plot()
            # Ubah warna BGR OpenCV menjadi RGB Matplotlib
            gambar_hasil_rgb = cv2.cvtColor(gambar_hasil_bgr, cv2.COLOR_BGR2RGB)

            ax = kumpulan_sub[i]
            ax.imshow(gambar_hasil_rgb)
            ax.set_title(os.path.basename(path_foto), fontsize=10)
            ax.axis('off')

        # Sembunyikan subplot sisa yang tidak terpakai
        for sisa_idx in range(len(daftar_foto_test), len(kumpulan_sub)):
            kumpulan_sub[sisa_idx].axis('off')

        plt.suptitle("Contoh Hasil Deteksi Model pada Foto Test (QRIS OCR)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        path_galeri_prediksi = os.path.join(folder_tujuan, "test_predictions_collage.png")
        plt.savefig(path_galeri_prediksi, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Galeri Foto Hasil Uji disimpan di: {path_galeri_prediksi}")

def buat_laporan_dan_plot_lengkap(run_dir, path_data_yaml):
    """
    Fungsi utama yang memanggil pembuatan grafik dan ujian akhir pada data test.
    """
    folder_laporan = os.path.join(run_dir, "summary_report")
    os.makedirs(folder_laporan, exist_ok=True)

    print(f"\n=======================================================")
    print(f"      MEMBUAT LAPORAN HASIL EVALUASI LENGKAP           ")
    print(f"=======================================================")
    print(f"Folder Hasil Latihan : {run_dir}")
    print(f"Folder Gambar Laporan: {folder_laporan}\n")

    # 1. Gambar Kurva Loss dan Metrik dari file csv
    path_csv = os.path.join(run_dir, "results.csv")
    buat_grafik_loss_dan_metrik(path_csv, folder_laporan)

    # 2. Cari file bobot model terbaik 'best.pt'
    path_best_pt = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(path_best_pt):
        path_best_pt = os.path.join(run_dir, "weights", "last.pt")

    # 3. Jalankan Ujian pada data Test
    if os.path.exists(path_best_pt):
        uji_model_pada_data_test(path_best_pt, path_data_yaml, folder_laporan)

    print(f"\n[SUKSES] Seluruh grafik laporan telah rapi tersimpan di:")
    print(f"         {folder_laporan}")
    print("=======================================================\n")

# Kode ini berjalan jika script ini dieksekusi langsung
if __name__ == "__main__":
    folder_saat_ini = os.path.dirname(os.path.abspath(__file__))
    folder_runs_detect = os.path.join(folder_saat_ini, "runs", "detect")
    run_terbaru = cari_folder_training_terbaru(folder_runs_detect)
    path_yaml = os.path.join(folder_saat_ini, "data.yaml")

    if run_terbaru:
        buat_laporan_dan_plot_lengkap(run_terbaru, path_yaml)
    else:
        print("[Info] Belum ada folder hasil latihan ('runs/detect/train*').")
        print("Silakan jalankan 'python train_yolo.py' terlebih dahulu untuk melatih model.")
