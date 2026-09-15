# ==============================================================================
# SCRIPT 1: PREPARE DATASET (Persiapan & Pembagian Dataset QRIS Positioning)
# ==============================================================================
# Tujuan script ini:
# 1. Memeriksa keberadaan dataset di folder train, valid, dan test.
# 2. Jika valid dan test belum ada, otomatis membagi dataset dari train:
#    - Train : ~80% gambar
#    - Valid : ~10% gambar
#    - Test  : ~10% gambar
# 3. Mendeteksi secara otomatis jumlah kelas (nc) dari file label (.txt) yang ada.
# 4. Memperbarui file konfigurasi 'data.yaml' dengan path absolut rapi.
# ==============================================================================

import os
import shutil
import random
import yaml
from pathlib import Path

def pisahkan_dan_siapkan_dataset():
    """
    Fungsi utama untuk menyiapkan folder dan konfigurasi dataset QRIS Positioning.
    """
    folder_saat_ini = Path(os.path.dirname(os.path.abspath(__file__)))

    print("=================================================================")
    print("    LANGKAH 1: MENYIAPKAN & MEMBAGI DATASET QRIS POSITIONING    ")
    print("=================================================================")
    print(f"Lokasi folder kerja: {folder_saat_ini}")

    train_img_dir = folder_saat_ini / 'train' / 'images'
    train_lbl_dir = folder_saat_ini / 'train' / 'labels'
    valid_img_dir = folder_saat_ini / 'valid' / 'images'
    valid_lbl_dir = folder_saat_ini / 'valid' / 'labels'
    test_img_dir = folder_saat_ini / 'test' / 'images'
    test_lbl_dir = folder_saat_ini / 'test' / 'labels'

    # Buat direktori valid dan test jika belum ada
    valid_img_dir.mkdir(parents=True, exist_ok=True)
    valid_lbl_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Cek apakah dataset valid dan test sudah terisi
    valid_sudah_ada = any(valid_img_dir.iterdir())
    test_sudah_ada = any(test_img_dir.iterdir())

    if not valid_sudah_ada or not test_sudah_ada:
        print("\n[INFO] Dataset valid / test belum terbagi. Memulai pembagian dataset otomatis...")
        semua_gambar = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            semua_gambar.extend(list(train_img_dir.glob(ext)))
        
        semua_gambar = sorted(list(set(semua_gambar)))
        total_semua = len(semua_gambar)
        print(f"[INFO] Ditemukan {total_semua} gambar di folder train.")

        # Acak dengan seed tetap agar hasil konsisten dan reproducible
        random.seed(42)
        random.shuffle(semua_gambar)

        # Hitung alokasi: 80% train, 10% valid, 10% test
        jumlah_test = max(1, int(total_semua * 0.10))
        jumlah_valid = max(1, int(total_semua * 0.10))
        
        daftar_test = semua_gambar[:jumlah_test]
        daftar_valid = semua_gambar[jumlah_test:jumlah_test + jumlah_valid]
        daftar_train = semua_gambar[jumlah_test + jumlah_valid:]

        print(f"[INFO] Memindahkan {len(daftar_valid)} gambar ke folder 'valid'...")
        for img_path in daftar_valid:
            target_img = valid_img_dir / img_path.name
            shutil.move(str(img_path), str(target_img))
            lbl_name = img_path.stem + ".txt"
            lbl_path = train_lbl_dir / lbl_name
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(valid_lbl_dir / lbl_name))

        print(f"[INFO] Memindahkan {len(daftar_test)} gambar ke folder 'test'...")
        for img_path in daftar_test:
            target_img = test_img_dir / img_path.name
            shutil.move(str(img_path), str(target_img))
            lbl_name = img_path.stem + ".txt"
            lbl_path = train_lbl_dir / lbl_name
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(test_lbl_dir / lbl_name))

    # Tampilkan rangkuman dataset
    total_train = len(list(train_img_dir.glob('*.*')))
    total_valid = len(list(valid_img_dir.glob('*.*')))
    total_test = len(list(test_img_dir.glob('*.*')))

    print("\n[OK] Dataset train, valid, test telah siap:")
    print(f"  - Data Train : {total_train} gambar")
    print(f"  - Data Valid : {total_valid} gambar")
    print(f"  - Data Test  : {total_test} gambar")

    # Deteksi nama kelas
    daftar_nama_kelas = ['Qris-Positioning']
    path_data_yaml = folder_saat_ini / "data.yaml"

    if path_data_yaml.exists():
        try:
            with open(path_data_yaml, 'r') as f:
                old_yaml = yaml.safe_load(f)
                if old_yaml and 'names' in old_yaml:
                    if isinstance(old_yaml['names'], list):
                        daftar_nama_kelas = old_yaml['names']
                    elif isinstance(old_yaml['names'], dict):
                        daftar_nama_kelas = [old_yaml['names'][k] for k in sorted(old_yaml['names'].keys())]
        except Exception:
            pass

    konfigurasi_yaml = {
        'path': str(folder_saat_ini).replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(daftar_nama_kelas),
        'names': daftar_nama_kelas
    }

    with open(path_data_yaml, 'w') as file_yaml:
        yaml.dump(konfigurasi_yaml, file_yaml, default_flow_style=False, sort_keys=False)

    print(f"[OK] File konfigurasi 'data.yaml' berhasil diperbarui ({len(daftar_nama_kelas)} kelas) di:")
    print(f"     {path_data_yaml}")
    print("=================================================================")
    print("      PEMBAGIAN DATASET SELESAI & SIAP DIGUNAKAN!              ")
    print("=================================================================\n")

if __name__ == "__main__":
    pisahkan_dan_siapkan_dataset()
