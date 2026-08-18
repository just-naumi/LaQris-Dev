# ==============================================================================
# SCRIPT 1: PREPARE DATASET (Persiapan & Konfigurasi Dataset OCR QRIS)
# ==============================================================================
# Tujuan script ini:
# 1. Memeriksa keberadaan dataset di folder train, valid, dan test.
# 2. Tidak mengubah / mengacak ulang berkas gambar jika dataset sudah terbagi rapi.
# 3. Mendeteksi secara otomatis jumlah kelas (nc) dari file label (.txt) yang ada.
# 4. Memperbarui file konfigurasi 'data.yaml' dengan path absolut rapi.
# ==============================================================================

import os
import shutil
import zipfile
import random
import yaml
from pathlib import Path

def pisahkan_dan_siapkan_dataset():
    """
    Fungsi utama untuk menyiapkan folder dan konfigurasi dataset OCR secara otomatis.
    """
    folder_saat_ini = Path(os.path.dirname(os.path.abspath(__file__)))
    folder_utama_project = folder_saat_ini.parent.parent

    print("=================================================================")
    print("       LANGKAH 1: MENSIAPKAN & MEMBAGI DATASET OCR QRIS         ")
    print("=================================================================")
    print(f"Lokasi folder kerja: {folder_saat_ini}")

    train_img_dir = folder_saat_ini / 'train' / 'images'
    valid_img_dir = folder_saat_ini / 'valid' / 'images'
    test_img_dir = folder_saat_ini / 'test' / 'images'

    sudah_terbagi = train_img_dir.exists() and any(train_img_dir.iterdir())

    if sudah_terbagi:
        total_train = len(list(train_img_dir.glob('*.*')))
        total_valid = len(list(valid_img_dir.glob('*.*'))) if valid_img_dir.exists() else 0
        total_test = len(list(test_img_dir.glob('*.*'))) if test_img_dir.exists() else 0
        print("[INFO] Dataset train, valid, test sudah terbagi dengan rapi:")
        print(f"  - Data Train : {total_train} gambar")
        print(f"  - Data Valid : {total_valid} gambar")
        print(f"  - Data Test  : {total_test} gambar")

    # 4. Deteksi jumlah kelas terbesar (nc) dari file label yang ada
    max_class_id = -1
    search_labels_dirs = [folder_saat_ini / 'train' / 'labels', folder_saat_ini / 'valid' / 'labels', folder_saat_ini / 'test' / 'labels']
    for l_dir in search_labels_dirs:
        if l_dir.exists():
            for txt_file in l_dir.glob('*.txt'):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and parts[0].isdigit():
                                c_id = int(parts[0])
                                if c_id > max_class_id:
                                    max_class_id = c_id
                except Exception:
                    pass

    total_nc = max(5, max_class_id + 1)
    
    # Daftar nama kelas default
    daftar_nama_kelas_default = ['acquirer', 'nama_merchant', 'nmid', 'qrcode', 'tid']
    if total_nc > len(daftar_nama_kelas_default):
        for i in range(len(daftar_nama_kelas_default), total_nc):
            daftar_nama_kelas_default.append(f"komponen_{i}")

    # Membaca data.yaml lama jika ada untuk mempertahankan nama kelas resmi jika cocok
    path_data_yaml = folder_saat_ini / "data.yaml"
    names_to_use = daftar_nama_kelas_default

    if path_data_yaml.exists():
        try:
            with open(path_data_yaml, 'r') as f:
                old_yaml = yaml.safe_load(f)
                if old_yaml and 'names' in old_yaml:
                    old_names = old_yaml['names']
                    if isinstance(old_names, list) and len(old_names) >= total_nc:
                        names_to_use = old_names
                    elif isinstance(old_names, dict) and len(old_names) >= total_nc:
                        names_to_use = [old_names[k] for k in sorted(old_names.keys())]
        except Exception:
            pass

    konfigurasi_yaml = {
        'path': str(folder_saat_ini).replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(names_to_use),
        'names': names_to_use
    }

    with open(path_data_yaml, 'w') as file_yaml:
        yaml.dump(konfigurasi_yaml, file_yaml, default_flow_style=False, sort_keys=False)

    print(f"[OK] File konfigurasi 'data.yaml' berhasil diperbarui ({len(names_to_use)} kelas) di:")
    print(f"     {path_data_yaml}")
    print("=================================================================")
    print("      PEMBAGIAN DATASET SELESAI & SIAP DIGUNAKAN!              ")
    print("=================================================================\n")

if __name__ == "__main__":
    pisahkan_dan_siapkan_dataset()
