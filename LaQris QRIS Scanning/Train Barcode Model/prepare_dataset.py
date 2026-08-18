# ==============================================================================
# SCRIPT 1: PREPARE DATASET (Persiapan & Pembagian Data Gambar & Label Barcode)
# ==============================================================================
# Tujuan script ini:
# 1. Mengekstrak file ZIP dataset barcode jika ada.
# 2. Mengumpulkan dan memastikan pasang gambar (.jpg/.png) dan label (.txt).
# 3. Jika dataset belum terbagi, bagi menjadi train (70%), valid (15%), test (15%).
# 4. Memperbarui file konfigurasi 'data.yaml' dengan path rapi dan kelas:
#    ['Barcode_Asli', 'Barcode_Palsu']
# ==============================================================================

import os
import shutil
import zipfile
import random
import yaml
from pathlib import Path

def pisahkan_dan_siapkan_dataset():
    """
    Fungsi utama untuk menyiapkan folder dan membagi dataset barcode secara otomatis.
    """
    folder_saat_ini = Path(os.path.dirname(os.path.abspath(__file__)))
    folder_utama_project = folder_saat_ini.parent.parent

    print("=================================================================")
    print("      LANGKAH 1: MENSIAPKAN & MEMBAGI DATASET BARCODE QRIS       ")
    print("=================================================================")
    print(f"Lokasi folder kerja: {folder_saat_ini}")

    # Cari file zip dataset jika ada
    path_zip_dataset = None
    for file_name in os.listdir(folder_utama_project):
        if file_name.lower().endswith('.zip') and 'barcode' in file_name.lower():
            path_zip_dataset = folder_utama_project / file_name
            break

    folder_ekstrak_sementara = folder_saat_ini / "_temp_extract"

    if path_zip_dataset and os.path.exists(path_zip_dataset):
        print(f"Membuka dan mengekstrak file ZIP: {path_zip_dataset.name} ...")
        if folder_ekstrak_sementara.exists():
            shutil.rmtree(folder_ekstrak_sementara)
        with zipfile.ZipFile(path_zip_dataset, 'r') as file_zip:
            file_zip.extractall(folder_ekstrak_sementara)
        print("Ekstraksi ZIP selesai!")

    # Cek apakah folder train, valid, test sudah ada dan terisi
    train_img_dir = folder_saat_ini / 'train' / 'images'
    valid_img_dir = folder_saat_ini / 'valid' / 'images'
    test_img_dir = folder_saat_ini / 'test' / 'images'

    sudah_siap = train_img_dir.exists() and any(train_img_dir.iterdir())

    if not sudah_siap and (folder_ekstrak_sementara.exists() or list(folder_saat_ini.glob("*.jpg"))):
        daftar_pasangan_data = []
        search_dir = folder_ekstrak_sementara if folder_ekstrak_sementara.exists() else folder_saat_ini

        for root, subfolders, files in os.walk(search_dir):
            for nama_file in files:
                if nama_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path_gambar = Path(root) / nama_file
                    path_label = path_gambar.with_suffix('.txt')
                    if not path_label.exists():
                        if path_gambar.parent.name == 'images':
                            path_label = path_gambar.parent.parent / 'labels' / f"{path_gambar.stem}.txt"
                    if path_label.exists():
                        daftar_pasangan_data.append((path_gambar, path_label))

        kamus_unik = {}
        for path_img, path_txt in daftar_pasangan_data:
            kamus_unik[path_img.name] = (path_img, path_txt)
        daftar_pasangan_data = list(kamus_unik.values())

        total_semua_data = len(daftar_pasangan_data)
        print(f"Total data pasangan gambar dan label yang ditemukan: {total_semua_data} buah.")

        if total_semua_data > 0:
            random.seed(42)
            random.shuffle(daftar_pasangan_data)

            jumlah_train = int(total_semua_data * 0.70)
            jumlah_valid = int(total_semua_data * 0.15)

            data_train = daftar_pasangan_data[:jumlah_train]
            data_valid = daftar_pasangan_data[jumlah_train : jumlah_train + jumlah_valid]
            data_test = daftar_pasangan_data[jumlah_train + jumlah_valid :]

            kumpulan_split = {
                'train': data_train,
                'valid': data_valid,
                'test': data_test
            }

            for nama_split, item_pasangan in kumpulan_split.items():
                folder_gambar_target = folder_saat_ini / nama_split / 'images'
                folder_label_target = folder_saat_ini / nama_split / 'labels'
                if folder_gambar_target.exists():
                    shutil.rmtree(folder_gambar_target)
                if folder_label_target.exists():
                    shutil.rmtree(folder_label_target)
                os.makedirs(folder_gambar_target, exist_ok=True)
                os.makedirs(folder_label_target, exist_ok=True)

                for path_img_asal, path_txt_asal in item_pasangan:
                    shutil.copy2(path_img_asal, folder_gambar_target / path_img_asal.name)
                    shutil.copy2(path_txt_asal, folder_label_target / path_txt_asal.name)

        if folder_ekstrak_sementara.exists():
            shutil.rmtree(folder_ekstrak_sementara)

    # 5. Memperbarui file 'data.yaml'
    path_data_yaml = folder_saat_ini / "data.yaml"

    konfigurasi_yaml = {
        'path': str(folder_saat_ini).replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 2,
        'names': ['Barcode_Asli', 'Barcode_Palsu']
    }

    with open(path_data_yaml, 'w') as file_yaml:
        yaml.dump(konfigurasi_yaml, file_yaml, default_flow_style=False, sort_keys=False)

    print(f"[OK] File konfigurasi 'data.yaml' berhasil diperbarui di:")
    print(f"     {path_data_yaml}")
    print("=================================================================")
    print("      PEMBAGIAN DATASET SELESAI & SIAP DIGUNAKAN!              ")
    print("=================================================================\n")

if __name__ == "__main__":
    pisahkan_dan_siapkan_dataset()
