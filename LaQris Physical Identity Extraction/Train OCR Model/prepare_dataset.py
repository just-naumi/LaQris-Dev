# ==============================================================================
# SCRIPT 1: PREPARE DATASET (Persiapan & Pembagian Data Gambar & Label)
# ==============================================================================
# Tujuan script ini:
# 1. Mengekstrak file ZIP berisi foto dan label koordinat teks QRIS.
# 2. Mengumpulkan semua pasang gambar (.jpg) dan label (.txt).
# 3. Membagi data menjadi 3 bagian:
#    - Data Train (70%): Data yang dipakai model untuk BELAJAR mengenali posisi teks.
#    - Data Valid (15%): Data yang dipakai model untuk UJIAN SEMENTARA saat latihan.
#    - Data Test  (15%): Data ujian akhir untuk mengetes kehebatan model yang sudah jadi.
# 4. Membuat file konfigurasi 'data.yaml' agar YOLO tahu lokasi folder & nama kelasnya.
# ==============================================================================

import os
import shutil
import zipfile
import random
import yaml
from pathlib import Path

def pisahkan_dan_siapkan_dataset():
    """
    Fungsi utama untuk menyiapkan folder dan membagi dataset secara otomatis.
    """
    # 1. Menentukan lokasi folder script ini berada
    folder_saat_ini = Path(os.path.dirname(os.path.abspath(__file__)))
    folder_utama_project = folder_saat_ini.parent.parent
    path_zip_dataset = folder_utama_project / "QRIS.v2-v1.yolo26.zip"
    
    print("=================================================================")
    print("       LANGKAH 1: MENSIAPKAN & MEMBAGI DATASET OCR QRIS         ")
    print("=================================================================")
    print(f"Lokasi folder kerja: {folder_saat_ini}")

    # 2. Jika file ZIP dataset ditemukan, kita ekstrak (buka) dulu isi zip-nya
    folder_ekstrak_sementara = folder_saat_ini / "_temp_extract"
    
    if os.path.exists(path_zip_dataset):
        print(f"Membuka dan mengekstrak file ZIP: {path_zip_dataset.name} ...")
        
        # Bersihkan folder sementara jika dulu pernah ada
        if folder_ekstrak_sementara.exists():
            shutil.rmtree(folder_ekstrak_sementara)
            
        # Ekstrak seluruh file di zip ke folder sementara
        with zipfile.ZipFile(path_zip_dataset, 'r') as file_zip:
            file_zip.extractall(folder_ekstrak_sementara)
            
        print("Ekstraksi ZIP selesai!")
    
    # 3. Mengumpulkan semua gambar (.jpg / .png) beserta label pasangannya (.txt)
    daftar_pasangan_data = [] # List untuk menyimpan pasangan (path_gambar, path_label)
    
    # Cari di dalam folder hasil ekstraksi sementara
    search_dir = folder_ekstrak_sementara if folder_ekstrak_sementara.exists() else folder_saat_ini
    
    for root, subfolders, files in os.walk(search_dir):
        for nama_file in files:
            # Cek apakah file ini adalah gambar
            if nama_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path_gambar = Path(root) / nama_file
                
                # Cari file label (.txt) yang namanya sama persis dengan gambar
                path_label = path_gambar.with_suffix('.txt')
                
                # Jika label tidak ada di folder yang sama, coba cari di folder 'labels'
                if not path_label.exists():
                    if path_gambar.parent.name == 'images':
                        path_label = path_gambar.parent.parent / 'labels' / f"{path_gambar.stem}.txt"
                
                # Jika pasangannya (file .txt) ditemukan, simpan ke daftar
                if path_label.exists():
                    daftar_pasangan_data.append((path_gambar, path_label))

    # Pastikan tidak ada data ganda (duplikat) berdasarkan nama file
    kamus_unik = {}
    for path_img, path_txt in daftar_pasangan_data:
        kamus_unik[path_img.name] = (path_img, path_txt)
    daftar_pasangan_data = list(kamus_unik.values())

    total_semua_data = len(daftar_pasangan_data)
    print(f"Total data pasangan gambar dan label yang ditemukan: {total_semua_data} buah.")

    if total_semua_data == 0:
        print("[ERROR] Tidak ada data gambar dan label yang ditemukan!")
        return

    # 4. Mengacak urutan data dengan angka acak tetap (seed=42)
    # Tujuan acak: Supaya pembagian data train, valid, dan test bervariasi dan adil.
    random.seed(42)
    random.shuffle(daftar_pasangan_data)

    # 5. Menghitung pembagian jumlah data:
    # 70% untuk Train (Belajar)
    # 15% untuk Valid (Evaluasi saat belajar)
    # 15% untuk Test (Ujian akhir)
    jumlah_train = int(total_semua_data * 0.70)
    jumlah_valid = int(total_semua_data * 0.15)
    jumlah_test = total_semua_data - jumlah_train - jumlah_valid

    data_train = daftar_pasangan_data[:jumlah_train]
    data_valid = daftar_pasangan_data[jumlah_train : jumlah_train + jumlah_valid]
    data_test = daftar_pasangan_data[jumlah_train + jumlah_valid :]

    print("\nRincian Pembagian Data:")
    print(f"  - Data Train (Belajar) : {len(data_train)} gambar ({len(data_train)/total_semua_data*100:.1f}%)")
    print(f"  - Data Valid (Ujian 1) : {len(data_valid)} gambar ({len(data_valid)/total_semua_data*100:.1f}%)")
    print(f"  - Data Test  (Ujian 2) : {len(data_test)} gambar ({len(data_test)/total_semua_data*100:.1f}%)\n")

    # 6. Membuat struktur folder bersih: train, valid, test (masing-masing punya folder images & labels)
    kumpulan_split = {
        'train': data_train,
        'valid': data_valid,
        'test': data_test
    }

    for nama_split, item_pasangan in kumpulan_split.items():
        # Buat folder images dan labels jika belum ada
        folder_gambar_target = folder_saat_ini / nama_split / 'images'
        folder_label_target = folder_saat_ini / nama_split / 'labels'
        
        # Kosongkan/hapus folder lama agar tidak tercampur
        if folder_gambar_target.exists():
            shutil.rmtree(folder_gambar_target)
        if folder_label_target.exists():
            shutil.rmtree(folder_label_target)
            
        os.makedirs(folder_gambar_target, exist_ok=True)
        os.makedirs(folder_label_target, exist_ok=True)

        # Salin (copy) file gambar dan label ke dalam foldernya masing-masing
        for path_img_asal, path_txt_asal in item_pasangan:
            shutil.copy2(path_img_asal, folder_gambar_target / path_img_asal.name)
            shutil.copy2(path_txt_asal, folder_label_target / path_txt_asal.name)

    # Bersihkan folder sementara bekas zip tadi
    if folder_ekstrak_sementara.exists():
        shutil.rmtree(folder_ekstrak_sementara)

    # 7. Membuat file 'data.yaml' (Setting konfigurasi dataset untuk YOLO)
    # File ini memberitahu YOLO letak folder data dan nama 5 komponen QRIS yang dideteksi
    path_data_yaml = folder_saat_ini / "data.yaml"
    
    konfigurasi_yaml = {
        'path': str(folder_saat_ini).replace('\\', '/'), # Path lokasi utama dataset
        'train': 'train/images',                         # Path folder train gambar
        'val': 'valid/images',                           # Path folder valid gambar
        'test': 'test/images',                           # Path folder test gambar
        'nc': 5,                                         # Total Jumlah Kelas / Label (5 kelas)
        'names': ['acquirer', 'nama_merchant', 'nmid', 'qrcode', 'tid'] # Nama 5 kelas QRIS
    }

    # Tulis file data.yaml dengan format YAML rapi
    with open(path_data_yaml, 'w') as file_yaml:
        yaml.dump(konfigurasi_yaml, file_yaml, default_flow_style=False, sort_keys=False)

    print(f"[OK] File konfigurasi 'data.yaml' berhasil diperbarui di:")
    print(f"     {path_data_yaml}")
    print("=================================================================")
    print("      PEMBAGIAN DATASET SELESAI & SIAP DIGUNAKAN!              ")
    print("=================================================================\n")

# Kode ini akan berjalan jika script ini dieksekusi langsung
if __name__ == "__main__":
    pisahkan_dan_siapkan_dataset()
