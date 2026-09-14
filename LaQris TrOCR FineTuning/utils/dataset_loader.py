"""
=============================================================================
LAQRIS TROCR FINE-TUNING — DATASET LOADER (utils/dataset_loader.py)
=============================================================================
TUGAS UTAMA FILE INI:
1. Membaca file `labels.csv` yang berisi daftar nama file gambar dan label teksnya.
2. MENANGANIN KOMA DENGAN AMAN: Jika label nama merchant memiliki koma (contoh:
   "merchant_00010.png,QRIS BN, PULSA & INTERNET"), script ini memisahkan hanya pada
   koma PERTAMA. Semua karakter setelah koma pertama akan dianggap sebagai SATU teks utuh.
3. Mengubah gambar menjadi format Tensor Piksel (pixel_values) untuk AI TrOCR.
4. Mengubah teks label menjadi ID Token angka (labels) untuk dilatih oleh PyTorch.
=============================================================================
"""

import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image

class TrOCRDataset(Dataset):
    """
    Class Dataset PyTorch Khusus Pembaca Data TrOCR.
    
    Parameter:
    - dataset_dir        : Jalur folder yang berisi folder `images/` dan file `labels.csv`
    - processor          : Objek TrOCRProcessor (membawa image processor + tokenizer)
    - max_target_length  : Batas maksimum panjang karakter token teks (default: 32)
    - transform          : Transformasi tambahan pada gambar (opsional)
    """
    def __init__(self, dataset_dir: str, processor, max_target_length: int = 32, transform=None):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.csv_path = os.path.join(dataset_dir, "labels.csv")
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform

        # [INSTRUKSI 1] Pastikan file labels.csv benar-benar ada di folder dataset
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"[ERROR] File labels.csv tidak ditemukan di: {self.csv_path}")

        # [INSTRUKSI 2] Pembacaan CSV cerdas (Smart CSV Parser)
        # Menangani nama merchant yang mengandung tanda koma (',')
        valid_rows = []
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError(f"[ERROR] File {self.csv_path} kosong!")

        # Lewati baris header pertama jika ada (contoh: "file_name,text")
        start_idx = 0
        header_candidate = lines[0].strip().lower()
        if "file_name" in header_candidate or "filename" in header_candidate:
            start_idx = 1

        # Process baris demi baris
        for line_num, line in enumerate(lines[start_idx:], start=start_idx + 1):
            line_str = line.strip()
            if not line_str:
                continue # Abaikan baris kosong

            # -------------------------------------------------------------------------
            # LOGIKA PENANGANAN KOMA:
            # Gunakan split(',', 1) -> hanya memecah pada koma PERTAMA!
            # Contoh: "merchant_00010.png,QRIS BN, PULSA & INTERNET"
            #   parts[0] = "merchant_00010.png" (Nama file gambar)
            #   parts[1] = "QRIS BN, PULSA & INTERNET" (Teks nama merchant lengkap)
            # -------------------------------------------------------------------------
            parts = line_str.split(',', 1)

            if len(parts) < 2:
                # Format tidak valid (tidak ada koma)
                continue

            # Bersihkan tanda petik ganda atau spasi berlebih jika ada
            img_name = parts[0].strip().strip('"').strip("'")
            text_str = parts[1].strip().strip('"').strip("'")

            # Cek apakah file gambar benar-benar ada di folder `images/`
            img_path = os.path.join(self.images_dir, img_name)
            if os.path.exists(img_path) and len(text_str) > 0:
                valid_rows.append({
                    "file_name": img_name,
                    "text": text_str
                })
            elif not os.path.exists(img_path):
                print(f"[WARNING] Gambar '{img_name}' di baris {line_num} tidak ditemukan di {self.images_dir}. Melewati...")

        self.data = valid_rows
        print(f"[DATASET] Berhasil memuat {len(self.data)} sampel valid dari: {dataset_dir}")
        
        # Cetak contoh 3 sampel pertama untuk verifikasi hasil bacaan teks yang ada komanya
        if len(self.data) > 0:
            print("[DATASET SANITY CHECK] Contoh data yang dibaca:")
            for sample in self.data[:3]:
                print(f"   -> File: {sample['file_name']} | Teks Label: '{sample['text']}'")

    def __len__(self):
        """Mengembalikan jumlah total data gambar yang siap dilatih."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        [INSTRUKSI 3] Mengambil 1 sampel data berdasarkan indeks:
        - Membaca gambar dan mengubahnya ke format RGB.
        - Memproses piksel gambar menjadi tensor piksel untuk model AI TrOCR.
        - Memproses teks menjadi token ID angka untuk target pelatihan PyTorch.
        """
        item = self.data[idx]
        img_path = os.path.join(self.images_dir, item["file_name"])
        text = item["text"]

        # 1. Buka file gambar dan konversi ke format RGB (3 Channel Warna)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback jika ada gambar yang rusak/corrupt (buat gambar putih kosong 128x64)
            image = Image.new("RGB", (128, 64), color=(255, 255, 255))

        # 2. Jalankan augmentasi gambar jika disediakan
        if self.transform:
            image = self.transform(image)

        # 3. Ubah gambar menjadi Tensor Piksel (pixel_values) yang siap diterima TrOCR
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        # 4. Ubah teks nama merchant menjadi token angka (labels) untuk TrOCR Tokenizer
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # 5. Ganti token padding (pad_token_id) dengan -100 agar diabaikan saat menghitung Loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text,
            "file_name": item["file_name"]
        }
