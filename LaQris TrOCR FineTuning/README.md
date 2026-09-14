# 🚀 LaQris TrOCR Fine-Tuning Pipeline

Modul dan workspace khusus untuk melatih / melakukan *fine-tuning* model **TrOCR (Microsoft VisionEncoderDecoderModel)** secara mandiri dengan 3 spesialisasi model yang terpisah:

1. **`trocr_merchant_name`**: Model Spesialis Teks Nama Merchant / Toko.
2. **`trocr_nmid`**: Model Spesialis Pola Angka & Karakter NMID (`ID1020...`).
3. **`trocr_general`**: Model Spesialis Teks Umum Stiker QRIS (Acquirer Bank, Header, Footer).

---

## 📁 Struktur Folder

```text
LaQris TrOCR FineTuning/
├── base_model/                        # Tempat penyimpanan lokal weights microsoft/trocr-base-printed
├── datasets/                          # Dataset gambar + label teks (Image + Text Label)
│   ├── dataset_merchant_name/
│   │   ├── images/                    # Simpan gambar potongan teks di sini (.png / .jpg)
│   │   └── labels.csv                 # Pemetaan file_name,text
│   ├── dataset_nmid/
│   │   ├── images/
│   │   └── labels.csv
│   └── dataset_general/
│       ├── images/
│       └── labels.csv
├── dataset_generators/                # Generator dataset sintetis otomatis dengan variasi font
│   ├── generate_merchant_dataset.py
│   ├── generate_nmid_dataset.py
│   └── generate_general_dataset.py
├── fonts/                             # Simpan file font kustom (.ttf / .otf) di sini
├── models/                            # Folder hasil training (Model Checkpoints)
│   ├── trocr_merchant_name/
│   ├── trocr_nmid/
│   └── trocr_general/
├── utils/
│   ├── dataset_loader.py             # PyTorch Dataset Loader
│   └── augmentations.py              # Efek blur, noise, lighting kamera HP
├── download_base_model.py             # Script unduh base model offline
├── train_model.py                     # Script utama pelatih 3 model
└── requirements.txt                   # Dependency Python
```

---

## 🛠️ Langkah-Langkah Penggunaan

### 1. Persiapan Environment & Install Dependency
```bash
pip install -r requirements.txt
```

### 2. Mengunduh Base Model TrOCR (Sekali Saja)
Unduh model dasar `microsoft/trocr-base-printed` agar tersimpan secara lokal di folder `./base_model/`:
```bash
python download_base_model.py
```

---

## 🖼️ CARA 1: Melatih Menggunakan Gambar Asli Anda Sendiri

Jika Anda memiliki gambar potong teks stiker asli (misal hasil crop kamera HP), masukkan ke dalam folder dataset terkait:

1. Masukkan file gambar (`.png` / `.jpg`) ke folder `datasets/<nama_dataset>/images/`.
2. Buka file `datasets/<nama_dataset>/labels.csv` dan tuliskan pasangan nama file dan teksnya:
   ```csv
   file_name,text
   potongan_toko_1.png,WARUNG BU SRI
   potongan_nmid_1.png,ID102039485712
   potongan_bank_1.png,BANK CENTRAL ASIA
   ```

---

## 🎨 CARA 2: Menghasilkan Dataset Sintetis Otomatis (Variasi Font)

Anda juga bisa menambahkan berbagai font `.ttf` / `.otf` ke dalam folder `fonts/`, lalu jalankan generator sintetis untuk membuat ratusan/ribuan gambar teks otomatis:

```bash
# Membuat 200 sampel dataset sintetis nama merchant
python dataset_generators/generate_merchant_dataset.py 200

# Membuat 200 sampel dataset sintetis NMID
python dataset_generators/generate_nmid_dataset.py 200

# Membuat 200 sampel dataset sintetis General QRIS
python dataset_generators/generate_general_dataset.py 200
```

---

## 🏋️ CARA 3: Menjalankan Training / Fine-Tuning 3 Model

Anda dapat melatih model secara spesifik satu per satu atau sekaligus seluruhnya:

### A. Melatih Model Spesialis NMID:
```bash
python train_model.py --model_type nmid --epochs 10 --batch_size 8
```

### B. Melatih Model Spesialis Nama Merchant:
```bash
python train_model.py --model_type merchant_name --epochs 10 --batch_size 8
```

### C. Melatih Model Spesialis General QRIS:
```bash
python train_model.py --model_type general --epochs 10 --batch_size 8
```

### D. Melatih Ketiga Model Sekaligus (Batch):
```bash
python train_model.py --model_type all --epochs 5
```

Hasil bobot model terbaik (*checkpoint*) akan tersimpan di folder:
- `models/trocr_merchant_name/`
- `models/trocr_nmid/`
- `models/trocr_general/`

---

## 🔗 Menghubungkan Model Hasil Fine-Tuning ke Backend LaQris

Setelah selesai melatih, Anda dapat langsung mengarahkan `MODEL_TROCR` pada file `LaQris/backend/engine.py` ke folder model spesialisasi hasil pelatihan Anda!
