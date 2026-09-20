# Panduan Menyiapkan Dataset Feedback QRIS LaQris

Folder ini (`backend/data/`) digunakan untuk menaruh file dataset kustom Anda.

## Format File Dataset
- **Nama File**: `dataset.csv`
- **Format**: CSV dengan pemisah koma (`,`)
- **Kolom yang Wajib Ada**:
  1. `text` : Kalimat cerita atau keluhan dari pengguna mengenai transaksi QRIS.
  2. `label` : Kategori masalah (pilih salah satu dari 6 label di bawah ini).

## 6 Label yang Dikenali:
| Label ID | Label Key | Arti / Kategori Masalah |
| :--- | :--- | :--- |
| 0 | `identity_mismatch` | Nama penerima pembayaran berbeda dengan nama merchant |
| 1 | `qr_replacement` | Stiker QRIS fisik diduga ditempel/ditimpa di atas QRIS asli |
| 2 | `additional_fees` | Pengguna dikenakan biaya tambahan/admin yang tidak sah |
| 3 | `suspicious_transaction` | Transaksi gagal berkali-kali atau status saldo mencurigakan |
| 4 | `safe_confirmation` | Pengguna mengonfirmasi QRIS/merchant aman dan transaksi lancar |
| 5 | `other` | Keluhan lain di luar kategori di atas (misal layanan kasir, parkir, dll) |

## Contoh Isi `dataset.csv`:
```csv
text,label
"Nama di rekening tujuan beda jauh sama plang toko",identity_mismatch
"Stiker QRIS ditempel ganda di atas kode QR aslinya",qr_replacement
"Kasir meminta tambahan biaya 2000 untuk transaksi QRIS",additional_fees
"Transaksi berkali-kali gagal tapi saldo saya terpotong",suspicious_transaction
"QRIS resmi nama toko sesuai dan transaksi sangat lancar",safe_confirmation
"Pelayanannya kurang ramah dan toko agak kotor",other
```

## Cara Melatih (Fine-Tuning) IndoBERT:
Setelah file `dataset.csv` Anda siap di dalam folder ini, jalankan perintah berikut dari folder `backend/`:
```bash
python train_indobert.py --data data/dataset.csv --epochs 3 --batch-size 16
```
Hasil model akan otomatis tersimpan di folder `backend/weights/indobert_feedback_model/` dan langsung otomatis digunakan oleh sistem LaQris.
