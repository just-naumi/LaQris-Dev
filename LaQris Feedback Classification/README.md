# LaQris Feedback Classification (IndoBERT Fine-Tuned)

Modul NLP Text Classification berbasis **IndoBERT** (`indobenchmark/indobert-base-p1`) untuk mengklasifikasikan laporan, aduan, dan feedback pengguna pada ekosistem **QRIS Statis** LaQris.

---

## 1. Latar Belakang & Fokus QRIS Statis

Pada QRIS Statis (kode QR fisik di akrilik meja, etalase kasir, stiker dinding, kotak amal):
- Pembeli memasukkan nominal secara manual di aplikasi mobile banking / e-wallet.
- Rentan terhadap kejahatan fisik (*sticker overlay* / penempelan stiker palsu di atas stiker resmi).
- Rentan terhadap ketidakcocokan identitas merchant (nama penerima rekening berbeda dengan plang fisik toko).
- Rentan terhadap kesalahan input nominal (kelebihan digit/nol) dan *double debit* saat kasir meminta scan ulang.

Modul ini secara otomatis menganalisis teks laporan pengguna dan mengkategorikannya ke dalam tingkat risiko serta tindakan mitigasi yang tepat.

---

## 2. Taksonomi 6 Kelas Kategori

| No | Label Kategori | Severity | Sentimen | Deskripsi & Contoh Kasus |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `PENIPUAN_STIKER_PALSU` | `CRITICAL` | `NEGATIVE` | Stiker QR fisik ditimpa stiker baru penipu di meja/kotak amal. Nama penerima berubah menjadi rekening pribadi. |
| 2 | `KETIDAKSESUAIAN_IDENTITAS_MERCHANT` | `HIGH` | `NEGATIVE` | Nama merchant di layar HP berbeda dari plang toko, rekening penampung pinjol/scam, atau lokasi kota tidak cocok. |
| 3 | `SALAH_INPUT_NOMINAL_DOUBLE_BAYAR` | `HIGH` | `NEGATIVE` | Pembeli salah ketik nominal manual (kelebihan nol), atau kasir meminta bayar ulang sehingga terpotong 2x. |
| 4 | `STIKER_QRIS_RUSAK_PUDAR` | `MEDIUM` | `NEGATIVE` | Stiker QR fisik lecet, sobek, pudar kena sinar matahari/air, kotor kena minyak, sehingga scanner gagal membaca. |
| 5 | `TRANSAKSI_PENDING_GAGAL_SISTEM` | `MEDIUM` | `NEGATIVE` | Saldo rekening pembeli sudah terdebet tapi kasir mengklaim dana belum masuk (timeout / pending switching PJP). |
| 6 | `TRANSAKSI_SUKSES_NORMAL` | `LOW` | `POSITIVE` | Transaksi QRIS statis sukses tanpa kendala, nama merchant cocok dengan plang fisik, verifikasi instan. |

---

## 3. Dataset (10.000 Sampel)

Dataset tersimpan di `dataset/qris_feedback_10k.csv` dengan metadata lengkap:
- `id`: ID unik (`FB-00001` s.d. `FB-10000`).
- `text`: Kalimat feedback dalam bahasa Indonesia sehari-hari, ragam formal, gaul (*gue/lu/aing/rek*), singkatan, dan typo realistis.
- `category`: Label 1 dari 6 kelas kategori di atas.
- `severity`: Tingkat urgensi (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
- `sentiment`: Sentimen teks (`NEGATIVE`, `POSITIVE`).
- `qr_type`: `STATIC`.

Untuk membuat ulang dataset:
```bash
python dataset/build_10k_dataset.py
```

---

## 4. Cara Menjalankan Fine-Tuning IndoBERT

### Persiapan Dependensi
```bash
pip install -r requirements.txt
```

### Jalankan Training
```bash
python training/train_indobert.py
```
- Split data: 80% Train, 10% Validation, 10% Test.
- Checkpoint model terbaik disimpan di `models/indobert_qris_feedback/best_model/`.

### Evaluasi Model
```bash
python training/evaluate_model.py
```

---

## 5. Uji Coba Inferensi (Prediksi Teks)

Uji coba inferensi kalimat baru secara langsung:
```bash
python inference/predict.py "Hati-hati qris di meja kasir ditempel stiker baru pas discan namanya rekening pribadi!"
```

Output:
```text
--------------------------------------------------
Teks Input        : Hati-hati qris di meja kasir ditempel stiker baru pas discan namanya rekening pribadi!
Kategori Prediksi : PENIPUAN_STIKER_PALSU
Confidence        : 95.00%
Severity          : CRITICAL
Rekomendasi Aksi  : Segera kirim peringatan darurat ke merchant & blokir sementara QRIS fisik tersebut.
--------------------------------------------------
```
