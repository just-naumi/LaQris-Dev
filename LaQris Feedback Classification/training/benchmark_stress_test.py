"""
Benchmark Stress Test Suite untuk Evaluasi Model IndoBERT LaQris EMRS
Menguji 50 Kasus Nyata (Adversarial, Hard Negatives, Surcharge, Fraud, Layanan Merchant, Ambigu).

7 Kategori Resmi LaQris EMRS:
1. PENIPUAN_STIKER_QRIS_PALSU
2. KETIDAKSESUAIAN_IDENTITAS_MERCHANT
3. PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE
4. KONDISI_FISIK_QRIS_RUSAK
5. KUALITAS_LAYANAN_MERCHANT
6. QRIS_NORMAL_MERCHANT_TERPERCAYA
7. FEEDBACK_AMBIGU
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TEST_CASES = [
    # ---------------------------------------------------------
    # 1. HARD NEGATIVES / QRIS NORMAL TERPERCAYA (10 Kasus)
    # Expected: QRIS_NORMAL_MERCHANT_TERPERCAYA
    # ---------------------------------------------------------
    {
        "id": "HN_01",
        "category": "HARD_NORMAL",
        "expected": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
        "text": "Saya membeli makanan di warung dekat kampus dan membayar menggunakan QRIS seperti biasanya. Nominal yang harus dibayar sesuai dengan harga makanan, transaksi berhasil dalam beberapa detik, dan bukti pembayaran juga langsung muncul di aplikasi."
    },
    {
        "id": "HN_02",
        "category": "HARD_NORMAL",
        "expected": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
        "text": "Kasir ramah menunjukkan barcode QRIS di meja, nominal pembayaran saya input pas 35 ribu tanpa biaya admin dan notifikasi uang masuk langsung bunyi di soundbox hp penjual."
    },
    {
        "id": "HN_03",
        "category": "HARD_NORMAL",
        "expected": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
        "text": "Proses pembayaran QRIS berlangsung sangat cepat tanpa kendala apapun, struk dan bukti transfer tersimpan rapi di riwayat aplikasi perbankan saya."
    },
    {
        "id": "HN_04",
        "category": "HARD_NORMAL",
        "expected": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
        "text": "Tadi bayar parkir pakai QRIS statis di pos keluar, nama penerima sesuai pengelola dan transaksi langsung terverifikasi hijau."
    },
    {
        "id": "HN_05",
        "category": "HARD_NORMAL",
        "expected": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
        "text": "Nominal belanjaan 150 ribu saya bayar dengan scan QRIS di kasir, semua detail informasi pembayaran cocok dan saya langsung dikasih barangnya beserta nota lunas."
    },
    {
        "id": "HN_06",
        "category": "HARD_NORMAL",
        "expected": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
        "text": "Pelayanan merchant sangat baik, qr code bersih dan terlindung akrilik, pembayaran sukses seketika tanpa ada biaya potongan tambahan."
    },
    {
        "id": "HN_07",
        "category": "HARD_NORMAL",
        "expected": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
        "text": "Cek mutasi setelah transaksi nominalnya pas sama yang saya ketik tadi siang di toko buku, pembayaran QRIS selesai aman sentosa."
    },
    {
        "id": "HN_08",
        "category": "HARD_NORMAL",
        "expected": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
        "text": "QRIS statis di warteg ini berfungsi normal, nama warteg muncul di layar konfirmasi dan transaksi langsung berhasil sekali scan."
    },
    {
        "id": "HN_09",
        "category": "HARD_NORMAL",
        "expected": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
        "text": "Tidak ada masalah sama sekali, pembayaran lewat qris merchant berjalan mulus dan bukti bayar resmi langsung keluar."
    },
    {
        "id": "HN_10",
        "category": "HARD_NORMAL",
        "expected": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
        "text": "Transaksi berhasil diproses, kasir mengecek terminal soundbox dan nominal uang masuk cocok dengan pesanan saya tanpa ada selisih."
    },

    # ---------------------------------------------------------
    # 2. IDENTITAS MERCHANT TIDAK SESUAI (10 Kasus)
    # Expected: KETIDAKSESUAIAN_IDENTITAS_MERCHANT
    # ---------------------------------------------------------
    {
        "id": "IM_01",
        "category": "IMPLICIT_IDENTITY",
        "expected": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
        "text": "Saya baru menyadari setelah pembayaran selesai bahwa nama merchant yang muncul pada detail transaksi tidak sesuai dengan nama toko tempat saya berbelanja, sehingga saya khawatir kode QRIS yang digunakan bukan milik merchant tersebut."
    },
    {
        "id": "IM_02",
        "category": "IMPLICIT_IDENTITY",
        "expected": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
        "text": "Saya makan di Restoran Padang Sederhana, tapi pas scan QRIS di aplikasi nama yang muncul malahan rekening atas nama pribadi Budi Santoso."
    },
    {
        "id": "IM_03",
        "category": "IMPLICIT_IDENTITY",
        "expected": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
        "text": "Setelah bayar di counter pulsa via QRIS, di struk digital nama merchant-nya Yayasan Amal Kasih padahal ini toko aksesoris hp."
    },
    {
        "id": "IM_04",
        "category": "IMPLICIT_IDENTITY",
        "expected": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
        "text": "Di banner tertulis Kopi Kenangan Mantan, pas di layar konfirmasi bayar QRIS judulnya Konveksi Baju Murah."
    },
    {
        "id": "IM_05",
        "category": "IMPLICIT_IDENTITY",
        "expected": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
        "text": "Transaksi berhasil tapi penjual bilang nama penerima yang ada di bukti transfer QRIS saya bukan rekening milik outlet mereka."
    },
    {
        "id": "IM_06",
        "category": "IMPLICIT_IDENTITY",
        "expected": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
        "text": "Plang depan warung namanya Mie Ayam Bakso Mas Joko, kenapa nama tujuan QRIS-nya CV Abadi Sejahtera di kota lain?"
    },
    {
        "id": "IM_07",
        "category": "IMPLICIT_IDENTITY",
        "expected": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
        "text": "Pas mau konfirmasi bayar QRIS tertera nama PT yang asing dan tidak ada hubungannya sama klinik tempat saya berobat ini."
    },
    {
        "id": "IM_08",
        "category": "IMPLICIT_IDENTITY",
        "expected": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
        "text": "Kasir kaget liat bukti bayar saya karena tujuan QRIS-nya ke rekening perorangan yang karyawan toko sendiri tidak tahu itu siapa."
    },
    {
        "id": "IM_09",
        "category": "IMPLICIT_IDENTITY",
        "expected": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
        "text": "Curiga pas bayar martabak pakai QRIS, nama di barcode akun personal nama cewek padahal abang penjualnya cowok dan bukan nama istrinya."
    },
    {
        "id": "IM_10",
        "category": "IMPLICIT_IDENTITY",
        "expected": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
        "text": "Setelah transaksi sukses baru sadar nama merchant QRIS terdaftar lokasinya di Medan padahal saya belinya di Bandung."
    },

    # ---------------------------------------------------------
    # 3. PUNGUTAN BIAYA TAMBAHAN / SURCHARGE (10 Kasus)
    # Expected: PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE
    # ---------------------------------------------------------
    {
        "id": "SC_01",
        "category": "SURCHARGE_FEE",
        "expected": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
        "text": "Ada tambahan biaya 1000 dari harga aslinya saat bayar pakai QRIS padahal kan aturan BI melarang membebankan biaya admin ke pembeli."
    },
    {
        "id": "SC_02",
        "category": "SURCHARGE_FEE",
        "expected": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
        "text": "Penjual minta saya lebihkan uang seribu rupiah buat biaya MDR katanya kalau bayar non tunai lewat barcode QRIS."
    },
    {
        "id": "SC_03",
        "category": "SURCHARGE_FEE",
        "expected": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
        "text": "Saat saya hendak membayar belanjaan, kasir secara sepihak meminta saya menambahkan biaya ekstra sebesar Rp1.000 di luar total harga pada struk dengan alasan biaya admin transaksi non tunai QRIS."
    },
    {
        "id": "SC_04",
        "category": "SURCHARGE_FEE",
        "expected": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
        "text": "Harga belanjaan dinaikkan kalau bayar pakai QRIS statis di toko ini, kasir membebankan potongan fee bank kepada pembeli."
    },
    {
        "id": "SC_05",
        "category": "SURCHARGE_FEE",
        "expected": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
        "text": "Dikenakan biaya surcharge 2000 oleh kasir toko kelontong dengan alasan potongan biaya aplikasi QRIS."
    },
    {
        "id": "SC_06",
        "category": "SURCHARGE_FEE",
        "expected": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
        "text": "Kasir meminta uang tambahan seribu per transaksi jika konsumen menggunakan barcode QRIS di etalase kasir."
    },
    {
        "id": "SC_07",
        "category": "SURCHARGE_FEE",
        "expected": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
        "text": "Total belanja 45 ribu tapi pas discan kasir nyuruh saya ketik 47 ribu di layar HP buat biaya charge admin QRIS."
    },
    {
        "id": "SC_08",
        "category": "SURCHARGE_FEE",
        "expected": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
        "text": "Pungutan liar biaya admin QRIS Rp2.000 sangat memberatkan konsumen, tolong tegur merchant ini."
    },
    {
        "id": "SC_09",
        "category": "SURCHARGE_FEE",
        "expected": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
        "text": "Kasir menolak mentransaksikan harga normal jika bayar lewat QRIS dan mewajibkan pembeli menambah nominal biaya ekstra seribu rupiah."
    },
    {
        "id": "SC_10",
        "category": "SURCHARGE_FEE",
        "expected": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
        "text": "Toko mengenakan biaya tambahan 1500 rupiah khusus untuk transaksi menggunakan pembayaran digital QRIS."
    },

    # ---------------------------------------------------------
    # 4. KUALITAS LAYANAN MERCHANT (10 Kasus)
    # Expected: KUALITAS_LAYANAN_MERCHANT
    # ---------------------------------------------------------
    {
        "id": "LM_01",
        "category": "MERCHANT_SERVICE",
        "expected": "KUALITAS_LAYANAN_MERCHANT",
        "text": "Kasir di toko ini menolak pembayaran QRIS secara sepihak dengan alasan minimal belanja harus di atas Rp30.000 padahal aturan resmi perbankan melarang pembatasan minimal transaksi QRIS."
    },
    {
        "id": "LM_02",
        "category": "MERCHANT_SERVICE",
        "expected": "KUALITAS_LAYANAN_MERCHANT",
        "text": "Kasir bersikap sangat judes dan ketus ketika saya bilang mau bayar pakai QRIS, kasir memaksa pembeli harus bayar uang tunai saja."
    },
    {
        "id": "LM_03",
        "category": "MERCHANT_SERVICE",
        "expected": "KUALITAS_LAYANAN_MERCHANT",
        "text": "Proses pembayaran di kasir sangat dipersulit oleh staf kasir yang beralasan barcode QRIS sedang disimpan dan malas mengeluarkan akrilik kasir."
    },
    {
        "id": "LM_04",
        "category": "MERCHANT_SERVICE",
        "expected": "KUALITAS_LAYANAN_MERCHANT",
        "text": "Kasir menahan barang belanjaan saya dengan sikap mencurigai pembeli secara tidak sopan padahal notifikasi bukti pembayaran QRIS sukses di aplikasi sudah saya perlihatkan."
    },
    {
        "id": "LM_05",
        "category": "MERCHANT_SERVICE",
        "expected": "KUALITAS_LAYANAN_MERCHANT",
        "text": "Merchant secara sepihak menolak menerima pembayaran QRIS dari aplikasi dompet digital tertentu dengan alasan kasir tidak paham, sangat mengecewakan layanannya."
    },
    {
        "id": "LM_06",
        "category": "MERCHANT_SERVICE",
        "expected": "KUALITAS_LAYANAN_MERCHANT",
        "text": "Kasir tidak mau scan barcode QRIS karena alasan antrean lagi ramai dan maunya cash saja, pembeli merasa dirugikan pelayanannya."
    },
    {
        "id": "LM_07",
        "category": "MERCHANT_SERVICE",
        "expected": "KUALITAS_LAYANAN_MERCHANT",
        "text": "Sikap kasir sangat buruk saat melayani pembayaran QRIS, malah memarahi pembeli yang tidak bawa uang pecahan pas."
    },
    {
        "id": "LM_08",
        "category": "MERCHANT_SERVICE",
        "expected": "KUALITAS_LAYANAN_MERCHANT",
        "text": "Kasir nolak QRIS dengan alasan sinyal toko lagi jelek padahal saya cek internet lancar, kasir malas cek mutasi masuk."
    },
    {
        "id": "LM_09",
        "category": "MERCHANT_SERVICE",
        "expected": "KUALITAS_LAYANAN_MERCHANT",
        "text": "Pembeli dipersulit saat mau bayar QRIS, kasir bilang QRIS hanya boleh untuk transaksi di atas 50 ribu saja."
    },
    {
        "id": "LM_10",
        "category": "MERCHANT_SERVICE",
        "expected": "KUALITAS_LAYANAN_MERCHANT",
        "text": "Pelayanan staf kasir toko sangat mengecewakan saat transaksi pembayaran QRIS, tidak ada sapaan dan tidak kooperatif memeriksa soundbox."
    },

    # ---------------------------------------------------------
    # 5. FEEDBACK AMBIGU / KURANG INFORMASI (10 Kasus)
    # Expected: FEEDBACK_AMBIGU
    # ---------------------------------------------------------
    {
        "id": "AM_01",
        "category": "AMBIGUOUS_EDGE",
        "expected": "FEEDBACK_AMBIGU",
        "text": "Saya sudah memastikan nominal belanja sebelum melakukan pembayaran dan prosesnya juga langsung selesai tanpa kendala, namun beberapa menit kemudian saya membuka kembali bukti transaksi karena ingin memastikan semuanya sudah benar, lalu saya menyadari bahwa ada bagian dari informasi pembayaran yang tidak seperti yang saya bayangkan ketika pertama kali melakukan transaksi."
    },
    {
        "id": "AM_02",
        "category": "AMBIGUOUS_EDGE",
        "expected": "FEEDBACK_AMBIGU",
        "text": "Tadi waktu bayar rasanya ada yang janggal deh pas saya scan barcode di kasir."
    },
    {
        "id": "AM_03",
        "category": "AMBIGUOUS_EDGE",
        "expected": "FEEDBACK_AMBIGU",
        "text": "Entah kenapa transaksi saya tadi agak aneh pas selesai, tolong dicek ya tim LaQris."
    },
    {
        "id": "AM_04",
        "category": "AMBIGUOUS_EDGE",
        "expected": "FEEDBACK_AMBIGU",
        "text": "Saya bayar pakai qris barusan di minimarket tapi kok perasaanku gak enak ya."
    },
    {
        "id": "AM_05",
        "category": "AMBIGUOUS_EDGE",
        "expected": "FEEDBACK_AMBIGU",
        "text": "Tadi mbak kasirnya sempat ngomong sesuatu pas saya lagi scan barcode tapi gak kedengeran jelas."
    },
    {
        "id": "AM_06",
        "category": "AMBIGUOUS_EDGE",
        "expected": "FEEDBACK_AMBIGU",
        "text": "Ada yang beda waktu saya coba transaksi barusan di meja depan."
    },
    {
        "id": "AM_07",
        "category": "AMBIGUOUS_EDGE",
        "expected": "FEEDBACK_AMBIGU",
        "text": "Pembayarannya selesai sih cuma saya bingung aja sama tampilannya."
    },
    {
        "id": "AM_08",
        "category": "AMBIGUOUS_EDGE",
        "expected": "FEEDBACK_AMBIGU",
        "text": "Kayaknya tadi ada hal yang gak biasa waktu saya mau bayar belanjaan saya."
    },
    {
        "id": "AM_09",
        "category": "AMBIGUOUS_EDGE",
        "expected": "FEEDBACK_AMBIGU",
        "text": "Transaksi qris barusan apakah sudah terverifikasi dengan benar di sistem kalian?"
    },
    {
        "id": "AM_10",
        "category": "AMBIGUOUS_EDGE",
        "expected": "FEEDBACK_AMBIGU",
        "text": "Saya tadi bayar tapi lupa liat detailnya langsung saya close aplikasinya."
    }
]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    model_dir = os.path.join(root_dir, "models", "indobert_qris_feedback", "best_model")

    if not os.path.exists(model_dir):
        print(f"[ERROR] Model tidak ditemukan di: {model_dir}")
        return

    print("=" * 80)
    print("LAQRIS FEEDBACK CLASSIFIER - STRESS BENCHMARK SUITE EMRS (50 KASUS)")
    print("Mengevaluasi Kualitas QRIS, Fraud, Surcharge, Layanan Merchant, dan Ambigu")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on Device: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    id2label = model.config.id2label

    results = []
    correct_count = 0

    print("\n[INFO] Menjalankan 50 Stress Test Cases...\n")
    print(f"{'ID':<6} | {'Kategori Test':<18} | {'Prediksi Model':<32} | {'Conf':<7} | {'Status'}")
    print("-" * 80)

    for case in TEST_CASES:
        text = case["text"]
        expected = case["expected"]
        cat_group = case["category"]

        inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            pred_label = id2label.get(pred_id, id2label.get(str(pred_id), f"LABEL_{pred_id}"))

        is_correct = (pred_label == expected)
        if is_correct:
            correct_count += 1
            status_str = "PASS"
            flag_mark = "[V]"
        else:
            status_str = "FAIL"
            flag_mark = "[X]"

        results.append({
            "id": case["id"],
            "group": cat_group,
            "text": text,
            "expected": expected,
            "predicted": pred_label,
            "confidence": round(confidence * 100, 2),
            "status": status_str,
            "is_correct": is_correct
        })

        print(f"{case['id']:<6} | {cat_group:<18} | {pred_label[:30]:<32} | {confidence*100:5.1f}% | {flag_mark} {status_str}")

    print("\n" + "=" * 80)
    print("HASIL ANALISIS BENCHMARK EMRS:")
    print("=" * 80)
    df_res = pd.DataFrame(results)
    for grp, grp_df in df_res.groupby("group"):
        pass_rate = (grp_df["is_correct"].sum() / len(grp_df)) * 100
        avg_conf = grp_df["confidence"].mean()
        print(f"- {grp:<18}: Pass = {grp_df['is_correct'].sum()}/{len(grp_df)} ({pass_rate:5.1f}%) | Avg Confidence: {avg_conf:5.1f}%")

    print("-" * 80)
    print(f"Total Kasus Diuji  : {len(TEST_CASES)}")
    print(f"Total Prediksi Benar: {correct_count}")
    print(f"Akurasi Keseluruhan : {(correct_count / len(TEST_CASES)) * 100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
