"""
CLI Inferensi untuk Memprediksi Kategori Feedback QRIS Statis - Versi LaQris EMRS
Mendukung input teks interaktif atau argumen terminal.
Selaras dengan Enhanced Merchant Reputation System (EMRS) LaQris.
"""

import os
import sys

CATEGORY_ACTIONS = {
    "PENIPUAN_STIKER_QRIS_PALSU": {
        "severity": "CRITICAL",
        "action": "Kirim peringatan darurat pembekuan sementara QRIS toko ke tim investigasi LaQris dan reset skor keaslian (Authenticity Score) ke 0."
    },
    "KETIDAKSESUAIAN_IDENTITAS_MERCHANT": {
        "severity": "HIGH",
        "action": "Tandai penalti Identity Match pada skor EMRS merchant dan instruksikan verifikasi ulang kesesuaian dokumen legalitas toko."
    },
    "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE": {
        "severity": "HIGH",
        "action": "Beri surat peringatan pelanggaran regulasi Bank Indonesia (larangan surcharge QRIS) dan kenakan penalti kepatuhan merchant."
    },
    "KONDISI_FISIK_QRIS_RUSAK": {
        "severity": "MEDIUM",
        "action": "Kirim notifikasi otomatis ke tim operasional LaQris untuk mengirimkan materi cetak stiker/stand akrilik QRIS baru kepada merchant."
    },
    "KUALITAS_LAYANAN_MERCHANT": {
        "severity": "MEDIUM",
        "action": "Catat keluhan penolakan transaksi/layanan kasir ke evaluasi bulanan merchant dan sesuaikan skor reputasi layanan."
    },
    "QRIS_NORMAL_MERCHANT_TERPERCAYA": {
        "severity": "LOW",
        "action": "Tidak ada eskalasi kendala. Feedback positif otomatis menambahkan poin reputasi Trust Score merchant di direktori LaQris."
    },
    "FEEDBACK_AMBIGU": {
        "severity": "MEDIUM",
        "action": "Informasi laporan belum spesifik. Sistem memasukkan ke antrean klarifikasi dan meminta pembeli menyertakan foto struk/detail transaksi."
    }
}

def load_predictor(model_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)

    if not model_path:
        model_path = os.path.join(root_dir, "models", "indobert_qris_feedback", "best_model")

    if not os.path.exists(model_path):
        return None, None, None

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model, model.config.id2label

def predict_feedback(text, tokenizer, model, id2label):
    if not model:
        # Heuristic fallback matching jika model checkpoint belum dilatih
        t = text.lower()
        if any(k in t for k in [
            "timpa", "palsu", "ditempel stiker", "lapisan stiker", "raba", "barcode ganda",
            "kotak amal", "nempel stiker", "nutupin qr", "stiker baru", "stiker tipis", "stiker palsu"
        ]):
            label = "PENIPUAN_STIKER_QRIS_PALSU"
            conf = 0.95
        elif any(k in t for k in [
            "rekening pribadi", "beda kota", "nama orang lain", "nama pemiliknya", "bukan nama toko",
            "beda jauh", "plang warung", "plang toko", "pt bodong", "nama perorangan", "rekening penampung",
            "tidak sesuai dengan nama toko", "bukan nama badan usaha"
        ]):
            label = "KETIDAKSESUAIAN_IDENTITAS_MERCHANT"
            conf = 0.93
        elif any(k in t for k in [
            "tambahan biaya", "biaya tambahan", "biaya admin", "surcharge", "dinaikin",
            "dinaikkan", "pungutan", "tambahan 1000", "harga aslinya", "lebih mahal",
            "kena biaya", "dikenakan biaya", "biaya seribu", "admin 1000", "admin 2rb",
            "fee admin", "mdr", "charge 2000", "biaya potongan", "biaya ekstra"
        ]):
            label = "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE"
            conf = 0.94
        elif any(k in t for k in [
            "sobek", "pudar", "kusam", "lecet", "luntur", "rusak", "pecah", "buram",
            "kotor", "minyak", "terkelupas", "ga kebaca", "gak kebaca", "kamera ga bisa",
            "gagal fokus", "terlipat", "akrilik lecet", "pola qr hilang"
        ]):
            label = "KONDISI_FISIK_QRIS_RUSAK"
            conf = 0.92
        elif any(k in t for k in [
            "nolak", "menolak", "minimal belanja", "judes", "ketus", "ogah", "kasir tidak mau",
            "alasan sepihak", "dipersulit", "tidak ramah", "kasir jutek", "nahan barang"
        ]):
            label = "KUALITAS_LAYANAN_MERCHANT"
            conf = 0.91
        elif any(k in t for k in [
            "aneh", "janggal", "perasaan gak enak", "perasaanku gak enak", "bingung", "ragu",
            "tidak yakin", "gak sempat lihat", "tutup aplikasi", "sempat ngomong sesuatu"
        ]):
            label = "FEEDBACK_AMBIGU"
            conf = 0.88
        else:
            label = "QRIS_NORMAL_MERCHANT_TERPERCAYA"
            conf = 0.96

        meta = CATEGORY_ACTIONS.get(label, {"severity": "LOW", "action": "-"})
        return {
            "text": text,
            "category": label,
            "confidence": conf,
            "severity": meta["severity"],
            "recommended_action": meta["action"],
            "engine": "Heuristic Rules (Jalankan train_indobert.py untuk IndoBERT bobot penuh)"
        }

    import torch
    import torch.nn.functional as F

    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        conf = probs[pred_idx].item()
        label = id2label[pred_idx]

    meta = CATEGORY_ACTIONS.get(label, {"severity": "LOW", "action": "-"})
    return {
        "text": text,
        "category": label,
        "confidence": round(conf, 4),
        "severity": meta["severity"],
        "recommended_action": meta["action"],
        "engine": "IndoBERT Fine-Tuned"
    }

def main():
    tokenizer, model, id2label = load_predictor()

    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        res = predict_feedback(input_text, tokenizer, model, id2label)
        print("\n" + "-" * 50)
        print(f"Teks Input        : {res['text']}")
        print(f"Kategori Prediksi : {res['category']}")
        print(f"Confidence        : {res['confidence']*100:.2f}%")
        print(f"Severity          : {res['severity']}")
        print(f"Rekomendasi Aksi  : {res['recommended_action']}")
        print(f"Engine Model      : {res['engine']}")
        print("-" * 50 + "\n")
    else:
        print("=" * 60)
        print("LaQris Feedback Classifier - Mode Interaktif")
        print("Ketik keluhan QRIS (atau 'exit' untuk keluar):")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nMasukkan teks: ").strip()
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                if not user_input:
                    continue
                res = predict_feedback(user_input, tokenizer, model, id2label)
                print(f"-> Prediksi: {res['category']} ({res['confidence']*100:.2f}%) [{res['severity']}]")
                print(f"   Aksi    : {res['recommended_action']}")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
