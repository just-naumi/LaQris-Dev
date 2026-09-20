"""
nlp_classifier.py - Modul NLP Klasifikasi Feedback QRIS untuk LaQris EMRS
Tersambung langsung dengan Model IndoBERT Fine-Tuned (7 Kelas Resmi LaQris).

7 Kelas EMRS:
0. PENIPUAN_STIKER_QRIS_PALSU        (CRITICAL)
1. KETIDAKSESUAIAN_IDENTITAS_MERCHANT (HIGH)
2. PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE (HIGH)
3. KONDISI_FISIK_QRIS_RUSAK           (MEDIUM)
4. KUALITAS_LAYANAN_MERCHANT          (MEDIUM)
5. QRIS_NORMAL_MERCHANT_TERPERCAYA    (LOW)
6. FEEDBACK_AMBIGU                    (MEDIUM)
"""

import os
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("laqris.nlp")

CATEGORIES = {
    "PENIPUAN_STIKER_QRIS_PALSU": {
        "title": "Penipuan Stiker QRIS Palsu / Ditimpa",
        "description": "Barcode QRIS fisik ditempeli stiker baru atau ditimpa di atas akrilik resmi toko",
        "severity": "CRITICAL",
        "action": "Kirim peringatan darurat pembekuan sementara QRIS toko ke tim investigasi LaQris dan reset skor keaslian (Authenticity Score) ke 0.",
        "emrs_penalty": "RESET_A"
    },
    "KETIDAKSESUAIAN_IDENTITAS_MERCHANT": {
        "title": "Ketidaksesuaian Identitas Merchant",
        "description": "Nama penerima atau NMID di aplikasi berbeda dengan nama usaha resmi toko",
        "severity": "HIGH",
        "action": "Tandai penalti Identity Match pada skor EMRS merchant dan instruksikan verifikasi ulang kesesuaian dokumen legalitas toko.",
        "emrs_penalty": "PENALTY_IDENTITY"
    },
    "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE": {
        "title": "Pungutan Biaya Tambahan / Surcharge",
        "description": "Pengguna dikenakan biaya admin/surcharge atau markup harga saat membayar dengan QRIS",
        "severity": "HIGH",
        "action": "Beri surat peringatan pelanggaran regulasi Bank Indonesia (larangan surcharge QRIS) dan kenakan penalti kepatuhan merchant.",
        "emrs_penalty": "PENALTY_COMPLIANCE"
    },
    "KONDISI_FISIK_QRIS_RUSAK": {
        "title": "Kondisi Fisik QRIS Rusak / Pudar",
        "description": "Stiker sobek, luntur panas, retak minyak, atau buram sehingga sulit discan",
        "severity": "MEDIUM",
        "action": "Kirim notifikasi otomatis ke tim operasional LaQris untuk mengirimkan materi cetak stiker/stand akrilik QRIS baru kepada merchant.",
        "emrs_penalty": "NOTICE_OPERATIONAL"
    },
    "KUALITAS_LAYANAN_MERCHANT": {
        "title": "Kualitas Layanan Merchant & Kepatuhan",
        "description": "Kasir menolak pembayaran QRIS secara sepihak, menetapkan batas minimal belanja, atau bersikap judes",
        "severity": "MEDIUM",
        "action": "Catat keluhan penolakan transaksi/layanan kasir ke evaluasi bulanan merchant dan sesuaikan skor reputasi layanan.",
        "emrs_penalty": "PENALTY_SERVICE"
    },
    "QRIS_NORMAL_MERCHANT_TERPERCAYA": {
        "title": "QRIS Normal & Merchant Terpercaya",
        "description": "Transaksi berhasil lancar, barcode resmi, nominal sesuai, dan identitas terverifikasi",
        "severity": "LOW",
        "action": "Tidak ada eskalasi kendala. Feedback positif otomatis menambahkan poin reputasi Trust Score merchant di direktori LaQris.",
        "emrs_penalty": "BOOST_TRUST"
    },
    "FEEDBACK_AMBIGU": {
        "title": "Feedback Ambigu / Belum Spesifik",
        "description": "Komentar umum, tidak jelas, atau tanpa rincian bukti transaksi yang konkret",
        "severity": "MEDIUM",
        "action": "Informasi laporan belum spesifik. Sistem memasukkan ke antrean klarifikasi dan meminta pembeli menyertakan foto struk/detail transaksi.",
        "emrs_penalty": "NONE"
    }
}

# Pemetaan mundur untuk kompatibilitas input legacy
LEGACY_MAP = {
    "identity_mismatch": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
    "qr_replacement": "PENIPUAN_STIKER_QRIS_PALSU",
    "additional_fees": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
    "suspicious_transaction": "FEEDBACK_AMBIGU",
    "safe_confirmation": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
    "other": "FEEDBACK_AMBIGU",
    "QRIS Replacement": "PENIPUAN_STIKER_QRIS_PALSU",
    "Merchant Mismatch": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
    "Additional Fee": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
    "General Complaint": "KUALITAS_LAYANAN_MERCHANT",
    "Verified Authentic": "QRIS_NORMAL_MERCHANT_TERPERCAYA"
}

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, "..", ".."))

# Path prioritas utama: Model IndoBERT yang baru selesai difine-tune
PRIMARY_INDOBERT_DIR = os.path.join(
    PROJECT_ROOT, "LaQris Feedback Classification", "models", "indobert_qris_feedback", "best_model"
)
FALLBACK_INDOBERT_DIR = os.path.join(BACKEND_DIR, "weights", "indobert_feedback_model")

_indobert_model = None
_indobert_tokenizer = None
_model_device = None
_model_loaded = False


def _load_custom_indobert():
    """Memuat model IndoBERT fine-tuned 7-kelas dari direktori training."""
    global _indobert_model, _indobert_tokenizer, _model_device, _model_loaded
    if _model_loaded:
        return _indobert_model is not None

    _model_loaded = True
    target_dir = None
    if os.path.exists(PRIMARY_INDOBERT_DIR):
        target_dir = PRIMARY_INDOBERT_DIR
    elif os.path.exists(FALLBACK_INDOBERT_DIR):
        target_dir = FALLBACK_INDOBERT_DIR

    if target_dir:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info("Memuat model IndoBERT 7-Kelas dari: %s", target_dir)
            _indobert_tokenizer = AutoTokenizer.from_pretrained(target_dir)
            _indobert_model = AutoModelForSequenceClassification.from_pretrained(target_dir)
            
            _model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _indobert_model.to(_model_device)
            _indobert_model.eval()
            
            logger.info("Model IndoBERT LaQris EMRS berhasil dimuat di device: %s", _model_device)
            return True
        except Exception as e:
            logger.warning("Gagal memuat model IndoBERT (%s). Menggunakan semantic fallback.", e)

    return False


def _semantic_baseline_classify(text: str) -> Dict[str, Any]:
    """
    Engine semantik baseline berbasis N-Gram kontekstual & operator relasional
    sebagai fallback instan jika model neural network belum dapat diakses.
    """
    clean_text = text.lower()
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", clean_text)
    text_lower = " " + clean_text + " "

    scores = {cat: 0.05 for cat in CATEGORIES}

    patterns = {
        "PENIPUAN_STIKER_QRIS_PALSU": [
            ("ditimpa", 4.0), ("ditempel", 3.5), ("stiker palsu", 4.5),
            ("stiker baru", 3.0), ("ditumpuk", 4.0), ("qris palsu", 4.5),
            ("lapisan stiker", 3.5), ("dobel stiker", 4.0), ("akrilik ditutup", 3.5),
            ("stiker siluman", 4.0)
        ],
        "KETIDAKSESUAIAN_IDENTITAS_MERCHANT": [
            ("nama beda", 3.5), ("nama berbeda", 4.0), ("bukan nama toko", 4.0),
            ("tidak sesuai", 3.0), ("rekening pribadi", 4.0), ("nama orang lain", 3.5),
            ("nama tidak cocok", 3.5), ("rekening perorangan", 3.5), ("bukan nama usaha", 3.5)
        ],
        "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE": [
            ("biaya tambahan", 4.5), ("surcharge", 4.5), ("biaya admin", 4.0),
            ("kena admin", 3.5), ("dipalak", 3.5), ("minta tambahan", 4.0),
            ("cas tambahan", 4.0), ("nambah 1000", 4.5), ("nambah 2000", 4.5),
            ("fee admin", 4.0), ("lebih mahal", 3.0)
        ],
        "KONDISI_FISIK_QRIS_RUSAK": [
            ("sobek", 4.0), ("luntur", 4.0), ("pudar", 4.0), ("pecah", 3.5),
            ("tergores", 3.5), ("retak", 3.5), ("buram", 3.5), ("kotor", 3.0),
            ("tidak terbaca", 3.5), ("minyak", 3.0), ("kusam", 3.5)
        ],
        "KUALITAS_LAYANAN_MERCHANT": [
            ("menolak", 4.0), ("tolak qris", 4.5), ("minimal belanja", 4.5),
            ("hanya tunai", 4.0), ("judes", 4.0), ("ketus", 4.0),
            ("marah", 3.5), ("disembunyikan", 4.0), ("malas", 3.5),
            ("tidak mau terima", 4.0)
        ],
        "QRIS_NORMAL_MERCHANT_TERPERCAYA": [
            ("sama persis", 4.5), ("sesuai", 3.5), ("cocok", 3.5),
            ("lancar", 3.5), ("berhasil", 3.0), ("normal", 3.5),
            ("tanpa kendala", 4.0), ("resmi", 3.5), ("puas", 3.0),
            ("tidak ada biaya", 3.5), ("ramah", 3.0)
        ]
    }

    for cat, kw_list in patterns.items():
        for kw, weight in kw_list:
            if kw in text_lower:
                scores[cat] += weight

    best_cat = max(scores, key=scores.get)
    max_score = scores[best_cat]

    if max_score <= 0.10:
        best_cat = "FEEDBACK_AMBIGU"
        confidence = 0.65
    else:
        confidence = min(0.98, 0.70 + (max_score * 0.05))

    cat_meta = CATEGORIES[best_cat]
    return {
        "category_key": best_cat,
        "category_title": cat_meta["title"],
        "confidence": round(confidence, 4),
        "severity": cat_meta["severity"],
        "action": cat_meta["action"],
        "emrs_penalty": cat_meta["emrs_penalty"],
        "model_used": "IndoSemantic-Heuristic-Fallback"
    }


def _apply_hybrid_guard(text: str, predicted_cat: str, confidence: float) -> tuple:
    """
    Hybrid Guard (P1 Item 11 di Readme.md):
    Mendeteksi pernyataan identitas positif / ulasan transaksi normal yang sering mengalami
    boundary false-positive pada model klasifikasi teks (misal: "nama merchant sama dengan nama toko").
    Mencegah feedback positif salah diprediksi sebagai KETIDAKSESUAIAN_IDENTITAS_MERCHANT.
    """
    text_clean = text.lower()
    positive_cues = [
        "nama merchant sama", "nama toko sama", "sama dengan nama toko", "sama persis",
        "nominal sesuai", "jumlah sesuai", "pembayaran berhasil", "tanpa kendala",
        "transaksi lancar", "lancar jaya", "kasir ramah", "resmi", "tidak ada biaya",
        "langsung masuk sesuai nama"
    ]
    negative_cues = [
        "beda", "berbeda", "tidak sesuai", "bukan nama", "salah nama", "nama orang lain",
        "ditimpa", "ditempel", "stiker palsu", "palsu", "rusak", "sobek", "luntur",
        "biaya tambahan", "surcharge", "dipalak", "menolak", "tolak qris", "judes"
    ]
    has_positive = any(cue in text_clean for cue in positive_cues)
    has_negative = any(cue in text_clean for cue in negative_cues)

    if has_positive and not has_negative:
        if predicted_cat in ["KETIDAKSESUAIAN_IDENTITAS_MERCHANT", "FEEDBACK_AMBIGU"]:
            return "QRIS_NORMAL_MERCHANT_TERPERCAYA", max(confidence, 0.94)
    return predicted_cat, confidence


def classify_feedback(text: Optional[str]) -> Dict[str, Any]:
    """
    Fungsi utama klasifikasi feedback:
    Menerima teks deskripsi dan mengembalikan kategori 7-kelas EMRS,
    confidence, severity, rekomendasi aksi, dan tipe penalti EMRS.
    Dilengkapi Hybrid Guard untuk mencegah boundary false-positive (P1 Item 11).
    """
    if not text or not text.strip():
        cat_meta = CATEGORIES["FEEDBACK_AMBIGU"]
        return {
            "category_key": "FEEDBACK_AMBIGU",
            "category_title": cat_meta["title"],
            "confidence": 0.0,
            "severity": cat_meta["severity"],
            "action": cat_meta["action"],
            "emrs_penalty": cat_meta["emrs_penalty"],
            "model_used": "None"
        }

    # Cek ketersediaan model IndoBERT fine-tuned
    has_indobert = _load_custom_indobert()
    if has_indobert and _indobert_model and _indobert_tokenizer:
        try:
            import torch
            inputs = _indobert_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128, padding=True
            ).to(_model_device)
            
            with torch.no_grad():
                outputs = _indobert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()

            id2label = _indobert_model.config.id2label
            if id2label and pred_idx in id2label:
                cat_key = id2label[pred_idx]
            else:
                cat_keys = list(CATEGORIES.keys())
                cat_key = cat_keys[pred_idx] if 0 <= pred_idx < len(cat_keys) else "FEEDBACK_AMBIGU"

            # Terapkan Hybrid Guard (P1 Item 11)
            cat_key, confidence = _apply_hybrid_guard(text, cat_key, confidence)

            if cat_key in CATEGORIES:
                cat_meta = CATEGORIES[cat_key]
                return {
                    "category_key": cat_key,
                    "category_title": cat_meta["title"],
                    "confidence": round(float(confidence), 4),
                    "severity": cat_meta["severity"],
                    "action": cat_meta["action"],
                    "emrs_penalty": cat_meta["emrs_penalty"],
                    "model_used": "IndoBERT-FineTuned-EMRS"
                }
        except Exception as err:
            logger.error("Error inferensi IndoBERT: %s. Fallback ke semantic baseline.", err)

    # Fallback ke semantic engine jika inferensi neural network gagal
    res = _semantic_baseline_classify(text)
    guarded_key, guarded_conf = _apply_hybrid_guard(text, res["category_key"], res["confidence"])
    if guarded_key != res["category_key"]:
        meta = CATEGORIES[guarded_key]
        res["category_key"] = guarded_key
        res["category_title"] = meta["title"]
        res["confidence"] = guarded_conf
        res["severity"] = meta["severity"]
        res["action"] = meta["action"]
        res["emrs_penalty"] = meta["emrs_penalty"]
    return res

