"""
nlp_classifier.py - Modul Kecerdasan Buatan (NLP) Pembaca Ulasan Pembeli
Berfungsi untuk membaca kalimat ulasan dari pengguna setelah berbelanja,
lalu mengelompokkannya secara otomatis ke dalam salah satu dari 7 kategori.

7 Kategori Deteksi LaQris:
1. PENIPUAN_STIKER_QRIS_PALSU        (Risiko Kritis  - Stiker QRIS ditimpa stiker penipu)
2. KETIDAKSESUAIAN_IDENTITAS_MERCHANT (Risiko Tinggi  - Nama rekening beda dari nama toko)
3. PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE (Risiko Tinggi  - Dimintai biaya admin tambahan)
4. KONDISI_FISIK_QRIS_RUSAK           (Risiko Sedang  - Stiker pudar, sobek, atau buram)
5. KUALITAS_LAYANAN_MERCHANT          (Risiko Sedang  - Kasir menolak QRIS / layanan kurang baik)
6. QRIS_NORMAL_MERCHANT_TERPERCAYA    (Risiko Rendah  - Transaksi lancar, aman, dan memuaskan)
7. FEEDBACK_AMBIGU                    (Perlu Pantau   - Ulasan terlalu singkat / belum jelas)
"""

import os
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("laqris.nlp")

# Rincian 7 Kategori Masalah beserta Tindakan Penanganannya
CATEGORIES = {
    "PENIPUAN_STIKER_QRIS_PALSU": {
        "title": "Penipuan Stiker QRIS Palsu / Ditimpa",
        "description": "Barcode fisik toko ditempeli stiker baru oleh oknum penipu",
        "severity": "CRITICAL",
        "action": "Kirim peringatan darurat ke tim investigasi dan kunci sementara status keaslian toko demi melindungi pembeli lain.",
        "emrs_penalty": "RESET_A"
    },
    "KETIDAKSESUAIAN_IDENTITAS_MERCHANT": {
        "title": "Ketidaksesuaian Identitas Toko",
        "description": "Nama penerima di aplikasi dompet digital berbeda dengan plang/nama toko sebenarnya",
        "severity": "HIGH",
        "action": "Beri tanda peringatan ketidaksesuaian identitas dan minta pemilik toko untuk verifikasi dokumen resmi.",
        "emrs_penalty": "PENALTY_IDENTITY"
    },
    "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE": {
        "title": "Pungutan Biaya Tambahan (Surcharge)",
        "description": "Pembeli ditarik biaya tambahan atau harga dinaikkan saat membayar menggunakan QRIS",
        "severity": "HIGH",
        "action": "Beri peringatan terkait larangan pungutan tambahan QRIS sesuai aturan Bank Indonesia.",
        "emrs_penalty": "PENALTY_COMPLIANCE"
    },
    "KONDISI_FISIK_QRIS_RUSAK": {
        "title": "Kondisi Fisik QRIS Rusak / Pudar",
        "description": "Stiker sobek, luntur kena panas, atau kusam sehingga susah dipindai kamera",
        "severity": "MEDIUM",
        "action": "Kirimkan pemberitahuan agar pemilik toko dapat mencetak ulang stiker QRIS yang baru dan jelas.",
        "emrs_penalty": "NOTICE_OPERATIONAL"
    },
    "KUALITAS_LAYANAN_MERCHANT": {
        "title": "Kualitas Layanan Toko",
        "description": "Kasir menolak menerima QRIS secara sepihak atau menentukan syarat minimal belanja",
        "severity": "MEDIUM",
        "action": "Catat masukan pelayanan ini sebagai bahan evaluasi kenyamanan pelanggan.",
        "emrs_penalty": "PENALTY_SERVICE"
    },
    "QRIS_NORMAL_MERCHANT_TERPERCAYA": {
        "title": "Transaksi Lancar & Toko Terpercaya",
        "description": "Pembayaran sukses tanpa kendala, nama penerima sesuai, dan kasir melayani dengan baik",
        "severity": "LOW",
        "action": "Ulasan positif ini otomatis menambah nilai reputasi dan kepercayaan toko di aplikasi LaQris.",
        "emrs_penalty": "BOOST_TRUST"
    },
    "FEEDBACK_AMBIGU": {
        "title": "Ulasan Kurang Spesifik / Perlu Rincian",
        "description": "Komentar terlalu pendek atau belum menyebutkan kejadian secara rinci",
        "severity": "MEDIUM",
        "action": "Ulasan dicatat dan sistem menyarankan pengguna melengkapi cerita jika menemukan kendala.",
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
    """
    Fungsi untuk memuat model kecerdasan buatan IndoBERT ke dalam memori.
    Model ini bertugas membaca teks ulasan berbahasa Indonesia dan memahami artinya.
    """
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

            logger.info("Memuat model kecerdasan buatan IndoBERT dari: %s", target_dir)
            _indobert_tokenizer = AutoTokenizer.from_pretrained(target_dir)
            _indobert_model = AutoModelForSequenceClassification.from_pretrained(target_dir)
            
            _model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _indobert_model.to(_model_device)
            _indobert_model.eval()
            
            logger.info("Model IndoBERT LaQris siap digunakan pada: %s", _model_device)
            return True
        except Exception as e:
            logger.warning("Model IndoBERT belum dapat dimuat (%s). Menggunakan sistem pencocokan kata cadangan.", e)

    return False


def _semantic_baseline_classify(text: str) -> Dict[str, Any]:
    """
    Sistem pencocokan kata cadangan (Semantic Fallback):
    Mencocokkan kata kunci keluhan atau pujian dalam kalimat ulasan secara cepat,
    sebagai cadangan jika model kecerdasan buatan utama sedang tidak aktif.
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
    Pelindung Ulasan Positif (Hybrid Guard):
    Memastikan ulasan pembeli yang bernada memuji dan menyatakan transaksi sukses
    (misal: "nama toko sama persis dan pembayaran lancar")
    tidak keliru dikategorikan sebagai masalah penipuan.
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
    Fungsi utama analisis ulasan pembeli:
    Menerima kalimat ulasan lalu menentukan kategori masalah, tingkat keparahan,
    serta rekomendasi tindak lanjut bagi toko.
    Dilengkapi pelindung otomatis agar ulasan kepuasan pelanggan dinilai secara akurat.
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

