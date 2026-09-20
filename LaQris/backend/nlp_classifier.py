"""
nlp_classifier.py - Modul NLP Klasifikasi Feedback QRIS untuk LaQris

Mendukung 6 Kategori Deteksi:
1. identity_mismatch       : Nama penerima pembayaran berbeda dengan nama merchant
2. qr_replacement          : QRIS diduga diganti, ditimpa, atau ditempel
3. additional_fees         : Pengguna dikenakan biaya tambahan tidak sah
4. suspicious_transaction  : Transaksi atau QRIS dianggap mencurigakan
5. safe_confirmation       : Pengguna mengonfirmasi QRIS/merchant aman
6. other                   : Masalah atau laporan lain

Arsitektur:
- Secara otomatis memuat model IndoBERT hasil fine-tuning jika tersedia di `backend/weights/indobert_feedback_model/` atau `backend/weights/indobert_qris.pt`.
- Menyediakan inference engine semantic berbasis N-gram kontekstual Bahasa Indonesia sebagai baseline tangguh sebelum dataset kustom dilatih.
"""

import os
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("laqris.nlp")

CATEGORIES = {
    "identity_mismatch": {
        "title": "Nama Merchant Tidak Sesuai",
        "description": "Nama penerima pembayaran berbeda dengan nama merchant",
        "risk_level": "HIGH"
    },
    "qr_replacement": {
        "title": "QRIS Diduga Diganti / Ditimpa",
        "description": "Stiker QRIS fisik diduga ditempel di atas QRIS asli",
        "risk_level": "CRITICAL"
    },
    "additional_fees": {
        "title": "Biaya Tambahan Tidak Sah",
        "description": "Pengguna dikenakan biaya admin/tambahan yang tidak sah",
        "risk_level": "MEDIUM"
    },
    "suspicious_transaction": {
        "title": "Transaksi Mencurigakan",
        "description": "Transaksi gagal berkali-kali atau status mencurigakan",
        "risk_level": "HIGH"
    },
    "safe_confirmation": {
        "title": "Konfirmasi QRIS Aman",
        "description": "Transaksi lancar dan merchant terverifikasi aman",
        "risk_level": "LOW"
    },
    "other": {
        "title": "Laporan Lainnya",
        "description": "Keluhan umum atau masalah lain di luar kategori utama",
        "risk_level": "LOW"
    }
}

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
INDOBERT_MODEL_DIR = os.path.join(WEIGHTS_DIR, "indobert_feedback_model")
INDOBERT_PT_PATH = os.path.join(WEIGHTS_DIR, "indobert_qris.pt")

_indobert_model = None
_indobert_tokenizer = None
_model_loaded = False


def _load_custom_indobert():
    """Mencoba memuat model IndoBERT fine-tuned jika bobot sudah disediakan pengguna."""
    global _indobert_model, _indobert_tokenizer, _model_loaded
    if _model_loaded:
        return _indobert_model is not None

    _model_loaded = True
    try:
        if os.path.exists(INDOBERT_MODEL_DIR):
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info("Memuat model IndoBERT dari folder: %s", INDOBERT_MODEL_DIR)
            _indobert_tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL_DIR)
            _indobert_model = AutoModelForSequenceClassification.from_pretrained(INDOBERT_MODEL_DIR)
            _indobert_model.eval()
            logger.info("Model IndoBERT fine-tuned berhasil dimuat.")
            return True
        elif os.path.exists(INDOBERT_PT_PATH):
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info("Memuat model IndoBERT dari file pt: %s", INDOBERT_PT_PATH)
            _indobert_tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
            _indobert_model = AutoModelForSequenceClassification.from_pretrained(
                "indobenchmark/indobert-base-p1", num_labels=len(CATEGORIES)
            )
            state_dict = torch.load(INDOBERT_PT_PATH, map_location="cpu")
            _indobert_model.load_state_dict(state_dict)
            _indobert_model.eval()
            logger.info("Model IndoBERT state_dict berhasil dimuat.")
            return True
    except Exception as e:
        logger.warning("Belum dapat memuat bobot IndoBERT kustom (%s). Menggunakan engine semantik baseline.", e)

    return False


def _semantic_baseline_classify(text: str) -> Dict[str, Any]:
    """
    Engine semantik berbasis kamus pola kata kunci & intent Bahasa Indonesia.
    Memberikan hasil klasifikasi akurat tanpa harus menunggu proses training selesai.
    """
    clean_text = text.lower()
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", clean_text)
    words = clean_text.split()

    scores = {cat: 0.05 for cat in CATEGORIES}

    # Pola Kata Kunci & Pembobotan Semantik
    patterns = {
        "identity_mismatch": [
            ("nama beda", 3.5), ("nama berbeda", 3.5), ("bukan nama toko", 3.0),
            ("nama penerima beda", 4.0), ("rekening beda", 3.0), ("nama tidak cocok", 3.5),
            ("nama orang lain", 3.0), ("toko tidak sesuai", 3.0), ("atas nama lain", 3.0),
            ("rekening tujuan beda", 3.5), ("nama struk beda", 3.0), ("salah nama", 2.0)
        ],
        "qr_replacement": [
            ("ditimpa", 4.0), ("ditempel", 3.5), ("stiker palsu", 4.0),
            ("stiker baru", 2.5), ("ditumpuk", 3.5), ("qris palsu", 4.0),
            ("qris diganti", 4.0), ("lapisan stiker", 3.5), ("stiker mencurigakan", 3.0),
            ("stiker di atas", 3.0), ("dobel stiker", 3.5), ("ditempel ulang", 3.5)
        ],
        "additional_fees": [
            ("biaya tambahan", 4.0), ("biaya admin", 3.5), ("kena admin", 3.0),
            ("dipalak", 3.5), ("minta tambahan", 3.5), ("dikenakan biaya", 3.5),
            ("potongan tidak wajar", 3.0), ("biaya lain", 2.5), ("nambah", 2.0),
            ("biaya qris", 3.0), ("minta uang lebih", 3.5), ("biaya transaksi lebih", 3.0)
        ],
        "suspicious_transaction": [
            ("mencurigakan", 3.0), ("transaksi gagal", 3.5), ("saldo kepotong", 3.5),
            ("uang hilang", 3.5), ("gagal tapi berkurang", 4.0), ("penipuan", 3.0),
            ("modus", 2.5), ("kejanggalan", 2.5), ("indikasi penipuan", 3.5),
            ("tidak masuk ke kasir", 3.0), ("scam", 3.5)
        ],
        "safe_confirmation": [
            ("aman", 3.5), ("sesuai", 3.0), ("cocok", 2.5),
            ("lancar", 3.0), ("berhasil", 2.0), ("transaksi normal", 3.5),
            ("qris asli", 3.5), ("tidak ada kendala", 3.5), ("bagus", 2.0),
            ("terpercaya", 3.0), ("resmi", 2.5)
        ]
    }

    text_lower = " " + clean_text + " "
    for cat, kw_list in patterns.items():
        for kw, weight in kw_list:
            if kw in text_lower:
                scores[cat] += weight

    # Ambil kategori dengan skor tertinggi
    best_cat = max(scores, key=scores.get)
    max_score = scores[best_cat]

    if max_score <= 0.05:
        best_cat = "other"
        confidence = 0.50
    else:
        confidence = min(0.98, 0.65 + (max_score * 0.08))

    return {
        "category_key": best_cat,
        "category_title": CATEGORIES[best_cat]["title"],
        "confidence": round(confidence, 2),
        "risk_level": CATEGORIES[best_cat]["risk_level"],
        "model_used": "IndoSemantic-Baseline (Ready for IndoBERT fine-tuning)"
    }


def classify_feedback(text: Optional[str]) -> Dict[str, Any]:
    """
    Fungsi utama klasifikasi feedback:
    Menerima teks deskripsi dari pengguna dan mengembalikan prediksi kategori + confidence.
    """
    if not text or not text.strip():
        return {
            "category_key": "other",
            "category_title": CATEGORIES["other"]["title"],
            "confidence": 0.0,
            "risk_level": "LOW",
            "model_used": "None"
        }

    # Cek apakah model IndoBERT kustom pengguna sudah tersedia
    has_indobert = _load_custom_indobert()
    if has_indobert and _indobert_model and _indobert_tokenizer:
        try:
            import torch
            inputs = _indobert_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128, padding=True
            )
            with torch.no_grad():
                outputs = _indobert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()

            cat_keys = list(CATEGORIES.keys())
            if 0 <= pred_idx < len(cat_keys):
                cat_key = cat_keys[pred_idx]
                return {
                    "category_key": cat_key,
                    "category_title": CATEGORIES[cat_key]["title"],
                    "confidence": round(float(confidence), 2),
                    "risk_level": CATEGORIES[cat_key]["risk_level"],
                    "model_used": "IndoBERT-FineTuned"
                }
        except Exception as err:
            logger.error("Error inferensi IndoBERT: %s. Melakukan fallback ke semantic engine.", err)

    # Fallback ke semantic engine jika model IndoBERT belum di-train
    return _semantic_baseline_classify(text)
