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

# ─────────────────────────────────────────────────────────────────────────────
# Kamus Normalisasi Slang / Singkatan Bahasa Indonesia (Preprocessing)
# Mengubah kata gaul, singkatan, dan typo umum ke bentuk baku
# sebelum teks dianalisis oleh model atau keyword matcher.
# ─────────────────────────────────────────────────────────────────────────────
SLANG_MAP = {
    # Singkatan umum
    "bgt":    "banget",
    "bgtt":   "banget",
    "bngt":   "banget",
    "bgt":    "banget",
    "yg":     "yang",
    "dgn":    "dengan",
    "dg":     "dengan",
    "krn":    "karena",
    "karna":  "karena",
    "utk":    "untuk",
    "tdk":    "tidak",
    "gak":    "tidak",
    "gk":     "tidak",
    "ga":     "tidak",
    "nggak":  "tidak",
    "ngga":   "tidak",
    "enggak": "tidak",
    "g":      "tidak",
    "sm":     "sama",
    "sama2":  "sama",
    "aja":    "saja",
    "aj":     "saja",
    "dah":    "sudah",
    "udah":   "sudah",
    "udh":    "sudah",
    "sdh":    "sudah",
    "emg":    "memang",
    "emang":  "memang",
    "hrs":    "harus",
    "kl":     "kalau",
    "klo":    "kalau",
    "kalo":   "kalau",
    "blm":    "belum",
    "blum":   "belum",
    "msh":    "masih",
    "tmn":    "teman",
    "nih":    "ini",
    "tuh":    "itu",
    "tp":     "tapi",
    "tpi":    "tapi",
    "cz":     "karena",
    "soalnya":"karena",
    "pdhl":   "padahal",
    "mau":    "mau",
    "mo":     "mau",

    # Kata positif pendek / informal
    "ok":       "oke",
    "oke":      "oke",
    "okelah":   "oke",
    "sip":      "oke",
    "sipp":     "oke",
    "sippp":    "oke",
    "jos":      "bagus",
    "joss":     "bagus",
    "josss":    "bagus",
    "mantap":   "mantap",
    "mntap":    "mantap",
    "mantul":   "mantap",
    "keren":    "bagus",
    "kece":     "bagus",
    "bgs":      "bagus",
    "bgus":     "bagus",
    "baguss":   "bagus",
    "baguus":   "bagus",
    "good":     "bagus",
    "nice":     "bagus",
    "top":      "bagus",
    "topp":     "bagus",
    "toppp":    "bagus",
    "work":     "berhasil",
    "works":    "berhasil",
    "aman":     "aman",
    "amaan":    "aman",
    "amann":    "aman",
    "puas":     "puas",
    "puass":    "puas",
    "lancar":   "lancar",
    "lancer":   "lancar",
    "beres":    "selesai",
    "bres":     "selesai",
    "sukses":   "berhasil",
    "suksess":  "berhasil",
    "berhasil": "berhasil",
    "works":    "berhasil",
    "fine":     "oke",

    # Kata negatif pendek / informal
    "palsu":    "palsu",
    "plsu":     "palsu",
    "palzuu":   "palsu",
    "paisu":    "palsu",
    "fake":     "palsu",
    "rusak":    "rusak",
    "rsk":      "rusak",
    "ancur":    "rusak",
    "hancur":   "rusak",
    "beda":     "berbeda",
    "bdo":      "berbeda",
    "bedo":     "berbeda",
    "bdes":     "berbeda",
    "berbeda":  "berbeda",
    "salah":    "salah",
    "curiga":   "curiga",
    "mencurigakan": "curiga",
    "tipu":     "penipuan",
    "nipu":     "penipuan",
    "ditipu":   "penipuan",
    "scam":     "penipuan",
    "bohong":   "penipuan",
    "nabrak":   "ditimpa",
    "ditimpa":  "ditimpa",
    "sobek":    "sobek",
    "sbk":      "sobek",
    "pudar":    "pudar",
    "pdr":      "pudar",
    "kotor":    "kotor",
    "ktr":      "kotor",
    "mahal":    "mahal",
    "kemahalan":"mahal",
    "dipalak":  "dipalak",
    "pungli":   "pungutan",
    "punglii":  "pungutan",
}

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


def _normalize_slang(text: str) -> str:
    """
    Preprocessing — Normalisasi Slang & Singkatan Bahasa Indonesia.
    Mengubah kata gaul, singkatan, dan typo umum ke bentuk baku
    agar keyword matcher dan IndoBERT bisa mengenali maknanya dengan lebih akurat.

    Contoh:
        "ok bgt aman" → "oke banget aman"
        "palzuu gk sesuai" → "palsu tidak sesuai"
        "joss mantul" → "bagus mantap"
    """
    # Hapus karakter berulang berlebihan ("amaaann" → "amaan" → diproses lanjut)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Tokenisasi sederhana per kata, lalu normalisasi satu per satu
    words = text.lower().split()
    normalized = [SLANG_MAP.get(w, w) for w in words]
    return " ".join(normalized)


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
            ("stiker siluman", 4.0),
            # Kata informal / sehari-hari
            ("palsu", 3.0), ("kayaknya palsu", 4.5), ("sepertinya palsu", 4.5),
            ("terindikasi palsu", 4.5), ("dicurigai palsu", 4.5),
            ("qris-nya palsu", 4.5), ("qrisnya palsu", 4.5),
            ("seperti ditimpa", 4.0), ("kayak ditempel", 4.0),
            # Hasil normalisasi slang
            ("penipuan", 3.5), ("curiga", 3.0), ("fake", 3.5)
        ],
        "KETIDAKSESUAIAN_IDENTITAS_MERCHANT": [
            ("nama beda", 3.5), ("nama berbeda", 4.0), ("bukan nama toko", 4.0),
            ("tidak sesuai", 3.0), ("rekening pribadi", 4.0), ("nama orang lain", 3.5),
            ("nama tidak cocok", 3.5), ("rekening perorangan", 3.5), ("bukan nama usaha", 3.5),
            # Frasa informal / kontekstual
            ("tidak sesuai dengan merchant", 4.5), ("tidak sesuai merchant", 4.5),
            ("tidak cocok dengan toko", 4.5), ("datanya tidak sesuai", 4.0),
            ("menunjukkan tidak sesuai", 4.0), ("data tidak sesuai", 4.0),
            ("tidak sesuai", 4.5), ("nama merchant tidak sesuai", 4.5),
            ("namanya beda", 4.0), ("beda sama toko", 4.0), ("nama tokonya beda", 4.5),
            # Hasil normalisasi slang ("beda" -> "berbeda", urutan kata berubah)
            ("berbeda", 3.0), ("salah nama", 3.5), ("nama salah", 3.5)
        ],
        "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE": [
            ("biaya tambahan", 4.5), ("surcharge", 4.5), ("biaya admin", 4.0),
            ("kena admin", 3.5), ("dipalak", 3.5), ("minta tambahan", 4.0),
            ("cas tambahan", 4.0), ("nambah 1000", 4.5), ("nambah 2000", 4.5),
            ("fee admin", 4.0), ("lebih mahal", 3.0),
            # Hasil normalisasi slang ("pungli" -> "pungutan", "mahal" sudah di map)
            ("pungutan", 3.5), ("mahal", 2.5)
        ],
        "KONDISI_FISIK_QRIS_RUSAK": [
            ("sobek", 4.0), ("luntur", 4.0), ("pudar", 4.0), ("pecah", 3.5),
            ("tergores", 3.5), ("retak", 3.5), ("buram", 3.5), ("kotor", 3.0),
            ("tidak terbaca", 3.5), ("minyak", 3.0), ("kusam", 3.5),
            ("tidak bisa scan", 4.0), ("gak bisa scan", 4.0), ("susah discan", 4.0)
        ],
        "KUALITAS_LAYANAN_MERCHANT": [
            ("menolak", 4.0), ("tolak qris", 4.5), ("minimal belanja", 4.5),
            ("hanya tunai", 4.0), ("judes", 4.0), ("ketus", 4.0),
            ("marah", 3.5), ("disembunyikan", 4.0), ("malas", 3.5),
            ("tidak mau terima", 4.0), ("kasir judes", 4.5), ("pelayanan buruk", 4.0),
            ("pelayanan jelek", 4.0), ("tidak bagus", 4.0), ("kurang ramah", 4.0),
            ("tidak ramah", 4.0), ("tidak aman", 4.0), ("gak lancar", 4.0),
            ("tidak lancar", 4.0), ("kecewa", 3.5), ("buruk", 3.5)
        ],
        "QRIS_NORMAL_MERCHANT_TERPERCAYA": [
            ("sama persis", 4.5), ("sesuai", 3.5), ("cocok", 3.5),
            ("lancar", 3.5), ("berhasil", 3.0), ("normal", 3.5),
            ("tanpa kendala", 4.0), ("resmi", 3.5), ("puas", 3.0),
            ("tidak ada biaya", 3.5), ("ramah", 3.0),
            # Kata positif pendek / informal (setelah slang normalization)
            ("bagus", 3.0), ("mantap", 3.0), ("oke", 2.5), ("aman", 3.0),
            ("selesai", 2.5), ("sukses", 3.0), ("top", 2.5),
            ("tidak ada masalah", 4.0), ("tidak ada kendala", 4.0),
            ("sangat puas", 4.0), ("sangat lancar", 4.0), ("sangat aman", 4.0),
            ("memuaskan", 3.5), ("terpercaya", 3.5), ("terjamin", 3.5)
        ]
    }

    for cat, kw_list in patterns.items():
        for kw, weight in kw_list:
            if kw in text_lower:
                if cat == "QRIS_NORMAL_MERCHANT_TERPERCAYA":
                    # Pastikan kata positif tidak dinegasi (misal: "tidak bagus", "gak lancar", "kurang ramah")
                    neg_pattern = rf"\b(tidak|bukan|kurang|gak|nggak|ga|tdk|belum|bkn)\s+(\w+\s+)?{re.escape(kw)}\b"
                    if re.search(neg_pattern, text_lower):
                        scores["KUALITAS_LAYANAN_MERCHANT"] += weight
                        continue
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


def _apply_hybrid_guard(text: str, predicted_cat: str, confidence: float, normalized_text: str = "") -> tuple:
    """
    Pelindung Ulasan Hybrid Guard & Ambiguity Resolver Cerdas (P1 Item 11):
    1. Positive Guard  — ulasan kepuasan/pujian yang keliru dikategorikan bermasalah/ambigu di-override ke NORMAL
    2. Negative Guard  — ulasan bermasalah (palsu, mismatch, surcharge, rusak, layanan) yang keliru
       diprediksi sebagai NORMAL oleh IndoBERT di-override ke kategori yang tepat.
    3. Ambiguity Resolver — jika IndoBERT memprediksi FEEDBACK_AMBIGU karena ulasan singkat atau informal,
       guard memeriksa kata kunci spesifik domain QRIS dan semantic engine untuk mengembalikan hasil yang rinci.
    """
    t_clean = text.lower()
    n_clean = (normalized_text or _normalize_slang(text)).lower()
    combined = f" {t_clean} {n_clean} "

    # 1. Cues per Kategori Masalah
    fraud_cues = [
        "stiker palsu", "qris palsu", "qris-nya palsu", "qrisnya palsu", "kayaknya palsu",
        "sepertinya palsu", "terindikasi palsu", "dicurigai palsu", "stiker siluman",
        "dobel stiker", "lapisan stiker", "stiker baru", "ditumpuk", "ditimpa stiker",
        "ditimpa", "ditempel", "palsu", "fake", "penipuan", "scam"
    ]

    mismatch_cues = [
        "tidak sesuai dengan merchant", "tidak sesuai merchant", "datanya tidak sesuai",
        "menunjukkan tidak sesuai", "data tidak sesuai", "tidak cocok dengan toko",
        "nama merchant tidak sesuai", "nama toko tidak sesuai", "tidak sesuai nama toko",
        "namanya beda", "nama tokonya beda", "nama toko beda", "beda sama toko",
        "nama beda", "nama berbeda", "namanya berbeda", "bukan nama toko", "bukan nama",
        "nama orang lain", "rekening pribadi", "rekening perorangan", "salah nama",
        "nama salah", "nama tidak cocok", "beda dengan nama", "berbeda dengan nama"
    ]

    surcharge_cues = [
        "biaya tambahan", "surcharge", "biaya admin", "kena admin", "dipalak",
        "minta tambahan", "cas tambahan", "charge tambahan", "nambah 1000", "nambah 2000",
        "nambah 500", "fee admin", "lebih mahal", "ada biaya tambahan", "pungutan", "pungli"
    ]

    damaged_cues = [
        "sobek", "luntur", "pudar", "pecah", "tergores", "retak", "buram", "kotor",
        "tidak terbaca", "susah discan", "gak bisa discan", "tidak bisa scan",
        "gabisa discan", "kusam", "rusak"
    ]

    service_cues = [
        "menolak", "tolak qris", "minimal belanja", "hanya tunai", "judes", "ketus",
        "marah", "disembunyikan", "malas", "tidak mau terima", "kasir judes", "pelayanan buruk",
        "pelayanan jelek", "tidak bagus", "kurang bagus", "gak bagus", "tidak ramah",
        "kurang ramah", "gak ramah", "tidak aman", "gak aman", "tidak lancar", "gak lancar",
        "tidak puas", "kurang puas", "gak puas", "kecewa", "buruk"
    ]

    positive_phrases = [
        "nama merchant sama", "nama toko sama", "sama dengan nama toko", "sama persis",
        "nominal sesuai", "jumlah sesuai", "pembayaran berhasil", "tanpa kendala",
        "transaksi lancar", "lancar jaya", "kasir ramah", "resmi", "tidak ada biaya",
        "tidak ada kendala", "tidak ada masalah", "langsung masuk sesuai nama",
        "sangat puas", "sangat lancar", "sangat aman", "pelayanan cepat",
        "sesuai nama toko", "sesuai dengan toko", "sesuai nama"
    ]

    positive_words = [
        "bagus", "aman", "terpercaya", "mantap", "puas", "lancar", "sukses",
        "berhasil", "cepat", "ramah", "recomended", "recommended", "terjamin",
        "memuaskan", "top", "jos", "joss", "josss", "sip", "oke", "good",
        "nice", "terima kasih", "makasih"
    ]

    has_fraud = any(cue in combined for cue in fraud_cues)
    has_mismatch = any(cue in combined for cue in mismatch_cues)
    has_surcharge = any(cue in combined for cue in surcharge_cues)
    has_damaged = any(cue in combined for cue in damaged_cues)
    has_service = any(cue in combined for cue in service_cues)

    # Deteksi kepuasan pelanggan dengan pengecekan negasi
    has_positive = any(phrase in combined for phrase in positive_phrases)
    if not has_positive:
        for pw in positive_words:
            if pw in combined:
                neg_pattern = rf"(tidak|bukan|kurang|gak|nggak|ga|tdk)\s+(\w+\s+)?{pw}"
                if not re.search(neg_pattern, combined):
                    has_positive = True
                    break

    has_specific_negative = has_fraud or has_mismatch or has_surcharge or has_damaged or has_service

    # ── GUARD 1: IndoBERT memprediksi FEEDBACK_AMBIGU padahal ada sinyal domain spesifik ──
    if predicted_cat == "FEEDBACK_AMBIGU":
        if has_fraud:
            return "PENIPUAN_STIKER_QRIS_PALSU", max(confidence, 0.95)
        if has_mismatch:
            return "KETIDAKSESUAIAN_IDENTITAS_MERCHANT", max(confidence, 0.95)
        if has_surcharge:
            return "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE", max(confidence, 0.95)
        if has_damaged:
            return "KONDISI_FISIK_QRIS_RUSAK", max(confidence, 0.95)
        if has_service:
            return "KUALITAS_LAYANAN_MERCHANT", max(confidence, 0.95)
        if has_positive and not has_specific_negative:
            return "QRIS_NORMAL_MERCHANT_TERPERCAYA", max(confidence, 0.95)

        # Cek semantic baseline jika cues langsung belum mencakup
        sem_res = _semantic_baseline_classify(n_clean)
        if sem_res["category_key"] != "FEEDBACK_AMBIGU":
            return sem_res["category_key"], max(float(sem_res["confidence"]), 0.85)

    # ── GUARD 2: IndoBERT memprediksi NORMAL padahal terdapat komplain atau keluhan nyata ──
    if predicted_cat == "QRIS_NORMAL_MERCHANT_TERPERCAYA":
        if has_fraud:
            return "PENIPUAN_STIKER_QRIS_PALSU", max(confidence, 0.90)
        if has_mismatch:
            return "KETIDAKSESUAIAN_IDENTITAS_MERCHANT", max(confidence, 0.90)
        if has_surcharge:
            return "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE", max(confidence, 0.90)
        if has_damaged:
            return "KONDISI_FISIK_QRIS_RUSAK", max(confidence, 0.90)
        if has_service:
            return "KUALITAS_LAYANAN_MERCHANT", max(confidence, 0.90)

    # ── GUARD 3: Ulasan positif murni keliru dikategorikan sebagai masalah ──
    if has_positive and not has_specific_negative:
        if predicted_cat in ["KETIDAKSESUAIAN_IDENTITAS_MERCHANT", "PENIPUAN_STIKER_QRIS_PALSU", "FEEDBACK_AMBIGU"]:
            return "QRIS_NORMAL_MERCHANT_TERPERCAYA", max(confidence, 0.95)

    return predicted_cat, confidence


def classify_feedback(text: Optional[str]) -> Dict[str, Any]:
    """
    Fungsi utama analisis ulasan pembeli:
    Menerima kalimat ulasan lalu menentukan kategori masalah, tingkat keparahan,
    serta rekomendasi tindak lanjut bagi toko.
    Dilengkapi pelindung otomatis agar ulasan kepuasan pelanggan dinilai secara akurat.
    Tahap preprocessing slang normalization dijalankan sebelum klasifikasi.
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

    # ── PREPROCESSING: Normalisasi slang & singkatan sebelum klasifikasi ──
    normalized_text = _normalize_slang(text)
    logger.debug("[Preprocessing] '%s' → '%s'", text.strip(), normalized_text)

    # Cek ketersediaan model IndoBERT fine-tuned
    has_indobert = _load_custom_indobert()
    if has_indobert and _indobert_model and _indobert_tokenizer:
        try:
            import torch
            # Gunakan teks yang sudah dinormalisasi untuk inferensi IndoBERT
            inputs = _indobert_tokenizer(
                normalized_text, return_tensors="pt", truncation=True, max_length=128, padding=True
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

            # Terapkan Hybrid Guard & Ambiguity Resolver Cerdas
            cat_key, confidence = _apply_hybrid_guard(text, cat_key, confidence, normalized_text)

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
    # Gunakan teks yang sudah dinormalisasi
    res = _semantic_baseline_classify(normalized_text)
    guarded_key, guarded_conf = _apply_hybrid_guard(text, res["category_key"], res["confidence"], normalized_text)
    if guarded_key != res["category_key"]:
        meta = CATEGORIES[guarded_key]
        res["category_key"] = guarded_key
        res["category_title"] = meta["title"]
        res["confidence"] = guarded_conf
        res["severity"] = meta["severity"]
        res["action"] = meta["action"]
        res["emrs_penalty"] = meta["emrs_penalty"]
    return res

