"""
Skrip Konversi Model TrOCR (PyTorch) ke Format ONNX Runtime
============================================================
Deskripsi:
  Mengonversi model TrOCR yang sudah di-finetune ke format ONNX menggunakan
  Hugging Face Optimum. Mendukung semua model spesialis LaQris:
    - merchant_name  : Model OCR khusus Nama Merchant
    - general        : Model OCR untuk label umum (NMID, acquirer, dll)

Cara Penggunaan:
  python export_onnx.py --model_type merchant_name
  python export_onnx.py --model_type general
  python export_onnx.py --model_type all
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Pastikan import pustaka yang dibutuhkan
try:
    from optimum.onnxruntime import ORTModelForVision2Seq
    from transformers import TrOCRProcessor
except ImportError:
    print("[ERROR] Pustaka 'optimum' atau 'transformers' belum terinstall.")
    print("Silakan jalankan: pip install optimum[onnxruntime-gpu] onnxruntime-gpu")
    sys.exit(1)

# Pemetaan nama model ke folder input/output
BASE_DIR = Path(__file__).parent

MODEL_EXPORT_CONFIGS = {
    "merchant_name": {
        "input_dir": BASE_DIR / "models" / "trocr_merchant_name",
        "output_dir": BASE_DIR / "models" / "trocr_merchant_name_onnx",
        "label": "Merchant Name Specialist",
    },
    "general": {
        "input_dir": BASE_DIR / "models" / "trocr_general",
        "output_dir": BASE_DIR / "models" / "trocr_general_onnx",
        "label": "General QRIS Specialist (NMID, Acquirer, dll)",
    },
    "base_printed": {
        "input_dir": "microsoft/trocr-base-printed",
        "output_dir": BASE_DIR / "models" / "trocr_base_printed_onnx",
        "label": "Original Base Model (microsoft/trocr-base-printed)",
    },
}


def convert_trocr_to_onnx(model_source: str, output_dir: str):
    """
    Mengonversi model VisionEncoderDecoder TrOCR ke ONNX.
    Menghasilkan 3 file: encoder_model.onnx, decoder_model.onnx,
    dan decoder_with_past_model.onnx.
    """
    is_local_path = isinstance(model_source, Path) or (isinstance(model_source, str) and (os.path.exists(model_source) or "\\" in model_source or "/" in model_source and not model_source.startswith("microsoft/")))
    
    if is_local_path:
        model_path = Path(model_source)
        if not model_path.exists():
            print(f"[ERROR] Folder model asal tidak ditemukan: {model_path.resolve()}")
            return False
        model_identifier = str(model_path.resolve())
    else:
        model_identifier = str(model_source)

    save_path = Path(output_dir)

    print("=" * 70)
    print("[START] Mengonversi Model TrOCR ke Format ONNX Runtime")
    print(f"  Model Asal    : {model_identifier}")
    print(f"  Folder Output : {save_path.resolve()}")
    print("=" * 70)


    start_time = time.time()

    # Step 1: Export Model ke ONNX
    # ORTModelForVision2Seq otomatis mengonversi encoder dan decoder terpisah
    print("\n[STEP 1/2] Mengekspor Bobot Model Encoder & Decoder ke ONNX...")
    print("  (Proses ini membutuhkan waktu 1-2 menit tergantung spesifikasi PC)...")

    model = ORTModelForVision2Seq.from_pretrained(
        model_identifier,
        export=True
    )

    # Step 2: Muat Processor & Simpan Kedua Komponen (encoder, decoder, processor)
    print("\n[STEP 2/2] Memuat Processor dan Menyimpan Artifact ONNX...")
    processor = TrOCRProcessor.from_pretrained(model_identifier)

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    processor.save_pretrained(str(save_path))

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[BERHASIL] Konversi ONNX Selesai dalam {elapsed:.2f} detik!")
    print(f"[OUTPUT] Model ONNX tersimpan di: {save_path.resolve()}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LaQris TrOCR ONNX Exporter")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(MODEL_EXPORT_CONFIGS.keys()) + ["all"],
        default="merchant_name",
        help="Model yang ingin dikonversi ke ONNX: merchant_name | general | all"
    )
    args = parser.parse_args()

    targets = list(MODEL_EXPORT_CONFIGS.keys()) if args.model_type == "all" else [args.model_type]

    for model_key in targets:
        cfg = MODEL_EXPORT_CONFIGS[model_key]
        print(f"\n{'='*70}")
        print(f"[TARGET] {cfg['label']}")
        print(f"{'='*70}")
        success = convert_trocr_to_onnx(str(cfg["input_dir"]), str(cfg["output_dir"]))
        if not success:
            print(f"[SKIP] Model '{model_key}' dilewati karena folder tidak ditemukan.")
