"""
=============================================================================
LAQRIS TROCR FINE-TUNING — CHECKPOINT COMPARATIVE EVALUATOR (test_eval.py)
=============================================================================
Script untuk menguji dan membandingkan akurasi prediksi OCR antara:
1. Base Model (microsoft/trocr-base-printed)
2. Checkpoint-60 (models/trocr_merchant_name/checkpoint-60)
3. Checkpoint-80 (models/trocr_merchant_name/checkpoint-80)
4. Model Final  (models/trocr_merchant_name)

Mencetak tabel komparasi teks Asli vs Prediksi masing-masing model & skor akurasi.
=============================================================================
"""

import os
import sys
import warnings
import torch
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, RobertaTokenizer, logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Konfigurasi encoding terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "ocr_datasets", "dataset_merchant_name")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH = os.path.join(DATASET_DIR, "labels.csv")

MODEL_BASE_DIR = os.path.join(BASE_DIR, "base_model")
MODEL_CKPT_60 = os.path.join(BASE_DIR, "models", "trocr_merchant_name", "checkpoint-60")
MODEL_CKPT_80 = os.path.join(BASE_DIR, "models", "trocr_merchant_name", "checkpoint-80")
MODEL_FINAL = os.path.join(BASE_DIR, "models", "trocr_merchant_name")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_trocr_model(model_path):
    """Fungsi helper memuat model TrOCR dari path tertentu."""
    if not os.path.exists(model_path):
        return None, None
    try:
        nama_base = MODEL_BASE_DIR if os.path.exists(MODEL_BASE_DIR) else "microsoft/trocr-base-printed"
        tokenizer = RobertaTokenizer.from_pretrained(nama_base)
        image_processor = ViTImageProcessor.from_pretrained(nama_base)
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(DEVICE)
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
            model.generation_config.max_length = None
        model.eval()
        return processor, model
    except Exception as e:
        print(f"[WARNING] Gagal memuat model dari {model_path}: {e}")
        return None, None

def run_inference(image_path, processor, model):
    """Fungsi inferensi 1 gambar menggunakan TrOCR."""
    if model is None or processor is None:
        return "-"
    try:
        img = Image.open(image_path).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=32)
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return pred_text
    except Exception:
        return "ERROR"

def main():
    print("=" * 90)
    print("🔍 [EVALUASI KOMPARASI CHECKPOINT 60 VS 80 LAQRIS TROCR]")
    print(f"Perangkat Ekspeksi: {DEVICE.upper()}")
    print("=" * 90)

    # 1. Muat Dataset labels.csv dengan penanganan koma yang aman
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] File labels.csv tidak ditemukan di: {CSV_PATH}")
        return

    records = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_idx = 1 if lines and "file_name" in lines[0].lower() else 0
    for line in lines[start_idx:]:
        line_str = line.strip()
        if not line_str:
            continue
        parts = line_str.split(',', 1)
        if len(parts) >= 2:
            img_name = parts[0].strip().strip('"').strip("'")
            text_gt = parts[1].strip().strip('"').strip("'")
            img_path = os.path.join(IMAGES_DIR, img_name)
            if os.path.exists(img_path):
                records.append({"file_name": img_name, "gt_text": text_gt, "path": img_path})

    print(f"[INFO] Memuat {len(records)} sampel untuk pengujian komparasi.\n")

    # 2. Muat Checkpoint Model 60, 80, dan Base
    print("[1/3] Memuat Checkpoint-60...")
    proc_60, model_60 = load_trocr_model(MODEL_CKPT_60)

    print("[2/3] Memuat Checkpoint-80...")
    proc_80, model_80 = load_trocr_model(MODEL_CKPT_80)

    print("[3/3] Memuat Model Final...")
    proc_final, model_final = load_trocr_model(MODEL_FINAL)

    match_60 = 0
    match_80 = 0
    match_final = 0
    total = len(records)

    report_path = os.path.join(BASE_DIR, "eval_report.txt")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 110 + "\n")
        rf.write(f"{'FILE NAME':<20} | {'GROUND TRUTH (ASLI)':<32} | {'CHECKPOINT-60':<32} | {'CHECKPOINT-80':<32}\n")
        rf.write("=" * 110 + "\n")

        for idx_row, item in enumerate(records, start=1):
            gt = item["gt_text"]
            pred_60 = run_inference(item["path"], proc_60, model_60) if model_60 else "-"
            pred_80 = run_inference(item["path"], proc_80, model_80) if model_80 else "-"
            pred_fn = run_inference(item["path"], proc_final, model_final) if model_final else "-"

            is_m60 = (pred_60.upper() == gt.upper())
            is_m80 = (pred_80.upper() == gt.upper())
            is_mfn = (pred_fn.upper() == gt.upper())

            if is_m60: match_60 += 1
            if is_m80: match_80 += 1
            if is_mfn: match_final += 1

            tag_60 = " [EXACT]" if is_m60 else ""
            tag_80 = " [EXACT]" if is_m80 else ""

            row_str = f"{item['file_name']:<20} | {gt[:30]:<32} | {(pred_60 + tag_60)[:30]:<32} | {(pred_80 + tag_80)[:30]:<32}"
            print(f"[{idx_row}/{total}] {row_str}", flush=True)
            rf.write(row_str + "\n")
            rf.flush()

        summary_str = "\n" + "=" * 110 + "\n📊 [RINGKASAN SKOR AKURASI KECOCOKAN PERSIS (EXACT MATCH)]\n"
        if model_60:
            acc_60 = (match_60 / total) * 100.0 if total > 0 else 0
            summary_str += f"  - Checkpoint-60 : {match_60}/{total} Tepat ({acc_60:.1f}%)\n"
        if model_80:
            acc_80 = (match_80 / total) * 100.0 if total > 0 else 0
            summary_str += f"  - Checkpoint-80 : {match_80}/{total} Tepat ({acc_80:.1f}%)\n"
        if model_final:
            acc_fn = (match_final / total) * 100.0 if total > 0 else 0
            summary_str += f"  - Model Final   : {match_final}/{total} Tepat ({acc_fn:.1f}%)\n"
        summary_str += "=" * 110 + "\n"

        print(summary_str, flush=True)
        rf.write(summary_str)

if __name__ == "__main__":
    main()
