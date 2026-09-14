"""
=============================================================================
LAQRIS TROCR FINE-TUNING — BASE MODEL DOWNLOADER
=============================================================================
Script ini mengunduh pre-trained model TrOCR (microsoft/trocr-base-printed)
dari Hugging Face Hub dan menyimpannya secara lokal di folder ./base_model/

Gunakan script ini agar pelatihan offline dan tidak perlu mengunduh ulang.
=============================================================================
"""

import os
import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, RobertaTokenizer

BASE_MODEL_NAME = "microsoft/trocr-base-printed"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_model")

def download_and_save_base_model():
    print(f"[INFO] Mengunduh Base Model TrOCR: '{BASE_MODEL_NAME}'...")
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL_NAME)
        image_processor = ViTImageProcessor.from_pretrained(BASE_MODEL_NAME)
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_NAME)

        # Simpan lokal
        processor.save_pretrained(SAVE_DIR)
        model.save_pretrained(SAVE_DIR)

        print(f"[SUCCESS] Model TrOCR berhasil disimpan secara lokal di: {SAVE_DIR}")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal mengunduh model: {e}")
        return False

if __name__ == "__main__":
    download_and_save_base_model()
