"""
=============================================================================
LAQRIS TROCR FINE-TUNING — SCRIPT PELATIHAN UTAMA (train_model.py)
=============================================================================
TUGAS UTAMA SCRIPT INI:
1. Memuat model awal TrOCR (microsoft/trocr-base-printed) dan TrOCRProcessor.
2. Membaca dataset gambar + label teks yang sudah Anda siapkan di folder datasets/
   (Termasuk menangani nama merchant yang mengandung koma secara aman).
3. Membagi dataset menjadi Data Latih (Train 85%) dan Data Uji (Validation 15%).
4. Melatih ulang model (fine-tuning) menggunakan PyTorch + Hugging Face Seq2SeqTrainer.
5. Menyimpan bobot AI hasil pelatihan terbaru ke folder `models/trocr_<nama_model>/`.

CARA MENJALANKAN PELATIHAN (CONTOH KHUSUS MERCHANT NAME):
  python train_model.py --model_type merchant_name --epochs 10 --batch_size 4
=============================================================================
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    RobertaTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

# Impor Dataset Loader khusus penanganan koma dari utils/dataset_loader.py
from utils.dataset_loader import TrOCRDataset

# Konfigurasi encoding stdout untuk terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Jalur direktori utama
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_DIR = os.path.join(BASE_DIR, "base_model")

# Pemetaan konfigurasi 3 model spesialisasi
MODEL_CONFIGS = {
    "merchant_name": {
        "dataset_dir": os.path.join(BASE_DIR, "ocr_datasets", "dataset_merchant_name"),
        "output_dir": os.path.join(BASE_DIR, "models", "trocr_merchant_name"),
        "name": "Model 1: Merchant Name Specialist"
    },
    "nmid": {
        "dataset_dir": os.path.join(BASE_DIR, "ocr_datasets", "dataset_nmid"),
        "output_dir": os.path.join(BASE_DIR, "models", "trocr_nmid"),
        "name": "Model 2: NMID Specialist"
    },
    "general": {
        "dataset_dir": os.path.join(BASE_DIR, "ocr_datasets", "dataset_general"),
        "output_dir": os.path.join(BASE_DIR, "models", "trocr_general"),
        "name": "Model 3: General QRIS Specialist"
    }
}


def load_processor_and_model():
    """
    [TUGAS INSTRUKSI]: Memuat Model & Processor TrOCR.
    Jika folder 'base_model' sudah diunduh secara lokal, script akan membaca dari lokal.
    Jika belum ada, script akan mengunduh dari Hugging Face secara otomatis.
    """
    if os.path.exists(BASE_MODEL_DIR) and os.path.exists(os.path.join(BASE_MODEL_DIR, "config.json")):
        print(f"[STEP 1] Memuat Base Model dari folder lokal: {BASE_MODEL_DIR}")
        processor = TrOCRProcessor.from_pretrained(BASE_MODEL_DIR)
        model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_DIR)
    else:
        nama_hf = "microsoft/trocr-base-printed"
        print(f"[STEP 1] Folder base_model belum ada. Mengunduh dari Hugging Face: {nama_hf}")
        tokenizer = RobertaTokenizer.from_pretrained(nama_hf)
        image_processor = ViTImageProcessor.from_pretrained(nama_hf)
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = VisionEncoderDecoderModel.from_pretrained(nama_hf)

    # [SETTING TROCR]: Konfigurasi token khusus decoder TrOCR Transformer
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Bersihkan parameter generasi dari model.config agar tidak melempar ValueError saat menyimpan checkpoint
    gen_keys = ["max_length", "early_stopping", "no_repeat_ngram_size", "length_penalty", "num_beams", "eos_token_id"]
    for key in gen_keys:
        if hasattr(model.config, key):
            delattr(model.config, key)

    # Set parameter generasi eksklusif pada model.generation_config
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
        model.generation_config.max_length = 32
        model.generation_config.early_stopping = True
        model.generation_config.no_repeat_ngram_size = 3
        model.generation_config.length_penalty = 2.0
        model.generation_config.num_beams = 4

    return processor, model


def train_single_model(target_key: str, epochs: int, batch_size: int, lr: float):
    """
    [TUGAS INSTRUKSI]: Fungsi utama yang mengeksekusi pelatihan untuk 1 model spesialis.
    - target_key : "merchant_name", "nmid", atau "general"
    - epochs     : Berapa kali AI membaca seluruh dataset (default: 5 atau 10)
    - batch_size : Berapa banyak gambar yang diproses bersamaan dalam 1 langkah
    - lr         : Learning rate / kecepatan belajar AI (default: 5e-5)
    """
    config = MODEL_CONFIGS[target_key]
    print("\n" + "=" * 75)
    print(f"[MEMULAI PELATIHAN] {config['name']}")
    print(f"[-] Folder Dataset : {config['dataset_dir']}")
    print(f"[-] Folder Output  : {config['output_dir']}")
    print("=" * 75)

    # 1. Memuat Processor dan Model Base
    processor, model = load_processor_and_model()

    # 2. Memuat Dataset menggunakan Custom Loader (Aman terhadap koma)
    print(f"[STEP 2] Membaca data gambar dan labels.csv dari folder...")
    full_dataset = TrOCRDataset(dataset_dir=config['dataset_dir'], processor=processor)
    total_len = len(full_dataset)

    if total_len == 0:
        print(f"[ERROR] Tidak ada data valid yang ditemukan di {config['dataset_dir']}.")
        print("Pastikan file labels.csv dan folder images/ berisi data gambar yang sesuai.")
        return

    # 3. Membagi Dataset menjadi Data Latih (Train 85%) dan Data Uji (Validation 15%)
    val_len = max(1, int(total_len * 0.15))
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    print(f"[STEP 3] Pembagian Data -> Train (Data Latih): {train_len} sampel | Val (Data Uji): {val_len} sampel")

    # 4. Menyiapkan Folder Simpan Model
    os.makedirs(config['output_dir'], exist_ok=True)

    # 5. Menentukan Parameter Pelatihan Hugging Face Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=torch.cuda.is_available(),           # Menggunakan GPU CUDA jika tersedia
        learning_rate=lr,                         # Kecepatan optimasi AI
        num_train_epochs=epochs,                  # Jumlah perulangan epoch
        logging_steps=5,                          # Cetak log setiap 5 langkah
        save_steps=20,                            # Simpan checkpoint setiap 20 langkah
        eval_steps=20,                            # Evaluasi setiap 20 langkah
        eval_strategy="steps",
        save_total_limit=2,                       # Simpan maksimal 2 checkpoint terbaik
        predict_with_generate=True,
        report_to="none"                          # Nonaktifkan wandb/tensorboard agar bersih
    )

    # 6. Inisialisasi Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor.image_processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    # 7. Eksekusi Pelatihan AI
    print(f"\n[STEP 4] Memulai iterasi pelatihan PyTorch ({epochs} epoch)...")
    trainer.train()

    # 8. Simpan Bobot Model Hasil Pelatihan
    print(f"\n[STEP 5] Pelatihan selesai! Menyimpan bobot model ke: {config['output_dir']}")
    model.save_pretrained(config['output_dir'])
    processor.save_pretrained(config['output_dir'])

    print(f"[SUCCESS] Fine-tuning untuk {config['name']} SELESAI DENGAN SUKSES!\n")


def main():
    """
    [TUGAS INSTRUKSI]: Entry point parser argumen baris perintah (CLI).
    Memungkinkan Anda menjalankan pelatihan dari terminal.
    """
    parser = argparse.ArgumentParser(description="LaQris TrOCR Fine-Tuning Manager")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["merchant_name", "nmid", "general", "all"],
        default="merchant_name",
        help="Pilih model yang ingin dilatih: merchant_name (default) | nmid | general | all"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Jumlah perulangan epoch pelatihan (default: 5)")
    parser.add_argument("--batch_size", type=int, default=4, help="Ukuran batch per langkah (default: 4)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 0.00005)")

    args = parser.parse_args()

    # Jika memilih 'all', latih ketiga model satu per satu
    if args.model_type == "all":
        for k in MODEL_CONFIGS.keys():
            train_single_model(k, args.epochs, args.batch_size, args.lr)
    else:
        train_single_model(args.model_type, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
