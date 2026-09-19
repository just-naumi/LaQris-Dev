"""
Script Fine-Tuning IndoBERT untuk Klasifikasi Feedback QRIS Statis
Model: indobenchmark/indobert-base-p1
Dataset: qris_feedback_10k.csv (10.000 sampel, 6 kelas)
"""

import os
import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

def main():
    # 1. Muat Konfigurasi
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("[INFO] Konfigurasi pelatihan berhasil dimuat.")
    root_dir = os.path.dirname(base_dir)
    csv_rel = cfg["data"]["dataset_path"]
    csv_path = os.path.normpath(os.path.join(root_dir, csv_rel.replace("./", "")))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {csv_path}")

    # 2. Muat Data
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset dimuat: {len(df)} baris.")
    print("Distribusi Kelas:")
    print(df["category"].value_counts())

    label2id = cfg["model"]["label2id"]
    id2label = {int(k): v for k, v in cfg["model"]["id2label"].items()}
    df["label"] = df["category"].map(label2id)

    # 3. Train-Val-Test Split (80 - 10 - 10)
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=cfg["data"]["random_seed"], stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=cfg["data"]["random_seed"], stratify=temp_df["label"]
    )

    print(f"[INFO] Ukuran Data Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Simpan test set untuk evaluasi nanti
    test_csv_path = os.path.join(root_dir, "dataset", "test_split.csv")
    test_df.to_csv(test_csv_path, index=False)

    # 4. Tokenisasi
    model_name = cfg["model"]["pretrained_model_name"]
    max_len = cfg["model"]["max_seq_length"]
    print(f"[INFO] Memuat Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])

    train_tokenized = train_ds.map(tokenize_func, batched=True)
    val_tokenized = val_ds.map(tokenize_func, batched=True)

    # 5. Inisialisasi Model IndoBERT
    print(f"[INFO] Memuat Pretrained Model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=cfg["model"]["num_labels"],
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    output_dir = os.path.normpath(os.path.join(root_dir, cfg["training"]["output_dir"].replace("./", "")))
    os.makedirs(output_dir, exist_ok=True)

    # 6. Training Arguments
    use_fp16 = torch.cuda.is_available() and cfg["training"].get("fp16", False)
    
    # Hitung warmup_steps (kompatibel dgn Transformers 5.x)
    total_steps = (len(train_tokenized) // cfg["training"]["per_device_train_batch_size"]) * cfg["training"]["num_train_epochs"]
    warmup_steps = int(total_steps * float(cfg["training"].get("warmup_ratio", 0.1)))

    train_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        warmup_steps=warmup_steps,
        weight_decay=cfg["training"]["weight_decay"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy=cfg["training"]["eval_strategy"],
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model=cfg["training"]["metric_for_best_model"],
        greater_is_better=True,
        fp16=use_fp16,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    # 7. Mulai Training
    print("[INFO] Memulai proses fine-tuning IndoBERT...")
    trainer.train()

    # 8. Simpan Model Terbaik
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"\n[SUKSES] Model IndoBERT fine-tuned berhasil disimpan di: {best_model_dir}")

if __name__ == "__main__":
    main()
