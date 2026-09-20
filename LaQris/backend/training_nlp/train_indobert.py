"""
train_indobert.py - Skrip Pelatihan (Fine-Tuning) IndoBERT untuk Klasifikasi Feedback LaQris

Format Dataset CSV yang Diharapkan:
----------------------------------
File: backend/data/dataset.csv
Kolom: text, label

Daftar label yang valid:
- identity_mismatch
- qr_replacement
- additional_fees
- suspicious_transaction
- safe_confirmation
- other

Contoh Baris:
"Nama di rekening tujuan beda jauh sama nama toko",identity_mismatch
"Stiker QRIS ditempel di atas kode lama",qr_replacement
"Kasir meminta tambahan biaya admin 2000",additional_fees
"QRIS asli dan pembayaran lancar",safe_confirmation

Cara Menjalankan:
-----------------
python train_indobert.py --data data/dataset.csv --epochs 3 --batch-size 16
"""

import os
import argparse
import logging
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_indobert")

LABEL2ID = {
    "identity_mismatch": 0,
    "qr_replacement": 1,
    "additional_fees": 2,
    "suspicious_transaction": 3,
    "safe_confirmation": 4,
    "other": 5
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def train(data_path: str, output_dir: str, epochs: int = 3, batch_size: int = 16, lr: float = 2e-5):
    if not os.path.exists(data_path):
        logger.error("File dataset tidak ditemukan di: %s", data_path)
        logger.info("Silakan siapkan file CSV dengan kolom 'text' dan 'label'.")
        return

    logger.info("Membaca dataset dari: %s", data_path)
    df = pd.read_csv(data_path, usecols=["text", "label"], on_bad_lines="skip")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset CSV wajib memiliki kolom 'text' dan 'label'.")

    # Filter data yang valid
    df = df.dropna(subset=["text", "label"])
    df["label_id"] = df["label"].map(LABEL2ID)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    logger.info("Total data valid: %d baris", len(df))
    
    if len(df) < 30:
        train_df, val_df = df, df
    else:
        train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label_id"])


    model_name = "indobenchmark/indobert-base-p1"
    logger.info("Mengunduh tokenizer dan pre-trained model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Menggunakan perangkat: %s", device)
    model.to(device)

    train_dataset = FeedbackDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer)
    val_dataset = FeedbackDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    logger.info("Memulai pelatihan (%d epochs)...", epochs)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for step, batch in enumerate(train_bar):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            avg_so_far = total_loss / (step + 1)
            train_bar.set_postfix(loss=f"{avg_so_far:.4f}")

        avg_loss = total_loss / len(train_loader)

        # Evaluasi Akurasi
        model.eval()
        correct, total = 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", unit="batch")
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_bar.set_postfix(acc=f"{(correct/total*100):.1f}%" if total > 0 else "0%")

        val_acc = (correct / total) * 100.0 if total > 0 else 0.0
        logger.info("Epoch %d/%d - Train Loss: %.4f - Val Accuracy: %.2f%%", epoch + 1, epochs, avg_loss, val_acc)

    # Simpan model
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Menyimpan model hasil training ke: %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Pelatihan selesai dengan sukses!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERT untuk Feedback LaQris")
    parser.add_argument("--data", type=str, default="dataset.csv", help="Path ke file dataset CSV")
    parser.add_argument("--output", type=str, default="weights/indobert_feedback_model", help="Folder penyimpanan model")
    parser.add_argument("--epochs", type=int, default=3, help="Jumlah epoch pelatihan")
    parser.add_argument("--batch-size", type=int, default=16, help="Ukuran batch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()
    train(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
