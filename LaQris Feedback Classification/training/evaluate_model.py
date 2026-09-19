"""
Script Evaluasi Model IndoBERT pada Test Set
Menghitung Confusion Matrix, Precision, Recall, F1-Score per Kelas
"""

import os
import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)

    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_dir = os.path.join(root_dir, "models", "indobert_qris_feedback", "best_model")
    if not os.path.exists(model_dir):
        print(f"[WARN] Checkpoint model belum tersedia di: {model_dir}")
        print("Jalankan 'train_indobert.py' terlebih dahulu untuk menghasilkan bobot model.")
        return

    test_csv_path = os.path.join(root_dir, "dataset", "test_split.csv")
    if not os.path.exists(test_csv_path):
        print(f"[WARN] Test dataset belum ditemukan di: {test_csv_path}")
        return

    print(f"[INFO] Memuat model dari: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    test_df = pd.read_csv(test_csv_path)
    print(f"[INFO] Mengevaluasi {len(test_df)} data uji...")

    label2id = cfg["model"]["label2id"]
    id2label = {int(k): v for k, v in cfg["model"]["id2label"].items()}

    y_true = test_df["label"].tolist()
    y_pred = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for text in test_df["text"]:
        inputs = tokenizer(text, truncation=True, max_length=cfg["model"]["max_seq_length"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            y_pred.append(pred)

    target_names = [id2label[i] for i in range(cfg["model"]["num_labels"])]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\n" + "="*70)
    print("HASIL CLASSIFICATION REPORT PADA TEST SET:")
    print("="*70)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
