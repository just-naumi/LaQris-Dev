"""
Evaluator 10.000 Hard Stress Test untuk Model IndoBERT LaQris
Menggunakan Batch Processing PyTorch langsung pada GPU untuk evaluasi kilat (~15-25 detik).

Metrik yang Dihitung:
- Akurasi Menyeluruh pada 10.000 Kasus Ekstrem
- Per-class Precision, Recall, F1-Score
- 7x7 Confusion Matrix Lengkap
- Distribusi Confidence & Deteksi Overconfidence
- False Positive Rate pada Transaksi Normal
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)

    model_dir = os.path.join(root_dir, "models", "indobert_qris_feedback", "best_model")
    csv_path = os.path.join(root_dir, "dataset", "qris_stress_test_10k.csv")

    if not os.path.exists(model_dir):
        print(f"[ERROR] Checkpoint model tidak ditemukan di: {model_dir}", flush=True)
        return

    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset stress test tidak ditemukan di: {csv_path}", flush=True)
        print("Jalankan 'dataset/build_10k_hard_stress.py' terlebih dahulu.", flush=True)
        return

    print("=" * 80, flush=True)
    print("LAQRIS INDOBERT - 10.000 SAMPLES HARD STRESS TEST EVALUATION", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Hardware Device : {device.type.upper()}", flush=True)
    if device.type == "cuda":
        print(f"[INFO] GPU Model       : {torch.cuda.get_device_name(0)}", flush=True)

    # 1. Muat Dataset
    print(f"[INFO] Memuat dataset dari: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    print(f"[INFO] Total data uji  : {len(df)} baris", flush=True)

    # 2. Muat Tokenizer & Model
    print(f"[INFO] Memuat model IndoBERT dari: {model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    id2label = model.config.id2label
    label2id = model.config.label2id

    df["label_id"] = df["expected_category"].map(label2id)

    # 3. Batch Processing Kilat
    BATCH_SIZE = 64
    texts = df["text"].tolist()
    total_samples = len(texts)
    total_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"[INFO] Mengeksekusi inferensi batch ({total_batches} batch, batch_size={BATCH_SIZE})...", flush=True)
    start_time = time.time()

    all_preds = []
    all_confs = []

    with torch.no_grad():
        for i in range(0, total_samples, BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            confs = np.max(probs, axis=-1)

            all_preds.extend(preds)
            all_confs.extend(confs)

            processed = min(i + BATCH_SIZE, total_samples)
            if (i // BATCH_SIZE) % 25 == 0 or processed == total_samples:
                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0
                print(f"  -> Selesai {processed}/{total_samples} sampel ({speed:.1f} sampel/detik)...", flush=True)

    elapsed_total = time.time() - start_time
    print(f"\n[SUKSES] 10.000 sampel selesai dievaluasi dalam: {elapsed_total:.2f} detik! ({total_samples/elapsed_total:.1f} sampel/detik)", flush=True)

    y_true = df["label_id"].tolist()
    y_pred = all_preds
    all_confs = np.array(all_confs)

    # 4. Hitung Metrik Evaluasi
    target_names = [id2label[i] for i in range(len(id2label))]
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=target_names, digits=4)

    cm = confusion_matrix(y_true, y_pred)

    overall_acc = report_dict["accuracy"] * 100
    macro_f1 = report_dict["macro avg"]["f1-score"] * 100

    # Analisis False Positives pada Transaksi Normal
    normal_id = label2id.get("QRIS_NORMAL_MERCHANT_TERPERCAYA", label2id.get("TRANSAKSI_SUKSES_NORMAL", 5))
    normal_total = sum(1 for y in y_true if y == normal_id)
    normal_correct = sum(1 for y, p in zip(y_true, y_pred) if y == normal_id and p == normal_id)
    normal_false_positives = normal_total - normal_correct

    print("\n" + "=" * 80, flush=True)
    print("HASIL EVALUASI 10.000 HARD STRESS TEST LAQRIS:", flush=True)
    print("=" * 80, flush=True)
    print(report_str, flush=True)

    print("-" * 80, flush=True)
    print("DISTRIBUSI CONFIDENCE (TINGKAT KEYAKINAN MODEL):", flush=True)
    print(f"- Rata-rata Confidence   : {np.mean(all_confs)*100:.2f}%", flush=True)
    print(f"- Median Confidence      : {np.median(all_confs)*100:.2f}%", flush=True)
    print(f"- Minimum Confidence     : {np.min(all_confs)*100:.2f}%", flush=True)
    print(f"- Sampel High Conf (>90%): {np.sum(all_confs >= 0.90)} / {len(df)} ({(np.sum(all_confs >= 0.90)/len(df))*100:.1f}%)", flush=True)
    print(f"- Sampel Low Conf (<70%) : {np.sum(all_confs < 0.70)} / {len(df)} ({(np.sum(all_confs < 0.70)/len(df))*100:.1f}%)", flush=True)

    print("\nANALISIS KEAMANAN TRANSAKSI NORMAL (HARD NEGATIVES):", flush=True)
    print(f"- Total Transaksi Normal Diuji  : {normal_total}", flush=True)
    print(f"- Lolos Akurat Tanpa Salah Flag : {normal_correct} ({(normal_correct/normal_total)*100:.2f}%)", flush=True)
    print(f"- False Positive (Salah Tuduh)  : {normal_false_positives} ({(normal_false_positives/normal_total)*100:.2f}%)", flush=True)
    print("-" * 80, flush=True)

    print("\nCONFUSION MATRIX (7x7):", flush=True)
    print(f"{'Kategori':<35} | " + " | ".join([f"K{i}" for i in range(len(target_names))]), flush=True)
    print("-" * 80, flush=True)
    for i, row in enumerate(cm):
        row_str = " | ".join([f"{val:4d}" for val in row])
        print(f"K{i}: {target_names[i][:31]:<31} | {row_str}", flush=True)

    # Simpan Laporan JSON
    report_output_path = os.path.join(base_dir, "stress_test_10k_report.json")
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(df),
            "execution_time_seconds": round(elapsed_total, 2),
            "overall_accuracy": round(overall_acc, 2),
            "macro_f1": round(macro_f1, 2),
            "mean_confidence": round(float(np.mean(all_confs) * 100), 2),
            "normal_false_positives": normal_false_positives,
            "normal_accuracy": round((normal_correct / normal_total) * 100, 2),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist()
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Laporan lengkap 10k disimpan di: {report_output_path}", flush=True)

if __name__ == "__main__":
    main()
