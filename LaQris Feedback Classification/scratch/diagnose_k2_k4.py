import os
import sys
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)

model_dir = os.path.join(root_dir, "models", "indobert_qris_feedback", "best_model")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
id2label = model.config.id2label

csv_path = os.path.join(root_dir, "dataset", "qris_stress_test_10k.csv")
df = pd.read_csv(csv_path)

k2_df = df[df["expected_category"] == "SALAH_INPUT_NOMINAL_DOUBLE_BAYAR"].copy()
texts = k2_df["text"].tolist()

preds = []
confs = []

with torch.no_grad():
    for i in range(0, len(texts), 64):
        b = texts[i:i+64]
        inp = tokenizer(b, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inp = {k: v.cuda() for k, v in inp.items()}
        out = model(**inp)
        p = torch.softmax(out.logits, dim=-1).cpu().numpy()
        preds.extend(np.argmax(p, axis=-1))
        confs.extend(np.max(p, axis=-1))

k2_df["pred"] = [id2label[p] for p in preds]
k2_df["conf"] = confs

err = k2_df[k2_df["pred"] == "TRANSAKSI_PENDING_GAGAL_SISTEM"]
print("=" * 80)
print(f"DIAGNOSIS 443 KASUS SALAH KLASIFIKASI: K2 -> K4")
print("=" * 80)
print(f"Total Kasus K2 -> K4 : {len(err)}")
print(f"Rata-rata Confidence : {err['conf'].mean()*100:.2f}%")
print(f"Median Confidence    : {err['conf'].median()*100:.2f}%")
print(f"High Conf (>90%)     : {(err['conf'] >= 0.90).sum()} / {len(err)} ({((err['conf'] >= 0.90).sum()/len(err))*100:.1f}%)")
print("-" * 80)

# Group by the main template prefix
err["template_key"] = err["text"].apply(lambda x: x[:60])
print("\nPENYEBARAN TEMPLATE YANG MENYEBABKAN ERROR:")
for key, grp in err.groupby("template_key"):
    sample_text = grp["text"].iloc[0]
    avg_c = grp["conf"].mean() * 100
    print(f"\n[Jumlah: {len(grp)} kasus | Avg Conf: {avg_c:.1f}%]")
    print(f"Teks Contoh: \"{sample_text}\"")

