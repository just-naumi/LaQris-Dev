"""
Skrip Pengujian & Benchmark Latency Inferensi TrOCR ONNX Runtime
================================================================
Deskripsi:
  Menguji model ONNX (models/trocr_merchant_name_onnx) menggunakan ONNX Runtime GPU (CUDAExecutionProvider).
  Mengukur latency per gambar (dalam milidetik) dan akurasi (Exact Match & CER).

Optimasi Utama:
  - Execution Provider: CUDAExecutionProvider (GPU Nvidia)
  - Decoding Strategy: Greedy Search (num_beams=1)
  - Token Capping: max_new_tokens=25 (Memangkas looping decoding yang sia-sia)
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from PIL import Image

try:
    from optimum.onnxruntime import ORTModelForVision2Seq
    from transformers import TrOCRProcessor
    import torch
except ImportError:
    print("[ERROR] Pustaka belum lengkap. Pastikan optimum & onnxruntime-gpu terpasang.")
    sys.exit(1)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Menghitung Character Error Rate (CER) antara referensi dan hipotesis."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    r_len = len(ref_chars)
    h_len = len(hyp_chars)
    
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
        
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                
    if r_len == 0:
        return 0.0 if h_len == 0 else 1.0
    return dp[r_len][h_len] / float(r_len)


def run_onnx_benchmark(onnx_model_dir: str, dataset_dir: str):
    onnx_path = Path(onnx_model_dir)
    dataset_path = Path(dataset_dir)
    csv_path = dataset_path / "labels.csv"
    images_dir = dataset_path / "images"

    if not onnx_path.exists():
        print(f"[ERROR] Folder ONNX tidak ditemukan: {onnx_path.resolve()}")
        print("Silakan jalankan `python export_onnx.py` terlebih dahulu!")
        return

    if not csv_path.exists():
        print(f"[ERROR] File dataset labels.csv tidak ditemukan: {csv_path.resolve()}")
        return

    print("=" * 80)
    print("[START] Benchmark Inferensi High-Speed ONNX Runtime (RTX 3050 GPU)")
    print(f"  Model ONNX Dir : {onnx_path.resolve()}")
    print(f"  Dataset Dir    : {dataset_path.resolve()}")
    print("=" * 80)

    # 1. Tentukan Provider (Prioritaskan CUDA, fallback ke CPU)
    # Cek apakah CUDA tersedia di ONNX Runtime DAN PyTorch sebelum mengaktifkan GPU
    import onnxruntime as ort
    import torch
    available_providers = ort.get_available_providers()
    print(f"[INFO] Provider ONNX Runtime yang tersedia: {available_providers}")

    # Verifikasi ganda: CUDA harus tersedia di ORT dan torch.cuda juga harus aktif
    cuda_ort_ok = "CUDAExecutionProvider" in available_providers
    cuda_torch_ok = torch.cuda.is_available()

    if cuda_ort_ok and cuda_torch_ok:
        provider = "CUDAExecutionProvider"
        use_cuda = True
        print("[HARDWARE ACCELERATION] Memakai NVIDIA GPU (CUDAExecutionProvider)")
    else:
        provider = "CPUExecutionProvider"
        use_cuda = False
        if not cuda_torch_ok:
            print("[INFO] PyTorch versi CPU-only terdeteksi, menggunakan CPUExecutionProvider.")
        else:
            print("[INFO] CUDA Runtime tidak cocok (butuh CUDA 13 + cuDNN 9), fallback ke CPU.")
        print("[INFO] Untuk GPU, install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

    # 2. Memuat Model ONNX & Processor
    # Export menghasilkan file terpisah (decoder_model.onnx + decoder_with_past_model.onnx)
    # Perlu deklarasi use_merged=False agar ORT tahu file mana yang dipakai
    print("\n[STEP 1] Memuat Model ONNX & Processor...")
    model = ORTModelForVision2Seq.from_pretrained(
        str(onnx_path),
        provider=provider,
        use_merged=False,  # Gunakan file decoder terpisah (bukan merged)
        decoder_file_name="decoder_model.onnx",
        decoder_with_past_file_name="decoder_with_past_model.onnx",
    )
    processor = TrOCRProcessor.from_pretrained(str(onnx_path))

    # Membaca labels.csv secara manual agar merchant name yang mengandung koma
    # (misal: "QRIS BN, PULSA & INTERNET") tidak salah diparse oleh pandas.
    # Caranya: split hanya pada koma PERTAMA di setiap baris.
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)  # Skip baris header (file_name,text)
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Pisah hanya di koma pertama -> kolom file_name dan text
            first_comma = line.index(",")
            file_name_col = line[:first_comma].strip()
            text_col = line[first_comma + 1:].strip().strip('"')
            rows.append({"file_name": file_name_col, "text": text_col})
    df = pd.DataFrame(rows)
    print(f"[STEP 2] Memuat {len(df)} sampel data uji dari labels.csv")


    latencies = []
    cer_scores = []
    exact_matches = 0

    print("\n" + "=" * 80)
    print(f"{'No':<4} | {'File Gambar':<22} | {'Teks Asli (Ground Truth)':<30} | {'Hasil ONNX TrOCR':<30} | {'Latency':<8}")
    print("-" * 105)

    # Warmup Model: jalankan satu inferensi dummy agar model warming up
    # (menghilangkan overhead cold-start dari statistik latency)
    print("[INFO] Warming up model...")
    dummy_img = Image.new("RGB", (384, 384), color="white")
    dummy_inputs = processor(dummy_img, return_tensors="pt")
    if use_cuda:
        # Pindahkan tensor ke GPU hanya jika CUDA benar-benar aktif
        dummy_inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in dummy_inputs.items()}
    _ = model.generate(**dummy_inputs, max_new_tokens=10, num_beams=1)
    print("[INFO] Warmup selesai!")

    for idx, row in df.iterrows():
        img_name = str(row['file_name']).strip()
        gt_text = str(row['text']).strip()
        img_file = images_dir / img_name

        if not img_file.exists():
            continue

        image = Image.open(img_file).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values

        if use_cuda:
            pixel_values = pixel_values.to("cuda")

        # Mengukur Waktu Inferensi Presisi Tinggi
        t0 = time.perf_counter()
        
        # Super-Fast Generation Config
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=25,  # QRIS Merchant Name rata-rata < 25 karakter
            num_beams=1          # Greedy Search untuk kecepatan maksimal
        )
        
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

        cer = compute_cer(gt_text, pred_text)
        cer_scores.append(cer)
        is_exact = (gt_text == pred_text)
        if is_exact:
            exact_matches += 1

        match_icon = "[OK]" if is_exact else "[MISMATCH]"
        print(f"{idx+1:<4} | {img_name:<22} | {gt_text[:30]:<30} | {pred_text[:30]:<30} | {latency_ms:6.1f} ms {match_icon}")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    avg_cer = (sum(cer_scores) / len(cer_scores)) * 100 if cer_scores else 0
    exact_acc = (exact_matches / len(df)) * 100 if len(df) > 0 else 0

    print("=" * 80)
    print("[RANGKUMAN HASIL BENCHMARK HIGH-SPEED ONNX TrOCR]")
    print(f"  - Total Sampel Uji  : {len(df)}")
    print(f"  - Rata-rata Latency : {avg_latency:.2f} ms / baris teks")
    print(f"  - Min / Max Latency : {min_latency:.2f} ms / {max_latency:.2f} ms")
    print(f"  - Exact Match Acc   : {exact_acc:.2f}% ({exact_matches}/{len(df)})")
    print(f"  - Avg CER           : {avg_cer:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test ONNX Inference Benchmark")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["merchant_name", "general"],
        default="merchant_name",
        help="Tipe model untuk diuji: merchant_name atau general"
    )
    args = parser.parse_args()

    BASE_DIR = Path(__file__).parent
    if args.model_type == "general":
        ONNX_DIR = BASE_DIR / "models" / "trocr_general_onnx"
        DATASET_DIR = BASE_DIR / "ocr_datasets" / "dataset_general"
    else:
        ONNX_DIR = BASE_DIR / "models" / "trocr_merchant_name_onnx"
        DATASET_DIR = BASE_DIR / "ocr_datasets" / "dataset_merchant_name"

    run_onnx_benchmark(str(ONNX_DIR), str(DATASET_DIR))

