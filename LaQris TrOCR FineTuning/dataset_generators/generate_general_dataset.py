"""
=============================================================================
LAQRIS TROCR FINE-TUNING — GENERAL QRIS TEXT DATASET GENERATOR
=============================================================================
Menghasilkan gambar teks sintetis untuk istilah umum stiker QRIS:
- Nama Bank Acquirer: "BANK CENTRAL ASIA", "GOPAY", "SHOPEEPAY", "DANA", "OVO"
- Header/Footer: "SATU QRIS UNTUK SEMUA", "DICETAK OLEH", "NATIONAL MERCHANT ID", "TERMINAL ID"
=============================================================================
"""

import os
import sys
import random
import csv
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.augmentations import apply_random_augmentations

OUTPUT_DIR = os.path.join(BASE_DIR, "ocr_datasets", "dataset_general")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

GENERAL_TEXTS = [
    "BANK CENTRAL ASIA", "BANK MANDIRI", "BANK RAKYAT INDONESIA", "BANK NEGARA INDONESIA",
    "GOPAY", "SHOPEEPAY", "DANA", "OVO", "LINKAJA", "BCA", "BNI", "BRI", "MANDIRI",
    "SATU QRIS UNTUK SEMUA", "PEMBAYARAN NASIONAL", "NATIONAL MERCHANT ID",
    "DICETAK OLEH", "DICEK OLEH", "TERMINAL ID", "TID: 10293847", "A01",
    "STANDAR QR CODE PEMBAYARAN", "ASPI INDONESIA", "GPN LOGO"
]

def get_available_fonts():
    fonts = []
    if os.path.exists(FONTS_DIR):
        for f in os.listdir(FONTS_DIR):
            if f.lower().endswith(('.ttf', '.otf')):
                fonts.append(os.path.join(FONTS_DIR, f))
    return fonts

def create_general_dataset(num_samples=100):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    font_paths = get_available_fonts()

    records = []
    print(f"[GENERATOR] Membuat {num_samples} gambar teks sintetis General QRIS...")

    for i in range(1, num_samples + 1):
        text = random.choice(GENERAL_TEXTS)
        img_name = f"general_{i:05d}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)

        bg_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
        text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))

        img_w = random.randint(220, 360)
        img_h = random.randint(48, 64)
        img = Image.new("RGB", (img_w, img_h), color=bg_color)
        draw = ImageDraw.Draw(img)

        font_size = random.randint(20, 26)
        font = None
        if font_paths:
            try:
                selected_font = random.choice(font_paths)
                font = ImageFont.truetype(selected_font, font_size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if tw > img_w - 20:
            img_w = tw + 40
            img = Image.new("RGB", (img_w, img_h), color=bg_color)
            draw = ImageDraw.Draw(img)

        tx = max(10, (img_w - tw) // 2)
        ty = max(5, (img_h - th) // 2)

        draw.text((tx, ty), text, fill=text_color, font=font)

        img = apply_random_augmentations(img)
        img.save(img_path)

        records.append({"file_name": img_name, "text": text})

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[SUCCESS] Dataset General QRIS berhasil disimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    create_general_dataset(count)
