"""
=============================================================================
LAQRIS TROCR FINE-TUNING — NMID DATASET GENERATOR
=============================================================================
Menghasilkan gambar teks sintetis khusus untuk NMID (National Merchant ID).
Variasi: "ID102039485712", "NMID : ID1020304050", "ID 102 039 485 102"
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

OUTPUT_DIR = os.path.join(BASE_DIR, "ocr_datasets", "dataset_nmid")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

def generate_random_nmid():
    prefix_opts = ["ID", "NMID: ID", "NMID : ID", "NMID ", ""]
    p = random.choice(prefix_opts)
    digits = "".join([str(random.randint(0, 9)) for _ in range(13)])
    
    if "ID" in p:
        return f"{p}{digits}"
    else:
        return f"ID{digits}"

def get_available_fonts():
    fonts = []
    if os.path.exists(FONTS_DIR):
        for f in os.listdir(FONTS_DIR):
            if f.lower().endswith(('.ttf', '.otf')):
                fonts.append(os.path.join(FONTS_DIR, f))
    return fonts

def create_nmid_dataset(num_samples=100):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    font_paths = get_available_fonts()

    records = []
    print(f"[GENERATOR] Membuat {num_samples} gambar teks sintetis NMID...")

    for i in range(1, num_samples + 1):
        text = generate_random_nmid()
        img_name = f"nmid_{i:05d}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)

        bg_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
        text_color = (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40))

        img_w = random.randint(240, 340)
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
            img_w = tw + 35
            img = Image.new("RGB", (img_w, img_h), color=bg_color)
            draw = ImageDraw.Draw(img)

        tx = max(8, (img_w - tw) // 2)
        ty = max(5, (img_h - th) // 2)

        draw.text((tx, ty), text, fill=text_color, font=font)

        img = apply_random_augmentations(img)
        img.save(img_path)

        records.append({"file_name": img_name, "text": text})

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[SUCCESS] Dataset NMID berhasil disimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    create_nmid_dataset(count)
