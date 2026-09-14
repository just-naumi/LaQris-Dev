"""
=============================================================================
LAQRIS TROCR FINE-TUNING — MERCHANT NAME DATASET GENERATOR
=============================================================================
Menghasilkan gambar teks sintetis untuk nama-nama merchant dengan variasi font,
ukuran, dan warna latar. Hasil disimpan ke datasets/dataset_merchant_name/
=============================================================================
"""

import os
import sys
import random
import csv
from PIL import Image, ImageDraw, ImageFont

# Tambahkan direktori utama ke sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.augmentations import apply_random_augmentations

OUTPUT_DIR = os.path.join(BASE_DIR, "ocr_datasets", "dataset_merchant_name")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

SAMPLE_MERCHANT_NAMES = [
    "WARUNG BU SRI", "TOKO MAJU JAYA", "CAFE KOPI KENANGAN", "CV. DIGITAL MEDIA",
    "RESTO SEDERHANA", "WARUNG MAKAN PADANG", "MARTABAK MANIS BANGKA",
    "APOTEK SEHAT BERSAMA", "DEPO AIR MINUM RO", "KEDAI KOPI NIKMAT",
    "BARBERSHOP CAPSTER", "LAUNDRY KILAT 24 JAM", "TOKO KELONTONG PAK HAJI",
    "BAKSO SOLO ASLI", "SOTO AYAM LAMONGAN", "BOUTIQUE FASHION INDAH",
    "JAYA MOTOR REPAIR", "FOTOCOPI STAR COPY", "MINI MARKET SEJAHTERA",
    "WARUNG SEAFOOD 99", "RM PADANG SALERO KITA", "ES TEH MANIS SOLO",
    "NASI GORENG GILA", "AYAM GEPREK BENSU", "TOKO OBAT MANCUR"
]

PREFIXES = ["PT", "CV", "UD", "TB", "PD", "WARUNG", "TOKO", "RM", "KEDAI", "CAFE", "DISTRO", "APOTEK"]
SURNAMES = ["MAJU", "JAYA", "SEJAHTERA", "BERKAH", "ABADI", "SENTOSA", "MAKMUR", "UTAMA", "KENCANA", "LUKISAN", "INDRA", "NUSANTARA", "GLOBAL"]

def generate_random_name():
    if random.random() > 0.4:
        return random.choice(SAMPLE_MERCHANT_NAMES)
    else:
        p = random.choice(PREFIXES)
        s1 = random.choice(SURNAMES)
        s2 = random.choice(SURNAMES)
        return f"{p} {s1} {s2}"

def get_available_fonts():
    fonts = []
    if os.path.exists(FONTS_DIR):
        for f in os.listdir(FONTS_DIR):
            if f.lower().endswith(('.ttf', '.otf')):
                fonts.append(os.path.join(FONTS_DIR, f))
    return fonts

def create_merchant_dataset(num_samples=100):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    font_paths = get_available_fonts()

    records = []
    print(f"[GENERATOR] Membuat {num_samples} gambar teks sintetis nama merchant...")

    for i in range(1, num_samples + 1):
        text = generate_random_name()
        img_name = f"merchant_{i:05d}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)

        # Buat gambar kanvas
        bg_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
        text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))

        img_w = random.randint(220, 360)
        img_h = random.randint(48, 64)
        img = Image.new("RGB", (img_w, img_h), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Pilihkah font jika ada, atau gunakan default
        font_size = random.randint(20, 28)
        font = None
        if font_paths:
            try:
                selected_font = random.choice(font_paths)
                font = ImageFont.truetype(selected_font, font_size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        # Posisi tengah
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Sesuaikan ukuran kanvas jika teks lebih lebar
        if tw > img_w - 20:
            img_w = tw + 40
            img = Image.new("RGB", (img_w, img_h), color=bg_color)
            draw = ImageDraw.Draw(img)

        tx = max(10, (img_w - tw) // 2)
        ty = max(5, (img_h - th) // 2)

        draw.text((tx, ty), text, fill=text_color, font=font)

        # Terapkan augmentasi
        img = apply_random_augmentations(img)
        img.save(img_path)

        records.append({"file_name": img_name, "text": text})

    # Simpan labels.csv
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[SUCCESS] Dataset nama merchant berhasil disimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    create_merchant_dataset(count)
