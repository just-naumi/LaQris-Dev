"""
=============================================================================
LAQRIS TROCR FINE-TUNING — AUGMENTATIONS (utils/augmentations.py)
=============================================================================
Fungsi augmentasi gambar simulasi stiker fisik QRIS:
- Noise cetakan printer thermal & kamera HP
- Blur lensa & pencahayaan buram
- Sedikit rotasi/kemiringan stiker
=============================================================================
"""

import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

def apply_random_augmentations(pil_img: Image.Image) -> Image.Image:
    """Menerapkan efek augmentasi realistis pada gambar potongan teks QRIS."""
    img = pil_img.copy()

    # 1. Acak Kecerahan & Kontras
    if random.random() > 0.4:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    if random.random() > 0.4:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.4))

    # 2. Acak Rotasi Ringan (-3 sampai +3 derajat)
    if random.random() > 0.5:
        angle = random.uniform(-3.0, 3.0)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    # 3. Acak Blur Lensa Ringan
    if random.random() > 0.6:
        blur_radius = random.uniform(0.3, 1.1)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 4. Tambahkan Noise Piksel (Gaussian Noise)
    if random.random() > 0.5:
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 12), np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)

    return img
