"""
Script Generator Dataset Feedback QRIS Statis (10.000 Sampel) - Versi 4.0 Primary Intent Hierarchy
Menerapkan Hierarki Niat Utama (Primary Intent) yang tegas untuk ekosistem LaQris EMRS:

1. K0: PENIPUAN_STIKER_QRIS_PALSU (Physical Sticker Tampering / Overlay)
2. K1: KETIDAKSESUAIAN_IDENTITAS_MERCHANT (Identity Mismatch Murni tanpa manipulasi fisik)
3. K2: PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE (Illegal Fee / Surcharge / Markup - Prioritas Finansial)
4. K3: KONDISI_FISIK_QRIS_RUSAK (Physical Wear, Tear, Sun Fading, Scratches)
5. K4: KUALITAS_LAYANAN_MERCHANT (Service Non-Compliance - Penolakan QRIS, Min. Belanja, Kasir Judes - Steril Fee)
6. K5: QRIS_NORMAL_MERCHANT_TERPERCAYA (Positive Trust Score + Hard Negatives)
7. K6: FEEDBACK_AMBIGU (Zero-Evidence / Vague / Incomplete)
"""

import os
import random
import csv

random.seed(42)

CATEGORIES = {
    "PENIPUAN_STIKER_QRIS_PALSU": {
        "severity": "CRITICAL",
        "sentiment": "NEGATIVE"
    },
    "KETIDAKSESUAIAN_IDENTITAS_MERCHANT": {
        "severity": "HIGH",
        "sentiment": "NEGATIVE"
    },
    "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE": {
        "severity": "HIGH",
        "sentiment": "NEGATIVE"
    },
    "KONDISI_FISIK_QRIS_RUSAK": {
        "severity": "MEDIUM",
        "sentiment": "NEGATIVE"
    },
    "KUALITAS_LAYANAN_MERCHANT": {
        "severity": "MEDIUM",
        "sentiment": "NEGATIVE"
    },
    "QRIS_NORMAL_MERCHANT_TERPERCAYA": {
        "severity": "LOW",
        "sentiment": "POSITIVE"
    },
    "FEEDBACK_AMBIGU": {
        "severity": "MEDIUM",
        "sentiment": "NEUTRAL"
    }
}

LOCATIONS = [
    "meja kasir", "akrilik meja", "etalase toko", "gerobak jualan", "plang kasir",
    "kotak amal masjid", "pintu masuk toko", "booth bazar", "warung makan",
    "counter pembayaran", "meja makan", "stand minuman", "kaca kasir",
    "meja barcode", "tent card meja", "stiker dinding", "kedai kopi"
]

MERCHANTS = [
    "Warung Nasi Bu Imas", "Kopi Kenangan Senja", "Toko Sembako Berkah",
    "Apotek Sehat Sejahtera", "RM Padang Murah", "Bakso Pak Kumis",
    "Martabak Bangka 88", "Sate Madura Cak Sholeh", "Laundry Kilat 99",
    "Masjid Al-Ikhlas", "Bengkel Motor Jaya", "Minimarket Barokah",
    "Kedai Mie Aceh", "Soto Ayam Lamongan Cak Har", "Ayam Geprek Juara",
    "Es Teh Nusantara", "Barbershop Bro & Co", "Counter Pulsa Cell"
]

CITIES = ["Bandung", "Jakarta", "Surabaya", "Medan", "Yogyakarta", "Semarang", "Malang", "Bekasi", "Tangerang", "Makassar"]
APPS = ["BCA Mobile", "wondr by BNI", "Livin by Mandiri", "BRImo", "DANA", "GoPay", "OVO", "ShopeePay", "LinkAja"]

FAKE_NAMES = [
    "rekening pribadi AGUS SETIAWAN", "rekening DANA 0812988xxx", "GOPAY TOPUP PENIPU",
    "rekening pribadi BAMBANG HERMANTO", "CV MAJU FIKTIF", "JOKO PRIBADI REK", "KASIH AMAL PALSU",
    "DEWI SARTIKA PRIVATE", "PENGUNDIAN RESMI ABAL", "DEPOSIT PULSA ONLINE",
    "rekening pribadi HENDRA", "RUDI HARTONO REK", "WARUNG PALSU 99"
]

NOMINALS = [
    ("15.000", "150.000"), ("25.000", "250.000"), ("50.000", "500.000"),
    ("10.000", "100.000"), ("35.000", "350.000"), ("75.000", "750.000")
]

# ── TEMPLATES POLA HIERARKI PRIMARY INTENT ───────────────────────────────────

TEMPLATES_STIKER_PALSU = [
    "Hati-hati stiker barcode QRIS fisik di {loc} {merch} {city} sengaja ditimpa stiker tempelan palsu lain oleh oknum penipu!",
    "Tolong dicek min, stiker QR code di akrilik meja {merch} ada dua lapisan bertumpuk di mana barcode atas adalah stiker palsu.",
    "Pas saya raba akrilik barcode di {loc} {merch}, permukaannya terasa ada stiker tempelan baru yang sengaja menutupi QR code resmi toko.",
    "Ada stiker QRIS ganda mencurigakan di etalase kasir {merch}, barcode asli milik kasir ditutup dengan stiker penipu.",
    "QRIS fisik di {loc} {merch} jelas-jelas ditempeli stiker baru di atas akrilik resmi toko, terindikasi kuat pemalsuan barcode fisik.",
    "Waspada modus penempelan stiker palsu di {loc} {merch}, kode QR asli toko sudah dilapisi stiker barcode cetakan luar.",
    "Stiker QRIS di {merch} {city} ini palsu bro, ada stiker tipis yang sengaja dipasang menutupi barcode resmi kasir toko.",
    "Jangan scan QRIS di {loc} {merch}, itu barcode QRIS palsu sengaja ditimpa stiker penipu untuk mengalihkan transaksi pelanggan!",
    "Saya menemukan stiker QRIS tempelan penipu di {loc} {merch}, ujung stikernya terkelupas dan di bawahnya masih ada barcode aslinya.",
    "Di {merch} {city}, stiker QRIS resmi di meja kasir ditutup stiker barcode lain oleh pihak tidak bertanggung jawab.",
    "Barcode QRIS fisik di {merch} kelihatan nempel stiker baru di atas akrilik meja kasir, modus penipuan tempel stiker barcode."
]

TEMPLATES_IDENTITAS_BEDA = [
    "Stiker barcode QRIS di meja kasir {merch} tampak asli dan bersih, tetapi saat saya scan nama merchant di aplikasi {app} muncul nama perorangan {fake_acc}.",
    "Tidak ada stiker palsu di akrilik {merch}, namun nama penerima digital yang tertera di layar ponsel saya beda jauh sama plang toko fisik.",
    "Lokasi toko fisik jelas {merch} di {city}, tapi saat scan QRIS resmi toko nama tujuan transfer terdaftar atas nama {fake_acc} di luar daerah.",
    "Saya batalkan bayar QRIS di {merch} karena nama rekening tujuan pada QRIS resminya mencurigakan atas nama rekening pribadi {fake_acc}.",
    "Plang fisik warung bertuliskan {merch}, tapi saat scan barcode QRIS di kasir nama penerima yang muncul justru PT Bodong / {fake_acc}.",
    "Apakah akun QRIS di {merch} ini resmi? Barcode tokonya bersih tapi nama penerima saat discan pakai {app} bukan nama toko pemiliknya.",
    "Identitas merchant QRIS tidak cocok antara banner fisik toko {merch} dengan nama digital NMID saat konfirmasi bayar di HP.",
    "QRIS toko {merch} dicurigai disalahgunakan karena nama penerima pembayaran pada aplikasi perbankan saya adalah {fake_acc}.",
    "Toko fisik {merch} berada di {city} tapi merchant NMID QRIS terdaftar atas nama orang lain yaitu {fake_acc} bukan nama usaha.",
    "Saya baru menyadari setelah pembayaran selesai bahwa nama merchant yang muncul pada detail transaksi QRIS tidak sesuai dengan nama toko tempat saya berbelanja di {merch}.",
    "Saya tadi langsung scan kode QRIS yang tersedia di meja kasir {merch}, namun setelah pembayaran berhasil saya melihat detail penerimanya bernama {fake_acc} yang tidak dikenal toko."
]

TEMPLATES_SURCHARGE = [
    "Kasir di {merch} mengenakan biaya tambahan admin Rp1.000 saat bayar pakai QRIS statis, padahal aturan Bank Indonesia melarang keras surcharge QRIS.",
    "Ada tambahan biaya seribu rupiah dari harga aslinya saat transaksi menggunakan QRIS di {merch}, kasir bilang untuk biaya admin QRIS.",
    "Di {merch}, harga belanjaan dinaikkan kalau bayar pakai QRIS statis, kasir membebankan potongan fee MDR QRIS kepada pembeli.",
    "Kena pungutan liar surcharge Rp2.000 saat transaksi QRIS di {loc} {merch}, mohon ditertibkan merchant nakal ini.",
    "Penjual di {merch} meminta saya melebihkan uang saat input nominal QRIS dengan alasan biaya administrasi jika bayar nontunai.",
    "Kasir meminta biaya tambahan Rp1.000 di luar total belanjaan nota jika pelanggan ingin menggunakan barcode QRIS di {merch}.",
    "Toko {merch} di {city} mengenakan biaya charge tambahan sebesar Rp2.000 khusus untuk setiap transaksi yang menggunakan QRIS statis.",
    "Total belanja 45 ribu tapi pas discan kasir {merch} menyuruh saya mengetik 47 ribu di HP buat biaya charge admin QRIS.",
    "Kenapa bayar pakai QRIS di {merch} ada biaya ekstra? Bukannya merchant dilarang membebankan biaya transaksi kepada pembeli?",
    "Pungutan biaya tambahan tidak resmi oleh kasir {merch} saat transaksi QRIS manual, ada mark up harga dari struk belanja.",
    "Kasir menolak mentransaksikan harga normal jika bayar lewat QRIS dan mewajibkan pembeli menambah nominal biaya ekstra seribu rupiah."
]

TEMPLATES_RUSAK_PUDAR = [
    "Stiker QRIS fisik di {loc} {merch} sudah sobek parah pada bagian pola deteksi sehingga kamera HP tidak bisa scan sama sekali.",
    "Barcode QRIS statis di kasir {merch} kusam dan pudar terkena terik matahari, autofocus kamera aplikasi {app} gagal memindai.",
    "Akrilik meja tempat barcode QRIS di {merch} banyak goresan lecet parah sampai pola matriks QRIS-nya hilang sebagian.",
    "Kamera pemindai {app} saya tidak bisa membaca stiker QRIS di {merch} karena stikernya basah dan tintanya luntur menyebar.",
    "Stiker QRIS fisik toko {merch} sudah buram dan terkelupas di sudut bawah sehingga scanner aplikasi selalu gagal deteksi.",
    "Susah sekali scan QRIS di {loc} {merch}, stiker barcode kotor terkena minyak gorengan dan debu tebal di meja kasir.",
    "QRIS fisik di gerobak {merch} rusak terlipat sehingga aplikasi perbankan menolak memproses gambar barcode tersebut.",
    "Resolusi cetakan stiker QRIS di kasir {merch} pecah dan buram, scanner HP tidak mampu mengunci titik fokus kode QR.",
    "Stiker QRIS statis di {loc} {merch} mengelupas separuh dari akrilik mejanya, pembeli kesulitan saat mau scan."
]

TEMPLATES_LAYANAN_MERCHANT = [
    "Pelayanan kasir di {merch} sangat mengecewakan, kasir menolak pembayaran QRIS secara sepihak dengan alasan minimal belanja harus Rp30.000.",
    "Kasir di toko {merch} bersikap sangat judes dan ketus ketika saya bilang mau bayar pakai QRIS, kasir memaksa pembeli harus bayar uang tunai saja.",
    "Kasir toko {merch} enggan melayani scan QRIS dan beralasan kasir lagi malas mengecek HP, pembeli dipaksa bayar cash.",
    "Pembeli dipersulit saat mau bayar QRIS di {merch}, kasir bilang barcode QRIS disimpan di laci dan menolak mengeluarkannya.",
    "Sikap staf kasir di {loc} {merch} sangat tidak sopan dan menolak scan QRIS dengan alasan sinyal toko jelek padahal koneksi internet lancar.",
    "Kasir di gerai {merch} menahan barang belanjaan dengan sikap mencurigai pembeli secara tidak sopan padahal bukti transaksi sukses sudah saya perlihatkan.",
    "Merchant {merch} di {city} menolak menerima transaksi QRIS dari aplikasi dompet digital tertentu secara sepihak tanpa alasan yang jelas.",
    "Kasir tidak kooperatif saat proses scan QRIS di meja kasir {merch}, pembeli disuruh menunggu lama dan diabaikan pelayanannya.",
    "Kasir nolak melayani pembayaran QRIS karena alasan antrean kasir sedang ramai, maunya hanya uang pas saja.",
    "Pelayanan kasir sangat buruk, malah memarahi pelanggan yang tidak membawa uang tunai saat mau bayar lewat QRIS."
]

TEMPLATES_NORMAL_SUKSES = [
    "Saya membeli makanan di warung makan {merch} dan membayar menggunakan QRIS seperti biasanya, nominal yang harus dibayar pas sesuai harga, transaksi berhasil dalam beberapa detik dan bukti pembayaran QRIS langsung muncul di aplikasi.",
    "Kasir ramah menunjukkan barcode QRIS di meja {merch}, nominal pembayaran saya input pas Rp{nom1} tanpa biaya admin dan notifikasi uang masuk QRIS langsung berbunyi di soundbox kasir.",
    "Cek mutasi setelah transaksi QRIS di {merch}, nominal pembayaran pas dengan yang saya ketik, tidak ada biaya potongan dan pembayaran QRIS selesai aman sentosa.",
    "Proses pembayaran QRIS statis berlangsung sangat cepat tanpa kendala apapun di {loc} {merch}, struk dan bukti transfer digital QRIS tersimpan rapi di aplikasi {app}.",
    "Tadi bayar pakai QRIS statis di {loc} {merch}, nama penerima di layar konfirmasi cocok dengan plang toko dan transaksi QRIS langsung terverifikasi hijau.",
    "Nominal belanjaan Rp{nom1} saya bayar dengan scan barcode QRIS di kasir {merch}, semua detail informasi pembayaran QRIS cocok dan kasir langsung menyerahkan nota lunas.",
    "Transaksi QRIS berhasil diproses seketika di {merch}, kasir mengecek terminal soundbox QRIS dan nominal uang masuk cocok dengan pesanan tanpa kendala.",
    "Tidak ada masalah sama sekali saat transaksi QRIS di {merch}, barcode bersih terlindung akrilik, pembayaran lancar dan bukti bayar resmi QRIS langsung keluar.",
    "Struk pembayaran QRIS resmi otomatis terbit di aplikasi {app}, nominal yang didebet pas Rp{nom1} tidak ada pungutan biaya admin QRIS tambahan.",
    "Uang masuk seketika ke mutasi QRIS toko {merch}, kasir langsung menyatakan transaksi lunas dan saya menerima pesanan saya dengan puas."
]

TEMPLATES_AMBIGU = [
    "Tadi waktu saya sedang melakukan proses pembayaran scan QRIS di meja kasir {merch} rasanya ada sesuatu yang janggal pada tampilan layar aplikasi saya tetapi saya tidak yakin apa yang sebenarnya terjadi.",
    "Entah kenapa transaksi pembayaran QRIS yang baru saja saya lakukan di {merch} {city} terasa agak aneh dan membingungkan ketika selesai diproses, tolong dicek ya tim LaQris.",
    "Saya bayar pakai QRIS barusan di minimarket {merch} tapi kok perasaan saya kurang enak ya, apakah transaksi QRIS saya tadi sudah benar-benar sah?",
    "Tadi mbak kasir di {merch} sempat mengucapkan sesuatu saat saya sedang memindai barcode QRIS di meja depan tetapi suaranya tidak terdengar jelas sehingga membuat saya bingung mengenai status pembayaran saya.",
    "Ada sedikit hal yang terasa tidak biasa waktu saya mencoba melakukan pembayaran QRIS di {loc} {merch} tadi siang, tampilannya agak berbeda dari biasanya dan saya ingin memastikan keamanannya.",
    "Proses pembayaran QRIS sebenarnya sudah selesai di layar HP saya, cuma saya merasa agak bingung saja dengan detail tampilan konfirmasi QRIS yang muncul di toko {merch}.",
    "Kayaknya tadi ada satu hal yang tidak biasa saat saya hendak membayar belanjaan menggunakan QRIS di {merch}, namun saya tidak begitu memperhatikan secara detail sehingga saya ragu.",
    "Saya tadi langsung menutup aplikasi setelah klik bayar QRIS di {merch} sehingga tidak sempat membaca pesan notifikasi yang sempat muncul sekilas di layar ponsel saya.",
    "Mohon bantuan pengecekan riwayat pembayaran QRIS terakhir saya di {merch}, tadi koneksi internet sempat tersendat saat menekan tombol bayar sehingga saya ragu statusnya.",
    "Saya tadi bayar belanjaan pakai barcode QRIS tapi lupa melihat detail layarnya karena terburu-buru pergi, tolong bantu verifikasi apakah transaksi saya aman."
]

def generate_sample(category):
    loc = random.choice(LOCATIONS)
    merch = random.choice(MERCHANTS)
    city = random.choice(CITIES)
    app = random.choice(APPS)
    fake_acc = random.choice(FAKE_NAMES)
    nom1, nom2 = random.choice(NOMINALS)

    if category == "PENIPUAN_STIKER_QRIS_PALSU":
        tmpl = random.choice(TEMPLATES_STIKER_PALSU)
    elif category == "KETIDAKSESUAIAN_IDENTITAS_MERCHANT":
        tmpl = random.choice(TEMPLATES_IDENTITAS_BEDA)
    elif category == "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE":
        tmpl = random.choice(TEMPLATES_SURCHARGE)
    elif category == "KONDISI_FISIK_QRIS_RUSAK":
        tmpl = random.choice(TEMPLATES_RUSAK_PUDAR)
    elif category == "KUALITAS_LAYANAN_MERCHANT":
        tmpl = random.choice(TEMPLATES_LAYANAN_MERCHANT)
    elif category == "FEEDBACK_AMBIGU":
        tmpl = random.choice(TEMPLATES_AMBIGU)
    else:
        tmpl = random.choice(TEMPLATES_NORMAL_SUKSES)

    text = tmpl.format(
        loc=loc, merch=merch, city=city, app=app,
        fake_acc=fake_acc, nom1=nom1, nom2=nom2
    )

    meta = CATEGORIES[category]
    return {
        "text": text.strip(),
        "category": category,
        "severity": meta["severity"],
        "sentiment": meta["sentiment"],
        "qr_type": "STATIC"
    }

def main():
    target_count = 10000
    cat_keys = list(CATEGORIES.keys())
    per_cat = target_count // len(cat_keys)
    remainder = target_count % len(cat_keys)

    print(f"[INFO] Memulai pembuatan {target_count} dataset feedback LaQris EMRS (Versi 4.0 Primary Intent)...")
    dataset = []

    for idx, cat in enumerate(cat_keys):
        count_to_gen = per_cat + (1 if idx < remainder else 0)
        print(f"  -> Menghasilkan {count_to_gen} sampel untuk kategori: {cat}")
        for _ in range(count_to_gen):
            sample = generate_sample(cat)
            dataset.append(sample)

    random.shuffle(dataset)

    output_rows = []
    for idx, item in enumerate(dataset, start=1):
        output_rows.append({
            "id": f"FB-{idx:05d}",
            "text": item["text"],
            "category": item["category"],
            "severity": item["severity"],
            "sentiment": item["sentiment"],
            "qr_type": item["qr_type"]
        })

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "qris_feedback_10k.csv")

    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "text", "category", "severity", "sentiment", "qr_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n[SUKSES] Dataset 10.000 baris LaQris EMRS (Versi 4.0) berhasil dibuat di: {output_path}")
    print(f"Total baris: {len(output_rows)}")

if __name__ == "__main__":
    main()
