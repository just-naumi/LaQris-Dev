"""
Script Generator Dataset Feedback QRIS Statis (14.000 Sampel Seimbang) - Versi 5.0 Contrastive Hard-Boundary
Ekosistem LaQris Enhanced Merchant Reputation System (EMRS).

Setiap kelas memiliki TEPAT 2.000 SAMPEL SEIMBANG (Total 14.000 baris):
1. K0: PENIPUAN_STIKER_QRIS_PALSU (2.000 sampel - Physical Sticker Tampering / Overlay)
2. K1: KETIDAKSESUAIAN_IDENTITAS_MERCHANT (2.000 sampel - 1.000 Mirror Contrastive Mismatch + 1.000 Diverse Mismatch)
3. K2: PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE (2.000 sampel - Illegal Fee / Surcharge / Markup)
4. K3: KONDISI_FISIK_QRIS_RUSAK (2.000 sampel - Physical Wear, Tear, Sun Fading, Scratches)
5. K4: KUALITAS_LAYANAN_MERCHANT (2.000 sampel - Penolakan QRIS, Min. Belanja, Kasir Judes - Bebas Fee)
6. K5: QRIS_NORMAL_MERCHANT_TERPERCAYA (2.000 sampel - 1.000 Mirror Contrastive Match + 1.000 Smooth Trust)
7. K6: FEEDBACK_AMBIGU (2.000 sampel - Zero-Evidence / Vague / Incomplete)
"""

import os
import random
import csv
import pandas as pd

random.seed(42)

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

# ── CONTRASTIVE MIRROR PAIRS COMPONENTS (K1 vs K5) ──────────────────────────
MERCHANT_TYPES = [
    ("warung makan", "Warung Bu Siti"),
    ("kedai kopi", "Kopi Kenangan Senja"),
    ("toko kelontong", "Toko Berkah Jaya"),
    ("apotek", "Apotek Sehat Medika"),
    ("minimarket", "Minimarket Barokah"),
    ("bengkel motor", "Bengkel Maju Lancar"),
    ("laundry", "Laundry Wangi Bersih"),
    ("counter pulsa", "Counter Cahaya Cell"),
    ("restoran padang", "Rumah Makan Sederhana Minang"),
    ("stand boba", "Boba Fresh Delight"),
    ("toko buah", "Toko Buah Segar Mandiri"),
    ("kedai bakso", "Bakso Super Urat Mas Bejo"),
    ("toko pakaian", "Fashion Style Boutique"),
    ("barbershop", "Ganteng Barbershop"),
    ("toko alat tulis", "Toko Buku Gemilang")
]

OPERATOR_PAIRS = [
    (
        "sama persis dengan nama toko tempat saya berbelanja",
        "berbeda dengan nama toko tempat saya berbelanja"
    ),
    (
        "benar-benar sesuai dengan nama tempat usaha yang saya datangi",
        "sama sekali tidak sesuai dengan nama tempat usaha yang saya datangi"
    ),
    (
        "cocok dan identik dengan plang nama toko di depan saya",
        "tidak cocok dan berbeda jauh dari plang nama toko di depan saya"
    ),
    (
        "memang resmi atas nama gerai tersebut",
        "malah mengarah ke rekening pribadi orang lain yang tidak dikenal"
    ),
    (
        "valid dan konsisten dengan identitas resmi merchant",
        "tidak konsisten dan mengindikasikan rekening pihak ketiga yang mencurigakan"
    ),
    (
        "tepat milik toko ini tanpa ada perbedaan sedikit pun",
        "bukan milik toko ini melainkan atas nama perseorangan"
    ),
    (
        "benar terdaftar atas nama merchant tersebut",
        "justru tertera nama merchant lain yang sama sekali berbeda"
    ),
    (
        "sesuai dengan nama usaha yang tertera di kasir",
        "tidak sesuai dengan nama usaha yang tertera di kasir"
    ),
    (
        "identik dengan nama toko dan tidak ada kejanggalan",
        "berbeda dari nama toko sehingga menimbulkan kecurigaan"
    ),
    (
        "sepenuhnya cocok dengan nama outlet tempat saya belanja",
        "justru tidak cocok dan memunculkan nama pribadi asing"
    )
]

CONTRASTIVE_CONTEXTS = [
    "Sebelum menekan tombol konfirmasi pembayaran QRIS di {store_type}, saya memeriksa nama penerima di layar aplikasi. Hasil pengecekan menunjukkan nama penerima {op}, dan transaksi berhasil diproses.",
    "Tadi saya bayar pakai QRIS di {store_name}. Pembayaran sukses, lalu saat mengecek rincian transaksi di aplikasi, nama merchant yang tercantum {op}.",
    "Kasir di {store_name} mengarahkan untuk scan barcode QRIS. Saya mencocokkan nama usaha pada aplikasi pembayaran dengan nama toko, dan hasilnya {op}.",
    "Saya memeriksa detail informasi QRIS termasuk NMID dan nama merchant sebelum transfer. Informasi yang tampil di layar HP {op}.",
    "Saat melakukan transaksi QRIS di meja kasir {store_type}, saya teliti melihat nama penerima transfer pada e-wallet. Ternyata nama merchant tersebut {op}.",
    "Nama merchant yang muncul di aplikasi setelah scan QRIS {op}.",
    "Setelah transaksi QRIS berhasil, saya cek notifikasi dan struk digital di perbankan saya. Nama penerima dana {op}.",
    "Demi keamanan sebelum bayar QRIS, saya memastikan identitas penerima terlebih dahulu. Nama yang terdaftar pada sistem {op}.",
    "Nominal belanja sudah pas dan saya cek nama merchant di aplikasi pembayaran. Nama penerima {op}, dan pembayaran selesai lancar.",
    "Beli barang di {store_name} bayar non-tunai via QRIS. Pas dicek di konfirmasi transaksi, nama penerimanya {op}."
]

# ── TEMPLATES K0: PENIPUAN_STIKER_QRIS_PALSU ─────────────────────────────────
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
    "Barcode QRIS fisik di {merch} kelihatan nempel stiker baru di atas akrilik meja kasir, modus penipuan tempel stiker barcode.",
    "Awas ada stiker QRIS siluman di {loc} {merch}, stikernya dobel bertumpuk dan sangat mencurigakan.",
    "Saya curiga barcode QRIS di kasir {merch} adalah tempelan palsu karena stiker aslinya tertutup stiker baru yang miring."
]

# ── TEMPLATES K1 (DIVERSE SCENARIOS): KETIDAKSESUAIAN_IDENTITAS_MERCHANT ──────
TEMPLATES_IDENTITAS_DIVERSE = [
    "Stiker barcode QRIS di meja kasir {merch} tampak asli dan bersih, tetapi saat saya scan nama merchant di aplikasi {app} muncul nama perorangan {fake_acc}.",
    "Tidak ada stiker palsu di akrilik {merch}, namun nama penerima digital yang tertera di layar ponsel saya beda jauh sama plang toko fisik.",
    "Barcode QRIS fisik di {loc} {merch} mulus tanpa tempelan ganda, tapi tujuan transfer di {app} bukan ke rekening resmi toko melainkan {fake_acc}.",
    "Saya scan QRIS asli toko di {merch}, pas masuk halaman konfirmasi {app} nama pemilik merchant yang keluar adalah nama pribadi asing {fake_acc}.",
    "Kondisi fisik QRIS di {merch} sangat rapi dan otentik, tapi data NMID dan nama merchant di aplikasi bank tidak cocok dengan nama usaha tempat belanja.",
    "QRIS fisik di {loc} {merch} dalam kondisi baik, namun ada perbedaan nama penerima di mana sistem menampilkan {fake_acc} bukannya nama toko resmi.",
    "Barcode QRIS di meja kasir {merch} tidak ada manipulasi stiker, tapi saya ragu bayar karena nama toko di aplikasi {app} menunjukkan nama orang lain.",
    "Saat transaksi di {merch}, fisik barcode terlihat resmi, tapi nama merchant penerima dana di m-banking berbeda dari nama toko aslinya.",
    "Saya cek teliti akrilik QRIS di {merch} semuanya bersih, cuma nama merchant yang tertera di aplikasi e-wallet beda dengan nama toko tempat saya beli.",
    "Tidak ada tanda stiker palsu di {loc} {merch}, tapi detail merchant yang muncul di {app} mencurigakan karena mengarah ke {fake_acc}."
]

# ── TEMPLATES K2: PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE ───────────────────────────
TEMPLATES_SURCHARGE = [
    "Kasir di {merch} {city} mengenakan biaya tambahan surcharge Rp 1.000 saat saya bayar tagihan {nom[0]} pakai QRIS, padahal BI melarang surcharge!",
    "Beli barang di {merch} harganya {nom[0]}, tapi kalau bayar pakai barcode QRIS disuruh nambah Rp 2.000 sama kasirnya, ini pungutan liar.",
    "Saya dipaksa bayar fee admin tambahan Rp 1.500 oleh kasir {merch} khusus untuk pembayaran scan QRIS.",
    "Di kasir {merch} {city} ada tulisan bayar QRIS kena biaya tambahan Rp 1.000 per transaksi, kasirnya bilang itu fee admin dari toko.",
    "Parah banget di {merch}, belanja {nom[1]} kalau bayar tunai harganya normal tapi kalau scan QRIS kasir naikin harga dan minta tambahan Rp 2.000.",
    "Kasir di {merch} memungut biaya surcharge Rp 1.000 tanpa persetujuan untuk setiap pembayaran QR code.",
    "Toko {merch} melanggar aturan Bank Indonesia karena menarik biaya tambahan 2% saat saya membayar via QRIS di meja kasir.",
    "Saya keberatan karena belanja di {merch} kena cas tambahan Rp 1.000 gara-gara bayar pakai QRIS bukannya uang tunai.",
    "Di {merch} kasirnya terang-terangan minta uang tambahan Rp 1.500 dengan alasan biaya potong saldo QRIS.",
    "Pungutan liar di kasir {merch}, harga belanja {nom[0]} dinaikkan jadi lebih mahal kalau bayar pakai scan barcode QRIS.",
    "Kasir di {merch} {city} memaksa saya membayar biaya tambahan surcharge Rp 1.000 saat saya bayar tagihan {nom[0]} pakai QRIS.",
    "Pelayanan kasir di {merch} arogan sekali dan memaksa pembeli nambah uang tunai Rp 2.000 dengan alasan biaya potong saldo QRIS.",
    "Kasir memarahi saya dan menuntut cas tambahan fee admin Rp 1.500 jika ingin melunasi tagihan {nom[0]} lewat scan barcode QRIS.",
    "Saya dipaksa kasir {merch} bayar lebih Rp 1.000 dari total belanjaan karena menggunakan pembayaran QR code di meja kasir.",
    "Kasir di {merch} bersikeras meminta uang cas Rp 2.000 tambahan sambil melayani dengan ketus saat saya scan barcode QRIS."
]

# ── TEMPLATES K3: KONDISI_FISIK_QRIS_RUSAK ───────────────────────────────────
TEMPLATES_FISIK_RUSAK = [
    "Stiker barcode QRIS di meja kasir {merch} sudah sobek parah di bagian tengah sampai kode matriksnya hilang dan gagal discan kamera ponsel.",
    "Akrilik QRIS di {loc} {merch} pecah dan retak melintang tepat di atas QR code, membuat aplikasi e-wallet tidak bisa membaca barcode.",
    "Tolong diganti stiker QRIS di {merch} {city}, warnanya sudah luntur dan pudar total terkena panas matahari sehingga barcode tidak terdeteksi.",
    "Permukaan stiker QRIS di {loc} {merch} kotor kena tumpahan minyak dan noda hitam tebal sampai tidak bisa dipindai sama sekali.",
    "Barcode QRIS fisik di meja kasir {merch} tergores benda tajam sangat parah, kamera HP berkali-kali mencoba selalu gagal baca kodenya.",
    "Plang barcode QRIS di {loc} {merch} sudah rusak mengelupas dan basah kuyup terkena hujan sampai kertas stikernya hancur.",
    "Stiker QR code di kasir {merch} burem banget dan buram terkena goresan, tolong segera kirimkan cetakan akrilik QRIS baru.",
    "Kondisi fisik QRIS di {loc} {merch} sangat memprihatinkan, tintanya luntur kecokelatan dan sudut stikernya koyak sobek.",
    "Scan QRIS di {merch} selalu error 'QR Code Tidak Terbaca' karena fisik stiker di meja kasir sudah kusut dan berkerut parah.",
    "Akrilik stand QRIS di {merch} pecah dan buram terkena debu tebal bertahun-tahun, kasir harus mencetak ulang kode barcode yang jelas."
]

# ── TEMPLATES K4: KUALITAS_LAYANAN_MERCHANT ───────────────────────────────────
TEMPLATES_LAYANAN = [
    "Kasir di {merch} {city} menolak pembayaran QRIS saya dengan alasan sinyal toko sedang gangguan dan memaksa harus bayar uang tunai pas.",
    "Di {merch} kasirnya judes banget dan pasang muka masam saat saya bilang mau bayar pakai QRIS di meja kasir.",
    "Toko {merch} menetapkan aturan sepihak minimal belanja Rp 50.000 baru boleh bayar pakai QRIS, padahal saya belanja {nom[0]}.",
    "Kasir {merch} menyembunyikan plang barcode QRIS di bawah meja dan enggan melayani pelanggan yang ingin membayar non-tunai.",
    "Pelayanan di {merch} sangat mengecewakan, kasirnya marah-marah saat disuruh cek notifikasi pembayaran QRIS yang sudah sukses di HP saya.",
    "Saya mau bayar {nom[0]} pakai QRIS di {merch} tapi kasir bilang QRIS hanya untuk belanja di atas Rp 30.000, sangat mengecewakan.",
    "Kasir di kasir {merch} sangat lambat dan tidak paham cara konfirmasi transaksi QRIS sampai saya harus menunggu 15 menit di antrean.",
    "Toko {merch} menempel plang 'QRIS Rusak' padahal kasirnya malas mengecek saldo masuk, pelayanan kasirnya sangat buruk.",
    "Kasir {merch} berkata kasar saat saya menunjukkan bukti berhasil bayar QRIS dan menuduh saya belum bayar sebelum ia melihat HP-nya.",
    "Penolakan sepihak transaksi QRIS di {merch}, kasir bersikeras hanya menerima uang cash fisik dengan nada ketus."
]

# ── TEMPLATES K5 (DIVERSE SCENARIOS): QRIS_NORMAL_MERCHANT_TERPERCAYA ────────
TEMPLATES_NORMAL_DIVERSE = [
    "Transaksi QRIS di {merch} {city} sangat cepat dan lancar, kasir ramah, barcode di akrilik bersih mulus, dan nominal {nom[0]} terdebet pas.",
    "Sangat puas bayar non-tunai di {merch}, scan barcode QRIS langsung terbaca dalam 1 detik via {app} tanpa kendala dan tanpa biaya tambahan.",
    "Stiker QRIS di meja kasir {merch} sangat rapi dan bersih, tidak ada biaya admin tambahan apapun, pelayanan kasir sangat profesional dan cepat.",
    "Pengalaman pembayaran QRIS di {loc} {merch} luar biasa aman dan nyaman, struk langsung tercetak dan nominal pembayaran tepat {nom[0]}.",
    "Barcode QRIS di {merch} sangat jelas dan terang, transaksi berhasil seketika lewat {app}, merchant terpercaya bintang lima.",
    "Belanja di {merch} bayar via QRIS bebas biaya tambahan surcharge, kasir ramah mengonfirmasi transaksi dengan senyum dan sopan.",
    "Semua berjalan sempurna saat bayar QRIS di {merch}, akrilik barcode mulus, tidak ada kejanggalan, kasir melayani dengan sangat baik.",
    "Proses checkout pakai QRIS di {merch} {city} memuaskan, scan cepat, tidak ada biaya tersembunyi, dan bukti transaksi langsung masuk ke e-wallet.",
    "Pelayanan di {merch} jempolan, pembayaran non-tunai QRIS nominal {nom[1]} berhasil diproses tanpa masalah apapun.",
    "QRIS di {loc} {merch} terawat sangat bersih, mudah discan dari jarak jauh, kasir sigap mengecek notifikasi masuk."
]

# ── TEMPLATES K6: FEEDBACK_AMBIGU ─────────────────────────────────────────────
TEMPLATES_AMBIGU = [
    "Saya bingung tadi waktu mau bayar di toko ini.",
    "Tolong dicek ya min.",
    "Gimana ini solusinya kok begini?",
    "Kemarin belanja di kasir rasanya aneh banget.",
    "Tadi ada masalah sedikit waktu transaksi.",
    "Mohon bantuan CS terkait pembayaran saya.",
    "Kurang paham sama sistemnya.",
    "Kayaknya ada yang salah tapi saya gak tahu apa.",
    "Kenapa bisa begitu ya min?",
    "Saya mau komplain soal pelayanan tadi siang.",
    "Tolong diperbaiki fiturnya.",
    "Transaksi tadi terasa mencurigakan.",
    "Ada kendala di kasir tolong bantu.",
    "Bisa tolong dicek akun saya?",
    "Mengecewakan sekali tadi."
]

LONG_TEMPLATES_AMBIGU = [
    "Tadi waktu saya sedang melakukan proses pembayaran di meja kasir {merch} rasanya ada sesuatu yang janggal pada tampilan layar aplikasi saya saat scan barcode, namun saya tidak yakin apa yang sebenarnya terjadi karena saya terlanjur menutup aplikasinya dengan tergesa-gesa tanpa bukti rincian apapun.",
    "Entah kenapa transaksi pembayaran yang baru saja saya lakukan di {merch} {city} terasa agak aneh dan membingungkan ketika selesai diproses, sehingga saya merasa ragu dan ingin meminta tolong pihak tim LaQris untuk membantu memeriksa status akun saya.",
    "Saya barusan menyelesaikan transaksi pembayaran menggunakan QRIS di toko {merch}, namun entah kenapa perasaan saya kurang tenang dan agak ragu apakah proses pembayarannya tadi sudah benar-benar selesai sesuai ketentuan sistem atau belum.",
    "Tadi mbak kasir di gerai {merch} sempat mengucapkan sesuatu kepada saya saat saya sedang melakukan pemindaian barcode di meja depan, tetapi suaranya tidak terdengar jelas karena suasana toko yang ramai sehingga membuat saya agak bingung.",
    "Ada sedikit hal yang terasa tidak biasa waktu saya mencoba melakukan pembayaran di {loc} {merch} tadi siang, tampilannya agak berbeda dari pengalaman saya biasanya dan saya merasa ragu meskipun tidak tahu apa penyebabnya.",
    "Proses pembayarannya sebenarnya terlihat sudah selesai di layar ponsel saya, cuma saya merasa agak bingung saja dengan alur tampilan konfirmasi yang muncul di toko {merch} dan tidak yakin apakah perlu konfirmasi tambahan.",
    "Saya tadi langsung menutup aplikasi perbankan saya setelah klik konfirmasi bayar di {merch} sehingga tidak sempat membaca pesan notifikasi yang sempat muncul sekilas di layar ponsel saya, apakah transaksi tersebut sudah beres?",
    "Mohon bantuan informasi min terkait transaksi terakhir saya di {merch}, tadi koneksi internet ponsel saya sempat tersendat saat menekan tombol bayar sehingga saya ragu statusnya meskipun tidak ada pesan kesalahan yang muncul.",
    "Saya lupa melihat rincian resi pembayaran di layar ponsel saya setelah scan barcode di {merch} karena terburu-buru mengejar kendaraan, saya hanya ingin memastikan apakah pembayaran saya sudah tercatat aman.",
    "Kemarin belanja di kasir {merch} tapi pas scan barcode HP saya agak nge-lag sebentar dan langsung kembali ke menu home, saya jadi bingung transaksinya sudah masuk atau belum.",
    "Saya kurang mengerti alur pembayaran QRIS di toko {merch} ini, kasirnya sibuk melayani pembeli lain jadi saya tidak sempat bertanya lebih lanjut dan langsung pergi dengan rasa ragu."
]

def build_dataset_14k():
    rows = []
    
    # 1. Generate 1.000 Mirror Contrastive Pairs untuk K1 & K5
    print("[1/3] Menghasilkan 1.000 pasang mirror contrastive (1.000 K5 + 1.000 K1)...")
    count = 0
    while count < 1000:
        store_type, store_name = random.choice(MERCHANT_TYPES)
        tmpl = random.choice(CONTRASTIVE_CONTEXTS)
        op_match, op_mismatch = random.choice(OPERATOR_PAIRS)

        text_k5 = tmpl.format(store_type=store_type, store_name=store_name, op=op_match)
        text_k1 = tmpl.format(store_type=store_type, store_name=store_name, op=op_mismatch)

        rows.append({"text": text_k5, "category": "QRIS_NORMAL_MERCHANT_TERPERCAYA", "severity": "LOW"})
        rows.append({"text": text_k1, "category": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT", "severity": "HIGH"})
        count += 1

    # 2. Lengkapi K1 dan K5 sisa 1.000 sampel masing-masing
    print("[2/3] Melengkapi sisa 1.000 sampel K1 (Diverse) dan 1.000 sampel K5 (Diverse)...")
    for _ in range(1000):
        # K1 Diverse
        tmpl = random.choice(TEMPLATES_IDENTITAS_DIVERSE)
        t = tmpl.format(
            merch=random.choice(MERCHANTS),
            loc=random.choice(LOCATIONS),
            city=random.choice(CITIES),
            app=random.choice(APPS),
            fake_acc=random.choice(FAKE_NAMES)
        )
        rows.append({"text": t, "category": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT", "severity": "HIGH"})

        # K5 Diverse
        tmpl = random.choice(TEMPLATES_NORMAL_DIVERSE)
        t = tmpl.format(
            merch=random.choice(MERCHANTS),
            loc=random.choice(LOCATIONS),
            city=random.choice(CITIES),
            app=random.choice(APPS),
            nom=random.choice(NOMINALS)
        )
        rows.append({"text": t, "category": "QRIS_NORMAL_MERCHANT_TERPERCAYA", "severity": "LOW"})

    # 3. Generate 2.000 sampel untuk K0, K2, K3, K4, K6
    print("[3/3] Menghasilkan 2.000 sampel untuk K0, K2, K3, K4, K6...")
    
    # K0: PENIPUAN_STIKER_QRIS_PALSU (2.000)
    for _ in range(2000):
        tmpl = random.choice(TEMPLATES_STIKER_PALSU)
        t = tmpl.format(merch=random.choice(MERCHANTS), loc=random.choice(LOCATIONS), city=random.choice(CITIES))
        rows.append({"text": t, "category": "PENIPUAN_STIKER_QRIS_PALSU", "severity": "CRITICAL"})

    # K2: PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE (2.000)
    for _ in range(2000):
        tmpl = random.choice(TEMPLATES_SURCHARGE)
        t = tmpl.format(merch=random.choice(MERCHANTS), city=random.choice(CITIES), nom=random.choice(NOMINALS))
        rows.append({"text": t, "category": "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE", "severity": "HIGH"})

    # K3: KONDISI_FISIK_QRIS_RUSAK (2.000)
    for _ in range(2000):
        tmpl = random.choice(TEMPLATES_FISIK_RUSAK)
        t = tmpl.format(merch=random.choice(MERCHANTS), loc=random.choice(LOCATIONS), city=random.choice(CITIES))
        rows.append({"text": t, "category": "KONDISI_FISIK_QRIS_RUSAK", "severity": "MEDIUM"})

    # K4: KUALITAS_LAYANAN_MERCHANT (2.000)
    for _ in range(2000):
        tmpl = random.choice(TEMPLATES_LAYANAN)
        t = tmpl.format(merch=random.choice(MERCHANTS), city=random.choice(CITIES), nom=random.choice(NOMINALS))
        rows.append({"text": t, "category": "KUALITAS_LAYANAN_MERCHANT", "severity": "MEDIUM"})

    # K6: FEEDBACK_AMBIGU (2.000 total: 1.000 Narasi Panjang + 1.000 Pendek/Sedang)
    # 1.000 Narasi Panjang (vaksinasi agar curhatan ambigu bertele-tele tidak salah lari ke K1 atau K4)
    for _ in range(1000):
        tmpl = random.choice(LONG_TEMPLATES_AMBIGU)
        t = tmpl.format(
            merch=random.choice(MERCHANTS),
            city=random.choice(CITIES),
            loc=random.choice(LOCATIONS)
        )
        rows.append({"text": t, "category": "FEEDBACK_AMBIGU", "severity": "MEDIUM"})

    # 1.000 Kalimat Ambigu Pendek / Sedang
    prefixes = ["Halo min, ", "Permisi, ", "Mau tanya dong, ", "Lapor min, ", "Bro, ", "Kak, ", "Min, ", "", "", ""]
    suffixes = [
        " tolong ditanggapi.", " gimana ya?", " mohon infonya.", " tolong dicek.",
        " bingung banget.", " ada tanggapan?", " tolong respon.", " terima kasih.", "", ""
    ]
    for _ in range(1000):
        core = random.choice(TEMPLATES_AMBIGU)
        t = f"{random.choice(prefixes)}{core}{random.choice(suffixes)}".strip()
        rows.append({"text": t, "category": "FEEDBACK_AMBIGU", "severity": "MEDIUM"})

    # Acak urutan baris
    random.shuffle(rows)
    return rows

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "qris_feedback_14k.csv")

    print("=" * 80)
    print("MEMULAI GENERASI DATASET 14.000 BARIS LAQRIS EMRS (VERSION 5.0)")
    print("7 Kelas Seimbang: Masing-masing Tepat 2.000 Sampel")
    print("=" * 80)

    rows = build_dataset_14k()
    df = pd.DataFrame(rows)

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[SUKSES] Dataset 14.000 baris berhasil disimpan di:\n  -> {output_path}")
    print(f"Total baris: {len(df)}")
    print("\nDistribusi Per Kategori:")
    print(df["category"].value_counts())

if __name__ == "__main__":
    main()
