"""
Generator Dataset 10.000 Hard Stress Test LaQris - Versi Narasi Panjang EMRS (Primary Intent)
Setiap sampel dirancang berupa kalimat/paragraf narasi panjang (25 - 50 kata) multi-klausa
yang secara ketat mematuhi aturan hierarki niat utama (Primary Intent) LaQris.

Hierarki Niat Utama:
1. K0: PENIPUAN_STIKER_QRIS_PALSU (Ada bukti fisik stiker barcode ditimpa / dimanipulasi)
2. K1: KETIDAKSESUAIAN_IDENTITAS_MERCHANT (Barcode fisik bersih toko, nama digital beda murni)
3. K2: PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE (Biaya admin / surcharge / kenaikan harga belanjaan)
4. K3: KONDISI_FISIK_QRIS_RUSAK (Stiker sobek, pudar matahari, kotor minyak gorengan)
5. K4: KUALITAS_LAYANAN_MERCHANT (Penolakan QRIS, minimal belanja, kasir judes - steril dari fee)
6. K5: QRIS_NORMAL_MERCHANT_TERPERCAYA (Transaksi sukses, struk resmi, kasir ramah, nominal pas)
7. K6: FEEDBACK_AMBIGU (Nol bukti teknis, murni perasaan ragu / terburu-buru)
"""

import os
import random
import csv

random.seed(2026)

CATEGORIES = [
    "PENIPUAN_STIKER_QRIS_PALSU",
    "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
    "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE",
    "KONDISI_FISIK_QRIS_RUSAK",
    "KUALITAS_LAYANAN_MERCHANT",
    "QRIS_NORMAL_MERCHANT_TERPERCAYA",
    "FEEDBACK_AMBIGU"
]

LOCATIONS = [
    "meja kasir", "akrilik meja", "etalase kaca kasir", "pintu masuk toko",
    "kotak amal", "plang barcode kasir", "counter pembayaran", "meja kasir utama",
    "stand pembayaran", "booth makanan", "pos kasir depan"
]

MERCHANTS = [
    "Kopi Kenangan Senja", "Warung Nasi Bu Imas", "Apotek Sehat 24 Jam", "RM Padang Saiyo",
    "Bakso Pak Granat", "Martabak Bangka Asli", "Soto Lamongan Cak Har", "Laundry Express 88",
    "Bengkel Mobil Prima", "Minimarket Barokah", "Kedai Mie Aceh", "Es Teh Nusantara",
    "Barbershop Mas Bro", "Toko Sembako Berkah", "Klinik Pratama Sehat", "Warteg Kharisma Bahari"
]

CITIES = ["Jakarta Selatan", "Kota Bandung", "Surabaya Pusat", "Medan Kota", "Yogyakarta", "Semarang", "Malang Kota", "Bekasi Barat"]
APPS = ["wondr by BNI", "BCA Mobile", "Livin by Mandiri", "BRImo", "DANA", "GoPay", "OVO", "ShopeePay"]

FAKE_ACCOUNTS = [
    "rekening pribadi Budi Santoso", "akun personal E-Wallet 0812988xxx", "GOPAY TOPUP ABAL-ABAL",
    "Yayasan Amal Kasih Fiktif", "CV Berkah Jaya Abadi di luar pulau", "rekening perorangan Dewi Anggraeni",
    "akun deposit game online", "rekening penampung PT Angin Ribut", "rekening pribadi kasir yang tidak dikenal"
]

NOMINALS = [
    ("15.000", "150.000"), ("25.000", "250.000"), ("35.000", "350.000"),
    ("50.000", "500.000"), ("75.000", "750.000"), ("100.000", "1.000.000")
]

# ── TEMPLATE NARASI PANJANG DENGAN BOUNDARY TEGAS ─────────────────────────────

LONG_TEMPLATES_STIKER_PALSU = [
    "Saat saya hendak melakukan pembayaran di {loc} {merch}, saya melihat dengan jelas bahwa barcode QRIS fisik di akrilik meja telah ditimpa oleh stiker tempelan palsu baru yang menutupi barcode resmi toko, sehingga saya langsung membatalkan transaksi karena ini merupakan modus pemalsuan stiker barcode fisik.",
    "Ketika saya memeriksa kode QRIS yang terpajang pada {loc} di {merch} {city}, saya menemukan adanya lapisan stiker ganda bertumpuk di mana stiker bagian atas sengaja ditempel orang luar untuk mengalihkan pembayaran ke {fake_acc}, mohon pihak LaQris segera menindaklanjuti temuan stiker palsu ini.",
    "Tadi waktu berbelanja di {merch}, kasir toko sempat kaget saat saya perlihatkan bahwa akrilik QRIS di mejanya sudah ditempeli stiker barcode lain oleh oknum tidak bertanggung jawab, sehingga pihak toko langsung mengamankan akrilik tersebut agar pelanggan tidak menjadi korban stiker tempel palsu.",
    "Saya curiga melihat kondisi fisik stiker barcode di {loc} {merch} yang tampak memiliki dua lapisan stiker di mana barcode aslinya ditutup rapat oleh stiker cetakan baru beratas nama {fake_acc}, sehingga saya tidak berani scan demi keamanan saldo perbankan saya.",
    "Ada indikasi kuat pemalsuan barcode fisik di gerai {merch} {city} karena stiker QRIS resmi di meja kasir telah ditimpa oleh stiker barcode palsu buatan penipu, sehingga siapapun yang memindai akrilik tersebut dananya akan dialihkan ke rekening pelaku kejahatan.",
    "Setelah saya amati lebih dekat saat hendak scan di {loc} {merch}, ternyata stiker barcode yang terpasang di akrilik bukan stiker bawaan toko melainkan stiker tempelan penipu yang sengaja dipasang menutupi kode QR kasir, sehingga ini jelas tindakan kriminal pemalsuan QRIS fisik.",
    "Waspada modus stiker palsu di {merch} karena barcode statis yang dipajang di etalase kasir ternyata sudah dilapisi stiker cetakan lain berukuran sama yang menutupi kode QR aslinya, terindikasi kuat kejahatan manipulasi stiker barcode toko."
]

LONG_TEMPLATES_IDENTITAS_BEDA = [
    "Kondisi fisik stiker barcode di meja kasir {merch} tampak bersih dan asli bawaan toko, namun saat saya melakukan scan nama merchant yang muncul pada detail transaksi justru tidak sesuai dengan nama toko fisik tempat saya berbelanja, sehingga saya khawatir ada ketidaksesuaian identitas akun QRIS.",
    "Saya sebenarnya berhasil melakukan transaksi pembayaran menggunakan QRIS di {merch} {city} pada barcode resmi kasir, tetapi setelah proses selesai saya baru melihat bahwa nama penerima yang tertera di aplikasi {app} adalah {fake_acc} dan bukan nama badan usaha toko yang sah.",
    "Plang nama toko dan spanduk di lokasi jelas-jelas bertuliskan {merch}, namun saat saya melakukan scan barcode statis resmi toko di meja kasir nama yang terverifikasi di layar ponsel saya justru {fake_acc}, sehingga ada ketidaksesuaian identitas yang mencurigakan antara nama fisik toko dan nama digital merchant.",
    "Ketika saya memeriksa kembali lembar bukti transfer digital setelah belanja di {merch}, tertera bahwa dana saya masuk ke rekening {fake_acc} yang terdaftar di kota lain padahal saya berbelanja di gerai resmi mereka di {city}, sehingga identitas merchant QRIS toko ini tidak cocok.",
    "Kasir di gerai {merch} sempat bingung ketika saya perlihatkan bukti bayar sukses karena nama tujuan transfer yang muncul di aplikasi saya adalah {fake_acc} dan bukan atas nama toko, sehingga identitas pemilik akun QRIS tersebut berbeda dengan identitas fisik toko ini.",
    "Setelah transaksi pembayaran belanjaan saya di {merch} terverifikasi selesai oleh bank, saya baru tersadar bahwa nama merchant yang tercantum pada resi digital sama sekali tidak ada hubungannya dengan nama toko ini melainkan atas nama {fake_acc}, mohon LaQris verifikasi kesesuaian identitasnya.",
    "Saya tadi scan kode QRIS resmi yang tersedia di meja kasir {merch}, namun setelah pembayaran berhasil saya melihat detail penerima digitalnya bernama {fake_acc} yang tidak dikenal toko, sehingga identitas merchant yang terdaftar di sistem tidak sinkron dengan nama toko."
]

LONG_TEMPLATES_SURCHARGE = [
    "Saat saya hendak membayar belanjaan di {merch}, kasir secara sepihak meminta saya menambahkan biaya ekstra sebesar Rp1.000 di luar total harga pada struk dengan alasan biaya admin transaksi non tunai QRIS, padahal menurut regulasi Bank Indonesia merchant dilarang keras mengenakan biaya surcharge kepada konsumen QRIS.",
    "Pihak pengelola toko {merch} di {city} mengenakan biaya surcharge tambahan sebesar Rp2.000 khusus untuk setiap transaksi yang menggunakan barcode QRIS statis, sehingga total yang harus saya bayar membengkak dari harga barang aslinya dan pungutan liar ini sangat merugikan pengguna QRIS.",
    "Kasir di gerai {merch} menaikkan harga belanjaan jika pembeli ingin menggunakan pembayaran QRIS statis, kasir berdalih bahwa biaya MDR potongan bank harus ditanggung sendiri oleh pembeli sehingga tindakan ini melanggar aturan kepatuhan merchant.",
    "Saya merasa kecewa karena kasir toko {merch} menambahkan biaya admin siluman sebesar seribu rupiah di nota saat saya scan QRIS, padahal regulasi resmi melarang merchant membebankan biaya tambahan kepada pembeli non tunai.",
    "Pungutan biaya surcharge Rp2.000 oleh kasir saat transaksi scan QRIS di {loc} {merch} sangat merugikan konsumen, mohon pihak LaQris memberikan teguran penalti kepatuhan regulasi pada merchant yang memungut fee tambahan ini.",
    "Kasir {merch} menolak mentransaksikan harga normal jika bayar lewat QRIS dan mewajibkan pembeli menambah nominal biaya ekstra seribu rupiah, sehingga pengenaan biaya tambahan sepihak ini melanggar ketentuan QRIS nasional."
]

LONG_TEMPLATES_RUSAK_PUDAR = [
    "Kondisi fisik stiker barcode QRIS yang terpasang pada {loc} di {merch} sudah mengalami sobek parah pada bagian pola tengah dan sudut pembacaan, sehingga kamera pemindai aplikasi perbankan saya berulang kali gagal mendeteksi kode tersebut dan transaksi tidak dapat diproses sama sekali.",
    "Stiker QRIS statis di etalase kasir {merch} {city} tampak sangat kusam dan pudar akibat terpapar panas matahari secara terus-menerus, sehingga kontras pola hitam putihnya hilang dan pelanggan harus mengantre panjang karena kamera smartphone tidak mampu membaca barcode.",
    "Lembaran akrilik pelindung kode QR di meja kasir {merch} dipenuhi banyak goresan lecet yang sangat dalam serta terkena tumpahan minyak dan noda kotoran, sehingga scanner aplikasi {app} saya selalu memunculkan pesan error gagal memindai gambar kode.",
    "Saya kesulitan melakukan pembayaran nontunai di {merch} karena stiker QRIS fisik yang ditempel di dinding kasir sudah terkelupas lebih dari separuh permukaannya, sehingga bagian data matriks barcode hilang dan sama sekali tidak dapat terbaca oleh sistem.",
    "Resolusi cetakan stiker barcode QRIS di gerai {merch} terlihat sangat buram dan pecah-pecah sehingga lensa kamera ponsel kesulitan menangkap fokus, akibatnya proses verifikasi scan selalu gagal meskipun sudah dicoba berkali-kali dari berbagai sudut.",
    "Stiker barcode di meja {merch} basah terkena cipratan air dan tintanya luntur menyebar, sehingga pola deteksi QRIS rusak total dan para pembeli terpaksa harus mencari uang tunai karena pembayaran digital tidak berfungsi."
]

LONG_TEMPLATES_LAYANAN_MERCHANT = [
    "Pelayanan kasir di {merch} sangat mengecewakan, kasir menolak pembayaran QRIS saya secara sepihak dengan alasan minimal belanja harus di atas Rp30.000 padahal aturan resmi perbankan melarang pembatasan minimal transaksi QRIS.",
    "Kasir di toko {merch} bersikap sangat judes dan ketus ketika saya bilang mau bayar pakai QRIS, kasir terang-terangan menolak melayani dan memaksa pembeli harus membayar menggunakan uang pas tunai saja.",
    "Proses pembayaran di {loc} {merch} sangat dipersulit oleh staf kasir yang beralasan barcode QRIS sedang disimpan di laci dan malas mengeluarkan akrilik kasir, sehingga pembeli dipaksa antre lama tanpa kejelasan pelayanan.",
    "Kasir di gerai {merch} menahan barang belanjaan saya dengan sikap mencurigai pembeli secara tidak sopan padahal notifikasi bukti pembayaran QRIS sukses di aplikasi {app} sudah saya perlihatkan dengan jelas di depan matanya.",
    "Merchant {merch} di {city} secara sepihak menolak menerima pembayaran QRIS dari aplikasi bank tertentu dengan alasan kasir tidak paham, sehingga sikap penolakan sepihak ini mencerminkan buruknya kualitas layanan toko.",
    "Staf kasir di {merch} melayani pelanggan dengan wajah masam dan memarahi pembeli yang ingin scan QRIS karena antrean sedang ramai, sangat tidak ramah dan merusak kenyamanan berbelanja."
]

LONG_TEMPLATES_NORMAL_SUKSES = [
    "Saya membeli makanan di warung makan {merch} dan membayar menggunakan QRIS seperti biasanya, di mana nominal yang harus dibayar pas sesuai dengan harga makanan, transaksi berhasil dalam beberapa detik, dan bukti pembayaran resmi juga langsung muncul seketika di aplikasi perbankan saya.",
    "Kasir yang melayani saya di {merch} sangat ramah dan segera menunjukkan barcode di meja kasir, nominal pembayaran saya input pas sebesar Rp{nom1} tanpa biaya admin, dan seketika itu juga notifikasi uang masuk langsung berbunyi di soundbox kasir toko.",
    "Saya memeriksa kembali mutasi rekening setelah transaksi di {merch} selesai dan nominal yang terdebet sangat cocok dengan yang saya ketik tadi siang, tidak ada potongan tersembunyi apapun dan seluruh proses pembayaran berjalan aman, cepat, dan terverifikasi hijau.",
    "Proses transaksi pembayaran nontunai berlangsung sangat lancar tanpa kendala apapun di {loc} {merch}, lembar struk dan bukti transfer digital tersimpan rapi pada riwayat aplikasi {app}, serta pihak penjual langsung menyerahkan pesanan saya dengan ramah.",
    "Tadi saya melakukan pembayaran menggunakan QRIS statis di {loc} {merch} {city}, nama penerima yang tertera pada layar konfirmasi bayar sama persis dengan nama toko pada plang kasir, dan transaksi langsung dinyatakan sukses dalam hitungan detik.",
    "Nominal belanjaan sebesar Rp{nom1} saya bayar secara tepat dengan scan barcode di kasir {merch}, semua detail informasi pembayaran terbukti akurat dan kasir langsung menyerahkan nota lunas beserta barang belanjaan tanpa ada biaya tambahan apapun.",
    "Tidak ada masalah sama sekali saat saya berbelanja di {merch}, barcode QRIS bersih terlindungi akrilik, verifikasi pembayaran berjalan sangat mulus, dan bukti bayar resmi langsung keluar di aplikasi ponsel saya tanpa kendala jaringan.",
    "Uang masuk seketika ke rekening mutasi pemilik toko {merch}, kasir langsung mengkonfirmasi bahwa tagihan sudah lunas dan saya menerima barang belanjaan saya dengan perasaan tenang dan puas."
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
    "Saya lupa melihat rincian resi pembayaran di layar ponsel saya setelah scan barcode di {merch} karena terburu-buru mengejar kendaraan, saya hanya ingin memastikan apakah pembayaran saya sudah tercatat aman."
]

def generate_long_sample(category):
    loc = random.choice(LOCATIONS)
    merch = random.choice(MERCHANTS)
    city = random.choice(CITIES)
    app = random.choice(APPS)
    fake_acc = random.choice(FAKE_ACCOUNTS)
    nom1, nom2 = random.choice(NOMINALS)

    if category == "PENIPUAN_STIKER_QRIS_PALSU":
        tmpl = random.choice(LONG_TEMPLATES_STIKER_PALSU)
    elif category == "KETIDAKSESUAIAN_IDENTITAS_MERCHANT":
        tmpl = random.choice(LONG_TEMPLATES_IDENTITAS_BEDA)
    elif category == "PUNGUTAN_BIAYA_TAMBAHAN_SURCHARGE":
        tmpl = random.choice(LONG_TEMPLATES_SURCHARGE)
    elif category == "KONDISI_FISIK_QRIS_RUSAK":
        tmpl = random.choice(LONG_TEMPLATES_RUSAK_PUDAR)
    elif category == "KUALITAS_LAYANAN_MERCHANT":
        tmpl = random.choice(LONG_TEMPLATES_LAYANAN_MERCHANT)
    elif category == "FEEDBACK_AMBIGU":
        tmpl = random.choice(LONG_TEMPLATES_AMBIGU)
    else:
        tmpl = random.choice(LONG_TEMPLATES_NORMAL_SUKSES)

    text = tmpl.format(
        loc=loc, merch=merch, city=city, app=app,
        fake_acc=fake_acc, nom1=nom1, nom2=nom2
    )

    if random.random() < 0.20:
        opener = random.choice([
            "Sebagai catatan pengalaman saya, ",
            "Sekadar laporan dari saya selaku pembeli, ",
            "Mohon perhatian pihak customer service, ",
            "Saya ingin menyampaikan feedback bahwa "
        ])
        text = f"{opener}{text[0].lower()}{text[1:]}"

    return text

def main():
    target_total = 10000
    per_cat = target_total // len(CATEGORIES)
    remainder = target_total % len(CATEGORIES)

    print("=" * 80)
    print("GENERATOR DATASET HARD STRESS TEST LAQRIS - PRIMARY INTENT HIERARCHY")
    print(f"Total Sampel Target: {target_total} baris (7 Kelas EMRS Seimbang)")
    print("=" * 80)

    rows = []
    current_id = 1

    for idx, cat in enumerate(CATEGORIES):
        count = per_cat + (1 if idx < remainder else 0)
        print(f"  [GEN] Membuat {count} sampel narasi panjang untuk: {cat}")
        for _ in range(count):
            txt = generate_long_sample(cat)
            rows.append({
                "test_id": f"STRESS-{current_id:05d}",
                "text": txt,
                "expected_category": cat
            })
            current_id += 1

    random.shuffle(rows)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    output_path = os.path.join(root_dir, "dataset", "qris_stress_test_10k.csv")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test_id", "text", "expected_category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SUKSES] Dataset 10.000 Hard Stress Test (Primary Intent) berhasil disimpan di:")
    print(f"  -> {output_path}")
    print(f"Total baris: {len(rows)}")

if __name__ == "__main__":
    main()
