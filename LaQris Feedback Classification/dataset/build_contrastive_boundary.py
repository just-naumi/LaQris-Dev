"""
Generator Dataset Hard-Boundary Contrastive Pairs (K1 vs K5) untuk LaQris EMRS
Membuat pasangan kalimat kontras langsung (Mirror Pairs) antara:
- K1: KETIDAKSESUAIAN_IDENTITAS_MERCHANT (Mismatch / !=)
- K5: QRIS_NORMAL_MERCHANT_TERPERCAYA (Match / ==)

Tujuan:
Mematahkan lexical bias pada IndoBERT sehingga model tidak lagi mengasosiasikan
vocabulary identitas ("nama merchant", "nama penerima", "toko", "NMID") ke K1,
melainkan membaca operator kesesuaian (relasi afirmatif vs relasi negasi/mismatch).
"""

import os
import sys
import random
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

random.seed(42)

# Komponen Pembentuk Pasangan Kalimat Kontras
MERCHANT_TYPES = [
    ("warung makan", "Warung Bu Siti", "kuliner"),
    ("kedai kopi", "Kopi Kenangan Senja", "cafe"),
    ("toko kelontong", "Toko Berkah Jaya", "sembako"),
    ("apotek", "Apotek Sehat Medika", "farmasi"),
    ("minimarket", "Minimarket Barokah", "ritel"),
    ("bengkel motor", "Bengkel Maju Lancar", "otomotif"),
    ("laundry", "Laundry Wangi Bersih", "jasa"),
    ("counter pulsa", "Counter Cahaya Cell", "gadget"),
    ("restoran padang", "Rumah Makan Sederhana Minang", "kuliner"),
    ("stand boba", "Boba Fresh Delight", "minuman"),
    ("toko buah", "Toko Buah Segar Mandiri", "buah"),
    ("kedai bakso", "Bakso Super Urat Mas Bejo", "kuliner"),
    ("toko pakaian", "Fashion Style Boutique", "fashion"),
    ("barbershop", "Ganteng Barbershop", "jasa"),
    ("toko alat tulis", "Toko Buku Gemilang", "retail")
]

# Pasangan Operator Kontras (Match vs Mismatch)
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

# Konteks Pembuka / Kronologi Transaksi
CONTEXT_TEMPLATES = [
    # Tipe 1: Pre-payment check
    (
        "Sebelum menekan tombol konfirmasi pembayaran QRIS di {store_type}, saya memeriksa nama penerima di layar aplikasi. Hasil pengecekan menunjukkan nama penerima {op}, dan transaksi berhasil diproses.",
    ),
    # Tipe 2: Post-payment check
    (
        "Tadi saya bayar pakai QRIS di {store_name}. Pembayaran sukses, lalu saat mengecek rincian transaksi di aplikasi, nama merchant yang tercantum {op}.",
    ),
    # Tipe 3: Verifikasi kasir & aplikasi
    (
        "Kasir di {store_name} mengarahkan untuk scan barcode QRIS. Saya mencocokkan nama usaha pada aplikasi pembayaran dengan nama toko, dan hasilnya {op}.",
    ),
    # Tipe 4: Multi-atribut (NMID & Nama)
    (
        "Saya memeriksa detail informasi QRIS termasuk NMID dan nama merchant sebelum transfer. Informasi yang tampil di layar HP {op}.",
    ),
    # Tipe 5: Pengalaman langsung di kasir
    (
        "Saat melakukan transaksi QRIS di meja kasir {store_type}, saya teliti melihat nama penerima transfer pada e-wallet. Ternyata nama merchant tersebut {op}.",
    ),
    # Tipe 6: Kalimat ringkas & lugas
    (
        "Nama merchant yang muncul di aplikasi setelah scan QRIS {op}.",
    ),
    # Tipe 7: Verifikasi resi / notifikasi
    (
        "Setelah transaksi QRIS berhasil, saya cek notifikasi dan struk digital di perbankan saya. Nama penerima dana {op}.",
    ),
    # Tipe 8: Skeptis di awal tapi diteliti
    (
        "Demi keamanan sebelum bayar QRIS, saya memastikan identitas penerima terlebih dahulu. Nama yang terdaftar pada sistem {op}.",
    ),
    # Tipe 9: Kasus nominal + identitas
    (
        "Nominal belanja sudah pas dan saya cek nama merchant di aplikasi pembayaran. Nama penerima {op}, dan pembayaran selesai lancar.",
    ),
    # Tipe 10: Belanja harian santai
    (
        "Beli barang di {store_name} bayar non-tunai via QRIS. Pas dicek di konfirmasi transaksi, nama penerimanya {op}."
    )
]

def generate_contrastive_pairs(num_pairs=1200):
    pairs = []
    
    # Loop hingga mencapai target pasang
    count = 0
    while count < num_pairs:
        store_type, store_name, category = random.choice(MERCHANT_TYPES)
        tmpl_tuple = random.choice(CONTEXT_TEMPLATES)
        tmpl = tmpl_tuple[0]
        op_match, op_mismatch = random.choice(OPERATOR_PAIRS)

        # Buat Kalimat K5 (Match)
        text_k5 = tmpl.format(
            store_type=store_type,
            store_name=store_name,
            op=op_match
        )

        # Buat Kalimat K1 (Mismatch)
        text_k1 = tmpl.format(
            store_type=store_type,
            store_name=store_name,
            op=op_mismatch
        )

        pairs.append({
            "pair_id": count + 1,
            "text": text_k5,
            "category": "QRIS_NORMAL_MERCHANT_TERPERCAYA",
            "relation_type": "IDENTITY_MATCH"
        })
        pairs.append({
            "pair_id": count + 1,
            "text": text_k1,
            "category": "KETIDAKSESUAIAN_IDENTITAS_MERCHANT",
            "relation_type": "IDENTITY_MISMATCH"
        })

        count += 1

    return pairs

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(base_dir, "k1_k5_contrastive_boundary.csv")

    num_target_pairs = 1200
    print("=" * 80)
    print("BUILDER HARD-BOUNDARY CONTRASTIVE PAIRS LAQRIS (K1 vs K5)")
    print(f"Target Pasang   : {num_target_pairs} pasang mirror ({num_target_pairs * 2} total baris)")
    print("=" * 80)

    data = generate_contrastive_pairs(num_target_pairs)
    df = pd.DataFrame(data)

    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"[SUKSES] File dataset kontras berhasil disimpan ke: {output_csv}")
    print(f"Total baris: {len(df)}")
    print("\nDistribusi Kategori:")
    print(df["category"].value_counts())

if __name__ == "__main__":
    main()
