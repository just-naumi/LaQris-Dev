import csv
import random
import os

random.seed(42)

# ── Augmentation helpers ──────────────────────────────────────────────────────

COLLOQUIAL = {
    "tidak": ["gak", "enggak", "nggak", "ga", "tidak"],
    "saya":  ["saya", "aku", "gue", "gw"],
    "dengan": ["dengan", "sama", "dgn"],
    "sudah": ["sudah", "udah", "dah"],
    "sangat": ["sangat", "banget", "amat"],
    "saat":  ["saat", "waktu", "ketika", "pas"],
}
INTROS = ["Aduh", "Wah", "Eh", "Loh", "Hmm", "Oh iya", "", "", "", ""]

def aug(text):
    for formal, opts in COLLOQUIAL.items():
        if formal in text and random.random() < 0.4:
            text = text.replace(formal, random.choice(opts), 1)
    intro = random.choice(INTROS)
    return ((intro + " " + text) if intro else text).strip()

def clean(text):
    return aug(text).replace("\n", " ").replace("\r", " ").strip()

# ── Per-class generators ──────────────────────────────────────────────────────

def gen_identity_mismatch(n):
    subjects = ["Nama penerima","Nama rekening tujuan","Nama merchant","Nama pemilik rekening",
                "Nama di layar pembayaran","Nama yang muncul saat scan","Nama akun tujuan","Identitas merchant"]
    predicates = ["tidak sesuai dengan nama toko","beda sama nama yang ada di plang",
                  "berbeda dengan nama toko yang tertera","gak cocok sama nama merchant resmi",
                  "tidak cocok dengan stiker toko","jauh berbeda dari nama di spanduk",
                  "beda banget dari nama tokonya","tidak sama dengan nama usaha",
                  "asing dan tidak dikenal","bukan nama tokonya sama sekali","nama orang bukan nama toko"]
    suffixes = [""," jadi saya ragu mau bayar"," apa ini aman?"," saya khawatir",
                " tolong dicek"," bikin curiga"," saya batalin aja transaksinya",
                " langsung saya laporin"," apa mungkin QRIS-nya dipalsukan?"," ini fraud ya?"]
    templates = [
        lambda s,p,x: f"{s} {p}{x}.",
        lambda s,p,x: f"Pas scan QRIS-nya {s} {p}{x}.",
        lambda s,p,x: f"Setelah scan {s} {p}{x}.",
        lambda s,p,x: f"Aneh banget {s} {p}{x}.",
        lambda s,p,x: f"Lho kok {s} {p}{x}?",
        lambda s,p,x: f"Kenapa ya {s} {p}{x}?",
        lambda s,p,x: f"Saya perhatikan {s} {p}{x}.",
    ]
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        rows.append(clean(t(random.choice(subjects), random.choice(predicates), random.choice(suffixes))))
    return rows

def gen_qr_replacement(n):
    objects = ["stiker QR palsu","stiker QRIS baru","QR code palsu","kode QR yang mencurigakan",
               "stiker QRIS asing","QR lain yang ditempel"]
    verbs = ["ditempel di atas QR asli","menutup QR code resmi","dipasang di atas QRIS toko",
             "ditimpa di kode QR yang sah","menimpa QRIS original","menutupi QR code merchantnya",
             "nempel di atas akrilik QRIS"]
    contexts = ["di kasir toko","di meja kasir","di depan toko","di akrilik stan","di tempat pembayaran"]
    suffixes = [""," saya curiga ini penipuan"," ini pasti fraud"," saya gak jadi bayar",
                " langsung saya laporkan"," saya foto dulu sebagai bukti"," tolong ditindaklanjuti"]
    templates = [
        lambda o,v,c,x: f"Saya lihat ada {o} yang {v} {c}{x}.",
        lambda o,v,c,x: f"Kelihatan ada {o} yang {v}{x}.",
        lambda o,v,c,x: f"Kayaknya ada {o} yang {v} {c}{x}.",
        lambda o,v,c,x: f"Waktu mau bayar saya notice {o} {v} {c}{x}.",
        lambda o,v,c,x: f"QRIS-nya tidak asli ada {o} yang {v}{x}.",
        lambda o,v,c,x: f"Ada {o} mencurigakan yang {v} {c}{x}.",
    ]
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        rows.append(clean(t(random.choice(objects), random.choice(verbs), random.choice(contexts), random.choice(suffixes))))
    return rows

def gen_additional_fees(n):
    actors = ["Kasir","Penjual","Pemilik toko","Petugasnya","Mbaknya","Masnya","Si penjualnya"]
    amounts = ["Rp500","Rp1.000","Rp1.500","Rp2.000","Rp2.500","Rp3.000","Rp5.000",
               "500 perak","seribu rupiah","dua ribu rupiah"]
    reasons = ["biaya admin QRIS","biaya layanan","biaya transaksi","ongkos pembayaran digital",
               "biaya sistem","biaya penggunaan QRIS","tanpa alasan jelas"]
    suffixes = [" padahal seharusnya gratis"," ini jelas melanggar aturan"," ini tidak boleh",
                " saya komplain tapi diabaikan"," apakah ini diperbolehkan?"," saya merasa dirugikan",
                " BI melarang ini",""," tanpa pemberitahuan sebelumnya"]
    templates = [
        lambda a,m,r,x: f"{a} minta {m} sebagai {r}{x}.",
        lambda a,m,r,x: f"Diminta bayar tambahan {m} untuk {r}{x}.",
        lambda a,m,r,x: f"Ada pungutan {m} sebagai {r}{x}.",
        lambda a,m,r,x: f"Saya dikenai biaya ekstra {m} atas nama {r}{x}.",
        lambda a,m,r,x: f"Tiba-tiba ada tambahan {m} katanya {r}{x}.",
        lambda a,m,r,x: f"Ditagih {m} lagi dengan alasan {r}{x}.",
    ]
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        rows.append(clean(t(random.choice(actors), random.choice(amounts), random.choice(reasons), random.choice(suffixes))))
    return rows

def gen_suspicious_transaction(n):
    issues = ["Transaksi QRIS gagal terus","Pembayaran tidak berhasil","QRIS tidak bisa dipindai",
              "Transaksi error berkali-kali","QR terbaca tapi transaksi gagal",
              "Sistem QRIS mengalami gangguan mencurigakan"]
    effects = ["tapi saldo sudah terpotong","namun saldo berkurang","padahal uang sudah keluar",
               "tapi rekening sudah didebet","walaupun tidak ada notifikasi sukses","saldo terpotong ganda"]
    contexts = ["saat saya coba bayar","ketika transaksi berlangsung","pas saya scan QR-nya",
                "waktu pembayaran diproses",""]
    suffixes = [" saya bingung harus lapor kemana"," tolong segera ditangani",
                " ini sangat merugikan"," saya minta refund"," dana saya hilang",
                " mohon bantuan",""," ini sudah kedua kalinya"]
    templates = [
        lambda i,e,c,x: f"{i} {c} {e}{x}.".replace("  ", " "),
        lambda i,e,c,x: f"Aneh banget {i} {c} {e}{x}.".replace("  ", " "),
        lambda i,e,c,x: f"Sudah coba berkali-kali tapi {i} {e}{x}.",
        lambda i,e,c,x: f"{i} dan {e} {c}{x}.".replace("  ", " "),
    ]
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        rows.append(clean(t(random.choice(issues), random.choice(effects), random.choice(contexts), random.choice(suffixes))))
    return rows

def gen_safe_confirmation(n):
    outcomes = ["Transaksi berhasil dan lancar","Pembayaran sukses tanpa masalah",
                "QRIS berjalan normal","Proses pembayaran mulus","Transaksi selesai dengan aman",
                "Semua berjalan baik","Pembayaran QRIS sukses"]
    confirmations = ["nama merchant sesuai dengan nama toko","nama penerima cocok dengan tokonya",
                     "tidak ada biaya tambahan","QRIS terlihat asli dan resmi","QR code tampak original",
                     "nama rekening tujuan sesuai","merchant terverifikasi","logo QRIS jelas dan tidak rusak"]
    details = ["Aman digunakan.","Recommended deh tokonya.","Tidak ada kendala sama sekali.",
               "Puas dengan layanannya.","Sangat mudah dan cepat.","Terpercaya tokonya.",""]
    templates = [
        lambda o,c,d: f"{o}, {c}. {d}",
        lambda o,c,d: f"{c} dan {o}. {d}",
        lambda o,c,d: f"Alhamdulillah {o}. {c}. {d}",
        lambda o,c,d: f"Syukurlah {o}, {c}. {d}",
        lambda o,c,d: f"Berhasil bayar QRIS, {c}. {o}. {d}",
    ]
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        rows.append(clean(t(random.choice(outcomes), random.choice(confirmations), random.choice(details))))
    return rows

def gen_other(n):
    topics = ["Pelayanan kasirnya kurang ramah","Antrian di kasir sangat panjang",
              "Produknya bagus tapi packaging kurang menarik","Tempat parkir susah sekali",
              "AC di dalam toko terlalu dingin","Toko terlalu ramai dan sempit",
              "Barang sering kehabisan stok","Harganya mahal dibanding toko lain",
              "Tempatnya bersih dan nyaman","Tidak ada wifi di toko","Jam buka toko tidak konsisten",
              "Kualitas produknya bagus sekali","Toko agak kotor di bagian belakang",
              "Pencahayaan toko kurang terang","Harga dan kualitas sebanding","Lokasi toko strategis banget"]
    suffixes = [""," perlu diperbaiki"," semoga bisa lebih baik"," tapi overall oke kok",
                " secara keseluruhan lumayan"," tidak masalah sih"]
    templates = [
        lambda t,x: f"{t}{x}.",
        lambda t,x: f"Saya mau kasih masukan: {t}{x}.",
        lambda t,x: f"Feedback untuk toko: {t}{x}.",
        lambda t,x: f"Pengalaman saya: {t}{x}.",
        lambda t,x: f"Catatan: {t}{x}.",
    ]
    rows = []
    for _ in range(n):
        tmpl = random.choice(templates)
        rows.append(clean(tmpl(random.choice(topics), random.choice(suffixes))))
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    N = 1000
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

    generators = {
        "identity_mismatch":      gen_identity_mismatch,
        "qr_replacement":         gen_qr_replacement,
        "additional_fees":        gen_additional_fees,
        "suspicious_transaction": gen_suspicious_transaction,
        "safe_confirmation":      gen_safe_confirmation,
        "other":                  gen_other,
    }

    rows = []
    for label, fn in generators.items():
        samples = fn(N)
        for text in samples:
            rows.append({"text": text, "label": label})
        print(f"  [OK] {label}: {N} samples")

    random.shuffle(rows)

    with open(dataset_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writerows(rows)

    print(f"\nDone! {len(rows)} baris di-append ke {dataset_path}")

if __name__ == "__main__":
    main()
