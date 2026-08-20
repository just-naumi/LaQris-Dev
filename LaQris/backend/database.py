import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Path ke file database SQLite
FOLDER_BACKEND = os.path.dirname(os.path.abspath(__file__))
PATH_SQLITE_DB = os.path.join(FOLDER_BACKEND, "database.sqlite")

SQLALCHEMY_DATABASE_URL = f"sqlite:///{PATH_SQLITE_DB}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Inisialisasi tabel SQLite dan auto-seed data reputasi merchant
    dengan skema EMRS (Evidence-Based Merchant Reputation Score).
    """
    import models
    models.Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        if db.query(models.Merchant).count() > 0:
            print(f"[OK] Database SQLite sudah terisi {db.query(models.Merchant).count()} merchant.")
            return

        print("[LOG] Mengisi data reputasi EMRS awal (seeding) ke SQLite Database...")
        now = datetime.utcnow()

        # ─────────────────────────────────────────────────────────
        # 1. BUDI PRIBADI — Merchant Penipu (EMRS sangat rendah)
        #    Banyak mismatch, banyak dispute, transaksi sering gagal
        # ─────────────────────────────────────────────────────────
        m1 = models.Merchant(
            nmid="ID1024309405321",
            merchant_name="BUDI PRIBADI",
            acquirer="93600915",           # DANA
            registered_at=now - timedelta(days=90),   # baru 3 bulan
            # Transaction Reliability (T): 60/100 berhasil
            verified_transactions=100,
            successful_transactions=60,
            failed_transactions=40,
            # Authenticity (A): banyak mismatch, 8 critical
            identity_match_count=5,
            identity_mismatch_count=15,
            critical_mismatch_count=8,
            # Legacy
            rating=2.1,
            total_reports=17,
            verified_reports=12,
        )
        db.add(m1)
        db.commit()
        db.refresh(m1)

        # Reports untuk BUDI PRIBADI (kategori berat, severity CRITICAL/HIGH)
        laporan_m1 = [
            # QRIS Replacement - CRITICAL (8 laporan, sebagian dengan bukti)
            ("QRIS Replacement", "CRITICAL", "Stiker QRIS ditimpa di warung Toko Berkah Jaya! Transfer masuk ke Budi Pribadi.", True, 2, now - timedelta(days=5)),
            ("QRIS Replacement", "CRITICAL", "QRIS toko dicopot dan diganti QR digital Budi Pribadi.", True, 2, now - timedelta(days=10)),
            ("QRIS Replacement", "CRITICAL", "Stiker QRIS penipuan menimpa banner pembayaran resmi.", True, 2, now - timedelta(days=15)),
            ("QRIS Replacement", "CRITICAL", "Penyelidikan mendapati QRIS palsu menempel di gerobak pedagang.", True, 2, now - timedelta(days=20)),
            ("QRIS Replacement", "CRITICAL", "Stiker palsu Budi Pribadi menimpa QRIS BCA toko.", True, 1, now - timedelta(days=25)),
            ("QRIS Replacement", "HIGH",     "Merchant fisik Toko Berkah tetapi transfer masuk ke Budi Pribadi.", True, 1, now - timedelta(days=30)),
            ("QRIS Replacement", "HIGH",     "Perbedaan nama rekening penerima dengan nama fisik gerai.", True, 1, now - timedelta(days=45)),
            ("QRIS Replacement", "HIGH",     "QRIS palsu terdeteksi di minimarket lokal.", True, 1, now - timedelta(days=60)),
            # Additional Fee - HIGH
            ("Additional Fee",   "HIGH",     "Biaya admin Rp5.000 ditarik tanpa konfirmasi.", True, 2, now - timedelta(days=12)),
            ("Additional Fee",   "HIGH",     "Pemotongan saldo tambahan saat scan QR tanpa notifikasi.", True, 2, now - timedelta(days=18)),
            ("Additional Fee",   "MEDIUM",   "Mengisi biaya ekstra diluar nominal barang.", True, 1, now - timedelta(days=35)),
            ("Additional Fee",   "MEDIUM",   "Merchant meminta biaya 3% tambahan.", True, 1, now - timedelta(days=50)),
            # Merchant Mismatch - CRITICAL
            ("Merchant Mismatch","CRITICAL", "Nama di aplikasi DANA tidak sesuai dengan fisik toko.", False, 1, now - timedelta(days=8)),
            ("Merchant Mismatch","HIGH",     "Indikasi pencurian identitas merchant.", False, 2, now - timedelta(days=22)),
            ("Merchant Mismatch","HIGH",     "Lokasi toko tidak sesuai alamat registrasi.", False, 1, now - timedelta(days=40)),
            ("Merchant Mismatch","MEDIUM",   "NMID tidak terdaftar di sistem ASPII.", False, 1, now - timedelta(days=55)),
            ("Additional Fee",   "LOW",      "Meminta charge tunai tambahan.", False, 1, now - timedelta(days=70)),
        ]
        for kat, sev, desc, is_ver, ev_lvl, created in laporan_m1:
            db.add(models.Report(
                merchant_id=m1.id,
                category=kat, severity=sev, description=desc,
                is_verified=is_ver, evidence_level=ev_lvl, created_at=created
            ))

        # Disputes untuk BUDI PRIBADI (terverifikasi dan berat)
        disputes_m1 = [
            ("Transaksi Rp150.000 masuk ke BUDI PRIBADI bukan Toko Berkah Jaya. Bukti transfer dilampirkan.", "TX-001", "CRITICAL", True,  now - timedelta(days=7)),
            ("User kehilangan Rp200.000 karena QR replacement. Sudah lapor polisi.", "TX-002",               "CRITICAL", True,  now - timedelta(days=14)),
            ("Sengketa pembayaran ganda akibat scan ulang QRIS palsu.",              "TX-003",               "HIGH",     True,  now - timedelta(days=28)),
            ("Bukti transaksi menunjukkan perbedaan penerima dari nama fisik toko.", None,                   "HIGH",     False, now - timedelta(days=42)),
        ]
        for desc, ev_ref, sev, is_ver, created in disputes_m1:
            db.add(models.Dispute(
                merchant_id=m1.id, description=desc, evidence_ref=ev_ref,
                severity=sev, is_verified=is_ver, created_at=created
            ))
        db.commit()

        # ─────────────────────────────────────────────────────────
        # 2. PUSKESMAS PUNGGING — Merchant Aman & Terpercaya
        #    Transaksi konsisten, tidak ada mismatch
        # ─────────────────────────────────────────────────────────
        m2 = models.Merchant(
            nmid="ID2023269910873",
            merchant_name="082 PUSK PUNGGING",
            acquirer="93600114",           # LINKAJA
            registered_at=now - timedelta(days=730),  # 2 tahun aktif
            # Transaction Reliability (T): 490/500 berhasil
            verified_transactions=500,
            successful_transactions=490,
            failed_transactions=10,
            # Authenticity (A): hampir selalu match
            identity_match_count=98,
            identity_mismatch_count=2,
            critical_mismatch_count=0,
            # Legacy
            rating=4.9,
            total_reports=2,
            verified_reports=2,
        )

        # ─────────────────────────────────────────────────────────
        # 3. ES COKLAT AJA 26 QR — Merchant Aman, Aktif Lama
        #    Keluhan minor, tidak ada mismatch serius
        # ─────────────────────────────────────────────────────────
        m3 = models.Merchant(
            nmid="ID1023286077558",
            merchant_name="ES COKLAT AJA 26 QR",
            acquirer="93600014",           # BCA
            registered_at=now - timedelta(days=1200), # lebih dari 3 tahun
            # Transaction Reliability (T): 285/300 berhasil
            verified_transactions=300,
            successful_transactions=285,
            failed_transactions=15,
            # Authenticity (A): konsisten match
            identity_match_count=55,
            identity_mismatch_count=5,
            critical_mismatch_count=0,
            # Legacy
            rating=4.8,
            total_reports=4,
            verified_reports=4,
        )
        db.add_all([m2, m3])
        db.commit()
        db.refresh(m2)
        db.refresh(m3)

        # Reports untuk PUSK PUNGGING — keluhan sangat ringan
        db.add(models.Report(
            merchant_id=m2.id, category="Additional Fee", severity="LOW",
            description="Pertanyaan mengenai admin nominal kecil.",
            is_verified=True, evidence_level=1, created_at=now - timedelta(days=180)
        ))
        db.add(models.Report(
            merchant_id=m2.id, category="General Complaint", severity="LOW",
            description="Proses scan agak lama saat jam sibuk.",
            is_verified=True, evidence_level=1, created_at=now - timedelta(days=90)
        ))

        # Reports untuk ES COKLAT AJA — keluhan ringan, lama
        db.add(models.Report(
            merchant_id=m3.id, category="Additional Fee", severity="LOW",
            description="Antrian kasir panjang saat scan QR.",
            is_verified=True, evidence_level=1, created_at=now - timedelta(days=365)
        ))
        db.add(models.Report(
            merchant_id=m3.id, category="Additional Fee", severity="LOW",
            description="Pernah mati lampu saat proses transaksi.",
            is_verified=True, evidence_level=1, created_at=now - timedelta(days=300)
        ))
        db.add(models.Report(
            merchant_id=m3.id, category="Merchant Mismatch", severity="LOW",
            description="Perubahan nama outlet cabang baru — sudah dikonfirmasi.",
            is_verified=True, evidence_level=1, created_at=now - timedelta(days=200)
        ))
        db.add(models.Report(
            merchant_id=m3.id, category="General Complaint", severity="LOW",
            description="Nominal QR berbeda tipis dari harga kasir — ternyata promo.",
            is_verified=True, evidence_level=1, created_at=now - timedelta(days=120)
        ))
        db.commit()

        print("[OK] Auto-seeding database EMRS berhasil diselesaikan!")
    finally:
        db.close()


def reset_db():
    """Drop semua tabel dan rebuild + re-seed dari scratch."""
    import models
    models.Base.metadata.drop_all(bind=engine)
    init_db()


if __name__ == "__main__":
    init_db()
