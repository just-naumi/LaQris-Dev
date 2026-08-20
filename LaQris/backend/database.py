import os
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
    Inisialisasi tabel database SQLite dan auto-seed data dummy reputasi merchant.
    """
    import models
    models.Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Cek apakah database sudah ada isi data merchant
        if db.query(models.Merchant).count() == 0:
            print("[LOG] Mengisi data reputasi awal (seeding) ke SQLite Database...")

            # 1. Merchant Penipu / Ditimpa (BUDI PRIBADI)
            m1 = models.Merchant(
                nmid="ID1024309405321",
                merchant_name="BUDI PRIBADI",
                acquirer="93600915",
                rating=2.1,
                total_reports=17,
                verified_reports=12
            )
            db.add(m1)
            db.commit()
            db.refresh(m1)

            # Laporan untuk BUDI PRIBADI
            laporan_m1 = [
                ("QRIS Replacement", "Stiker QRIS di warung ditimpa stiker atas nama Budi Pribadi! Mismatch parah.", True),
                ("QRIS Replacement", "QRIS toko dicopot dan diganti QR digital Budi Pribadi.", True),
                ("QRIS Replacement", "Stiker QRIS penipuan menimpa banner pembayaran resmi.", True),
                ("QRIS Replacement", "Penyelidikan mendapati QRIS palsu menempel di gerobak pedagang.", True),
                ("QRIS Replacement", "Stiker palsu Budi Pribadi menimpa QRIS BCA toko.", True),
                ("QRIS Replacement", "Merchant fisik Toko Berkah tetapi transfer masuk ke Budi Pribadi.", True),
                ("QRIS Replacement", "Perbedaan nama rekening penerima dengan nama fisik gerai.", True),
                ("QRIS Replacement", "QRIS palsu terdeteksi di minimarket lokal.", True),
                ("Additional Fee", "Biaya admin ditarik tanpa konfirmasi.", True),
                ("Additional Fee", "Pemotongan saldo tambahan saat scan QR.", True),
                ("Additional Fee", "Mengisi biaya ekstra diluar nominal barang.", True),
                ("Additional Fee", "Merchant meminta biaya 3% tambahan.", True),
                ("Merchant Mismatch", "Nama di aplikasi DANA tidak sesuai dengan fisik toko.", False),
                ("Merchant Mismatch", "Indikasi pencurian identitas merchant.", False),
                ("Merchant Mismatch", "Lokasi toko tidak sesuai alamat registrasi.", False),
                ("Merchant Mismatch", "NMID tidak terdaftar di sistem ASPII.", False),
                ("Additional Fee", "Meminta charge tunai tambahan.", False)
            ]
            for kat, desc, is_ver in laporan_m1:
                db.add(models.Report(merchant_id=m1.id, category=kat, description=desc, is_verified=is_ver))

            # 2. Merchant Resmi / Aman (ES COKLAT AJA 26 QR / PUSKESMAS PUNGGING)
            m2 = models.Merchant(
                nmid="ID2023269910873",
                merchant_name="082 PUSK PUNGGING",
                acquirer="93600114",
                rating=4.9,
                total_reports=1,
                verified_reports=1
            )
            m3 = models.Merchant(
                nmid="ID1023286077558",
                merchant_name="ES COKLAT AJA 26 QR",
                acquirer="93600014",
                rating=4.8,
                total_reports=3,
                verified_reports=3
            )
            db.add_all([m2, m3])
            db.commit()
            db.refresh(m2)
            db.refresh(m3)

            db.add(models.Report(merchant_id=m2.id, category="Additional Fee", description="Pertanyaan mengenai admin nominal kecil", is_verified=True))
            db.add(models.Report(merchant_id=m3.id, category="Additional Fee", description="Antrian kasir panjang saat scan QR", is_verified=True))
            db.add(models.Report(merchant_id=m3.id, category="Additional Fee", description="Pernah mati lampu saat proses transaksi", is_verified=True))
            db.add(models.Report(merchant_id=m3.id, category="Merchant Mismatch", description="Perubahan nama outlet cabang baru", is_verified=True))

            db.commit()
            print("[OK] Auto-seeding database SQLite berhasil diselesaikan!")
        else:
            print(f"[OK] Database SQLite sudah terisi {db.query(models.Merchant).count()} merchant.")
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
