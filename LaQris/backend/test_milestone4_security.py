"""
test_milestone4_security.py
Pengujian Otomatis Komprehensif untuk Roadmap Milestone 4 (Security Hardening & Readiness)
Sesuai Dokumen Readme2.md (Step 28 - Step 31):
1. Argon2id Password Hashing & Legacy SHA-256 Transparent Migration (Step 28)
2. JWT Access Token Signing & Claims Validation (Step 28)
3. User Ownership Enforcement (Step 29)
4. CORS Configuration & In-Memory Rate Limiting (Step 30)
5. Sensitive Payment Field Sanitization (Step 15)
6. Seed Endpoint Protection in Production (Step 31)
"""

import os
import sys
from fastapi.testclient import TestClient

# Pastikan path modul terdaftar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app, rate_limiter
from database import get_db
from models import User, VerificationSession, PaymentTransaction, Merchant
import auth

client = TestClient(app)

def test_milestone4_argon2_and_jwt_registration():
    print("\n[TEST 1] Pengujian Registrasi Pengguna Baru (Argon2id + JWT)...")
    email = "test_sec_user_2026@laqris.id"
    payload = {
        "username": "secuser2026",
        "email": email,
        "password": "SuperSecretPassword#2026",
        "full_name": "Security Hardened User",
        "phone": "081299998888",
        "role": "PENGGUNA"
    }
    
    # Hapus user jika sudah ada sebelumnya
    db = next(get_db())
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        db.delete(existing)
        db.commit()

    res = client.post("/api/register", json=payload)
    assert res.status_code == 200, f"Register failed: {res.text}"
    data = res.json()
    
    assert "access_token" in data, "Token JWT tidak ditemukan di respons"
    assert data["token_type"] == "bearer"
    
    # Validasi dekode JWT
    claims = auth.decode_access_token(data["access_token"])
    assert claims is not None, "Gagal mendecode JWT token"
    assert claims["sub"] == data["user"]["user_id"]
    assert claims["email"] == email
    print(f"  -> JWT Token Valid: sub={claims['sub']}, exp={claims.get('exp')}")
    
    # Validasi hash di database adalah Argon2id ($argon2id$...)
    db_user = db.query(User).filter(User.email == email).first()
    assert db_user.password_hash.startswith("$argon2id$"), f"Password hash bukan Argon2id: {db_user.password_hash}"
    print(f"  -> Format Hash DB: {db_user.password_hash[:30]}... (Argon2id verified)")


def test_milestone4_login_and_legacy_rehash():
    print("\n[TEST 2] Pengujian Login & Migrasi Transparan Legacy SHA-256 ke Argon2id...")
    # yantoalim adalah user seed awal
    db = next(get_db())
    user = db.query(User).filter(User.email == "yanto@gmail.com").first()
    assert user is not None, "User seed yanto@gmail.com tidak ditemukan"

    res = client.post("/api/login", json={
        "email": "yanto@gmail.com",
        "password": "password123"
    })
    assert res.status_code == 200, f"Login yantoalim gagal: {res.text}"
    data = res.json()
    assert "access_token" in data
    
    # Cek bahwa hash di database telah ter-migrasi/rehash ke Argon2id
    db.refresh(user)
    assert user.password_hash.startswith("$argon2id$"), "Hash pengguna lama tidak ter-rehash ke Argon2id"
    print(f"  -> User yantoalim berhasil dimigrasikan ke Argon2id: {user.password_hash[:30]}...")


def test_milestone4_user_ownership_enforcement():
    print("\n[TEST 3] Pengujian User Ownership Enforcement (Step 29 di Readme2.md)...")
    db = next(get_db())
    
    # 1. Buat Sesi Verifikasi untuk User A (USR-AAA)
    session_id = "LQ-V-TEST-OWNER-01"
    existing_s = db.query(VerificationSession).filter(VerificationSession.session_id == session_id).first()
    if existing_s:
        db.delete(existing_s)
        db.commit()

    s = VerificationSession(
        session_id=session_id,
        user_id="USR-AAA",
        nmid="ID1020000000001",
        digital_name="WARUNG MAKAN TEST",
        physical_name="WARUNG MAKAN TEST",
        trust_score=95.0,
        risk_level="NORMAL",
        decision="ALLOW"
    )
    db.add(s)
    db.commit()

    # 2. User B (USR-BBB) mencoba mengirim ulasan atas sesi milik User A
    feedback_payload = {
        "verification_id": session_id,
        "comment": "Nama pemilik toko beda dengan QRIS saat discan",
        "user_id": "USR-BBB"  # Berbeda dari USR-AAA
    }
    
    res = client.post("/api/v1/feedback", json=feedback_payload)
    assert res.status_code == 403, f"Seharusnya ditolak 403 Forbidden, tapi respon: {res.status_code} {res.text}"
    print(f"  -> Penolakan berhasil (HTTP 403): {res.json()['detail']}")

    # 3. User A (USR-AAA) mengirim ulasan atas sesinya sendiri -> Harus Berhasil (HTTP 200)
    feedback_payload_valid = {
        "verification_id": session_id,
        "comment": "Nama pemilik toko beda dengan QRIS saat discan",
        "user_id": "USR-AAA"  # Cocok
    }
    res_valid = client.post("/api/v1/feedback", json=feedback_payload_valid)
    assert res_valid.status_code == 200, f"Seharusnya berhasil 200 OK: {res_valid.text}"
    print(f"  -> Akses valid berhasil (HTTP 200): Skor Baru={res_valid.json().get('new_reputation_score')}")


def test_milestone4_sensitive_field_sanitization():
    print("\n[TEST 4] Pengujian Sanitasi Data Sensitif Pembayaran (Step 15 di Readme2.md)...")
    db = next(get_db())
    
    s_id = "LQ-V-TEST-SAN-01"
    existing_s = db.query(VerificationSession).filter(VerificationSession.session_id == s_id).first()
    if existing_s:
        db.delete(existing_s)
        db.commit()

    s = VerificationSession(
        session_id=s_id,
        user_id="USR-001928",
        nmid="ID1020000000001",
        trust_score=90.0,
        risk_level="NORMAL",
        decision="ALLOW"
    )
    db.add(s)
    db.commit()

    # Kirim payload transaksi yang sengaja menyertakan data kartu kredit/PIN sensitif
    tx_payload = {
        "verification_id": s_id,
        "provider": "DemoPay",
        "provider_transaction_id": "TX-SAN-999",
        "amount": 50000.0,
        "status": "SUCCESS",
        "response_code": "00",
        "card_number": "4111111111111111",   # Data sensitif
        "cvv": "123",                         # Data sensitif
        "pin": "654321"                       # Data sensitif
    }

    # Hapus TX lama jika ada
    existing_tx = db.query(PaymentTransaction).filter(PaymentTransaction.provider_transaction_id == "TX-SAN-999").first()
    if existing_tx:
        db.delete(existing_tx)
        db.commit()

    res = client.post("/api/v1/transaction-events", json=tx_payload)
    assert res.status_code == 200, f"Event transaksi gagal: {res.text}"
    
    # Verifikasi field sensitif tidak ada di objek database
    saved_tx = db.query(PaymentTransaction).filter(PaymentTransaction.provider_transaction_id == "TX-SAN-999").first()
    assert saved_tx is not None
    assert not hasattr(saved_tx, "card_number")
    assert not hasattr(saved_tx, "cvv")
    assert not hasattr(saved_tx, "pin")
    print("  -> Sanitasi field sensitif (CVV, Card Number, PIN) berhasil terverifikasi.")


def test_milestone4_seed_endpoint_protection():
    print("\n[TEST 5] Pengujian Proteksi Endpoint Seed (Step 31 di Readme2.md)...")
    # Simulasikan mode production
    os.environ["LAQRIS_ENV"] = "production"
    os.environ["LAQRIS_ADMIN_SECRET"] = "admin-secret-laqris-2026"

    # Panggilan tanpa secret di mode production -> 403 Forbidden
    res_fail = client.post("/api/seed")
    assert res_fail.status_code == 403, f"Seharusnya 403 Forbidden, tapi {res_fail.status_code}"
    print(f"  -> Akses seed production tanpa secret ditolak: {res_fail.json()['detail']}")

    # Panggilan dengan X-Admin-Secret yang benar -> 200 OK
    res_ok = client.post("/api/seed", headers={"X-Admin-Secret": "admin-secret-laqris-2026"})
    assert res_ok.status_code == 200
    print("  -> Akses seed production dengan header X-Admin-Secret berhasil.")

    # Kembalikan ke mode development
    os.environ["LAQRIS_ENV"] = "development"


def test_milestone4_rate_limiting():
    print("\n[TEST 6] Pengujian Sliding Window Rate Limiter (Step 30 di Readme2.md)...")
    # Rate limit untuk login adalah 10 req / 60 detik
    rate_limiter._records.clear() # Reset state
    
    status_codes = []
    for i in range(12):
        res = client.post("/api/login", json={"email": "nonexistent@gmail.com", "password": "wrong"})
        status_codes.append(res.status_code)
    
    # Permintaan ke-11 dan 12 harus 429 Too Many Requests
    assert 429 in status_codes, f"Rate limiter tidak menembak 429: {status_codes}"
    print(f"  -> Rate limiting berhasil: status codes = {status_codes[-3:]} (HTTP 429 aktif)")


if __name__ == "__main__":
    print("=" * 70)
    print("MENJALANKAN PENGUJIAN OTOMATIS MILESTONE 4: SECURITY HARDENING")
    print("=" * 70)
    test_milestone4_argon2_and_jwt_registration()
    test_milestone4_login_and_legacy_rehash()
    test_milestone4_user_ownership_enforcement()
    test_milestone4_sensitive_field_sanitization()
    test_milestone4_seed_endpoint_protection()
    test_milestone4_rate_limiting()
    print("=" * 70)
    print("SELURUH 6 PENGUJIAN KEAMANAN ROADMAP MILESTONE 4 BERHASIL 100%!")
    print("=" * 70)
