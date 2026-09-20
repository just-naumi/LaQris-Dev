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
import json
import hmac
import hashlib
from datetime import datetime, timedelta
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
    rate_limiter._records.clear()  # Reset state

    status_codes = []
    for i in range(12):
        res = client.post("/api/login", json={"email": "nonexistent@gmail.com", "password": "wrong"})
        status_codes.append(res.status_code)

    # Permintaan ke-11 dan 12 harus 429 Too Many Requests
    assert 429 in status_codes, f"Rate limiter tidak menembak 429: {status_codes}"
    print(f"  -> Rate limiting berhasil: status codes = {status_codes[-3:]} (HTTP 429 aktif)")


def test_milestone4_pin_and_pan_credential_hygiene():
    print("\n[TEST 7] Pengujian Credential Hygiene: PIN Tidak Boleh Bocor di API (P0 Item 4)...")
    rate_limiter._records.clear()  # Reset rate limiter agar tidak terhalang test 6
    # 1. Cek response register
    email = "test_hygiene@laqris.id"
    payload = {
        "username": "hygieneuser",
        "email": email,
        "password": "Password123#",
        "full_name": "Hygiene User"
    }
    db = next(get_db())
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        db.delete(existing)
        db.commit()

    res_reg = client.post("/api/register", json=payload)
    assert res_reg.status_code == 200
    assert "pin" not in res_reg.json()["user"], "PIN bocor di response register!"

    # 2. Cek response login
    res_login = client.post("/api/login", json={"email": email, "password": "Password123#"})
    assert res_login.status_code == 200
    assert "pin" not in res_login.json()["user"], "PIN bocor di response login!"

    # 3. Cek endpoint /api/user/current
    res_curr = client.get("/api/user/current")
    assert res_curr.status_code == 200
    assert "pin" not in res_curr.json(), "PIN bocor di response /api/user/current!"

    # 4. Cek secure pin verification via constant time
    res_pin_ok = client.post("/api/user/verify-pin", json={"pin": "123456"})
    assert res_pin_ok.status_code == 200
    assert res_pin_ok.json()["valid"] is True

    res_pin_fail = client.post("/api/user/verify-pin", json={"pin": "999999"})
    assert res_pin_fail.status_code == 200
    assert res_pin_fail.json()["valid"] is False
    print("  -> Credential Hygiene lolos: PIN tidak pernah diekspos di endpoint publik.")


def test_milestone4_provider_authentication():
    print("\n[TEST 8] Pengujian Service-to-Service Provider Authentication (P1 Item 5)...")
    db = next(get_db())
    s_id = "LQ-V-TEST-PROV-AUTH"
    existing_s = db.query(VerificationSession).filter(VerificationSession.session_id == s_id).first()
    if existing_s:
        db.delete(existing_s)
        db.commit()

    s = VerificationSession(
        session_id=s_id,
        user_id="USR-001928",
        nmid="ID1020000000001",
        trust_score=95.0,
        risk_level="NORMAL",
        decision="ALLOW"
    )
    db.add(s)
    db.commit()

    # Simulasikan mode production
    os.environ["LAQRIS_ENV"] = "production"

    tx_payload = {
        "verification_session_id": s_id,
        "provider": "DemoPay",
        "provider_transaction_id": "TX-PROV-001",
        "amount": 25000.0,
        "status": "SUCCESS",
        "response_code": "00"
    }

    # 1. Panggilan tanpa API Key di mode production -> 401 Unauthorized
    res_fail = client.post("/api/v1/transactions/events", json=tx_payload)
    assert res_fail.status_code == 401, f"Harusnya ditolak 401, tapi {res_fail.status_code}"
    print(f"  -> Provider tanpa auth ditolak 401: {res_fail.json()['detail']}")

    # 2. Panggilan dengan X-Provider-Api-Key yang valid -> 200 OK
    res_ok = client.post(
        "/api/v1/transactions/events",
        json=tx_payload,
        headers={"X-Provider-Api-Key": "demopay-live-key-2026-auth"}
    )
    assert res_ok.status_code == 200, f"Provider auth gagal: {res_ok.text}"
    print("  -> Provider dengan X-Provider-Api-Key valid diterima (HTTP 200).")

    # 3. Panggilan dengan X-LaQris-Signature (HMAC-SHA256 of raw_body) valid
    s.is_bound = False
    db.commit()
    tx_hmac_payload = {
        "verification_session_id": s_id,
        "provider": "DemoPay",
        "provider_transaction_id": "TX-PROV-HMAC-001",
        "amount": 30000.0,
        "status": "SUCCESS",
        "response_code": "00"
    }
    raw_payload_bytes = json.dumps(tx_hmac_payload).encode("utf-8")
    hmac_sig = hmac.new(
        b"demopay-service-hmac-secret-2026",
        raw_payload_bytes,
        hashlib.sha256
    ).hexdigest()
    res_hmac = client.post(
        "/api/v1/transactions/events",
        content=raw_payload_bytes,
        headers={
            "Content-Type": "application/json",
            "X-LaQris-Signature": hmac_sig
        }
    )
    assert res_hmac.status_code == 200, f"HMAC provider auth gagal: {res_hmac.text}"
    print("  -> Provider dengan X-LaQris-Signature (HMAC-SHA256) valid diterima (HTTP 200).")

    # 4. Panggilan dengan signature palsu -> 401
    s.is_bound = False
    db.commit()
    res_bad_sig = client.post(
        "/api/v1/transactions/events",
        content=raw_payload_bytes,
        headers={
            "Content-Type": "application/json",
            "X-LaQris-Signature": "invalid-hmac-signature-deadbeef"
        }
    )
    assert res_bad_sig.status_code == 401
    print("  -> Provider dengan signature palsu berhasil ditolak (HTTP 401).")

    os.environ["LAQRIS_ENV"] = "development"


def test_milestone4_strict_merchant_binding():
    print("\n[TEST 9] Pengujian Strict Merchant/Session Binding (P1 Item 6)...")
    db = next(get_db())
    s_id = "LQ-V-TEST-STRICT-NMID"
    existing_s = db.query(VerificationSession).filter(VerificationSession.session_id == s_id).first()
    if existing_s:
        db.delete(existing_s)
        db.commit()

    s = VerificationSession(
        session_id=s_id,
        user_id="USR-001928",
        nmid="ID1020000000123", # NMID sesi resmi
        trust_score=95.0,
        risk_level="NORMAL",
        decision="ALLOW"
    )
    db.add(s)
    db.commit()

    # Provider mengirim transaksi dengan NMID toko lain yang berbeda!
    tx_mismatch_payload = {
        "verification_session_id": s_id,
        "provider": "DemoPay",
        "provider_transaction_id": "TX-MISMATCH-001",
        "amount": 50000.0,
        "status": "SUCCESS",
        "response_code": "00",
        "nmid": "ID9999999999999" # Mismatch sengaja
    }

    res_mismatch = client.post(
        "/api/v1/transactions/events",
        json=tx_mismatch_payload,
        headers={"X-Provider-Api-Key": "demopay-live-key-2026-auth"}
    )
    assert res_mismatch.status_code == 422, f"Harusnya ditolak 422 Unprocessable, tapi {res_mismatch.status_code}"
    assert "Strict merchant binding mismatch" in res_mismatch.json()["detail"]
    print(f"  -> Strict NMID mismatch berhasil ditolak (HTTP 422): {res_mismatch.json()['detail']}")


def test_milestone4_transaction_status_enum_validation():
    print("\n[TEST 10] Pengujian Validasi Mandatory Status Enum (P1 Item 7)...")
    s_id = "LQ-V-TEST-ENUM"
    # Status invalid (bukan salah satu dari SUCCESS, FAILED, TIMEOUT, CANCELLED, REVERSED)
    tx_invalid_status = {
        "verification_session_id": s_id,
        "provider": "DemoPay",
        "provider_transaction_id": "TX-ENUM-001",
        "amount": 10000.0,
        "status": "PENDING_UNKNOWN",
        "response_code": "00"
    }
    res_inv = client.post(
        "/api/v1/transactions/events",
        json=tx_invalid_status,
        headers={"X-Provider-Api-Key": "demopay-live-key-2026-auth"}
    )
    assert res_inv.status_code == 422, "Status transaksi invalid harusnya ditolak Pydantic 422"
    print("  -> Status transaksi di luar enum berhasil ditolak oleh Pydantic schema (HTTP 422).")


def test_milestone4_demopay_server_processing():
    print("\n[TEST 11] Pengujian DemoPay Server-Side Payment Processing Gateway (P0 Item 1-3)...")
    db = next(get_db())

    # 1. Uji penolakan sesi BLOCK / DANGER oleh DemoPay Server
    block_s_id = "LQ-V-TEST-DEMOPAY-BLOCK"
    existing_b = db.query(VerificationSession).filter(VerificationSession.session_id == block_s_id).first()
    if existing_b:
        db.delete(existing_b)
        db.commit()

    s_block = VerificationSession(
        session_id=block_s_id,
        user_id="USR-001928",
        nmid="ID1020000000999",
        trust_score=10.0,
        risk_level="DANGER",
        decision="BLOCK"
    )
    db.add(s_block)
    db.commit()

    res_block = client.post("/api/v1/demopay/process-payment", json={
        "session_id": block_s_id,
        "amount": 50000.0,
        "pin": "123456",
        "user_id": "USR-001928"
    })
    assert res_block.status_code == 403, f"Harusnya 403 BLOCK, dapat {res_block.status_code}"
    assert "DANGER" in res_block.json()["detail"] or "BLOCK" in res_block.json()["detail"]
    print(f"  -> DemoPay menolak sesi berstatus BLOCK / DANGER: {res_block.json()['detail']}")

    # 2. Uji PIN salah pada DemoPay Server
    allow_s_id = "LQ-V-TEST-DEMOPAY-ALLOW"
    existing_a = db.query(VerificationSession).filter(VerificationSession.session_id == allow_s_id).first()
    if existing_a:
        db.delete(existing_a)
        db.commit()

    s_allow = VerificationSession(
        session_id=allow_s_id,
        user_id="USR-001928",
        nmid="ID1020000000001",
        digital_name="WARUNG MAKAN SEDAP",
        trust_score=98.0,
        risk_level="NORMAL",
        decision="ALLOW"
    )
    db.add(s_allow)
    db.commit()

    # Uji penolakan jika user_id tidak cocok dengan pemilik sesi (P0 Item 1.A)
    res_user_mismatch = client.post("/api/v1/demopay/process-payment", json={
        "session_id": allow_s_id,
        "amount": 25000.0,
        "pin": "123456",
        "user_id": "USR-OTHER-USER"
    })
    assert res_user_mismatch.status_code == 403
    print(f"  -> DemoPay menolak transaksi dari user bukan pemilik sesi (HTTP 403).")

    res_wrong_pin = client.post("/api/v1/demopay/process-payment", json={
        "session_id": allow_s_id,
        "amount": 25000.0,
        "pin": "999999",
        "user_id": "USR-001928"
    })
    assert res_wrong_pin.status_code == 400
    assert "PIN" in res_wrong_pin.json()["detail"]
    print("  -> DemoPay Server berhasil menolak PIN yang salah (HTTP 400).")

    # 3. Uji pemrosesan pembayaran sukses & pengukuran metrik latensi nyata
    res_pay_ok = client.post("/api/v1/demopay/process-payment", json={
        "session_id": allow_s_id,
        "amount": 25000.0,
        "pin": "123456",
        "terminal_id": "A01",
        "user_id": "USR-001928"
    })
    assert res_pay_ok.status_code == 200, f"DemoPay pay gagal: {res_pay_ok.text}"
    p_data = res_pay_ok.json()
    assert p_data["success"] is True
    assert p_data["status"] == "SUCCESS"
    assert p_data["transaction_id"].startswith("TX-")
    assert p_data["invoice_number"].startswith("INV-")
    assert p_data["latency_ms"] >= 100, f"Latency harus diukur nyata (>100ms), dapat {p_data['latency_ms']}ms"
    assert "transaction_time" in p_data
    print(f"  -> DemoPay Server sukses memproses: TX={p_data['transaction_id']}, Invoice={p_data['invoice_number']}")
    print(f"  -> Actual Measured Latency: {p_data['latency_ms']} ms (bukan hard-coded!)")

    # 4. Verifikasi state database (Anti-replay & PaymentTransaction persistence)
    db.expire_all()
    s_updated = db.query(VerificationSession).filter(VerificationSession.session_id == allow_s_id).first()
    assert s_updated.is_bound is True, "Sesi verifikasi harus diikat (is_bound=True)"

    tx_rec = db.query(PaymentTransaction).filter(PaymentTransaction.provider_transaction_id == p_data["transaction_id"]).first()
    assert tx_rec is not None, "Record PaymentTransaction harus tersimpan di database"
    assert tx_rec.latency_ms == p_data["latency_ms"]
    assert tx_rec.status == "SUCCESS"
    print("  -> Database persistence terverifikasi: Session terikat & PaymentTransaction tersimpan aman.")


def test_milestone4_nominal_locking_and_anti_tamper():
    print("\n[TEST 12] Pengujian Anti-Tamper Nominal Locking (P0 Item 2 di Readme.md)...")
    db = next(get_db())
    s_id = "LQ-V-TEST-LOCK-NOMINAL"
    db.query(VerificationSession).filter(VerificationSession.session_id == s_id).delete()
    db.commit()

    s = VerificationSession(
        session_id=s_id,
        user_id="USR-001928",
        nmid="ID1020000000001",
        trust_score=95.0,
        risk_level="NORMAL",
        decision="ALLOW"
    )
    db.add(s)
    db.commit()

    # 1. Kunci nominal Rp27.000 via Payment Intent
    res_intent = client.post("/api/v1/payment-intent", json={
        "session_id": s_id,
        "amount": 27000.0
    })
    assert res_intent.status_code == 200
    assert res_intent.json()["amount_locked"] is True
    assert res_intent.json()["amount"] == 27000.0

    # 2. Penyerang / client mencoba memanipulasi nominal menjadi Rp270.000 -> Harus ditolak HTTP 422!
    res_tamper = client.post("/api/v1/demopay/process-payment", json={
        "session_id": s_id,
        "amount": 270000.0,
        "pin": "123456",
        "user_id": "USR-001928"
    })
    assert res_tamper.status_code == 422, f"Harusnya 422 Unprocessable, tapi {res_tamper.status_code}"
    assert "Anti-tamper" in res_tamper.json()["detail"]
    print(f"  -> Manipulasi nominal berhasil digagalkan (HTTP 422): {res_tamper.json()['detail']}")

    # 3. Transaksi dengan nominal yang sesuai -> Berhasil (HTTP 200)
    res_valid_nom = client.post("/api/v1/demopay/process-payment", json={
        "session_id": s_id,
        "amount": 27000.0,
        "pin": "123456",
        "user_id": "USR-001928"
    })
    assert res_valid_nom.status_code == 200
    assert res_valid_nom.json()["amount"] == 27000.0
    print("  -> Pembayaran dengan nominal terkunci yang sah berhasil diproses.")


def test_milestone4_demopay_multistatus_scenarios():
    print("\n[TEST 13] Pengujian Skenario Multi-Status DemoPay (P0 Item 4 di Readme.md)...")
    db = next(get_db())

    # Skenario FAILED
    s_failed_id = "LQ-V-TEST-STATUS-FAIL"
    db.query(VerificationSession).filter(VerificationSession.session_id == s_failed_id).delete()
    db.commit()
    s_f = VerificationSession(
        session_id=s_failed_id,
        user_id="USR-001928",
        nmid="ID1020000000001",
        trust_score=95.0,
        risk_level="NORMAL",
        decision="ALLOW"
    )
    db.add(s_f)
    db.commit()

    res_failed = client.post("/api/v1/demopay/process-payment", json={
        "session_id": s_failed_id,
        "amount": 15000.0,
        "pin": "123456",
        "scenario": "FAILED",
        "user_id": "USR-001928"
    })
    assert res_failed.status_code == 200
    f_data = res_failed.json()
    assert f_data["status"] == "FAILED"
    assert f_data["response_code"] == "05"
    assert f_data["success"] is False
    print(f"  -> Skenario FAILED terverifikasi: Status={f_data['status']}, RC={f_data['response_code']}")

    # Skenario TIMEOUT
    s_to_id = "LQ-V-TEST-STATUS-TIMEOUT"
    db.query(VerificationSession).filter(VerificationSession.session_id == s_to_id).delete()
    db.commit()
    s_to = VerificationSession(
        session_id=s_to_id,
        user_id="USR-001928",
        nmid="ID1020000000001",
        trust_score=95.0,
        risk_level="NORMAL",
        decision="ALLOW"
    )
    db.add(s_to)
    db.commit()

    res_to = client.post("/api/v1/demopay/process-payment", json={
        "session_id": s_to_id,
        "amount": 20000.0,
        "pin": "123456",
        "scenario": "TIMEOUT",
        "user_id": "USR-001928"
    })
    assert res_to.status_code == 200
    to_data = res_to.json()
    assert to_data["status"] == "TIMEOUT"
    assert to_data["response_code"] == "68"
    assert to_data["success"] is False
    print(f"  -> Skenario TIMEOUT terverifikasi: Status={to_data['status']}, RC={to_data['response_code']}")


def test_milestone4_anti_replay_nonce_and_timestamp():
    print("\n[TEST 14] Pengujian Anti-Replay Nonce & Timestamp Header (P1 Item 7 di Readme.md)...")
    os.environ["LAQRIS_ENV"] = "production"
    db = next(get_db())
    s_id = "LQ-V-TEST-NONCE-01"
    db.query(VerificationSession).filter(VerificationSession.session_id == s_id).delete()
    db.commit()
    s = VerificationSession(
        session_id=s_id,
        user_id="USR-001928",
        nmid="ID1020000000001",
        trust_score=95.0,
        risk_level="NORMAL",
        decision="ALLOW"
    )
    db.add(s)
    db.commit()

    # 1. Test expired timestamp (> 300 detik)
    old_ts = (datetime.utcnow() - timedelta(seconds=350)).isoformat() + "Z"
    res_old = client.post(
        "/api/v1/transactions/events",
        json={
            "verification_session_id": s_id,
            "provider": "DemoPay",
            "provider_transaction_id": "TX-OLD-TS",
            "amount": 10000.0,
            "status": "SUCCESS",
            "response_code": "00"
        },
        headers={
            "X-Provider-Api-Key": "demopay-live-key-2026-auth",
            "X-LaQris-Timestamp": old_ts,
            "X-LaQris-Nonce": "nonce-old-123"
        }
    )
    assert res_old.status_code == 401
    assert "anti-replay" in res_old.json()["detail"].lower()
    print("  -> Request dengan timestamp kedaluwarsa berhasil ditolak (HTTP 401).")

    # 2. Test valid nonce pertama kali
    current_ts = datetime.utcnow().isoformat() + "Z"
    test_nonce = f"nonce-test-{int(datetime.utcnow().timestamp())}"
    res_nonce_1 = client.post(
        "/api/v1/transactions/events",
        json={
            "verification_session_id": s_id,
            "provider": "DemoPay",
            "provider_transaction_id": "TX-NONCE-01",
            "amount": 10000.0,
            "status": "SUCCESS",
            "response_code": "00"
        },
        headers={
            "X-Provider-Api-Key": "demopay-live-key-2026-auth",
            "X-LaQris-Timestamp": current_ts,
            "X-LaQris-Nonce": test_nonce
        }
    )
    assert res_nonce_1.status_code == 200
    print("  -> Request pertama dengan nonce unik berhasil diterima (HTTP 200).")

    # 3. Test replay attack: nonce yang sama dikirim ulang -> Harus ditolak 401!
    res_nonce_replay = client.post(
        "/api/v1/transactions/events",
        json={
            "verification_session_id": s_id,
            "provider": "DemoPay",
            "provider_transaction_id": "TX-NONCE-REPLAY",
            "amount": 10000.0,
            "status": "SUCCESS",
            "response_code": "00"
        },
        headers={
            "X-Provider-Api-Key": "demopay-live-key-2026-auth",
            "X-LaQris-Timestamp": current_ts,
            "X-LaQris-Nonce": test_nonce
        }
    )
    assert res_nonce_replay.status_code == 401
    assert "anti-replay" in res_nonce_replay.json()["detail"].lower()
    print("  -> Replay attack dengan nonce sama berhasil dicegah (HTTP 401).")

    os.environ["LAQRIS_ENV"] = "development"


def test_milestone4_hybrid_nlp_guard():
    print("\n[TEST 15] Pengujian Hybrid Guard NLP Feedback (P1 Item 11 di Readme.md)...")
    from nlp_classifier import classify_feedback

    text = "Nama merchant sama dengan nama toko, nominal sesuai, pembayaran berhasil tanpa kendala."
    res = classify_feedback(text)
    assert res["category_key"] == "QRIS_NORMAL_MERCHANT_TERPERCAYA", f"Hasil tidak terproteksi guard: {res}"
    assert res["severity"] == "LOW"
    assert res["emrs_penalty"] == "BOOST_TRUST"
    assert res["confidence"] >= 0.90
    print(f"  -> Hybrid Guard berhasil memproteksi boundary test: category={res['category_key']}, confidence={res['confidence']}")


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
    test_milestone4_pin_and_pan_credential_hygiene()
    test_milestone4_provider_authentication()
    test_milestone4_strict_merchant_binding()
    test_milestone4_transaction_status_enum_validation()
    test_milestone4_demopay_server_processing()
    test_milestone4_nominal_locking_and_anti_tamper()
    test_milestone4_demopay_multistatus_scenarios()
    test_milestone4_anti_replay_nonce_and_timestamp()
    test_milestone4_hybrid_nlp_guard()
    print("=" * 70)
    print("SELURUH 15 PENGUJIAN KEAMANAN ROADMAP MILESTONE 4 BERHASIL 100%!")
    print("=" * 70)
