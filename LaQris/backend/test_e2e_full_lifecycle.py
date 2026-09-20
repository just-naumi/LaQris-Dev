"""
test_e2e_full_lifecycle.py
Pengujian End-to-End Kedua Skenario Utama LaQris (Step 38 & 39 di Readme2.md):
- Demo A: Legitimate Payment Loop (Verify -> ALLOW -> Payment Event TX -> IndoBERT Feedback -> EMRS Recalculated)
- Demo B: Tampered Attack Loop (Verify -> DANGER / BLOCK -> Payment Prevented -> TX Binding Rejected)
"""

import os
import sys
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app
from database import get_db, reset_db
from models import VerificationSession, PaymentTransaction, Merchant

client = TestClient(app)

def test_demo_a_legitimate_full_loop():
    print("\n" + "=" * 65)
    print("DEMO A: SKENARIO TRANSAKSI ASLI & LEGITIMATE (ALLOW FLOW)")
    print("=" * 65)
    
    db = next(get_db())
    session_id = f"LQ-V-DEMO-A-{int(datetime.utcnow().timestamp())}"
    
    # 1. Mock Hasil Pre-Payment Verification (ALLOW)
    session = VerificationSession(
        session_id=session_id,
        user_id="USR-001928",
        nmid="ID1020000000001",
        digital_name="WARUNG MAKAN SEDAP",
        physical_name="WARUNG MAKAN SEDAP",
        scanned_at=datetime.utcnow(),
        status="MATCH",
        trust_score=100.0,
        risk_level="NORMAL",
        reputation_score=75.0,
        decision="ALLOW",
        reason_codes="[]",
        expires_at=datetime.utcnow() + timedelta(minutes=15),
        is_bound=False
    )
    db.add(session)
    db.commit()

    # Step 1: DemoPay memanggil Contract A (/api/v1/verify)
    res_verify = client.post("/api/v1/verify", json={"session_id": session_id})
    assert res_verify.status_code == 200
    v_data = res_verify.json()
    assert v_data["decision"] == "ALLOW"
    assert v_data["can_proceed_payment"] is True
    print(f"[1] Pre-Payment Verify: Keputusan={v_data['decision']}, Risk={v_data['risk_level']} (Pembayaran Diizinkan)")

    # Step 2: Payment Berhasil -> Kirim Contract B (/api/v1/transactions/events)
    tx_id = f"TX-DEMO-A-{int(datetime.utcnow().timestamp())}"
    tx_payload = {
        "verification_session_id": session_id,
        "provider": "DemoPay",
        "provider_transaction_id": tx_id,
        "amount": 35000.0,
        "status": "SUCCESS",
        "response_code": "00",
        "user_id": "USR-001928",
        "card_number": "4111222233334444", # Field sensitif untuk uji sanitasi
        "cvv": "999"
    }
    res_tx = client.post("/api/v1/transactions/events", json=tx_payload)
    assert res_tx.status_code == 200
    tx_data = res_tx.json()
    assert tx_data["is_bound"] is True
    print(f"[2] Post-Payment Event: Transaksi={tx_data['transaction_id']}, Status={tx_data['status']}, Sesi Bound={tx_data['is_bound']}")

    # Step 3: User Memberikan Ulasan Naratif -> Contract C (/api/v1/feedback)
    comment_text = "Pelayanan toko cepat dan kasir ramah, transaksi QRIS langsung masuk sesuai nama toko."
    fb_payload = {
        "verification_id": session_id,
        "transaction_id": tx_id,
        "comment": comment_text,
        "user_id": "USR-001928"
    }
    res_fb = client.post("/api/v1/feedback", json=fb_payload)
    assert res_fb.status_code == 200
    fb_data = res_fb.json()
    assert fb_data["evidence_level"] == 2 # Level 2 karena terikat transaksi resmi
    assert fb_data["merchant_reputation_updated"] is True
    print(f"[3] Post-Payment Feedback (IndoBERT AI):")
    print(f"    - Event Type: {fb_data['event_type']} ({fb_data['category_title']})")
    print(f"    - Tingkat Bukti: Level {fb_data['evidence_level']} (Transaksi Terverifikasi)")
    print(f"    - Reputasi Toko: {fb_data['previous_reputation_score']} -> {fb_data['new_reputation_score']}")
    print("DEMO A BERHASIL DENGAN SEMPURNA!")


def test_demo_b_tampered_attack_loop():
    print("\n" + "=" * 65)
    print("DEMO B: SKENARIO SERANGAN QRIS TAMPERED / PALSU (BLOCK FLOW)")
    print("=" * 65)
    
    db = next(get_db())
    session_id = f"LQ-V-DEMO-B-{int(datetime.utcnow().timestamp())}"
    
    # Mock Hasil Pre-Payment Verification (DANGER / BLOCK akibat stiker ditimpa)
    session = VerificationSession(
        session_id=session_id,
        user_id="USR-001928",
        nmid="ID1020000000999", # Mismatch
        digital_name="PENIPU STIKER QRIS",
        physical_name="WARUNG MAKAN ASLI",
        scanned_at=datetime.utcnow(),
        status="MISMATCH",
        trust_score=10.0,
        risk_level="DANGER",
        reputation_score=20.0,
        decision="BLOCK",
        reason_codes='["NMID_MISMATCH", "STICKER_TAMPERED"]',
        expires_at=datetime.utcnow() + timedelta(minutes=15),
        is_bound=False
    )
    db.add(session)
    db.commit()

    # Step 1: DemoPay memanggil Contract A (/api/v1/verify)
    res_verify = client.post("/api/v1/verify", json={"session_id": session_id})
    assert res_verify.status_code == 200
    v_data = res_verify.json()
    assert v_data["decision"] == "BLOCK"
    assert v_data["can_proceed_payment"] is False
    print(f"[1] Pre-Payment Verify: Keputusan={v_data['decision']}, Risk={v_data['risk_level']}")
    print(f"    - Reason Codes: {v_data['reason_codes']}")
    print("    - INTERVENSI KEAMANAN: Payment Provider membatalkan otorisasi pembayaran!")

    # Step 2: Jika penipu mencoba memanggil API event transaksi secara ilegal -> Ditolak HTTP 403
    tx_payload = {
        "verification_session_id": session_id,
        "provider": "DemoPay",
        "provider_transaction_id": f"TX-ILLEGAL-{int(datetime.utcnow().timestamp())}",
        "amount": 100000.0,
        "status": "SUCCESS"
    }
    res_tx = client.post("/api/v1/transactions/events", json=tx_payload)
    assert res_tx.status_code == 403
    print(f"[2] Enforcement DB Transaction Layer:")
    print(f"    - Hasil: HTTP 403 Forbidden - {res_tx.json()['detail']}")
    print("DEMO B BERHASIL: Serangan berhasil diblokir sebelum transaksi terjadi!")


if __name__ == "__main__":
    test_demo_a_legitimate_full_loop()
    test_demo_b_tampered_attack_loop()
    print("\n" + "=" * 65)
    print("SEMUA SKENARIO END-TO-END BERHASIL 100% SESUAI ROADMAP README2.MD!")
    print("=" * 65)
