"""
adapters.py - Provider Payload Normalizer & Canonical Adapter (P1 Item 9 di Readme.md)
Mendukung normalisasi payload transaksi dari beragam payment provider (DemoPay, DANA, GoPay, dll.)
menjadi format Canonical Transaction Event LaQris yang aman, tervalidasi, dan tersanitasi.
"""

import re
from datetime import datetime
from typing import Dict, Any, Optional

SENSITIVE_FIELDS = {
    "card_number", "pan", "full_pan", "cvv", "cvc", "pin", "password",
    "auth_token", "secret", "card_expiry", "private_key"
}

STATUS_MAPPING = {
    "00": "SUCCESS",
    "0": "SUCCESS",
    "SUCCESS": "SUCCESS",
    "PAID": "SUCCESS",
    "SETTLEMENT": "SUCCESS",
    "05": "FAILED",
    "FAILED": "FAILED",
    "DENIED": "FAILED",
    "REJECTED": "FAILED",
    "DECLINED": "FAILED",
    "68": "TIMEOUT",
    "TIMEOUT": "TIMEOUT",
    "EXPIRED": "TIMEOUT",
    "99": "CANCELLED",
    "CANCELLED": "CANCELLED",
    "CANCELED": "CANCELLED",
    "REVERSED": "REVERSED"
}


def sanitize_raw_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Menghapus seluruh field sensitif pembayaran (P1 Item 9 & Step 15)."""
    clean = {}
    for k, v in payload.items():
        if k.lower() in SENSITIVE_FIELDS:
            continue
        clean[k] = v
    return clean


def normalize_provider_transaction(
    raw_payload: Dict[str, Any],
    provider_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Mengonversi payload heterogen dari berbagai payment provider menjadi Canonical Event LaQris.
    """
    clean_dict = sanitize_raw_payload(raw_payload)
    
    # 1. Resolve Provider
    detected_provider = provider_name or clean_dict.get("provider") or clean_dict.get("issuer") or "DemoPay"

    # 2. Resolve Verification Session ID
    v_id = (
        clean_dict.get("verification_session_id") or
        clean_dict.get("verification_id") or
        clean_dict.get("session_id") or
        clean_dict.get("v_id")
    )

    # 3. Resolve Provider Transaction ID
    tx_id = (
        clean_dict.get("provider_transaction_id") or
        clean_dict.get("transaction_id") or
        clean_dict.get("tx_id") or
        clean_dict.get("ref_id") or
        clean_dict.get("partner_reference_no")
    )

    # 4. Resolve Amount
    raw_amount = clean_dict.get("amount") or clean_dict.get("total_amount") or 0.0
    try:
        amount = float(raw_amount)
    except (ValueError, TypeError):
        amount = 0.0

    # 5. Resolve & Canonicalize Status and Response Code
    raw_status = str(clean_dict.get("status") or "SUCCESS").upper().strip()
    raw_rc = str(clean_dict.get("response_code") or clean_dict.get("rc") or "00").strip()
    canonical_status = STATUS_MAPPING.get(raw_status, STATUS_MAPPING.get(raw_rc, "SUCCESS"))

    # 6. Resolve Transaction Time
    tx_time_raw = (
        clean_dict.get("transaction_time") or
        clean_dict.get("time") or
        clean_dict.get("created_at") or
        clean_dict.get("timestamp")
    )
    if isinstance(tx_time_raw, str):
        try:
            tx_time = datetime.fromisoformat(tx_time_raw.replace("Z", "+00:00"))
        except Exception:
            tx_time = datetime.utcnow()
    elif isinstance(tx_time_raw, datetime):
        tx_time = tx_time_raw
    else:
        tx_time = datetime.utcnow()

    # 7. Construct Canonical Event
    return {
        "verification_session_id": v_id,
        "provider": detected_provider,
        "provider_transaction_id": tx_id,
        "merchant_id": clean_dict.get("merchant_id"),
        "merchant_name": clean_dict.get("merchant_name") or clean_dict.get("digital_name"),
        "nmid": clean_dict.get("nmid"),
        "user_id": clean_dict.get("user_id"),
        "amount": amount,
        "status": canonical_status,
        "response_code": raw_rc,
        "invoice_number": clean_dict.get("invoice_number") or clean_dict.get("invoice_id"),
        "terminal_id": clean_dict.get("terminal_id") or "A01",
        "transaction_time": tx_time,
        "latency_ms": int(clean_dict.get("latency_ms") or 0),
        "retry_count": int(clean_dict.get("retry_count") or 0)
    }
