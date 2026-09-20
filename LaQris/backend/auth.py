"""
auth.py - LaQris Security Hardening Module (Milestone 4 / Step 28 & 29)
Mengimplementasikan:
1. Argon2id Password Hashing dengan verifikasi mundur kompatibel (SHA-256 migration).
2. JWT Access Token (HS256) untuk autentikasi stateless yang aman.
3. User Ownership Enforcement (mencegah feedback/transaksi unauthorized antar pengguna).
"""

import os
import time
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

# Inisialisasi Password Hasher Argon2id (rekomendasi OWASP)
argon2_hasher = PasswordHasher(
    time_cost=3,        # 3 iterasi
    memory_cost=65536,  # 64 MB
    parallelism=4,      # 4 thread paralel
    hash_len=32,
    salt_len=16
)

# Konfigurasi JWT Secret & Expiry
SECRET_KEY = os.getenv("LAQRIS_JWT_SECRET", "laqris-security-hardened-jwt-secret-key-2026-v2-production-grade")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 Jam

# OAuth2 Scheme untuk FastAPI Swagger UI & Bearer extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login", auto_error=False)


# ─────────────────────────────────────────────────────────────────────────────
# Password Hashing & Verification (Argon2id + Legacy SHA-256 Migration)
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Menghasilkan hash password Argon2id dengan salt kriptografis unik."""
    return argon2_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Memvalidasi kecocokan password.
    Mendukung format Argon2id ($argon2id$...) dan backward-compatibility SHA-256
    agar user existing/seed data tidak terkunci.
    """
    if not hashed_password or not plain_password:
        return False

    # Format Argon2id
    if hashed_password.startswith("$argon2"):
        try:
            return argon2_hasher.verify(hashed_password, plain_password)
        except (VerifyMismatchError, InvalidHashError):
            return False

    # Format Legacy SHA-256 (64 hex characters)
    legacy_hash = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
    return legacy_hash == hashed_password


def is_legacy_hash(hashed_password: str) -> bool:
    """Mengecek apakah hash masih menggunakan algoritma lama (SHA-256) yang butuh rehash."""
    return not (hashed_password and hashed_password.startswith("$argon2"))


# ─────────────────────────────────────────────────────────────────────────────
# JWT Token Generation & Verification
# ─────────────────────────────────────────────────────────────────────────────

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Membuat token JWT terenkripsi dengan klaim sub (user_id), email, role,
    serta expiry timestamp (iat & exp).
    """
    to_encode = data.copy()
    now = datetime.utcnow()
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    
    to_encode.update({
        "iat": now,
        "exp": expire,
        "iss": "LaQris-Auth-Service"
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Membedah dan memvalidasi integritas JWT token.
    Mengembalikan payload dictionary jika valid, atau None jika expired/tampered.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except (jwt.PyJWTError, Exception):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Dependencies & User Ownership
# ─────────────────────────────────────────────────────────────────────────────

def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Optional[Session] = None
) -> Optional[Dict[str, Any]]:
    """
    Ekstrak data pengguna dari Bearer token tanpa melempar error 401 jika anonim.
    Sangat berguna untuk endpoint hybrid (bisa anonim atau terotentikasi).
    """
    if not token:
        return None
    return decode_access_token(token)


def get_current_user_required(
    token: Optional[str] = Depends(oauth2_scheme)
) -> Dict[str, Any]:
    """
    Wajibkan Bearer token valid. Melempar HTTP 401 jika token hilang atau kedaluwarsa.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token otentikasi Bearer diperlukan untuk mengakses endpoint ini.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(token)
    if not payload or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token otentikasi tidak valid atau sudah kedaluwarsa.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def verify_user_ownership(requester_user_id: Optional[str], resource_owner_id: Optional[str], resource_name: str = "transaksi"):
    """
    Enforce Step 29 di Readme2.md & P1 Item 8 di Readme.md (Strict User Ownership Rule):
    User A tidak dapat mengakses/memodifikasi transaksi/sesi milik User B.
    Jika resource memiliki pemilik, requester WAJIB terotentikasi (fail closed).
    """
    if resource_owner_id:
        if not requester_user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Autentikasi diperlukan: Akses atas {resource_name} memerlukan akun pengguna terverifikasi."
            )
        if requester_user_id != resource_owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User ownership violation: Akun '{requester_user_id}' tidak memiliki hak otorisasi atas {resource_name} milik '{resource_owner_id}'."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Step 25 di Readme2.md & P1 Item 7: Payment Provider S2S Auth + Anti-Replay
# ─────────────────────────────────────────────────────────────────────────────
import hmac

PROVIDER_SECRET_KEY = os.getenv("LAQRIS_PROVIDER_SECRET", "demopay-service-hmac-secret-2026")
PROVIDER_API_KEYS = {
    "DemoPay": os.getenv("DEMOPAY_API_KEY", "demopay-live-key-2026-auth"),
    "DANA": os.getenv("DANA_API_KEY", "dana-gw-secret-key-2026"),
    "GOPAY": os.getenv("GOPAY_API_KEY", "gopay-gw-secret-key-2026")
}

# Cache in-memory untuk melacak nonce guna mencegah serangan replay (P1 Item 7)
SEEN_NONCES = set()

def verify_provider_authentication(
    provider: str,
    api_key: Optional[str] = None,
    signature: Optional[str] = None,
    raw_body: Optional[bytes] = None,
    timestamp: Optional[str] = None,
    nonce: Optional[str] = None
) -> bool:
    """
    Verifikasi service-to-service autentikasi payment provider (Step 25 / P1 Item 7).
    Mendukung X-Provider-Api-Key atau X-LaQris-Signature (HMAC-SHA256) dengan
    proteksi anti-replay callback (X-LaQris-Timestamp & X-LaQris-Nonce).
    """
    # 1. Anti-Replay Check (Timestamp freshness & Nonce uniqueness)
    if timestamp:
        try:
            if str(timestamp).replace(".", "", 1).isdigit():
                req_ts = float(timestamp)
            else:
                clean_ts = str(timestamp).replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean_ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                req_ts = dt.timestamp()
            now_ts = time.time()
            # Toleransi jeda waktu jaringan maksimal 5 menit (300 detik)
            if abs(now_ts - req_ts) > 300:
                return False
        except Exception:
            return False

    if nonce:
        if nonce in SEEN_NONCES:
            return False  # Nonce sudah pernah dipakai -> REPLAY ATTACK TERTOLAK!
        SEEN_NONCES.add(nonce)
        if len(SEEN_NONCES) > 10000:
            SEEN_NONCES.clear()

    # 2. Verifikasi Kredensial Provider
    if api_key:
        expected_key = PROVIDER_API_KEYS.get(provider, PROVIDER_API_KEYS.get("DemoPay"))
        if expected_key and hmac.compare_digest(api_key, expected_key):
            return True

    if signature and raw_body:
        # A. Cek kecocokan HMAC langsung terhadap raw_body
        expected_sig_raw = hmac.new(
            PROVIDER_SECRET_KEY.encode("utf-8"),
            raw_body,
            hashlib.sha256
        ).hexdigest()
        if hmac.compare_digest(signature, expected_sig_raw):
            return True

        # B. Cek kecocokan HMAC dengan format timestamp + "." + raw_body
        if timestamp:
            expected_sig_ts = hmac.new(
                PROVIDER_SECRET_KEY.encode("utf-8"),
                f"{timestamp}.".encode("utf-8") + raw_body,
                hashlib.sha256
            ).hexdigest()
            if hmac.compare_digest(signature, expected_sig_ts):
                return True

    # 3. Fail-Closed: Di mode non-development, tolak seluruh request tanpa auth valid
    if is_development_mode():
        # Hanya izinkan jika request sama sekali tidak membawa header auth di mode lokal
        if not api_key and not signature:
            return True

    return False


def is_development_mode() -> bool:
    """Mengecek apakah sistem berjalan dalam mode development lokal."""
    return os.getenv("LAQRIS_ENV", "development").lower() == "development"


def verify_pin_secure(input_pin: str, stored_pin: Optional[str] = None) -> bool:
    """
    Verifikasi PIN transaksi secara aman dengan constant-time comparison (P0 Item 1.B).
    Mencegah timing attack. PIN wajib diverifikasi dari database user;
    di mode local development, jika stored_pin belum diset maka fallback ke '123456'.
    """
    if not input_pin:
        return False
    if not stored_pin:
        if is_development_mode():
            stored_pin = "123456"
        else:
            return False
    return hmac.compare_digest(str(input_pin).strip(), str(stored_pin).strip())


