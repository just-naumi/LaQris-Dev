"""
auth.py - LaQris Security Hardening Module (Milestone 4 / Step 28 & 29)
Mengimplementasikan:
1. Argon2id Password Hashing dengan verifikasi mundur kompatibel (SHA-256 migration).
2. JWT Access Token (HS256) untuk autentikasi stateless yang aman.
3. User Ownership Enforcement (mencegah feedback/transaksi unauthorized antar pengguna).
"""

import os
import hashlib
from datetime import datetime, timedelta
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
    Enforce Step 29 di Readme2.md (User Ownership Rule):
    User A tidak dapat mengirim feedback atau memodifikasi transaksi/sesi milik User B.
    """
    if not requester_user_id or not resource_owner_id:
        return  # Jika salah satu tidak terikat user, izinkan dengan fallback

    if requester_user_id != resource_owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User ownership violation: Akun '{requester_user_id}' tidak memiliki hak otorisasi atas {resource_name} milik '{resource_owner_id}'."
        )
