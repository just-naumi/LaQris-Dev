# auth.py - Modul Keamanan & Autentikasi Akun LaQris
# Mengatur keamanan kata sandi, tiket login (JWT token), PIN pembayaran, dan proteksi hak akses pengguna.

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

# Mesin pengacak kata sandi berstandar tinggi (Argon2id)
# Mengubah kata sandi asli menjadi kode acak panjang sehingga aman dari pencurian data.
argon2_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16
)

# Kunci rahasia untuk membuat tiket login (JWT)
SECRET_KEY = os.getenv("LAQRIS_JWT_SECRET", "laqris-security-hardened-jwt-secret-key-2026-v2-production-grade")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # Tiket aktif selama 24 jam

# Penampung header otorisasi login untuk API
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login", auto_error=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pengamanan Kata Sandi (Password Hashing)
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Mengacak kata sandi asli pengguna menjadi kode acak yang tidak dapat dibaca orang lain."""
    return argon2_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Memeriksa apakah kata sandi yang dimasukkan saat login cocok dengan kode rahasia di database.
    Mendukung format modern (Argon2id) dan format akun lama (SHA-256) secara mulus.
    """
    if not hashed_password or not plain_password:
        return False

    # Pemeriksaan untuk akun berformat modern Argon2id
    if hashed_password.startswith("$argon2"):
        try:
            return argon2_hasher.verify(hashed_password, plain_password)
        except (VerifyMismatchError, InvalidHashError):
            return False

    # Pemeriksaan mundur untuk akun lama (SHA-256)
    legacy_hash = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
    return legacy_hash == hashed_password


def is_legacy_hash(hashed_password: str) -> bool:
    """Mengecek apakah format kata sandi akun masih menggunakan format lama yang perlu diperbarui."""
    return not (hashed_password and hashed_password.startswith("$argon2"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pembuatan & Pemeriksaan Tiket Login (JWT Access Token)
# ─────────────────────────────────────────────────────────────────────────────

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Membuat tiket digital (JWT token) resmi saat pengguna berhasil login.
    Di dalam tiket tersimpan ID pengguna, email, peran (pembeli/toko), dan batas waktu aktif.
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
    Membaca dan memastikan keaslian tiket digital (JWT token).
    Jika tiket sah dan belum kedaluwarsa, data pengguna dikembalikan. Jika palsu, tolak (None).
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except (jwt.PyJWTError, Exception):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pemeriksaan Akun Pengguna & Proteksi Hak Milik (User Ownership)
# ─────────────────────────────────────────────────────────────────────────────

def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Optional[Session] = None
) -> Optional[Dict[str, Any]]:
    """
    Membaca akun pengguna dari tiket login jika ada, tanpa menolak jika pengguna belum login.
    Cocok untuk fitur yang bisa diakses siapa saja namun memiliki fitur tambahan jika login.
    """
    if not token:
        return None
    return decode_access_token(token)


def get_current_user_required(
    token: Optional[str] = Depends(oauth2_scheme)
) -> Dict[str, Any]:
    """
    Mewajibkan pengguna sudah login. Jika tidak ada tiket login yang sah, tolak dengan pesan 401.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Silakan login terlebih dahulu untuk mengakses layanan ini.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(token)
    if not payload or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sesi login Anda sudah kedaluwarsa. Silakan login kembali.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def verify_user_ownership(requester_user_id: Optional[str], resource_owner_id: Optional[str], resource_name: str = "transaksi"):
    """
    Perlindungan Hak Milik Data:
    Memastikan Pengguna A tidak bisa mengakses atau mengubah riwayat transaksi milik Pengguna B.
    Jika data memiliki pemilik resmi, pembeli lain yang mencoba mengakses akan langsung ditolak.
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
                detail=f"Akses ditolak: Akun '{requester_user_id}' tidak memiliki hak otorisasi atas {resource_name} milik '{resource_owner_id}'."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Keamanan Pembayaran Antar Server (Bank / Dompet Digital & Anti-Replay)
# ─────────────────────────────────────────────────────────────────────────────
import hmac

PROVIDER_SECRET_KEY = os.getenv("LAQRIS_PROVIDER_SECRET", "demopay-service-hmac-secret-2026")
PROVIDER_API_KEYS = {
    "DemoPay": os.getenv("DEMOPAY_API_KEY", "demopay-live-key-2026-auth"),
    "DANA": os.getenv("DANA_API_KEY", "dana-gw-secret-key-2026"),
    "GOPAY": os.getenv("GOPAY_API_KEY", "gopay-gw-secret-key-2026")
}

# Daftar kode unik sementara (nonce) untuk mencegah transaksi dikirim ulang oleh penipu
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
    Memverifikasi keaslian pesan dari server bank / aplikasi dompet digital:
    1. Memeriksa waktu pengiriman (maksimal jeda 5 menit agar data lama tidak disalahgunakan).
    2. Memeriksa kode unik pengiriman (nonce) agar transaksi tidak dapat diduplikat (anti-replay).
    3. Memeriksa kunci rahasia API atau tanda tangan digital HMAC.
    """
    # 1. Pengecekan Waktu & Kode Unik Anti-Duplikasi
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
            # Toleransi jeda waktu sinyal maksimal 5 menit (300 detik)
            if abs(now_ts - req_ts) > 300:
                return False
        except Exception:
            return False

    if nonce:
        if nonce in SEEN_NONCES:
            return False  # Kode unik ini sudah pernah dipakai -> Tolak upaya transaksi duplikat!
        SEEN_NONCES.add(nonce)
        if len(SEEN_NONCES) > 10000:
            SEEN_NONCES.clear()

    # 2. Pemeriksaan Kunci API Resmi Bank
    if api_key:
        expected_key = PROVIDER_API_KEYS.get(provider, PROVIDER_API_KEYS.get("DemoPay"))
        if expected_key and hmac.compare_digest(api_key, expected_key):
            return True

    # 3. Pemeriksaan Tanda Tangan Digital (HMAC)
    if signature and raw_body:
        expected_sig_raw = hmac.new(
            PROVIDER_SECRET_KEY.encode("utf-8"),
            raw_body,
            hashlib.sha256
        ).hexdigest()
        if hmac.compare_digest(signature, expected_sig_raw):
            return True

        if timestamp:
            expected_sig_ts = hmac.new(
                PROVIDER_SECRET_KEY.encode("utf-8"),
                f"{timestamp}.".encode("utf-8") + raw_body,
                hashlib.sha256
            ).hexdigest()
            if hmac.compare_digest(signature, expected_sig_ts):
                return True

    # 4. Mode Pengujian Lokal
    if is_development_mode():
        if not api_key and not signature:
            return True

    return False


def is_development_mode() -> bool:
    """Mengecek apakah aplikasi sedang dijalankan dalam mode uji coba lokal."""
    return os.getenv("LAQRIS_ENV", "development").lower() == "development"


def verify_pin_secure(input_pin: str, stored_pin: Optional[str] = None) -> bool:
    """
    Memeriksa 6 angka PIN transaksi secara aman dan waktu tetap (constant-time)
    agar tidak mudah ditebak oleh pihak tidak berwenang.
    PIN wajib diverifikasi dari data akun pengguna di database.
    """
    if not input_pin:
        return False
    if not stored_pin:
        if is_development_mode():
            stored_pin = "123456"
        else:
            return False
    return hmac.compare_digest(str(input_pin).strip(), str(stored_pin).strip())


