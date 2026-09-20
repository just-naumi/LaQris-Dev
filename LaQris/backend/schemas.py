from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Modul Skema Data LaQris (Pydantic)
# Berfungsi untuk mendefinisikan format data yang dikirim dan diterima oleh aplikasi.
# Memastikan setiap data (seperti scan QRIS, pembayaran, feedback, dan login)
# memiliki format yang benar dan aman sebelum diproses.


class TransactionStatusEnum(str, Enum):
    """Status resmi hasil pembayaran dari bank/penyedia dompet digital."""
    SUCCESS = "SUCCESS"      # Pembayaran berhasil
    FAILED = "FAILED"        # Pembayaran gagal
    TIMEOUT = "TIMEOUT"      # Waktu transaksi habis
    CANCELLED = "CANCELLED"  # Dibatalkan oleh pengguna
    REVERSED = "REVERSED"    # Dana dikembalikan


# ─────────────────────────────────────────────────────────────
# 1. Skema Data Laporan dan Sengketa Toko
# ─────────────────────────────────────────────────────────────

class ReportSchema(BaseModel):
    id: int
    category: str
    severity: str
    description: Optional[str] = None
    evidence_level: int
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


class DisputeSchema(BaseModel):
    id: int
    description: Optional[str] = None
    severity: str
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────
# 2. Skema Riwayat Pengamatan QRIS Toko
# Dikumpulkan dari riwayat scan para pengguna LaQris untuk melihat
# konsistensi dan rekam jejak toko dari waktu ke waktu.
# ─────────────────────────────────────────────────────────────

class ObservationHistorySchema(BaseModel):
    total_observations: int           # Berapa kali QRIS toko ini pernah di-scan oleh pengguna
    unique_observers: int             # Berapa orang unik yang pernah melakukan scan
    first_observed: Optional[str]     # Tanggal pertama kali QRIS toko ini dipindai
    last_observed: Optional[str]      # Tanggal terakhir kali QRIS toko ini dipindai
    identity_match: int               # Jumlah scan di mana nama stiker fisik cocok dengan data QR
    identity_mismatch: int            # Jumlah scan yang nama fisiknya mencurigakan / berbeda
    physical_anomaly: int             # Jumlah scan dengan stiker tumpukan atau anomali fisik
    identity_match_rate: float        # Persentase kecocokan nama (0% - 100%)
    complaint_rate: Optional[float]   # Persentase keluhan dari total scan (%)
    verified_feedback: int            # Total ulasan yang terbukti sah
    complaints: int                   # Jumlah laporan keluhan
    disputes: int                     # Jumlah sengketa transaksi


# ─────────────────────────────────────────────────────────────
# 3. Komponen Nilai Reputasi Toko (EMRS)
# Nilai 0 - 100 yang menilai kelayakan dan keamanan sebuah toko.
# ─────────────────────────────────────────────────────────────

class EMRSComponents(BaseModel):
    A: float                          # Nilai Keaslian & Kesesuaian Nama Toko (bobot 40%)
    C: float                          # Nilai Bebas Keluhan (bobot 30%)
    D: float                          # Nilai Bebas Sengketa Pembayaran (bobot 20%)
    L: float                          # Nilai Masa Aktif & Pengalaman Toko (bobot 10%)
    T_observed: Optional[float] = None


class ReputationScoreSchema(BaseModel):
    reputation_score: Optional[float]          # Skor akhir 0 - 100 (kosong jika toko baru pertama kali ditemukan)
    grade: str                         # Predikat toko: "Sangat Baik", "Baik", "Cukup", atau "Buruk"
    confidence_level: str              # Tingkat keyakinan data: TINGGI, SEDANG, atau RENDAH
    confidence_score: float            # Angka persentase keyakinan (0 - 100%)
    data_sufficiency_status: str       # Apakah data historis sudah mencukupi atau masih baru
    components: EMRSComponents
    evidence_quality: str              # Kualitas bukti laporan
    total_evidence_count: int
    found_in_db: bool
    nmid: Optional[str] = None
    merchant_name: Optional[str] = None
    registered_at: Optional[datetime] = None
    first_seen_observed: Optional[str] = None
    last_seen_observed: Optional[str] = None
    observation_history: Optional[ObservationHistorySchema] = None


# ─────────────────────────────────────────────────────────────
# 4. Skema Analisis Kode QRIS dan Bukti Identitas
# ─────────────────────────────────────────────────────────────

class QRISRawAnalysisSchema(BaseModel):
    point_of_initiation: str
    initiation_type_code: str
    mcc_code: str
    mcc_category: str
    nmid_parsed: Dict[str, Any]
    currency: str
    crc_checksum: Optional[str] = None


class IdentityFieldEvidenceSchema(BaseModel):
    physical: Optional[str] = None
    digital: Optional[str] = None
    match: bool = False
    similarity: Optional[float] = None
    priority: str = "SUPPORTING_SIGNAL"


class IdentityEvidenceSchema(BaseModel):
    nmid: IdentityFieldEvidenceSchema
    merchant_name: IdentityFieldEvidenceSchema
    acquirer: IdentityFieldEvidenceSchema
    tid: IdentityFieldEvidenceSchema


# ─────────────────────────────────────────────────────────────
# 5. Skema Tingkat Risiko Scan QRIS Saat Ini
# Menentukan keputusan apakah pembayaran: AMAN (ALLOW),
# WASPADA (WARN), atau DIBLOKIR KARENA BAHAYA (BLOCK).
# ─────────────────────────────────────────────────────────────

class CurrentQRRiskSchema(BaseModel):
    risk_level: str                    # Tingkat risiko: "NORMAL", "CAUTION", "WARNING", "DANGER"
    risk_label: str                    # Keterangan yang mudah dibaca pengguna
    risk_color: str                    # Warna indikator: hijau, kuning, oranye, merah
    overall_risk_score: float
    trust_score: float
    decision: str = "ALLOW"            # Keputusan sistem: "ALLOW", "WARN", atau "BLOCK"
    reason_codes: List[str] = []       # Kode alasan (contoh: NMID_MISMATCH jika nama beda)
    is_mismatch: bool
    name_similarity: float
    match_level: str
    explanation: str                   # Pesan penjelasan ramah untuk pengguna
    physical_merchant: str
    digital_merchant: str
    digital_city: str
    physical_nmid: str
    digital_nmid: str
    physical_acquirer: str
    digital_acquirer: str
    physical_tid: str
    digital_tid: str
    technical_info: Dict[str, Any]
    qris_raw_analysis: Optional[QRISRawAnalysisSchema] = None
    identity_evidence: Optional[Dict[str, Any]] = None


class ScanResponseSchema(BaseModel):
    session_id: str
    current_qr_risk: CurrentQRRiskSchema
    merchant_reputation: ReputationScoreSchema
    visualization_url: str


# ─────────────────────────────────────────────────────────────
# 6. Skema Verifikasi Pra-Pembayaran (Sebelum Saldo Dipotong)
# Digunakan oleh aplikasi pembayaran untuk mengecek keaslian QRIS.
# ─────────────────────────────────────────────────────────────

class VerifyRequestSchema(BaseModel):
    session_id: Optional[str] = None
    verification_id: Optional[str] = None
    provider: str = "DemoPay"
    qr_payload: Optional[str] = None
    user_id: Optional[str] = None
    amount: Optional[float] = None


class VerifyMerchantInfoSchema(BaseModel):
    digital_name: Optional[str] = None
    physical_name: Optional[str] = None
    nmid: Optional[str] = None
    acquirer: Optional[str] = None
    terminal_id: Optional[str] = None


class VerifyResponseSchema(BaseModel):
    verification_id: str
    decision: str                      # "ALLOW" | "WARN" | "BLOCK"
    risk_level: str                    # "NORMAL" | "CAUTION" | "WARNING" | "DANGER"
    risk_score: float
    trust_score: float
    reason_codes: List[str] = []
    merchant_info: VerifyMerchantInfoSchema
    can_proceed_payment: bool          # True jika ALLOW atau WARN, False jika BLOCK
    is_bound: bool = False
    expires_at: Optional[datetime] = None
    message: str = "Verifikasi pre-payment berhasil."


# ─────────────────────────────────────────────────────────────
# 7. Skema Detail Sesi Pemindaian QRIS
# ─────────────────────────────────────────────────────────────

class VerificationSessionDetailSchema(BaseModel):
    id: int
    session_id: str
    user_id: Optional[str] = None
    nmid: Optional[str] = None
    digital_name: Optional[str] = None
    physical_name: Optional[str] = None
    scanned_at: datetime
    status: str
    trust_score: float
    risk_level: str
    reputation_score: float
    decision: str = "ALLOW"
    reason_codes: List[str] = []
    expires_at: Optional[datetime] = None
    is_bound: bool = False
    bound_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────
# 8. Skema Pencatatan Bukti Transaksi Pembayaran
# Menerima laporan sukses/gagal dari bank/dompet digital setelah bayar.
# ─────────────────────────────────────────────────────────────

class PaymentTransactionSchema(BaseModel):
    id: int
    verification_session_id: Optional[str] = None
    provider: str
    provider_transaction_id: str
    merchant_id: Optional[int] = None
    nmid: Optional[str] = None
    user_id: Optional[str] = None
    amount: float
    status: str
    response_code: str
    invoice_number: Optional[str] = None
    terminal_id: Optional[str] = None
    transaction_time: Optional[datetime] = None
    latency_ms: int = 0
    retry_count: int = 0
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PaymentTransactionCreateSchema(BaseModel):
    verification_id: Optional[str] = None
    verification_session_id: Optional[str] = None
    provider: str = "DemoPay"
    provider_transaction_id: str
    amount: float
    status: TransactionStatusEnum        # Status transaksi (contoh: SUCCESS)
    response_code: str                   # Kode respon bank (contoh: "00" jika berhasil)
    merchant_id: Optional[Any] = None
    merchant_name: Optional[str] = None
    nmid: Optional[str] = None
    user_id: Optional[str] = None
    terminal_id: Optional[str] = None
    invoice_number: Optional[str] = None
    transaction_time: Optional[str] = None
    latency_ms: int = 0
    retry_count: int = 0


class PaymentTransactionResponseSchema(BaseModel):
    success: bool
    message: str
    transaction_id: str
    verification_id: str
    status: str
    amount: float
    is_bound: bool = True
    merchant_reputation_impact: Optional[Dict[str, Any]] = None


# Alias nama skema transaksi untuk kemudahan pemanggilan
TransactionEventCreateSchema = PaymentTransactionCreateSchema
TransactionEventResponseSchema = PaymentTransactionResponseSchema


# ─────────────────────────────────────────────────────────────
# 9. Skema Proses Pembayaran Dompet Digital DemoPay
# Digunakan saat pengguna memasukkan PIN dan menekan tombol Bayar.
# ─────────────────────────────────────────────────────────────

class DemoPayProcessPaymentSchema(BaseModel):
    session_id: str
    amount: float
    pin: str
    terminal_id: Optional[str] = "A01"
    user_id: Optional[str] = None
    account_number: Optional[str] = None
    scenario: Optional[str] = "SUCCESS"  # Pilihan uji coba: "SUCCESS", "FAILED", "TIMEOUT", "CANCELLED"


class DemoPayProcessPaymentResponseSchema(BaseModel):
    success: bool
    message: str
    transaction_id: str
    invoice_number: str
    terminal_id: str
    amount: float
    status: str
    response_code: str
    transaction_time: datetime
    latency_ms: int
    retry_count: int = 0
    session_id: str
    nmid: Optional[str] = None
    merchant_name: Optional[str] = None
    acquirer: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# 10. Skema Kunci Nominal Pembayaran
# Memastikan nominal yang dibayar cocok persis dengan sesi scan.
# ─────────────────────────────────────────────────────────────

class PaymentIntentSchema(BaseModel):
    session_id: str
    amount: float


class PaymentIntentResponseSchema(BaseModel):
    success: bool
    message: str
    session_id: str
    amount: float
    amount_locked: bool


# ─────────────────────────────────────────────────────────────
# 11. Skema Informasi Lengkap Profil Toko
# ─────────────────────────────────────────────────────────────

class MerchantSchema(BaseModel):
    id: int
    nmid: str
    merchant_name: str
    acquirer: str
    rating: float
    total_reports: int
    verified_reports: int
    verified_transactions: int
    successful_transactions: int
    identity_match_count: int
    identity_mismatch_count: int
    reputation_score: float
    registered_at: datetime
    reports: List[ReportSchema] = []
    disputes: List[DisputeSchema] = []

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────
# 12. Skema Pengiriman Ulasan / Feedback Pembeli
# Menerima ulasan pengalaman berbelanja untuk dianalisis oleh AI.
# ─────────────────────────────────────────────────────────────

class FeedbackSubmitSchema(BaseModel):
    nmid: str
    category: str           # Kategori keluhan (misal: penipuan stiker atau salah nama)
    severity: str = "MEDIUM"# Tingkat keparahan: "LOW", "MEDIUM", "HIGH", "CRITICAL"
    description: Optional[str] = None
    transaction_ref: Optional[str] = None
    has_evidence: bool = False


class FeedbackResponseSchema(BaseModel):
    success: bool
    message: str
    evidence_level: int
    new_reputation_score: float
    detected_category: Optional[str] = None
    detected_category_key: Optional[str] = None
    confidence: Optional[float] = None
    severity: Optional[str] = None
    action: Optional[str] = None


class FeedbackContractCSchema(BaseModel):
    """
    Format pengiriman ulasan setelah transaksi berhasil.
    Pengguna dapat menceritakan pengalamannya tanpa perlu menghafal kategori teknis.
    """
    verification_id: Optional[str] = None      # Nomor verifikasi scan QRIS
    transaction_id: Optional[str] = None       # Nomor transaksi pembayaran
    comment: str                               # Teks ulasan atau keluhan dari pengguna
    nmid: Optional[str] = None                 # Nomor identitas QRIS toko
    manual_category: Optional[str] = None      # Kategori manual jika dipilih pengguna
    has_evidence: bool = False                 # Apakah menyertakan foto bukti fisik
    user_id: Optional[str] = None              # Identitas akun pengguna yang memberi ulasan


class FeedbackContractCResponseSchema(BaseModel):
    """
    Format balasan setelah ulasan dibaca oleh AI IndoBERT.
    Menampilkan kategori masalah yang terdeteksi dan pembaruan reputasi toko.
    """
    success: bool = True
    message: str
    event_type: str                            # Kategori masalah yang terdeteksi AI
    category_title: str                        # Judul masalah dalam bahasa yang ramah pengguna
    severity: str                              # Tingkat keparahan ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    evidence_level: int                        # Tingkat keabsahan bukti (0: hanya teks, 1: ada scan, 2: ada transaksi resmi)
    confidence: float                          # Tingkat keyakinan AI (0.0 - 1.0)
    merchant_reputation_updated: bool = True
    previous_reputation_score: Optional[float] = None
    new_reputation_score: float
    recommended_action: Optional[str] = None
    model_used: Optional[str] = "IndoBERT-FineTuned-EMRS"
    verification_id: Optional[str] = None
    transaction_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# 13. Skema Pendaftaran dan Masuk Akun Pengguna
# ─────────────────────────────────────────────────────────────

class UserRegisterSchema(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    role: str = "PENGGUNA"  # "PENGGUNA" | "MERCHANT"


class UserLoginSchema(BaseModel):
    email: str
    password: str


class UserResponseSchema(BaseModel):
    id: int
    user_id: str
    username: str
    email: str
    status: str
    full_name: Optional[str] = None
    role: str
    account_number: Optional[str] = "1858868768"
    account_type: Optional[str] = "TAPLUS"
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponseSchema(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponseSchema
