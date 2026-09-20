Bro, saya cek ulang repository LaQris sekarang. Repo publiknya sudah 102 commits dan core-nya sudah mencakup physical identity extraction, EMVCo parsing, identity matching, EMRS, serta layer payment. Commit 919f036 juga sudah menambahkan server-side DemoPay, pengukuran latency, session binding, dan HMAC-SHA256 untuk transaction event. Jadi posisi kita sekarang bukan lagi “bangun fitur utama”, tetapi hardening + memastikan seluruh lifecycle benar-benar nyambung + menyiapkan bukti PoC.

1. Prioritas terbesar: amankan payment lifecycle

Ini menurut saya P0 — harus selesai 20–21 September.

A. DemoPay masih belum benar-benar mengikat transaksi ke user yang sedang login

Pada endpoint POST /api/v1/demopay/process-payment, schema masih menerima user_id dari payload, dan implementasinya bahkan memiliki fallback:

payload.user_id
→ session.user_id
→ "USR-001928"

Endpoint tersebut juga belum terlihat menggunakan dependency get_current_user_required(). Jadi secara konsep, browser masih bisa mengirim identitas user sendiri.

Revisi:

JWT user
↓
DemoPay endpoint
↓
current_user.sub
↓
session.user_id
↓
harus sama

Jangan lagi:

target_user_id = payload.user_id or session.user_id or "USR-001928"

Menjadi konsep:

current_user = auth.get_current_user_required(...)
target_user_id = current_user["sub"]

if session.user_id != target_user_id:
raise HTTPException(403)

Untuk POC, user_id boleh tetap ada di schema kalau diperlukan compatibility, tetapi jangan dipercaya dari client.

B. PIN masih memiliki fallback 123456

Di auth.py, verify_pin_secure() masih menggunakan:

target = stored_pin if stored_pin else "123456"

Artinya kalau stored_pin tidak dikirim, PIN valid secara default adalah 123456. Untuk demo lokal memang praktis, tetapi untuk proyek fintech-security ini akan sangat mudah dipertanyakan juri.

Revisi:

PIN harus berasal dari user yang sudah terautentikasi, bukan default global.

Minimal:

JWT → User → stored PIN verifier → compare

Dan jangan menyimpan PIN plaintext.

C. Secret masih hardcoded di auth.py

Saat ini ada default untuk:

LAQRIS_JWT_SECRET
LAQRIS_PROVIDER_SECRET
DEMOPAY_API_KEY
DANA_API_KEY
GOPAY_API_KEY

Selain itu, verify_provider_authentication() masih mengizinkan request tanpa kredensial ketika:

LAQRIS_ENV=development

karena terdapat development bypass.

Untuk submission hackathon, ini salah satu hal yang jangan sampai kelihatan saat security review.

Target final:

.env / environment
↓
secret
↓
backend

Tidak ada fallback secret production.

Development bypass boleh ada, tetapi:

LOCAL DEVELOPMENT ONLY

dan harus fail closed ketika mode demo/staging/production.

2. P0 berikutnya: cegah manipulasi nominal

Ini justru sangat penting karena LaQris adalah security layer pembayaran.

Schema DemoPay menerima:

session_id
amount
pin
terminal_id
user_id
account_number

sehingga amount berasal dari client.

Bayangkan user sudah melihat:

Rp27.000

kemudian browser dimodifikasi menjadi:

Rp270.000

Server tidak boleh hanya percaya payload tersebut.

Solusi paling aman untuk POC

Buat konsep:

SCAN
↓
VerificationSession
↓
Payment Intent / Confirmed Amount
↓
PIN
↓
DemoPay

Setelah nominal dikonfirmasi, backend menyimpan:

session.amount = 27000

Kemudian saat pembayaran:

if payload.amount != session.amount:
reject

Lebih bagus lagi, server menjadikan session.amount sebagai source of truth.

Untuk static QRIS, nominal memang dapat dimasukkan oleh pengguna, tetapi setelah pengguna melakukan konfirmasi, nominal tersebut harus dikunci di server.

3. P0: lifecycle session harus konsisten

Ada inkonsistensi yang saya temukan di implementasi sekarang.

Backend memberi pesan:

Sesi verifikasi telah kedaluwarsa (>15 menit)

pada endpoint verification dan DemoPay.

Padahal konsep LaQris yang kita susun sebelumnya menggunakan Verification Session 30 menit.

Jangan sampai demo UI mengatakan:

30 min

tetapi backend:

15 min
Revisi

Tetapkan satu konstanta:

VERIFICATION_SESSION_TTL_MINUTES = 30

Semua tempat menggunakan konstanta yang sama:

create session
verify
payment
transaction event
feedback

Dengan begitu tidak ada angka 15, 30, 10 tersebar di beberapa file.

4. P0: DemoPay sekarang masih hanya SUCCESS

Ini poin yang cukup besar untuk PoC.

Server sekarang melakukan:

~250 ms simulated host processing
↓
status = SUCCESS
response_code = 00
retry_count = 0

Latency memang diukur menggunakan perf_counter(), tetapi delay 250 ms tersebut tetap merupakan simulasi processing host, bukan latency bank/BI-FAST yang sesungguhnya.

Jangan dipresentasikan sebagai:

“LaQris mengukur latency BI-FAST real-time.”

Yang benar:

“DemoPay mensimulasikan downstream payment processing dan LaQris mengukur end-to-end processing latency pada POC.”

Revisi yang saya sarankan

Tambahkan scenario DemoPay:

SUCCESS
FAILED
TIMEOUT
CANCELLED

Misalnya:

Demo A → SUCCESS
Demo B → BLOCK sebelum payment

Demo C → payment FAILED
Demo D → timeout

Tidak perlu bikin banking engine rumit.

Cukup:

scenario = "SUCCESS"

tetapi keputusan akhirnya dibuat server-side, bukan browser.

Ini akan membuat Contract B jauh lebih bermakna karena transaction event sekarang memang punya variasi status.

5. P0: anti-replay belum cukup kuat secara concurrency

Sekarang ada pengecekan:

if session.is_bound:
raise HTTPException(...)

kemudian:

session.is_bound = True

dan commit dilakukan setelahnya.

Masalahnya adalah kalau ada dua request hampir bersamaan:

Request A → is_bound = False
Request B → is_bound = False
Request A → payment
Request B → payment

Keduanya bisa melewati pengecekan sebelum salah satu commit.

Revisi

Untuk POC, minimal lakukan:

SELECT session
FOR UPDATE

atau atomic update:

UPDATE verification_session
SET is_bound = TRUE
WHERE session_id = ?
AND is_bound = FALSE

lalu cek affected rows.

Tambahkan juga unique constraint untuk:

provider + provider_transaction_id

Dan transaction ID jangan hanya 6 digit random seperti:

TX-123456

lebih aman:

TX-<UUID> 6. P1: Contract A juga harus diamankan

/api/v1/verify saat ini menerima session_id, melakukan rate limit, lalu mengambil session dari database dan mengembalikan decision. Endpoint itu sendiri tidak terlihat menggunakan authentication provider seperti Contract B.

Secara arsitektur kita ingin:

DANA / GoPay / Bank
↓
LaQris /verify

maka perlu:

Provider Authentication

- HMAC/API Key

Jadi nanti:

Contract A
Pre-payment Verify
→ provider authenticated

Contract B
Post-payment Event
→ provider authenticated

Konsisten.

7. P1: HMAC sudah bagus, tapi tambahkan anti-replay callback

Commit 919f036 sudah memperbaiki HMAC agar signature diverifikasi menggunakan raw request body, ini sudah langkah bagus.

Tetapi HMAC saja belum otomatis mencegah:

Provider legitimate request
↓
attacker captures it
↓
replay request berkali-kali

Tambahkan:

X-LaQris-Timestamp
X-LaQris-Nonce

Contoh:

timestamp = 2026-09-20T16:30:00Z
nonce = random-unique-value
signature = HMAC(secret, timestamp + "." + raw_body)

LaQris:

timestamp terlalu lama → reject
nonce pernah digunakan → reject
signature invalid → reject

Ini bagus sekali untuk cerita security saat presentasi.

8. P1: ownership helper masih permissive

Di verify_user_ownership() sekarang:

if not requester_user_id or not resource_owner_id:
return

artinya kalau salah satu user ID hilang, fungsi justru mengizinkan request.

Untuk data yang seharusnya private, lebih baik:

resource seharusnya punya owner
↓
owner tidak ada
↓
reject / invalid state

Bukan:

owner tidak ada
↓
allow

Untuk anonymous feedback yang tidak terkait transaksi, boleh anonymous.

Tetapi:

transaction-linked feedback

harus strict:

JWT user == transaction owner 9. P1: Transaction Event jangan dipercaya mentah dari provider

Sekarang \_sanitize_payment_payload() sudah menyaring beberapa field sensitif seperti:

PAN
CVV
PIN
password
secret
auth token

Ini sudah bagus.

Tetapi arsitektur final sebaiknya jelas:

Raw Provider Payload
↓
Provider Adapter
↓
Sanitize + Validate
↓
Canonical LaQris Event

Contoh:

{
"provider": "DemoPay",
"provider_transaction_id": "TX-...",
"verification_id": "VFY-...",
"merchant_id": "...",
"nmid": "...",
"terminal_id": "...",
"amount": 27000,
"status": "SUCCESS",
"response_code": "00",
"transaction_time": "...",
"latency_ms": 281,
"retry_count": 0
}

Jadi nanti saat ditanya:

“Kalau DANA/GoPay bagaimana?”

Jawabannya:

“Kami menggunakan provider adapter. Payload masing-masing provider dinormalisasi menjadi canonical LaQris Transaction Event.”

Bukan membuat LaQris bergantung pada payload DemoPay.

10. P1: Error message masih membocorkan exception internal

Ada bagian:

detail=f"Gagal memproses ...: {str(e)}"

yang mengembalikan exception internal ke client.

Ganti menjadi:

Gagal memproses transaksi.
Transaction ID / trace ID: ...

Detail sebenarnya hanya masuk log server.

Ini kecil, tapi sangat mudah dijadikan pertanyaan juri security.

11. P1: Feedback NLP — jangan retrain dulu

Untuk model feedback, saya tidak menyarankan fine-tuning ulang sekarang.

Masalah boundary test sebelumnya cukup jelas:

"nama merchant sama..."
→ malah diprediksi identity mismatch

Padahal stress test secara keseluruhan sudah menunjukkan model mampu mengenali beberapa kelas dengan cukup baik.

Karena deadline tinggal 13 hari, jangan masuk spiral:

dataset
→ retrain
→ overfit
→ retrain
→ evaluasi
→ ubah dataset
→ retrain

Lebih aman gunakan hybrid guard.

Misalnya:

NLP IndoBERT
↓
negative cue detection
↓
positive identity statement detection
↓
final classification

Kalimat:

Nama merchant sama dengan nama toko, nominal sesuai, pembayaran berhasil tanpa kendala.

memiliki sinyal eksplisit:

sama
sesuai
berhasil
tanpa kendala

dan tidak memiliki:

berbeda
tidak sesuai
ditimpa
mencurigakan
palsu

Maka bisa diarahkan ke:

QRIS_NORMAL_MERCHANT_TERPERCAYA

Ini lebih aman untuk memperbaiki boundary behavior daripada memaksa model belajar ulang beberapa hari sebelum lomba.

12. P1: jangan masukkan model confidence ke risk score

Ini justru saya sarankan dibekukan.

Confidence:

YOLO confidence
TrOCR confidence
IndoBERT confidence

cukup untuk:

diagnostic
benchmark
debugging

Bukan:

risk score

Karena dataset kecil dan model confidence belum cukup layak dijadikan faktor finansial.

Jadi struktur:

AI Prediction
↓
Evidence
↓
Rule / Identity / Technical Analysis
↓
Risk

bukan:

AI confidence 87%
↓
risk calculation 13. P1: rapikan hubungan EMRS dengan transaction intelligence

Ada satu hal konseptual yang perlu diputuskan.

Sekarang calculate_emrs() menghitung:

A = Authenticity 40%
C = Complaints 30%
D = Disputes 20%
L = Longevity 10%

dan memang menghitung T_observed dari transaksi sukses/terverifikasi, tetapi T_observed belum dimasukkan ke formula final R.

Jadi jangan sampai presentasi bilang:

“Setiap transaksi sukses langsung menaikkan EMRS.”

Karena implementasi sekarang belum seperti itu.

Ada dua opsi:

Opsi aman sebelum deadline:

Tetap formula sekarang, tetapi jelaskan:

Transaction Intelligence
= reliability telemetry

sedangkan:

EMRS
= authenticity + complaints + disputes + longevity

Saya lebih memilih opsi ini menjelang lomba daripada mengubah formula besar-besaran.

14. P2: frontend sudah jauh lebih baik, tetapi wajib dites secara runtime

payment.html pada commit yang saya cek sudah memiliki:

security status banner,
blocked screen,
nominal,
confirmation,
PIN,
receipt.

Jadi sekarang jangan fokus mempercantik UI dulu.

Yang wajib dibuktikan:

Browser
↓
login JWT
↓
scan
↓
verification session
↓
payment
↓
server-side DemoPay
↓
transaction
↓
feedback

Pastikan browser tidak pernah mengirim provider secret/API key.
