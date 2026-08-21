/* ======================================================
   LaQris — API Configuration
   ======================================================
   CARA KONFIGURASI:
   1. Local dev (FastAPI berjalan di localhost:8000):
      Ubah baris di bawah menjadi:
      window.LAQRIS_API_URL = "http://localhost:8000";

   2. Production (backend sudah di-deploy, misal Railway):
      Ubah baris di bawah menjadi:
      window.LAQRIS_API_URL = "https://your-backend.railway.app";

   JANGAN pernah biarkan kosong ("") saat deploy ke Vercel,
   karena Vercel tidak memiliki backend dan akan mengembalikan
   HTML 404 bukan JSON.
   ====================================================== */

window.LAQRIS_API_URL = "https://YOUR-BACKEND-URL";

window.API_BASE = (function () {
    // 1. Prioritas tertinggi: env yang di-inject oleh CI/CD / build tool
    if (typeof __API_BASE__ !== "undefined" && __API_BASE__) return __API_BASE__;
    // 2. URL yang di-set manual di atas
    if (window.LAQRIS_API_URL && window.LAQRIS_API_URL !== "https://YOUR-BACKEND-URL") {
        return window.LAQRIS_API_URL.replace(/\/$/, ""); // hapus trailing slash
    }
    // 3. Fallback: tampilkan placeholder (akan menghasilkan error yang jelas)
    return "https://YOUR-BACKEND-URL";
})();
