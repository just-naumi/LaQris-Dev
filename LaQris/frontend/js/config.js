/* ======================================================
   LaQris — API Configuration
   ======================================================
   Ubah nilai API_BASE sesuai URL backend Anda:
   - Development lokal : ""  (empty, proxy ke localhost)
   - Production Vercel : "https://your-backend.railway.app"
   ====================================================== */

window.API_BASE = (function() {
    // Jika ada env yang di-inject oleh CI/CD, gunakan itu
    if (typeof __API_BASE__ !== "undefined") return __API_BASE__;
    // Deteksi otomatis: jika di Vercel (domain vercel.app), gunakan env dari window
    if (window.LAQRIS_API_URL) return window.LAQRIS_API_URL;
    // Default: kosong = relative URL (cocok untuk local dev dengan FastAPI)
    return "";
})();

