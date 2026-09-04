/* ======================================================
   LaQris — API Configuration
   ======================================================
   SATU-SATUNYA file yang perlu diubah untuk konfigurasi API.

   LOCAL DEV  → tidak perlu ubah apapun (auto-detect localhost)
   PRODUCTION → ubah baris LAQRIS_API_URL di bawah:
                window.LAQRIS_API_URL = "https://nama-app.railway.app";
   ====================================================== */

window.LAQRIS_API_URL = "https://YOUR-BACKEND-URL";

window.API_BASE = (function () {
    // Lokal: FastAPI serve frontend & backend di port yang sama
    if (window.location.hostname === "localhost" ||
        window.location.hostname === "127.0.0.1") {
        return window.location.origin; // e.g. "http://localhost:5000"
    }
    // Production: wajib set LAQRIS_API_URL di atas
    if (window.LAQRIS_API_URL && window.LAQRIS_API_URL !== "https://YOUR-BACKEND-URL") {
        return window.LAQRIS_API_URL.replace(/\/$/, "");
    }
    // Belum dikonfigurasi — kembalikan placeholder agar error jelas
    return "https://YOUR-BACKEND-URL";
})();
