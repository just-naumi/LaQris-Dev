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
    // Jika diakses via browser (localhost, 127.0.0.1, atau IP Wi-Fi HP seperti 192.168.x.x):
    // FastAPI menyajikan frontend & API di origin yang sama
    if (window.location.origin && window.location.origin.startsWith("http")) {
        return window.location.origin;
    }
    // Production override jika frontend di-hosting terpisah:
    if (window.LAQRIS_API_URL && window.LAQRIS_API_URL !== "https://YOUR-BACKEND-URL") {
        return window.LAQRIS_API_URL.replace(/\/$/, "");
    }
    return "http://localhost:5000";
})();
