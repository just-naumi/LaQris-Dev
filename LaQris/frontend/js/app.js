/* ==============================================================================
   LaQris POC v2.0 — EMRS Frontend Logic
   ============================================================================== */

// API_BASE dikonfigurasi melalui js/config.js
// JANGAN ubah baris ini — ubah window.LAQRIS_API_URL di config.js
const API_BASE = (window.API_BASE || "").replace(/\/$/, "");

// ── API Helper ────────────────────────────────────────────────
// Semua request ke backend wajib melalui fungsi ini.
async function apiCall(path, options = {}) {
    if (!API_BASE || API_BASE === "https://YOUR-BACKEND-URL") {
        throw new Error(
            "Backend belum dikonfigurasi.\n" +
            "Buka js/config.js dan ubah window.LAQRIS_API_URL " +
            "ke URL backend Anda (Railway, Render, dll)."
        );
    }
    const url = `${API_BASE}${path}`;
    const response = await fetch(url, options);
    if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try { const err = await response.json(); detail = err.detail || detail; } catch (_) {}
        throw new Error(detail);
    }
    return response.json();
}

// State: simpan data NMID aktif untuk feedback modal
let currentNmid = null;

document.addEventListener("DOMContentLoaded", () => {
    console.log("LaQris EMRS v2.0 Frontend Loaded.");
    if (!API_BASE || API_BASE === "https://YOUR-BACKEND-URL") {
        console.warn(
            "%c[LaQris] Backend belum dikonfigurasi!\n" +
            "Buka js/config.js dan set window.LAQRIS_API_URL ke URL backend Anda.",
            "color: #f59e0b; font-weight: bold;"
        );
    }
});


// ══════════════════════════════════════════════════════════════
// Scan Functions
// ══════════════════════════════════════════════════════════════

async function runScanSample(sampleFilename) {
    startLoading();

    const formData = new FormData();
    formData.append("sample_name", sampleFilename);

    try {
        await tickSteps();
        const data = await apiCall("/api/scan", { method: "POST", body: formData });
        finishLoading();
        setTimeout(() => renderScanResults(data), 400);
    } catch (error) {
        console.error("Gagal scan sampel:", error);
        showError("Gagal memproses sampel QRIS.", error.message);
        resetScanner();
    }
}

async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    startLoading();

    const formData = new FormData();
    formData.append("file", file);

    try {
        await tickSteps();
        const data = await apiCall("/api/scan", { method: "POST", body: formData });
        finishLoading();
        setTimeout(() => renderScanResults(data), 400);
    } catch (error) {
        console.error("Gagal upload file:", error);
        showError("Gagal mengunggah foto QRIS.", error.message);
        resetScanner();
    }
}


// ══════════════════════════════════════════════════════════════
// Loading State
// ══════════════════════════════════════════════════════════════

function startLoading() {
    document.getElementById("uploadPanel").classList.add("d-none");
    document.getElementById("loadingPanel").classList.remove("d-none");
    document.getElementById("resultSection").classList.add("d-none");
    setStep(1);
}

function finishLoading() {
    document.getElementById("loadingPanel").classList.add("d-none");
    document.getElementById("resultSection").classList.remove("d-none");
}

function setStep(n) {
    for (let i = 1; i <= 5; i++) {
        const el = document.getElementById(`step${i}`);
        if (!el) continue;
        el.className = i < n ? "step-item complete" : i === n ? "step-item active" : "step-item";
    }
}

async function tickSteps() {
    const delays = [200, 300, 300, 400, 0];
    for (let i = 1; i <= 5; i++) {
        setStep(i);
        await sleep(delays[i - 1]);
    }
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}


// ══════════════════════════════════════════════════════════════
// Render Results — Dua Panel Terpisah
// ══════════════════════════════════════════════════════════════

function renderScanResults(data) {
    const qr = data.current_qr_risk || {};
    const rep = data.merchant_reputation || {};

    // Simpan NMID untuk feedback
    currentNmid = qr.digital_nmid || rep.nmid || null;

    // ── PANEL A: Current QR Risk ──────────────────────────────
    renderQRRiskPanel(qr);

    // ── PANEL B: Merchant Reputation (EMRS) ──────────────────
    renderReputationPanel(rep);

    // Visualisasi YOLO
    if (data.visualization_url) {
        document.getElementById("visImage").src = data.visualization_url + "?t=" + Date.now();
        document.getElementById("visBox").style.display = "block";
    }

    // Action buttons
    const btnCancel = document.getElementById("btnCancelPayment");
    if (qr.is_mismatch || qr.risk_level === "HIGH_RISK") {
        btnCancel.style.display = "block";
    } else {
        btnCancel.style.display = "none";
    }

    document.getElementById("resultSection").scrollIntoView({ behavior: "smooth" });

    // High risk modal
    if (qr.risk_level === "HIGH_RISK") {
        setTimeout(() => showWarningModal(qr, rep), 600);
    }
}

function renderQRRiskPanel(qr) {
    // Trust Score Bar
    const score = qr.trust_score ?? 0;
    document.getElementById("trustScoreValue").textContent = `${score.toFixed(0)} / 100`;

    const bar = document.getElementById("trustProgressBar");
    bar.style.width = `${score}%`;
    bar.className = "score-bar-fill";
    if (score >= 80) bar.classList.add("fill-safe");
    else if (score >= 50) bar.classList.add("fill-moderate");
    else bar.classList.add("fill-danger");

    // Risk Badge
    const badgeLg = document.getElementById("riskBadgeLg");
    const icon = document.getElementById("riskPanelIcon");
    const rl = qr.risk_level || "SAFE";

    const riskMap = {
        "HIGH_RISK":      { label: "HIGH RISK",      cls: "badge-danger",   iconCls: "icon-danger",   iconI: "fa-triangle-exclamation" },
        "ELEVATED_RISK":  { label: "ELEVATED RISK",  cls: "badge-warning",  iconCls: "icon-warning",  iconI: "fa-circle-exclamation" },
        "MODERATE_RISK":  { label: "MODERATE RISK",  cls: "badge-moderate", iconCls: "icon-moderate", iconI: "fa-circle-info" },
        "SAFE":           { label: "AMAN ✓",          cls: "badge-safe",    iconCls: "icon-safe",     iconI: "fa-shield-check" }
    };
    const rm = riskMap[rl] || riskMap["SAFE"];
    badgeLg.textContent = rm.label;
    badgeLg.className = `risk-badge-lg ${rm.cls}`;
    icon.className = `panel-icon ${rm.iconCls}`;
    icon.innerHTML = `<i class="fa-solid ${rm.iconI}"></i>`;

    // Explanation
    const expBox = document.getElementById("explanationBox");
    expBox.textContent = qr.explanation || "—";
    expBox.className = `explanation-box ${qr.is_mismatch ? "exp-danger" : "exp-safe"}`;

    // Physical vs Digital
    document.getElementById("physMerchantName").textContent = qr.physical_merchant || "Tidak terbaca";
    document.getElementById("physNmid").textContent         = qr.physical_nmid    || "Tidak terbaca";
    document.getElementById("physAcquirer").textContent     = qr.physical_acquirer || "Tidak terbaca";
    document.getElementById("physTid").textContent          = qr.physical_tid     || "Tidak terbaca";

    document.getElementById("digMerchantName").textContent  = qr.digital_merchant || "Tidak ditemukan";
    document.getElementById("digNmid").textContent          = qr.digital_nmid     || "Tidak ditemukan";
    document.getElementById("digAcquirer").textContent      = qr.digital_acquirer || "Tidak ditemukan";
    document.getElementById("digCity").textContent          = qr.digital_city     || "Tidak ditemukan";

    // Match badge
    const matchBadge = document.getElementById("matchBadge");
    matchBadge.className = `kv-vs-badge ${qr.is_mismatch ? "vs-mismatch" : "vs-match"}`;
    matchBadge.innerHTML = qr.is_mismatch
        ? `<i class="fa-solid fa-xmark"></i>`
        : `<i class="fa-solid fa-check"></i>`;

    // Match details
    document.getElementById("similarityVal").textContent  = `${qr.name_similarity ?? 0}%`;
    document.getElementById("matchLevelVal").textContent  = qr.match_level || "—";
    document.getElementById("qrStatusVal").textContent    = qr.technical_info?.status || "—";
}

function renderReputationPanel(rep) {
    const score = rep.reputation_score ?? 50;
    const comps = rep.components || {};

    // EMRS Score Circle
    document.getElementById("emrsScoreNum").textContent = score.toFixed(0);
    const circle = document.getElementById("emrsScoreCircle");
    circle.className = "emrs-score-circle";
    if (score >= 85)      circle.classList.add("circle-excellent");
    else if (score >= 70) circle.classList.add("circle-good");
    else if (score >= 55) circle.classList.add("circle-fair");
    else if (score >= 40) circle.classList.add("circle-poor");
    else                  circle.classList.add("circle-critical");

    // Grade Banner
    const gradeMap = {
        "Excellent": { cls: "grade-excellent", icon: "🏆" },
        "Very Good":  { cls: "grade-verygood",  icon: "⭐" },
        "Good":       { cls: "grade-good",      icon: "👍" },
        "Fair":       { cls: "grade-fair",      icon: "⚠️" },
        "Poor":       { cls: "grade-poor",      icon: "🔴" }
    };
    const gm = gradeMap[rep.grade] || gradeMap["Fair"];
    const gradeBanner = document.getElementById("gradeBanner");
    gradeBanner.className = `grade-banner ${gm.cls}`;
    document.getElementById("gradeLabel").textContent = `${gm.icon} ${rep.grade ?? "—"}`;

    const evQ = rep.evidence_quality || "INSUFFICIENT";
    const evLabel = { "HIGH": "High Evidence", "MEDIUM": "Medium Evidence", "LOW": "Low Evidence", "INSUFFICIENT": "Insufficient Data" };
    document.getElementById("gradeEvidence").textContent = `${evLabel[evQ]} (${rep.total_evidence_count ?? 0} poin)`;

    // EMRS Components
    const compData = [
        { id: "T", label: "T", val: comps.T },
        { id: "A", label: "A", val: comps.A },
        { id: "L", label: "L", val: comps.L },
        { id: "C", label: "C", val: comps.C },
        { id: "D", label: "D", val: comps.D }
    ];
    for (const c of compData) {
        const v = c.val ?? 50;
        document.getElementById(`score${c.id}`).textContent = `${v.toFixed(0)}`;
        const bar = document.getElementById(`bar${c.id}`);
        bar.style.width = `${v}%`;
    }

    // Merchant Info Row
    if (rep.found_in_db) {
        document.getElementById("merchantInfoRow").style.display = "flex";
        document.getElementById("repNmid").textContent = rep.nmid || "—";

        if (rep.registered_at) {
            const regDate = new Date(rep.registered_at);
            document.getElementById("repRegistered").textContent = `Aktif sejak ${regDate.toLocaleDateString("id-ID", { year: "numeric", month: "long" })}`;
        }
    } else {
        document.getElementById("merchantInfoRow").style.display = "none";
    }

    // Enable/disable feedback btn
    document.getElementById("btnFeedback").disabled = !rep.found_in_db;
}


// ══════════════════════════════════════════════════════════════
// Warning Modal
// ══════════════════════════════════════════════════════════════

function showWarningModal(qr, rep) {
    document.getElementById("modalPhysName").textContent = qr.physical_merchant || "TIDAK TERBACA";
    document.getElementById("modalDigName").textContent  = qr.digital_merchant  || "TIDAK DITEMUKAN";

    const repDetail = document.getElementById("modalReputationDetail");
    if (rep && rep.found_in_db) {
        repDetail.innerHTML = `Merchant <strong>${rep.merchant_name}</strong> memiliki EMRS Score <strong>${(rep.reputation_score ?? 0).toFixed(0)} / 100</strong> (${rep.grade}). Terdapat <strong>${rep.total_evidence_count ?? 0} poin evidence</strong> dalam database LaQris.`;
    } else {
        repDetail.innerHTML = `Identitas stiker fisik tidak cocok dengan identitas penerima digital QRIS. Merchant <strong>belum dikenal</strong> di database LaQris.`;
    }
    new bootstrap.Modal(document.getElementById("warningModal")).show();
}

function closeWarningModal() {
    const el = document.getElementById("warningModal");
    bootstrap.Modal.getInstance(el)?.hide();
}

function cancelTransactionAction() {
    closeWarningModal();
    Swal.fire({
        icon: "success",
        title: "❌ Transaksi Dibatalkan!",
        html: `<strong style="color:#34d399;font-size:1.05rem;">Potensi Penipuan Berhasil Dicegah!</strong><br><br>Anda berhasil membatalkan pembayaran dan terhindar dari penipuan stiker QRIS ditimpa.`,
        background: "#18181b",
        color: "#fff",
        confirmButtonColor: "#10b981",
        confirmButtonText: "Kembali ke Beranda",
    }).then(() => resetScanner());
}


// ══════════════════════════════════════════════════════════════
// Feedback Modal
// ══════════════════════════════════════════════════════════════

function openFeedbackModal() {
    if (!currentNmid) return;
    new bootstrap.Modal(document.getElementById("feedbackModal")).show();
}

function closeFeedbackModal() {
    bootstrap.Modal.getInstance(document.getElementById("feedbackModal"))?.hide();
}

function toggleEvidence() {
    const current = document.getElementById("fbHasEvidence").value === "true";
    const next = !current;
    document.getElementById("fbHasEvidence").value = next.toString();
    const icon = document.getElementById("fbEvidenceIcon");
    const toggle = document.getElementById("fbEvidenceToggle");
    icon.innerHTML = next ? '<i class="fa-solid fa-square-check"></i>' : '<i class="fa-regular fa-square"></i>';
    toggle.className = next ? "fb-evidence-toggle active" : "fb-evidence-toggle";
}

async function submitFeedback() {
    if (!currentNmid) return;

    const payload = {
        nmid: currentNmid,
        category: document.getElementById("fbCategory").value,
        severity: document.getElementById("fbSeverity").value,
        description: document.getElementById("fbDescription").value || null,
        transaction_ref: document.getElementById("fbTxRef").value || null,
        has_evidence: document.getElementById("fbHasEvidence").value === "true"
    };

    try {
        const result = await apiCall("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        closeFeedbackModal();
        Swal.fire({
            icon: "success",
            title: "Laporan Terkirim!",
            html: `Evidence Level: <strong>${payload.has_evidence ? "2 (Verified)" : "1 (Unverified)"}</strong><br>EMRS baru: <strong>${result.new_reputation_score?.toFixed(1) ?? "—"} / 100</strong>`,
            background: "#18181b",
            color: "#fff",
            confirmButtonColor: "#6366f1"
        });
    } catch (e) {
        Swal.fire({ icon: "error", title: "Network Error", text: e.message, background: "#18181b", color: "#fff" });
    }
}


// ══════════════════════════════════════════════════════════════
// Database Modal
// ══════════════════════════════════════════════════════════════

async function openDatabaseModal() {
    try {
        const merchants = await apiCall("/api/merchants");

        const tbody = document.getElementById("merchantTableBody");
        tbody.innerHTML = "";

        for (const m of merchants) {
            // Get reputation score
            let emrsScore = m.reputation_score ?? 50;
            let grade = "—";

            // Determine grade from score
            if (emrsScore >= 85) grade = "Excellent";
            else if (emrsScore >= 70) grade = "Very Good";
            else if (emrsScore >= 55) grade = "Good";
            else if (emrsScore >= 40) grade = "Fair";
            else grade = "Poor";

            const gradeCls = {
                "Excellent": "badge-excellent", "Very Good": "badge-verygood",
                "Good": "badge-good", "Fair": "badge-fair", "Poor": "badge-poor"
            }[grade] || "badge-fair";

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><code class="code-nmid">${m.nmid}</code></td>
                <td><strong>${m.merchant_name}</strong></td>
                <td><span class="badge-acquirer">${m.acquirer}</span></td>
                <td><strong class="emrs-val">${emrsScore.toFixed(1)}</strong></td>
                <td><span class="db-grade ${gradeCls}">${grade}</span></td>
                <td><span class="badge-reports">${m.total_reports} (${m.verified_reports} verified)</span></td>
            `;
            tbody.appendChild(tr);
        }

        new bootstrap.Modal(document.getElementById("databaseModal")).show();
    } catch (error) {
        console.error("Gagal membaca database:", error);
    }
}


// ══════════════════════════════════════════════════════════════
// Reset
// ══════════════════════════════════════════════════════════════

function resetScanner() {
    currentNmid = null;
    document.getElementById("fileInput").value = "";
    document.getElementById("uploadPanel").classList.remove("d-none");
    document.getElementById("loadingPanel").classList.add("d-none");
    document.getElementById("resultSection").classList.add("d-none");
    window.scrollTo({ top: 0, behavior: "smooth" });
}


// ══════════════════════════════════════════════════════════════
// Utilities
// ══════════════════════════════════════════════════════════════

function showError(title, text) {
    Swal.fire({ icon: "error", title, text, background: "#18181b", color: "#fff" });
}
