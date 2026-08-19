/* ==============================================================================
   LaQris POC Tahap 1 — Dynamic Frontend Logic & API Integration
   ============================================================================== */

const API_BASE = "";

document.addEventListener("DOMContentLoaded", () => {
    console.log("LaQris POC Tahap 1 Frontend App Loaded.");
});

// Run scan with pre-loaded sample
async function runScanSample(sampleFilename) {
    showStepProgress(1);
    
    const formData = new FormData();
    formData.append("sample_name", sampleFilename);

    try {
        updateStepIndicator(2);
        const response = await fetch(`${API_BASE}/api/scan`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        updateStepIndicator(4);
        const data = await response.json();
        updateStepIndicator(5);
        
        setTimeout(() => {
            renderScanResults(data);
        }, 400);

    } catch (error) {
        console.error("Gagal melakukan scan sampel:", error);
        Swal.fire({
            icon: 'error',
            title: 'Gagal Scan',
            text: 'Terjadi kesalahan saat memproses gambar QRIS.',
            background: '#1e293b',
            color: '#fff'
        });
    }
}

// Handle file selection from upload dropzone
async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    showStepProgress(1);

    const formData = new FormData();
    formData.append("file", file);

    try {
        updateStepIndicator(2);
        const response = await fetch(`${API_BASE}/api/scan`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        updateStepIndicator(4);
        const data = await response.json();
        updateStepIndicator(5);

        setTimeout(() => {
            renderScanResults(data);
        }, 400);

    } catch (error) {
        console.error("Gagal upload file:", error);
        Swal.fire({
            icon: 'error',
            title: 'Gagal Scan File',
            text: error.message || 'Terjadi kesalahan saat mengunggah foto.',
            background: '#1e293b',
            color: '#fff'
        });
    }
}

// Render Results to UI
function renderScanResults(data) {
    document.getElementById("resultSection").style.display = "flex";

    // 1. Visual Image Bounding Box
    if (data.visualization_image_url) {
        document.getElementById("visImage").src = data.visualization_image_url;
    }

    // 2. Physical vs Digital Comparison
    const phys = data.physical_entity || {};
    const dig = data.digital_entity || {};

    document.getElementById("physMerchantName").textContent = phys.merchant_name || "Tidak terbaca";
    document.getElementById("physNmid").textContent = `NMID: ${phys.nmid || 'Tidak terbaca'}`;
    document.getElementById("physAcquirer").textContent = `Acquirer: ${phys.acquirer || 'Tidak terbaca'}`;

    document.getElementById("digMerchantName").textContent = dig.merchant_name || "Tidak ditemukan";
    document.getElementById("digNmid").textContent = `NMID: ${dig.nmid || 'Tidak ditemukan'}`;
    document.getElementById("digAcquirer").textContent = `Acquirer: ${dig.acquirer || 'Tidak ditemukan'}`;

    // 3. Trust Score Bar
    const score = data.trust_score || 0;
    document.getElementById("trustScoreValue").textContent = `${score}%`;
    const progressBar = document.getElementById("trustProgressBar");
    progressBar.style.width = `${score}%`;
    
    if (score >= 85) {
        progressBar.className = "progress-bar bg-success";
    } else if (score >= 50) {
        progressBar.className = "progress-bar bg-warning";
    } else {
        progressBar.className = "progress-bar bg-danger";
    }

    // 4. Verdict Status Banner
    const verdictCard = document.getElementById("verdictCard");
    const riskBadge = document.getElementById("riskBadge");
    const verdictTitle = document.getElementById("verdictTitle");
    const verdictExp = document.getElementById("verdictExplanation");
    const verdictIcon = document.getElementById("verdictIcon");

    verdictTitle.textContent = data.verdict_status;
    verdictExp.textContent = data.explanation;

    if (data.risk_level === "HIGH_RISK") {
        riskBadge.textContent = "HIGH RISK (BAHAYA)";
        riskBadge.className = "badge bg-danger mb-1";
        verdictIcon.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-danger"></i>`;
        verdictCard.style.borderColor = "var(--accent-rose)";
    } else if (data.risk_level === "SAFE") {
        riskBadge.textContent = "VERIFIED SAFE (AMAN)";
        riskBadge.className = "badge bg-success mb-1";
        verdictIcon.innerHTML = `<i class="fa-solid fa-shield-check text-emerald"></i>`;
        verdictCard.style.borderColor = "var(--accent-emerald)";
    } else {
        riskBadge.textContent = "MEDIUM RISK (PERHATIAN)";
        riskBadge.className = "badge bg-warning text-dark mb-1";
        verdictIcon.innerHTML = `<i class="fa-solid fa-circle-exclamation text-warning"></i>`;
        verdictCard.style.borderColor = "var(--accent-amber)";
    }

    // 5. SQLite Merchant Reputation Widget
    const rep = data.reputation;
    if (rep) {
        document.getElementById("repNmidBadge").textContent = `NMID: ${rep.nmid}`;
        document.getElementById("repRating").textContent = rep.rating.toFixed(1);
        document.getElementById("repTotalReports").textContent = rep.total_reports;
        document.getElementById("repVerifiedReports").textContent = rep.verified_reports;

        const bc = rep.breakdown_categories || { qris_replacement: 0, additional_fee: 0, merchant_mismatch: 0 };
        document.getElementById("catReplacementCount").textContent = bc.qris_replacement;
        document.getElementById("catFeeCount").textContent = bc.additional_fee;
        document.getElementById("catMismatchCount").textContent = bc.merchant_mismatch;

        const maxRep = Math.max(rep.total_reports, 1);
        document.getElementById("catReplacementBar").style.width = `${(bc.qris_replacement / maxRep) * 100}%`;
        document.getElementById("catFeeBar").style.width = `${(bc.additional_fee / maxRep) * 100}%`;
        document.getElementById("catMismatchBar").style.width = `${(bc.merchant_mismatch / maxRep) * 100}%`;
    } else {
        document.getElementById("repNmidBadge").textContent = `NMID: ${dig.nmid || '-'}`;
        document.getElementById("repRating").textContent = "5.0";
        document.getElementById("repTotalReports").textContent = "0";
        document.getElementById("repVerifiedReports").textContent = "0";
        document.getElementById("catReplacementCount").textContent = "0";
        document.getElementById("catFeeCount").textContent = "0";
        document.getElementById("catMismatchCount").textContent = "0";
        document.getElementById("catReplacementBar").style.width = "0%";
        document.getElementById("catFeeBar").style.width = "0%";
        document.getElementById("catMismatchBar").style.width = "0%";
    }

    // Scroll smoothly to results
    document.getElementById("resultSection").scrollIntoView({ behavior: 'smooth' });

    // 6. Trigger HIGH RISK Warning Modal if High Risk Detected!
    if (data.risk_level === "HIGH_RISK") {
        setTimeout(() => {
            showWarningModal(data);
        }, 500);
    }
}

// Show Warning Modal
function showWarningModal(data) {
    const physName = data.physical_entity?.merchant_name || "TOKO BERKAH JAYA";
    const digName = data.digital_entity?.merchant_name || "BUDI PRIBADI";
    const digNmid = data.digital_entity?.nmid || "ID1024309405321";

    document.getElementById("modalPhysName").textContent = physName;
    document.getElementById("modalDigName").textContent = digName;

    document.getElementById("modalBoxPhys").textContent = physName;
    document.getElementById("modalBoxDig").textContent = `${digName} (NMID: ${digNmid})`;

    const rep = data.reputation;
    if (rep) {
        document.getElementById("modalReputationDetail").innerHTML = 
            `Merchant <strong>${rep.merchant_name}</strong> memiliki Rating <strong>${rep.rating.toFixed(1)} / 5.0</strong> dengan total <strong>${rep.total_reports} laporan penipuan</strong> (${rep.breakdown_categories?.qris_replacement || 0} Laporan QRIS Replacement).`;
    } else {
        document.getElementById("modalReputationDetail").innerHTML = 
            `Identitas stiker fisik (${physName}) tidak cocok dengan identitas penerima digital (${digName}). Terindikasi stiker QRIS ditimpa.`;
    }

    const modal = new bootstrap.Modal(document.getElementById('warningModal'));
    modal.show();
}

function closeWarningModal() {
    const modalEl = document.getElementById('warningModal');
    const modal = bootstrap.Modal.getInstance(modalEl);
    if (modal) modal.hide();
}

// Action button: CANCEL PAYMENT
function cancelTransactionAction() {
    closeWarningModal();
    Swal.fire({
        icon: 'success',
        title: '❌ TRANSACTION CANCELLED!',
        html: `<strong style="color: #34d399; font-size: 1.1rem;">Suspicious Transaction Prevented!</strong><br><br>Pembayaran berhasil dibatalkan oleh pengguna. Anda terhindar dari potensi penipuan QRIS ditimpa.`,
        background: '#18181b',
        color: '#fff',
        confirmButtonColor: '#10b981',
        confirmButtonText: 'Kembali ke Beranda'
    });
}

// Step Progress Animations
function showStepProgress(step) {
    updateStepIndicator(step);
}

function updateStepIndicator(step) {
    for (let i = 1; i <= 5; i++) {
        const el = document.getElementById(`step${i}`);
        if (!el) continue;
        if (i < step) {
            el.className = "step-item complete";
        } else if (i === step) {
            el.className = "step-item active";
        } else {
            el.className = "step-item";
        }
    }
}

// Open Database Directory Modal
async function openDatabaseModal() {
    try {
        const response = await fetch(`${API_BASE}/api/merchants`);
        const merchants = await response.json();

        const tbody = document.getElementById("merchantTableBody");
        tbody.innerHTML = "";

        merchants.forEach(m => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><code>${m.nmid}</code></td>
                <td><strong class="text-white">${m.merchant_name}</strong></td>
                <td><span class="badge bg-secondary">${m.acquirer}</span></td>
                <td><span class="badge bg-warning text-dark"><i class="fa-solid fa-star"></i> ${m.rating.toFixed(1)}</span></td>
                <td><span class="badge bg-danger">${m.total_reports} Laporan (${m.verified_reports} Verified)</span></td>
            `;
            tbody.appendChild(tr);
        });

        const modal = new bootstrap.Modal(document.getElementById('databaseModal'));
        modal.show();

    } catch (error) {
        console.error("Gagal membaca database SQLite:", error);
    }
}
