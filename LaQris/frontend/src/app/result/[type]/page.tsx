"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import type { ScanResponse } from "@/types/detection";

export default function ResultPage() {
  const params = useParams();
  const type = (params?.type as string) || "aman";
  const [scanData, setScanData] = useState<ScanResponse | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const storedScan = sessionStorage.getItem("laqris:last-scan");
      if (storedScan) {
        try {
          setScanData(JSON.parse(storedScan) as ScanResponse);
        } catch {
          sessionStorage.removeItem("laqris:last-scan");
        }
      }
    }, 0);

    return () => window.clearTimeout(timer);
  }, []);

  // Data config for Aman, Waspada, and Bahaya
  const configs = {
    aman: {
      title: "STATUS: AMAN",
      subtitle: "Kode QRIS Terverifikasi & Tepercaya",
      badgeText: "Terverifikasi 100%",
      badgeClass: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20",
      iconBg: "bg-emerald-50 text-emerald-600 border-emerald-200",
      iconSvg: (
        <svg className="w-8 h-8 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      reputationScore: "98%",
      reputationText: "Sangat Baik & Aman",
      merchantName: "Toko Berkah Jaya",
      qrisCategory: "Umum / Retail",
      bankName: "Bank Central Asia (BCA)",
      details: [
        "Identitas merchant sesuai dengan pendaftaran resmi QRIS.",
        "Tidak ditemukan laporan indikasi penipuan atau fraud.",
        "Rekening tujuan aktif dan memiliki track record positif.",
      ],
      buttonText: "Pindai QRIS Lain",
      btnClass: "bg-neutral-900 text-white hover:bg-neutral-800",
    },
    waspada: {
      title: "STATUS: WASPADA",
      subtitle: "Perlu Perhatian Ekstra Sebelum Transaksi",
      badgeText: "Perlu Diwaspadai",
      badgeClass: "bg-amber-500/10 text-amber-600 border-amber-500/20",
      iconBg: "bg-amber-50 text-amber-600 border-amber-200",
      iconSvg: (
        <svg className="w-8 h-8 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
        </svg>
      ),
      reputationScore: "65%",
      reputationText: "Cukup Baik (Ada Catatan)",
      merchantName: "Kedai Kopi Utama (Perseorangan)",
      qrisCategory: "Usaha Mikro",
      bankName: "E-Wallet / Bank Digital",
      details: [
        "Nama merchant berbeda tipis dengan nama pemilik rekening.",
        "Akun QRIS baru terdaftar dalam 30 hari terakhir.",
        "Pastikan mengonfirmasi nama penerima sebelum menekan tombol bayar.",
      ],
      buttonText: "Pindai QRIS Lain",
      btnClass: "bg-amber-600 text-white hover:bg-amber-700",
    },
    bahaya: {
      title: "STATUS: BAHAYA",
      subtitle: "Terdeteksi Indikasi Penipuan / QR Palsu",
      badgeText: "Terindikasi Fraud!",
      badgeClass: "bg-rose-500/10 text-rose-600 border-rose-500/20",
      iconBg: "bg-rose-50 text-rose-600 border-rose-200",
      iconSvg: (
        <svg className="w-8 h-8 text-rose-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m0 3.75h.008v.008H12v-.008zM21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      reputationScore: "15%",
      reputationText: "Sangat Berbahaya",
      merchantName: "SatuQR_FakeMerchant_Unverified",
      qrisCategory: "Tidak Terdaftar",
      bankName: "Rekening Mencurigakan",
      details: [
        "QRIS ditempel di atas QRIS asli (Modus Penipuan Stiker QRIS).",
        "Terdapat 12+ laporan masyarakat terkait penipuan pada rekening ini.",
        "JANGAN MELAKUKAN TRANSFER ATAU PEMBAYARAN KE KODE INI!",
      ],
      buttonText: "Laporkan / Pindai Lagi",
      btnClass: "bg-rose-600 text-white hover:bg-rose-700",
    },
  };

  const currentConfig = configs[type as keyof typeof configs] || configs.aman;
  const merchantName = scanData?.digital_merchant || currentConfig.merchantName;
  const bankName = scanData?.digital_acquirer || currentConfig.bankName;
  const reputationScore = scanData ? `${Math.round(scanData.trust_score)}%` : currentConfig.reputationScore;
  const details = scanData ? [scanData.explanation, `Kecocokan nama: ${scanData.match_level}`, `NMID digital: ${scanData.digital_nmid}`] : currentConfig.details;

  return (
    <div className="min-h-screen bg-neutral-900 py-0 sm:py-8 flex items-center justify-center font-sans antialiased">
      {/* Mobile Screen Container */}
      <div className="w-full max-w-[390px] h-[844px] bg-white sm:rounded-[44px] shadow-2xl overflow-hidden flex flex-col justify-between relative border-0 sm:border-[8px] border-neutral-800 text-neutral-900">
        
        {/* Soft Background Decorative Circles */}
        <div className="absolute -top-12 -left-12 w-64 h-64 bg-neutral-100/80 rounded-full blur-2xl pointer-events-none" />
        <div className="absolute top-1/3 -right-16 w-60 h-60 bg-neutral-100/60 rounded-full blur-2xl pointer-events-none" />

        {/* TOP NAVBAR */}
        <div className="pt-6 px-6 pb-3 flex items-center justify-between relative z-10 border-b border-neutral-100 bg-white/80 backdrop-blur">
          <Link href="/dashboard" className="text-2xl font-extrabold tracking-tight text-neutral-900">
            LàQris.
          </Link>
          <Link href="/dashboard" className="text-xs font-bold text-neutral-500 hover:text-neutral-900">
            Kembali
          </Link>
        </div>

        {/* MAIN RESULT BODY AREA */}
        <div className="flex-1 overflow-y-auto px-6 py-5 relative z-10 space-y-4">
          
          {/* STATUS HEADER CARD */}
          <div className="p-5 rounded-3xl bg-neutral-50 border border-neutral-200/80 text-center space-y-3 shadow-xs">
            <div className="w-16 h-16 rounded-full bg-white border border-neutral-200 shadow-sm mx-auto flex items-center justify-center">
              {currentConfig.iconSvg}
            </div>

            <div>
              <span className={`inline-block text-[9px] font-extrabold px-2.5 py-0.5 rounded-full border mb-1 ${currentConfig.badgeClass}`}>
                {currentConfig.badgeText}
              </span>
              <h2 className="text-xl font-extrabold text-neutral-900 tracking-tight">{currentConfig.title}</h2>
              <p className="text-xs font-medium text-neutral-500 mt-0.5">{currentConfig.subtitle}</p>
            </div>
          </div>

          {/* MERCHANT & REPUTATION DETAILS */}
          <div className="p-4 rounded-3xl bg-white border border-neutral-200/80 shadow-xs space-y-3">
            <h3 className="text-xs font-extrabold text-neutral-900 tracking-tight border-b border-neutral-100 pb-2">
              Informasi Pemilik QRIS
            </h3>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-neutral-400 font-medium">Nama Merchant:</span>
                <span className="font-bold text-neutral-900">{merchantName}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-400 font-medium">Kategori:</span>
                <span className="font-semibold text-neutral-700">{currentConfig.qrisCategory}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-400 font-medium">Penyedia Jasa:</span>
                <span className="font-semibold text-neutral-700">{bankName}</span>
              </div>
              <div className="flex justify-between items-center pt-1 border-t border-neutral-100">
                <span className="text-neutral-400 font-medium">Skor Reputasi:</span>
                <span className="font-extrabold text-neutral-900">{reputationScore} ({currentConfig.reputationText})</span>
              </div>
            </div>
          </div>

          {/* ANALYSIS VERIFICATION CHECKLIST */}
          <div className="p-4 rounded-3xl bg-neutral-50 border border-neutral-200/80 shadow-xs space-y-2.5">
            <h3 className="text-xs font-extrabold text-neutral-900 tracking-tight">
              Hasil Analisis AI
            </h3>

            <ul className="space-y-2 text-[11px] font-medium text-neutral-600">
              {details.map((detail, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-emerald-500 font-bold shrink-0 mt-0.5">•</span>
                  <span>{detail}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* SIMULATION DEMO SWITCHER BUTTONS */}
          <div className="pt-2">
            <span className="text-[9px] font-extrabold text-neutral-400 uppercase tracking-wider block text-center mb-1.5">
              Coba Simulasi Status Lainnya:
            </span>
            <div className="grid grid-cols-3 gap-1.5">
              <Link
                href="/result/aman"
                className={`py-2 text-[10px] font-extrabold text-center rounded-xl border transition-all ${
                  type === "aman"
                    ? "bg-emerald-500 text-white border-emerald-500 shadow-sm"
                    : "bg-neutral-50 text-neutral-600 border-neutral-200 hover:bg-neutral-100"
                }`}
              >
                🟢 Aman
              </Link>
              <Link
                href="/result/waspada"
                className={`py-2 text-[10px] font-extrabold text-center rounded-xl border transition-all ${
                  type === "waspada"
                    ? "bg-amber-500 text-white border-amber-500 shadow-sm"
                    : "bg-neutral-50 text-neutral-600 border-neutral-200 hover:bg-neutral-100"
                }`}
              >
                🟡 Waspada
              </Link>
              <Link
                href="/result/bahaya"
                className={`py-2 text-[10px] font-extrabold text-center rounded-xl border transition-all ${
                  type === "bahaya"
                    ? "bg-rose-500 text-white border-rose-500 shadow-sm"
                    : "bg-neutral-50 text-neutral-600 border-neutral-200 hover:bg-neutral-100"
                }`}
              >
                🔴 Bahaya
              </Link>
            </div>
          </div>

        </div>

        {/* FOOTER MAIN ACTION BUTTON */}
        <div className="p-6 border-t border-neutral-100 bg-white relative z-10">
          <Link
            href="/dashboard"
            className={`w-full py-3.5 px-6 rounded-full font-extrabold text-xs transition-all flex items-center justify-center gap-2 shadow-md active:scale-[0.98] ${currentConfig.btnClass}`}
          >
            {currentConfig.buttonText}
          </Link>
        </div>

      </div>
    </div>
  );
}
