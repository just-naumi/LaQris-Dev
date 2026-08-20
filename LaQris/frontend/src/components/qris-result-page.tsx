"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { ScanResponse } from "@/types/detection";

type ResultType = "aman" | "waspada" | "bahaya";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:5000";

function displayValue(value: string | undefined, fallback: string) {
  return value && value !== "Tidak ditemukan" && value !== "Tidak terbaca" ? value : fallback;
}

export default function QrisResultPage({ type }: { type: ResultType }) {
  const [scanData, setScanData] = useState<ScanResponse | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const stored = sessionStorage.getItem("laqris:last-scan");
      if (!stored) return;
      try {
        setScanData(JSON.parse(stored) as ScanResponse);
      } catch {
        sessionStorage.removeItem("laqris:last-scan");
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const isSafe = type === "aman";
  const isDanger = type === "bahaya";
  const trustScore = scanData ? Math.round(scanData.trust_score) : isSafe ? 100 : isDanger ? 50 : 65;
  const physicalMerchant = displayValue(scanData?.physical_merchant, isDanger ? "Tidak terbaca" : "Toko Berkah Jaya");
  const digitalMerchant = displayValue(scanData?.digital_merchant, isDanger ? "Tidak ditemukan" : "Toko Berkah Jaya");
  const physicalNmid = displayValue(scanData?.physical_nmid, "Tidak terbaca");
  const digitalNmid = displayValue(scanData?.digital_nmid, "Tidak ditemukan");
  const city = displayValue(scanData?.digital_city, "Tidak ditemukan");
  const acquirer = displayValue(scanData?.digital_acquirer, "Tidak ditemukan");
  const terminalId = displayValue(scanData?.digital_tid, "Tidak ditemukan");
  const visualizationUrl = scanData?.visualization_url ? `${API_BASE}${scanData.visualization_url}` : null;
  const reportCount = typeof scanData?.reputation?.total_reports === "number" ? scanData.reputation.total_reports : isDanger ? 14 : 0;
  const explanation = scanData?.explanation || (isSafe ? "Identitas QRIS sesuai dan tidak ditemukan indikasi bahaya." : "Terdapat ketidaksesuaian identitas yang perlu diperiksa.");
  const matchText = scanData ? `${Math.round(scanData.name_similarity)}% (${scanData.match_level})` : isSafe ? "100% (EXACT_MATCH)" : "0% (COMPLETELY_DIFFERENT)";

  return (
    <main className="h-dvh overflow-hidden bg-slate-100 flex items-center justify-center font-sans antialiased">
      <div className="w-full max-w-[390px] h-full sm:h-[min(844px,calc(100dvh-4rem))] bg-white sm:rounded-[22px] shadow-xl overflow-hidden flex flex-col relative border border-slate-200 text-slate-900">
        <header className="h-[46px] shrink-0 border-b border-slate-100 flex items-center justify-center">
          <h1 className="text-xs font-extrabold">Status Verifikasi QRIS</h1>
        </header>

        <div className="scrollbar-hidden flex-1 overflow-y-auto px-3 py-3 space-y-2.5">
          <section className={`rounded-2xl border p-3.5 text-center ${isSafe ? "bg-emerald-50 border-emerald-200" : isDanger ? "bg-rose-50 border-rose-200" : "bg-amber-50 border-amber-200"}`}>
            <div className={`mx-auto mb-2 flex h-11 w-11 items-center justify-center rounded-full bg-white shadow-sm ${isSafe ? "text-emerald-500" : isDanger ? "text-rose-600" : "text-amber-500"}`}>
              {isSafe ? (
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="m5 12 4 4L19 6" /></svg>
              ) : (
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM12 16.5h.008v.008H12V16.5Z" /></svg>
              )}
            </div>
            <h2 className="text-sm font-black tracking-tight">{isSafe ? "Verifikasi Berhasil (Aman)" : isDanger ? "Indikasi Identitas Tidak Sesuai" : "Perlu Perhatian"}</h2>
            <div className="mt-1 inline-flex rounded-full bg-white px-2.5 py-1 text-[9px] font-black text-slate-600 shadow-sm">Trust Score: {trustScore} / 100</div>
          </section>

          <div className={`rounded-xl border px-3 py-2 text-[9px] leading-relaxed ${isSafe ? "bg-slate-50 border-slate-200 text-slate-500" : "bg-rose-50 border-rose-200 text-rose-600"}`}>
            {!isSafe && <span className="mr-1">⚠</span>}{explanation}
          </div>

          {visualizationUrl && (
            <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
              <img src={visualizationUrl} alt="Visual Deteksi YOLO" className="mx-auto h-[165px] w-full object-contain" />
            </div>
          )}

          <section className="overflow-hidden rounded-xl border border-slate-200 bg-white">
            <div className="divide-y divide-slate-100 text-[9px]">
              <div className="flex items-start justify-between gap-3 px-2 py-2"><span className="text-slate-500">Merchant Terlihat<br />(Fisik)</span><strong className={`max-w-[58%] text-right ${isDanger ? "text-rose-600" : ""}`}>{physicalMerchant}</strong></div>
              <div className="flex items-start justify-between gap-3 px-2 py-2"><span className="text-slate-500">Penerima QRIS (Digital)</span><strong className="max-w-[58%] text-right text-blue-600">{digitalMerchant}</strong></div>
              <div className="flex items-start justify-between gap-3 px-2 py-2"><span className="text-slate-500">Kesesuaian Identitas</span><strong className="max-w-[58%] text-right">{matchText}</strong></div>
              <div className="flex items-start justify-between gap-3 px-2 py-2"><span className="text-slate-500">NMID Fisik</span><strong className="text-right font-mono">{physicalNmid}</strong></div>
              <div className="flex items-start justify-between gap-3 px-2 py-2"><span className="text-slate-500">NMID Digital</span><strong className="text-right font-mono">{digitalNmid}</strong></div>
              <div className="flex items-start justify-between gap-3 px-2 py-2"><span className="text-slate-500">Kota Merchant (EMV)</span><strong className="text-right">{city}</strong></div>
              <div className="flex items-start justify-between gap-3 px-2 py-2"><span className="text-slate-500">Acquirer Bank</span><strong className="text-right">{acquirer}</strong></div>
              <div className="flex items-start justify-between gap-3 px-2 py-2"><span className="text-slate-500">Terminal ID</span><strong className="text-right font-mono">{terminalId}</strong></div>
              <div className="flex items-start justify-between gap-3 px-2 py-2"><span className="text-slate-500">Reputasi SQLite</span><strong className="text-right">Rating 5.0 / 5.0 ({reportCount} Laporan)</strong></div>
            </div>
          </section>

          <section className="rounded-xl border border-slate-200 bg-slate-50 p-3">
            <h2 className="mb-2 text-[10px] font-black">Hasil Analisis AI</h2>
            <div className="space-y-1.5 text-[9px] leading-relaxed text-slate-500">
              <p><strong className="text-slate-900">1. Keaslian Fisik</strong><br />{isDanger ? "Terdeteksi indikasi stiker QRIS palsu atau ditimpa." : "QRIS terverifikasi original, tidak ada stiker timpa."}</p>
              <p><strong className="text-slate-900">2. Kesesuaian Data</strong><br />{scanData ? `${scanData.match_level} dengan skor kecocokan ${Math.round(scanData.name_similarity)}%.` : "Data fisik sesuai dengan data digital."}</p>
              <p><strong className="text-slate-900">3. Reputasi QRIS</strong><br />{isDanger ? `Reputasi perlu diperiksa dengan ${reportCount} laporan.` : "Reputasi baik dan bebas dari laporan bahaya."}</p>
            </div>
          </section>
        </div>

        <footer className="shrink-0 border-t border-slate-100 bg-white px-3 py-3">
          <Link href={isDanger || type === "waspada" ? "/report" : "/dashboard"} className={`flex w-full items-center justify-center rounded-xl py-2.5 text-[10px] font-black text-white shadow-sm ${isSafe ? "bg-slate-900" : isDanger ? "bg-rose-600" : "bg-amber-500"}`}>
            {isSafe ? "Scan Stiker Lain" : isDanger ? "Batalkan Pembayaran" : "Laporkan Masalah"}
          </Link>
          {!isSafe && <Link href="/dashboard" className="mt-2 block text-center text-[9px] font-bold text-slate-500">Scan Stiker Lain</Link>}
        </footer>
      </div>
    </main>
  );
}
