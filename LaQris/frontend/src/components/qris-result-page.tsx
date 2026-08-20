"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { ScanResponse } from "@/types/detection";

type ResultType = "aman" | "waspada" | "bahaya";

const resultConfig: Record<ResultType, {
  title: string;
  subtitle: string;
  accent: string;
  soft: string;
  border: string;
  action: string;
}> = {
  aman: {
    title: "QRIS Ini Aman Digunakan",
    subtitle: "Merchant terverifikasi resmi dan tidak ditemukan indikasi bahaya.",
    accent: "text-emerald-600",
    soft: "bg-emerald-50",
    border: "border-emerald-200",
    action: "Pindai QRIS Lain",
  },
  waspada: {
    title: "Perlu Perhatian Sebelum Membayar",
    subtitle: "Ditemukan perbedaan yang perlu Anda periksa kembali.",
    accent: "text-amber-600",
    soft: "bg-amber-50",
    border: "border-amber-200",
    action: "Laporkan Masalah",
  },
  bahaya: {
    title: "Indikasi Identitas Tidak Sesuai",
    subtitle: "Terdeteksi indikasi fraud atau ketidaksesuaian QRIS.",
    accent: "text-rose-600",
    soft: "bg-rose-50",
    border: "border-rose-200",
    action: "Lanjut Laporan",
  },
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:5000";

function valueOrFallback(value: string | undefined, fallback: string) {
  return value && value !== "Tidak ditemukan" && value !== "Tidak terbaca" ? value : fallback;
}

export default function QrisResultPage({ type }: { type: ResultType }) {
  const config = resultConfig[type];
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

  const trustScore = scanData ? Math.round(scanData.trust_score) : type === "aman" ? 98 : type === "waspada" ? 65 : 50;
  const physicalMerchant = valueOrFallback(scanData?.physical_merchant, type === "bahaya" ? "Tidak terbaca" : "Toko Berkah Jaya");
  const digitalMerchant = valueOrFallback(scanData?.digital_merchant, type === "bahaya" ? "Tidak ditemukan" : "Toko Berkah Jaya");
  const physicalNmid = valueOrFallback(scanData?.physical_nmid, "Tidak terbaca");
  const digitalNmid = valueOrFallback(scanData?.digital_nmid, "Tidak ditemukan");
  const physicalAcquirer = valueOrFallback(scanData?.physical_acquirer, "Tidak terbaca");
  const digitalAcquirer = valueOrFallback(scanData?.digital_acquirer, "Tidak ditemukan");
  const physicalTid = valueOrFallback(scanData?.physical_tid, "Tidak terbaca");
  const digitalTid = valueOrFallback(scanData?.digital_tid, "Tidak ditemukan");
  const visualizationUrl = scanData?.visualization_url ? `${API_BASE}${scanData.visualization_url}` : null;
  const explanation = scanData?.explanation || config.subtitle;
  const reportCount = typeof scanData?.reputation?.total_reports === "number" ? scanData.reputation.total_reports : type === "bahaya" ? 14 : 0;

  const actionHref = type === "bahaya" || type === "waspada" ? "/report" : "/dashboard";

  return (
    <main className="h-dvh overflow-hidden bg-slate-100 py-0 sm:py-4 flex items-center justify-center font-sans antialiased">
      <div className="w-full max-w-[620px] h-full sm:h-[calc(100dvh-2rem)] bg-white sm:rounded-[32px] shadow-xl overflow-hidden flex flex-col relative border-0 sm:border border-slate-200 text-slate-900">
        <header className="px-7 py-5 border-b border-slate-100 flex items-center justify-between shrink-0">
          <Link href="/dashboard" className="text-xl font-black tracking-tight">LàQris.</Link>
          <Link href="/dashboard" className="text-sm font-bold text-slate-500 hover:text-slate-900">Kembali</Link>
        </header>

        <div className="flex-1 overflow-y-auto px-7 py-7 space-y-6">
          <section className={`${config.soft} ${config.border} rounded-3xl border p-7 text-center`}>
            <div className={`mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-white shadow-sm ${config.accent}`}>
              {type === "aman" ? (
                <svg className="h-11 w-11" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="m5 12 4 4L19 6" /></svg>
              ) : (
                <svg className="h-11 w-11" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM12 16.5h.008v.008H12V16.5Z" /></svg>
              )}
            </div>
            <h1 className="text-3xl font-black tracking-tight">{config.title}</h1>
            <p className="mx-auto mt-2 max-w-[430px] text-base font-medium leading-relaxed text-slate-500">{config.subtitle}</p>
            <div className="mt-5 inline-flex rounded-full bg-white px-5 py-2 text-lg font-black text-slate-600 shadow-sm">
              Trust Score: {trustScore} / 100
            </div>
          </section>

          {scanData?.explanation && (
            <div className={`${config.soft} ${config.border} rounded-2xl border px-5 py-4 text-base font-medium leading-relaxed ${config.accent}`}>
              {explanation}
            </div>
          )}

          {visualizationUrl && (
            <section className="overflow-hidden rounded-3xl border border-slate-200 bg-slate-50">
              <img src={visualizationUrl} alt="Visualisasi hasil deteksi QRIS" className="mx-auto max-h-[390px] w-auto max-w-full object-contain" />
            </section>
          )}

          <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-3 flex items-center justify-between border-b border-slate-100 pb-3">
              <h2 className="text-lg font-black">Informasi Kode QRIS</h2>
              <span className={`text-sm font-bold ${config.accent}`}>{type === "aman" ? "Resmi" : "Perlu Perhatian"}</span>
            </div>
            <div className="divide-y divide-slate-100 text-base">
              <div className="flex items-start justify-between gap-5 py-3"><span className="text-slate-500">Merchant Terlihat (Fisik)</span><strong className="text-right">{physicalMerchant}</strong></div>
              <div className="flex items-start justify-between gap-5 py-3"><span className="text-slate-500">Penerima QRIS (Digital)</span><strong className={`text-right ${digitalMerchant === "Tidak ditemukan" ? "text-blue-600" : ""}`}>{digitalMerchant}</strong></div>
              <div className="flex items-start justify-between gap-5 py-3"><span className="text-slate-500">NMID Fisik / Digital</span><strong className="text-right font-mono text-sm">{physicalNmid} / {digitalNmid}</strong></div>
              <div className="flex items-start justify-between gap-5 py-3"><span className="text-slate-500">Acquirer Fisik / Digital</span><strong className="text-right">{physicalAcquirer} / {digitalAcquirer}</strong></div>
              <div className="flex items-start justify-between gap-5 py-3"><span className="text-slate-500">Terminal ID Fisik / Digital</span><strong className="text-right font-mono text-sm">{physicalTid} / {digitalTid}</strong></div>
              <div className="flex items-start justify-between gap-5 py-3"><span className="text-slate-500">Reputasi SQLite</span><strong className="text-right">{type === "bahaya" ? "Berisiko" : "Rating baik"} ({reportCount} laporan)</strong></div>
            </div>
          </section>

          <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
            <h2 className="mb-4 text-lg font-black">Hasil Analisis AI</h2>
            <div className="space-y-3">
              <div className="rounded-2xl bg-white p-4"><strong className="block text-base">1. Keaslian Fisik</strong><span className="mt-1 block text-sm text-slate-500">{type === "bahaya" ? "Terdeteksi indikasi stiker QRIS palsu atau ditimpa." : "QRIS terverifikasi dan tidak ditemukan indikasi stiker timpa."}</span></div>
              <div className="rounded-2xl bg-white p-4"><strong className="block text-base">2. Kesesuaian Data</strong><span className="mt-1 block text-sm text-slate-500">{scanData ? `${scanData.match_level} dengan skor kecocokan ${Math.round(scanData.name_similarity)}%.` : "Perbandingan data fisik dan digital tersedia setelah pemindaian."}</span></div>
              <div className="rounded-2xl bg-white p-4"><strong className="block text-base">3. Reputasi QRIS</strong><span className="mt-1 block text-sm text-slate-500">{type === "bahaya" ? `Ditemukan ${reportCount} laporan atau indikasi yang perlu ditindaklanjuti.` : "Tidak ditemukan laporan berbahaya pada pemeriksaan ini."}</span></div>
            </div>
          </section>
        </div>

        <footer className="shrink-0 border-t border-slate-100 bg-white px-7 py-5">
          <Link href={actionHref} className={`flex w-full items-center justify-center rounded-full py-4 text-base font-black text-white shadow-md transition-transform active:scale-[0.98] ${type === "aman" ? "bg-slate-900 hover:bg-slate-800" : type === "waspada" ? "bg-amber-500 hover:bg-amber-600" : "bg-rose-600 hover:bg-rose-700"}`}>
            {config.action}
          </Link>
          {type !== "aman" && <Link href="/dashboard" className="mt-3 block text-center text-sm font-bold text-slate-500 hover:text-slate-900">Scan Stiker Lain</Link>}
        </footer>
      </div>
    </main>
  );
}
