"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { detectImage } from "@/lib/api";
import type { DetectionResponse } from "@/types/detection";

interface HistoryItem {
  id: string;
  merchantName: string;
  reputationScore: string;
  textColor: string;
  timeAgo: string;
}

export default function DashboardPage() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const cameraInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 7 Days Data (Senin - Minggu)
  const daysData = [
    { day: "Sen", height: 45, isToday: false },
    { day: "Sel", height: 70, isToday: false },
    { day: "Rab", height: 35, isToday: false },
    { day: "Kam", height: 90, isToday: true },
    { day: "Jum", height: 60, isToday: false },
    { day: "Sab", height: 80, isToday: false },
    { day: "Min", height: 50, isToday: false },
  ];

  // Dynamic History Aktivitas QRIS Data
  const [qrisHistory, setQrisHistory] = useState<HistoryItem[]>([
    {
      id: "1",
      merchantName: "Toko Berkah Jaya",
      reputationScore: "98% · Sangat Baik",
      textColor: "text-emerald-600",
      timeAgo: "10m lalu",
    },
    {
      id: "2",
      merchantName: "Kedai Kopi Utama",
      reputationScore: "95% · Tepercaya",
      textColor: "text-amber-500",
      timeAgo: "1j lalu",
    },
    {
      id: "3",
      merchantName: "Warung Sederhana Abadi",
      reputationScore: "99% · Sangat Tinggi",
      textColor: "text-emerald-600",
      timeAgo: "3j lalu",
    },
  ]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setErrorMsg(null);
      router.push("/scan");
    }
  };

  const handleRunDetection = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const data = await detectImage(selectedFile);
      setResult(data);

      const firstLabel = data.detections[0]?.label || "Pemindaian Objek";
      const newEntry: HistoryItem = {
        id: Date.now().toString(),
        merchantName: `Scan: ${firstLabel.toUpperCase()}`,
        reputationScore: `${Math.round((data.detections[0]?.confidence || 0.95) * 100)}% · Terverifikasi`,
        textColor: "text-emerald-600",
        timeAgo: "Baru saja",
      };
      setQrisHistory((prev) => [newEntry, ...prev.slice(0, 3)]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Gagal memproses gambar";
      setErrorMsg(msg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-neutral-900 py-0 sm:py-8 flex items-center justify-center font-sans antialiased">
      {/* Mobile Screen Container */}
      <div className="w-full max-w-[390px] h-[844px] bg-base-100 sm:rounded-[44px] shadow-2xl overflow-hidden flex flex-col justify-between relative border-0 sm:border-[8px] border-neutral-800 text-base-content">
        
        {/* Soft Background Decorative Circles */}
        <div className="absolute -top-12 -left-12 w-64 h-64 bg-base-200/80 rounded-full blur-2xl pointer-events-none" />
        <div className="absolute top-1/3 -right-16 w-60 h-60 bg-base-200/60 rounded-full blur-2xl pointer-events-none" />

        {/* DAISYUI NAVBAR */}
        <div className="navbar bg-base-100/90 backdrop-blur border-b border-base-200 px-6 min-h-[58px] z-10">
          <div className="flex-1">
            <h1 className="text-2xl font-black tracking-tight text-base-content">
              LàQris.
            </h1>
          </div>
          <div className="flex-none flex items-center gap-1.5">
            <button
              type="button"
              aria-label="Notifikasi"
              className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-base-content transition-colors hover:bg-base-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral focus-visible:ring-offset-2"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M14.857 17.082a23.848 23.848 0 005.454-1.31A8.967 8.967 0 0118 9.75v-.7V9A6 6 0 006 9v.75a8.967 8.967 0 01-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 01-5.714 0m5.714 0a3 3 0 11-5.714 0" />
              </svg>
              <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-emerald-500 ring-2 ring-base-100" />
            </button>

            {/* DaisyUI Avatar */}
            <div className="relative group">
              <Link
                href="/"
                aria-label="Keluar dari aplikasi"
                className="flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-full bg-neutral text-neutral-content focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral focus-visible:ring-offset-2"
              >
                <span className="block text-[10px] font-bold leading-none tracking-tight">JD</span>
              </Link>
              <span className="pointer-events-none absolute right-0 top-full z-30 mt-2 w-max max-w-[150px] translate-y-1 rounded-md bg-neutral px-2.5 py-1.5 text-center text-[10px] font-medium text-neutral-content opacity-0 shadow-lg transition-all duration-150 group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:translate-y-0 group-focus-within:opacity-100">
                Klik untuk keluar
              </span>
            </div>
          </div>
        </div>

        {/* MAIN BODY AREA */}
        <div className="flex-1 px-6 py-4 relative z-10 space-y-3.5 overflow-y-auto">

          {/* PERMANENT 1/4 DISPLAY BANNER WITH DAISYUI STAT OVERLAY */}
          <div className="relative w-full h-[135px] rounded-3xl bg-neutral-950 overflow-hidden shadow-lg group shrink-0">
            <img
              src="https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=800&auto=format&fit=crop"
              alt="Visual Background Display"
              className="w-full h-full object-cover opacity-30 scale-105"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-neutral-950 via-neutral-950/60 to-transparent" />
            
            <div className="absolute inset-0 p-3.5 flex flex-col justify-between z-10 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[9px] font-bold text-neutral-400 uppercase tracking-wider block">Grafik Activity</span>
                  <p className="text-xs font-extrabold text-white mt-0.5">Aktivitas 7 Hari Terakhir</p>
                </div>
                {/* DaisyUI Badge */}
                <div className="badge badge-success badge-outline text-[8px] font-bold py-1">
                  7 Hari
                </div>
              </div>

              {/* 7-DAY BOLD BARS */}
              <div className="w-full h-12 flex items-end justify-between gap-2 pt-1">
                {daysData.map((item, idx) => (
                  <div key={idx} className="flex-1 flex flex-col items-center gap-1 h-full justify-end">
                    <div className="w-full bg-white/10 rounded-lg h-8 flex items-end overflow-hidden">
                      <div
                        className={`w-full rounded-lg transition-all duration-300 ${
                          item.isToday
                            ? "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]"
                            : "bg-white/80 hover:bg-emerald-400/80"
                        }`}
                        style={{ height: `${item.height}%` }}
                      />
                    </div>
                    <span className={`text-[8px] font-bold ${item.isToday ? "text-emerald-400" : "text-neutral-400"}`}>
                      {item.day}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* HIDDEN INPUTS */}
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileChange}
            className="hidden"
          />
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />

          {/* DAISYUI CARD: PEMINDAI VISUAL AI */}
          <div className="card bg-base-200/50 border border-base-300 shadow-sm rounded-3xl">
            <div className="card-body p-4 space-y-2.5">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="font-extrabold text-xs text-base-content">Pemindai Visual AI</h3>
                  <p className="text-[10px] font-medium text-base-content/60">Ambil foto atau pilih dari file</p>
                </div>
                {/* DaisyUI Badge */}
                <span className="badge badge-neutral badge-sm font-bold text-[9px]">
                  Active
                </span>
              </div>

              {/* DASHED BOX */}
              <div className="w-full py-4 rounded-2xl border-2 border-dashed border-base-300 flex flex-col items-center justify-center bg-base-100 space-y-1 overflow-hidden relative min-h-[95px]">
                {previewUrl ? (
                  <div className="relative w-full h-24">
                    <img src={previewUrl} alt="Preview" className="w-full h-full object-contain" />
                    <button
                      onClick={() => {
                        setSelectedFile(null);
                        setPreviewUrl(null);
                      }}
                      className="btn btn-xs btn-neutral rounded-full absolute top-1 right-1 font-bold text-[9px]"
                    >
                      Reset
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="w-8 h-8 rounded-full bg-base-200 flex items-center justify-center text-base-content/50">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <p className="text-xs font-bold text-base-content">Pilih Foto Gambar</p>
                    <p className="text-[9px] font-medium text-base-content/50">Format JPG, PNG, WebP</p>
                  </>
                )}
              </div>

              {/* DAISYUI PILL BUTTONS */}
              <div className="space-y-1.5 pt-1">
                <button
                  onClick={() => cameraInputRef.current?.click()}
                  className="btn btn-outline btn-neutral btn-block rounded-full btn-sm text-xs font-bold gap-2"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.039l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z" />
                  </svg>
                  Ambil Foto Kamera
                </button>

                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn btn-ghost bg-base-100 hover:bg-base-200 border border-base-300 btn-block rounded-full btn-sm text-xs font-semibold gap-2 shadow-2xs"
                >
                  <svg className="w-3.5 h-3.5 text-base-content/70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
                  </svg>
                  Masukkan Gambar dari File
                </button>
              </div>

              {/* START DETECTION BUTTON */}
              {selectedFile && (
                <button
                  onClick={handleRunDetection}
                  disabled={isLoading}
                  className="btn btn-neutral btn-block rounded-full btn-sm text-xs font-extrabold gap-2 shadow-md mt-1"
                >
                  {isLoading ? (
                    <span className="loading loading-spinner loading-xs"></span>
                  ) : (
                    "✨ Mulai Pindai Gambar"
                  )}
                </button>
              )}

              {errorMsg && (
                <div className="alert alert-error text-[11px] p-2 rounded-xl">
                  <span>{errorMsg}</span>
                </div>
              )}
            </div>
          </div>

          {/* DYNAMIC HISTORY AKTIVITAS WITH DAISYUI CARDS */}
          <div className="space-y-2 pb-2">
            <div className="flex justify-between items-center px-1">
              <h3 className="font-extrabold text-xs text-base-content tracking-tight">History Aktivitas</h3>
              {qrisHistory.length > 0 && (
                <span className="text-[10px] font-semibold text-base-content/50 hover:text-base-content cursor-pointer">Lihat Semua</span>
              )}
            </div>

            {qrisHistory.length === 0 ? (
              <div className="card bg-base-200/50 border border-base-300 p-5 text-center space-y-1.5 rounded-3xl">
                <div className="w-8 h-8 rounded-full bg-base-300/60 mx-auto flex items-center justify-center text-base-content/40">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <p className="text-xs font-bold text-base-content">Belum Ada Aktivitas</p>
                <p className="text-[10px] font-medium text-base-content/50">Riwayat pemindaian Anda akan muncul di sini</p>
              </div>
            ) : (
              <div className="space-y-2">
                {qrisHistory.map((item) => (
                  <div
                    key={item.id}
                    className="p-3 rounded-2xl bg-base-200/50 border border-base-300/80 flex items-center justify-between shadow-2xs transition-all hover:bg-base-200"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="w-10 h-10 rounded-xl bg-base-100 border border-base-300 flex items-center justify-center text-base-content shrink-0 shadow-2xs">
                        <svg className="w-5 h-5 text-base-content/80" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 4.875c0-.621.504-1.125 1.125-1.125h4.5c.621 0 1.125.504 1.125 1.125v4.5c0 .621-.504 1.125-1.125 1.125h-4.5A1.125 1.125 0 013.75 9.375v-4.5zM3.75 14.625c0-.621.504-1.125 1.125-1.125h4.5c.621 0 1.125.504 1.125 1.125v4.5c0 .621-.504 1.125-1.125 1.125h-4.5a1.125 1.125 0 01-1.125-1.125v-4.5zM13.5 4.875c0-.621.504-1.125 1.125-1.125h4.5c.621 0 1.125.504 1.125 1.125v4.5c0 .621-.504 1.125-1.125 1.125h-4.5A1.125 1.125 0 0113.5 9.375v-4.5z" />
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6.75 6.75h.008v.008H6.75V6.75zM6.75 16.5h.008v.008H6.75V16.5zM16.5 6.75h.008v.008H16.5V6.75zM13.5 13.5h1.5v1.5h-1.5v-1.5zM16.5 13.5h3v1.5h-3v-1.5zM13.5 16.5h1.5v3h-1.5v-3zM16.5 18h3v1.5h-3V18z" />
                        </svg>
                      </div>
                      <div className="truncate">
                        <p className="text-xs font-bold text-base-content truncate">{item.merchantName}</p>
                        <p className="text-[10px] font-medium text-base-content/60 mt-0.5">
                          Reputasi: <span className={`font-bold ${item.textColor}`}>{item.reputationScore}</span>
                        </p>
                      </div>
                    </div>
                    <span className="text-[9px] font-medium text-base-content/40 shrink-0 pl-2">
                      {item.timeAgo}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>

      </div>
    </div>
  );
}
