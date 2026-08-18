"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function ScanAnimationPage() {
  const router = useRouter();
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState("Menganalisis Kode QRIS...");

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 5;
      });
    }, 100);

    const t1 = setTimeout(() => setStatusText("Memeriksa Database Rekening..."), 1000);
    const t2 = setTimeout(() => setStatusText("Verifikasi Reputasi Merchant..."), 2000);

    const t3 = setTimeout(() => {
      const results = ["aman", "waspada", "bahaya"];
      const randomResult = results[Math.floor(Math.random() * results.length)];
      router.push(`/result/${randomResult}`);
    }, 2800);

    return () => {
      clearInterval(interval);
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [router]);

  return (
    <div className="min-h-screen bg-neutral-900 py-0 sm:py-8 flex items-center justify-center font-sans antialiased">
      {/* Mobile Screen Container */}
      <div className="w-full max-w-[390px] h-[844px] bg-neutral-950 sm:rounded-[44px] shadow-2xl overflow-hidden flex flex-col justify-between p-8 relative border-0 sm:border-[8px] border-neutral-800 text-white">
        
        {/* Soft Background Decorative Circles */}
        <div className="absolute -top-12 -left-12 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute top-1/2 -right-16 w-60 h-60 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />

        {/* BRAND LOGO & DAISYUI BADGE */}
        <div className="pt-6 relative z-10 flex items-center justify-between">
          <h1 className="text-3xl font-black tracking-tight text-white">
            LàQris.
          </h1>
          <span className="badge badge-success badge-outline gap-1 text-[10px] font-extrabold py-2 animate-pulse">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            SCANNING...
          </span>
        </div>

        {/* MIDDLE SCANNING ANIMATION CONTAINER */}
        <div className="my-auto space-y-6 relative z-10 flex flex-col items-center w-full">
          
          {/* CAMERA SCANNER VIEWFINDER BOX WITH LASER ANIMATION */}
          <div className="relative w-full aspect-square max-w-[280px] rounded-3xl bg-neutral-900 border-2 border-neutral-800 overflow-hidden shadow-2xl flex items-center justify-center">
            
            {/* Viewfinder Corners */}
            <div className="absolute top-3 left-3 w-5 h-5 border-t-2 border-l-2 border-emerald-400 rounded-tl" />
            <div className="absolute top-3 right-3 w-5 h-5 border-t-2 border-r-2 border-emerald-400 rounded-tr" />
            <div className="absolute bottom-3 left-3 w-5 h-5 border-b-2 border-l-2 border-emerald-400 rounded-bl" />
            <div className="absolute bottom-3 right-3 w-5 h-5 border-b-2 border-r-2 border-emerald-400 rounded-br" />

            {/* Mock QR Code Pattern Background */}
            <div className="w-40 h-40 opacity-25 flex flex-wrap justify-between p-2">
              <div className="w-12 h-12 border-4 border-white rounded-lg p-1">
                <div className="w-full h-full bg-white rounded-xs" />
              </div>
              <div className="w-12 h-12 border-4 border-white rounded-lg p-1">
                <div className="w-full h-full bg-white rounded-xs" />
              </div>
              <div className="w-12 h-12 border-4 border-white rounded-lg p-1 mt-auto">
                <div className="w-full h-full bg-white rounded-xs" />
              </div>
            </div>

            {/* MOVING LASER SCAN LINE */}
            <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-emerald-400 to-transparent shadow-[0_0_15px_#34d399] animate-[bounce_2s_infinite]" />
            <div className="absolute inset-0 bg-emerald-500/5 animate-pulse" />
          </div>

          {/* STATUS TEXT & DAISYUI PROGRESS BAR */}
          <div className="w-full space-y-3 text-center px-2">
            <p className="text-xs font-bold text-white tracking-wide flex items-center justify-center gap-2">
              <span className="loading loading-dots loading-xs text-emerald-400"></span>
              {statusText}
            </p>

            {/* DaisyUI Progress Component */}
            <progress className="progress progress-success w-full h-2" value={progress} max="100"></progress>
            <p className="text-[10px] font-semibold text-neutral-500">{progress}% Selesai</p>
          </div>

        </div>

        {/* FOOTER INSTRUCTION */}
        <div className="pb-6 text-center relative z-10">
          <p className="text-[11px] font-medium text-neutral-500">
            Harap tunggu sebentar, AI sedang bekerja.
          </p>
        </div>

      </div>
    </div>
  );
}
