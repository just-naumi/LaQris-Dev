"use client";

import Link from "next/link";

export default function AmanResultPage() {
  return (
    <div className="min-h-screen bg-neutral-900 py-0 sm:py-8 flex items-center justify-center font-sans antialiased">
      {/* Mobile Screen Container */}
      <div className="w-full max-w-[390px] h-[844px] bg-base-100 sm:rounded-[44px] shadow-2xl overflow-hidden flex flex-col justify-between relative border-0 sm:border-[8px] border-neutral-800 text-base-content">
        
        {/* Soft Background Decorative Circles */}
        <div className="absolute -top-12 -left-12 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute top-1/3 -right-16 w-60 h-60 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />

        {/* TOP NAVBAR */}
        <div className="navbar bg-base-100/90 backdrop-blur border-b border-base-200 px-6 min-h-[58px] z-10">
          <div className="flex-1">
            <Link href="/dashboard" className="text-2xl font-black tracking-tight text-base-content">
              LàQris.
            </Link>
          </div>
          <div className="flex-none">
            <Link href="/dashboard" className="btn btn-ghost btn-xs text-xs font-bold text-base-content/60 hover:text-base-content">
              Kembali
            </Link>
          </div>
        </div>

        {/* MAIN RESULT BODY AREA */}
        <div className="flex-1 overflow-y-auto px-6 py-4 relative z-10 space-y-4">
          
          {/* STATUS BANNER CLEAN */}
          <div className="card bg-emerald-500/10 border border-emerald-500/20 shadow-xs rounded-3xl p-5 text-center space-y-2">
            <div className="w-14 h-14 rounded-full bg-emerald-500 text-white flex items-center justify-center mx-auto shadow-md shadow-emerald-500/30">
              <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
              </svg>
            </div>

            <div>
              <h2 className="text-xl font-black text-base-content tracking-tight">
                QRIS Ini Aman Digunakan
              </h2>
              <p className="text-xs font-medium text-base-content/60 mt-1 max-w-[240px] mx-auto leading-relaxed">
                Merchant terverifikasi resmi dan tidak ditemukan indikasi bahaya.
              </p>
            </div>
          </div>

          {/* STATISTIK REPUTASI QRIS (3 COLUMNS) */}
          <div className="grid grid-cols-3 gap-2 bg-base-200/50 p-2.5 rounded-3xl border border-base-300">
            <div className="text-center p-2 rounded-2xl bg-base-100 border border-base-200/60 shadow-2xs">
              <span className="text-[9px] font-extrabold text-base-content/50 uppercase tracking-wide block">Transaksi</span>
              <span className="text-sm font-black text-base-content mt-0.5 block">1.420</span>
            </div>
            
            <div className="text-center p-2 rounded-2xl bg-base-100 border border-base-200/60 shadow-2xs">
              <span className="text-[9px] font-extrabold text-base-content/50 uppercase tracking-wide block">Laporan</span>
              <span className="text-sm font-black text-emerald-600 mt-0.5 block">0</span>
            </div>
            
            <div className="text-center p-2 rounded-2xl bg-base-100 border border-base-200/60 shadow-2xs">
              <span className="text-[9px] font-extrabold text-base-content/50 uppercase tracking-wide block">Reputasi</span>
              <span className="text-sm font-black text-emerald-600 mt-0.5 block">98%</span>
            </div>
          </div>

          {/* DETAIL INFORMASI KODE QRIS */}
          <div className="card bg-base-100 border border-base-300 shadow-2xs rounded-3xl">
            <div className="card-body p-4 space-y-2.5">
              <div className="flex items-center justify-between border-b border-base-200 pb-2">
                <h3 className="text-xs font-black text-base-content tracking-tight">
                  Informasi Kode QRIS
                </h3>
                <span className="text-[10px] font-bold text-emerald-600">Resmi</span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-base-content/60 font-medium">Nama Merchant:</span>
                  <span className="font-extrabold text-base-content">TOKO BERKAH JAYA</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-base-content/60 font-medium">NMID:</span>
                  <span className="font-mono font-semibold text-base-content/80">ID1024598124001</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-base-content/60 font-medium">Penyedia Jasa:</span>
                  <span className="font-semibold text-base-content/80">BCA (Bank Central Asia)</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-base-content/60 font-medium">ID Terminal (TID):</span>
                  <span className="font-mono font-semibold text-base-content/80">A01-09823412</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-base-content/60 font-medium">Kategori Usaha:</span>
                  <span className="font-semibold text-base-content/80">UMI (Usaha Mikro)</span>
                </div>
              </div>
            </div>
          </div>

          {/* HASIL ANALISIS AI (3 POIN SANTAI - CLEAN TEXT) */}
          <div className="card bg-base-200/50 border border-base-300 shadow-2xs rounded-3xl">
            <div className="card-body p-4 space-y-2.5">
              <h3 className="text-xs font-black text-base-content tracking-tight">
                Hasil Analisis AI
              </h3>

              <div className="space-y-2 text-[11px] font-medium text-base-content/80">
                <div className="p-3 rounded-2xl bg-base-100 border border-base-200/80 space-y-0.5">
                  <strong className="text-base-content font-bold block text-xs">1. Keaslian Fisik</strong>
                  <span className="text-base-content/70">QRIS terverifikasi original, tidak ada stiker timpa.</span>
                </div>

                <div className="p-3 rounded-2xl bg-base-100 border border-base-200/80 space-y-0.5">
                  <strong className="text-base-content font-bold block text-xs">2. Kesesuaian Data</strong>
                  <span className="text-base-content/70">Nama cetakan kertas fisik sesuai 100% dengan data digital.</span>
                </div>

                <div className="p-3 rounded-2xl bg-base-100 border border-base-200/80 space-y-0.5">
                  <strong className="text-base-content font-bold block text-xs">3. Reputasi QRIS</strong>
                  <span className="text-base-content/70">Reputasi sangat baik &amp; bebas dari laporan bahaya.</span>
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* FOOTER ACTION BUTTON */}
        <div className="p-6 border-t border-base-200 bg-base-100 relative z-10">
          <Link
            href="/dashboard"
            className="w-full py-3.5 px-6 rounded-full border border-neutral-900 bg-neutral-900 hover:bg-neutral-800 text-white font-bold text-xs transition-all duration-200 active:scale-[0.98] shadow-md flex items-center justify-center"
          >
            Lanjut Laporan
          </Link>
        </div>

      </div>
    </div>
  );
}
