"use client";

import Link from "next/link";
import { useState } from "react";

const reportTypes = [
  "QRIS palsu atau ditimpa",
  "Nama merchant tidak sesuai",
  "Rekening mencurigakan",
  "Saya mengalami kerugian",
];

export default function ReportPage() {
  const [reportType, setReportType] = useState(reportTypes[0]);
  const [description, setDescription] = useState("");
  const [evidenceName, setEvidenceName] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitted(true);
  };

  return (
    <main className="min-h-screen bg-neutral-900 py-0 sm:py-8 flex items-center justify-center font-sans antialiased">
      <div className="w-full max-w-[390px] h-[844px] bg-white sm:rounded-[44px] shadow-2xl overflow-hidden flex flex-col relative border-0 sm:border-[8px] border-neutral-800 text-neutral-900">
        <header className="px-6 pt-6 pb-4 border-b border-neutral-100 flex items-center justify-between">
          <Link href="/result/bahaya" className="text-xs font-bold text-neutral-500 hover:text-neutral-900">
            Kembali
          </Link>
          <h1 className="text-sm font-extrabold">Laporkan QRIS</h1>
          <span className="w-12" aria-hidden="true" />
        </header>

        <div className="scrollbar-hidden flex-1 overflow-y-auto px-6 py-5 space-y-4">
          {submitted ? (
            <section className="min-h-full flex flex-col items-center justify-center text-center space-y-4">
              <div className="w-16 h-16 rounded-full bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-600 text-2xl">
                ✓
              </div>
              <div>
                <h2 className="text-xl font-extrabold">Laporan tersimpan</h2>
                <p className="text-xs text-neutral-500 mt-2 leading-relaxed">
                  Terima kasih. Laporan Anda akan ditinjau oleh tim LaQris.
                </p>
              </div>
              <Link href="/dashboard" className="w-full rounded-full bg-neutral-900 text-white py-3.5 text-xs font-extrabold">
                Kembali ke Beranda
              </Link>
            </section>
          ) : (
            <>
              <section className="rounded-3xl border border-rose-200 bg-rose-50 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-[10px] font-extrabold uppercase tracking-wider text-rose-600">Hasil pemindaian</p>
                    <h2 className="text-base font-extrabold text-rose-800 mt-1">Status: Bahaya</h2>
                  </div>
                  <span className="rounded-full bg-white px-3 py-1 text-[10px] font-bold text-rose-700 border border-rose-200">Perlu ditinjau</span>
                </div>
                <p className="text-xs text-rose-800/80 mt-3 leading-relaxed">
                  Laporkan detail ini agar tim dapat menindaklanjuti QRIS yang mencurigakan.
                </p>
              </section>

              <form onSubmit={handleSubmit} className="space-y-4">
                <section className="rounded-3xl border border-neutral-200 p-4 space-y-3">
                  <div>
                    <h2 className="text-sm font-extrabold">Jenis laporan</h2>
                    <p className="text-[10px] text-neutral-500 mt-1">Pilih alasan yang paling sesuai.</p>
                  </div>
                  <div className="space-y-2">
                    {reportTypes.map((type) => (
                      <label key={type} className={`flex items-center gap-3 rounded-2xl border p-3 cursor-pointer transition-colors ${reportType === type ? "border-rose-400 bg-rose-50" : "border-neutral-200 hover:bg-neutral-50"}`}>
                        <input
                          type="radio"
                          name="reportType"
                          value={type}
                          checked={reportType === type}
                          onChange={() => setReportType(type)}
                          className="radio radio-error radio-xs"
                        />
                        <span className="text-xs font-semibold">{type}</span>
                      </label>
                    ))}
                  </div>
                </section>

                <section className="rounded-3xl border border-neutral-200 p-4 space-y-3">
                  <label htmlFor="description" className="text-sm font-extrabold">Ceritakan kejadian</label>
                  <textarea
                    id="description"
                    value={description}
                    onChange={(event) => setDescription(event.target.value)}
                    placeholder="Tambahkan detail yang membantu pemeriksaan..."
                    className="textarea textarea-bordered w-full min-h-28 rounded-2xl text-xs leading-relaxed"
                    required
                  />
                  <label className="flex items-center justify-center rounded-2xl border border-dashed border-neutral-300 p-3 cursor-pointer hover:bg-neutral-50">
                    <span className="text-xs font-bold text-neutral-600">{evidenceName || "Tambahkan foto bukti (opsional)"}</span>
                    <input type="file" accept="image/*" className="hidden" onChange={(event) => setEvidenceName(event.target.files?.[0]?.name || "")} />
                  </label>
                </section>

                <label className="flex items-start gap-2 px-1 text-[10px] font-medium text-neutral-500">
                  <input type="checkbox" required className="checkbox checkbox-xs checkbox-error mt-0.5" />
                  <span>Saya menyatakan informasi yang diberikan benar.</span>
                </label>

                <div className="flex gap-2 pt-1 pb-5">
                  <Link href="/result/bahaya" className="flex-1 rounded-full border border-neutral-300 py-3.5 text-center text-xs font-extrabold text-neutral-700 hover:bg-neutral-50">
                    Batal
                  </Link>
                  <button type="submit" className="flex-[1.5] rounded-full bg-rose-600 py-3.5 text-xs font-extrabold text-white shadow-md hover:bg-rose-700 active:scale-[0.98]">
                    Kirim Laporan
                  </button>
                </div>
              </form>
            </>
          )}
        </div>
      </div>
    </main>
  );
}
