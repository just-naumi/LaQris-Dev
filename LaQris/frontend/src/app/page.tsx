"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const router = useRouter();
  const [showEmailForm, setShowEmailForm] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Google OAuth Popup Modal States
  const [showGoogleOAuthModal, setShowGoogleOAuthModal] = useState(false);
  const [isGoogleSigningIn, setIsGoogleSigningIn] = useState(false);
  const [showCustomGoogleForm, setShowCustomGoogleForm] = useState(false);
  const [customName, setCustomName] = useState("");
  const [customEmail, setCustomEmail] = useState("");

  const handleLoginSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg("");
    setIsLoading(true);

    setTimeout(() => {
      let registeredUsers: Array<{ email: string; password: string }> = [];
      try {
        const stored = localStorage.getItem("laqris_users");
        if (stored) {
          registeredUsers = JSON.parse(stored);
        }
      } catch (err) {
        console.error(err);
      }

      const defaultUsers = [
        { email: "admin@laqris.id", password: "admin123" },
        { email: "demo@laqris.com", password: "laqris123" },
        { email: "user@laqris.id", password: "laqris123" },
      ];

      const allUsers = [...defaultUsers, ...registeredUsers];

      const foundUser = allUsers.find(
        (u) => u.email.toLowerCase() === email.trim().toLowerCase() && u.password === password
      );

      if (foundUser) {
        setIsLoading(false);
        localStorage.setItem("laqris_logged_in_user", JSON.stringify({ email: foundUser.email, provider: "email" }));
        router.push("/dashboard");
      } else {
        setIsLoading(false);
        setErrorMsg("Email atau password tidak valid / belum terdaftar.");
      }
    }, 400);
  };

  const handleSelectGoogleAccount = (accountName: string, accountEmail: string) => {
    setIsGoogleSigningIn(true);
    setTimeout(() => {
      setIsGoogleSigningIn(false);
      setShowGoogleOAuthModal(false);
      localStorage.setItem(
        "laqris_logged_in_user",
        JSON.stringify({
          name: accountName,
          email: accountEmail,
          provider: "google",
        })
      );
      router.push("/dashboard");
    }, 600);
  };

  const handleCustomGoogleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customEmail) return;
    const name = customName.trim() || customEmail.split("@")[0];
    handleSelectGoogleAccount(name, customEmail.trim());
  };

  return (
    <div className="min-h-screen bg-neutral-950 py-0 sm:py-8 flex items-center justify-center font-sans antialiased selection:bg-emerald-500 selection:text-white">
      {/* Mobile Screen Container */}
      <div className="w-full max-w-[410px] min-h-[850px] bg-gradient-to-b from-white via-slate-50 to-emerald-50/40 sm:rounded-[44px] shadow-2xl overflow-hidden flex flex-col justify-between p-6 relative border-0 sm:border-[8px] border-neutral-800 text-neutral-900">
        
        {/* Soft Background Decorative Glows */}
        <div className="absolute -top-16 -left-16 w-64 h-64 bg-emerald-100/60 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute top-1/3 -right-20 w-72 h-72 bg-emerald-200/40 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-16 -left-16 w-64 h-64 bg-slate-200/70 rounded-full blur-3xl pointer-events-none" />

        {/* TOP BRAND HEADER */}
        <div className="pt-4 relative z-10 flex items-center justify-between">
          <h1 className="text-3xl font-black tracking-tight text-neutral-900">
            LàQris.
          </h1>
          {showEmailForm && (
            <button
              onClick={() => {
                setShowEmailForm(false);
                setErrorMsg("");
              }}
              className="text-xs font-semibold text-neutral-500 hover:text-neutral-900 transition-colors flex items-center gap-1 bg-neutral-100 px-3 py-1.5 rounded-full"
            >
              ← Back
            </button>
          )}
        </div>

        {/* MAIN HERO CONTENT */}
        <div className="my-auto space-y-4 relative z-10 py-2">
          
          {/* HERO SECTION TEXT */}
          <div className="relative py-2">
            <div className="space-y-2.5 max-w-full z-10">
              <h2 className="text-3xl font-extrabold tracking-tight text-neutral-900 leading-[1.15]">
                {showEmailForm ? (
                  "Welcome back."
                ) : (
                  <>
                    Scan First.<br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-neutral-900 via-neutral-800 to-emerald-700">
                      Pay Safe.
                    </span>
                  </>
                )}
              </h2>
              <p className="text-xs font-medium text-neutral-600 leading-relaxed max-w-[280px]">
                {showEmailForm
                  ? "Enter your registered email and password to access your account."
                  : "Verify every QRIS before payment using AI-powered fraud detection."}
              </p>
            </div>
          </div>

          {/* 3-COLUMN FEATURE CARDS */}
          {!showEmailForm && (
            <div className="grid grid-cols-3 gap-2 bg-white/80 backdrop-blur-md p-3 rounded-2xl border border-neutral-200/80 shadow-sm shadow-neutral-200/50">
              <div className="flex flex-col items-center text-center px-1">
                <div className="w-8 h-8 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center mb-1.5 shadow-sm">
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                    <path d="M9 12l2 2 4-4" />
                  </svg>
                </div>
                <h4 className="text-[11px] font-bold text-neutral-900 leading-tight">Fraud Detection</h4>
                <p className="text-[9px] font-medium text-neutral-500 leading-tight mt-0.5">AI secures your payment.</p>
              </div>

              <div className="flex flex-col items-center text-center px-1 border-x border-neutral-200/80">
                <div className="w-8 h-8 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center mb-1.5 shadow-sm">
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
                  </svg>
                </div>
                <h4 className="text-[11px] font-bold text-neutral-900 leading-tight">Instant Scan</h4>
                <p className="text-[9px] font-medium text-neutral-500 leading-tight mt-0.5">Fast & accurate in seconds.</p>
              </div>

              <div className="flex flex-col items-center text-center px-1">
                <div className="w-8 h-8 rounded-full bg-teal-100 text-teal-700 flex items-center justify-center mb-1.5 shadow-sm">
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <circle cx="12" cy="8" r="6" />
                    <path d="M15.477 12.89L17 22l-5-3-5 3 1.523-9.11" />
                  </svg>
                </div>
                <h4 className="text-[11px] font-bold text-neutral-900 leading-tight">Merchant Verified</h4>
                <p className="text-[9px] font-medium text-neutral-500 leading-tight mt-0.5">Trusted merchant, safer transaction.</p>
              </div>
            </div>
          )}

          {/* DYNAMIC FORM / BUTTONS CONTAINER WITH ANIMATION */}
          <div className="relative overflow-hidden min-h-[180px] flex flex-col justify-end pt-1">

            {/* INITIAL LOGIN SELECTION (Sign in to LaQris + Google) */}
            <div
              className={`space-y-3 transition-all duration-400 ease-in-out ${
                showEmailForm
                  ? "opacity-0 pointer-events-none translate-y-6 absolute inset-x-0 bottom-0"
                  : "opacity-100 translate-y-0 relative"
              }`}
            >
              {/* SIGN IN TO LAQRIS BUTTON */}
              <button
                onClick={() => {
                  setShowEmailForm(true);
                  setErrorMsg("");
                }}
                className="w-full py-3.5 px-5 rounded-2xl bg-neutral-900 hover:bg-neutral-800 text-white font-bold text-xs transition-all duration-200 active:scale-[0.98] shadow-lg shadow-neutral-900/20 flex items-center justify-between group"
              >
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-neutral-300 group-hover:text-white transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  <span>Sign in to LaQris</span>
                </div>
                <svg className="w-4 h-4 text-neutral-400 group-hover:translate-x-1 group-hover:text-white transition-all" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </button>

              {/* DIVIDER */}
              <div className="relative flex py-0.5 items-center">
                <div className="grow border-t border-neutral-200"></div>
                <span className="shrink mx-3 text-[11px] font-bold text-neutral-600">
                  Or Continue With
                </span>
                <div className="grow border-t border-neutral-200"></div>
              </div>

              {/* GOOGLE BUTTON */}
              <div>
                <button
                  type="button"
                  onClick={() => {
                    setShowGoogleOAuthModal(true);
                    setShowCustomGoogleForm(false);
                  }}
                  className="w-full py-3.5 px-4 rounded-2xl bg-white hover:bg-neutral-50 text-neutral-900 font-bold text-xs transition-all flex items-center justify-center gap-2.5 border border-neutral-200/90 shadow-sm active:scale-[0.98] group"
                >
                  <svg className="w-4 h-4 group-hover:scale-110 transition-transform" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
                  </svg>
                  <span>Continue with Google</span>
                </button>
              </div>
            </div>

            {/* EXPANDED EMAIL INPUT FORM */}
            <form
              onSubmit={handleLoginSubmit}
              className={`space-y-3 transition-all duration-400 ease-in-out ${
                showEmailForm
                  ? "opacity-100 translate-y-0 relative"
                  : "opacity-0 pointer-events-none translate-y-6 absolute inset-x-0 bottom-0"
              }`}
            >
              {errorMsg && (
                <div className="p-3 bg-red-50 border border-red-200 text-red-700 text-xs font-semibold rounded-xl flex items-center gap-2">
                  <svg className="w-4 h-4 shrink-0 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>{errorMsg}</span>
                </div>
              )}

              <div className="space-y-1 text-left">
                <label className="text-[11px] font-semibold text-neutral-600 block pl-1">
                  Email Address
                </label>
                <input
                  type="email"
                  placeholder="admin@laqris.id"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required={showEmailForm}
                  className="w-full py-3 px-4 rounded-xl border border-neutral-200 bg-white text-neutral-900 text-xs font-medium focus:outline-none focus:border-emerald-600 focus:ring-1 focus:ring-emerald-600 transition-all placeholder:text-neutral-300 shadow-sm"
                />
              </div>

              <div className="space-y-1 text-left">
                <label className="text-[11px] font-semibold text-neutral-600 block pl-1">
                  Password
                </label>
                <input
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required={showEmailForm}
                  className="w-full py-3 px-4 rounded-xl border border-neutral-200 bg-white text-neutral-900 text-xs font-medium focus:outline-none focus:border-emerald-600 focus:ring-1 focus:ring-emerald-600 transition-all placeholder:text-neutral-300 shadow-sm"
                />
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-3.5 px-6 rounded-xl bg-neutral-900 hover:bg-neutral-800 text-white font-bold text-xs transition-all duration-200 active:scale-[0.98] shadow-lg shadow-neutral-900/20 mt-1 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <svg className="w-4 h-4 animate-spin text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span>Verifying...</span>
                  </>
                ) : (
                  "Sign In"
                )}
              </button>

              <p className="text-[10px] text-center text-neutral-400 font-medium pt-1">
                Demo Account: <span className="font-semibold text-neutral-700">admin@laqris.id</span> / <span className="font-semibold text-neutral-700">admin123</span>
              </p>
            </form>

          </div>

          {/* BOTTOM SECURITY PRIORITY BADGE */}
          {!showEmailForm && (
            <div className="flex items-center gap-3 bg-white/80 backdrop-blur-md p-3 rounded-2xl border border-neutral-200/70 shadow-sm mt-3">
              <div className="w-8 h-8 rounded-full bg-neutral-100 text-neutral-800 flex items-center justify-center shrink-0 shadow-sm">
                <svg className="w-4 h-4 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <div>
                <h5 className="text-[11px] font-bold text-neutral-900 leading-tight">Your security is our priority.</h5>
                <p className="text-[9.5px] font-medium text-neutral-500 leading-tight">LàQris protects you from QRIS fraud.</p>
              </div>
            </div>
          )}

        </div>

        {/* FOOTER LINK */}
        <div className="pb-2 text-center relative z-10">
          <p className="text-xs font-semibold text-neutral-700">
            Don&apos;t have an account?{" "}
            <Link href="/register" className="font-extrabold text-neutral-900 underline underline-offset-2 hover:text-emerald-600 transition-colors">
              Register Here
            </Link>
          </p>
        </div>

      </div>

      {/* GOOGLE OAUTH MODAL (EXACT MATCH FOR IMAGE 2) */}
      {showGoogleOAuthModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="w-full max-w-[680px] bg-[#121316] text-white rounded-3xl shadow-2xl border border-neutral-800 overflow-hidden relative">
            
            {/* TOP HEADER BAR */}
            <div className="px-6 py-4 border-b border-neutral-800/80 flex items-center justify-between bg-[#18191c]">
              <div className="flex items-center gap-2.5">
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
                </svg>
                <span className="text-xs font-semibold text-neutral-300">Login dengan Google</span>
              </div>
              <button
                onClick={() => {
                  if (!isGoogleSigningIn) setShowGoogleOAuthModal(false);
                }}
                className="w-7 h-7 rounded-full bg-neutral-800 text-neutral-400 hover:text-white flex items-center justify-center text-xs font-bold transition-colors"
              >
                ✕
              </button>
            </div>

            {/* MODAL BODY (TWO COLUMNS / GRID) */}
            <div className="p-6 sm:p-8 grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
              
              {/* LEFT COLUMN: BRAND & HEADING */}
              <div className="space-y-4 pt-1">
                {/* BRAND ICON */}
                <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20 text-white font-black text-xl">
                  <svg className="w-7 h-7" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                    <path d="M9 12l2 2 4-4" />
                  </svg>
                </div>

                <div>
                  <h3 className="text-3xl font-extrabold text-white tracking-tight leading-tight">
                    Pilih akun
                  </h3>
                  <p className="text-sm font-medium text-neutral-400 mt-1">
                    Lanjutkan ke <span className="text-emerald-400 font-bold">LàQris</span>
                  </p>
                </div>
              </div>

              {/* RIGHT COLUMN: GOOGLE ACCOUNTS & SELECTION */}
              <div className="space-y-4">
                {isGoogleSigningIn ? (
                  <div className="py-10 text-center space-y-3">
                    <div className="w-10 h-10 border-3 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto" />
                    <p className="text-xs font-medium text-neutral-300">
                      Menghubungkan akun Google...
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="space-y-2">
                      {/* DEFAULT ACCOUNT 1: Al Ghifariy */}
                      <button
                        type="button"
                        onClick={() => handleSelectGoogleAccount("Al Ghifariy", "ghifariyal@gmail.com")}
                        className="w-full p-3 rounded-xl hover:bg-neutral-800/80 transition-all flex items-center justify-between text-left group border border-transparent hover:border-neutral-700"
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-9 h-9 rounded-full bg-cyan-500 text-white font-bold text-xs flex items-center justify-center shadow-sm shrink-0">
                            AI
                          </div>
                          <div>
                            <h4 className="text-xs font-bold text-white group-hover:text-emerald-400 transition-colors">
                              Al Ghifariy
                            </h4>
                            <p className="text-[11px] font-medium text-neutral-400">
                              ghifariyal@gmail.com
                            </p>
                          </div>
                        </div>
                      </button>

                      {/* DIVIDER */}
                      <div className="border-t border-neutral-800 my-1" />

                      {/* GUNAKAN AKUN LAIN */}
                      {!showCustomGoogleForm ? (
                        <button
                          type="button"
                          onClick={() => setShowCustomGoogleForm(true)}
                          className="w-full p-3 rounded-xl hover:bg-neutral-800/80 transition-all flex items-center gap-3 text-left text-neutral-300 hover:text-white group"
                        >
                          <div className="w-9 h-9 rounded-full border border-neutral-700 flex items-center justify-center text-neutral-400 group-hover:text-white shrink-0">
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                          </div>
                          <span className="text-xs font-medium">Gunakan akun lain</span>
                        </button>
                      ) : (
                        <form onSubmit={handleCustomGoogleSubmit} className="space-y-2 pt-1">
                          <input
                            type="text"
                            placeholder="Nama Lengkap"
                            value={customName}
                            onChange={(e) => setCustomName(e.target.value)}
                            className="w-full py-2 px-3 rounded-lg bg-neutral-900 border border-neutral-700 text-xs text-white placeholder:text-neutral-500 focus:outline-none focus:border-emerald-500"
                          />
                          <input
                            type="email"
                            placeholder="nama@gmail.com"
                            value={customEmail}
                            onChange={(e) => setCustomEmail(e.target.value)}
                            required
                            className="w-full py-2 px-3 rounded-lg bg-neutral-900 border border-neutral-700 text-xs text-white placeholder:text-neutral-500 focus:outline-none focus:border-emerald-500"
                          />
                          <div className="flex items-center gap-2 pt-1">
                            <button
                              type="submit"
                              className="w-full py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs transition-all"
                            >
                              Login
                            </button>
                            <button
                              type="button"
                              onClick={() => setShowCustomGoogleForm(false)}
                              className="py-2 px-3 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-300 text-xs font-medium"
                            >
                              Batal
                            </button>
                          </div>
                        </form>
                      )}
                    </div>

                    {/* DISCLAIMER */}
                    <p className="text-[10px] text-neutral-500 leading-relaxed pt-2">
                      Sebelum menggunakan aplikasi ini, Anda dapat meninjau{" "}
                      <span className="text-emerald-400 cursor-pointer hover:underline">Kebijakan Privasi</span> dan{" "}
                      <span className="text-emerald-400 cursor-pointer hover:underline">Persyaratan Layanan</span> LàQris.
                    </p>
                  </>
                )}
              </div>

            </div>

            {/* BOTTOM BAR */}
            <div className="px-6 py-3 bg-[#18191c] border-t border-neutral-800/80 flex items-center justify-between text-[10px] text-neutral-500">
              <span className="cursor-pointer hover:text-neutral-300">Indonesia ▾</span>
              <div className="flex items-center gap-4">
                <span className="cursor-pointer hover:text-neutral-300">Bantuan</span>
                <span className="cursor-pointer hover:text-neutral-300">Privasi</span>
                <span className="cursor-pointer hover:text-neutral-300">Persyaratan</span>
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
