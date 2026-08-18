"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const router = useRouter();
  const [showEmailForm, setShowEmailForm] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLoginSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    router.push("/dashboard");
  };

  return (
    <div className="min-h-screen bg-neutral-900 py-0 sm:py-8 flex items-center justify-center font-sans antialiased">
      {/* Mobile Screen Container */}
      <div className="w-full max-w-[390px] h-[844px] bg-white sm:rounded-[44px] shadow-2xl overflow-hidden flex flex-col justify-between p-8 relative border-0 sm:border-[8px] border-neutral-800 text-neutral-900">
        
        {/* Soft Background Decorative Circles */}
        <div className="absolute -top-12 -left-12 w-64 h-64 bg-neutral-100/80 rounded-full blur-2xl pointer-events-none" />
        <div className="absolute top-1/2 -right-16 w-60 h-60 bg-neutral-100/60 rounded-full blur-2xl pointer-events-none" />
        <div className="absolute -bottom-16 -left-16 w-64 h-64 bg-neutral-100/80 rounded-full blur-2xl pointer-events-none" />

        {/* BRAND LOGO */}
        <div className="pt-8 relative z-10 flex items-center justify-between">
          <h1 className="text-4xl font-extrabold tracking-tight text-neutral-900">
            LàQris.
          </h1>
          {showEmailForm && (
            <button
              onClick={() => setShowEmailForm(false)}
              className="text-xs font-semibold text-neutral-400 hover:text-neutral-900 transition-colors flex items-center gap-1"
            >
              ← Back
            </button>
          )}
        </div>

        {/* MIDDLE HERO CONTENT */}
        <div className="my-auto space-y-5 relative z-10 pt-2">
          <div className="space-y-2">
            <h2 className="text-3xl font-extrabold tracking-tight text-neutral-900 leading-tight transition-all duration-300">
              {showEmailForm ? (
                "Welcome back."
              ) : (
                <>
                  Scan First.<br />
                  Pay First.
                </>
              )}
            </h2>
            <p className="text-xs font-medium text-neutral-500 leading-relaxed max-w-[260px] transition-all duration-300">
              {showEmailForm
                ? "Enter your email and password to access your account."
                : (
                    <>
                      Verify every QRIS before payment using<br />
                      AI-powered fraud detection.
                    </>
                  )}
            </p>
          </div>

          {/* DYNAMIC FORM / BUTTONS CONTAINER WITH ANIMATION */}
          <div className="relative overflow-hidden min-h-[220px] flex flex-col justify-end pt-2">

            {/* INITIAL LOGIN SELECTION (Login by Email + Google) */}
            <div
              className={`space-y-4 transition-all duration-400 ease-in-out ${
                showEmailForm
                  ? "opacity-0 pointer-events-none translate-y-6 absolute inset-x-0 bottom-0"
                  : "opacity-100 translate-y-0 relative"
              }`}
            >
              <button
                onClick={() => setShowEmailForm(true)}
                className="w-full py-3.5 px-6 rounded-full border border-neutral-900 bg-transparent hover:bg-neutral-900 hover:text-white text-neutral-900 font-bold text-xs transition-all duration-200 active:scale-[0.98] shadow-sm"
              >
                Login by Email
              </button>

              {/* DIVIDER */}
              <div className="relative flex py-1 items-center">
                <div className="grow border-t border-neutral-200"></div>
                <span className="shrink mx-3 text-[10px] font-semibold text-neutral-400">
                  Or Continue With
                </span>
                <div className="grow border-t border-neutral-200"></div>
              </div>

              {/* GOOGLE BUTTON */}
              <div>
                <button
                  onClick={() => router.push("/dashboard")}
                  className="w-full py-3.5 px-4 rounded-full bg-neutral-50 hover:bg-neutral-100 text-neutral-900 font-semibold text-xs transition-all flex items-center justify-center gap-2 border border-neutral-200 shadow-sm active:scale-[0.98]"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
                  </svg>
                  Google
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
              <div className="space-y-1 text-left">
                <label className="text-[11px] font-semibold text-neutral-600 block pl-1">
                  Email Address
                </label>
                <input
                  type="email"
                  placeholder="name@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required={showEmailForm}
                  className="w-full py-3 px-4 rounded-full border border-neutral-200 bg-neutral-50/50 text-neutral-900 text-xs font-medium focus:outline-none focus:border-neutral-900 focus:bg-white transition-all placeholder:text-neutral-300"
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
                  className="w-full py-3 px-4 rounded-full border border-neutral-200 bg-neutral-50/50 text-neutral-900 text-xs font-medium focus:outline-none focus:border-neutral-900 focus:bg-white transition-all placeholder:text-neutral-300"
                />
              </div>

              <button
                type="submit"
                className="w-full py-3.5 px-6 rounded-full bg-neutral-900 hover:bg-neutral-800 text-white font-bold text-xs transition-all duration-200 active:scale-[0.98] shadow-lg shadow-neutral-900/10 mt-1"
              >
                Sign In
              </button>
            </form>

          </div>
        </div>

        {/* FOOTER LINK */}
        <div className="pb-6 text-center relative z-10">
          <p className="text-xs font-medium text-neutral-400">
            Don&apos;t have an account?{" "}
            <Link href="/register" className="font-bold text-neutral-900 underline underline-offset-2 hover:text-primary transition-colors">
              Register Here
            </Link>
          </p>
        </div>

      </div>
    </div>
  );
}
