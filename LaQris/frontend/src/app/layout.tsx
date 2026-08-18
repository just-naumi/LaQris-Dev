import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LaQris — AI Object Detection",
  description: "Deteksi objek real-time dengan YOLOv8 langsung dari kamera HP kamu.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id" data-theme="night">
      <body>{children}</body>
    </html>
  );
}
