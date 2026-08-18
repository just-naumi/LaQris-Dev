import os
import sys
import subprocess


def find_file_test():
    """Cari file_test.py di workspace relatif terhadap folder `api/`."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(base, "LaQris Physical Identity Extraction", "file_test.py")
    if os.path.exists(candidate):
        return candidate

    # fallback: cari rekursif jika struktur beda
    for root, dirs, files in os.walk(base):
        if "file_test.py" in files:
            return os.path.join(root, "file_test.py")

    raise FileNotFoundError("file_test.py not found in workspace")


def verify_image(image_path: str, timeout: int = 300):
    """
    Jalankan `file_test.py` sebagai subprocess dengan argumen path gambar.
    Kembalikan dict berisi `returncode`, `stdout`, `stderr`.

    Catatan: Ini implementasi sederhana untuk scaffold API. Langkah berikutnya
    adalah memuat pipeline model in-process (singleton) agar tidak memuat ulang
    model untuk setiap request.
    """
    file_test = find_file_test()
    cmd = [sys.executable, file_test, image_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
