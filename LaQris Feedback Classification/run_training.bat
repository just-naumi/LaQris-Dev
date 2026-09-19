@echo off
echo ================================================================
echo    LaQris - Fine-Tuning IndoBERT Feedback Classifier (GPU)
echo ================================================================
echo.

REM Pindah ke direktori root repository
cd /d "%~dp0\.."

REM Cek apakah virtual environment GPU ada
if exist ".venv-gpu\Scripts\python.exe" (
    echo [INFO] Menggunakan Python dari .venv-gpu (NVIDIA CUDA)...
    set "PYTHON_EXEC=.venv-gpu\Scripts\python.exe"
    set "PIP_EXEC=.venv-gpu\Scripts\pip.exe"
) else (
    echo [INFO] .venv-gpu tidak ditemukan, menggunakan python sistem/default...
    set "PYTHON_EXEC=python"
    set "PIP_EXEC=pip"
)

echo.
echo [1/3] Memeriksa dan menginstal pustaka yang diperlukan...
%PIP_EXEC% install transformers datasets accelerate scikit-learn pyyaml

echo.
echo [2/3] Memulai proses fine-tuning IndoBERT (10.000 dataset)...
%PYTHON_EXEC% "LaQris Feedback Classification\training\train_indobert.py"

echo.
echo [3/3] Menjalankan evaluasi model pada data uji (Test Set)...
%PYTHON_EXEC% "LaQris Feedback Classification\training\evaluate_model.py"

echo.
echo ================================================================
echo    Training Selesai! Checkpoint tersimpan di models/
echo ================================================================
pause
