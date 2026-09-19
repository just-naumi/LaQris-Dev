@echo off
title Pelatihan Model YOLO26 Small OCR QRIS - LaQris
echo ===================================================================
echo    LAQRIS PHYSICAL IDENTITY EXTRACTION - TRAINING YOLO26 SMALL
echo ===================================================================
echo [1/2] Mengaktifkan Python GPU Virtual Environment...
call "..\..\.venv-gpu\Scripts\activate.bat"

echo [2/2] Memulai Pelatihan Model (yolo26s.pt)...
python train_yolo.py

echo.
echo ===================================================================
echo    PELATIHAN SELESAI! Tekan sembarang tombol untuk keluar.
echo ===================================================================
pause
