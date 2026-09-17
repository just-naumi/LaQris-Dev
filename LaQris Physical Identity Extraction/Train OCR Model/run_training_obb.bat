@echo off
title Pelatihan Model YOLO26 Small OBB (Oriented Bounding Box) - LaQris
echo ===================================================================
echo    LAQRIS PHYSICAL IDENTITY EXTRACTION - TRAINING YOLO26 OBB
echo    Kotak Deteksi Berotasi / Miring Mengikuti Orientasi Teks
echo ===================================================================
echo [1/2] Mengaktifkan Python GPU Virtual Environment...
call "..\..\.venv-gpu\Scripts\activate.bat"

echo [2/2] Memulai Pelatihan Model (yolo26s-obb.pt) di GPU RTX 3050...
python train_yolo_obb.py

echo.
echo ===================================================================
echo    PELATIHAN OBB SELESAI! Tekan sembarang tombol untuk keluar.
echo ===================================================================
pause
