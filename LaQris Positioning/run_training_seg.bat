@echo off
title Pelatihan Model YOLO26s Instance Segmentation - LaQris Positioning
echo ===================================================================
echo    LAQRIS POSITIONING - TRAINING YOLO26s INSTANCE SEGMENTATION
echo ===================================================================
echo [1/2] Mengaktifkan Python GPU Virtual Environment (.venv-gpu)...
call "..\.venv-gpu\Scripts\activate.bat"

echo.
echo [2/2] Menjalankan Script Pelatihan (train_yolo_seg.py)...
echo Target Hardware: NVIDIA RTX 3050 Laptop GPU (CUDA: 0)
echo Backbone Model : yolo26s-seg.pt
echo Dataset        : 108 Train, 31 Valid, 15 Test
echo.
python train_yolo_seg.py

echo.
echo ===================================================================
echo    PELATIHAN SELESAI! Tekan sembarang tombol untuk menutup jendela.
echo ===================================================================
pause
