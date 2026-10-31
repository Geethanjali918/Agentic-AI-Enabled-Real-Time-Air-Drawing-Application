@echo off
cd /d "%~dp0"
title Agentic AI Air Drawing System

echo =======================================================
echo    Agentic AI-Enabled Real-Time Air Drawing Application
echo    Proportional Stack: Dataset + ML + DL + Data Analytics
echo =======================================================
echo.

:: Detect Python executable
set "PY_CMD="
where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_CMD=py"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10+ and make sure to check
    echo the box "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:menu
echo.
echo Please choose an option:
echo   [1] Launch Real-Time Air Drawing Application (Live Camera)
echo   [2] Train Machine Learning Models (Random Forest, SVM, KNN)
echo   [3] Train Deep Learning Models (DeepNet DNN + StrokeCNN)
echo   [4] Generate Data Analytics Dashboard & EDA Figures
echo   [5] Collect Custom Hand Gesture Dataset via Webcam
echo   [6] Run Complete 10-Point Automated Test Suite
echo   [7] Exit
echo.
set /p opt="Enter choice [1-7]: "

if "%opt%"=="1" (
    echo.
    echo [AirDrawing] Launching Real-Time Application...
    %PY_CMD% main.py
    goto menu
) else if "%opt%"=="2" (
    echo.
    echo [AirDrawing] Training Machine Learning Models...
    %PY_CMD% ml_models/train_ml.py
    pause
    goto menu
) else if "%opt%"=="3" (
    echo.
    echo [AirDrawing] Training Deep Learning Models...
    %PY_CMD% dl_models/train_dl.py
    pause
    goto menu
) else if "%opt%"=="4" (
    echo.
    echo [AirDrawing] Generating Analytics Dashboard & Reports...
    %PY_CMD% analytics/run_all_analytics.py
    pause
    goto menu
) else if "%opt%"=="5" (
    echo.
    echo [AirDrawing] Launching Live Dataset Collector...
    %PY_CMD% dataset/dataset_collector.py
    pause
    goto menu
) else if "%opt%"=="6" (
    echo.
    echo [AirDrawing] Running Test Suite...
    %PY_CMD% test_app.py
    pause
    goto menu
) else if "%opt%"=="7" (
    echo Exiting. Goodbye!
    exit /b 0
) else (
    echo Invalid selection. Please try again.
    goto menu
)
