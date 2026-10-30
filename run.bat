@echo off
cd /d "%~dp0"
title Real-Time Air Drawing Application

echo =======================================================
echo    Agentic AI-Enabled Real-Time Air Drawing Application
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

:: Quick dependency check
echo [AirDrawing] Verifying dependencies...
%PY_CMD% -c "import cv2, mediapipe, numpy" >nul 2>nul
if %errorlevel% neq 0 (
    echo [AirDrawing] Required packages not found. Installing from requirements.txt...
    %PY_CMD% -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to install dependencies automatically.
        echo Please run: pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

echo [AirDrawing] Launching application...
echo.
%PY_CMD% main.py

if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] Application closed with exit code %errorlevel%.
)

pause
