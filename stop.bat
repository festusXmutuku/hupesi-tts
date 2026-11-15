@echo off
setlocal enabledelayedexpansion
title Hupesi TTS - Dung ung dung
cd /d "%~dp0"

echo ================================================================
echo         Hupesi TTS - Free Vietnamese Text-to-Speech
echo         Tac gia: Festus Mutuku (https://github.com/festusxmutukugithub)
echo         Phat trien dua tren: VieNeu-TTS (Pham Nguyen Ngoc Bao)
echo ================================================================
echo.
echo [+] Dang tim va dong cac tien trinh Hupesi TTS dang chay...
echo.

set "STOPPED=0"

:: Dong tien trinh dang lang nghe tren Port 8001 (may chu cua ung dung)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8001 ^| findstr LISTENING 2^>nul') do (
    set "STOPPED=1"
    echo [+] Dang dong tien trinh Hupesi TTS tren Port 8001 (PID: %%a)...
    taskkill /f /pid %%a >nul 2>&1
)

echo.
if "!STOPPED!"=="1" (
    echo ================================================================
    echo  [+] Da dong thanh cong tat ca tien trinh Hupesi TTS!
    echo ================================================================
) else (
    echo [i] Khong tim thay tien trinh Hupesi TTS nao dang chay.
)
echo.
pause
