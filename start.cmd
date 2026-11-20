@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Mancu Launcher
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -NonInteractive %*
set "MANCU_EXIT=%ERRORLEVEL%"
if not "%MANCU_EXIT%"=="0" (
  echo.
  echo Mancu could not start. Check logs\start-powershell.log for details.
  if not "%MANCU_NO_PAUSE%"=="1" pause
)
exit /b %MANCU_EXIT%

