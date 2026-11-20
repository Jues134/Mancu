@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Stop Mancu
set "MANCU_ROOT=%~dp0"
set "MANCU_STOP_PORT=8100"
if not "%~1"=="" set "MANCU_STOP_PORT=%~1"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { $port=[int]$env:MANCU_STOP_PORT; $root=[IO.Path]::GetFullPath($env:MANCU_ROOT); $expected=@([IO.Path]::GetFullPath((Join-Path $root '.venv\Scripts\python.exe')),[IO.Path]::GetFullPath((Join-Path $root 'runtime\python.exe'))); $ids=@(); if(Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue){ try{$ids=@(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique)}catch{} }; if($ids.Count -eq 0){ foreach($line in (& netstat.exe -ano -p tcp)){ if($line -match ('^\s*TCP\s+\S+:'+$port+'\s+\S+\s+LISTENING\s+(\d+)\s*$')){$ids+=[int]$Matches[1]} } }; $ids=@($ids | Select-Object -Unique); if($ids.Count -eq 0){Write-Host ('Port '+$port+' is already free.');exit 0}; foreach($processId in $ids){$cmd='';$exe='';$name='unknown';try{$info=Get-CimInstance Win32_Process -Filter ('ProcessId='+$processId) -ErrorAction Stop;$cmd=[string]$info.CommandLine;$exe=[string]$info.ExecutablePath;$name=[string]$info.Name}catch{try{$p=Get-Process -Id $processId -ErrorAction Stop;$name=$p.ProcessName;$exe=[string]$p.Path}catch{}};$isExpected=$false;if($exe){$isExpected=$expected -contains [IO.Path]::GetFullPath($exe)};$isMancu=(($cmd -match 'app\.serve') -and ($cmd -match 'python')) -or $isExpected;if(-not $isMancu){throw ('Port '+$port+' belongs to PID '+$processId+' ('+$name+'), not Mancu. Nothing was stopped.')};Stop-Process -Id $processId -Force -ErrorAction Stop;Write-Host ('Stopped Mancu PID '+$processId+' on port '+$port+'.') } } catch { Write-Host ('[ERROR] '+$_.Exception.Message) -ForegroundColor Red; exit 1 }"
set "MANCU_EXIT=%ERRORLEVEL%"
echo.
if not "%MANCU_NO_PAUSE%"=="1" pause
exit /b %MANCU_EXIT%

