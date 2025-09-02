@echo off
setlocal

REM ===========================
REM Spark-TTS manager (start/stop/restart)
REM Usage:
REM   tts start [port]
REM   tts stop [port]
REM   tts restart [port]
REM Default port: 8000
REM ===========================

:: ---------- args ----------
set ACTION=%1
set PORT=%2
if "%PORT%"=="" set PORT=8000

if /I "%ACTION%"=="start"    goto :START
if /I "%ACTION%"=="stop"     goto :STOP
if /I "%ACTION%"=="restart"  goto :RESTART

echo Usage: tts start^|stop^|restart [port]
goto :END

:STOP
echo Stopping anything listening on :%PORT% ...

REM 1) netstat IPv4/IPv6 LISTENING -> kill
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
  taskkill /PID %%p /F >nul 2>&1
)

REM 2) if previously started via "start", also try killing by window title
taskkill /FI "WINDOWTITLE eq SparkTTS %PORT%" /T /F >nul 2>&1

REM 3) PowerShell fallback: kill any process that still owns the port
for /f %%p in ('powershell -NoP -C "(Get-NetTCPConnection -LocalPort %PORT% -State Listen ^| Select-Object -Expand OwningProcess) 2^>^&1"') do (
  taskkill /PID %%p /F >nul 2>&1
)

REM 4) wait until released (max ~5s)
set /a _tries=0
:WAIT_FREE
set /a _tries+=1
for /f %%x in ('powershell -NoP -C "(@(Get-NetTCPConnection -LocalPort %PORT% -State Listen).Count)"') do set _cnt=%%x
if "%_cnt%"=="" set _cnt=0
if %_cnt% GTR 0 (
  if %_tries% LSS 6 (
    timeout /t 1 >nul
    goto :WAIT_FREE
  )
)
echo Done.
goto :END

:START
echo Stopping anything on :%PORT% first...
call "%~f0" stop %PORT%
echo Starting Spark-TTS on port %PORT% ...

REM sanity check: ensure port is free
for /f %%x in ('powershell -NoP -C "(@(Get-NetTCPConnection -LocalPort %PORT% -State Listen).Count)"') do set _cnt=%%x
if "%_cnt%"=="" set _cnt=0
if %_cnt% GTR 0 (
  echo Port %PORT% is still in use. Try another port or stop the process manually.
  goto :END
)

REM Activate conda env (adjust if needed)
where conda >nul 2>&1
if errorlevel 1 (
  REM If conda is not on PATH, set your activate.bat path here:
  REM call "C:\Miniconda3\Scripts\activate.bat" sparktts
  call conda activate sparktts
) else (
  call conda activate sparktts
)

REM Launch uvicorn in a new window
start "SparkTTS %PORT%" cmd /k uvicorn app_fastapi_tts:app --host 0.0.0.0 --port %PORT%
goto :END

:RESTART
call "%~f0" stop %PORT%
timeout /t 2 >nul
call "%~f0" start %PORT%
goto :END

:END
endlocal
