@echo off
set PORT=%1
if "%PORT%"=="" set PORT=8000

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do taskkill /F /PID %%p
echo Stopped anything on :%PORT%
