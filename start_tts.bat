@echo off
set PORT=%1
if "%PORT%"=="" set PORT=8000

call conda activate sparktts
uvicorn app_fastapi_tts:app --host 0.0.0.0 --port %PORT%
