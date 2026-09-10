@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" start_app.py
) else (
  "..\photo-to-print-runtime\Scripts\python.exe" start_app.py
)
pause
