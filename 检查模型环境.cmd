@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts/configure_engine.py
) else (
  "..\photo-to-print-runtime\Scripts\python.exe" scripts/configure_engine.py
)
pause
