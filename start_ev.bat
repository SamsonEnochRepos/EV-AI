@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  echo Run setup first: see README.md
  pause
  exit /b 1
)
start "" /B .venv\Scripts\pythonw.exe -m ev
