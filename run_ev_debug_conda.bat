@echo off
rem Foreground run with logs in the console (Ctrl+C to quit). Good for first-time testing.
cd /d "%~dp0"
set "CONDA_PY=%USERPROFILE%\miniconda3\envs\ev\python.exe"
if not exist "%CONDA_PY%" set "CONDA_PY=%USERPROFILE%\anaconda3\envs\ev\python.exe"
"%CONDA_PY%" -m ev --debug
