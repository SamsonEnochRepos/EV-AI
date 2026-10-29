@echo off
rem Starts EV from the "ev" conda environment (no console window). Use --debug via run_ev_debug_conda.bat.
cd /d "%~dp0"
set "CONDA_PY=%USERPROFILE%\miniconda3\envs\ev\pythonw.exe"
if not exist "%CONDA_PY%" set "CONDA_PY=%USERPROFILE%\anaconda3\envs\ev\pythonw.exe"
if not exist "%CONDA_PY%" (
  echo Conda env "ev" not found. See README: Conda setup.
  pause
  exit /b 1
)
start "" /B "%CONDA_PY%" -m ev
