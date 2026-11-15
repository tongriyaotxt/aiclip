@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [!] missing .venv, run setup_env.ps1 first
  pause
  exit /b 1
)
echo Starting AIClip UI at http://127.0.0.1:7860
".venv\Scripts\python.exe" -m aiclip ui
pause
