@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [!] missing .venv, run setup_env.ps1 first
  exit /b 1
)
".venv\Scripts\python.exe" -m aiclip %*
