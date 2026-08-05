@echo off
setlocal
cd /d "%~dp0"
py -m pip install --upgrade pip
py -m pip install -r requirements.txt pyinstaller
py -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name "MPC Wavetable Studio" ^
  --collect-all soundfile ^
  mpc_wavetable_splitter.py
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)
explorer dist
pause
