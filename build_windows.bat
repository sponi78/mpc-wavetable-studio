@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo  MPC Wavetable Studio - Windows Build
echo ============================================

where py >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or 3.12 from python.org.
  pause
  exit /b 1
)

py -m pip install --upgrade pip
if errorlevel 1 goto :error
py -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

py -m PyInstaller --noconfirm --clean --onedir --windowed ^
  --name "MPC Wavetable Studio" ^
  --collect-all soundfile ^
  mpc_wavetable_splitter.py
if errorlevel 1 goto :error

echo.
echo Build finished successfully.
echo Output: dist\MPC Wavetable Studio\MPC Wavetable Studio.exe
explorer "dist\MPC Wavetable Studio"
pause
exit /b 0

:error
echo.
echo Build failed. Please review the error messages above.
pause
exit /b 1
