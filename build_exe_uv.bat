@echo off
setlocal
cd /d "%~dp0"

echo Creating/updating uv environment...
if not exist ".venv\Scripts\python.exe" (
    uv venv
    if errorlevel 1 goto fail
)

set "PY=.venv\Scripts\python.exe"

echo Installing dependencies with uv...
uv pip install --python "%PY%" -r requirements.txt
if errorlevel 1 goto fail

uv pip install --python "%PY%" -r requirements-build.txt
if errorlevel 1 goto fail

echo Building EXE...
"%PY%" -m PyInstaller TradingAgentDemo.spec --clean --noconfirm
if errorlevel 1 goto fail

if not exist "dist\TradingAgentDemo\data" mkdir "dist\TradingAgentDemo\data"
if not exist "dist\TradingAgentDemo\artifacts" mkdir "dist\TradingAgentDemo\artifacts"

xcopy /E /I /Y "data" "dist\TradingAgentDemo\data" >nul
xcopy /E /I /Y "artifacts" "dist\TradingAgentDemo\artifacts" >nul

if not exist "dist\TradingAgentDemo\.streamlit" mkdir "dist\TradingAgentDemo\.streamlit"
xcopy /E /I /Y ".streamlit" "dist\TradingAgentDemo\.streamlit" >nul

echo.
echo Done. Run:
echo dist\TradingAgentDemo\TradingAgentDemo.exe
pause
exit /b 0

:fail
echo Build failed.
pause
exit /b 1
