@echo off
setlocal
cd /d "%~dp0"

echo.
echo ==========================================
echo  Building TradingAgentDemo.exe with uv
echo ==========================================
echo.

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found in this terminal.
    echo Install uv or open a terminal where uv is available.
    echo.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    echo .venv was not found. Creating it with uv...
    uv venv
    if errorlevel 1 goto fail
    set "PY=.venv\Scripts\python.exe"
)

echo Using Python:
"%PY%" -c "import sys; print(sys.executable)"
if errorlevel 1 goto fail

echo.
echo Installing runtime requirements with uv...
uv pip install --python "%PY%" -r requirements.txt
if errorlevel 1 goto fail

echo.
echo Installing PyInstaller requirements with uv...
uv pip install --python "%PY%" -r requirements-build.txt
if errorlevel 1 goto fail

echo.
echo Cleaning old build output...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo Running PyInstaller...
"%PY%" -m PyInstaller TradingAgentDemo.spec --clean --noconfirm
if errorlevel 1 goto fail

echo.
echo Copying local data and artifact folders beside the EXE...
if not exist "dist\TradingAgentDemo\data" mkdir "dist\TradingAgentDemo\data"
if not exist "dist\TradingAgentDemo\artifacts" mkdir "dist\TradingAgentDemo\artifacts"

xcopy /E /I /Y "data" "dist\TradingAgentDemo\data" >nul
xcopy /E /I /Y "artifacts" "dist\TradingAgentDemo\artifacts" >nul

if not exist "dist\TradingAgentDemo\.streamlit" mkdir "dist\TradingAgentDemo\.streamlit"
xcopy /E /I /Y ".streamlit" "dist\TradingAgentDemo\.streamlit" >nul

echo.
echo ==========================================
echo  Build complete
echo ==========================================
echo.
echo Run:
echo   dist\TradingAgentDemo\TradingAgentDemo.exe
echo.
echo IMPORTANT:
echo   Distribute the whole dist\TradingAgentDemo folder,
echo   not only the .exe file.
echo.
pause
exit /b 0

:fail
echo.
echo Build failed.
echo Check the error above.
pause
exit /b 1
