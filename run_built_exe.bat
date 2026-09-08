@echo off
cd /d "%~dp0"
if exist "dist\TradingAgentDemo\TradingAgentDemo.exe" (
    start "" "dist\TradingAgentDemo\TradingAgentDemo.exe"
) else (
    echo EXE not found. Run build_exe.bat first.
    pause
)
