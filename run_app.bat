@echo off
cd /d "%~dp0"
echo Starting local Streamlit trading-agent demo...
python -m streamlit run app.py
pause
