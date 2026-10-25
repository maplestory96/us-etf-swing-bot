@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ========================================================
echo   미국 ETF 스윙 시그널 AI 봇 (차트 주력 + 뉴스 보조)
echo ========================================================
echo.
".\.venv\Scripts\python.exe" main.py
echo.
pause
