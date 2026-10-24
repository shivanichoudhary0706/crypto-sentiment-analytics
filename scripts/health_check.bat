@echo off
REM Pipeline health check + self-repair. Double-click, or run from cmd.
REM   scripts\health_check.bat                  report -> repair -> report
REM   scripts\health_check.bat --check-only     report only
REM   scripts\health_check.bat --with-news      also fill OLD news gaps (spends CryptoCompare quota)
REM   scripts\health_check.bat --with-reports   also re-run lag / event study / backtest / tuning
cd /d "%~dp0.."
call venv\Scripts\activate.bat
if not exist logs mkdir logs
python -m tools.health_check %* 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath logs\health_check_last.txt"
echo.
echo Full output saved to logs\health_check_last.txt
pause
