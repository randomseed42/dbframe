@echo off
setlocal EnableDelayedExpansion

cd /D "%~dp0\.."
call .venv\Scripts\activate

echo === run post install test ===
python -m unittest discover tests
echo.&echo.

endlocal