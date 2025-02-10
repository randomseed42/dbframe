@echo off
setlocal EnableDelayedExpansion

cd /D "%~dp0\.."
call .venv\Scripts\activate

echo === run post install test ===
@REM python -m unittest discover tests
python -m unittest tests.test_dbframe tests.test_sqlite_handler_v2
echo.&echo.

endlocal