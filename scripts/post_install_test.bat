@echo off
setlocal EnableDelayedExpansion

cd /D "%~dp0\.."
call .venv\Scripts\activate

echo === run post install test ===
python -m unittest discover tests
@REM python -m unittest tests.test_dbframe tests.test_sqlite_handler
@REM python -m unittest tests.test_dbframe tests.test_pg_handler
echo.&echo.

endlocal