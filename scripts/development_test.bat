@echo off
setlocal EnableDelayedExpansion

cd /D "%~dp0\.."
call .venv\Scripts\activate

echo === run development test ===
set pythonpath=src
python -m unittest discover tests
@REM python -m unittest tests.test_dbframe tests.test_sqlite_handler_v2
@REM python -m unittest tests.test_dbframe tests.test_pg_handler_v2
echo.&echo.

endlocal