@echo off
setlocal EnableDelayedExpansion

cd /D "%~dp0\.."
call .venv\Scripts\activate

echo === run development test ===
set pythonpath=src
@REM python -m unittest discover tests
@REM python -m unittest tests.test_dbframe tests.test_sqlite_handler
@REM python -m unittest tests.test_dbframe tests.test_pg_handler
@REM python -m unittest tests.test_dbframe tests.test_utils
python -m unittest tests.test_utils.TestUtils.test_dataframe_to_sql_columns
echo.&echo.

endlocal