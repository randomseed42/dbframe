@echo off
setlocal EnableDelayedExpansion

cd /D "%~dp0\.."
call .venv\Scripts\activate

echo === run build and install ===
python -m build
for /f "delims=" %%i in ('python -c "from src.dbframe import __version__;print(__version__)"') do set VER=%%i
pip install dist/dbframe-%VER%-py3-none-any.whl --force-reinstall --no-deps
echo.&echo.

endlocal