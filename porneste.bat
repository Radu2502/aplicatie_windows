@echo off
rem Porneste aplicatia "Postura calaretului". La prima rulare instaleaza mediul (Python si bibliotecile).
setlocal
cd /d "%~dp0"
title Postura calaretului

set "UV=uv"
where uv >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
        echo.
        echo Programul uv nu este instalat sau nu este gasit.
        echo Instaleaza-l din PowerShell cu comanda din README.md, sectiunea "Instalare pe Windows 11",
        echo apoi inchide aceasta fereastra si da din nou dublu click pe porneste.bat.
        echo.
        pause
        exit /b 1
    )
)

if not exist ".venv\" (
    echo Prima pornire: instalez Python si bibliotecile. Poate dura cateva minute, nu inchide fereastra.
)
"%UV%" sync --frozen --no-dev
if errorlevel 1 (
    echo.
    echo Instalarea nu a reusit. Verifica legatura la internet si incearca din nou.
    echo.
    pause
    exit /b 1
)

echo.
echo Pornesc aplicatia. Browserul se deschide singur la http://localhost:8501
echo Ca sa opresti aplicatia, inchide aceasta fereastra sau apasa Ctrl+C.
echo.
"%UV%" run --frozen --no-dev streamlit run app\app.py
pause
