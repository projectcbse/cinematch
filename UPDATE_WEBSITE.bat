@echo off
title CineMatch - update website
cd /d "%~dp0"
echo.
echo ==========================================
echo    CineMatch - rebuilding the website
echo ==========================================
echo.
set PY=
where py >nul 2>nul && set PY=py
if "%PY%"=="" where python >nul 2>nul && set PY=python
if "%PY%"=="" (
  echo [PROBLEM] Python was not found. Install Python 3.12 from https://www.python.org/downloads/
  pause
  exit /b
)
%PY% -c "import Orange" >nul 2>nul
if errorlevel 1 (
  echo [STEP 1] Installing Orange Data Mining - first time only, a few minutes...
  %PY% -m pip install orange3
  echo.
)
echo [STEP 2] Running the Orange AI and building the website...
%PY% build_website.py
if errorlevel 1 (
  echo.
  echo [PROBLEM] Something went wrong - read the message above.
  pause
  exit /b
)
echo.
echo Opening the website...
start "" "website\index.html"
timeout /t 4 >nul
