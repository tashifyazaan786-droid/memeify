@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -m venv .venv
  if errorlevel 1 (
    echo Failed to create .venv. Make sure Python 3.10+ is installed and the py launcher works.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if not exist ".env" (
  copy /Y ".env.example" ".env" >nul
  echo Created .env from .env.example. Add your GROQ_API_KEY before continuing.
)

echo.
echo Starting Meme Knowledge Agent...
python app.py
pause
