@echo off
echo.
echo  ========================================
echo   College Culture Matcher - Setup ^& Run
echo   Bama Builds Hackathon
echo  ========================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download it from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/2] Installing dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies. Try running:
    echo   pip install streamlit plotly pandas numpy scikit-learn requests vaderSentiment
    pause
    exit /b 1
)

echo.
echo [2/2] Launching app...
echo.
echo  The app will open in your browser automatically.
echo  If it doesn't, go to: http://localhost:8501
echo.
echo  Press Ctrl+C in this window to stop the app.
echo.
streamlit run app.py
