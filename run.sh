#!/bin/bash
echo ""
echo "  ========================================"
echo "   College Culture Matcher - Setup & Run"
echo "   Bama Builds Hackathon"
echo "  ========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[ERROR] Python is not installed."
        echo "Install it from https://www.python.org/downloads/"
        exit 1
    fi
    PY=python
else
    PY=python3
fi

echo "[1/2] Installing dependencies..."
$PY -m pip install -r requirements.txt --quiet

echo ""
echo "[2/2] Launching app..."
echo ""
echo "  The app will open in your browser automatically."
echo "  If it doesn't, go to: http://localhost:8501"
echo ""
echo "  Press Ctrl+C to stop the app."
echo ""
$PY -m streamlit run app.py
