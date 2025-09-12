@echo off
echo 🦅 Sistema de Vision Tetrocromatica - Modo Simplificado
echo ========================================================
echo.

echo Instalando dependencias minimas...
pip install streamlit numpy pandas pillow matplotlib opencv-python-headless scikit-learn joblib --quiet

echo.
echo Iniciando aplicacion...
streamlit run app.py

echo.
echo Presiona cualquier tecla para salir...
pause > nul