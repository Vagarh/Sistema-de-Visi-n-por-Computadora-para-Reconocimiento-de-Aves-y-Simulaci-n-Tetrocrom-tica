#!/usr/bin/env python3
"""
Script simple para ejecutar la aplicación con dependencias mínimas
"""

import subprocess
import sys
import os

def install_minimal_deps():
    """Instalar solo las dependencias mínimas necesarias"""
    minimal_deps = [
        "streamlit",
        "numpy", 
        "pandas",
        "pillow",
        "matplotlib",
        "opencv-python-headless",  # Versión headless más compatible
        "scikit-learn",
        "joblib"
    ]
    
    print("📦 Instalando dependencias mínimas...")
    
    for dep in minimal_deps:
        try:
            print(f"Instalando {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep, "--quiet"
            ])
            print(f"✅ {dep} instalado")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error instalando {dep}: {e}")
            continue
    
    print("✅ Instalación de dependencias mínimas completada")

def main():
    print("🦅 Sistema de Visión Tetrocromática - Modo Simplificado")
    print("=" * 55)
    
    # Verificar Python
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        input("Presiona Enter para salir...")
        return
    
    print(f"✅ Python {sys.version.split()[0]} detectado")
    
    # Instalar dependencias mínimas
    install_minimal_deps()
    
    # Ejecutar aplicación
    print("\n🚀 Iniciando aplicación Streamlit...")
    print("La aplicación se abrirá en tu navegador web")
    print("Presiona Ctrl+C para detener la aplicación")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Aplicación detenida por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando Streamlit: {e}")
        print("\nIntenta ejecutar manualmente:")
        print("streamlit run app.py")

if __name__ == "__main__":
    main()