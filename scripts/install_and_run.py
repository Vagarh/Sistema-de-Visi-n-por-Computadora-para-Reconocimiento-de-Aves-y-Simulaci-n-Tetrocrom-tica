#!/usr/bin/env python3
"""
Script de instalación y ejecución para la aplicación Streamlit
Sistema de Visión Tetrocromática para Aves
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Verificar versión de Python"""
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"Versión actual: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detectado")
    return True

def install_requirements():
    """Instalar dependencias desde requirements.txt"""
    print("\n📦 Instalando dependencias...")
    
    # Primero actualizar pip
    try:
        print("🔄 Actualizando pip...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
    except subprocess.CalledProcessError:
        print("⚠️ No se pudo actualizar pip, continuando...")
    
    # Instalar dependencias básicas primero
    basic_deps = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "pillow>=8.3.0",
        "matplotlib>=3.5.0",
        "streamlit>=1.28.0"
    ]
    
    print("📦 Instalando dependencias básicas...")
    for dep in basic_deps:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ])
            print(f"✅ {dep.split('>=')[0]} instalado")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error instalando {dep}: {e}")
    
    # Instalar OpenCV con manejo especial
    print("📦 Instalando OpenCV...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "opencv-python>=4.5.0"
        ])
        print("✅ OpenCV instalado")
    except subprocess.CalledProcessError:
        print("⚠️ Error con opencv-python, intentando opencv-python-headless...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "opencv-python-headless>=4.5.0"
            ])
            print("✅ OpenCV headless instalado")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando OpenCV: {e}")
    
    # Instalar el resto desde requirements.txt
    try:
        print("📦 Instalando dependencias restantes...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Todas las dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Algunas dependencias no se pudieron instalar: {e}")
        print("La aplicación funcionará con funcionalidad limitada")
        return True  # Continuar de todos modos

def check_models():
    """Verificar que los modelos estén disponibles"""
    model_files = [
        "Modelos/uv_regressor_hgb_2.joblib",
        "Modelos/uv_regressor_hgb_meta_2.json"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Archivos de modelo faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        print("La aplicación funcionará con funcionalidad limitada")
        return False
    
    print("✅ Modelos encontrados")
    return True

def run_streamlit():
    """Ejecutar la aplicación Streamlit"""
    print("\n🚀 Iniciando aplicación Streamlit...")
    print("La aplicación se abrirá en tu navegador web")
    print("Presiona Ctrl+C para detener la aplicación")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Aplicación detenida por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando Streamlit: {e}")

def main():
    """Función principal"""
    print("🦅 Sistema de Visión Tetrocromática para Aves")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        input("Presiona Enter para salir...")
        return
    
    # Verificar sistema operativo
    os_name = platform.system()
    print(f"💻 Sistema operativo: {os_name}")
    
    # Instalar dependencias
    if not install_requirements():
        print("\n❌ No se pudieron instalar las dependencias")
        input("Presiona Enter para salir...")
        return
    
    # Verificar modelos
    check_models()
    
    # Ejecutar aplicación
    run_streamlit()

if __name__ == "__main__":
    main()