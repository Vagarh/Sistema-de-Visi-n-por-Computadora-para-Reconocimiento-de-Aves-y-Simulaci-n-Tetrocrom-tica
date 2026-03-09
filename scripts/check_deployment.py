#!/usr/bin/env python3
"""
Script para verificar que el proyecto esté listo para deployment en Streamlit Cloud
"""

import os
import sys

def check_file_exists(filepath, required=True):
    """Verificar si un archivo existe"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = "(REQUERIDO)" if required else "(OPCIONAL)"
    print(f"{status} {filepath} {req_text}")
    return exists

def check_deployment_readiness():
    """Verificar que el proyecto esté listo para deployment"""
    print("🚀 Verificando preparación para Streamlit Cloud")
    print("=" * 50)
    
    # Archivos requeridos
    print("\n📁 Archivos Requeridos:")
    required_files = [
        "app.py",
        "requirements.txt",
        "utils.py"
    ]
    
    all_required = True
    for file in required_files:
        if not check_file_exists(file, required=True):
            all_required = False
    
    # Archivos opcionales
    print("\n📁 Archivos Opcionales:")
    optional_files = [
        ".streamlit/config.toml",
        "Modelos/uv_regressor_hgb_2.joblib",
        "Modelos/uv_regressor_hgb_meta_2.json",
        "README_DEPLOYMENT.md"
    ]
    
    for file in optional_files:
        check_file_exists(file, required=False)
    
    # Verificar contenido de requirements.txt
    print("\n📋 Verificando requirements.txt:")
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            content = f.read()
            
        required_packages = ["streamlit", "numpy", "pandas", "matplotlib", "scikit-learn"]
        for package in required_packages:
            if package in content:
                print(f"✅ {package} encontrado")
            else:
                print(f"❌ {package} faltante")
                all_required = False
    
    # Verificar tamaño de archivos
    print("\n📊 Verificando tamaños de archivos:")
    for root, dirs, files in os.walk("."):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                if size > 100 * 1024 * 1024:  # 100MB
                    print(f"⚠️ {filepath}: {size/1024/1024:.1f}MB (muy grande para Streamlit Cloud)")
                elif size > 10 * 1024 * 1024:  # 10MB
                    print(f"⚠️ {filepath}: {size/1024/1024:.1f}MB (grande)")
            except:
                pass
    
    # Resultado final
    print("\n" + "=" * 50)
    if all_required:
        print("✅ PROYECTO LISTO PARA DEPLOYMENT")
        print("\n🚀 Pasos siguientes:")
        print("1. Sube los archivos a GitHub")
        print("2. Ve a https://share.streamlit.io")
        print("3. Conecta tu repositorio")
        print("4. Especifica 'app.py' como Main file path")
        print("5. ¡Deploy!")
    else:
        print("❌ PROYECTO NO LISTO - Faltan archivos requeridos")
        print("\n🔧 Soluciona los problemas marcados arriba")
    
    return all_required

if __name__ == "__main__":
    check_deployment_readiness()