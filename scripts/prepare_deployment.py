#!/usr/bin/env python3
"""
Script para preparar el proyecto para deployment en Streamlit Cloud
"""

import os
import shutil
import sys

def prepare_for_deployment():
    """Preparar archivos para deployment"""
    print("🚀 Preparando proyecto para Streamlit Cloud...")
    print("=" * 50)
    
    # 1. Usar configuración de cloud
    if os.path.exists(".streamlit/config_cloud.toml"):
        shutil.copy(".streamlit/config_cloud.toml", ".streamlit/config.toml")
        print("✅ Configuración de cloud aplicada")
    
    # 2. Verificar archivos esenciales
    essential_files = [
        "app.py",
        "requirements.txt", 
        "utils.py",
        ".streamlit/config.toml"
    ]
    
    missing_files = []
    for file in essential_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    if missing_files:
        print(f"\n❌ Archivos faltantes: {', '.join(missing_files)}")
        return False
    
    # 3. Crear archivo de información del deployment
    deployment_info = f"""# Deployment Info
Proyecto: Sistema de Visión Tetrocromática para Aves
Desarrollador: Juan Felipe Cardona Arango
Fecha: Enero 2025
Archivo principal: app.py
Configuración: Optimizada para Streamlit Cloud
"""
    
    with open("deployment_info.txt", "w", encoding="utf-8") as f:
        f.write(deployment_info)
    
    print("✅ deployment_info.txt creado")
    
    # 4. Mostrar instrucciones finales
    print("\n" + "=" * 50)
    print("✅ PROYECTO LISTO PARA DEPLOYMENT")
    print("\n📋 Archivos preparados:")
    print("- app.py (archivo principal)")
    print("- requirements.txt (dependencias)")
    print("- utils.py (utilidades)")
    print("- .streamlit/config.toml (configuración)")
    print("- Modelos/ (si existen)")
    
    print("\n🚀 Pasos para subir a Streamlit Cloud:")
    print("1. Crea un repositorio en GitHub")
    print("2. Sube estos archivos al repositorio")
    print("3. Ve a https://share.streamlit.io")
    print("4. Conecta tu repositorio")
    print("5. Especifica 'app.py' como Main file path")
    print("6. ¡Deploy!")
    
    print(f"\n🌐 Tu aplicación estará disponible en:")
    print("https://[tu-usuario]-[nombre-repo]-main-[hash].streamlit.app")
    
    return True

def restore_local_config():
    """Restaurar configuración para desarrollo local"""
    print("🔧 Restaurando configuración local...")
    
    local_config = """[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 50
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false"""
    
    with open(".streamlit/config.toml", "w") as f:
        f.write(local_config)
    
    print("✅ Configuración local restaurada")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore_local_config()
    else:
        prepare_for_deployment()