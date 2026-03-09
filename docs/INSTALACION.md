# 🛠️ Guía de Instalación - Sistema de Visión Tetrocromática

## Opciones de Instalación

### Opción 1: Instalación Automática (Recomendada)

**Windows:**
```bash
# Ejecutar archivo batch
run_app.bat
```

**Python directo:**
```bash
python install_and_run.py
```

### Opción 2: Instalación Simplificada

Si tienes problemas con la instalación completa:

**Windows:**
```bash
run_simple.bat
```

**Python directo:**
```bash
python run_simple.py
```

### Opción 3: Instalación Manual

1. **Instalar dependencias básicas:**
```bash
pip install streamlit numpy pandas pillow matplotlib scikit-learn joblib scipy
```

2. **Instalar OpenCV (una de estas opciones):**
```bash
# Opción A (completa)
pip install opencv-python

# Opción B (si la A falla)
pip install opencv-python-headless
```

3. **Ejecutar aplicación:**
```bash
streamlit run app.py
```

## Solución de Problemas Comunes

### Error: "No module named 'cv2'"

**Solución 1:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

**Solución 2:**
```bash
pip install --upgrade pip
pip install opencv-python --force-reinstall
```

### Error: "Microsoft Visual C++ 14.0 is required"

**Windows:**
1. Descargar e instalar "Microsoft C++ Build Tools"
2. O instalar Visual Studio Community
3. Reiniciar y volver a intentar

**Alternativa:**
```bash
pip install opencv-python-headless
```

### Error: "Failed building wheel"

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Problemas con PyTorch/timm

Si no necesitas funcionalidad completa de embeddings:
```bash
# Instalar solo dependencias básicas
pip install streamlit numpy pandas pillow matplotlib opencv-python-headless scikit-learn joblib scipy
```

La aplicación funcionará con funcionalidad limitada pero operativa.

## Verificación de Instalación

Ejecuta este comando para verificar las dependencias:

```python
python -c "
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sklearn
print('✅ Todas las dependencias básicas están instaladas')
"
```

## Dependencias por Funcionalidad

### Funcionalidad Básica (Requerida)
- `streamlit` - Interfaz web
- `numpy` - Operaciones numéricas
- `pandas` - Manejo de datos
- `pillow` - Procesamiento de imágenes
- `matplotlib` - Visualizaciones
- `opencv-python` o `opencv-python-headless` - Visión por computadora
- `scikit-learn` - Machine learning
- `joblib` - Carga de modelos
- `scipy` - Análisis científico

### Funcionalidad Avanzada (Opcional)
- `seaborn` - Visualizaciones mejoradas
- `umap-learn` - Proyecciones UMAP
- `hdbscan` - Clustering avanzado
- `torch` + `timm` - Embeddings con transformers
- `rembg` - Remoción de fondo profesional

## Estructura de Archivos Necesarios

```
├── app.py                    # Aplicación principal
├── utils.py                  # Utilidades (opcional)
├── requirements.txt          # Dependencias
├── Modelos/                  # Modelos entrenados
│   ├── uv_regressor_hgb_2.joblib
│   └── uv_regressor_hgb_meta_2.json
└── .streamlit/
    └── config.toml          # Configuración
```

## Contacto y Soporte

Si continúas teniendo problemas:

1. Verifica tu versión de Python: `python --version` (requiere 3.8+)
2. Actualiza pip: `pip install --upgrade pip`
3. Intenta la instalación simplificada
4. Revisa los logs de error para dependencias específicas

La aplicación está diseñada para funcionar con dependencias mínimas, por lo que siempre tendrás funcionalidad básica disponible.