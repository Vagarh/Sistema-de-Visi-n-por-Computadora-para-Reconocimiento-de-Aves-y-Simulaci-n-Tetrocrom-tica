# 🚀 Deployment en Streamlit Cloud

## Archivos Necesarios

Para subir este proyecto a Streamlit Cloud, necesitas estos archivos:

### Archivos Obligatorios:
- `app.py` ⭐ (archivo principal)
- `requirements.txt`
- `utils.py`

### Archivos Opcionales:
- `.streamlit/config.toml` (configuración de la app)
- `Modelos/uv_regressor_hgb_2.joblib` (modelo entrenado)
- `Modelos/uv_regressor_hgb_meta_2.json` (metadata del modelo)

## 📋 Pasos para Deployment

### 1. Preparar Repositorio GitHub
```bash
# Crear repositorio en GitHub con estos archivos:
app.py
utils.py
requirements.txt
.streamlit/config.toml
Modelos/ (carpeta con modelos)
```

### 2. Configurar Streamlit Cloud
1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu cuenta de GitHub
3. Selecciona tu repositorio
4. **Main file path**: `app.py`
5. **Python version**: 3.9 o superior

### 3. Variables de Entorno (si necesarias)
No se requieren variables de entorno especiales para este proyecto.

## ⚠️ Consideraciones Importantes

### Limitaciones de Streamlit Cloud:
- **Memoria**: ~1GB RAM disponible
- **CPU**: Limitado para modelos pesados
- **Almacenamiento**: ~1GB para archivos

### Optimizaciones Implementadas:
- ✅ OpenCV headless (más ligero)
- ✅ Modelos opcionales (PyTorch comentado)
- ✅ Carga lazy de dependencias pesadas
- ✅ Fallbacks para funcionalidad limitada

## 🔧 Troubleshooting

### Si falla la instalación:
1. Verifica que `requirements.txt` esté en la raíz
2. Usa `opencv-python-headless` en lugar de `opencv-python`
3. Comenta PyTorch si hay problemas de memoria

### Si falta el modelo UV:
- La app funcionará con modelo simulado
- Mensaje informativo al usuario
- Funcionalidad completa disponible

## 📊 Funcionalidad en Streamlit Cloud

### Disponible:
- ✅ Carga de imágenes
- ✅ Segmentación básica
- ✅ Predicción UV (simulada si no hay modelo)
- ✅ Análisis EDA completo
- ✅ Clustering con scikit-learn
- ✅ Visualizaciones matplotlib
- ✅ Panel técnico
- ✅ Exportación de resultados

### Limitada:
- ⚠️ Modelos PyTorch (por memoria)
- ⚠️ Procesamiento de imágenes muy grandes
- ⚠️ Análisis batch de múltiples imágenes

## 🎯 URL de la Aplicación

Una vez deployada, tu aplicación estará disponible en:
```
https://[tu-usuario]-[nombre-repo]-[branch]-[hash].streamlit.app
```

## 📞 Soporte

Si tienes problemas con el deployment:
1. Revisa los logs en Streamlit Cloud
2. Verifica que todos los archivos estén en GitHub
3. Asegúrate de que `app.py` esté en la raíz del repositorio