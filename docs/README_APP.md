# 🦅 Aplicación Streamlit - Sistema de Visión Tetrocromática para Aves

## Descripción

Esta aplicación web interactiva permite analizar imágenes de aves simulando la percepción visual tetrocromática (incluyendo canal ultravioleta). La app integra todo el pipeline de procesamiento desarrollado en los notebooks del proyecto.

## Características Principales

### 🔬 Procesamiento de Imágenes
- **Segmentación automática**: Remoción de fondo de la imagen
- **Predicción UV**: Estimación del canal ultravioleta usando modelo entrenado
- **Visualización tetrocromática**: Representación RGB + UVB

### 📊 Análisis Exploratorio (EDA)
- Distribución de píxeles por canal (R, G, B, UVB)
- Estadísticas descriptivas por canal
- Visualización de intensidades UVB como heatmap

### 🧠 Análisis de Embeddings
- Extracción de características usando modelos BEiT
- Comparación RGB vs RGB+UVB
- Métricas de clustering (Silhouette, Davies-Bouldin, Calinski-Harabasz)

### 🗺️ Visualizaciones
- Proyecciones UMAP comparativas
- Cladogramas jerárquicos
- Análisis de coeficientes cophenéticos

## Instalación y Ejecución

### Opción 1: Ejecución Automática (Windows)
```bash
# Ejecutar el archivo batch
run_app.bat
```

### Opción 2: Ejecución Manual
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
```

## Uso de la Aplicación

1. **Cargar Imagen**: Sube una imagen de ave (JPG, PNG)
2. **Configurar Parámetros**: Ajusta el número de clusters en la barra lateral
3. **Visualizar Resultados**: 
   - Imagen original vs imagen con canal UV
   - Análisis EDA con distribuciones de píxeles
   - Métricas de clustering comparativas
   - Proyecciones UMAP
   - Cladogramas RGB vs RGB+UVB

## Estructura de Archivos

```
├── app.py                 # Aplicación principal Streamlit
├── run_app.bat           # Script de ejecución para Windows
├── requirements.txt      # Dependencias Python
├── .streamlit/
│   └── config.toml      # Configuración de Streamlit
├── Modelos/             # Modelos entrenados
│   ├── uv_regressor_hgb_2.joblib
│   └── uv_regressor_hgb_meta_2.json
└── README_APP.md        # Este archivo
```

## Funcionalidades Técnicas

### Modelos Utilizados
- **Modelo UV**: HistGradientBoostingRegressor para predicción de canal ultravioleta
- **BEiT**: Vision Transformer para extracción de embeddings
- **Clustering**: K-means y HDBSCAN para agrupamiento

### Métricas de Evaluación
- **Silhouette Score**: Calidad de separación de clusters
- **Davies-Bouldin Index**: Compacidad intra-cluster
- **Calinski-Harabasz Index**: Separación inter-cluster
- **Coeficiente Cophenético**: Fidelidad del dendrograma

## Interpretación de Resultados

### Mejoras con Canal UVB
- ✅ **Silhouette Score mayor**: Mejor separación de clusters
- ✅ **Davies-Bouldin menor**: Clusters más compactos
- ✅ **Cambios en coeficiente cophenético**: Nueva estructura jerárquica

### Validación de Hipótesis
La aplicación valida automáticamente si:
1. El canal UVB mejora la diferenciación cromática
2. Se incrementa la separación de clusters
3. Se revelan relaciones evolutivas adicionales

## Limitaciones Actuales

- **Segmentación**: Implementación simplificada (en producción usar rembg completo)
- **Embeddings**: Para demo se simulan múltiples muestras
- **Modelos BEiT**: Adaptación básica para 4 canales

## Próximas Mejoras

1. **Integración completa con rembg** para segmentación profesional
2. **Carga de múltiples imágenes** para análisis batch
3. **Exportación de resultados** en PDF/Excel
4. **Comparación con base de datos** de especies conocidas
5. **Análisis geoespacial** de patrones UV

## Soporte

Para problemas o sugerencias, consulta el README principal del proyecto o los notebooks de desarrollo.

---

**Nota**: Esta aplicación es una demostración interactiva del pipeline completo desarrollado en los notebooks. Para análisis de investigación, se recomienda usar los notebooks originales con datasets completos.