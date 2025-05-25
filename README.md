## Sistema de Visión por Computadora para Reconocimiento de Aves y Simulación Tetrocromática

### Descripción del proyecto

Este proyecto implementa una solución de Ciencia de Datos para simular la visión tetrocromática de las aves y realizar el reconocimiento y agrupación de especies a gran escala utilizando el dataset LaSBiRD y técnicas de visión por computadora.

### Justificación Biológica

Las aves poseen cuatro tipos de conos sensibles a distintas longitudes de onda (UV, corto, medio y largo), lo que les permite percibir rangos de luz —especialmente ultravioleta— que quedan fuera del espectro visible humano. Simular este canal UV proxy en imágenes RGB enriquece los descriptores de color y refuerza la diferenciación de especies según patrones evolutivos reales.

### Objetivos

1. Simular la percepción visual de las aves añadiendo un canal ultravioleta (UV) proxy a las imágenes RGB.
2. Segmentar automáticamente cada ave del fondo para aislar el sujeto.
3. Normalizar y preprocesar las imágenes recortadas (equalización de histogramas, z-score por canal).
4. Extraer descriptores de color, textura y forma de cada ave.
5. Agrupar especies según patrones de coloración y estrategias evolutivas (mimetismo, camuflaje, dimorfismo).
6. Clasificar especies empleando técnicas supervisadas (fine-tuning de CNNs 4-canal) y validar resultados.
7. Evaluar hipótesis sobre patrones de color independientes de las etiquetas taxonómicas.

### Dataset

**LaSBiRD (Large Scale Bird Recognition Dataset)**

* \~1.2 millones de imágenes.
* Anotaciones: bounding boxes, máscaras de segmentación, taxonomía (familia, género, especie), metadatos geográficos y temporales.

### Metodología (CRISP-DM)

#### 1. Comprensión del negocio

* Definir alcance: mejorar herramientas de investigación ecológica y conservación mediante análisis cuantitativo de patrones de color y segmentación precisa de sujetos.
* Identificar stakeholders: biólogos evolutivos, ornitólogos, equipos de conservación.

#### 2. Comprensión de los datos

* Exploración de LaSBiRD: distribución de especies, condiciones de iluminación y cobertura espacial.
* Análisis de calidad: revisión de bounding boxes, máscaras y metadatos faltantes.

#### 3. Preparación de los datos

1. **Detección y segmentación híbrida**: uso de YOLOv8-Seg (detección rápida + segmentación ligera) o Mask R-CNN para aislar aves.
2. **Recorte de sujetos**: extraer ROI de cada ave usando máscaras binaras.
3. **Simulación tetrocromática**:

   * Generar canal UV proxy (lectura empírica o estimación espectral).
   * Construir tensor de imagen 4-canal \[UV, R, G, B].
4. **Normalización**: equalización de histogramas (CLAHE) y estandarización (z-score) por canal.
5. División en train/validation/test estratificada.

#### 4. Modelado

* **Extracción de características**:

  * Histogramas multicanal y momentos (media, varianza).
  * Descriptores de textura (GLCM, LBP) y super-píxeles (SLIC).
* **Clustering y análisis no supervisado**:

  * PCA/UMAP + K-Means/HDBSCAN para descubrir agrupaciones según patrones de color.
* **Clasificación supervisada**:

  * Fine-tuning de CNNs 4-canal (ResNet50/EfficientNet o BEiT modificado).
  * Métricas: precisión, recall, F1 y matriz de confusión.

#### 5. Evaluación y validación

* Validación cruzada y análisis de errores (especies confundidas, impacto del canal UV).
* Pruebas estadísticas (ANOVA, tests post-hoc) para contrastar patrones de color entre clases.
* Análisis de mimetismo y convergencia en clusters no taxonómicos.
* Validación geoespacial: efecto de variabilidad ambiental en agrupaciones.

### Técnicas y Herramientas

* **Segmentación**: Ultralytics YOLOv8-Seg, Detectron2 (Mask R-CNN).
* **Transformación tetrocromática**: OpenCV, NumPy, SciPy.
* **Extracción de features**: scikit-image, scikit-learn.
* **Modelado**: PyTorch/Torchvision, timm (BEiT), scikit-learn, UMAP, HDBSCAN.
* **Visualización**: Matplotlib, Seaborn.
* **Gestión**: Python 3.9+, Git + GitHub.

### Estructura del repositorio

```
repositorio-lasbird/
├── README.md
├── data/                  # Scripts de descarga y preprocesamiento de LaSBiRD
├── notebooks/             # Exploración y prototipos en Jupyter
├── src/
│   ├── preprocessing/     # Detección, segmentación y simulación tetrocromática
│   ├── features/          # Extracción de descriptores de color y textura
│   ├── modeling/          # Clustering y clasificación supervisada
│   └── evaluation/        # Validación de resultados y análisis de errores
└── requirements.txt       # Dependencias Python
```

### Diferenciadores del Proyecto

* **Visión Tetrocromática Proxy-UV** a gran escala: integración de un canal UV estimado con calibración espectral.
* **Segmentación híbrida**: combinación de YOLOv8-Seg y Mask R-CNN para aislar aves con precisión y velocidad.
* **Embeddings avanzados**: fine-tuning de BEiT 4-canal y/o aprendizaje auto-supervisado (SimCLR/MoCo) para enriquecer representaciones de color y textura.
* **Análisis ecológico profundo**: estudio de mimetismo, convergencia y patrones evolutivos fuera de la taxonomía.
* **Reproducibilidad**: aplicación rigurosa del ciclo CRISP-DM con trazabilidad en cada fase.

---

*Este README está alineado con las recomendaciones de segmentación, normalización y clasificación basadas en patrones de color discutidas previamente.*

