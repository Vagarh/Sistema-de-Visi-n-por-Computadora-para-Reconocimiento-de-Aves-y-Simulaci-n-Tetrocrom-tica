## Sistema de Visión por Computadora para Reconocimiento de Aves y Simulación Tetrocromática

### Descripción del proyecto

Este proyecto implementa una solución de **Ciencia de Datos** para simular la visión tetrocromática de las aves y realizar el reconocimiento y agrupación de especies a gran escala utilizando el dataset **LaSBiRD** y técnicas de visión por computadora.

### Justificación Biológica

Las aves poseen cuatro tipos de conos sensibles a distintas longitudes de onda (UV, corto, medio y largo), lo que les permite percibir rangos de luz—especialmente ultravioleta—que quedan fuera del espectro visible humano. Esta capacidad influye en conducta de apareamiento, detección de alimento y camuflaje. Simular este canal UV proxy en imágenes RGB enriquece los descriptores de color y refuerza la diferenciación de especies según patrones evolutivos reales.

### Objetivos

* **Simular la percepción visual** de las aves añadiendo un canal ultravioleta (UV) proxy a las imágenes RGB.
* **Extraer descriptores de color y forma** de cada ave recortada.
* **Agrupar** especies según patrones de coloración y estrategias evolutivas (mimetismo, camuflaje, dimorfismo).
* **Clasificar** especies empleando técnicas supervisadas (fine-tuning de CNNs) y validar resultados frente a las etiquetas de LaSBiRD.

### Dataset

* **LaSBiRD (Large Scale Bird Recognition Dataset)**

  * \~1.2 millones de imágenes.
  * Anotaciones: bounding boxes, máscaras de segmentación, taxonomía (familia, género, especie), metadatos geográficos y temporales.

### Metodología (CRISP-DM)

1. **Comprensión del negocio**

   * Definir alcance: mejorar herramientas de investigación ecológica y conservación mediante análisis cuantitativo de patrones de color.
   * Identificar stakeholders: biólogos evolutivos, ornitólogos, equipos de conservación.

2. **Comprensión de los datos**

   * Exploración de LaSBiRD: distribución de especies, condiciones de iluminación y variabilidad geográfica.
   * Análisis de calidad: revisión de bounding boxes, máscaras y metadatos incompletos.

3. **Preparación de los datos**

   * **Detección y segmentación**: usar modelos como YOLOv5 para detección de aves y Mask R-CNN para segmentación precisa de plumaje.
   * **Recorte automático** de aves: aislar cada ave usando bounding boxes y máscaras con OpenCV/PIL.
   * **Simulación tetrocromática**: generar canal UV proxy y aplicar matriz de transformación sobre canales RGB.
   * **Normalización y partición** en `train`/`val`/`test`.

4. **Modelado**

   * **Extracción de características**: histogramas multicanal, momentos de color, descriptores de textura.
   * **Reducción de dimensionalidad**: PCA o UMAP para visualización y preprocesamiento.
   * **Clustering**: K-means o HDBSCAN; evaluación con ARI y purity contra taxonomía real.
   * **Clasificación supervisada**: fine-tuning de ResNet50/EfficientNet para 4 canales; métricas de precisión, recall y F1.

5. **Evaluación**

   * Validación cruzada y análisis de errores: especies confundidas, impacto de proxy UV y calidad de segmentación.
   * Estudio de mimetismo: análisis de clusters atípicos.

### Técnicas de Visión por Computadora

* **YOLOv5/YOLOv7** para detección rápida de aves en imágenes de alta resolución.
* **Mask R-CNN** para segmentación detallada del contorno y áreas de plumaje.
* **Redes neuronales convolucionales (CNN)** preentrenadas (ResNet, EfficientNet) adaptadas a 4 canales para clasificación.
* **Aprendizaje auto-supervisado (SimCLR, MoCo)** opcional para mejorar embeddings de color y textura.

### Estructura del repositorio

```text
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

### Tecnologías y herramientas

* **Lenguaje**: Python 3.9+
* **Librerías**: NumPy, OpenCV, PyTorch/Torchvision, scikit-learn, UMAP, HDBSCAN, Detectron2 (Mask R-CNN), Ultralytics YOLO
* **Control de versiones**: Git + GitHub

---

### Diferenciador del Proyecto

* **Visión Tetrocromática Proxy-UV a Gran Escala**: Integra un canal UV aproximado con técnicas de calibración basadas en estudios espectrales, aplicado a \~1.2 M de imágenes de LaSBiRD.
* **Segmentación de Precisión Híbrida**: Combina detección ultrarrápida (YOLOv5/YOLOv7) con segmentación detallada (Mask R-CNN) para aislar plumajes complejos.
* **Embeddings Mejorados por Aprendizaje Auto‑Supervisado**: Incorpora SimCLR/MoCo para enriquecer representaciones de color y textura que capturan señales evolutivas (mimetismo, dimorfismo).
* **Análisis Ecológico Avanzado**: Más allá de la clasificación, estudia patrones de mimetismo y convergencia ecológica en clusters no taxonómicos.
* **Metodología Integral CRISP‑DM** aplicada de forma estructurada en todas las fases, asegurando reproducibilidad y trazabilidad.
* **Validación Geoespacial**: Aprovecha metadatos de ubicación y fecha para evaluar robustez de clusters según variabilidad ambiental.

---

*Este README muestra la descripción completa del proyecto alineada con CRISP-DM y las técnicas de visión por computadora planeadas.* la descripción completa del proyecto alineada con CRISP-DM y las técnicas de visión por computadora planeadas.\*
