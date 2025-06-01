# Sistema de Visión por Computadora para Reconocimiento de Aves y Simulación Tetrocromática

---

## Descripción del Proyecto

Este repositorio implementa una solución de Ciencia de Datos que combina técnicas de visión por computadora, simulación tetrocromática y análisis ecológico. El objetivo principal es simular la percepción visual de las aves (cuatro canales: UV, R, G, B) y utilizar dicha información para:

1. Reconocimiento y clasificación de especies.
2. Agrupación no supervisada basada en patrones de coloración y estrategias evolutivas.
3. Análisis de hipótesis sobre mimetismo, camuflaje y convergencia evolutiva.

El proyecto se basa en dos fuentes de datos principales:

- **LaSBiRD**: Conjunto de imágenes de aves con bounding boxes y metadatos taxonómicos.
- **FeathersV1**: Imágenes de plumas de aves en formato TIFF (RGB + canal alfa).

A lo largo de las fases de CRISP-DM, se abordan desde la extracción y segmentación de aves, hasta la extracción de descriptores y el entrenamiento de modelos supervisados y no supervisados.

---

<!-- Espacio destinado a la imagen de portada del proyecto -->
  
![Ejemplo de visión tetrocromática](https://github.com/Vagarh/Sistema-de-Visi-n-por-Computadora-para-Reconocimiento-de-Aves-y-Simulaci-n-Tetrocrom-tica/blob/main/Imagenes/VISIOTETRACROMICA.jpg_large)

  

---

## Justificación Biológica

Las aves poseen un sistema visual tetrocromático: cuatro tipos de conos sensibles a distintas longitudes de onda (incluyendo ultravioleta). Esta capacidad les permite:

- Percibir patrones de color invisible para el ojo humano.
- Comunicar señales visuales relacionadas con cortejo, defensa territorial y reconocimiento de especies.

La simulación de un canal UV proxy über imágenes RGB amplía los descriptores de color y enriquece la extracción de características relevantes para diferenciar especies según patrones evolutivos reales. Esto resulta especialmente útil para:

- Estudios de biodiversidad y conservación.
- Investigación de mimetismo y estrategias de camuflaje.
- Análisis comparativo entre taxonomías tradicionales y agrupamientos basados en rasgos visuales.

---

## Objetivos

1. **Simular la percepción tetrocromática** de las aves añadiendo un canal ultravioleta (UV) proxy a imágenes originalmente capturadas en RGB.  
2. **Segmentar automáticamente** cada ave del fondo (ROI) usando modelos de segmentación hibrida (YOLOv8-Seg / Mask R-CNN).  
3. **Normalizar y preprocesar** las imágenes recortadas:
   - Equalización de histogramas (CLAHE) por canal.
   - Estandarización (z-score) por canal (UV, R, G, B).  
4. **Extraer descriptores** de color, textura y forma:
   - Histogramas multicanal y momentos estadísticos (media, varianza).  
   - Descriptores de textura: GLCM (Gray Level Co-Occurrence Matrix) y LBP (Local Binary Patterns).  
   - Super-píxeles (SLIC) para análisis de regiones homogéneas.  
5. **Agrupar especies** usando técnicas de clustering y reducción de dimensiones (PCA, UMAP + KMeans/HDBSCAN).  
6. **Clasificar especies** mediante métodos supervisados:
   - Fine-tuning de CNNs 4-canal ( BEiT adaptado).  
   - Métricas de desempeño: precisión, recall, F1 y matriz de confusión.  
7. **Evaluar hipótesis ecológicas**:
   - Contrastar patrones de color entre clusters no taxonómicos.
   - Analizar mimetismo y convergencia evolutiva mediante pruebas estadísticas (ANOVA, tests post-hoc).  
   - Validación geoespacial para estudiar el efecto de la variabilidad ambiental en agrupamientos.

---

## Datasets

### 1. LaSBiRD

- **Repositorio:** [LaSBiRD en GitHub](https://github.com/BirdColorBase/home)  
- **Descripción:** Base de datos espectral con mediciones de reflectancia (300–700 nm) para más de 2,500 especies de aves.  
- **Contenido:**
  - Archivos CSV con percent reflectance por “patch” de plumaje.
  - Espectros promediados en intervalos de 2 nm.
  - Metadatos de especies y parches medidos en museos (Ocean Optics USB2000 + PX-2).  

### 2. FeathersV1

- **Repositorio:** [FeathersV1 en GitHub](https://github.com/feathers-dataset/feathersv1-dataset)  
- **Descripción:** Conjunto de 28,272 imágenes de plumas de aves (cortadas) recopiladas de foros, sitios web públicos y motores de búsqueda.  
- **Estructura:**
  - Organización taxonómica simplificada: carpeta por Orden, subcarpeta por Especie.
  - Imágenes en TIFF con fondo variable (RGB; algunas con canal alfa tras remoción de fondo).  

---

## Metodología (CRISP-DM)

### 1. Comprensión del Negocio

- **Alcance:** Desarrollar herramientas cuantitativas para investigación ecológica y conservación basadas en análisis de patrones de color y segmentación de sujetos.  
- **Stakeholders:**  
  - Biólogos evolutivos.  
  - Ornitólogos.  
  - Equipos de conservación y gestión de biodiversidad.  

---

<!-- Espacio para un diagrama de flujo de CRISP-DM -->
  
  

### 2. Comprensión de los Datos

1. **Exploración de LaSBiRD**  
   - Distribución de especies por región geográfica.  
   - Calidad de metadatos: bounding boxes, máscaras, filtración de duplicados.  
   - Inspección de las longitudes de onda: uniformidad de las lecturas UV vs. visible.  
2. **Análisis de FeathersV1**  
   - Verificación de formatos (TIFF multicanal).  
   - Revisión de metadatos (nombres de archivos, correspondencia taxonómica).  
   - Mapeo de imágenes por especie y cálculo de estadísticas básicas de píxeles.

---

### 3. Preparación de los Datos

1. **Detección y Segmentación Híbrida**  
   - **YOLOv8-Seg**: detección rápida de aves con segmentación ligera.  
   - **Mask R-CNN (Detectron2)**: segmentación más precisa en casos complejos.  
   - Pipeline:  
     - Detectar bounding box con YOLOv8.  
     - Refinar máscara con Mask R-CNN si la forma es irregular.  
     - Extraer ROI (región de interés) de cada ave.  
2. **Recorte de Sujetos**  
   - Aplicar máscara binaria para aislar ave sobre fondo.  
   - Guardar recortes en carpeta `data/recortes/`.  
3. **Simulación Tetrocromática**  
   - **Generación de canal UV proxy:**  
     - Entrenar un modelo de regresión (HistGradientBoostingRegressor) en datos espectrales de LaSBiRD.  
     - Variables de entrada (X): valores RGB simulados a partir de espectros.  
     - Variable objetivo (y): reflectancia UV (300–400 nm).  
     - Pipeline de scikit-learn:  
       1. `PolynomialFeatures` (grado 2-3).  
       2. `StandardScaler`.  
       3. `HistGradientBoostingRegressor` con búsqueda aleatoria de hiperparámetros (RandomizedSearchCV).  
   - **Construcción de tensor 4-canal:**  
     - Leer imagen RGB recortada.  
     - Predecir valor UV para cada píxel con el modelo entrenado.  
     - Concatenar canales: `[UV_pred, R, G, B]`.  
     - Guardar imagen 4-canal en formato TIFF.  
4. **Normalización**  
   - **Ecualización de histogramas (CLAHE)** por canal (UV, R, G, B).  
   - **Estandarización (z-score)**: `(valor − media) / desviación_estándar` por canal.  
5. **División en Conjuntos**  
   - Partición estratificada (taxonómica/por especie) en:  
     - **Train**: 70 %  
     - **Validation**: 15 %  
     - **Test**: 15 %  

---

### 4. Modelado

1. **Extracción de Características**  
   - **Descriptores de Color:**  
     - Histogramas multicanal (UV, R, G, B).  
     - Momentos estadísticos: media, varianza, sesgo, curtosis.  
   - **Descriptores de Textura:**  
     - GLCM (Gray Level Co-Occurrence Matrix): contraste, correlación, homogeneidad, entropía.  
     - LBP (Local Binary Patterns).  
   - **Super-píxeles (SLIC)**: agrupación en regiones homogéneas, extracción de features localizadas.  
2. **Clustering y Análisis No Supervisado**  
   - **Reducción de Dimensión:**  
     - PCA para entender varianza explicada.  
     - UMAP para visualización en 2D/3D.  
   - **Algoritmos de Agrupamiento:**  
     - K-Means: evaluación con métricas internas (Silhouette, Davies-Bouldin, Calinski-Harabasz).  
     - HDBSCAN para detección de clusters de densidad variable.  
   - **Objetivo:** Identificar agrupaciones basadas en patrones de color + UV que puedan revelar mimetismo, convergencia o características evolutivas independientes de la taxonomía.  
3. **Clasificación Supervisada**  
   - **Arquitecturas CNN 4-canal:**  
     - **ResNet50 / EfficientNet modificados** para entrada de 4 canales.  
     - **BEiT (Vision Transformer) adaptado:**  
       - Ajuste de la primera capa patch_embed.proj para recibir 4 canales.  
       - Inicialización de pesos del canal UV con ligeras perturbaciones (copia de canal R + ruido).  
   - **Pipeline de Entrenamiento:**  
     1. Carga de imágenes 4-canal (224 × 224) con transformaciones (normalización, augmentation ligera).  
     2. Fine-tuning de la red preentrenada en ImageNet (RGB) adaptada a 4 canales.  
     3. Optimización con AdamW / SGD con scheduler de tasa de aprendizaje (CosineAnnealingLR).  
   - **Métricas de Evaluación:**  
     - Precisión por clase, recall, F1-score.  
     - Matriz de confusión global y por familia taxonómica.  
     - Curva ROC-AUC (por especie si hay binarización).  

---

### 5. Evaluación y Validación

1. **Validación Cruzada**  
   - K-Fold estratificado (por especie) con K = 5.  
   - Análisis de varianza de métricas entre folds.  
2. **Análisis de Errores**  
   - Inspección de clases confundidas:  
     - Especies con similitud morfológica cercana.  
     - Impacto del canal UV en la discriminación.  
3. **Pruebas Estadísticas**  
   - ANOVA y tests post-hoc (Tukey) para comparar promedios de métricas entre grupos taxonómicos.  
   - Pruebas de correlación (Spearman / Pearson) entre estadísticos de clusters y variables ambientales.  
4. **Evaluación Ecológica**  
   - Identificación de clústeres no taxonómicos con alta similitud cromática (posible mimetismo).  
   - Correlación geoespacial:  
     - Superponer coordenadas de muestreo (metadatos) con resultados de clustering.  
     - Análisis de variabilidad ambiental (temperatura, vegetación) vs. agrupamientos.  
5. **Visualización de Resultados**  
   - **Dendrogramas (Cladogramas):**  
     - Clustering jerárquico de promedios de embeddings por especie (RGB vs. RGB+UV).  
     - Cophenetic Correlation Coefficient para cuantificar ajuste.  
     - Tanglegram para comparar estructuras de dendrogramas.  
   - **Gráficos de UMAP / t-SNE:**  
     - Proyección 2D de embeddings para inspección visual de separabilidad.  
     - Coloreado por familia taxonómica o por cluster HDBSCAN.  
   - **Curvas Espectrales Promedio:**  
     - Graficar reflectancia media por canal (UV, R, G, B) vs. longitud de onda.  
     - Invertir eje X para mostrar UV (corto) a la izquierda.  

---

## Técnicas y Herramientas

| Fase                       | Herramientas / Bibliotecas                                                                                     |
|----------------------------|------------------------------------------------------------------------------------------------------------------|
| Segmentación               | Ultralytics YOLOv8-Seg, Detectron2 (Mask R-CNN)                                                                 |
| Simulación Tetrocromática  | OpenCV, NumPy, SciPy                                                                                             |
| Entrenamiento UV-Model     | scikit-learn (PolynomialFeatures, StandardScaler, HistGradientBoostingRegressor, RandomizedSearchCV)            |
| Extracción de Features     | scikit-image (GLCM, LBP), scikit-learn (PCA, UMAP, HDBSCAN), NumPy                                              |
| Modelado Supervisado       | PyTorch, Torchvision, timm (BEiT), torchvision.transforms                                                          |
| Evaluación de Clustering   | scikit-learn (KMeans, Silhouette Score, Davies-Bouldin, Calinski-Harabasz), scipy (Mantel Test)                |
| Visualización              | Matplotlib, Seaborn                                                                                                |
| Lectura/Escritura de TIFF  | tifffile, OpenCV                                                                                                  |
| Gestión de Datos y IO      | pandas, os, glob                                                                                                  |
| Orquestación de Notebooks  | Jupyter Notebook / Google Colab                                                                                   |
| Control de Versiones       | Git, GitHub                                                                                                       |
| Entorno de Desarrollo      | Python 3.9+, Conda / virtualenv                                                                                   |

---


## Resumen de Resultados

https://github.com/Vagarh/Sistema-de-Visi-n-por-Computadora-para-Reconocimiento-de-Aves-y-Simulaci-n-Tetrocrom-tica/blob/main/Imagenes/RGB-PLUMA.png

https://github.com/Vagarh/Sistema-de-Visi-n-por-Computadora-para-Reconocimiento-de-Aves-y-Simulaci-n-Tetrocrom-tica/blob/main/Imagenes/RGB%20VS%20RGB+UVA%20UMAP.png

- **Mejora en el Clustering No Supervisado**  
  - Utilizando solo canales RGB, el coeficiente de Silhouette promedio alcanzó **0.39**.  
  - Al incorporar el canal UV sintético (RGB+UVB), el coeficiente de Silhouette subió a **0.42**, lo que evidencia una mejor cohesión interna y separación de clusters.

- **Reorganización de la Jerarquía Evolutiva**  
  - El coeficiente cophenético para embeddings RGB fue **0.3501**, mientras que para RGB+UVB bajó a **0.2591**.  
  - Esta disminución refleja que el canal UV sintético introduce nuevas relaciones jerárquicas basadas en patrones ultravioleta, permitiendo identificar agrupaciones evolutivas que no se detectan con RGB únicamente.

- **Modificación en el Espacio de Embeddings**  
  - Al comparar las proyecciones en 2D, se observa que los embeddings RGB+UVB desplazan y reorganizan la nube de puntos.  
  - Esta reconfiguración sugiere que la dimensión UV aporta información discriminativa que modifica las distancias latentes entre imágenes, facilitando la separación de subconjuntos específicos.

- **Variabilidad Espectral del Canal UV**  
  - La curva espectral promedio muestra reflectancia más alta en R (~0.50) y G (~0.45), y disminuye hacia UV (~0.10).  
  - No obstante, existen muestras individuales con picos de reflectancia UV cercanos a 0.8, lo cual confirma que el canal UV sintético aporta suficiente variabilidad para mejorar la diferenciación de especies.

> **Conclusión breve:**  
> La integración del canal UV sintético **mejora la capacidad de separación** en tareas de clustering, **reconfigura la jerarquía** de similitud entre especies y añade **información espectral relevante**. Estos resultados validan el valor agregado del canal UV-proxy para el análisis de patrones de color en aves y su aplicación en estudios ecológicos y evolutivos.


```


