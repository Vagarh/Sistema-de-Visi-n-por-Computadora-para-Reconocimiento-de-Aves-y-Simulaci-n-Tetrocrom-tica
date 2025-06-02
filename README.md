# Sistema de Visión por Computadora para Reconocimiento de Aves con simulación Tetrocromática mediante la adicion de un canal UVB estimado en imagenes RGB

---

## Descripción del Proyecto

Este repositorio implementa una solución de Ciencia de Datos que combina técnicas de visión por computadora, simulación tetrocromática y análisis ecológico. El objetivo principal es simular la percepción visual de las aves (cuatro canales: UV, R, G, B) y utilizar dicha información para:

1. Reconocimiento y clasificación de especies.
2. Agrupación no supervisada basada en patrones de coloración y estrategias evolutivas.(BEIT ajustado)

El proyecto se basa en tres fuentes de datos principales:

- **BirdColorbase** Conjunto de mediciones de espectofotometris de 2500 especies de aves.
- **FeathersV1**: Imágenes de plumas de aves en formato JGP.

A lo largo de las fases de CRISP-DM, se abordan desde la extracción y segmentación de aves, hasta la extracción de descriptores y el entrenamiento de modelos supervisados y no supervisados.

## Hipotesis 

La adición de un canal ultravioleta estimado a las imágenes RGB de aves incrementa la diferenciación cromática, mejora la separación de clústeres y revela relaciones evolutivas y ecológicas que no se detectan utilizando solamente el espacio de color visible.

## Criterios de Exito

En conjunto, la hipótesis se valida cuando se demuestre que:

- Métricas cuantitativas (Silhouette, cophenetic, precisión de clasificación) mejoran al incluir UVB.

- Visualmente, la separación de grupos en proyecciones UMAP/PCA es más nítida con el canal UV.

- Dendrogramas muestran reagrupamientos significativos en RGB+UVB que no aparecen en RGB, indicando patrones evolutivos o ecológicos adicionales.

De confirmarse, este resultado avalaría la pertinencia de emplear simulación tetrocromática en aplicaciones de visión por computadora orientadas a ornitología, conservación y análisis evolutivo.

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

## Objetivos

1. **Simular la percepción tetrocromática**  
   - Generar un canal ultravioleta (UV) proxy a partir de imágenes RGB de plumas, usando un modelo de regresión entrenado con datos espectrales.

2. **Segmentar automáticamente las plumas**  
   - Aplicar `rembg` o un modelo basado en U²-Net para obtener una máscara alfa que aísle la pluma del fondo, dejando únicamente el sujeto en primer plano.

3. **Normalizar y preprocesar las imágenes**  
   - Redimensionar cada recorte a 224 × 224 píxeles, conservando los cuatro canales (UV, R, G, B).  
   - Estandarizar (z-score) cada canal de las imágenes 4-canal para garantizar comparabilidad.

4. **Extraer descriptores y embeddings**  
   - Emplear un modelo BEiT preentrenado (3 canales) para extraer embeddings RGB.  
   - Adaptar BEiT a entrada 4 canales (RGB+UV) y extraer los embeddings correspondientes.  
   - Calcular la diferencia y la distancia entre embeddings RGB vs. RGB+UVB para cuantificar el aporte del canal UV.

5. **Agrupar plumas y especies**  
   - Reducir dimensión con PCA/UMAP sobre los embeddings.  
   - Aplicar K-Means y HDBSCAN para identificar clusters basados en patrones de coloración (incluyendo UV).

6. **Clasificar especies con redes supervisadas**  
   - Realizar fine-tuning de BEiT 4-canal para clasificación de especies de aves a partir de plumas.    
   - Calcular el coeficiente de correlación cophenética en el espacio jerárquico.  
   - Generar cladogramas a partir de los embeddings promedio por especie.

7. **Evaluar hipótesis ecológicas y evolutivas Pendiente **  
   - Contrastar patrones de coloración entre clusters no taxonómicos (mimetismo, camuflaje).  
   - Realizar pruebas estadísticas (ANOVA y tests post-hoc) para identificar diferencias significativas en reflectancia UV/RGB.  
   - Validar geoespacialmente los agrupamientos: analizar cómo la variabilidad ambiental (radiación UV, hábitat) influye en la organización de clusters.

---

## Datasets

### 1. BirdColorbase

- **Repositorio:** [BirdColorBase en GitHub](https://github.com/BirdColorBase/home)  
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

## Metodología Ajustada (CRISP-DM)

### 1. Comprensión del Negocio

- **Alcance:**  
  Desarrollar un pipeline de análisis de imágenes de plumas que combine simulación de visión tetrocromática con extracción de embeddings, clustering y análisis jerárquico. El objetivo es proporcionar a ornitólogos y equipos de conservación herramientas cuantitativas para:
  - Differenciar patrones de coloración incluidos aquellos en ultravioleta.
  - Agrupar plumas y especies según similitudes visuales.
  - Generar cladogramas basados en embeddings para apoyar estudios evolutivos y ecológicos.

- **Stakeholders:**  
  - Biólogos evolutivos  
  - Ornitólogos  
  - Equipos de conservación y biodiversidad  
  - Data scientists interesados en visión computacional aplicada a ecología  

---

### 2. Comprensión de los Datos

1. **Exploración de FeathersV1**  
   - Verificar que las imágenes de plumas estén en formato TIFF y, tras remoción de fondo, continúen con 4 canales (RGBA o RGB+canal alfa).  
   - Validar nombres de archivo y estructura de carpetas para asegurar correspondencia con metadatos taxonómicos (si existieran).  
   - Calcular estadísticas básicas de píxeles (forma, rango de valores) para cada canal (R, G, B, Alfa).

2. **Inspección del Modelo de Simulación UV**  
   - Revisar el dataset espectral usado para entrenar el modelo de regresión UV (por ejemplo, datos de LaSBiRD transformados a valores RGB/UV).  
   - Confirmar calidad de ajuste (R², MAE) en los datos espectrales antes de proceder a predecir UV para cada píxel.

---

### 3. Preparación de los Datos

1. **Segmentación Automática de Plumas**  
   - Utilizar `rembg` (que emplea U²-Net internamente) para eliminar el fondo de cada imagen de pluma y obtener una máscara alfa que aísle el sujeto.  
   - Guardar el resultado como imagen RGBA en `data/processed/feathers/segmented/` (el canal Alfa marca la pluma).

2. **Construcción del Tensor 4-Canal (RGB+UV)**  
   - **Predicción del canal UV proxy:**  
     - Cargar el modelo de regresión entrenado previamente con datos espectrales.  
     - Por cada píxel RGB de la imagen segmentada, normalizada a [0,1], predecir su valor UV.  
   - **Concatenación y resize:**  
     - Combinar los cuatro canales: `[UV_pred, R, G, B]`.  
     - Redimensionar a 224 × 224 píxeles manteniendo los cuatro canales.  
   - Almacenar las imágenes resultantes en `data/processed/feathers/4channel/`.

3. **Normalización**  
   - Para cada imagen 4-Canal (224×224), calcular media y desviación estándar de cada uno de los cuatro canales (UV, R, G, B) sobre el dataset completo.  
   - Aplicar estandarización por canal (z-score) siguiendo:  
     \[
       \text{valor\_normalizado} \;=\; \frac{\text{valor} - \mu_{\text{canal}}}{\sigma_{\text{canal}}}
     \]
   - Guardar las matrices normalizadas en memoria o disco según se requiera para las siguientes fases.

4. **División de Conjuntos (Opcional)**  
   - Si se dispone de alguna etiqueta parcial (por ejemplo, especie) para un subconjunto de plumas, reservar un 70 % de esas imágenes para extracción de embeddings y clustering inicial, y el 30 % restante para pruebas de recuperación o validación.  
   - En ausencia de etiquetas, todo el dataset se procesará para extracción de embeddings y análisis no supervisado.

---

### 4. Extracción de Embeddings

1. **BEiT Preentrenado (3 Canales)**  
   - Cargar la versión estándar de BEiT (preentrenado en ImageNet) y adaptar su pipeline de preprocesamiento para recibir entradas de 224 × 224 × 3 (RGB sin canal UV).  
   - Para cada imagen original (RGB), extraer el embedding de la capa penúltima (vector de características fijas).

2. **BEiT Adaptado (4 Canales)**  
   - Modificar la capa inicial de BEiT (`patch_embed.proj`) para aceptar 4 canales en lugar de 3.  
     - Inicializar los pesos del canal UV copiando levemente los del canal R y añadiendo ruido gaussiano pequeño (permitiendo que el modelo aprenda a usar UV en posteriores ajustes).  
   - Conservar el resto de la arquitectura idéntica para fine-tuning si se quisiera entrenar; sin embargo, si no se ajusta, usar directamente como extractor de características 4-canal.  
   - Para cada imagen normalizada 4-Canal (224 × 224 × 4), extraer el embedding.

3. **Cálculo de Diferencias entre Embeddings**  
   - Para cada par de embeddings (RGB vs. RGB+UV), calcular:  
     - **Distancia coseno** \(\; d_{\cos}(\mathbf{e}_{RGB}, \mathbf{e}_{RGB+UV}) = 1 - \frac{\mathbf{e}_{RGB} \cdot \mathbf{e}_{RGB+UV}}{\lVert \mathbf{e}_{RGB}\rVert \,\lVert \mathbf{e}_{RGB+UV}\rVert}\).  
     - **Norma de la diferencia** \(\; \lVert \mathbf{e}_{RGB} - \mathbf{e}_{RGB+UV} \rVert_{2}\).  
   - Almacenar las dos versiones de embeddings y sus distancias en CSVs:  
     - `embeddings_rgb.csv`  
     - `embeddings_rgb_uv.csv`  
     - `distancias_embeddings.csv`  

---

### 5. Clustering y Análisis No Supervisado

1. **Reducción de Dimensión**  
   - Aplicar **PCA** sobre los embeddings (tanto RGB como RGB+UV) para determinar cuántas componentes explican, por ejemplo, el 95 % de la varianza.  
   - Utilizar **UMAP** (con vecinos = 15, distancia mínima = 0.1) para proyectar los embeddings en 2D, facilitando la visualización.

2. **Clustering**  
   - **K-Means** (k = 5 a 10, según heurísticas como elbow method): agrupar puntos en el espacio reducido (PCA o UMAP). Calcular:  
     - Silhouette Score  
     - Davies-Bouldin Index  
     - Calinski-Harabasz Score  
   - **HDBSCAN** (mínimo de 10 muestras por cluster): detectar agrupaciones de densidad variable, útil si existen subgrupos más pequeños o ruido.  
   - Comparar métricas entre versiones RGB vs. RGB+UV:  
     - Evaluar si agregar UV mejora Silhouette, reduce Davies-Bouldin y/o incrementa Calinski-Harabasz.

3. **Análisis de Clusters**  
   - Para cada cluster obtenido (RGB y RGB+UV), calcular estadísticos de distancia interna (promedio coseno entre pares de embeddings dentro del mismo cluster).  
   - Visualizar los clusters en el espacio UMAP, coloreando por etiqueta (si se dispone) o por clusters HDBSCAN.  
   - Inspeccionar clusters cualitativamente: revisar ejemplos de plumas en cada clúster para interpretar agrupamientos (mimetismo, convergencia).

---

### 6. Análisis Jerárquico y Cladogramas

1. **Promedio de Embeddings por “Grupo”**  
   - Si existe metadato de “especie” o “familia” para un subconjunto, agrupar embeddings por ese campo y calcular el embedding promedio (centroide) de cada grupo.  
   - En ausencia de etiquetas fijas, se puede promediar embeddings de subclusters (p. ej., agrupaciones HDBSCAN).

2. **Distancia de Coseno entre Embeddings Promedios**  
   - Generar la matriz de distancias (coseno) entre centroides de cada grupo.  
   - Construir el linkage jerárquico (método promedio) a partir de dicha matriz.

3. **Coeficiente Cophenético**  
   - Calcular el coeficiente cophenético para los dendrogramas basados en embeddings RGB y en embeddings RGB+UV.  
   - Comparar valores:  
     - Un coeficiente más alto indica que el dendrograma refleja mejor las distancias originales.  
     - Un valor más bajo en RGB+UV —en comparación con RGB— puede sugerir reorganizaciones importantes basadas en información UV.

4. **Tanglegram**  
   - Visualizar ambos dendrogramas (RGB vs. RGB+UV) en un tanglegram, conectando los mismos grupos en los dos árboles.  
   - Observar reordenamientos de ramas que indiquen convergencias o divergencias de agrupamientos al añadir el canal UV.


---

### 7. Evaluación de Hipótesis Ecológicas ( NO DESARROLLADO)

1. **Comparación de Clusters No Taxonómicos**  
   - Identificar clusters que combinen plumas de distintas especies o familias en el agrupamiento RGB+UV pero que estén separadas en RGB.  
   - Interpretar si dichos clusters corresponden a mimetismo (colores UV similares en especies no emparentadas) o adaptaciones comunes al mismo hábitat.

2. **Pruebas Estadísticas**  
   - Para cada cluster (RGB+UV), extraer métricas de reflectancia promedio por canal (UV, R, G, B).  
   - Realizar **ANOVA** para comparar valores medios de canal UV entre clusters, verificando si existen diferencias significativas.  
   - Si el ANOVA resulta significativo, aplicar **tests post-hoc (Tukey)** para identificar pares de clusters con diferencias reales en reflectancia UV.

3. **Validación Geoespacial**  
   - Si se dispone de coordenadas de muestreo para las plumas, superponer clusters sobre un mapa.  
   - Evaluar si clusters basados en UV tienden a corresponder con regiones de alta radiación UV o hábitats similares (e.g., montañas vs. costa).  
   - Calcular correlaciones (Spearman o Pearson) entre promedios de UV en un cluster y variables ambientales (radiación UV promedio de la región, altitud).

---

## Técnicas y Herramientas

| Fase                       | Herramientas / Bibliotecas                                                                                     |
|----------------------------|------------------------------------------------------------------------------------------------------------------|
| Segmentación               | `rembg` (U²-Net internamente)                                                                                   |
| Simulación Tetrocromática  | OpenCV, NumPy, SciPy, scikit-learn (HistGradientBoostingRegressor, PolynomialFeatures, StandardScaler)           |
| Extracción de Embeddings   | PyTorch, timm (BEiT), `torchvision.transforms`                                                                    |
| Reducción de Dimensión     | scikit-learn (PCA), UMAP                                                                                          |
| Clustering                 | scikit-learn (KMeans, Silhouette Score, Davies-Bouldin, Calinski-Harabasz), HDBSCAN                              |
| Análisis Jerárquico        | SciPy (`linkage`, `cophenet`), scikit-bio (Mantel test)                                                            |
| Pruebas Estadísticas       | SciPy (`f_oneway`, `pairwise_tukeyhsd` de `statsmodels`), pandas                                                 |
| Visualización              | Matplotlib, Seaborn                                                                                                |
| Gestión de Imágenes TIFF   | tifffile, OpenCV                                                                                                  |
| Manejo de Datos y IO       | pandas, os, glob                                                                                                  |
| Orquestación de Notebooks  | Jupyter Notebook / Google Colab                                                                                   |
| Control de Versiones       | Git, GitHub                                                                                                       |
| Entorno de Desarrollo      | Python 3.9+, Conda / virtualenv                                                                                   |

---


## Resumen de Resultados

## Resumen de Resultados

[**Figura 1. Proyección UMAP**](https://github.com/Vagarh/Sistema-de-Visi-n-por-Computadora-para-Reconocimiento-de-Aves-y-Simulaci-n-Tetrocrom-tica/blob/ecd3d561c257b562e2e775cb47c48c45425ec1c7/Imagenes/UMAP.png)  
[**Figura 2. Comparación de dendrogramas**](https://github.com/Vagarh/Sistema-de-Visi-n-por-Computadora-para-Reconocimiento-de-Aves-y-Simulaci-n-Tetrocrom-tica/blob/ecd3d561c257b562e2e775cb47c48c45425ec1c7/Imagenes/comparacion%20dendrogramas.png)  

La incorporación del canal UVB modifica de forma notable la estructura jerárquica de agrupamiento. Al añadir esta dimensión, las distancias entre las muestras de plumaje cambian, generando un dendrograma distinto: la correlación cophenética pasa de **0,3501** en el espacio solo RGB a **0,2591** en el espacio RGB + UVB. Esto confirma que la jerarquía resultante ya no refleja únicamente las distancias basadas en colores visibles.

Desde el punto de vista espectral, el canal UVB aporta información clave. Algunas especies presentan reflectancia en ultravioleta que no se detecta en RGB, por lo que UVB captura características biológicamente relevantes y “ocultas” al ojo humano. Esa señal complementaria enriquece el espacio de características y permite distinguir ejemplares cuyas plumas, en RGB, podrían parecer idénticas.

Además, la inclusión de UVB produce clusters más cohesionados y mejor separados. En el análisis con K-Means y validación interna, la puntuación de **Silhouette aumenta de 0,39 a 0,42** al sumar el canal UVB, indicando que cada punto se agrupa de modo más compacto y queda mejor aislado de puntos ajenos. Simultáneamente, el índice de **Davies–Bouldin disminuye** (clusters más compactos) y el índice de **Calinski–Harabasz crece ligeramente** (mayor disparidad entre varianza intra- e intercluster).

La proyección UMAP (Figura 1) corrobora estas mejoras, mostrando que los cinco grupos resultantes en el espacio RGB + UVB están más claramente diferenciados y presentan menos solapamientos que en el espacio solo RGB. En conjunto, estos hallazgos demuestran que el canal UVB revela patrones espectrales ocultos en el plumaje de aves, validando su uso para descubrir agrupaciones que no son detectables únicamente con la información visible. Por tanto, la integración de UVB constituye una estrategia eficaz para identificar características “invisibles” en las plumas y avanzar en el estudio de su ecología y taxonomía.




```


