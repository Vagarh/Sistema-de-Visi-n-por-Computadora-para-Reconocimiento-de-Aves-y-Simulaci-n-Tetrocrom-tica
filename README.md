<div align="center">

# 🦅 AvianVision: Sistema de Reconocimiento y Simulación Tetrocromática

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-PyTorch%20%7C%20Scikit--Learn-FF6F00?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Una plataforma avanzada de visión computacional y ciencia de datos que recrea la percepción visual de las aves mediante la predicción de un cuarto canal espectral (Ultravioleta - UVB).**

![Visión Tetrocromática](https://github.com/Vagarh/Sistema-de-Visi-n-por-Computadora-para-Reconocimiento-de-Aves-y-Simulaci-n-Tetrocrom-tica/blob/main/Imagenes/VISIOTETRACROMICA.jpg_large)

</div>

---

## 📖 Descripción del Proyecto

Las aves perciben el mundo de una forma fundamentalmente distinta a los humanos. Su sistema visual **tetrocromático** cuenta con cuatro tipos de conos fotorreceptores, incluyendo uno especializado en el espectro ultravioleta (UV), invisible para nosotros.

**AvianVision** es un proyecto pionero de Machine Learning aplicado a la ornitología y conservación. Su propósito principal es simular esta rica experiencia visual prediciendo un canal UVB profundo a partir de imágenes RGB de plumaje de aves. Al revelar esta dimensión "oculta", el sistema potencia las capacidades de:

1. **Reconocimiento y Diferenciación Taxonómica Avanzada**
2. **Análisis Ecológico y Descubrimiento de Relaciones Evolutivas**
3. **Agrupación Filogenética Visual Asistida por IA Transformers (BEiT)**

El flujo de trabajo (basado en CRISP-DM) unifica procesamiento avanzado de imágenes, redes neuronales Vision Transformers (BEiT para 4 canales) y aprendizaje no supervisado.

---

## ✨ Características Principales

*   **🪄 Predicción Tetrocromática**: Transforma imágenes RGB convencionales en tensores de 4 canales (`UV, R, G, B`), estimando la reflectancia ultravioleta mediante modelos ensamblados rigurosamente evaluados sobre datos espectroscópicos *(BirdColorBase)*.
*   **🧠 Redes Vision Transformers Adaptativas**: Integra modelos pre-entrenados **BEiT** con canales de entrada modificados para asimilar información cromática tetradimensional y extraer embeddings de altísima fidelidad.
*   **📊 Clustering Evolutivo Interactivo**: Agrupa especies sin supervisión empleando HDBSCAN, K-Means e innovadoras reducciones topológicas en subespacios de embeddings multidimensionales.
*   **🛰 Interfaz Premium UX/UI**: Presenta una aplicación moderna construida con Streamlit y Plotly, ofreciendo visualizaciones UMAP interactivas, mapas térmicos de correlación espectral y dendrogramas para comparativa visual fluida.
*   **🧩 Arquitectura Modular Robusta**: Código altamente escalable, diseñado como paquete fuente (`src/`) facilitando la incorporación futura a pipelines en producción.

---

## 🧬 Justificación Biológica

Las plumas de las aves ocultan un universo comunicativo clave en la banda ultravioleta, que emplean para:

*   Reconocimiento certero de individuos y evaluación del sexo/aptitud.
*   Señalización críptica o cortejo.
*   Mimetismo o defensa antidepredatoria.

La hipótesis fundamental de *AvianVision* estipula que **la adición de nuestro canal ultravioleta estimado (tensor RGB+UVB) incrementa mediblemente la separabilidad y cohesión de las clases filogenéticas** en espacios latentes por encima del espacio RGB tradicional (validado mediante *Silhouette e índices de Davies-Bouldin*).

---

## 🚀 Inicio Rápido

### Requisitos Previos
* Python 3.9 o superior.
* Se recomienda encarecidamente una GPU para la extracción rápida de embeddings Transformer.

### Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/Vagarh/Sistema-de-Visi-n-por-Computadora-para-Reconocimiento-de-Aves-y-Simulaci-n-Tetrocrom-tica.git
   cd Sistema-de-Visi-n-por-Computadora-para-Reconocimiento-de-Aves-y-Simulaci-n-Tetrocrom-tica
   ```

2. Configura tu entorno virtual e instala dependencias:
   ```bash
   python -m venv venv
   # En Windows: venv\Scripts\activate
   # En Linux/macOS: source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Lanza la interfaz de usuario de Streamlit:
   ```bash
   streamlit run app.py
   # Alternativa en Windows: ./scripts/run_app.bat
   ```

---

## 🏗️ Arquitectura del Proyecto

El código base fue recientemente reestructurado bajo un estándar profesional:

```text
📁 AvianVision
├── 📄 app.py                     # Interfaz de Usuario Principal (Streamlit)
├── 📁 src/                       # Núcleo Modular
│   ├── 📁 models/                # Lógica AI: Clustering, extracción BEiT y regresión UVB
│   ├── 📁 processing/            # Ingeniería de features y procesamiento U-Net
│   └── 📁 utils/                 # Visualizaciones avanzadas e índices espectrales
├── 📁 notebooks/                 # EDA originario, pruebas espectrales e inv. de hipótesis
├── 📁 scripts/                   # Utilidades CLI de despliegue y validación de entornos
├── 📁 data/                      # Volúmenes de imágenes crudas y procesadas (FeathersV1)
└── 📄 GEMINI.MD                  # Documento estratégico de Roadmap y limitaciones
```

---

## 📈 Resultados y Validación

Los análisis demuestran sólidamente beneficios tangibles en el espacio tetrocromático:

*   **Identificación Topológica Más Clara:** La proyección UMAP evidencia una aglomeración intra-cluster mucho más densa frente a RGB.
*   **Reestructuración Jerárquica:** El coeficiente Cofenético de dendrogramas decae de 0.35 para RGB a 0.25 en RGB+UVB, confirmando el resurgimiento de características ortogonales no deducibles con óptica humana.
*   **Métricas Internas de Agrupación:** Se documenta un incremento sistemático del *Silhouette Score* (+10% relativo) y *Calinski-Harabasz*, garantizando un mapeo más intrínsecamente coherente con la divergencia visual de las aves.

<p align="center">
  <img src="https://github.com/Vagarh/Sistema-de-Visi-n-por-Computadora-para-Reconocimiento-de-Aves-y-Simulaci-n-Tetrocrom-tica/blob/ecd3d561c257b562e2e775cb47c48c45425ec1c7/Imagenes/UMAP.png" alt="Proyección UMAP" width="48%">
  <img src="https://github.com/Vagarh/Sistema-de-Visi-n-por-Computadora-para-Reconocimiento-de-Aves-y-Simulaci-n-Tetrocrom-tica/blob/ecd3d561c257b562e2e775cb47c48c45425ec1c7/Imagenes/comparacion%20dendrogramas.png" alt="Comparativa de Dendrogramas" width="48%">
</p>

---

## 🗺️ Roadmap de Desarrollo

Este repositorio pasó recientemente de ser un monolito de investigación a una aplicación lista para su uso:

*   **[✅] Fase 1: Reestructuración de Cimientos** (Pipelines empaquetados como `src/`).
*   **[✅] Fase 2: Modularización** (Separación de Transformers, Modelos Sklearn y Segmentadores).
*   **[✅] Fase 3: Experiencia UI/UX Premium** (Reconstrucción total con gráficas dinámicas Plotly).
*   **[⏳] Fase 4: Preparación Prod (Scale-Up)** (Batched Inference, Docstrings exhaustivos, Caché persistente, Dockerización).

Consulta [`GEMINI.MD`](./GEMINI.MD) para la visión estratégica detallada del proyecto.

---

## 🤝 Contribuciones

Agradecemos sinceramente contribuciones tanto en el espectro biológico / ornitológico como ingenieril. Por favor, abre un *Issue* describiendo la mejora propuesta antes de enviar un *Pull Request*.

## 👨‍💻 Acerca del Autor

Desarrollado y mantenido por **Juan Felipe Cardona Arango**, enfocando de forma transdisciplinaria Ciencias de Datos, Computer Vision y Machine Learning hacia la biodiversidad y bioinformática.

---
<p align="center">Construido con propósitos de investigación ornitológica. Universidad y Año de Defensa - 2025.</p>
