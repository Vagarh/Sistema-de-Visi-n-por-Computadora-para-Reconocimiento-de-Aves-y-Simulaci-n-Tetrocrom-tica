# GEMINI.MD - Análisis del Proyecto

## Estado Actual del Proyecto
El proyecto se encuentra en una fase avanzada de desarrollo funcional, con los siguientes componentes implementados:
- **Simulación Tetrocromática**: Un pipeline funcional que estima el canal UVB a partir de imágenes RGB utilizando modelos de regresión entrenados con datos espectrales de *BirdColorBase*.
- **Procesamiento de Imágenes**: Segmentación automática de plumas mediante `rembg` (U²-Net) y normalización de tensores de 4 canales.
- **Extracción de Embeddings**: Adaptación de modelos *BEiT* (Vision Transformers) para procesar entradas de 4 canales y extraer descriptores visuales profundos.
- **Análisis de Agrupamiento**: Implementación de proyecciones dimensionales (PCA/UMAP) y algoritmos de clustering (K-Means, HDBSCAN) para validar la separación de especies.
- **Interfaz de Usuario**: Una aplicación interactiva en *Streamlit* (`app.py`) que permite visualizar la simulación UV y comparar resultados de clustering.

## Objetivos
1. **Simulación Visual**: Recrear la percepción visual aviar incorporando información del espectro ultravioleta "oculta" para los humanos.
2. **Mejora del Reconocimiento**: Incrementar la precisión en la clasificación y diferenciación de especies de aves mediante la adición del canal UVB.
3. **Validación Filogenética**: Generar cladogramas basados en embeddings visuales que permitan contrastar la taxonomía tradicional con patrones de coloración evolutivos.
4. **Análisis Ecológico**: Identificar convergencias evolutivas como mimetismo o camuflaje que solo son evidentes en el espectro tetrocromático.

## Limitaciones
- **Naturaleza del Canal UV**: El canal UVB es una estimación estadística (proxy) y no una medición directa por sensor UV, lo que introduce un margen de error dependiente del modelo de regresión.
- **Sesgo de Datos**: La base de datos *FeathersV1* contiene imágenes con fondos variables y condiciones de iluminación diversas que pueden afectar la consistencia de la predicción UV.
- **Validación Ecológica**: Aún está pendiente la validación geoespacial completa y pruebas estadísticas exhaustivas (ANOVA/Tukey) para correlacionar los clusters con variables ambientales reales.
- **Complejidad Computacional**: El ajuste fino (fine-tuning) de modelos *BEiT* de 4 canales requiere recursos computacionales significativos para entrenamiento a gran escala.

---
## Hoja de Ruta (Roadmap) para Aplicación Real

Para transformar este proyecto de investigación en una aplicación profesional, se han definido las siguientes fases de trabajo:

### Fase 1: Reestructuración de Cimientos (✅ COMPLETADO)
- **Organización de Directorios**: Implementar una estructura estándar (`src/`, `notebooks/`, `data/`, `scripts/`).
- **Gestión de Dependencias**: Limpiar y optimizar `requirements.txt` y entornos virtuales.

### Fase 2: Modularización y Refactorización (✅ COMPLETADO)
- **Extracción de Lógica Core**: Mover el preprocesamiento, segmentación y predicción UV a módulos reutilizables en `src/`.
- **Arquitectura de Modelos**: Estandarizar la carga y uso de modelos BEiT y regresores UV.
- **Manejo de Errores**: Implementar logging y validaciones robustas.

### Fase 3: Experiencia de Usuario (UI/UX) Premium (✅ COMPLETADO)
- **Rediseño de Interfaz**: Aplicar principios de diseño moderno en Streamlit (animaciones, micro-interacciones).
- **Visualización Avanzada**: Gráficos interactivos y comparativas dinámicas de alta fidelidad con Plotly.
- **Feedback en Tiempo Real**: Notificaciones y barras de progreso para procesos pesados.

### Fase 4: Preparación para Producción (⏳ PENDIENTE)
- **Optimización de Rendimiento**: Caché eficiente y procesamiento por lotes.
- **Documentación Técnica**: Docstrings y guías de contribución claras.
- **Empaquetado**: Facilitar la instalación y despliegue en la nube o local.

---
*Este documento fue generado automáticamente por Antigravity tras el análisis del repositorio en Marzo 2026.*
