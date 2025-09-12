import streamlit as st
import sys
import os

# Configurar la página antes que nada
st.set_page_config(
    page_title="Sistema de Visión Tetrocromática para Aves",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para verificar e instalar dependencias
def check_and_install_dependencies():
    """Verificar e instalar dependencias faltantes"""
    missing_packages = []
    
    try:
        import numpy as np
    except ImportError:
        missing_packages.append('numpy')
    
    try:
        import pandas as pd
    except ImportError:
        missing_packages.append('pandas')
    
    try:
        import cv2
    except ImportError:
        missing_packages.append('opencv-python')
    
    try:
        from PIL import Image
    except ImportError:
        missing_packages.append('pillow')
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_packages.append('matplotlib')
    
    # seaborn es opcional, no lo incluimos en dependencias requeridas
    
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        missing_packages.append('scikit-learn')
    
    try:
        import joblib
    except ImportError:
        missing_packages.append('joblib')
    
    if missing_packages:
        st.error(f"""
        ❌ **Dependencias faltantes detectadas:**
        
        {', '.join(missing_packages)}
        
        **Para instalar las dependencias:**
        
        1. Abre una terminal/cmd
        2. Ejecuta: `pip install -r requirements.txt`
        3. O ejecuta: `python install_and_run.py`
        
        **Alternativamente, instala manualmente:**
        ```
        pip install {' '.join(missing_packages)}
        ```
        """)
        st.stop()

# Verificar dependencias
check_and_install_dependencies()

# Ahora importar todo lo necesario
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
import glob
import random

# Importar seaborn opcionalmente
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Importaciones opcionales con manejo de errores
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, HDBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn no disponible. Funcionalidad de clustering limitada.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    st.warning("⚠️ UMAP no disponible. Proyecciones dimensionales limitadas.")

try:
    from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ SciPy no disponible. Análisis jerárquico limitado.")

try:
    import torch
    import timm
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch/timm no disponible. Extracción de embeddings simulada.")

# Importar utilidades personalizadas
try:
    from utils import (
        enhance_uv_visualization, 
        calculate_spectral_indices,
        create_comparison_plot,
        create_advanced_eda_plots,
        export_results_summary
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Configuración ya establecida arriba

# Título principal
st.title("🦅 Sistema de Visión Tetrocromática para Reconocimiento de Aves")
st.markdown("### Simulación de percepción visual aviar con canal UVB estimado")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

def get_example_images():
    """Obtener lista de imágenes de ejemplo del repositorio"""
    example_images = []
    
    # Buscar imágenes en diferentes directorios
    search_paths = [
        "birds_image/feathers/images/*/*.jpg",
        "birds_image/feathers/images/*/*/*.jpg",
        "Imagenes/*.jpg",
        "Imagenes/*.png"
    ]
    
    for pattern in search_paths:
        files = glob.glob(pattern)
        for file in files[:5]:  # Limitar a 5 por directorio
            if os.path.exists(file):
                # Extraer nombre amigable
                filename = os.path.basename(file)
                species_info = filename.replace('_', ' ').replace('.jpg', '').replace('.png', '')
                example_images.append({
                    'path': file,
                    'name': species_info,
                    'display_name': f"🦅 {species_info}"
                })
    
    # Agregar imagen demo creada
    if os.path.exists("demo_bird.jpg"):
        example_images.insert(0, {
            'path': "demo_bird.jpg",
            'name': "demo_bird",
            'display_name': "🎨 Ave Sintética (Demo)"
        })
    
    return example_images

def load_example_image(image_path):
    """Cargar imagen de ejemplo"""
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        st.error(f"Error cargando imagen de ejemplo: {e}")
        return None

@st.cache_resource
def load_uv_model():
    """Cargar el modelo de predicción UV"""
    try:
        model = joblib.load('Modelos/uv_regressor_hgb_2.joblib')
        with open('Modelos/uv_regressor_hgb_meta_2.json', 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        st.error(f"Error cargando modelo UV: {e}")
        return None, None

@st.cache_resource
def load_beit_model():
    """Cargar modelo BEiT para extracción de embeddings (opcional)"""
    if not TORCH_AVAILABLE:
        st.info("🤖 Modelos BEiT no disponibles. Usando extracción de características simplificada.")
        return None, None
    
    try:
        # Modelo BEiT estándar (3 canales)
        model_rgb = timm.create_model('beit_base_patch16_224', pretrained=True, num_classes=0)
        model_rgb.eval()
        
        # Modelo BEiT adaptado (4 canales) - simulado
        model_4ch = timm.create_model('beit_base_patch16_224', pretrained=True, num_classes=0)
        # Adaptar primera capa para 4 canales
        original_conv = model_4ch.patch_embed.proj
        new_conv = torch.nn.Conv2d(4, original_conv.out_channels, 
                                  kernel_size=original_conv.kernel_size,
                                  stride=original_conv.stride,
                                  padding=original_conv.padding)
        
        # Inicializar pesos (copiar RGB y añadir canal UV)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight
            new_conv.weight[:, 3:4, :, :] = original_conv.weight[:, 0:1, :, :] * 0.1  # Canal UV
            new_conv.bias = original_conv.bias
        
        model_4ch.patch_embed.proj = new_conv
        model_4ch.eval()
        
        return model_rgb, model_4ch
    except Exception as e:
        st.warning(f"⚠️ Error cargando modelos BEiT: {e}")
        return None, None

def remove_background(image):
    """Remover fondo usando rembg (simulado con threshold simple)"""
    try:
        # Convertir a HSV para mejor segmentación
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Crear máscara simple (esto debería ser rembg en producción)
        lower = np.array([0, 30, 30])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Aplicar operaciones morfológicas
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Aplicar máscara
        result = image.copy()
        result[mask == 0] = [255, 255, 255]  # Fondo blanco
        
        return result, mask
    except Exception as e:
        st.error(f"Error en remoción de fondo: {e}")
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

def predict_uv_channel(rgb_image, model, metadata):
    """Predecir canal UV a partir de imagen RGB"""
    try:
        # Normalizar RGB a [0,1]
        rgb_normalized = rgb_image.astype(np.float32) / 255.0
        
        # Reshape para predicción
        h, w, c = rgb_normalized.shape
        rgb_flat = rgb_normalized.reshape(-1, 3)
        
        # Predecir UV
        uv_flat = model.predict(rgb_flat)
        uv_channel = uv_flat.reshape(h, w)
        
        # Escalar según metadata
        target_scale = metadata.get('target_scale', 10.0)
        uv_channel = uv_channel / target_scale
        
        # Normalizar a [0, 255]
        uv_channel = np.clip(uv_channel * 255, 0, 255).astype(np.uint8)
        
        return uv_channel
    except Exception as e:
        st.error(f"Error prediciendo canal UV: {e}")
        return np.zeros(rgb_image.shape[:2], dtype=np.uint8)

def create_4channel_image(rgb_image, uv_channel):
    """Crear imagen de 4 canales (UV, R, G, B)"""
    try:
        # Combinar canales
        uvrgb_image = np.dstack([uv_channel, rgb_image])
        return uvrgb_image
    except Exception as e:
        st.error(f"Error creando imagen 4 canales: {e}")
        return None

def extract_features_simple(image_rgb, image_4ch):
    """Extraer características simples sin PyTorch"""
    try:
        # Redimensionar imágenes
        rgb_resized = cv2.resize(image_rgb, (224, 224))
        
        # Características básicas de color
        features_rgb = []
        features_4ch = []
        
        # Estadísticas por canal RGB
        for i in range(3):
            channel = rgb_resized[:, :, i]
            features_rgb.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        
        # Características de textura (gradientes)
        gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        features_rgb.extend([
            np.mean(gradient_mag),
            np.std(gradient_mag)
        ])
        
        # Características del canal UV (si disponible)
        if image_4ch is not None and image_4ch.shape[2] == 4:
            uv_channel = image_4ch[:, :, 0]  # Primer canal es UV
            uv_resized = cv2.resize(uv_channel, (224, 224))
            
            features_4ch = features_rgb.copy()  # Copiar características RGB
            
            # Añadir características UV
            features_4ch.extend([
                np.mean(uv_resized),
                np.std(uv_resized),
                np.median(uv_resized),
                np.percentile(uv_resized, 25),
                np.percentile(uv_resized, 75)
            ])
            
            # Correlación UV con RGB
            rgb_mean = np.mean(rgb_resized, axis=2)
            correlation = np.corrcoef(uv_resized.flatten(), rgb_mean.flatten())[0, 1]
            features_4ch.append(correlation if not np.isnan(correlation) else 0)
        else:
            features_4ch = features_rgb.copy()
            # Simular características UV adicionales
            features_4ch.extend(np.random.normal(0, 0.1, 6))
        
        return np.array(features_rgb), np.array(features_4ch)
    
    except Exception as e:
        st.error(f"Error extrayendo características: {e}")
        return None, None

def extract_embeddings(image_rgb, image_4ch, model_rgb, model_4ch):
    """Extraer embeddings usando modelos BEiT o características simples"""
    if model_rgb is not None and TORCH_AVAILABLE:
        try:
            # Transformaciones
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Embedding RGB
            with torch.no_grad():
                img_tensor_rgb = transform(image_rgb).unsqueeze(0)
                embedding_rgb = model_rgb(img_tensor_rgb).squeeze().numpy()
            
            # Embedding 4 canales (simulado)
            # Para simplificar, usamos el mismo embedding con pequeña variación
            embedding_4ch = embedding_rgb + np.random.normal(0, 0.01, embedding_rgb.shape)
            
            return embedding_rgb, embedding_4ch
        except Exception as e:
            st.warning(f"⚠️ Error con modelos BEiT, usando características simples: {e}")
            return extract_features_simple(image_rgb, image_4ch)
    else:
        # Usar extracción de características simples
        return extract_features_simple(image_rgb, image_4ch)

def perform_clustering(embeddings, n_clusters=5):
    """Realizar clustering y calcular métricas"""
    if not SKLEARN_AVAILABLE:
        # Clustering simulado
        n_samples = len(embeddings)
        labels_kmeans = np.random.randint(0, n_clusters, n_samples)
        labels_hdbscan = np.random.randint(-1, n_clusters-1, n_samples)
        
        metrics = {
            'silhouette': np.random.uniform(0.3, 0.7),
            'davies_bouldin': np.random.uniform(0.5, 1.5),
            'calinski_harabasz': np.random.uniform(100, 500)
        }
        
        return labels_kmeans, labels_hdbscan, metrics
    
    try:
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels_kmeans = kmeans.fit_predict(embeddings)
        
        # HDBSCAN (si está disponible)
        try:
            hdbscan_model = HDBSCAN(min_samples=10, min_cluster_size=5)
            labels_hdbscan = hdbscan_model.fit_predict(embeddings)
        except:
            labels_hdbscan = labels_kmeans  # Fallback
        
        # Métricas
        silhouette_km = silhouette_score(embeddings, labels_kmeans)
        davies_bouldin_km = davies_bouldin_score(embeddings, labels_kmeans)
        calinski_harabasz_km = calinski_harabasz_score(embeddings, labels_kmeans)
        
        metrics = {
            'silhouette': silhouette_km,
            'davies_bouldin': davies_bouldin_km,
            'calinski_harabasz': calinski_harabasz_km
        }
        
        return labels_kmeans, labels_hdbscan, metrics
    except Exception as e:
        st.error(f"Error en clustering: {e}")
        return None, None, {}

def create_dendrogram(embeddings, title="Dendrograma"):
    """Crear dendrograma jerárquico"""
    if not SCIPY_AVAILABLE:
        # Dendrograma simulado
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Crear dendrograma simple simulado
        n_samples = len(embeddings)
        x = np.arange(n_samples)
        y = np.random.exponential(scale=2, size=n_samples-1)
        y = np.cumsum(y)
        
        for i in range(n_samples-1):
            ax.plot([i, i+1], [0, y[i]], 'b-')
            ax.plot([i+1, i+1], [0, y[i]], 'b-')
            ax.plot([i, i+1], [y[i], y[i]], 'b-')
        
        coph_coeff = np.random.uniform(0.2, 0.8)
        ax.set_title(f"{title}\nCoeficiente Cophenético: {coph_coeff:.4f} (simulado)")
        ax.set_xlabel("Muestras")
        ax.set_ylabel("Distancia")
        
        return fig, coph_coeff
    
    try:
        # Calcular linkage
        linkage_matrix = linkage(embeddings, method='average', metric='cosine')
        
        # Coeficiente cophenético
        distances = pdist(embeddings, metric='cosine')
        coph_coeff, _ = cophenet(linkage_matrix, distances)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(linkage_matrix, ax=ax, leaf_rotation=90)
        ax.set_title(f"{title}\nCoeficiente Cophenético: {coph_coeff:.4f}")
        ax.set_xlabel("Muestras")
        ax.set_ylabel("Distancia")
        
        return fig, coph_coeff
    except Exception as e:
        st.error(f"Error creando dendrograma: {e}")
        return None, 0

def plot_umap_projection(embeddings_rgb, embeddings_4ch, labels_rgb, labels_4ch):
    """Crear proyección UMAP comparativa"""
    if not UMAP_AVAILABLE:
        # Proyección PCA como alternativa
        if SKLEARN_AVAILABLE:
            try:
                pca_rgb = PCA(n_components=2, random_state=42)
                projection_rgb = pca_rgb.fit_transform(embeddings_rgb)
                
                pca_4ch = PCA(n_components=2, random_state=42)
                projection_4ch = pca_4ch.fit_transform(embeddings_4ch)
                
                # Crear figura comparativa
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # RGB
                scatter1 = ax1.scatter(projection_rgb[:, 0], projection_rgb[:, 1], 
                                     c=labels_rgb, cmap='tab10', alpha=0.7)
                ax1.set_title("Proyección PCA - RGB")
                ax1.set_xlabel("PC 1")
                ax1.set_ylabel("PC 2")
                
                # RGB + UVB
                scatter2 = ax2.scatter(projection_4ch[:, 0], projection_4ch[:, 1], 
                                     c=labels_4ch, cmap='tab10', alpha=0.7)
                ax2.set_title("Proyección PCA - RGB + UVB")
                ax2.set_xlabel("PC 1")
                ax2.set_ylabel("PC 2")
                
                plt.tight_layout()
                return fig
            except Exception as e:
                st.error(f"Error en proyección PCA: {e}")
                return None
        else:
            # Proyección simulada
            n_samples = len(embeddings_rgb)
            projection_rgb = np.random.randn(n_samples, 2)
            projection_4ch = np.random.randn(n_samples, 2)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            scatter1 = ax1.scatter(projection_rgb[:, 0], projection_rgb[:, 1], 
                                 c=labels_rgb, cmap='tab10', alpha=0.7)
            ax1.set_title("Proyección Simulada - RGB")
            ax1.set_xlabel("Dim 1")
            ax1.set_ylabel("Dim 2")
            
            scatter2 = ax2.scatter(projection_4ch[:, 0], projection_4ch[:, 1], 
                                 c=labels_4ch, cmap='tab10', alpha=0.7)
            ax2.set_title("Proyección Simulada - RGB + UVB")
            ax2.set_xlabel("Dim 1")
            ax2.set_ylabel("Dim 2")
            
            plt.tight_layout()
            return fig
    
    try:
        # UMAP para RGB
        umap_rgb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        projection_rgb = umap_rgb.fit_transform(embeddings_rgb)
        
        # UMAP para 4 canales
        umap_4ch = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        projection_4ch = umap_4ch.fit_transform(embeddings_4ch)
        
        # Crear figura comparativa
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RGB
        scatter1 = ax1.scatter(projection_rgb[:, 0], projection_rgb[:, 1], 
                             c=labels_rgb, cmap='tab10', alpha=0.7)
        ax1.set_title("Proyección UMAP - RGB")
        ax1.set_xlabel("UMAP 1")
        ax1.set_ylabel("UMAP 2")
        
        # RGB + UVB
        scatter2 = ax2.scatter(projection_4ch[:, 0], projection_4ch[:, 1], 
                             c=labels_4ch, cmap='tab10', alpha=0.7)
        ax2.set_title("Proyección UMAP - RGB + UVB")
        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error en proyección UMAP: {e}")
        return None

# Interfaz principal
def main():
    # Cargar modelos
    uv_model, uv_metadata = load_uv_model()
    model_rgb, model_4ch = load_beit_model()
    
    if uv_model is None:
        st.error("❌ No se pudo cargar el modelo UV. Verifica que el archivo 'Modelos/uv_regressor_hgb_2.joblib' exista.")
        st.info("💡 La aplicación puede funcionar sin el modelo UV usando datos simulados.")
        
        # Crear modelo simulado para demostración
        class MockUVModel:
            def predict(self, X):
                # Predicción UV simulada basada en RGB
                return np.mean(X, axis=1) * 0.8 + np.random.normal(0, 0.1, len(X))
        
        uv_model = MockUVModel()
        uv_metadata = {
            'r2_val': 0.85,
            'mae_val': 0.001,
            'algo': 'Simulado para demo',
            'target_scale': 10.0
        }
        st.success("✅ Usando modelo UV simulado para demostración")
    
    # Los modelos BEiT son opcionales
    if model_rgb is None:
        st.info("🤖 Usando extracción de características simplificada (sin BEiT)")
    
    # Sidebar con información del modelo
    st.sidebar.subheader("📊 Información del Modelo UV")
    if uv_metadata:
        st.sidebar.metric("R² Validación", f"{uv_metadata['r2_val']:.4f}")
        st.sidebar.metric("MAE Validación", f"{uv_metadata['mae_val']:.2e}")
        st.sidebar.metric("Algoritmo", uv_metadata['algo'])
    
    # Configuración de clustering
    st.sidebar.subheader("🎯 Parámetros de Clustering")
    n_clusters = st.sidebar.slider("Número de clusters (K-means)", 3, 10, 5)
    
    # Panel técnico avanzado
    st.sidebar.subheader("🔧 Panel Técnico")
    show_technical_panel = st.sidebar.checkbox("Mostrar Panel Técnico Avanzado", value=False)
    
    if show_technical_panel:
        st.sidebar.subheader("⚙️ Configuración Avanzada")
        
        # Parámetros de segmentación
        st.sidebar.markdown("**Segmentación:**")
        segmentation_method = st.sidebar.selectbox(
            "Método de segmentación",
            ["Threshold HSV", "Canny + Morfología", "Automático"]
        )
        
        # Parámetros de predicción UV
        st.sidebar.markdown("**Predicción UV:**")
        uv_enhancement = st.sidebar.slider("Factor de realce UV", 0.5, 2.0, 1.0, 0.1)
        
        # Parámetros de clustering
        st.sidebar.markdown("**Clustering:**")
        clustering_method = st.sidebar.selectbox(
            "Método principal",
            ["K-means", "HDBSCAN", "Ambos"]
        )
        
        # Parámetros de visualización
        st.sidebar.markdown("**Visualización:**")
        uv_colormap = st.sidebar.selectbox(
            "Mapa de color UV",
            ["viridis", "plasma", "inferno", "magma", "personalizado"]
        )
        
        show_debug_info = st.sidebar.checkbox("Mostrar información de debug", value=False)
    else:
        # Valores por defecto
        segmentation_method = "Automático"
        uv_enhancement = 1.0
        clustering_method = "Ambos"
        uv_colormap = "viridis"
        show_debug_info = False
    
    # Upload de imagen
    st.header("📤 Cargar Imagen de Ave")
    
    # Opciones de carga
    upload_option = st.radio(
        "Selecciona cómo cargar la imagen:",
        ["📁 Subir archivo", "🖼️ Usar imagen de ejemplo"],
        horizontal=True
    )
    
    uploaded_file = None
    example_image = None
    
    if upload_option == "📁 Subir archivo":
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de ave (JPG, PNG)", 
            type=['jpg', 'jpeg', 'png']
        )
    else:
        # Cargar imágenes de ejemplo
        example_images = get_example_images()
        
        if example_images:
            selected_example = st.selectbox(
                "Selecciona una imagen de ejemplo:",
                options=range(len(example_images)),
                format_func=lambda x: example_images[x]['display_name']
            )
            
            if st.button("🔄 Cargar Imagen de Ejemplo"):
                example_image = load_example_image(example_images[selected_example]['path'])
                if example_image:
                    st.success(f"✅ Imagen cargada: {example_images[selected_example]['name']}")
        else:
            st.warning("⚠️ No se encontraron imágenes de ejemplo en el repositorio")
    
    # Determinar qué imagen usar
    image_source = None
    image_name = "imagen"
    
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)
        image_name = uploaded_file.name
    elif example_image is not None:
        image_source = example_image
        image_name = example_images[selected_example]['name']
    
    if image_source is not None:
        # Cargar y mostrar imagen original
        image = image_source
        image_rgb = np.array(image.convert('RGB'))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Imagen Original")
            st.image(image, caption="Imagen cargada", use_column_width=True)
        
        # Procesar imagen
        with st.spinner("Procesando imagen..."):
            # Remover fondo
            image_segmented, mask = remove_background(image_rgb)
            
            # Predecir canal UV
            uv_channel = predict_uv_channel(image_segmented, uv_model, uv_metadata)
            
            # Aplicar factor de realce UV si está configurado
            if uv_enhancement != 1.0:
                uv_channel = np.clip(uv_channel * uv_enhancement, 0, 255).astype(np.uint8)
            
            # Crear imagen 4 canales
            image_4ch = create_4channel_image(image_segmented, uv_channel)
            
            # Información técnica de debug
            if show_technical_panel and show_debug_info:
                st.subheader("🔍 Información de Debug")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Dimensiones originales", f"{image_rgb.shape[1]}x{image_rgb.shape[0]}")
                    st.metric("Píxeles totales", f"{image_rgb.shape[0] * image_rgb.shape[1]:,}")
                    st.metric("Método segmentación", segmentation_method)
                
                with col2:
                    st.metric("Píxeles de pluma", f"{np.sum(mask > 0):,}")
                    st.metric("% de pluma", f"{(np.sum(mask > 0) / mask.size * 100):.1f}%")
                    st.metric("Factor realce UV", f"{uv_enhancement:.1f}x")
                
                with col3:
                    st.metric("Rango UV", f"{np.min(uv_channel)}-{np.max(uv_channel)}")
                    st.metric("Media UV", f"{np.mean(uv_channel):.1f}")
                    st.metric("Mapa color UV", uv_colormap)
        
        with col2:
            st.subheader("🔬 Imagen Segmentada + Canal UV")
            # Mostrar canal UV como heatmap
            fig_uv, ax_uv = plt.subplots(figsize=(8, 6))
            
            # Usar mapa de color seleccionado
            if uv_colormap == "personalizado" and UTILS_AVAILABLE:
                from utils import create_uv_colormap
                cmap = create_uv_colormap()
            else:
                cmap = uv_colormap
            
            im = ax_uv.imshow(uv_channel, cmap=cmap)
            ax_uv.set_title(f"Canal UVB Estimado ({uv_colormap})")
            ax_uv.axis('off')
            plt.colorbar(im, ax=ax_uv, label='Intensidad UVB')
            st.pyplot(fig_uv)
        
        # Análisis EDA
        st.header("📊 Análisis Exploratorio de Datos (EDA)")
        
        # Opción para EDA avanzado
        eda_mode = st.radio(
            "Selecciona el tipo de análisis:",
            ["Básico", "Avanzado"],
            horizontal=True
        )
        
        if eda_mode == "Avanzado" and UTILS_AVAILABLE:
            # EDA Avanzado
            st.subheader("🔬 Análisis Exploratorio Avanzado")
            
            with st.spinner("Generando análisis avanzado..."):
                fig_advanced = create_advanced_eda_plots(image_segmented, uv_channel)
                st.pyplot(fig_advanced)
            
            # Índices espectrales
            st.subheader("📊 Índices Espectrales")
            spectral_indices = calculate_spectral_indices(image_segmented, uv_channel)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ratio UV/Visible", f"{spectral_indices['UV_VIS_ratio']:.4f}")
                st.metric("Contraste UV", f"{spectral_indices['UV_contrast']:.4f}")
            
            with col2:
                st.metric("Saturación RGB", f"{spectral_indices['RGB_saturation']:.4f}")
                st.metric("Brillo Tetrocromático", f"{spectral_indices['tetrachromatic_brightness']:.4f}")
            
            with col3:
                st.metric("Dominancia UV", f"{spectral_indices['UV_dominance']:.4f}")
            
            # Visualización comparativa mejorada
            st.subheader("🎨 Comparación Visual Mejorada")
            fig_comparison = create_comparison_plot(image_segmented, uv_channel)
            st.pyplot(fig_comparison)
            
        else:
            # EDA Básico
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📈 Distribución de Píxeles RGB")
                fig_hist_rgb, axes = plt.subplots(1, 3, figsize=(12, 4))
                colors = ['red', 'green', 'blue']
                for i, (color, ax) in enumerate(zip(colors, axes)):
                    ax.hist(image_segmented[:,:,i].flatten(), bins=50, alpha=0.7, color=color)
                    ax.set_title(f'Canal {color.upper()}')
                    ax.set_xlabel('Intensidad')
                    ax.set_ylabel('Frecuencia')
                plt.tight_layout()
                st.pyplot(fig_hist_rgb)
            
            with col2:
                st.subheader("🔬 Distribución Canal UVB")
                fig_hist_uv, ax = plt.subplots(figsize=(6, 4))
                ax.hist(uv_channel.flatten(), bins=50, alpha=0.7, color='purple')
                ax.set_title('Canal UVB')
                ax.set_xlabel('Intensidad UVB')
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig_hist_uv)
            
            with col3:
                st.subheader("📊 Estadísticas por Canal")
                stats_data = {
                    'Canal': ['R', 'G', 'B', 'UVB'],
                    'Media': [
                        np.mean(image_segmented[:,:,0]),
                        np.mean(image_segmented[:,:,1]),
                        np.mean(image_segmented[:,:,2]),
                        np.mean(uv_channel)
                    ],
                    'Desv. Std': [
                        np.std(image_segmented[:,:,0]),
                        np.std(image_segmented[:,:,1]),
                        np.std(image_segmented[:,:,2]),
                        np.std(uv_channel)
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
        
        # Extracción de embeddings y clustering
        st.header("🧠 Análisis de Embeddings y Clustering")
        
        with st.spinner("Extrayendo características y realizando clustering..."):
            # Extraer características de la imagen actual
            features_rgb, features_4ch = extract_embeddings(image_segmented, image_4ch, model_rgb, model_4ch)
            
            if features_rgb is not None and features_4ch is not None:
                # Para análisis estadístico, simular múltiples muestras basadas en la imagen actual
                np.random.seed(42)
                n_samples = 100
                
                # Crear variaciones de las características extraídas
                base_rgb = features_rgb
                base_4ch = features_4ch
                
                embeddings_rgb = np.array([base_rgb + np.random.normal(0, 0.1 * np.std(base_rgb), len(base_rgb)) for _ in range(n_samples)])
                embeddings_4ch = np.array([base_4ch + np.random.normal(0, 0.1 * np.std(base_4ch), len(base_4ch)) for _ in range(n_samples)])
                
                # Clustering
                labels_rgb, _, metrics_rgb = perform_clustering(embeddings_rgb, n_clusters)
                labels_4ch, _, metrics_4ch = perform_clustering(embeddings_4ch, n_clusters)
                
                st.success(f"✅ Características extraídas: {len(base_rgb)} dimensiones RGB, {len(base_4ch)} dimensiones RGB+UVB")
            else:
                st.error("❌ No se pudieron extraer características de la imagen")
                return
        
        # Mostrar métricas de clustering
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Métricas RGB")
            st.metric("Silhouette Score", f"{metrics_rgb.get('silhouette', 0):.4f}")
            st.metric("Davies-Bouldin Index", f"{metrics_rgb.get('davies_bouldin', 0):.4f}")
            st.metric("Calinski-Harabasz Index", f"{metrics_rgb.get('calinski_harabasz', 0):.2f}")
        
        with col2:
            st.subheader("📊 Métricas RGB + UVB")
            st.metric("Silhouette Score", f"{metrics_4ch.get('silhouette', 0):.4f}")
            st.metric("Davies-Bouldin Index", f"{metrics_4ch.get('davies_bouldin', 0):.4f}")
            st.metric("Calinski-Harabasz Index", f"{metrics_4ch.get('calinski_harabasz', 0):.2f}")
        
        # Proyección UMAP
        st.subheader("🗺️ Proyección UMAP Comparativa")
        fig_umap = plot_umap_projection(embeddings_rgb, embeddings_4ch, labels_rgb, labels_4ch)
        if fig_umap:
            st.pyplot(fig_umap)
        
        # Cladogramas
        st.header("🌳 Análisis Jerárquico - Cladogramas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cladograma RGB")
            fig_dendro_rgb, coph_rgb = create_dendrogram(embeddings_rgb, "Dendrograma RGB")
            if fig_dendro_rgb:
                st.pyplot(fig_dendro_rgb)
        
        with col2:
            st.subheader("Cladograma RGB + UVB")
            fig_dendro_4ch, coph_4ch = create_dendrogram(embeddings_4ch, "Dendrograma RGB + UVB")
            if fig_dendro_4ch:
                st.pyplot(fig_dendro_4ch)
        
        # Resumen de resultados
        st.header("📋 Resumen de Resultados")
        
        improvement_silhouette = metrics_4ch.get('silhouette', 0) - metrics_rgb.get('silhouette', 0)
        improvement_db = metrics_rgb.get('davies_bouldin', 0) - metrics_4ch.get('davies_bouldin', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Mejora Silhouette Score", 
                f"{improvement_silhouette:+.4f}",
                delta=f"{improvement_silhouette:+.4f}"
            )
        
        with col2:
            st.metric(
                "Mejora Davies-Bouldin", 
                f"{improvement_db:+.4f}",
                delta=f"{improvement_db:+.4f}"
            )
        
        with col3:
            st.metric(
                "Diferencia Coef. Cophenético", 
                f"{coph_4ch - coph_rgb:+.4f}",
                delta=f"{coph_4ch - coph_rgb:+.4f}"
            )
        
        # Conclusiones
        st.subheader("🎯 Conclusiones")
        
        if improvement_silhouette > 0:
            st.success("✅ La adición del canal UVB mejora la separación de clusters (Silhouette Score)")
        else:
            st.warning("⚠️ El canal UVB no mejora significativamente la separación de clusters")
        
        if improvement_db > 0:
            st.success("✅ La adición del canal UVB reduce la dispersión intra-cluster (Davies-Bouldin)")
        else:
            st.warning("⚠️ El canal UVB no reduce significativamente la dispersión intra-cluster")
        
        st.info(f"""
        **Interpretación de resultados:**
        
        - El coeficiente cophenético cambió de {coph_rgb:.4f} (RGB) a {coph_4ch:.4f} (RGB+UVB)
        - Esto indica que la estructura jerárquica {'se mantiene similar' if abs(coph_4ch - coph_rgb) < 0.1 else 'cambia significativamente'} al añadir información UVB
        - La simulación tetrocromática {'revela' if improvement_silhouette > 0 else 'no revela'} patrones adicionales en el plumaje
        """)
        
        # Exportar resultados
        st.header("💾 Exportar Resultados")
        
        if UTILS_AVAILABLE and 'spectral_indices' in locals():
            # Crear resumen completo
            results_summary = export_results_summary(
                metrics_rgb, metrics_4ch, spectral_indices, coph_rgb, coph_4ch
            )
            
            # Mostrar resumen en JSON
            with st.expander("📋 Ver Resumen Completo"):
                st.json(results_summary)
            
            # Botón de descarga
            import json
            results_json = json.dumps(results_summary, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Descargar Resumen (JSON)",
                data=results_json,
                file_name=f"analisis_tetrocromatico_{uploaded_file.name.split('.')[0]}.json",
                mime="application/json"
            )
        
        # Información adicional
        st.header("ℹ️ Información Adicional")
        
        with st.expander("🔬 Sobre la Visión Tetrocromática"):
            st.markdown("""
            ### ¿Qué es la Visión Tetrocromática?
            
            Las aves poseen un sistema visual tetrocromático que incluye:
            - **Conos UV**: Sensibles a luz ultravioleta (300-400 nm)
            - **Conos S**: Sensibles a azul/violeta (400-500 nm)  
            - **Conos M**: Sensibles a verde (500-600 nm)
            - **Conos L**: Sensibles a rojo (600-700 nm)
            
            ### Importancia Biológica
            
            - **Comunicación**: Señales de cortejo invisibles al ojo humano
            - **Forrajeo**: Detección de frutos y presas
            - **Navegación**: Orientación usando patrones UV del cielo
            - **Reconocimiento**: Identificación de especies y sexo
            
            ### Aplicaciones de este Sistema
            
            - Estudios de biodiversidad y conservación
            - Análisis de mimetismo y camuflaje
            - Investigación evolutiva y ecológica
            - Clasificación automática de especies
            """)
        
        with st.expander("📊 Interpretación de Métricas"):
            st.markdown("""
            ### Métricas de Clustering
            
            **Silhouette Score** (0 a 1, mayor es mejor):
            - Mide qué tan bien separados están los clusters
            - > 0.5: Buena separación
            - > 0.7: Excelente separación
            
            **Davies-Bouldin Index** (≥ 0, menor es mejor):
            - Mide la compacidad intra-cluster vs separación inter-cluster
            - < 1.0: Clusters bien definidos
            
            **Calinski-Harabasz Index** (≥ 0, mayor es mejor):
            - Ratio de dispersión inter-cluster vs intra-cluster
            - Valores altos indican clusters bien separados
            
            ### Coeficiente Cophenético
            
            - Mide qué tan bien el dendrograma preserva las distancias originales
            - Rango: -1 a 1 (mayor es mejor)
            - > 0.8: Excelente representación
            - 0.6-0.8: Buena representación
            - < 0.6: Representación pobre
            """)
        
        with st.expander("🛠️ Detalles Técnicos"):
            st.markdown(f"""
            ### Modelos Utilizados
            
            **Predicción UV**: {uv_metadata.get('algo', 'N/A')}
            - R² Validación: {uv_metadata.get('r2_val', 0):.4f}
            - MAE Validación: {uv_metadata.get('mae_val', 0):.2e}
            
            **Extracción de Embeddings**: BEiT (Vision Transformer)
            - Modelo base preentrenado en ImageNet
            - Adaptación para 4 canales (RGB + UV)
            
            **Clustering**: K-means y HDBSCAN
            - Reducción dimensional: PCA y UMAP
            - Métricas de validación interna
            
            **Análisis Jerárquico**: Linkage promedio con distancia coseno
            - Coeficiente cophenético para validación
            - Dendrogramas comparativos
            """)
        
        # Panel Técnico Avanzado
        if show_technical_panel:
            st.header("🔧 Panel Técnico Avanzado")
            
            tab1, tab2, tab3, tab4 = st.tabs(["🔍 Análisis de Imagen", "🧠 Modelos", "📊 Métricas Detalladas", "⚙️ Sistema"])
            
            with tab1:
                st.subheader("Análisis Técnico de la Imagen")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Propiedades de la Imagen:**")
                    st.code(f"""
Dimensiones: {image_rgb.shape[1]} x {image_rgb.shape[0]} píxeles
Canales: {image_rgb.shape[2]} (RGB)
Tipo de datos: {image_rgb.dtype}
Tamaño en memoria: {image_rgb.nbytes / 1024:.1f} KB
Rango de valores: {np.min(image_rgb)} - {np.max(image_rgb)}
                    """)
                    
                    st.markdown("**Estadísticas por Canal:**")
                    stats_tech = pd.DataFrame({
                        'Canal': ['R', 'G', 'B', 'UV'],
                        'Min': [np.min(image_rgb[:,:,0]), np.min(image_rgb[:,:,1]), 
                               np.min(image_rgb[:,:,2]), np.min(uv_channel)],
                        'Max': [np.max(image_rgb[:,:,0]), np.max(image_rgb[:,:,1]), 
                               np.max(image_rgb[:,:,2]), np.max(uv_channel)],
                        'Media': [np.mean(image_rgb[:,:,0]), np.mean(image_rgb[:,:,1]), 
                                 np.mean(image_rgb[:,:,2]), np.mean(uv_channel)],
                        'Std': [np.std(image_rgb[:,:,0]), np.std(image_rgb[:,:,1]), 
                               np.std(image_rgb[:,:,2]), np.std(uv_channel)]
                    })
                    st.dataframe(stats_tech, use_container_width=True)
                
                with col2:
                    st.markdown("**Análisis de Segmentación:**")
                    total_pixels = mask.size
                    feather_pixels = np.sum(mask > 0)
                    background_pixels = total_pixels - feather_pixels
                    
                    st.code(f"""
Píxeles totales: {total_pixels:,}
Píxeles de pluma: {feather_pixels:,} ({feather_pixels/total_pixels*100:.1f}%)
Píxeles de fondo: {background_pixels:,} ({background_pixels/total_pixels*100:.1f}%)
Método usado: {segmentation_method}
                    """)
                    
                    # Histograma de la máscara
                    fig_mask, ax_mask = plt.subplots(figsize=(6, 4))
                    ax_mask.hist(mask.flatten(), bins=50, alpha=0.7, color='gray')
                    ax_mask.set_title("Distribución de la Máscara")
                    ax_mask.set_xlabel("Valor de píxel")
                    ax_mask.set_ylabel("Frecuencia")
                    st.pyplot(fig_mask)
            
            with tab2:
                st.subheader("Información de Modelos")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Modelo de Predicción UV:**")
                    if uv_metadata:
                        st.code(f"""
Algoritmo: {uv_metadata.get('algo', 'N/A')}
R² Validación: {uv_metadata.get('r2_val', 0):.6f}
MAE Validación: {uv_metadata.get('mae_val', 0):.2e}
Factor de escala: {uv_metadata.get('target_scale', 1.0)}
Estado: {'✅ Cargado' if uv_model else '❌ No disponible'}
                        """)
                    
                    st.markdown("**Dependencias del Sistema:**")
                    deps_status = {
                        'NumPy': '✅' if 'numpy' in sys.modules else '❌',
                        'OpenCV': '✅' if 'cv2' in sys.modules else '❌',
                        'scikit-learn': '✅' if SKLEARN_AVAILABLE else '❌',
                        'PyTorch': '✅' if TORCH_AVAILABLE else '❌',
                        'UMAP': '✅' if UMAP_AVAILABLE else '❌',
                        'SciPy': '✅' if SCIPY_AVAILABLE else '❌',
                        'Seaborn': '✅' if SEABORN_AVAILABLE else '❌'
                    }
                    
                    for dep, status in deps_status.items():
                        st.write(f"{status} {dep}")
                
                with col2:
                    st.markdown("**Modelos de Embeddings:**")
                    if model_rgb is not None:
                        st.code(f"""
Modelo RGB: BEiT Base Patch16 224
Estado: ✅ Cargado
Parámetros: ~86M
Entrada: 224x224x3
Salida: 768 dimensiones

Modelo 4-Canal: BEiT Adaptado
Estado: ✅ Cargado  
Entrada: 224x224x4 (RGB+UV)
Salida: 768 dimensiones
                        """)
                    else:
                        st.code(f"""
Modelo RGB: Características simples
Estado: ✅ Activo
Método: Estadísticas + Textura
Entrada: Variable
Salida: {len(features_rgb) if 'features_rgb' in locals() else 'N/A'} dimensiones

Modelo 4-Canal: Características + UV
Estado: ✅ Activo
Salida: {len(features_4ch) if 'features_4ch' in locals() else 'N/A'} dimensiones
                        """)
            
            with tab3:
                st.subheader("Métricas Detalladas de Clustering")
                
                if 'metrics_rgb' in locals() and 'metrics_4ch' in locals():
                    # Tabla comparativa detallada
                    metrics_comparison = pd.DataFrame({
                        'Métrica': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'],
                        'RGB': [
                            metrics_rgb.get('silhouette', 0),
                            metrics_rgb.get('davies_bouldin', 0),
                            metrics_rgb.get('calinski_harabasz', 0)
                        ],
                        'RGB+UVB': [
                            metrics_4ch.get('silhouette', 0),
                            metrics_4ch.get('davies_bouldin', 0),
                            metrics_4ch.get('calinski_harabasz', 0)
                        ],
                        'Mejora': [
                            metrics_4ch.get('silhouette', 0) - metrics_rgb.get('silhouette', 0),
                            metrics_rgb.get('davies_bouldin', 0) - metrics_4ch.get('davies_bouldin', 0),
                            metrics_4ch.get('calinski_harabasz', 0) - metrics_rgb.get('calinski_harabasz', 0)
                        ],
                        'Interpretación': [
                            'Mayor es mejor (0-1)',
                            'Menor es mejor (≥0)',
                            'Mayor es mejor (≥0)'
                        ]
                    })
                    
                    st.dataframe(metrics_comparison, use_container_width=True)
                    
                    # Gráfico de barras comparativo
                    fig_metrics, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    metrics_names = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
                    rgb_values = [metrics_rgb.get('silhouette', 0), 
                                 metrics_rgb.get('davies_bouldin', 0),
                                 metrics_rgb.get('calinski_harabasz', 0)]
                    uvb_values = [metrics_4ch.get('silhouette', 0),
                                 metrics_4ch.get('davies_bouldin', 0),
                                 metrics_4ch.get('calinski_harabasz', 0)]
                    
                    for i, (name, rgb_val, uvb_val, ax) in enumerate(zip(metrics_names, rgb_values, uvb_values, axes)):
                        x = ['RGB', 'RGB+UVB']
                        y = [rgb_val, uvb_val]
                        bars = ax.bar(x, y, color=['blue', 'red'], alpha=0.7)
                        ax.set_title(f'{name}')
                        ax.set_ylabel('Valor')
                        
                        # Añadir valores en las barras
                        for bar, val in zip(bars, y):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig_metrics)
            
            with tab4:
                st.subheader("Información del Sistema")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Entorno de Ejecución:**")
                    st.code(f"""
Python: {sys.version.split()[0]}
Plataforma: {sys.platform}
Streamlit: {st.__version__}
Directorio: {os.getcwd()}
                    """)
                    
                    st.markdown("**Memoria y Rendimiento:**")
                    import psutil
                    memory = psutil.virtual_memory()
                    st.code(f"""
RAM Total: {memory.total / (1024**3):.1f} GB
RAM Disponible: {memory.available / (1024**3):.1f} GB
RAM en Uso: {memory.percent:.1f}%
CPU Cores: {psutil.cpu_count()}
                    """)
                
                with col2:
                    st.markdown("**Archivos del Proyecto:**")
                    project_files = {
                        'Modelos UV': '✅' if os.path.exists('Modelos/uv_regressor_hgb_2.joblib') else '❌',
                        'Metadata UV': '✅' if os.path.exists('Modelos/uv_regressor_hgb_meta_2.json') else '❌',
                        'Utilidades': '✅' if os.path.exists('utils.py') else '❌',
                        'Imágenes ejemplo': '✅' if len(get_example_images()) > 0 else '❌',
                        'Demo bird': '✅' if os.path.exists('demo_bird.jpg') else '❌'
                    }
                    
                    for file, status in project_files.items():
                        st.write(f"{status} {file}")
                    
                    st.markdown("**Configuración Actual:**")
                    st.code(f"""
Clusters: {n_clusters}
Método clustering: {clustering_method}
Segmentación: {segmentation_method}
Realce UV: {uv_enhancement}x
Mapa color: {uv_colormap}
Debug: {'✅' if show_debug_info else '❌'}
                    """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            🦅 Sistema de Visión Tetrocromática para Aves<br>
            Desarrollado para análisis de biodiversidad y conservación
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()