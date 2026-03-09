"""
Utilidades para la aplicación Streamlit de Visión Tetrocromática
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Importar seaborn opcionalmente
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

def enhance_uv_visualization(uv_channel, method='viridis'):
    """
    Mejorar la visualización del canal UV con diferentes mapas de color
    """
    colormaps = {
        'viridis': plt.cm.viridis,
        'plasma': plt.cm.plasma,
        'inferno': plt.cm.inferno,
        'magma': plt.cm.magma,
        'uv_custom': create_uv_colormap()
    }
    
    return colormaps.get(method, plt.cm.viridis)

def create_uv_colormap():
    """
    Crear un mapa de color personalizado para UV que simule la percepción aviar
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    colors = ['#000033', '#000066', '#0000CC', '#3333FF', '#6666FF', '#9999FF', '#CCCCFF', '#FFFFFF']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('uv_custom', colors, N=n_bins)
    return cmap

def calculate_spectral_indices(rgb_image, uv_channel):
    """
    Calcular índices espectrales relevantes para análisis ornitológico
    """
    # Normalizar canales
    r = rgb_image[:,:,0].astype(np.float32) / 255.0
    g = rgb_image[:,:,1].astype(np.float32) / 255.0
    b = rgb_image[:,:,2].astype(np.float32) / 255.0
    uv = uv_channel.astype(np.float32) / 255.0
    
    # Evitar división por cero
    epsilon = 1e-8
    
    indices = {}
    
    # Índice UV/Visible
    indices['UV_VIS_ratio'] = np.mean(uv) / (np.mean(r + g + b) + epsilon)
    
    # Índice de contraste UV
    indices['UV_contrast'] = np.std(uv) / (np.mean(uv) + epsilon)
    
    # Índice de saturación RGB
    rgb_max = np.maximum(np.maximum(r, g), b)
    rgb_min = np.minimum(np.minimum(r, g), b)
    indices['RGB_saturation'] = np.mean((rgb_max - rgb_min) / (rgb_max + epsilon))
    
    # Índice de brillo tetrocromático
    indices['tetrachromatic_brightness'] = np.mean(uv + r + g + b) / 4.0
    
    # Índice de dominancia UV
    total_intensity = uv + r + g + b + epsilon
    indices['UV_dominance'] = np.mean(uv / total_intensity)
    
    return indices

def create_comparison_plot(rgb_image, uv_channel, title="Comparación RGB vs UVB"):
    """
    Crear una visualización comparativa mejorada
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Imagen RGB original
    axes[0,0].imshow(rgb_image)
    axes[0,0].set_title("Imagen RGB Original")
    axes[0,0].axis('off')
    
    # Canal UV
    im_uv = axes[0,1].imshow(uv_channel, cmap='viridis')
    axes[0,1].set_title("Canal UVB Estimado")
    axes[0,1].axis('off')
    plt.colorbar(im_uv, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    # Composición falso color (UV como rojo)
    false_color = np.zeros_like(rgb_image)
    false_color[:,:,0] = uv_channel  # UV en canal rojo
    false_color[:,:,1] = rgb_image[:,:,1]  # Verde original
    false_color[:,:,2] = rgb_image[:,:,2]  # Azul original
    axes[0,2].imshow(false_color)
    axes[0,2].set_title("Falso Color (UV-G-B)")
    axes[0,2].axis('off')
    
    # Histogramas comparativos
    axes[1,0].hist(rgb_image[:,:,0].flatten(), bins=50, alpha=0.7, color='red', label='R')
    axes[1,0].hist(rgb_image[:,:,1].flatten(), bins=50, alpha=0.7, color='green', label='G')
    axes[1,0].hist(rgb_image[:,:,2].flatten(), bins=50, alpha=0.7, color='blue', label='B')
    axes[1,0].set_title("Distribución RGB")
    axes[1,0].set_xlabel("Intensidad")
    axes[1,0].set_ylabel("Frecuencia")
    axes[1,0].legend()
    
    axes[1,1].hist(uv_channel.flatten(), bins=50, alpha=0.7, color='purple')
    axes[1,1].set_title("Distribución UVB")
    axes[1,1].set_xlabel("Intensidad UVB")
    axes[1,1].set_ylabel("Frecuencia")
    
    # Mapa de calor de correlación
    # Crear matriz de correlación entre canales
    h, w = uv_channel.shape
    sample_size = min(10000, h * w)  # Muestrear para eficiencia
    
    indices = np.random.choice(h * w, sample_size, replace=False)
    r_flat = rgb_image[:,:,0].flatten()[indices]
    g_flat = rgb_image[:,:,1].flatten()[indices]
    b_flat = rgb_image[:,:,2].flatten()[indices]
    uv_flat = uv_channel.flatten()[indices]
    
    corr_data = pd.DataFrame({
        'R': r_flat,
        'G': g_flat,
        'B': b_flat,
        'UV': uv_flat
    })
    
    corr_matrix = corr_data.corr()
    
    if SEABORN_AVAILABLE:
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    ax=axes[1,2], square=True)
    else:
        # Fallback usando matplotlib
        im = axes[1,2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1,2].set_xticks(range(len(corr_matrix.columns)))
        axes[1,2].set_yticks(range(len(corr_matrix.columns)))
        axes[1,2].set_xticklabels(corr_matrix.columns)
        axes[1,2].set_yticklabels(corr_matrix.columns)
        
        # Añadir valores de correlación como texto
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                axes[1,2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center', color='black')
        
        plt.colorbar(im, ax=axes[1,2])
    
    axes[1,2].set_title("Correlación entre Canales")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def analyze_texture_features(image, mask=None):
    """
    Analizar características de textura en la imagen
    """
    from skimage.feature import graycomatrix, graycoprops
    from skimage import img_as_ubyte
    
    # Convertir a escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Aplicar máscara si se proporciona
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Calcular matriz de co-ocurrencia
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    gray_ubyte = img_as_ubyte(gray)
    glcm = graycomatrix(gray_ubyte, distances, angles, levels=256, symmetric=True, normed=True)
    
    # Calcular propiedades de textura
    texture_features = {}
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    for prop in properties:
        values = graycoprops(glcm, prop)
        texture_features[f'{prop}_mean'] = np.mean(values)
        texture_features[f'{prop}_std'] = np.std(values)
    
    return texture_features

def create_advanced_eda_plots(rgb_image, uv_channel):
    """
    Crear visualizaciones EDA más avanzadas
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Layout de subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Imagen original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_image)
    ax1.set_title("Imagen RGB Original")
    ax1.axis('off')
    
    # 2. Canal UV con diferentes visualizaciones
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(uv_channel, cmap='viridis')
    ax2.set_title("Canal UVB - Viridis")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(uv_channel, cmap=create_uv_colormap())
    ax3.set_title("Canal UVB - Personalizado")
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 3. Análisis de intensidad por regiones
    ax4 = fig.add_subplot(gs[0, 3])
    # Dividir imagen en cuadrantes y analizar
    h, w = uv_channel.shape
    quadrants = [
        uv_channel[:h//2, :w//2],  # Superior izquierdo
        uv_channel[:h//2, w//2:],  # Superior derecho
        uv_channel[h//2:, :w//2],  # Inferior izquierdo
        uv_channel[h//2:, w//2:]   # Inferior derecho
    ]
    
    quad_means = [np.mean(q) for q in quadrants]
    ax4.bar(['SI', 'SD', 'II', 'ID'], quad_means, color=['red', 'green', 'blue', 'purple'])
    ax4.set_title("Intensidad UVB por Cuadrante")
    ax4.set_ylabel("Intensidad Media")
    
    # 4. Distribuciones con estadísticas
    ax5 = fig.add_subplot(gs[1, :2])
    colors = ['red', 'green', 'blue', 'purple']
    channels = ['R', 'G', 'B', 'UV']
    data = [rgb_image[:,:,0].flatten(), rgb_image[:,:,1].flatten(), 
            rgb_image[:,:,2].flatten(), uv_channel.flatten()]
    
    for i, (channel, color, channel_data) in enumerate(zip(channels, colors, data)):
        ax5.hist(channel_data, bins=50, alpha=0.6, color=color, label=f'{channel}')
    
    ax5.set_title("Distribuciones de Intensidad por Canal")
    ax5.set_xlabel("Intensidad")
    ax5.set_ylabel("Frecuencia")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 5. Box plots comparativos
    ax6 = fig.add_subplot(gs[1, 2:])
    box_data = [rgb_image[:,:,0].flatten(), rgb_image[:,:,1].flatten(), 
                rgb_image[:,:,2].flatten(), uv_channel.flatten()]
    
    bp = ax6.boxplot(box_data, labels=['R', 'G', 'B', 'UV'], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax6.set_title("Distribución de Intensidades - Box Plots")
    ax6.set_ylabel("Intensidad")
    ax6.grid(True, alpha=0.3)
    
    # 6. Análisis de gradientes
    ax7 = fig.add_subplot(gs[2, 0])
    # Calcular gradientes en UV
    grad_x = cv2.Sobel(uv_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(uv_channel, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    im7 = ax7.imshow(gradient_magnitude, cmap='hot')
    ax7.set_title("Gradientes UVB")
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    # 7. Mapa de calor de correlación espacial
    ax8 = fig.add_subplot(gs[2, 1])
    # Correlación local entre RGB y UV
    kernel_size = 15
    correlation_map = np.zeros_like(uv_channel, dtype=np.float32)
    
    for i in range(kernel_size//2, uv_channel.shape[0] - kernel_size//2):
        for j in range(kernel_size//2, uv_channel.shape[1] - kernel_size//2):
            uv_patch = uv_channel[i-kernel_size//2:i+kernel_size//2+1, 
                                 j-kernel_size//2:j+kernel_size//2+1].flatten()
            rgb_patch = np.mean(rgb_image[i-kernel_size//2:i+kernel_size//2+1, 
                                        j-kernel_size//2:j+kernel_size//2+1], axis=2).flatten()
            
            if len(uv_patch) > 1 and len(rgb_patch) > 1:
                correlation_map[i, j] = np.corrcoef(uv_patch, rgb_patch)[0, 1]
    
    im8 = ax8.imshow(correlation_map, cmap='RdBu_r', vmin=-1, vmax=1)
    ax8.set_title("Correlación Local RGB-UV")
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    # 8. Estadísticas resumidas
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.axis('off')
    
    # Calcular índices espectrales
    indices = calculate_spectral_indices(rgb_image, uv_channel)
    
    stats_text = f"""
    ESTADÍSTICAS ESPECTRALES
    
    Ratio UV/Visible: {indices['UV_VIS_ratio']:.4f}
    Contraste UV: {indices['UV_contrast']:.4f}
    Saturación RGB: {indices['RGB_saturation']:.4f}
    Brillo Tetrocromático: {indices['tetrachromatic_brightness']:.4f}
    Dominancia UV: {indices['UV_dominance']:.4f}
    
    ESTADÍSTICAS BÁSICAS UV
    Media: {np.mean(uv_channel):.2f}
    Mediana: {np.median(uv_channel):.2f}
    Desv. Std: {np.std(uv_channel):.2f}
    Min: {np.min(uv_channel):.2f}
    Max: {np.max(uv_channel):.2f}
    """
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle("Análisis Exploratorio Avanzado - Visión Tetrocromática", fontsize=16)
    
    return fig

def export_results_summary(metrics_rgb, metrics_4ch, spectral_indices, coph_rgb, coph_4ch):
    """
    Crear un resumen exportable de los resultados
    """
    summary = {
        'Análisis de Clustering': {
            'RGB': {
                'Silhouette Score': metrics_rgb.get('silhouette', 0),
                'Davies-Bouldin Index': metrics_rgb.get('davies_bouldin', 0),
                'Calinski-Harabasz Index': metrics_rgb.get('calinski_harabasz', 0)
            },
            'RGB+UVB': {
                'Silhouette Score': metrics_4ch.get('silhouette', 0),
                'Davies-Bouldin Index': metrics_4ch.get('davies_bouldin', 0),
                'Calinski-Harabasz Index': metrics_4ch.get('calinski_harabasz', 0)
            }
        },
        'Análisis Jerárquico': {
            'Coeficiente Cophenético RGB': coph_rgb,
            'Coeficiente Cophenético RGB+UVB': coph_4ch,
            'Diferencia': coph_4ch - coph_rgb
        },
        'Índices Espectrales': spectral_indices,
        'Mejoras con UVB': {
            'Silhouette': metrics_4ch.get('silhouette', 0) - metrics_rgb.get('silhouette', 0),
            'Davies-Bouldin': metrics_rgb.get('davies_bouldin', 0) - metrics_4ch.get('davies_bouldin', 0),
            'Calinski-Harabasz': metrics_4ch.get('calinski_harabasz', 0) - metrics_rgb.get('calinski_harabasz', 0)
        }
    }
    
    return summary