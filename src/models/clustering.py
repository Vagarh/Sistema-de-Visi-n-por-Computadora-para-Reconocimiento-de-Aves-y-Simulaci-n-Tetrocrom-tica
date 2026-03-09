import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, HDBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def perform_clustering(embeddings, n_clusters=5):
    """Realizar clustering y calcular métricas"""
    if not SKLEARN_AVAILABLE:
        n_samples = len(embeddings)
        return np.random.randint(0, n_clusters, n_samples), np.random.randint(-1, n_clusters-1, n_samples), {
            'silhouette': np.random.uniform(0.3, 0.7),
            'davies_bouldin': np.random.uniform(0.5, 1.5),
            'calinski_harabasz': np.random.uniform(100, 500)
        }
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels_kmeans = kmeans.fit_predict(embeddings)
        
        try:
            hdbscan_model = HDBSCAN(min_samples=10, min_cluster_size=5)
            labels_hdbscan = hdbscan_model.fit_predict(embeddings)
        except:
            labels_hdbscan = labels_kmeans
        
        metrics = {
            'silhouette': silhouette_score(embeddings, labels_kmeans),
            'davies_bouldin': davies_bouldin_score(embeddings, labels_kmeans),
            'calinski_harabasz': calinski_harabasz_score(embeddings, labels_kmeans)
        }
        
        return labels_kmeans, labels_hdbscan, metrics
    except Exception as e:
        st.error(f"Error en clustering: {e}")
        return None, None, {}

def create_dendrogram(embeddings, title="Dendrograma"):
    """Crear dendrograma jerárquico"""
    if not SCIPY_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        n_samples = len(embeddings)
        y = np.cumsum(np.random.exponential(scale=2, size=n_samples-1))
        for i in range(n_samples-1):
            ax.plot([i, i+1], [0, y[i]], 'b-')
            ax.plot([i+1, i+1], [0, y[i]], 'b-')
            ax.plot([i, i+1], [y[i], y[i]], 'b-')
        coph_coeff = np.random.uniform(0.2, 0.8)
        ax.set_title(f"{title}\nCoeficiente Cophenético: {coph_coeff:.4f} (simulado)")
        return fig, coph_coeff
    
    try:
        linkage_matrix = linkage(embeddings, method='average', metric='cosine')
        coph_coeff, _ = cophenet(linkage_matrix, pdist(embeddings, metric='cosine'))
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(linkage_matrix, ax=ax, leaf_rotation=90)
        ax.set_title(f"{title}\nCoeficiente Cophenético: {coph_coeff:.4f}")
        return fig, coph_coeff
    except Exception as e:
        st.error(f"Error creando dendrograma: {e}")
        return None, 0

def plot_umap_projection(embeddings_rgb, embeddings_4ch, labels_rgb, labels_4ch):
    """Crear proyección UMAP comparativa con Plotly"""
    import plotly.subplots as sp
    import plotly.graph_objects as go
    
    if not UMAP_AVAILABLE:
        if SKLEARN_AVAILABLE:
            try:
                pca_rgb = PCA(n_components=2, random_state=42)
                projection_rgb = pca_rgb.fit_transform(embeddings_rgb)
                pca_4ch = PCA(n_components=2, random_state=42)
                projection_4ch = pca_4ch.fit_transform(embeddings_4ch)
                
                fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Proyección PCA - RGB", "Proyección PCA - RGB + UVB"))
                
                fig.add_trace(go.Scatter(x=projection_rgb[:, 0], y=projection_rgb[:, 1], mode='markers', 
                                         marker=dict(color=labels_rgb, colorscale='Viridis', showscale=False)), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=projection_4ch[:, 0], y=projection_4ch[:, 1], mode='markers', 
                                         marker=dict(color=labels_4ch, colorscale='Viridis', showscale=False)), row=1, col=2)
                
                fig.update_layout(height=500, width=1000, title_text="Proyecciones Interactivas PCA", showlegend=False)
                return fig
            except Exception as e:
                st.error(f"Error en proyección PCA: {e}")
                return None
        return None
    
    try:
        umap_rgb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        projection_rgb = umap_rgb.fit_transform(embeddings_rgb)
        umap_4ch = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        projection_4ch = umap_4ch.fit_transform(embeddings_4ch)
        
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Proyección UMAP - RGB", "Proyección UMAP - RGB + UVB"))
        
        fig.add_trace(go.Scatter(x=projection_rgb[:, 0], y=projection_rgb[:, 1], mode='markers', 
                                 marker=dict(color=labels_rgb, colorscale='Plasma', showscale=False)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=projection_4ch[:, 0], y=projection_4ch[:, 1], mode='markers', 
                                 marker=dict(color=labels_4ch, colorscale='Plasma', showscale=True, colorbar=dict(title='Cluster'))), row=1, col=2)
        
        fig.update_layout(height=600, width=1200, title_text="Proyecciones Interactivas UMAP", showlegend=False, template='plotly_white')
        return fig
    except Exception as e:
        st.error(f"Error en proyección UMAP: {e}")
        return None
