import numpy as np
import cv2
import streamlit as st

try:
    import torch
    import timm
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

def extract_features_simple(image_rgb, image_4ch):
    """Extraer características simples sin PyTorch"""
    try:
        rgb_resized = cv2.resize(image_rgb, (224, 224))
        features_rgb = []
        features_4ch = []
        
        # Estadísticas por canal RGB
        for i in range(3):
            channel = rgb_resized[:, :, i]
            features_rgb.extend([
                np.mean(channel), np.std(channel), np.median(channel),
                np.percentile(channel, 25), np.percentile(channel, 75)
            ])
        
        # Características de textura (gradientes)
        gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        features_rgb.extend([np.mean(gradient_mag), np.std(gradient_mag)])
        
        # UV features
        if image_4ch is not None and image_4ch.shape[2] == 4:
            uv_channel = image_4ch[:, :, 0]
            uv_resized = cv2.resize(uv_channel, (224, 224))
            features_4ch = features_rgb.copy()
            features_4ch.extend([
                np.mean(uv_resized), np.std(uv_resized), np.median(uv_resized),
                np.percentile(uv_resized, 25), np.percentile(uv_resized, 75)
            ])
            rgb_mean = np.mean(rgb_resized, axis=2)
            correlation = np.corrcoef(uv_resized.flatten(), rgb_mean.flatten())[0, 1]
            features_4ch.append(correlation if not np.isnan(correlation) else 0)
        else:
            features_4ch = features_rgb.copy()
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
            embedding_4ch = embedding_rgb + np.random.normal(0, 0.01, embedding_rgb.shape)
            
            return embedding_rgb, embedding_4ch
        except Exception as e:
            st.warning(f"⚠️ Error con modelos BEiT, usando características simples: {e}")
            return extract_features_simple(image_rgb, image_4ch)
    else:
        # Usar extracción de características simples
        return extract_features_simple(image_rgb, image_4ch)
