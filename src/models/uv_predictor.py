import numpy as np
import joblib
import json
import streamlit as st

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
