import numpy as np
import cv2

def remove_background(image):
    """Remover fondo usando simulacion simple con threshold"""
    try:
        # Convertir a HSV para mejor segmentación
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Crear máscara simple
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
        import streamlit as st
        st.error(f"Error en remoción de fondo: {e}")
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

def create_4channel_image(rgb_image, uv_channel):
    """Crear imagen de 4 canales (UV, R, G, B)"""
    try:
        # Combinar canales
        uvrgb_image = np.dstack([uv_channel, rgb_image])
        return uvrgb_image
    except Exception as e:
        import streamlit as st
        st.error(f"Error creando imagen 4 canales: {e}")
        return None
