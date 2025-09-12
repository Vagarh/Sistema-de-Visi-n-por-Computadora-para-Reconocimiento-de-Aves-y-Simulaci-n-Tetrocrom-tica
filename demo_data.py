"""
Datos de demostración para la aplicación Streamlit
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def create_synthetic_bird_image(width=400, height=300):
    """
    Crear una imagen sintética de ave para demostración
    """
    # Crear imagen base
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fondo (cielo azul)
    image[:, :] = [135, 206, 235]  # Sky blue
    
    # Cuerpo del ave (elipse)
    center = (width // 2, height // 2)
    axes = (width // 4, height // 6)
    
    # Cuerpo principal (marrón)
    cv2.ellipse(image, center, axes, 0, 0, 360, (101, 67, 33), -1)
    
    # Cabeza (círculo)
    head_center = (center[0], center[1] - height // 4)
    head_radius = height // 8
    cv2.circle(image, head_center, head_radius, (139, 69, 19), -1)
    
    # Pico
    beak_points = np.array([
        [head_center[0] - head_radius//2, head_center[1]],
        [head_center[0] - head_radius - 20, head_center[1] - 5],
        [head_center[0] - head_radius - 20, head_center[1] + 5]
    ], np.int32)
    cv2.fillPoly(image, [beak_points], (255, 140, 0))
    
    # Ojo
    eye_center = (head_center[0] - 10, head_center[1] - 5)
    cv2.circle(image, eye_center, 5, (0, 0, 0), -1)
    cv2.circle(image, eye_center, 2, (255, 255, 255), -1)
    
    # Ala
    wing_points = np.array([
        [center[0] - 10, center[1] - 20],
        [center[0] + 40, center[1] - 30],
        [center[0] + 60, center[1] + 10],
        [center[0] + 20, center[1] + 20]
    ], np.int32)
    cv2.fillPoly(image, [wing_points], (160, 82, 45))
    
    # Plumas del ala (detalles)
    for i in range(3):
        y_offset = i * 15
        feather_points = np.array([
            [center[0] + 20, center[1] - 15 + y_offset],
            [center[0] + 45, center[1] - 20 + y_offset],
            [center[0] + 50, center[1] - 5 + y_offset],
            [center[0] + 25, center[1] + y_offset]
        ], np.int32)
        cv2.fillPoly(image, [feather_points], (205, 133, 63))
    
    # Cola
    tail_points = np.array([
        [center[0] + axes[0] - 10, center[1]],
        [center[0] + axes[0] + 40, center[1] - 20],
        [center[0] + axes[0] + 50, center[1] + 20]
    ], np.int32)
    cv2.fillPoly(image, [tail_points], (139, 69, 19))
    
    # Patas
    leg1_start = (center[0] - 15, center[1] + axes[1] - 5)
    leg1_end = (center[0] - 15, center[1] + axes[1] + 25)
    cv2.line(image, leg1_start, leg1_end, (255, 140, 0), 3)
    
    leg2_start = (center[0] + 5, center[1] + axes[1] - 5)
    leg2_end = (center[0] + 5, center[1] + axes[1] + 25)
    cv2.line(image, leg2_start, leg2_end, (255, 140, 0), 3)
    
    # Añadir algo de ruido para hacer más realista
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def create_sample_uv_pattern(rgb_image):
    """
    Crear un patrón UV sintético basado en la imagen RGB
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Crear patrón UV basado en características de la imagen
    uv_channel = np.zeros_like(gray, dtype=np.float32)
    
    # Las plumas suelen tener alta reflectancia UV
    # Crear patrón basado en gradientes y texturas
    
    # Detectar bordes (las plumas tienen bordes definidos)
    edges = cv2.Canny(gray, 50, 150)
    
    # Las áreas con bordes tendrán más UV
    uv_channel += edges.astype(np.float32) * 0.3
    
    # Añadir patrón basado en intensidad RGB
    # Las áreas más oscuras en RGB pueden tener más UV
    uv_from_rgb = (255 - gray).astype(np.float32) / 255.0
    uv_channel += uv_from_rgb * 0.4
    
    # Añadir patrón de textura
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    texture = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    texture = np.abs(texture)
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    uv_channel += texture * 0.3
    
    # Normalizar a rango [0, 255]
    uv_channel = (uv_channel - uv_channel.min()) / (uv_channel.max() - uv_channel.min() + 1e-8)
    uv_channel = (uv_channel * 255).astype(np.uint8)
    
    return uv_channel

def save_demo_image(filename="demo_bird.jpg"):
    """
    Guardar imagen de demostración
    """
    image = create_synthetic_bird_image()
    pil_image = Image.fromarray(image)
    pil_image.save(filename)
    print(f"Imagen de demostración guardada como: {filename}")
    return filename

if __name__ == "__main__":
    # Crear y guardar imagen de demostración
    demo_file = save_demo_image()
    
    # Mostrar la imagen
    image = create_synthetic_bird_image()
    uv_pattern = create_sample_uv_pattern(image)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(image)
    ax1.set_title("Imagen RGB Sintética")
    ax1.axis('off')
    
    im2 = ax2.imshow(uv_pattern, cmap='viridis')
    ax2.set_title("Patrón UV Sintético")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig("demo_preview.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Archivos de demostración creados:")
    print(f"- {demo_file}")
    print("- demo_preview.png")