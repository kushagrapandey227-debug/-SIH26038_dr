"""
sample_generator.py - Generates realistic synthetic fundus images for testing
-----------------------------------------------------------------------------
Creates sample retinal images (Normal, Mild DR, Severe DR) so users can test
the Streamlit dashboard immediately without needing external fundus datasets.
"""

import os
import cv2
import numpy as np
from PIL import Image

def draw_fundus_base(width=512, height=512):
    """Draws a synthetic retinal fundus base image (orange-red disk with optic disc & vessels)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. Background dark space
    center = (width // 2, height // 2)
    radius = int(width * 0.45)
    
    # 2. Reddish-orange fundus disk gradient
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    fundus_mask = dist_from_center <= radius
    
    # Fundus gradient color (Deep orange-red)
    fundus_color = np.zeros((height, width, 3), dtype=np.float32)
    fundus_color[:, :, 0] = 180 - (dist_from_center / radius) * 40  # Red
    fundus_color[:, :, 1] = 70 - (dist_from_center / radius) * 30   # Green
    fundus_color[:, :, 2] = 20 - (dist_from_center / radius) * 10   # Blue
    
    # Apply fundus mask
    img[fundus_mask] = np.clip(fundus_color[fundus_mask], 0, 255).astype(np.uint8)
    
    # 3. Optic Disc (Bright yellowish circle at center-left)
    optic_center = (int(width * 0.35), int(height * 0.5))
    cv2.circle(img, optic_center, 35, (240, 230, 160), -1)
    cv2.circle(img, optic_center, 38, (210, 190, 120), 2)
    
    # 4. Macula (Darker reddish region at center-right)
    macula_center = (int(width * 0.65), int(height * 0.5))
    cv2.circle(img, macula_center, 40, (140, 45, 15), -1)
    
    # 5. Retinal Blood Vessels radiating from optic disc
    vessel_color = (110, 20, 10)
    for angle in [-60, -30, 0, 30, 60, 120, 150, 180, 210, 240]:
        rad = np.radians(angle)
        end_x = int(optic_center[0] + np.cos(rad) * 200)
        end_y = int(optic_center[1] + np.sin(rad) * 200)
        ctrl_x = int(optic_center[0] + np.cos(rad + 0.2) * 100)
        ctrl_y = int(optic_center[1] + np.sin(rad + 0.2) * 100)
        
        pts = np.array([optic_center, (ctrl_x, ctrl_y), (end_x, end_y)], np.int32)
        cv2.polylines(img, [pts], False, vessel_color, thickness=3, lineType=cv2.LINE_AA)
        
    return img, fundus_mask


def generate_sample_images(output_dir="samples"):
    """Generates 3 synthetic fundus images for demonstration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Sample 1: Normal Fundus ---
    normal_img, mask = draw_fundus_base()
    # Add subtle natural texture noise
    noise = np.random.normal(0, 3, normal_img.shape).astype(np.int16)
    normal_img = np.clip(normal_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    normal_img[~mask] = 0
    cv2.imwrite(os.path.join(output_dir, "sample_normal.jpg"), cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))
    
    # --- Sample 2: Mild DR Fundus (with Microaneurysms) ---
    mild_img, mask = draw_fundus_base()
    # Add tiny red dot microaneurysms
    np.random.seed(42)
    for _ in range(12):
        rx = np.random.randint(180, 380)
        ry = np.random.randint(150, 350)
        cv2.circle(mild_img, (rx, ry), np.random.randint(2, 4), (160, 10, 10), -1)
    mild_img[~mask] = 0
    cv2.imwrite(os.path.join(output_dir, "sample_mild_dr.jpg"), cv2.cvtColor(mild_img, cv2.COLOR_RGB2BGR))
    
    # --- Sample 3: Severe DR Fundus (Exudates + Hemorrhages) ---
    severe_img, mask = draw_fundus_base()
    np.random.seed(100)
    # Add bright yellow hard exudates
    for _ in range(15):
        ex = np.random.randint(200, 400)
        ey = np.random.randint(120, 380)
        cv2.circle(severe_img, (ex, ey), np.random.randint(4, 8), (240, 240, 180), -1)
    # Add dark red hemorrhages
    for _ in range(10):
        hx = np.random.randint(180, 380)
        hy = np.random.randint(150, 380)
        cv2.ellipse(severe_img, (hx, hy), (np.random.randint(5, 12), np.random.randint(3, 8)), np.random.randint(0, 180), 0, 360, (130, 5, 5), -1)
    severe_img[~mask] = 0
    cv2.imwrite(os.path.join(output_dir, "sample_severe_dr.jpg"), cv2.cvtColor(severe_img, cv2.COLOR_RGB2BGR))
    
    print(f"Successfully generated sample fundus images in '{output_dir}/'")

if __name__ == "__main__":
    generate_sample_images()
