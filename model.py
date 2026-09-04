"""
model.py - Diabetic Retinopathy Machine Learning Model & Preprocessing Pipeline
--------------------------------------------------------------------------------
This module handles:
1. Loading a pre-trained deep learning backbone (PyTorch ResNet-50).
2. Retinal fundus image preprocessing (resizing, normalization, optional Ben Graham enhancement).
3. Running model inference to return DR severity grade, class description, and confidence score.

Dataset & Scale Reference:
APTOS 2019 / International Clinical Diabetic Retinopathy (ICDR) Scale:
  0 - No DR (Healthy retinal fundus)
  1 - Mild DR (Microaneurysms only)
  2 - Moderate DR (Hemorrhages, microaneurysms, hard exudates)
  3 - Severe DR (Severe hemorrhages, cotton wool spots, venous beading)
  4 - Proliferative DR (Neovascularization present, high risk of vision loss)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------
# Diabetic Retinopathy Severity Mapping & Descriptions
# ---------------------------------------------------------
DR_CLASSES = {
    0: {
        "name": "No DR",
        "severity": "Normal",
        "badge_color": "green",
        "description": "No clinical signs of diabetic retinopathy detected in the fundus image.",
        "recommendation": "Routine annual eye check-up recommended."
    },
    1: {
        "name": "Mild DR",
        "severity": "Mild",
        "badge_color": "blue",
        "description": "Mild non-proliferative diabetic retinopathy. Early microaneurysms detected.",
        "recommendation": "Monitor blood sugar levels closely. Follow-up eye exam in 6-12 months."
    },
    2: {
        "name": "Moderate DR",
        "severity": "Moderate",
        "badge_color": "orange",
        "description": "Moderate non-proliferative DR. Presence of exudates, microaneurysms, or minor hemorrhages.",
        "recommendation": "Consult an ophthalmologist for comprehensive evaluation within 3 months."
    },
    3: {
        "name": "Severe DR",
        "severity": "Severe",
        "badge_color": "red",
        "description": "Severe non-proliferative DR. Widespread hemorrhages or cotton wool spots present.",
        "recommendation": "Urgent ophthalmologist consultation required within 2-4 weeks."
    },
    4: {
        "name": "Proliferative DR",
        "severity": "Proliferative",
        "badge_color": "darkred",
        "description": "Advanced proliferative DR. Abnormal new blood vessel growth (neovascularization) visible.",
        "recommendation": "Immediate specialized retinal care required to prevent irreversible vision loss."
    }
}


def build_dr_model(num_classes: int = 5) -> nn.Module:
    """
    Builds a PyTorch ResNet-50 classification network.
    Replaces the final fully-connected (fc) layer with 5 output units corresponding
    to the DR severity grades.
    """
    # Load pre-trained ResNet-50 architecture
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # Replace classification head for 5 DR grades
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model


@torch.no_grad()
def load_model(weights_path: str = None, device: str = "cpu") -> nn.Module:
    """
    Loads and initializes the DR classification model.
    If weights_path is provided and exists, loads custom weights.
    Otherwise, loads the pre-trained transfer learning baseline model.
    """
    model = build_dr_model(num_classes=5)
    
    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded custom weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path} ({e}). Using baseline model.")
            
    model.to(device)
    model.eval()
    return model


def ben_graham_enhancement(image_np: np.ndarray) -> np.ndarray:
    """
    Applies Ben Graham's method for fundus image contrast enhancement.
    This technique subtracts the local Gaussian blurred image color to accentuate
    retinal lesions (microaneurysms, hemorrhages, exudates).
    """
    # Convert RGB to BGR for OpenCV processing
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Crop borders/black padding around fundus circle
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    if mask.any():
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        bgr = bgr[y:y+h, x:x+w]
        
    # Resize to standard scale before blurring
    bgr = cv2.resize(bgr, (512, 512))
    
    # Subtract local mean color (Gaussian Blur)
    blurred = cv2.GaussianBlur(bgr, (0, 0), 512 / 30)
    enhanced = cv2.addWeighted(bgr, 4, blurred, -4, 128)
    
    # Circle crop to isolate retinal region
    circle_mask = np.zeros(enhanced.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (256, 256), int(512 * 0.9 / 2), (1, 1, 1), -1, 8, 0)
    enhanced = enhanced * circle_mask + 128 * (1 - circle_mask)
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return enhanced_rgb


def preprocess_image(pil_img: Image.Image, apply_enhancement: bool = False):
    """
    Preprocesses input PIL image into PyTorch tensor.
    
    Args:
        pil_img: Input retinal fundus PIL image.
        apply_enhancement: If True, applies Ben Graham preprocessing.
        
    Returns:
        tensor: Formatted PyTorch tensor (1, 3, 224, 224) ready for model input.
        display_np: Processed RGB image array (224, 224, 3) for UI visualization.
    """
    img_np = np.array(pil_img.convert("RGB"))
    
    if apply_enhancement:
        img_np = ben_graham_enhancement(img_np)
        
    processed_pil = Image.fromarray(img_np).resize((224, 224))
    
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform_pipeline(processed_pil).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    display_np = np.array(processed_pil)
    
    return tensor, display_np


def predict_dr(model: nn.Module, tensor: torch.Tensor, device: str = "cpu"):
    """
    Runs model inference on input tensor and returns prediction results.
    
    Returns:
        dict containing predicted_class, class_name, severity, confidence,
        all_probabilities dict, description, and recommendation.
    """
    tensor = tensor.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
    pred_index = int(np.argmax(probabilities))
    confidence_pct = float(probabilities[pred_index] * 100)
    
    class_info = DR_CLASSES[pred_index]
    
    probs_dict = {
        DR_CLASSES[i]["name"]: float(probabilities[i] * 100)
        for i in range(len(DR_CLASSES))
    }
    
    return {
        "class_id": pred_index,
        "class_name": class_info["name"],
        "severity": class_info["severity"],
        "badge_color": class_info["badge_color"],
        "confidence": confidence_pct,
        "probabilities": probs_dict,
        "description": class_info["description"],
        "recommendation": class_info["recommendation"]
    }
