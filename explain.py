"""
explain.py - Explainable AI (XAI) engine using Grad-CAM
-------------------------------------------------------
Grad-CAM (Gradient-weighted Class Activation Mapping) produces visual explanations 
for decisions from Convolutional Neural Network (CNN) models.

How it works for Diabetic Retinopathy:
1. It registers hooks on the final convolutional layer of the network (e.g. model.layer4).
2. It tracks the feature map activations during forward inference.
3. During backpropagation, it computes gradients of the target class score with respect to the feature map.
4. It weights each feature map channel by its average gradient importance.
5. It applies a ReLU to keep positive influence regions (lesions, hemorrhages, exudates).
6. It resizes the heatmap to match the original fundus image and blends it as a colored overlay.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        """
        Args:
            model: PyTorch classification model (e.g., ResNet-50).
            target_layer: Target conv layer for Grad-CAM. Defaults to model.layer4[-1] if None.
        """
        self.model = model
        self.model.eval()
        
        # If target layer is not explicitly provided, find final convolutional layer in ResNet-50
        if target_layer is None:
            if hasattr(model, "layer4"):
                target_layer = model.layer4[-1]
            elif hasattr(model, "features"):
                target_layer = model.features[-1]
            else:
                # Fallback: search for last Conv2d layer
                conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                if conv_layers:
                    target_layer = conv_layers[-1]
                else:
                    raise ValueError("Could not automatically locate a Convolutional layer for Grad-CAM.")
                    
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Hook to capture target layer output activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Hook to capture gradients flowing back to target layer."""
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM heatmap for a specified class index.
        
        Args:
            input_tensor: PyTorch input tensor (1, 3, H, W).
            class_idx: Integer target class index. If None, targets top predicted class.
            
        Returns:
            heatmap: 2D numpy array (H, W) with values normalized between 0.0 and 1.0.
        """
        self.model.zero_grad()
        
        # Enable gradient computation on input tensor pass for backpropagation
        tensor_req = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(tensor_req)
        
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()
            
        # Target score for backpropagation
        score = outputs[0, class_idx]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            # Fallback if hooks were not triggered properly
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5

        # Global average pooling of gradients: weights alpha_k
        gradients = self.gradients.cpu().data.numpy()[0]  # Shape: (C, H_conv, W_conv)
        activations = self.activations.cpu().data.numpy()[0]  # Shape: (C, H_conv, W_conv)
        
        weights = np.mean(gradients, axis=(1, 2))  # Shape: (C,)
        
        # Weighted combination of forward activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to keep only positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)
            
        # Resize heatmap to match input image spatial dimensions (H, W)
        img_h, img_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam, (img_w, img_h))
        
        return cam_resized


def generate_gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image_rgb: np.ndarray,
    target_class_idx: int = None,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
):
    """
    Generates Grad-CAM visual output including raw heatmap, colorized heatmap, and blended overlay.
    
    Args:
        model: PyTorch trained model.
        input_tensor: PyTorch normalized input tensor (1, 3, H, W).
        original_image_rgb: Processed RGB image array (H, W, 3) in range [0, 255].
        target_class_idx: Target class index to explain.
        alpha: Blending ratio (0.0 = original image only, 1.0 = heatmap only).
        colormap: OpenCV colormap code (e.g. cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS).
        
    Returns:
        heatmap_2d: 2D float array (0.0 to 1.0).
        heatmap_color_rgb: RGB colorized heatmap image.
        overlay_rgb: RGB blended heatmap overlay image.
    """
    cam_generator = GradCAM(model)
    heatmap_2d = cam_generator.generate_heatmap(input_tensor, target_class_idx)
    
    # Scale heatmap to [0, 255] uint8
    heatmap_uint8 = np.uint8(255 * heatmap_2d)
    
    # Apply OpenCV color map (returns BGR)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize original image to match heatmap dimensions if needed
    h, w = heatmap_2d.shape
    if original_image_rgb.shape[:2] != (h, w):
        original_image_rgb = cv2.resize(original_image_rgb, (w, h))
        
    # Create blended overlay: overlay = alpha * heatmap + (1 - alpha) * original
    overlay_bgr = cv2.addWeighted(
        cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR),
        1.0 - alpha,
        heatmap_bgr,
        alpha,
        0
    )
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    return heatmap_2d, heatmap_color_rgb, overlay_rgb


def get_heatmap_focus_summary(heatmap_2d: np.ndarray) -> str:
    """
    Analyzes spatial density of heatmap activations and provides human-readable summary.
    """
    high_activation_ratio = np.mean(heatmap_2d > 0.6) * 100
    peak_intensity = np.max(heatmap_2d) * 100
    
    if high_activation_ratio > 25:
        pattern = "Diffuse / Widespread"
        clinical_note = "High model focus detected across multiple sectors of the retina, consistent with widespread pathology or vascular changes."
    elif high_activation_ratio > 10:
        pattern = "Focal / Regional"
        clinical_note = "Moderate focal model concentration around localized fundus structures (potential exudates, microaneurysms, or micro-hemorrhages)."
    else:
        pattern = "Isolated / Low Density"
        clinical_note = "Model attention is low or distributed neutrally, typical for healthy retina or very subtle lesions."
        
    return f"**Attention Pattern:** {pattern} (Peak Intensity: {peak_intensity:.1f}%, High-Attention Area: {high_activation_ratio:.1f}% of image).\n\n{clinical_note}"
