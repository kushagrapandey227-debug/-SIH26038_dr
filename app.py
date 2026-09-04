"""
app.py - Tele-EYE DR: Explainable AI for Diabetic Retinopathy Screening
-----------------------------------------------------------------------
Streamlit web application for SIH26038 prototype demonstration.
Features:
- Retinal fundus image upload & pre-loaded sample images.
- Ben Graham fundus image contrast enhancement.
- ResNet-50 deep learning model prediction with confidence score.
- Grad-CAM Explainable AI (XAI) heatmap visualization overlay.
- Comprehensive clinical guidance & mandatory medical disclaimer.
"""

import os
import streamlit as st
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import custom modules
import model as dr_model
import explain as dr_explain

# ---------------------------------------------------------
# Ensure Sample Images Exist at Startup
# ---------------------------------------------------------
SAMPLES_DIR = "samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

if not os.path.exists(os.path.join(SAMPLES_DIR, "sample_normal.jpg")):
    try:
        import sample_generator
        sample_generator.generate_sample_images(output_dir=SAMPLES_DIR)
    except Exception as err:
        print(f"Sample generation warning: {err}")

# ---------------------------------------------------------
# Page Configuration & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tele-EYE DR | Explainable AI Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and responsive design
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .badge-normal { background-color: #10B981; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-mild { background-color: #3B82F6; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-moderate { background-color: #F59E0B; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-severe { background-color: #EF4444; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-proliferative { background-color: #881337; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 5px solid #DC2626;
        padding: 1rem;
        border-radius: 8px;
        color: #991B1B;
        font-size: 0.95rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model Loading (Cached to avoid reloading on user interactions)
# ---------------------------------------------------------
@st.cache_resource
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = dr_model.load_model(device=device)
    return net, device

model, device = get_model()


# ---------------------------------------------------------
# Sidebar Setup
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/eye-checked.png", width=70)
st.sidebar.title("Tele-EYE DR")
st.sidebar.caption("SIH26038 Prototype | Rural Telemedicine XAI")

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Sample Retinal Fundus Images")
sidebar_sample = st.sidebar.radio(
    "Choose a test sample image:",
    [
        "Upload Custom Image",
        "Sample 1: Normal Retina (Grade 0)",
        "Sample 2: Mild DR (Grade 1)",
        "Sample 3: Severe DR (Grade 3)"
    ],
    help="Select a synthetic fundus image to test predictions immediately without uploading your own file."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Screening Settings")

# Enhancement option
apply_ben_graham = st.sidebar.checkbox(
    "Apply Ben Graham Preprocessing",
    value=True,
    help="Enhances fundus contrast by removing local mean illumination (useful for low-cost rural fundus cameras)."
)

# Grad-CAM opacity slider
cam_alpha = st.sidebar.slider(
    "Grad-CAM Heatmap Opacity",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Adjust blending ratio between original fundus image and XAI heatmap."
)

st.sidebar.markdown("---")
st.sidebar.info("""
**APTOS / ICDR Scale Summary:**
- **0: No DR**: Healthy fundus
- **1: Mild**: Microaneurysms only
- **2: Moderate**: Exudates & Hemorrhages
- **3: Severe**: Extensive hemorrhages
- **4: Proliferative**: Neovascularization
""")


# ---------------------------------------------------------
# Header & Introduction
# ---------------------------------------------------------
st.markdown("<div class='main-header'>👁️ Tele-EYE DR: Explainable AI for Diabetic Retinopathy</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A decision-support prototype tailored for primary health centers and telemedicine units in rural India (SIH Problem Statement SIH26038).</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Image Input Handling (Upload or Sample selection)
# ---------------------------------------------------------
st.markdown("### 📤 Step 1: Input Retinal Image")
col_upload, col_sample_buttons = st.columns([1.2, 1])

uploaded_file = None
pil_image = None
selected_sample_path = None

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Retinal Fundus Image (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        help="Upload a retinal fundus photograph taken with a fundus camera."
    )

with col_sample_buttons:
    st.write("**Or Click to Load a Demo Sample Image:**")
    btn_normal = st.button("🟢 Load Normal Retina", use_container_width=True)
    btn_mild = st.button("🔵 Load Mild DR Sample", use_container_width=True)
    btn_severe = st.button("🔴 Load Severe DR Sample", use_container_width=True)
    
    if btn_normal:
        selected_sample_path = "samples/sample_normal.jpg"
        st.session_state["active_sample"] = "Sample 1: Normal Retina"
    elif btn_mild:
        selected_sample_path = "samples/sample_mild_dr.jpg"
        st.session_state["active_sample"] = "Sample 2: Mild DR"
    elif btn_severe:
        selected_sample_path = "samples/sample_severe_dr.jpg"
        st.session_state["active_sample"] = "Sample 3: Severe DR"

# Check sidebar selection if main buttons were not clicked
if selected_sample_path is None:
    if sidebar_sample == "Sample 1: Normal Retina (Grade 0)":
        selected_sample_path = "samples/sample_normal.jpg"
    elif sidebar_sample == "Sample 2: Mild DR (Grade 1)":
        selected_sample_path = "samples/sample_mild_dr.jpg"
    elif sidebar_sample == "Sample 3: Severe DR (Grade 3)":
        selected_sample_path = "samples/sample_severe_dr.jpg"

# Resolution logic
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.success(f"📁 Custom image uploaded: **{uploaded_file.name}**")
elif selected_sample_path is not None and os.path.exists(selected_sample_path):
    pil_image = Image.open(selected_sample_path).convert("RGB")
    st.info(f"🧪 Demo sample loaded: **{os.path.basename(selected_sample_path)}**")

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
if pil_image is not None:
    # 1. Preprocessing
    input_tensor, display_np = dr_model.preprocess_image(pil_image, apply_enhancement=apply_ben_graham)
    
    # 2. Prediction Inference
    results = dr_model.predict_dr(model, input_tensor, device=device)
    
    # 3. Grad-CAM Explanation
    heatmap_2d, heatmap_color, overlay_rgb = dr_explain.generate_gradcam_overlay(
        model,
        input_tensor,
        display_np,
        target_class_idx=results["class_id"],
        alpha=cam_alpha
    )
    
    st.markdown("---")
    
    # ---------------------------------------------------------
    # Results Dashboard Layout
    # ---------------------------------------------------------
    col_results, col_probs = st.columns([1.2, 1])
    
    with col_results:
        st.subheader("📋 Screening Assessment")
        
        # Badge rendering based on severity
        badge_style_map = {
            "No DR": "badge-normal",
            "Mild DR": "badge-mild",
            "Moderate DR": "badge-moderate",
            "Severe DR": "badge-severe",
            "Proliferative DR": "badge-proliferative"
        }
        badge_class = badge_style_map.get(results["class_name"], "badge-normal")
        
        st.markdown(f"""
        <div class="card">
            <h3>Prediction: <span class="{badge_class}">{results['class_name']}</span></h3>
            <p><strong>Severity Level:</strong> {results['severity']} (Grade {results['class_id']})</p>
            <p><strong>Model Confidence:</strong> <span style="font-size:1.4rem; font-weight:700; color:#1E40AF;">{results['confidence']:.1f}%</span></p>
            <hr/>
            <p><strong>Clinical Summary:</strong> {results['description']}</p>
            <p><strong>Recommended Action:</strong> <em>{results['recommendation']}</em></p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_probs:
        st.subheader("📊 Class Probability Distribution")
        probs = results["probabilities"]
        
        # Display horizontal progress bars for all 5 grades
        for class_name, prob_val in probs.items():
            st.write(f"**{class_name}**: {prob_val:.1f}%")
            st.progress(min(int(prob_val), 100))
            
    # ---------------------------------------------------------
    # Explainable AI (XAI) Visualizer
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🔍 Explainable AI (Grad-CAM Heatmap)")
    st.caption("Grad-CAM highlights specific retinal regions that influenced the model's prediction score.")
    
    cam_col1, cam_col2, cam_col3 = st.columns(3)
    
    with cam_col1:
        st.markdown("##### 1. Processed Retinal Image")
        st.image(display_np, use_container_width=True, caption="Preprocessed Fundus")
        
    with cam_col2:
        st.markdown("##### 2. XAI Activation Heatmap")
        st.image(heatmap_color, use_container_width=True, caption="Grad-CAM Activation Map")
        
    with cam_col3:
        st.markdown("##### 3. Blended Overlay")
        st.image(overlay_rgb, use_container_width=True, caption=f"Overlay (Alpha={cam_alpha:.2f})")
        
    # XAI Clinical Interpretation Box
    xai_summary = dr_explain.get_heatmap_focus_summary(heatmap_2d)
    st.info(f"💡 **XAI Feature Analysis:**\n\n{xai_summary}")

else:
    # Placeholder state when no image is loaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; border: 2px dashed #CBD5E1; border-radius: 12px; margin-top: 1rem;">
        <img src="https://img.icons8.com/illustrations/120/000000/microscope.png" style="margin-bottom:1rem;" />
        <h3 style="color:#475569;">No Retinal Image Loaded</h3>
        <p style="color:#64748B;">Please upload a fundus photograph above or click one of the demo sample buttons to begin screening.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Mandatory Medical Disclaimer Footer
# ---------------------------------------------------------
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>MANDATORY MEDICAL DISCLAIMER:</strong><br/>
    This application is an artificial intelligence prototype developed for research, demonstration, and decision-support purposes under Smart India Hackathon (SIH26038). 
    It is <strong>NOT</strong> a certified medical device and must <strong>NOT</strong> be used for definitive diagnostic or clinical treatment decisions. 
    All retinal screenings must be evaluated and confirmed by a licensed ophthalmologist or qualified healthcare provider.
</div>
""", unsafe_allow_html=True)
