"""
app.py - Tele-EYE DR: AI-Assisted Diabetic Retinopathy Screening & Hindi Voice Assistant
-----------------------------------------------------------------------------------------
Smart India Hackathon (SIH26038) Prototype Demonstration.
Features:
1. Professional 4-step screening workflow (Upload -> Quality Check -> Analyze -> Report).
2. Deep learning DR severity prediction (PyTorch ResNet-50 baseline).
3. Grad-CAM Explainable AI (XAI) heatmap visualizer.
4. Fundus image quality check (resolution, exposure, blurriness).
5. Downloadable clinical screening report.
6. Rural Accessibility Hindi Voice Assistant (Web Speech API TTS + Speech Input + Quick Commands).
7. Mandatory medical disclaimers & bilingual support (English / हिंदी).
"""

import os
import io
import time
import datetime
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import numpy as np
import torch

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
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tele-EYE DR | AI Retinal Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Clinical Visual System & Custom CSS Styling
st.markdown("""
<style>
    /* Main Layout Aesthetics */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header Card */
    .hero-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #0D9488 100%);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.15);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        opacity: 0.92;
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }
    .hero-tag {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        display: inline-block;
    }
    
    /* Section Step Cards */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    .step-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Severity Badges */
    .badge-normal { background-color: #10B981; color: white; padding: 5px 14px; border-radius: 20px; font-weight: 700; }
    .badge-mild { background-color: #3B82F6; color: white; padding: 5px 14px; border-radius: 20px; font-weight: 700; }
    .badge-moderate { background-color: #F59E0B; color: white; padding: 5px 14px; border-radius: 20px; font-weight: 700; }
    .badge-severe { background-color: #EF4444; color: white; padding: 5px 14px; border-radius: 20px; font-weight: 700; }
    .badge-proliferative { background-color: #881337; color: white; padding: 5px 14px; border-radius: 20px; font-weight: 700; }

    /* Result Metric Card */
    .result-card {
        background: #F1F5F9;
        border-left: 6px solid #2563EB;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* Voice Assistant Box */
    .voice-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #F0FDFA 100%);
        border: 2px solid #60A5FA;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .voice-title {
        color: #1E40AF;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 5px solid #DC2626;
        padding: 1.1rem 1.4rem;
        border-radius: 10px;
        color: #991B1B;
        font-size: 0.95rem;
        margin-top: 2rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

if "active_sample" not in st.session_state:
    st.session_state["active_sample"] = None

if "case_id" not in st.session_state:
    st.session_state["case_id"] = f"CASE-{datetime.datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"

if "assistant_speech" not in st.session_state:
    st.session_state["assistant_speech"] = "नमस्ते! मैं Tele-EYE DR सहायक हूँ। अपनी आँख की फोटो अपलोड करें या डेमो सैंपल चुनें।"


# ---------------------------------------------------------
# Cached PyTorch Model Loading
# ---------------------------------------------------------
@st.cache_resource
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = dr_model.load_model(device=device)
    return net, device

model, device = get_model()


# ---------------------------------------------------------
# Sidebar Controls & Settings
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/eye-checked.png", width=70)
st.sidebar.title("Tele-EYE DR")
st.sidebar.caption("SIH26038 Prototype | Telemedicine XAI")

# Bilingual Selector
st.sidebar.markdown("---")
selected_lang = st.sidebar.selectbox(
    "🌐 Select Language / भाषा चुनें:",
    options=["English", "हिंदी"],
    index=0 if st.session_state["lang"] == "en" else 1
)
st.session_state["lang"] = "en" if selected_lang == "English" else "hi"
is_hi = st.session_state["lang"] == "hi"

# Sidebar Demo Samples
st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ " + ("नमूना रेटिना फोटो" if is_hi else "Sample Fundus Images"))

sidebar_sample = st.sidebar.radio(
    "Choose a demo image:" if not is_hi else "नमूना फोटो चुनें:",
    [
        "Upload Custom Image" if not is_hi else "अपनी फोटो अपलोड करें",
        "Sample 1: Normal Retina (Grade 0)" if not is_hi else "सैंपल 1: सामान्य रेटिना (ग्रेड 0)",
        "Sample 2: Mild DR (Grade 1)" if not is_hi else "सैंपल 2: हल्का DR (ग्रेड 1)",
        "Sample 3: Severe DR (Grade 3)" if not is_hi else "सैंपल 3: गंभीर DR (ग्रेड 3)"
    ],
    help="Quick test images for immediate demonstration."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ " + ("स्क्रीनिंग सेटिंग्स" if is_hi else "Screening Settings"))

apply_ben_graham = st.sidebar.checkbox(
    "Apply Ben Graham Preprocessing" if not is_hi else "बेन ग्राहम कंट्रास्ट फ़िल्टर लागू करें",
    value=True,
    help="Enhances fundus illumination and lesion contrast."
)

cam_alpha = st.sidebar.slider(
    "Grad-CAM Heatmap Opacity" if not is_hi else "XAI हीटमैप पारदर्शिता",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Adjust overlay blending ratio."
)

st.sidebar.markdown("---")
st.sidebar.info("""
**APTOS / ICDR Scale:**
- **0: No DR**: Healthy retina
- **1: Mild**: Microaneurysms
- **2: Moderate**: Exudates & Hemorrhages
- **3: Severe**: Widespread Hemorrhages
- **4: Proliferative**: Neovascularization
""")


# ---------------------------------------------------------
# Header Banner Section
# ---------------------------------------------------------
title_text = "Tele-EYE DR: टेली-आई डीआर" if is_hi else "Tele-EYE DR"
subtitle_text = "ग्रामीण स्वास्थ्य केंद्रों के लिए एआई-आधारित डायबिटिक रेटिनोपैथी स्क्रीनिंग सहायता" if is_hi else "AI-Assisted Diabetic Retinopathy Screening for Rural Primary Healthcare"

st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">👁️ {title_text}</div>
    <div class="hero-subtitle">{subtitle_text}</div>
    <div>
        <span class="hero-tag">🎯 SIH Problem Statement SIH26038</span>
        <span class="hero-tag">✨ AI Screening Prototype</span>
        <span class="hero-tag">🌾 Rural Accessibility (Hindi Voice)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 🎙️ HINDI VOICE ASSISTANT SECTION
# ---------------------------------------------------------
st.markdown(f"""
<div class="voice-box">
    <div class="voice-title">🎙️ { 'ग्रामीण उपयोगकर्ताओं के लिए हिंदी आवाज़ सहायक (Hindi Voice Assistant)' if is_hi else 'Hindi Voice Assistant for Rural Accessibility' }</div>
    <p style="color:#334155; margin-bottom:0.8rem;">
        { '“नमस्ते! मैं Tele-EYE DR सहायक हूँ। मैं आपको आँखों की जाँच की प्रक्रिया समझाने में मदद कर सकता हूँ।”' if is_hi else '"Hello! I am your Tele-EYE DR Assistant. I can guide you through the retinal screening process in simple Hindi."' }
    </p>
</div>
""", unsafe_allow_html=True)

# Voice Controls & Quick Command Buttons
v_col1, v_col2 = st.columns([1.2, 1])

with v_col1:
    st.write("**" + ("त्वरित आवाज़ निर्देश (Quick Commands):" if is_hi else "Quick Voice Commands:") + "**")
    q1, q2, q3 = st.columns(3)
    btn_q1 = q1.button("🖼️ " + ("फोटो निर्देश" if is_hi else "Upload Info"), use_container_width=True)
    btn_q2 = q2.button("🔍 " + ("जाँच निर्देश" if is_hi else "Analysis Info"), use_container_width=True)
    btn_q3 = q3.button("👨‍⚕️ " + ("डॉक्टर सलाह" if is_hi else "Doctor Advice"), use_container_width=True)

with v_col2:
    st.write("**" + ("वॉइस असिस्टेंट स्थिति:" if is_hi else "Voice Speech Output:") + "**")
    if btn_q1:
        st.session_state["assistant_speech"] = "अपनी आँख की रेटिना की फोटो अपलोड करें या नीचे दिए गए बटन से सैंपल चुनें।"
    elif btn_q2:
        st.session_state["assistant_speech"] = "फोटो लोड होने के बाद 'Analyze Retinal Image' बटन दबाएं।"
    elif btn_q3:
        st.session_state["assistant_speech"] = "यदि परिणाम में कोई समस्या पाई जाती है, तो कृपया तुरंत आँखों के डॉक्टर से संपर्क करें।"

    st.info(f"🔊 **{st.session_state['assistant_speech']}**")

# HTML5 Web Speech API Component (Natively browser-supported TTS in Hindi)
speech_js_code = f"""
<script>
function speakHindi(text) {{
    if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        var utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'hi-IN';
        utterance.rate = 0.95;
        window.speechSynthesis.speak(utterance);
    }}
}}
</script>
<div style="display:flex; gap:10px; margin-top:5px;">
    <button onclick="speakHindi('{st.session_state['assistant_speech']}')" style="background:#2563EB; color:white; border:none; padding:8px 16px; border-radius:6px; font-weight:600; cursor:pointer;">
        🔊 { 'सुनें / Read Aloud' if is_hi else 'Listen in Hindi' }
    </button>
    <button onclick="window.speechSynthesis.cancel()" style="background:#64748B; color:white; border:none; padding:8px 16px; border-radius:6px; font-weight:600; cursor:pointer;">
        ⏹️ { 'रोकें / Stop' if is_hi else 'Stop Voice' }
    </button>
</div>
"""
components.html(speech_js_code, height=50)


st.markdown("---")

# ---------------------------------------------------------
# STEP 1: UPLOAD RETINAL IMAGE
# ---------------------------------------------------------
st.markdown(f"### 📤 Step 1: { 'रेटिना की फोटो अपलोड करें' if is_hi else 'Upload Retinal Image' }")

col_up, col_sample = st.columns([1.2, 1])
selected_sample_path = None
uploaded_file = None
pil_image = None
filename_str = "None"

with col_up:
    uploaded_file = st.file_uploader(
        "Select Retinal Fundus Photo (.jpg, .jpeg, .png)" if not is_hi else "रेटिना फंडस फोटो चुनें (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        help="Supports standard fundus camera outputs."
    )

with col_sample:
    st.write("**" + ("या डेमो सैंपल फोटो लोड करें:" if is_hi else "Or Click to Load a Demo Sample:") + "**")
    b_norm = st.button("🟢 Load Normal Retina" if not is_hi else "🟢 सामान्य रेटिना (Grade 0)", use_container_width=True)
    b_mild = st.button("🔵 Load Mild DR Sample" if not is_hi else "🔵 हल्का DR (Grade 1)", use_container_width=True)
    b_sev = st.button("🔴 Load Severe DR Sample" if not is_hi else "🔴 गंभीर DR (Grade 3)", use_container_width=True)
    
    if b_norm:
        selected_sample_path = "samples/sample_normal.jpg"
    elif b_mild:
        selected_sample_path = "samples/sample_mild_dr.jpg"
    elif b_sev:
        selected_sample_path = "samples/sample_severe_dr.jpg"

# Sidebar fallback check
if selected_sample_path is None:
    if "Normal" in sidebar_sample:
        selected_sample_path = "samples/sample_normal.jpg"
    elif "Mild" in sidebar_sample:
        selected_sample_path = "samples/sample_mild_dr.jpg"
    elif "Severe" in sidebar_sample:
        selected_sample_path = "samples/sample_severe_dr.jpg"

# Resolve Image Source
if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        filename_str = uploaded_file.name
        st.success(f"📁 Custom image loaded: **{filename_str}**")
        st.session_state["assistant_speech"] = "फोटो अपलोड हो गई है। अब आप Step 3 में 'Analyze Retinal Image' बटन दबाकर जाँच शुरू कर सकते हैं।"
    except Exception as e:
        st.error(f"Invalid image file uploaded: {e}")
elif selected_sample_path and os.path.exists(selected_sample_path):
    try:
        pil_image = Image.open(selected_sample_path).convert("RGB")
        filename_str = os.path.basename(selected_sample_path)
        st.info(f"🧪 Demo sample loaded: **{filename_str}**")
        st.session_state["assistant_speech"] = f"डेमो सैंपल {filename_str} लोड हो गया है। जाँच के लिए 'Analyze Retinal Image' बटन दबाएं।"
    except Exception as e:
        st.error(f"Could not load sample image: {e}")


# ---------------------------------------------------------
# STEP 2 & MAIN PIPELINE EXECUTION
# ---------------------------------------------------------
if pil_image is not None:
    # ---------------------------------------------------------
    # STEP 2: IMAGE QUALITY CHECK
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### 📋 Step 2: { 'इमेज गुणवत्ता एवं प्री-प्रोसेसिंग जाँच' if is_hi else 'Image Quality Check & Preprocessing' }")
    
    quality = dr_model.check_image_quality(pil_image)
    
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    q_col1.metric("Resolution / रिज़ॉल्यूशन", f"{quality['width']}x{quality['height']} px")
    q_col2.metric("Brightness / चमक", f"{quality['mean_brightness']} / 255")
    q_col3.metric("Sharpness / तीक्ष्णता", f"{quality['sharpness']}")
    q_col4.metric("Quality Pass / गुणवत्ता स्थिति", "✅ Pass" if quality["passed"] else "⚠️ Warning")
    
    if not quality["passed"]:
        st.warning("⚠️ **Image quality may be insufficient for reliable screening.** Please upload a clearer retinal image.")
        for w in quality["warnings"]:
            st.write(f"- ⚠️ {w}")
    else:
        st.success("✅ Retinal image meets resolution and contrast requirements for AI analysis.")

    # ---------------------------------------------------------
    # STEP 3: ANALYZE & VIEW RESULTS
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### 🔍 Step 3: { 'जाँच शुरू करें एवं परिणाम देखें' if is_hi else 'Analyze Image & View Screening Result' }")
    
    btn_analyze = st.button("🚀 Analyze Retinal Image / रेटिना की जाँच करें", type="primary", use_container_width=True)
    
    # Run Inference on click or store in session state
    if btn_analyze or "last_results" in st.session_state:
        if btn_analyze:
            with st.spinner("Running PyTorch ResNet-50 AI inference & generating Grad-CAM XAI map..."):
                time.sleep(0.4)
                input_tensor, display_np = dr_model.preprocess_image(pil_image, apply_enhancement=apply_ben_graham)
                results = dr_model.predict_dr(model, input_tensor, device=device)
                heatmap_2d, heatmap_color, overlay_rgb = dr_explain.generate_gradcam_overlay(
                    model, input_tensor, display_np, target_class_idx=results["class_id"], alpha=cam_alpha
                )
                
                st.session_state["last_results"] = results
                st.session_state["last_display_np"] = display_np
                st.session_state["last_heatmap_color"] = heatmap_color
                st.session_state["last_overlay_rgb"] = overlay_rgb
                st.session_state["last_heatmap_2d"] = heatmap_2d
                
                # Context aware speech update
                st.session_state["assistant_speech"] = f"जाँच पूरी हो गई है। परिणाम {results['class_name']} (ग्रेड {results['class_id']}) पाया गया है। मॉडल का विश्वास {results['confidence']:.1f}% है।"

        # Retrieve processed state
        results = st.session_state["last_results"]
        display_np = st.session_state["last_display_np"]
        heatmap_color = st.session_state["last_heatmap_color"]
        overlay_rgb = st.session_state["last_overlay_rgb"]
        heatmap_2d = st.session_state["last_heatmap_2d"]
        
        # Display Results Dashboard Cards
        col_res_card, col_prob_card = st.columns([1.2, 1])
        
        with col_res_card:
            badge_style_map = {
                "No DR": "badge-normal",
                "Mild DR": "badge-mild",
                "Moderate DR": "badge-moderate",
                "Severe DR": "badge-severe",
                "Proliferative DR": "badge-proliferative"
            }
            b_class = badge_style_map.get(results["class_name"], "badge-normal")
            
            st.markdown(f"""
            <div class="result-card">
                <h2 style="margin:0 0 0.5rem 0; color:#1E293B;">Screening Result: <span class="{b_class}">{results['class_name']}</span></h2>
                <p style="font-size:1.1rem; margin:0.3rem 0;"><strong>Predicted DR Grade:</strong> Grade {results['class_id']} ({results['severity']})</p>
                <p style="font-size:1.2rem; margin:0.3rem 0; color:#1E40AF;"><strong>Confidence Score:</strong> <strong>{results['confidence']:.1f}%</strong></p>
                <hr style="margin:0.8rem 0;"/>
                <p style="margin:0.3rem 0;"><strong>Clinical Summary:</strong> {results['description']}</p>
                <p style="margin:0.3rem 0; color:#065F46;"><strong>Recommended Action:</strong> <em>{results['recommendation']}</em></p>
            </div>
            """, unsafe_allow_html=True)

        with col_prob_card:
            st.subheader("📊 Class Probability Breakdown")
            probs = results["probabilities"]
            for c_name, p_val in probs.items():
                st.write(f"**{c_name}**: {p_val:.1f}%")
                st.progress(min(int(p_val), 100))

        # Visual Result & Explainable AI (Grad-CAM)
        st.markdown("#### 🔍 Explainable AI (Grad-CAM Heatmap Visualization)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Processed Fundus Image**")
            st.image(display_np, use_container_width=True)
        with c2:
            st.markdown("**2. Grad-CAM Activation Heatmap**")
            st.image(heatmap_color, use_container_width=True)
        with c3:
            st.markdown(f"**3. Blended Overlay (Alpha={cam_alpha:.2f})**")
            st.image(overlay_rgb, use_container_width=True)

        xai_summary = dr_explain.get_heatmap_focus_summary(heatmap_2d)
        st.info(f"💡 **XAI Feature Analysis:**\n\n{xai_summary}")

        # ---------------------------------------------------------
        # STEP 4: SCREENING REPORT & DOWNLOAD
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown(f"### 📋 Step 4: { 'स्क्रीनिंग रिपोर्ट एवं डाउनलोड' if is_hi else 'Screening Report & Download' }")
        
        report_text = f"""==================================================
           TELE-EYE DR SCREENING REPORT
==================================================
Case / Patient ID : {st.session_state['case_id']}
Date & Time        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image Filename     : {filename_str}

--------------------------------------------------
SCREENING ASSESSMENT
--------------------------------------------------
Predicted Condition : {results['class_name']}
Severity Grade      : Grade {results['class_id']} ({results['severity']})
Model Confidence    : {results['confidence']:.1f}%

Clinical Summary    : {results['description']}
Recommended Action  : {results['recommendation']}

--------------------------------------------------
EXPLAINABLE AI (XAI) ATTENTION
--------------------------------------------------
{xai_summary.replace('**', '')}

--------------------------------------------------
RECOMMENDATION & DISCLAIMER
--------------------------------------------------
Please consult a qualified ophthalmologist for clinical evaluation.
This AI system is intended for screening assistance and does not replace
professional diagnosis.
=================================================="""

        st.code(report_text, language="text")
        
        st.download_button(
            label="📥 Download Screening Report (TXT)",
            data=report_text,
            file_name=f"tele_eye_dr_report_{st.session_state['case_id']}.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    # Placeholder empty state
    st.markdown(f"""
    <div style="text-align: center; padding: 3.5rem; border: 2px dashed #CBD5E1; border-radius: 14px; background: white; margin-top: 1rem;">
        <img src="https://img.icons8.com/illustrations/120/000000/microscope.png" style="margin-bottom:1rem;" />
        <h3 style="color:#334155;">{ 'कोई रेटिना फोटो अपलोड नहीं की गई' if is_hi else 'No Retinal Fundus Image Loaded' }</h3>
        <p style="color:#64748B;">{ 'कृपया ऊपर दिए गए बटन से अपनी आँख की फोटो अपलोड करें या एक नमूना फोटो चुनें।' if is_hi else 'Please upload a fundus photograph above or click one of the demo sample buttons to begin screening.' }</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# ABOUT THE PROJECT (EXPANDABLE SIH SECTION)
# ---------------------------------------------------------
st.markdown("---")
with st.expander("ℹ️ About the Project / परियोजना की जानकारी (SIH Problem Statement SIH26038)"):
    st.markdown("""
    **Problem Statement**: Smart Scan Strategy for Diabetic Retinopathy Screening in Rural India (SIH26038).
    
    - **Context**: Rural telemedicine centers often lack resident ophthalmologists. Early screening of diabetic patients prevents irreversible blindness.
    - **Solution**: **Tele-EYE DR** provides automated, explainable screening (Grad-CAM heatmaps) to assist healthcare workers and primary health center (PHC) technicians.
    - **Rural Accessibility**: Includes a bilingual Hindi voice assistant for patients and healthcare workers with limited English/digital literacy.
    - **Clinical Scope**: Research decision-support prototype. Requires confirmation by an ophthalmologist.
    """)


# ---------------------------------------------------------
# MANDATORY MEDICAL DISCLAIMER
# ---------------------------------------------------------
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>MANDATORY MEDICAL DISCLAIMER / चिकित्सा अस्वीकरण:</strong><br/>
    Tele-EYE DR is an AI-assisted screening prototype intended for research, demonstration, and decision-support purposes under Smart India Hackathon (SIH26038). 
    It does <strong>NOT</strong> provide a definitive medical diagnosis and should <strong>NOT</strong> replace evaluation by a qualified healthcare professional or ophthalmologist.
    <br/>
    <em>(यह केवल AI-आधारित स्क्रीनिंग सहायक है। अंतिम निदान के लिए योग्य नेत्र चिकित्सक से परामर्श आवश्यक है।)</em>
</div>
""", unsafe_allow_html=True)

