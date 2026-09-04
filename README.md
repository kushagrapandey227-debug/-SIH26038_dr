# SIH26038: Explainable AI for Diabetic Retinopathy Screening in Rural India 👁️✨

An intuitive, beginner-friendly Python prototype for **Smart India Hackathon (Problem Statement SIH26038)**. 

This system enables primary health centers and telemedicine workers in rural India to upload retinal fundus images, receive instant AI-driven Diabetic Retinopathy (DR) screening predictions with confidence scores, and view **Grad-CAM Explainable AI (XAI)** heatmaps showing exactly which retinal lesions influenced the prediction.

---

## 🌟 Key Features

- **Pre-trained Deep Learning Model**: Uses PyTorch (ResNet-50 backbone) tuned for the standard 5-grade APTOS/ICDR Diabetic Retinopathy severity scale.
- **Explainable AI (Grad-CAM)**: Generates color activation heatmaps that highlight microaneurysms, hemorrhages, and exudates.
- **Ben Graham Contrast Enhancement**: Option to remove local mean background illumination—a standard pre-processing technique for fundus photography in rural settings.
- **Interactive Streamlit Dashboard**: Clean visual interface with side-by-side fundus vs. heatmap comparison, progress bars, confidence metrics, and sample test images.
- **Zero External Dataset Dependency**: Comes with a built-in synthetic fundus image generator so you can test the application immediately!
- **Medical Disclaimer**: Clear header and footer indicating that the tool is for research and decision-support only.

---

## 📂 Project Architecture

```
SIH26038_dr/
├── app.py                # Streamlit web dashboard interface & UI logic
├── model.py              # PyTorch model architecture, preprocessing & inference
├── explain.py            # Grad-CAM XAI engine & heatmap overlay blending
├── sample_generator.py   # Synthetic test fundus image generator
├── requirements.txt      # Required Python packages
└── README.md             # Windows setup instructions & project documentation
```

---

## 🖥️ Windows Setup Instructions (Step-by-Step)

Follow these simple steps in **Windows PowerShell** to set up and run the project on your machine.

### Step 1: Open PowerShell and Navigate to Project Directory
```powershell
cd e:\SIH26038_dr
```

### Step 2: Create a Python Virtual Environment
Creating a virtual environment ensures all libraries are isolated cleanly:
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
Allow script execution for the current session and activate the environment:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
*(You will see `(venv)` appear at the beginning of your terminal prompt).*

### Step 4: Install Required Packages
Install PyTorch, Streamlit, OpenCV, and other required libraries:
```powershell
pip install -r requirements.txt
```

### Step 5: Generate Sample Test Images
Run the sample generator script to create test fundus images in a `samples/` directory:
```powershell
python sample_generator.py
```

### Step 6: Launch the Streamlit Dashboard! 🚀
```powershell
streamlit run app.py
```
Streamlit will automatically open your default web browser to `http://localhost:8501`.

---

## 🩺 APTOS / ICDR Diabetic Retinopathy Scale

The model classifies fundus images into five clinical severity grades:

| Grade | Severity | Clinical Characteristics | Color Code |
|---|---|---|---|
| **0** | **No DR** | Healthy retina, no microaneurysms detected | 🟢 Green |
| **1** | **Mild DR** | Microaneurysms only | 🔵 Blue |
| **2** | **Moderate DR** | Microaneurysms, hemorrhages, or hard exudates | 🟡 Orange |
| **3** | **Severe DR** | Extensive intraretinal hemorrhages or cotton wool spots | 🔴 Red |
| **4** | **Proliferative DR** | Neovascularization (new abnormal blood vessels) | 🔴 Dark Red |

---

## 💡 How Grad-CAM Explainable AI Works

In rural telemedicine, healthcare workers and doctors need to trust AI predictions. **Grad-CAM (Gradient-weighted Class Activation Mapping)** provides transparency by showing a visual overlay:

1. **Red / Yellow Hotspots**: Spatial regions of the retina that strongly contributed to a high severity score (often pointing to exudates, hemorrhages, or microvascular abnormalities).
2. **Blue / Cool Areas**: Regions that had low influence on the DR classification score.

---

## ⚠️ Mandatory Medical Disclaimer

This application is an artificial intelligence prototype developed for research, demonstration, and decision-support purposes under Smart India Hackathon (SIH26038). It is **NOT** a certified medical device and must **NOT** be used for definitive diagnostic or clinical treatment decisions. All retinal screenings must be evaluated and confirmed by a licensed ophthalmologist.
