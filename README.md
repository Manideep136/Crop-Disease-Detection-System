# Crop-Disease-Detection-System
<div align="center">

# 🌾 CROP DISEASE DETECTION SYSTEM

### AI-Powered Plant Disease Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> Upload a leaf image → Get instant AI diagnosis → Know the disease, severity & treatment steps

<br/>


</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation & Setup](#-installation--setup)
- [How to Use](#-how-to-use)
- [App Pages](#-app-pages)
- [Supported Crops & Diseases](#-supported-crops--diseases)
- [Leaf Validation System](#-leaf-validation-system)
- [Results & Performance](#-results--performance)
- [Screenshots](#-screenshots)
- [Future Scope](#-future-scope)
- [Developer](#-developer)

---

## 🧠 Overview

The **Crop Disease Detection System** is a deep learning-powered web application that enables farmers, agronomists, and researchers to instantly identify plant diseases from leaf images. Built with TensorFlow and Streamlit, it classifies **38 distinct disease conditions across 14 crop species** with **98.7% validation accuracy**.

Unlike basic classifiers, this system also:
- **Rejects non-leaf images** using a pixel-level color validation engine
- Provides **disease severity ratings** (Critical / High / Moderate / Low)
- Displays **treatment steps**, **cause**, **symptoms**, and **prevention** for each detected disease
- Shows a **confidence score meter** so users know how reliable the diagnosis is

---

## 🔗 Live Demo

> 🌐 **[Launch App →](https://crop-disease-detection-system-md8siq49w46nfoudounw3t.streamlit.app/)**

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔬 **AI Diagnosis** | CNN-based classification across 38 disease classes |
| 🚫 **Leaf Validator** | Rejects non-plant images using HSV color analysis |
| 📊 **Confidence Meter** | Visual score bar showing how certain the model is |
| 💊 **Treatment Guide** | Step-by-step treatment for every detected disease |
| 🛡️ **Prevention Tips** | Seasonal and cultural prevention recommendations |
| 📚 **Disease Library** | Searchable reference for all 38 conditions |
| 🌿 **38+ Diseases** | Covers fungal, bacterial, viral, and pest conditions |
| ⚡ **Real-time** | Diagnosis delivered in under 5 seconds |
| 📱 **Responsive UI** | Works on desktop and mobile browsers |
| 🌙 **Dark Theme** | Professional dark UI designed for field use |

---

## 📁 Project Structure

```
CROP-DISEASE-DETECTION-SYSTEM/
│
├── app.py                    # Main Streamlit application
├── trained_model.keras       # Trained CNN model (TensorFlow SavedModel)
├── homeIMG.jpg               # Hero image for home page
├── logo3.png                 # App logo
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
└── README.md                 # Project documentation
```

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────────────┐
│                   CROP DISEASE DETECTION                │
├──────────────────┬──────────────────┬───────────────────┤
│   FRONTEND       │   BACKEND/ML     │   DEPLOYMENT      │
│                  │                  │                   │
│  Streamlit       │  TensorFlow 2.0  │  Streamlit Cloud  │
│  HTML/CSS        │  Keras           │  Local Server     │
│  Custom CSS      │  NumPy           │                   │
│                  │  Pillow (PIL)    │                   │
│                  │  colorsys        │                   │
└──────────────────┴──────────────────┴───────────────────┘
```

| Layer | Technology |
|---|---|
| **UI Framework** | Streamlit |
| **Deep Learning** | TensorFlow 2.0 + Keras |
| **Model Type** | Custom CNN (16-layer architecture) |
| **Image Processing** | PIL (Pillow), NumPy |
| **Leaf Validation** | colorsys (HSV color space analysis) |
| **Language** | Python 3.10+ |

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [New Plant Diseases Dataset — Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) |
| **Total Images** | 87,000+ labeled RGB images |
| **Training Set** | 70,295 images (80%) |
| **Validation Set** | 17,572 images (20%) |
| **Disease Classes** | 38 categories |
| **Input Resolution** | Resized to 128×128 pixels |
| **Augmentation** | Rotation, horizontal flip, zoom variations |

---

## 🧬 Model Architecture

```
Input Layer       → 128×128×3 RGB Image
Conv2D Block 1    → 32 filters, 3×3, ReLU + MaxPooling
Conv2D Block 2    → 64 filters, 3×3, ReLU + MaxPooling
Conv2D Block 3    → 128 filters, 3×3, ReLU + MaxPooling
Conv2D Block 4    → 256 filters, 3×3, ReLU + MaxPooling
Flatten           → Dense(1500) → Dropout(0.4)
Dense(1500)       → Dropout(0.4)
Output Layer      → Dense(38) + Softmax
```

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Epochs:** 50
- **Batch Size:** 32
- **Validation Accuracy:** 98.7%

---

## ⚙️ Installation & Setup

### Prerequisites

Make sure you have the following installed:
- Python 3.10 or higher
- pip package manager
- Git

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Manideep136/CROP-DISEASE-DETECTION-SYSTEM.git
cd CROP-DISEASE-DETECTION-SYSTEM
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the App

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:**

---

### `requirements.txt`

```
streamlit>=1.28.0
tensorflow>=2.13.0
numpy>=1.24.0
Pillow>=10.0.0
```

---

## 📖 How to Use

```
1. Open the app → Navigate to 🔬 Diagnose from the sidebar

2. Upload a leaf image (JPG / PNG / JPEG)
   └── Must be a clear, well-lit photo of a plant leaf

3. Click "Analyze Leaf" button

4. The system first validates your image:
   ├── ✅ Valid leaf image → runs disease classification
   └── 🚫 Non-leaf image → shows a helpful error with tips

5. View your results:
   ├── Disease name + plant species
   ├── Severity badge (Critical / High / Moderate / Low / Healthy)
   ├── Confidence score with visual meter
   ├── Tab 1: Disease Info (cause, symptoms)
   ├── Tab 2: Treatment steps (numbered action plan)
   └── Tab 3: Prevention strategy
```

---

## 📄 App Pages

### 🏠 Home
Hero section with key stats, supported crops, and a step-by-step guide on how the system works.

### 🔬 Diagnose
The main detection page. Upload a leaf image to get an instant AI-powered diagnosis with full disease details.

### 📚 Disease Library
Browse all 38 detectable conditions organized by crop. Includes search and filter functionality (Diseases Only / Healthy Only).

### ℹ️ About
Project overview, model performance metrics, dataset details, and technical architecture.

---

## 🌿 Supported Crops & Diseases

<details>
<summary><strong>Click to expand full disease list (38 classes)</strong></summary>

| Crop | Condition |
|------|-----------|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🌽 Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| 🍊 Orange | Huanglongbing (Citrus Greening) |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Bell Pepper | Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍓 Raspberry | Healthy |
| 🌱 Soybean | Healthy |
| 🎃 Squash | Powdery Mildew |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

</details>

---

## 🔍 Leaf Validation System

A unique feature that prevents the model from making false predictions on non-leaf images.

**How it works:**

```
1. Resize uploaded image to 100×100 pixels
2. Convert each pixel from RGB → HSV color space
3. Count pixels in green range  (hue 60°–165°) → healthy leaves
4. Count pixels in brown/tan range (hue 15°–55°) → diseased/dry leaves
5. Calculate plant_ratio = (green + brown pixels) / total pixels
6. If plant_ratio < 10% → REJECT with helpful error message
7. If plant_ratio ≥ 10% → PASS to disease classification model
```

This ensures users always receive meaningful, accurate results rather than incorrect disease predictions on random images.

---

## 📈 Results & Performance

| Metric | Value |
|---|---|
| **Validation Accuracy** | 98.7% |
| **Training Accuracy** | ~99.2% |
| **Total Parameters** | ~5.2M |
| **Inference Time** | < 5 seconds |
| **Model Size** | ~63 MB (.keras format) |
| **Supported Classes** | 38 |

---

## 📸 Screenshots

> Add screenshots of your app here after deployment

| Home Page | Diagnosis Page | Disease Library |
|---|---|---|
| *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## 🚀 Future Scope

- [ ] 🌐 **Multi-language support** — Hindi, Telugu, Marathi for Indian farmers
- [ ] 📱 **Android/iOS app** using Flutter + TensorFlow Lite
- [ ] 🗺️ **Disease heatmap** — regional disease outbreak tracking
- [ ] 🌦️ **Weather integration** — disease risk prediction based on local climate
- [ ] 📷 **Real-time camera feed** — live leaf scanning without upload
- [ ] 🤖 **LLM chatbot** — conversational agronomist powered by Claude AI
- [ ] 🛒 **Pesticide recommendation** — direct product suggestions with dosage

---

## 👨‍💻 Developer

<div align="center">

**Manideep**

[![GitHub](https://img.shields.io/badge/GitHub-Manideep136-181717?style=for-the-badge&logo=github)](https://github.com/Manideep136)

*Built with ❤️ for Indian farmers and precision agriculture*

</div>

---

<div align="center">

**⭐ If this project helped you, please give it a star on GitHub!**


</div>
