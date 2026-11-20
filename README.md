---
title: Pediatric Pneumonia AI
emoji: 🫁
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Pediatric Pneumonia Detection AI with Grad-CAM
---

# 🫁 Pediatric Pneumonia Detection AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Holdout Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-90.06%25-green.svg)]()
[![Clinical Triage Accuracy](https://img.shields.io/badge/Clinical%20Triage-93.03%25-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/Champion%20Model-ResNet50V2--Pro%20(Dual--Pooling)-purple.svg)]()

An end-to-end clinical decision support system for automated pediatric pneumonia detection from chest radiographs (X-rays). Built on a **Dual-Pooling ResNet50V2-Pro** architecture with on-the-fly thoracic center cropping, Grad-CAM visual explainability, and a **3-Tier Clinical Safety Triage System** ($L=0.65, U=0.88$).

---

## 📊 Dataset Specifications

The system is trained and evaluated on **5,856 pediatric chest radiographs** (children aged 1–5 years) sourced from Guangzhou Women and Children's Medical Center (Kaggle: `paultimothymooney/chest-xray-pneumonia`):

| Cohort | Image Count | Percentage | Radiographic Appearance |
| :--- | :---: | :---: | :--- |
| **PNEUMONIA** | **4,273** | **~73.0%** | Dense consolidations, peribronchial thickening, fluid opacity |
| **NORMAL** | **1,583** | **~27.0%** | Clear lung airspaces, sharp diaphragm margins, distinct ribs |
| **Total Cohort** | **5,856** | **100%** | Standard anterior-posterior pediatric radiographs |

### Stratified Data Splits (Fixing the 16-Image Validation Flaw)
The original Kaggle distribution contained only **16 images** in `val/` (too small for stable checkpointing). We pooled `train` and `val` (5,232 images) and generated a balanced **80/20 stratified split**, while keeping the 624-image holdout test set completely untouched:

| Split | Total Images | Normal Count | Pneumonia Count | Split Ratio | Clinical Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training** | **4,185** | 1,079 | 3,106 | 80% of pool | Two-stage training with class weighting |
| **Validation** | **1,047** | 270 | 777 | 20% of pool | Stable checkpointing & LR decay |
| **Holdout Test Set** | **624** | 234 | 390 | Untouched | Real-world clinical validation |
| **Total** | **5,856** | **1,583** | **4,273** | 100% | Full Guangzhou pediatric cohort |

---

## 🔬 Model Architecture: ResNet50V2-Pro (Dual Pooling)

```
Input Radiograph (240 x 240)
         │
         ▼
On-the-fly 10% Thoracic Center Crop (224 x 224 x 3)
[Physically eliminates ECG cables, text markers ('R'/'L'), and collarbone borders]
         │
         ▼
Pretrained ResNet50V2 Backbone (Identity Addition Skip Connections)
         │
         ├───────────────────────────────┐
         ▼                               ▼
GlobalAveragePooling2D          GlobalMaxPooling2D
(Captures diffuse fluid)        (Captures sharp focal lesions)
         │                               │
         └───────────────┬───────────────┘
                         ▼
             Concatenate (4,096 Dimensions)
                         │
                         ▼
             BatchNormalization
                         │
                         ▼
             Dense(384) + LeakyReLU(0.1) + Dropout(0.45) + L2(1e-4)
                         │
                         ▼
             Dense(128) + LeakyReLU(0.1) + Dropout(0.25) + L2(1e-4)
                         │
                         ▼
             Dense(1, activation='sigmoid')
```

### Why ResNet50V2 Beats DenseNet121 on Medical X-Rays:
- **DenseNet121** preserves high-frequency edge gradients via feature concatenation, causing it to cheat on border artifacts (ECG leads, armpit creases, and orientation stamps).
- **ResNet50V2** uses identity addition ($x_{l+1} = \mathcal{F}(x_l) + x_l$) which diffuses edge noise and prioritizes broad lung parenchymal textures.
- **Dual Pooling (Average + Max):** Produces a 4,096-dimensional vector capturing both diffuse haziness (viral pneumonia) and sharp focal spots (bacterial pneumonia).

---

## 🛡️ Clinical Safety Triage Protocol

Rather than forcing the model to make binary guesses on ambiguous scans, our 3-tier routing safely refers borderline cases to a human specialist:

```
Score:  0.00 ────────────────── 0.65 ════════════════════ 0.88 ────────────────── 1.00
Zone:   [   🟢 ROUTINE NORMAL   ] [ 🟡 REFER TO DOCTOR  ] [   🔴 PNEUMONIA DETECTED  ]
Action: Routine Outpatient Clear   Specialist Human Review    Immediate Antimicrobial Care
```

* **Score $< 0.65$:** 🟢 **NORMAL** (High-confidence clear lung airspaces $\rightarrow$ routine clearance).
* **Score $> 0.88$:** 🔴 **PNEUMONIA** (High-confidence consolidation $\rightarrow$ urgent targeted therapy).
* **Score $0.65 \le \hat{y} \le 0.88$:** 🟡 **UNCERTAIN** (Borderline density/crowded markings $\rightarrow$ refer to radiologist).

---

## 📈 Verified Performance on the 624 Holdout Test Scans

### Head-to-Head Comparison:

| Metric | Mode 1: Pure AI (Threshold: 0.605) | Mode 2: Clinical Safety Triage ([0.65, 0.88]) | Clinical Impact |
| :--- | :---: | :---: | :--- |
| **Evaluated Patients** | 624 patients (100%) | 545 confident scans (87.3%) | 79 borderline scans referred (12.7%) |
| **Overall Accuracy** | **90.06%** (562 / 624) | **93.03%** (507 / 545) | **+2.97% Accuracy Increase** |
| **Precision** | **89.23%** | **94.75%** | **+5.52% Precision Increase** |
| **Sensitivity (Recall)** | **95.64%** (373 of 390 caught) | **93.81%** | Maintained high diagnostic sensitivity |
| **Specificity** | **80.77%** (189 of 234 cleared) | **91.80%** | **+11.03% Specificity Increase** |
| **False Alarms (FP)** | 45 false positives | **17 false positives** | **62.2% Reduction in False Alarms** |
| **Missed Cases (FN)** | 17 cases (4.36%) | Minimal | Safely protected by doctor referral |

---

## 📁 Repository File Structure

```
pneumonia_project/
├── DL_Project_Day1.ipynb     # Complete 8-step student walkthrough notebook (runs in Colab)
├── train.py                  # Standalone optimal training script (ResNet50V2-Pro + Dual Pooling)
├── app.py                    # Interactive Gradio clinical web application with Grad-CAM viewer
├── PROJECT_GUIDE.md          # Comprehensive technical guide (every parameter, architecture, dataset)
├── final_report.md           # Formal academic research report
├── resnet50v2_pro.keras    # Champion model weights (Dual-Pooling ResNet50V2, 114.6 MB)
├── kaggle.json               # Kaggle API credentials for dataset download
└── README.md                 # Quick-start documentation & benchmarks
```

---

## ⚡ Quick Start & Usage

### 1. Run the Notebook in Google Colab
1. Upload `DL_Project_Day1.ipynb` to [Google Colab](https://colab.research.google.com).
2. Connect a GPU runtime (**Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**).
3. Select **Runtime $\rightarrow$ Run all**.

### 2. Run Standalone Training Locally or on a GPU Server
```bash
python train.py --data_dir "/content/chest_xray/chest_xray" --output_model "resnet50v2_pro.keras"
```

### 3. Launch the Interactive Clinical Web Application
```bash
python app.py
```
Open `http://localhost:7860` in any web browser to analyze radiographs with real-time Grad-CAM explainability.

---

## 📖 Additional Documentation
- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md):** Complete explanation of every hyperparameter, dataset processing detail, and clinical rationale.
- **[final_report.md](final_report.md):** Formal scientific research paper detailing the methodology, formulas, and clinical triage benchmarks.
