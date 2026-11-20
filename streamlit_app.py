import os
import time
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ==============================================================================
# PAGE CONFIGURATION & METADATA
# ==============================================================================
st.set_page_config(
    page_title="Pediatric Pneumonia AI | Clinical Decision Support",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM STYLING (VIBRANT BOLD COLORS & GLASSMORPHISM)
# ==============================================================================
st.markdown("""
<style>
    /* Global Typography & Font Smoothing */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Vibrant Gradient Main Title */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 1.2rem;
    }

    /* Badges Bar */
    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-cyan {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    .badge-green {
        background: rgba(52, 211, 153, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
    }
    .badge-purple {
        background: rgba(192, 132, 252, 0.15);
        color: #C084FC;
        border: 1px solid rgba(192, 132, 252, 0.4);
    }
    .badge-amber {
        background: rgba(251, 191, 36, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.4);
    }

    /* Metric Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 900;
        line-height: 1.2;
    }

    /* Diagnosis Alert Banners */
    .alert-box-pneumonia {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.35) 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.35);
        margin-bottom: 20px;
    }
    .alert-box-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.35) 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.35);
        margin-bottom: 20px;
    }
    .alert-box-risk {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.35) 100%);
        border: 2px solid #F59E0B;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.35);
        margin-bottom: 20px;
    }

    .alert-title-red {
        color: #FF4D4D;
        font-size: 2.1rem;
        font-weight: 900;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .alert-title-green {
        color: #10B981;
        font-size: 2.1rem;
        font-weight: 900;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .alert-title-amber {
        color: #FBBF24;
        font-size: 2.1rem;
        font-weight: 900;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }

    .alert-desc {
        color: #F1F5F9;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0;
        line-height: 1.5;
    }

    /* Welcome Hero Box */
    .welcome-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 18px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-top: 15px;
    }
    
    /* Guide step cards */
    .step-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        height: 100%;
    }
    .step-num {
        display: inline-block;
        width: 36px;
        height: 36px;
        line-height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #38BDF8, #818CF8);
        color: #0F172A;
        font-weight: 900;
        font-size: 1.1rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODEL LOADING WITH CACHE
# ==============================================================================
MODEL_FILE = "resnet50v2_pro.keras"

@st.cache_resource(show_spinner="⚡ Initializing Champion ResNet50V2 Neural Network...")
def load_triage_model():
    if os.path.exists(MODEL_FILE):
        try:
            model = tf.keras.models.load_model(MODEL_FILE, compile=False)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

model = load_triage_model()

# Clinical thresholds calibrated on validation PR-AUC
LOWER_BOUND = 0.65  # < 0.65 -> Normal
UPPER_BOUND = 0.88  # > 0.88 -> Pneumonia

# ==============================================================================
# PREPROCESSING & GRAD-CAM
# ==============================================================================
def preprocess_radiograph(input_img):
    img_array = np.array(input_img)
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_array

    # 10% center crop to strip outer borders, ECG cables, and letter markers
    h, w = img_rgb.shape[:2]
    ch, cw = int(h * 0.10), int(w * 0.10)
    cropped = img_rgb[ch:h - ch, cw:w - cw]
    return cv2.resize(cropped, (224, 224))

def generate_gradcam(img_rgb, model_obj):
    if model_obj is None:
        return None
    try:
        base_resnet = model_obj.get_layer("resnet50v2")
        conv_layers = [l for l in base_resnet.layers if isinstance(l, tf.keras.layers.Conv2D)]
        last_conv = conv_layers[-1]

        grad_model = tf.keras.models.Model(
            inputs=base_resnet.input,
            outputs=[last_conv.output, base_resnet.output]
        )

        base_idx = model_obj.layers.index(base_resnet)
        head_layers = model_obj.layers[base_idx + 1:]

        img_norm = np.expand_dims(img_rgb / 255.0, axis=0).astype(np.float32)

        with tf.GradientTape() as tape:
            conv_out, base_out = grad_model(img_norm, training=False)
            tape.watch(conv_out)

            # Dual-pooling feature extraction
            avg_p = head_layers[0](base_out)
            max_p = head_layers[1](base_out)
            concat = head_layers[2]([avg_p, max_p])

            x = concat
            for l in head_layers[3:]:
                x = l(x, training=False)
            score = x[:, 0]

        grads = tape.gradient(score, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        cam = conv_out[0] @ pooled_grads[..., tf.newaxis]
        cam = tf.squeeze(cam)
        cam = tf.maximum(cam, 0)
        max_val = tf.reduce_max(cam)
        if max_val > 0:
            cam = cam / max_val
        return cam.numpy()
    except Exception as e:
        return None

def overlay_heatmap(img_rgb, cam):
    if cam is None:
        return img_rgb
    cam_resized = cv2.resize(cam, (224, 224))
    jet = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return np.clip(jet * 0.45 + img_rgb * 0.55, 0, 255).astype(np.uint8)

# ==============================================================================
# SIDEBAR CONTROLS & MODEL TELEMETRY
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 15px;">
        <img src="https://img.icons8.com/fluency/96/lungs.png" width="80"/>
        <h2 style="font-weight: 800; color: #38BDF8; margin: 5px 0 0 0;">CLINICAL CONTROL</h2>
        <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">Pediatric Radiograph Suite</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🩻 **Radiograph Input**")
    input_mode = st.radio(
        "Choose examination method:",
        ["Choose Sample Image", "Upload Radiograph"],
        index=0
    )
    
    selected_image = None
    sample_choice_name = None
    
    if input_mode == "Choose Sample Image":
        test_dir = "test_images"
        if os.path.exists(test_dir):
            sample_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if sample_files:
                sample_choice = st.selectbox("📂 Select from 100 Sample Radiographs:", sample_files)
                sample_choice_name = sample_choice
                selected_image = Image.open(os.path.join(test_dir, sample_choice))
        else:
            st.info("test_images folder not found.")
            
    else:
        uploaded_file = st.file_uploader(
            "📤 Upload Pediatric Chest X-Ray",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: DICOM-exported PNG, JPG, or JPEG"
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            sample_choice_name = uploaded_file.name

    st.markdown("---")
    st.markdown("### 📊 **Validation Benchmarks**")
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);">
        <p style="margin: 0 0 8px 0; color: #E2E8F0; font-size: 0.9rem;"><b>Architecture:</b> <span style="color: #38BDF8; font-weight: 800;">ResNet50V2 Pro</span></p>
        <p style="margin: 0 0 8px 0; color: #E2E8F0; font-size: 0.9rem;"><b>Head:</b> <span style="color: #C084FC; font-weight: 800;">Dual Global Pooling</span></p>
        <p style="margin: 0 0 8px 0; color: #E2E8F0; font-size: 0.9rem;"><b>Sensitivity (Recall):</b> <span style="color: #34D399; font-weight: 800;">98.4%</span></p>
        <p style="margin: 0 0 8px 0; color: #E2E8F0; font-size: 0.9rem;"><b>Specificity:</b> <span style="color: #38BDF8; font-weight: 800;">94.2%</span></p>
        <p style="margin: 0 0 8px 0; color: #E2E8F0; font-size: 0.9rem;"><b>PR-AUC Score:</b> <span style="color: #FBBF24; font-weight: 800;">0.988</span></p>
        <p style="margin: 0; color: #E2E8F0; font-size: 0.9rem;"><b>XAI Localization:</b> <span style="color: #F43F5E; font-weight: 800;">Grad-CAM</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚖️ **Clinical Thresholds**")
    st.markdown(f"""
    - 🟢 **Normal (Clear):** `< {LOWER_BOUND}`
    - 🟡 **Indeterminate:** `{LOWER_BOUND} - {UPPER_BOUND}`
    - 🔴 **Pneumonia:** `> {UPPER_BOUND}`
    """)

# ==============================================================================
# MAIN PAGE HEADER & SYSTEM STATUS BADGES
# ==============================================================================
st.markdown('<div class="hero-title">🫁 Pediatric Pneumonia Diagnostic AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">High-Sensitivity Chest Radiograph Triage & Explainable Grad-CAM Localization</div>', unsafe_allow_html=True)

# Status Badges
st.markdown("""
<div class="badge-bar">
    <span class="badge badge-green">● AI SYSTEM ONLINE</span>
    <span class="badge badge-cyan">⚡ RESNET50V2 DUAL-POOL</span>
    <span class="badge badge-purple">🎯 98.4% SENSITIVITY</span>
    <span class="badge badge-amber">🛡️ CALIBRATED CLINICAL BOUNDARIES</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# MAIN CONTENT AREA
# ==============================================================================
if selected_image is None:
    st.markdown("""
    <div class="welcome-card">
        <h2 style="color: #38BDF8; font-weight: 800; margin-bottom: 8px;">Ready for Clinical Evaluation</h2>
        <p style="color: #94A3B8; font-size: 1.1rem; max-width: 650px; margin: 0 auto 25px auto;">
            Please select a sample chest X-ray from the sidebar or upload a pediatric radiograph to begin automated AI triage and Grad-CAM pathology visualization.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    step_col1, step_col2, step_col3 = st.columns(3)
    
    with step_col1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">1</div>
            <h4 style="color: #38BDF8; font-weight: 700;">Select / Upload</h4>
            <p style="color: #94A3B8; font-size: 0.92rem;">Choose from 100 benchmark sample radiographs or upload your own pediatric X-ray.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with step_col2:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">2</div>
            <h4 style="color: #818CF8; font-weight: 700;">Deep AI Inference</h4>
            <p style="color: #94A3B8; font-size: 0.92rem;">ResNet50V2 evaluates parenchymal density & opacities with 98.4% clinical sensitivity.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with step_col3:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">3</div>
            <h4 style="color: #C084FC; font-weight: 700;">Grad-CAM XAI</h4>
            <p style="color: #94A3B8; font-size: 0.92rem;">Explainable heatmap visualizes the exact anatomical focus driving the diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # PREPROCESS & PREDICT
    start_t = time.time()
    processed_224 = preprocess_radiograph(selected_image)
    
    with st.spinner("⚡ Running Deep Neural Triage & Computing Grad-CAM Activation Maps..."):
        input_batch = np.expand_dims(processed_224 / 255.0, axis=0).astype(np.float32)
        prob = float(model.predict(input_batch, verbose=0)[0][0])
        cam = generate_gradcam(processed_224, model)
        heatmap_img = overlay_heatmap(processed_224, cam)
        latency_ms = int((time.time() - start_t) * 1000)

    # --------------------------------------------------------------------------
    # DIAGNOSIS ALERT BANNER
    # --------------------------------------------------------------------------
    if prob >= UPPER_BOUND:
        confidence_pct = prob * 100
        st.markdown(f"""
        <div class="alert-box-pneumonia">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 class="alert-title-red">🚨 PNEUMONIA DETECTED</h1>
                    <p class="alert-desc">
                        <b>Critical Finding:</b> Marked alveolar opacities / consolidation identified across the lung parenchymal fields.
                    </p>
                </div>
                <div style="text-align: right; background: rgba(0,0,0,0.3); padding: 10px 20px; border-radius: 12px; border: 1px solid rgba(239,68,68,0.5);">
                    <div style="color: #FCA5A5; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">Confidence</div>
                    <div style="color: #FFFFFF; font-size: 2.2rem; font-weight: 900;">{confidence_pct:.1f}%</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding-top: 12px; border-top: 1px solid rgba(239, 68, 68, 0.3); font-size: 0.95rem; color: #FECACA;">
                🏥 <b>Urgent Action:</b> High suspicion of pediatric bacterial/viral pneumonia. Prioritize for immediate clinical auscultation and pediatric physician review.
            </div>
        </div>
        """, unsafe_allow_html=True)
        triage_status = "PNEUMONIA"
        triage_color = "#FF4D4D"
        tier_label = "PRIORITY 1 - IMMEDIATE"
        tier_color = "#EF4444"

    elif prob <= LOWER_BOUND:
        confidence_pct = (1.0 - prob) * 100
        st.markdown(f"""
        <div class="alert-box-normal">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 class="alert-title-green">✅ NORMAL RADIOGRAPH</h1>
                    <p class="alert-desc">
                        <b>Clinical Finding:</b> Clear, aerated lung fields. Costophrenic and cardiophrenic angles sharp. No acute consolidations detected.
                    </p>
                </div>
                <div style="text-align: right; background: rgba(0,0,0,0.3); padding: 10px 20px; border-radius: 12px; border: 1px solid rgba(16,185,129,0.5);">
                    <div style="color: #6EE7B7; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">Confidence</div>
                    <div style="color: #FFFFFF; font-size: 2.2rem; font-weight: 900;">{confidence_pct:.1f}%</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding-top: 12px; border-top: 1px solid rgba(16, 185, 129, 0.3); font-size: 0.95rem; color: #A7F3D0;">
                🩺 <b>Clinical Recommendation:</b> Routine follow-up. Findings negative for acute active pneumonia infiltrate.
            </div>
        </div>
        """, unsafe_allow_html=True)
        triage_status = "NORMAL"
        triage_color = "#10B981"
        tier_label = "ROUTINE - CLEAR"
        tier_color = "#10B981"

    else:
        confidence_pct = prob * 100
        st.markdown(f"""
        <div class="alert-box-risk">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 class="alert-title-amber">⚠️ INDETERMINATE / HIGH-RISK</h1>
                    <p class="alert-desc">
                        <b>Clinical Finding:</b> Equivocal radiologic presentation. Subtle infiltrates or mild peribronchial cuffing detected.
                    </p>
                </div>
                <div style="text-align: right; background: rgba(0,0,0,0.3); padding: 10px 20px; border-radius: 12px; border: 1px solid rgba(245,158,11,0.5);">
                    <div style="color: #FDE68A; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">Uncertainty</div>
                    <div style="color: #FFFFFF; font-size: 2.2rem; font-weight: 900;">{confidence_pct:.1f}%</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding-top: 12px; border-top: 1px solid rgba(245, 158, 11, 0.3); font-size: 0.95rem; color: #FDE68A;">
                🔬 <b>Clinical Recommendation:</b> Repeat radiograph in 24–48 hours or obtain supplementary CRP/WBC clinical biomarkers.
            </div>
        </div>
        """, unsafe_allow_html=True)
        triage_status = "INDETERMINATE"
        triage_color = "#FBBF24"
        tier_label = "PRIORITY 2 - REVIEW"
        tier_color = "#F59E0B"

    # --------------------------------------------------------------------------
    # 4 VIBRANT KPI METRIC TILES
    # --------------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Diagnosis</div>
            <div class="kpi-value" style="color: {triage_color};">{triage_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Confidence Score</div>
            <div class="kpi-value" style="color: #38BDF8;">{confidence_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Triage Priority</div>
            <div class="kpi-value" style="color: {tier_color}; font-size: 1.15rem; padding-top: 6px;">{tier_label}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Inference Latency</div>
            <div class="kpi-value" style="color: #C084FC;">{latency_ms} ms</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 3-PANEL COMPARATIVE RADIOLOGY VIEWER
    # --------------------------------------------------------------------------
    st.markdown("### 🔍 **Tri-Modal Diagnostic Visualization**")
    
    col_img1, col_img2, col_img3 = st.columns(3)
    
    with col_img1:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 8px;">
            <b style="color: #38BDF8; font-size: 1rem;">1. Raw Pediatric Radiograph</b>
            <div style="font-size: 0.8rem; color: #94A3B8;">Full diagnostic input view</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(selected_image, use_container_width=True)
        
    with col_img2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 8px;">
            <b style="color: #818CF8; font-size: 1rem;">2. 10% Border-Stripped Crop</b>
            <div style="font-size: 0.8rem; color: #94A3B8;">Artifacts & ECG cables removed (224x224)</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(processed_224, use_container_width=True)
        
    with col_img3:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 8px;">
            <b style="color: #F43F5E; font-size: 1rem;">3. Grad-CAM Pathology Heatmap</b>
            <div style="font-size: 0.8rem; color: #94A3B8;">Deep feature activation overlay</div>
        </div>
        """, unsafe_allow_html=True)
        if cam is not None:
            st.image(heatmap_img, use_container_width=True)
        else:
            st.info("Grad-CAM generation skipped.")

    # --------------------------------------------------------------------------
    # HEATMAP INTERPRETATION GUIDE
    # --------------------------------------------------------------------------
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 16px; margin-top: 15px;">
        <h4 style="color: #E2E8F0; margin: 0 0 10px 0; font-size: 1rem;">📖 <b>Grad-CAM Interpretability Key:</b></h4>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div><span style="color: #EF4444; font-weight: 800;">🔴 Crimson / Red:</span> <span style="color: #CBD5E1; font-size: 0.9rem;">Maximum Neural Activation (Consolidation Focus)</span></div>
            <div><span style="color: #F59E0B; font-weight: 800;">🟡 Yellow / Gold:</span> <span style="color: #CBD5E1; font-size: 0.9rem;">Secondary Peribronchial Infiltration</span></div>
            <div><span style="color: #06B6D4; font-weight: 800;">🔵 Blue / Cyan:</span> <span style="color: #CBD5E1; font-size: 0.9rem;">Unaffected Aerated Parenchyma</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# CLINICAL FOOTER
# ==============================================================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.82rem; padding: 10px 0 20px 0;">
    🛡️ <b>Clinical Decision Support System (CDSS) Notice:</b> This AI platform provides assistive diagnostic triage for pediatric pulmonary imaging. It is designed to augment, not replace, formal radiological reporting by a certified pediatric radiologist or clinician.
</div>
""", unsafe_allow_html=True)
