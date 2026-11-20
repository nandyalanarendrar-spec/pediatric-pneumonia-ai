import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Pediatric Pneumonia AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #0284C7;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODEL LOADING WITH CACHE
# ==============================================================================
MODEL_FILE = "resnet50v2_pro.keras"

@st.cache_resource(show_spinner="Loading Champion ResNet50V2 Architecture...")
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
        st.warning(f"Grad-CAM generation notice: {e}")
        return None

def overlay_heatmap(img_rgb, cam):
    if cam is None:
        return img_rgb
    cam_resized = cv2.resize(cam, (224, 224))
    jet = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return np.clip(jet * 0.40 + img_rgb * 0.60, 0, 255).astype(np.uint8)

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=70)
    st.title("Settings & Samples")
    
    st.markdown("### Choose Input Source")
    input_mode = st.radio("Select input mode:", ["Upload Radiograph", "Choose Sample Image"])
    
    selected_image = None
    
    if input_mode == "Choose Sample Image":
        test_dir = "test_images"
        if os.path.exists(test_dir):
            sample_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if sample_files:
                sample_choice = st.selectbox("Select Test Radiograph:", sample_files)
                selected_image = Image.open(os.path.join(test_dir, sample_choice))
        else:
            st.info("No test_images directory found.")
            
    else:
        uploaded_file = st.file_uploader("Upload Pediatric Chest X-Ray", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            
    st.markdown("---")
    st.markdown("### Model Specifications")
    st.markdown("""
    - **Architecture**: ResNet50V2 Dual-Pool
    - **Input Resolution**: 224 x 224 x 3
    - **Sensitivity**: 98.4%
    - **Specificity**: 94.2%
    - **XAI Engine**: Grad-CAM (post-conv)
    """)

# ==============================================================================
# MAIN PAGE INTERFACE
# ==============================================================================
st.markdown('<div class="main-header">🫁 Pediatric Pneumonia Diagnostic AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Chest Radiograph Triage & Explainable Grad-CAM Localization</div>', unsafe_allow_html=True)

if selected_image is None:
    st.info("👈 Please select a sample image or upload a pediatric chest X-ray from the left sidebar to begin diagnosis.")
else:
    processed_224 = preprocess_radiograph(selected_image)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Radiograph")
        st.image(selected_image, caption="Original Pediatric Chest Radiograph", use_container_width=True)
        
    with col2:
        st.subheader("Clinical AI Evaluation")
        
        if model is None:
            st.error("Model weights file (`resnet50v2_pro.keras`) not found.")
        else:
            with st.spinner("Analyzing radiograph with ResNet50V2 & generating Grad-CAM..."):
                input_batch = np.expand_dims(processed_224 / 255.0, axis=0).astype(np.float32)
                prob = float(model.predict(input_batch, verbose=0)[0][0])
                cam = generate_gradcam(processed_224, model)
                heatmap_img = overlay_heatmap(processed_224, cam)
                
            # Clinical Decision Triage
            if prob >= UPPER_BOUND:
                st.error(f"### 🚨 PNEUMONIA DETECTED\n**Confidence: {prob * 100:.2f}%**")
                st.write("**Recommendation:** Immediate clinical evaluation required. Bilateral/unilateral pulmonary infiltrates identified.")
            elif prob <= LOWER_BOUND:
                st.success(f"### ✅ NORMAL RADIOGRAPH\n**Confidence: {(1.0 - prob) * 100:.2f}%**")
                st.write("**Recommendation:** No acute consolidation or significant infiltrates detected. Clear lung fields.")
            else:
                st.warning(f"### ⚠️ INDETERMINATE / HIGH RISK\n**Confidence: {prob * 100:.2f}%**")
                st.write("**Recommendation:** Equivocal presentation. Radiologist consultation and serial follow-up recommended.")
                
            st.progress(prob)
            st.caption(f"Raw Output Score: `{prob:.4f}` | Decision Boundaries: Normal < {LOWER_BOUND} | High Risk: {LOWER_BOUND} - {UPPER_BOUND} | Pneumonia > {UPPER_BOUND}")

    st.markdown("---")
    st.subheader("🔍 Explainable AI: Grad-CAM Pathology Heatmap")
    
    cam_col1, cam_col2 = st.columns(2)
    with cam_col1:
        st.image(processed_224, caption="Preprocessed Center-Cropped Lung Field (224x224)", use_container_width=True)
    with cam_col2:
        if cam is not None:
            st.image(heatmap_img, caption="Grad-CAM Activation Map (Warm colors = High Neural Attention)", use_container_width=True)
        else:
            st.write("Heatmap unavailable for this sample.")

st.markdown("---")
st.caption("⚠️ **Clinical Disclaimer:** This tool is an investigational clinical decision support system designed to assist healthcare professionals. It does not replace professional radiologic interpretation or clinical judgment.")
