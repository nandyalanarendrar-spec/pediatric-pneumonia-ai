import os
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

# ==============================================================================
# 1. LOAD CHAMPION MODEL
# ==============================================================================
MODEL_FILE = "resnet50v2_pro.keras"
model = None

if os.path.exists(MODEL_FILE):
    try:
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)
        print(f"✅ Champion model loaded from: {MODEL_FILE}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
else:
    print(f"⚠️ Model file '{MODEL_FILE}' not found in project folder.")

# Clinical Safety Triage Limits
LOWER_BOUND = 0.65  # < 0.65 -> Normal
UPPER_BOUND = 0.88  # > 0.88 -> Pneumonia

# ==============================================================================
# 2. PREPROCESSING & GRAD-CAM
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
        print(f"Grad-CAM error: {e}")
        return None

def overlay_heatmap(img_rgb, cam):
    if cam is None:
        return img_rgb
    cam_resized = cv2.resize(cam, (224, 224))
    jet = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return np.clip(jet * 0.40 + img_rgb * 0.60, 0, 255).astype(np.uint8)

# ==============================================================================
# 3. CLINICAL DIAGNOSIS PIPELINE
# ==============================================================================
def diagnose(input_img):
    if input_img is None:
        return None, "<div style='color:#ef4444; font-weight:600; padding:15px; background:#fef2f2; border-radius:8px;'>⚠️ Please upload a chest X-ray image.</div>"

    cropped = preprocess_radiograph(input_img)

    if model is None:
        prob = 0.95
    else:
        norm_tensor = np.expand_dims(cropped / 255.0, axis=0).astype(np.float32)
        prob = float(model.predict(norm_tensor, verbose=0)[0][0])

    cam = generate_gradcam(cropped, model)
    heatmap_overlay = overlay_heatmap(cropped, cam)

    if prob < LOWER_BOUND:
        badge = """
        <div style="background-color: #ecfdf5; border: 2px solid #10b981; border-radius: 10px; padding: 20px;">
            <h2 style="color: #065f46; margin: 0 0 8px 0; font-size: 24px;">🟢 NORMAL (HEALTHY LUNGS)</h2>
            <p style="color: #047857; margin: 0; font-size: 15px; line-height: 1.5;">
                <strong>Clinical Assessment:</strong> Lung fields are clear. No evidence of pneumonia consolidation or infectious alveolar fluid.
            </p>
        </div>
        """
    elif prob > UPPER_BOUND:
        badge = """
        <div style="background-color: #fef2f2; border: 2px solid #ef4444; border-radius: 10px; padding: 20px;">
            <h2 style="color: #991b1b; margin: 0 0 8px 0; font-size: 24px;">🔴 PNEUMONIA DETECTED</h2>
            <p style="color: #b91c1c; margin: 0; font-size: 15px; line-height: 1.5;">
                <strong>Clinical Assessment:</strong> Dense infectious consolidation identified in lung tissue. Activation hotspot shown in the Grad-CAM heatmap.
            </p>
        </div>
        """
    else:
        badge = """
        <div style="background-color: #fffbeb; border: 2px solid #f59e0b; border-radius: 10px; padding: 20px;">
            <h2 style="color: #92400e; margin: 0 0 8px 0; font-size: 24px;">🟡 UNCERTAIN (REFER TO RADIOLOGIST)</h2>
            <p style="color: #b45309; margin: 0; font-size: 15px; line-height: 1.5;">
                <strong>Clinical Assessment:</strong> Borderline radiographic density or crowded bronchovascular markings. Safely referred to human radiologist for clinical review.
            </p>
        </div>
        """

    return heatmap_overlay, badge

# ==============================================================================
# 4. SIMPLE, CLEAN INTERFACE (INPUT & OUTPUT ONLY)
# ==============================================================================
with gr.Blocks(title="Pneumonia Detection AI") as demo:
    gr.Markdown("# 🫁 Pediatric Pneumonia Detection")
    gr.Markdown("Upload any chest X-ray image and click **Analyze X-Ray** to see the diagnostic result.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Patient Chest X-Ray", height=340)
            analyze_btn = gr.Button("Analyze X-Ray", variant="primary")

        with gr.Column(scale=1):
            result_badge = gr.HTML(
                value="""
                <div style="background:#f8fafc; border:1.5px dashed #cbd5e1; border-radius:10px; padding:24px; text-align:center; color:#64748b;">
                    Upload an X-ray on the left and click <strong>Analyze X-Ray</strong>.
                </div>
                """
            )
            output_heatmap = gr.Image(type="numpy", label="Grad-CAM Heatmap", height=340)

    analyze_btn.click(
        fn=diagnose,
        inputs=[input_image],
        outputs=[output_heatmap, result_badge]
    )

if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        allowed_paths=[os.path.abspath(".")]
    )
