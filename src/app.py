"""
PHASE 10 – USER INTERFACE
==========================
Script: src/app.py

Streamlit-based diagnostic interface implementing all UI requirements
from the Proposal (Section 2 – Objectives, Point 5):

  ✓ MRI image upload
  ✓ Image preview (original + preprocessing stages)
  ✓ Prediction with selected model
  ✓ Display predicted class with confidence score
  ✓ Per-class probability bar chart

Launch: streamlit run src/app.py
"""

import sys
import os
from pathlib import Path

# Add project root to path so src imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st
from tensorflow import keras
import plotly.graph_objects as go

from src.preprocess import preprocess_image
from src.models import get_preprocess_input

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .main-header h1 { color: white; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .main-header p  { color: rgba(255,255,255,0.85); font-size: 1.1rem; margin: 0.5rem 0 0; }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.2rem;
    }

    /* Metric boxes */
    .metric-box {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .metric-label { color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 500; margin-bottom: 0.3rem; }
    .metric-value { color: white; font-size: 1.8rem; font-weight: 700; }

    /* Result badges */
    .result-glioma      { background: linear-gradient(135deg,#ff6b6b,#ff8e53); }
    .result-meningioma  { background: linear-gradient(135deg,#a18cd1,#fbc2eb); }
    .result-pituitary   { background: linear-gradient(135deg,#43e97b,#38f9d7); }
    .result-notumor     { background: linear-gradient(135deg,#4facfe,#00f2fe); }
    .result-default     { background: linear-gradient(135deg,#667eea,#764ba2); }
    .result-badge {
        padding: 1.5rem 2rem;
        border-radius: 14px;
        text-align: center;
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 1rem 0;
        letter-spacing: 1px;
        text-shadow: 0 2px 6px rgba(0,0,0,0.3);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }

    /* Sidebar */
    .css-1d391kg { background: rgba(15,12,41,0.8) !important; }

    /* Step labels */
    .step-label {
        text-align: center;
        color: rgba(255,255,255,0.75);
        font-size: 0.8rem;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(255,200,0,0.1);
        border: 1px solid rgba(255,200,0,0.3);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: rgba(255,220,50,0.9);
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODELS_DIR  = ROOT / "models"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma":     "🔴  Glioma",
    "meningioma": "🟣  Meningioma",
    "notumor":    "🟢  No Tumor",
    "pituitary":  "🔵  Pituitary Tumor",
}
CLASS_CSS = {
    "glioma":     "result-glioma",
    "meningioma": "result-meningioma",
    "pituitary":  "result-pituitary",
    "notumor":    "result-notumor",
}
CLASS_INFO = {
    "glioma": (
        "Gliomas arise from glial cells in the brain or spine. "
        "They are the most common type of primary brain tumour. "
        "Types include astrocytomas, oligodendrogliomas, and glioblastomas."
    ),
    "meningioma": (
        "Meningiomas arise from the meninges – the membranes surrounding "
        "the brain and spinal cord. They are usually benign (non-cancerous) "
        "and slow-growing."
    ),
    "pituitary": (
        "Pituitary tumours develop in the pituitary gland at the base of "
        "the brain. Most are benign adenomas. They can affect hormone levels, "
        "vision, and other bodily functions."
    ),
    "notumor": (
        "No brain tumour was detected in this MRI scan. The scan appears "
        "to show normal brain tissue without any evident tumour mass."
    ),
}

MODEL_DISPLAY = {
    "custom_cnn":   "Custom CNN",
    "vgg16":        "VGG16 (Transfer Learning)",
    "resnet50":     "ResNet50 (Transfer Learning)",
    "efficientnet": "EfficientNetB0 (Transfer Learning)",
}


# ── Model Loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load and cache a Keras model by name."""
    best_path  = MODELS_DIR / f"{model_name}_best.keras"
    final_path = MODELS_DIR / f"{model_name}_final.keras"
    for path in [best_path, final_path]:
        if path.exists():
            return keras.models.load_model(str(path))
    return None


# ── Preprocessing Preview ──────────────────────────────────────────────────────
def show_preprocessing_stages(img_bytes: bytes):
    """Display all preprocessing steps as an image gallery."""
    import tempfile, io
    from PIL import Image

    # Save bytes to temp file so preprocess_image can load via OpenCV
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        stages = preprocess_image(tmp_path, return_stages=True)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        os.unlink(tmp_path)
        return None, None

    os.unlink(tmp_path)

    stage_keys = ["raw", "gray3", "gaussian", "median", "stripped", "normalized"]
    stage_labels = ["Original", "Grayscale (3ch)", "Gaussian Filter", "Median Filter", "Skull Stripped", "Normalised"]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Preprocessing Pipeline")
    cols = st.columns(len(stage_keys))

    for col, key, label in zip(cols, stage_keys, stage_labels):
        arr = stages[key]
        # Convert to uint8 for display
        if arr.dtype in [np.float32, np.float64]:
            arr_disp = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr_disp = arr.astype(np.uint8)
        # BGR → RGB for display
        if len(arr_disp.shape) == 3 and arr_disp.shape[2] == 3:
            arr_disp = cv2.cvtColor(arr_disp, cv2.COLOR_BGR2RGB)

        col.image(arr_disp, use_column_width=True)
        col.markdown(f"<div class='step-label'>{label}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    return stages["normalized"], stages["raw"]


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(model, normalized_img: np.ndarray) -> tuple:
    """
    Run inference and return (class_name, confidence, all_probs).

    Args:
        model:          Loaded Keras model.
        normalized_img: Float32 (IMG_SIZE, IMG_SIZE, 3) in [0, 1].

    Returns:
        (predicted_class_name, confidence_float, probs_array)
    """
    x = np.expand_dims(normalized_img, axis=0)     # (1, H, W, 3)

    # Apply the correct preprocessing for the selected backbone.
    model_name = getattr(model, "name", "custom_cnn").lower()
    is_transfer = False
    if "vgg16" in model_name:
        preprocess_fn = get_preprocess_input("vgg16")
        is_transfer = True
    elif "resnet50" in model_name:
        preprocess_fn = get_preprocess_input("resnet50")
        is_transfer = True
    elif "efficientnet" in model_name:
        preprocess_fn = get_preprocess_input("efficientnet")
        is_transfer = True
    else:
        preprocess_fn = get_preprocess_input("custom_cnn")

    if is_transfer:
        probs = model.predict(preprocess_fn(x * 255.0), verbose=0)[0]         # (num_classes,)
    else:
        probs = model.predict(preprocess_fn(x), verbose=0)[0]         # (num_classes,)
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_class, confidence, probs


# ── Probability Chart ──────────────────────────────────────────────────────────
def plot_probability_chart(probs: np.ndarray) -> go.Figure:
    """Create an interactive Plotly bar chart of class probabilities."""
    colors = ["#ff6b6b", "#a18cd1", "#4facfe", "#43e97b"]
    labels = [CLASS_DISPLAY[c] for c in CLASS_NAMES]
    values = [float(p) * 100 for p in probs]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Prediction Confidence per Class",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 115],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='main-header'>
        <h1>🧠 Brain Tumor MRI Classifier</h1>
        <p>Automated detection and classification of brain tumors using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        model_key = st.selectbox(
            "Select Model",
            options=list(MODEL_DISPLAY.keys()),
            format_func=lambda k: MODEL_DISPLAY[k],
            index=2,   # Default: ResNet50
        )

        st.markdown("---")
        st.markdown("### 📂 Tumor Classes")
        for cls, disp in CLASS_DISPLAY.items():
            st.markdown(f"- **{disp}**")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "This system classifies brain MRI scans using CNNs and "
            "Transfer Learning. For research and educational purposes only."
        )
        st.markdown("---")
        st.markdown(
            "<div class='disclaimer'>⚠️ <b>Medical Disclaimer</b>: "
            "This tool is for research purposes only. "
            "Always consult a qualified medical professional for diagnosis.</div>",
            unsafe_allow_html=True,
        )

    # ── Model Loading ──────────────────────────────────────────────────────────
    with st.spinner(f"Loading {MODEL_DISPLAY[model_key]}…"):
        model = load_model(model_key)

    if model is None:
        st.error(
            f"⚠️ Model **{MODEL_DISPLAY[model_key]}** not found. "
            "Please train the models first:\n\n"
            "```\npython src/train.py\n```"
        )
        st.info("After training completes, refresh this page.")
        return

    st.success(f"✅ **{MODEL_DISPLAY[model_key]}** loaded successfully.")

    # ── File Upload ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a T1/T2/FLAIR brain MRI image for classification.",
    )

    if uploaded_file is None:
        st.info("👆 Please upload an MRI image to begin classification.")

        # Demo placeholder
        st.markdown("---")
        st.markdown("### 💡 How to Use")
        col1, col2, col3 = st.columns(3)
        col1.markdown("<div class='metric-box'><div class='metric-label'>Step 1</div><div class='metric-value'>📤</div><div class='metric-label'>Upload MRI Image</div></div>", unsafe_allow_html=True)
        col2.markdown("<div class='metric-box'><div class='metric-label'>Step 2</div><div class='metric-value'>🔬</div><div class='metric-label'>View Preprocessing</div></div>", unsafe_allow_html=True)
        col3.markdown("<div class='metric-box'><div class='metric-label'>Step 3</div><div class='metric-value'>🧬</div><div class='metric-label'>Get Diagnosis</div></div>", unsafe_allow_html=True)
        return

    # ── Process Uploaded Image ─────────────────────────────────────────────────
    img_bytes = uploaded_file.read()

    # Raw image preview
    st.markdown("---")
    col_preview, col_info = st.columns([1, 2])
    with col_preview:
        st.image(img_bytes, caption="Uploaded MRI Image", use_column_width=True)
    with col_info:
        st.markdown("### 📋 Image Information")
        import io
        from PIL import Image
        pil_img = Image.open(io.BytesIO(img_bytes))
        w, h = pil_img.size
        st.markdown(f"""
        <div class='card'>
            <p>📁 <b>File:</b> {uploaded_file.name}</p>
            <p>📐 <b>Original size:</b> {w} × {h} px</p>
            <p>📄 <b>Format:</b> {pil_img.format or 'Unknown'}</p>
            <p>🎨 <b>Mode:</b> {pil_img.mode}</p>
            <p>🤖 <b>Model:</b> {MODEL_DISPLAY[model_key]}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔬 Preprocessing Pipeline")
    with st.spinner("Applying preprocessing pipeline…"):
        normalized_img, raw_img = show_preprocessing_stages(img_bytes)

    if normalized_img is None:
        return

    # ── Prediction ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🧬 Classification Result")

    with st.spinner("Running inference…"):
        pred_class, confidence, probs = predict(model, normalized_img)

    # Result badge
    css_class = CLASS_CSS.get(pred_class, "result-default")
    display   = CLASS_DISPLAY.get(pred_class, pred_class.upper())
    st.markdown(
        f"<div class='result-badge {css_class}'>{display}</div>",
        unsafe_allow_html=True,
    )

    # Metrics row
    col_acc, col_model, col_class = st.columns(3)
    col_acc.markdown(
        f"<div class='metric-box'><div class='metric-label'>Confidence</div>"
        f"<div class='metric-value'>{confidence*100:.1f}%</div></div>",
        unsafe_allow_html=True,
    )
    col_model.markdown(
        f"<div class='metric-box'><div class='metric-label'>Model Used</div>"
        f"<div class='metric-value' style='font-size:1rem'>{MODEL_DISPLAY[model_key]}</div></div>",
        unsafe_allow_html=True,
    )
    col_class.markdown(
        f"<div class='metric-box'><div class='metric-label'>Detected Class</div>"
        f"<div class='metric-value' style='font-size:1.1rem'>{pred_class.title()}</div></div>",
        unsafe_allow_html=True,
    )

    # Probability chart
    st.markdown("---")
    fig = plot_probability_chart(probs)
    st.plotly_chart(fig, use_container_width=True)

    # Per-class breakdown table
    st.markdown("### 📊 Per-Class Probabilities")
    prob_data = {
        "Class": [CLASS_DISPLAY[c] for c in CLASS_NAMES],
        "Probability": [f"{p*100:.2f}%" for p in probs],
    }
    st.table(prob_data)

    # Clinical info
    st.markdown("---")
    st.markdown("### 🏥 Clinical Information")
    st.markdown(
        f"<div class='card'><b>{display}</b><br><br>{CLASS_INFO.get(pred_class, '')}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
