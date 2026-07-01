"""
MINIMAL UI  —  Brain Tumor MRI Classifier
=========================================
Script: src/app_minimal.py

A deliberately small, single-file Streamlit app for **testing & demonstration**.
It complements the full-featured src/app.py but keeps only the essentials:

    1. Pick a model (only the ones actually trained show up)
    2. Upload an MRI image
    3. See the 6-stage preprocessing pipeline
    4. Get the predicted class + confidence + a matplotlib probability chart

Design goals:
  • Works with ZERO trained models — the preprocessing demo still runs, so you
    can present the image-processing pipeline even before training finishes.
  • Pure matplotlib for the chart (no plotly) → minimal dependencies.
  • ~150 lines, easy to read and explain in a viva.

Launch:
    streamlit run src/app_minimal.py
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless: we hand the figure to Streamlit
import matplotlib.pyplot as plt
import streamlit as st

# Make `src` importable whether run from repo root or elsewhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import preprocess_image          # noqa: E402
from src.models import get_preprocess_input          # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────────
MODELS_DIR  = ROOT / "models"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_EMOJI = {"glioma": "🔴", "meningioma": "🟣", "notumor": "🟢", "pituitary": "🔵"}
BAR_COLORS  = ["#e15759", "#a855f7", "#59a14f", "#4e79a7"]
MODEL_DISPLAY = {
    "custom_cnn":   "Custom CNN",
    "vgg16":        "VGG16 (Transfer)",
    "resnet50":     "ResNet50 (Transfer)",
    "efficientnet": "EfficientNetB0 (Transfer)",
}
# Test-set accuracy (from reports/model_comparison.csv) — used to rank the picker
# best-first and to label each option. ResNet50 is the overall best model.
MODEL_ACC  = {"resnet50": 85.8, "vgg16": 84.6, "efficientnet": 83.8, "custom_cnn": 71.7}
MODEL_RANK = ["resnet50", "vgg16", "efficientnet", "custom_cnn"]   # best → worst
BEST_MODEL = MODEL_RANK[0]

st.set_page_config(page_title="Brain Tumor Classifier — Minimal", page_icon="🧠", layout="centered")


# ── Helpers ──────────────────────────────────────────────────────────────────────
def available_models() -> list[str]:
    """Model keys that have a saved .keras file, ordered best→worst by test accuracy."""
    if not MODELS_DIR.exists():
        return []
    return [
        key for key in MODEL_RANK
        if (MODELS_DIR / f"{key}_final.keras").exists()
        or (MODELS_DIR / f"{key}_best.keras").exists()
    ]


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """Load & cache a Keras model. Imported lazily so the app starts fast."""
    from tensorflow import keras
    for suffix in ("_final.keras", "_best.keras"):
        path = MODELS_DIR / f"{model_key}{suffix}"
        if path.exists():
            return keras.models.load_model(str(path))
    return None


def predict(model, model_key: str, normalized_img: np.ndarray):
    """Run inference on a [0,1] normalized (224,224,3) image → (class, conf, probs)."""
    x = np.expand_dims(normalized_img, axis=0)          # (1, H, W, 3)
    pre = get_preprocess_input(model_key)
    # Transfer backbones expect their own channel-mean preprocessing on a [0,255] range;
    # the custom CNN consumes the [0,1] image directly (its preprocess is identity).
    x = pre(x * 255.0) if model_key != "custom_cnn" else pre(x)
    # Direct call (not model.predict) — for a single image it skips the predict
    # loop / callback machinery and returns in milliseconds.
    probs = model(x, training=False).numpy()[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


def probability_chart(probs: np.ndarray):
    """Horizontal matplotlib bar chart of the 4 class probabilities."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    labels = [f"{CLASS_EMOJI[c]} {c.capitalize()}" for c in CLASS_NAMES]
    vals = [p * 100 for p in probs]
    bars = ax.barh(labels, vals, color=BAR_COLORS)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.invert_yaxis()                                    # first class on top
    for b, v in zip(bars, vals):
        ax.text(min(v + 1.5, 96), b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


# ── App ──────────────────────────────────────────────────────────────────────────
st.title("🧠 Brain Tumor MRI Classifier")
st.caption("Minimal demo UI · upload an MRI → preprocess → classify. Research/education only.")

models = available_models()

with st.sidebar:
    st.header("⚙️ Settings")
    if models:
        def _label(k):
            star = "⭐ " if k == BEST_MODEL else ""
            return f"{star}{MODEL_DISPLAY[k]} · {MODEL_ACC[k]:.1f}% acc"
        # `models` is sorted best-first, so index 0 defaults to the most accurate
        # model available (⭐ ResNet50 once it is trained/present).
        model_key = st.selectbox("Model", models, index=0, format_func=_label)
        if BEST_MODEL in models:
            st.caption("⭐ ResNet50 is the most accurate model (85.8%) — selected by default.")
        else:
            st.caption(
                f"Showing trained models only. Best available: **{MODEL_DISPLAY[models[0]]}**. "
                "Add ResNet50 to `models/` to demo the top model."
            )
    else:
        model_key = None
        st.warning(
            "No trained models found in `models/`.\n\n"
            "The **preprocessing demo still works** below. To enable prediction, "
            "train at least one model:\n\n```\npython -m src.train --models custom_cnn\n```"
        )
    st.markdown("---")
    st.markdown("**Classes**")
    for c in CLASS_NAMES:
        st.markdown(f"{CLASS_EMOJI[c]} {c.capitalize()}")

uploaded = st.file_uploader("Upload a brain MRI (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("👆 Upload an MRI image to begin.")
    st.stop()

# Persist the upload to a temp file so OpenCV (used inside preprocess_image) can read it
img_bytes = uploaded.read()
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    tmp.write(img_bytes)
    tmp_path = tmp.name

try:
    stages = preprocess_image(tmp_path, return_stages=True)
finally:
    os.unlink(tmp_path)

# ── Preprocessing gallery (works with or without a model) ────────────────────────
st.subheader("🔬 Preprocessing pipeline")
keys   = ["raw", "gray3", "gaussian", "median", "stripped", "normalized"]
labels = ["Original", "Grayscale", "Gaussian", "Median", "Skull-strip", "Normalized"]
cols = st.columns(6)
for col, k, lab in zip(cols, keys, labels):
    arr = stages[k]
    disp = (arr * 255).clip(0, 255).astype(np.uint8) if arr.dtype.kind == "f" else arr.astype(np.uint8)
    if disp.ndim == 3 and disp.shape[2] == 3:            # OpenCV is BGR → flip to RGB for display
        disp = disp[:, :, ::-1]
    col.image(disp, use_container_width=True)
    col.caption(lab)

# ── Prediction (only if a model is available) ────────────────────────────────────
st.subheader("🧬 Classification")
if model_key is None:
    st.warning("Prediction disabled — no trained model available (see sidebar).")
    st.stop()

# Load the model first (the first run imports TensorFlow — a few seconds), then
# infer. Splitting the two makes it obvious which step you're waiting on, and the
# try/except surfaces any real error instead of leaving the spinner stuck forever.
with st.spinner("Loading model (first run imports TensorFlow — ~5–10s)…"):
    model = load_model(model_key)
if model is None:
    st.error("Model file vanished. Retrain and refresh.")
    st.stop()

try:
    with st.spinner(f"Running {MODEL_DISPLAY[model_key]}…"):
        pred_class, conf, probs = predict(model, model_key, stages["normalized"])
except Exception as e:                    # noqa: BLE001 — show the user what broke
    st.error("Inference failed — details below:")
    st.exception(e)
    st.stop()

c1, c2 = st.columns([1, 1])
c1.metric("Prediction", f"{CLASS_EMOJI[pred_class]} {pred_class.capitalize()}")
c2.metric("Confidence", f"{conf * 100:.1f}%")
st.pyplot(probability_chart(probs))
