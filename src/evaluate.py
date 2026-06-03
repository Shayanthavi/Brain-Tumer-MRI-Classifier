"""
PHASE 9 – EVALUATION
======================
Script: src/evaluate.py

Evaluates all four trained models on the held-out test set.

Required metrics (Proposal Section 4.4):
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - F1-Score
  - Confusion Matrix

Outputs:
  - Console: per-class classification report
  - reports/confusion_matrix_{model}.png
  - reports/training_curves_{model}.png
  - reports/model_comparison.png
  - reports/model_comparison.csv
  - reports/classification_reports.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.models import get_preprocess_input

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = 224
BATCH_SIZE  = 32
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_NAMES = ["custom_cnn", "vgg16", "resnet50", "efficientnet"]


# ── Load Test Dataset ──────────────────────────────────────────────────────────
def _load_test_dataset(model_name: str) -> tuple:
    """
    Load test images and return (true_labels, image_arrays).

    Returns:
        y_true:  1D array of integer labels.
        images:  4D float32 array (N, IMG_SIZE, IMG_SIZE, 3).
    """
    test_dir = PROCESSED_DIR / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory '{test_dir}' not found. "
            "Run src/preprocess.py first."
        )

    ds = keras.utils.image_dataset_from_directory(
        directory=str(test_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False,
        seed=42,
    )
    preprocess_fn = get_preprocess_input(model_name)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    if model_name == "custom_cnn":
        ds = ds.map(lambda x, y: (x, y))
    else:
        ds = ds.map(lambda x, y: (preprocess_fn(x * 255.0), y))

    all_images, all_labels = [], []
    for imgs, lbls in ds:
        all_images.append(imgs.numpy())
        all_labels.append(np.argmax(lbls.numpy(), axis=1))

    return np.concatenate(all_labels), np.concatenate(all_images, axis=0)


# ── Load Model ────────────────────────────────────────────────────────────────
def _load_trained_model(model_name: str) -> keras.Model:
    """Load the best saved model weights."""
    best_path  = MODELS_DIR / f"{model_name}_best.keras"
    final_path = MODELS_DIR / f"{model_name}_final.keras"

    for path in [best_path, final_path]:
        if path.exists():
            print(f"[INFO] Loading {model_name} from {path}")
            return keras.models.load_model(str(path))

    raise FileNotFoundError(
        f"No saved model found for '{model_name}'. "
        "Run src/train.py first."
    )


# ── Predict ───────────────────────────────────────────────────────────────────
def _predict(model: keras.Model, images: np.ndarray) -> tuple:
    """
    Run inference and return (predicted_labels, confidence_scores).
    """
    probs = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    return preds, confs


# ── Plot: Confusion Matrix ─────────────────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix – {model_name.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = REPORTS_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Confusion matrix saved → {save_path}")


# ── Plot: Training Curves ──────────────────────────────────────────────────────
def _plot_training_curves(model_name: str):
    hist_path = MODELS_DIR / f"{model_name}_history.json"
    if not hist_path.exists():
        print(f"  [WARN] No history file for {model_name}. Skipping training curves.")
        return

    with open(hist_path) as f:
        history = json.load(f)

    # Concatenate stages
    acc     = history.get("stage1_acc", []) + history.get("stage2_acc", [])
    val_acc = history.get("stage1_val_acc", []) + history.get("stage2_val_acc", [])
    loss    = history.get("stage1_loss", []) + history.get("stage2_loss", [])
    val_loss = history.get("stage1_val_loss", []) + history.get("stage2_val_loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(acc) + 1)

    axes[0].plot(epochs, acc,     "b-o", markersize=3, label="Train Accuracy")
    axes[0].plot(epochs, val_acc, "r-o", markersize=3, label="Val Accuracy")
    axes[0].set_title(f"Accuracy – {model_name.upper()}", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, loss,     "b-o", markersize=3, label="Train Loss")
    axes[1].plot(epochs, val_loss, "r-o", markersize=3, label="Val Loss")
    axes[1].set_title(f"Loss – {model_name.upper()}", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Categorical Cross-Entropy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Mark Stage 2 boundary if applicable
    s1_len = len(history.get("stage1_acc", []))
    if s1_len > 0 and history.get("stage2_acc"):
        for ax in axes:
            ax.axvline(x=s1_len + 0.5, color="green", linestyle="--",
                       alpha=0.7, label="Stage 2 Start")
        axes[0].legend()
        axes[1].legend()

    plt.tight_layout()
    save_path = REPORTS_DIR / f"training_curves_{model_name}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Training curves saved → {save_path}")


# ── Plot: Model Comparison Bar Chart ──────────────────────────────────────────
def _plot_comparison(df: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(df["Model"], df[metric] * 100, color=color, edgecolor="white", width=0.5)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)")
        ax.set_xticks(range(len(df["Model"])))
        ax.set_xticklabels(df["Model"], rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Comparison – Brain Tumour Classification", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = REPORTS_DIR / "model_comparison.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Comparison chart saved → {save_path}")


# ── Main Evaluation ────────────────────────────────────────────────────────────
def evaluate_all(model_names: list = None) -> pd.DataFrame:
    """
    Evaluate all models on the test set.

    Returns:
        DataFrame with per-model metrics.
    """
    if model_names is None:
        model_names = MODEL_NAMES

    results = []
    all_reports = {}

    for model_name in model_names:
        print(f"\n{'='*55}")
        print(f"  EVALUATING: {model_name.upper()}")
        print("=" * 55)

        try:
            model = _load_trained_model(model_name)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        print("[INFO] Loading test dataset…")
        y_true, images = _load_test_dataset(model_name)
        print(f"[OK] Test set: {len(y_true)} images\n")

        y_pred, _ = _predict(model, images)

        # Metrics (Proposal Section 4.4)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
        f1   = f1_score(y_true, y_pred,        average="weighted", zero_division=0)

        report = classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        print(f"\n  Accuracy : {acc*100:.2f}%")
        print(f"  Precision: {prec*100:.2f}%")
        print(f"  Recall   : {rec*100:.2f}%")
        print(f"  F1-Score : {f1*100:.2f}%")
        print(f"\n  Per-class report:")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

        results.append({
            "Model":     model_name,
            "Accuracy":  acc,
            "Precision": prec,
            "Recall":    rec,
            "F1-Score":  f1,
        })
        all_reports[model_name] = report

        # Plots
        _plot_confusion_matrix(y_true, y_pred, model_name)
        _plot_training_curves(model_name)

        # Free memory
        keras.backend.clear_session()
        del model

    if not results:
        print("[ERROR] No models evaluated. Train models first.")
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = REPORTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Comparison table saved → {csv_path}")

    # Save JSON reports
    rpt_path = REPORTS_DIR / "classification_reports.json"
    with open(rpt_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"[OK] Classification reports saved → {rpt_path}")

    # Comparison chart
    _plot_comparison(df)

    # Print best model
    best = df.iloc[0]
    print(f"\n{'='*55}")
    print(f"  BEST MODEL: {best['Model'].upper()}")
    print(f"  Accuracy : {best['Accuracy']*100:.2f}%")
    print(f"  F1-Score : {best['F1-Score']*100:.2f}%")
    print("=" * 55)

    return df


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Brain Tumour Classification Models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help="Models to evaluate (default: all)",
    )
    args = parser.parse_args()

    df = evaluate_all(args.models)
    if not df.empty:
        print("\n" + df.to_string(index=False))
