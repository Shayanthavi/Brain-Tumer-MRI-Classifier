"""
PHASE 8 – TRAINING PIPELINE
==============================
Script: src/train.py

Trains all four models with the strategy required by the Proposal
(Section 4.4 – Phase 4: Training and Evaluation Strategy):

  Dataset split : 70% Train | 15% Val | 15% Test
  Optimizer     : Adam
  Loss          : Categorical Cross-Entropy
  Batch size    : 32 (adjustable)
  Epochs        : up to 50 with Early Stopping (patience=5)

Training is two-stage for Transfer Learning models:
  Stage 1 → Train head only (frozen backbone)
  Stage 2 → Unfreeze top layers & fine-tune at low LR

All metrics, weights, and training history are saved to disk.
"""

import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.models import get_model, MODEL_BUILDERS, get_preprocess_input
from src.augment import build_augmentation_pipeline

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
LOGS_DIR      = BASE_DIR / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE     = 224
BATCH_SIZE   = 32        # Proposal: 32 or 64 depending on GPU memory
STAGE1_EPOCHS = 1        # Head training
STAGE2_EPOCHS = 1        # Fine-tuning
STAGE1_LR    = 1e-3
STAGE2_LR    = 1e-5
EARLY_STOP_PATIENCE = 5  # Proposal: patience of 5 epochs
NUM_CLASSES  = 4

CLASS_NAMES  = ["glioma", "meningioma", "notumor", "pituitary"]


# ── Dataset Loaders ────────────────────────────────────────────────────────────
def _make_dataset(split: str, model_name: str, augment: bool = False) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from the processed images for a given split.

    Args:
        split:   'train', 'val', or 'test'.
        augment: Apply augmentation if True (training only).

    Returns:
        Batched and prefetched tf.data.Dataset.
    """
    split_dir = PROCESSED_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Processed directory '{split_dir}' not found. "
            "Run src/preprocess.py first."
        )

    ds = keras.utils.image_dataset_from_directory(
        directory=str(split_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=(split == "train"),
        seed=42,
    )

    preprocess_fn = get_preprocess_input(model_name)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug_model = build_augmentation_pipeline()
        ds = ds.map(
            lambda x, y: (aug_model(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if model_name == "custom_cnn":
        ds = ds.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (preprocess_fn(x * 255.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def load_datasets(model_name: str):
    """Load train, val, and test tf.data.Dataset objects."""
    print("[INFO] Loading datasets…")
    train_ds = _make_dataset("train", model_name, augment=True)
    val_ds   = _make_dataset("val",   model_name, augment=False)
    test_ds  = _make_dataset("test",  model_name, augment=False)

    # Count samples
    train_n = sum(1 for _ in train_ds.unbatch())
    val_n   = sum(1 for _ in val_ds.unbatch())
    test_n  = sum(1 for _ in test_ds.unbatch())
    print(f"[OK] Datasets loaded → Train: {train_n} | Val: {val_n} | Test: {test_n}\n")
    return train_ds, val_ds, test_ds


# ── Callbacks ─────────────────────────────────────────────────────────────────
def _make_callbacks(model_name: str, stage: int) -> list:
    """
    Create training callbacks.

    Includes:
      - EarlyStopping (patience=5, Proposal requirement)
      - ModelCheckpoint (save best weights)
      - ReduceLROnPlateau (reduce LR on plateau)
      - TensorBoard logging
    """
    weights_path = str(MODELS_DIR / f"{model_name}_best.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / model_name / f"stage{stage}"),
            histogram_freq=0,
        ),
    ]
    return callbacks


# ── Single Model Training ──────────────────────────────────────────────────────
def train_model(
    model_name: str,
    train_ds: tf.data.Dataset,
    val_ds:   tf.data.Dataset,
) -> dict:
    """
    Train a single model using a two-stage strategy (for Transfer Learning)
    or a single stage (Custom CNN).

    Returns:
        dict with training history and final validation metrics.
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_name.upper()}")
    print("=" * 60)
    t0 = time.time()

    is_transfer = (model_name != "custom_cnn")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"[INFO] Stage 1: Training classification head…")
    model = get_model(model_name, fine_tune=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS if is_transfer else STAGE1_EPOCHS + STAGE2_EPOCHS,
        callbacks=_make_callbacks(model_name, stage=1),
        verbose=1,
    )

    history = {
        "stage1_acc":     hist1.history["accuracy"],
        "stage1_val_acc": hist1.history["val_accuracy"],
        "stage1_loss":    hist1.history["loss"],
        "stage1_val_loss":hist1.history["val_loss"],
    }

    # ── Stage 2 (Transfer Learning only) ──────────────────────────────────────
    if is_transfer:
        print(f"\n[INFO] Stage 2: Fine-tuning top layers at LR={STAGE2_LR}…")
        # Rebuild with fine-tune layers unfrozen
        model_ft = get_model(model_name, fine_tune=True)
        # Copy weights from Stage 1 best model
        best_weights_path = str(MODELS_DIR / f"{model_name}_best.keras")
        if Path(best_weights_path).exists():
            model_ft.load_weights(best_weights_path)

        model_ft.compile(
            optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        hist2 = model_ft.fit(
            train_ds,
            validation_data=val_ds,
            epochs=STAGE2_EPOCHS,
            callbacks=_make_callbacks(model_name, stage=2),
            verbose=1,
        )

        history.update({
            "stage2_acc":     hist2.history["accuracy"],
            "stage2_val_acc": hist2.history["val_accuracy"],
            "stage2_loss":    hist2.history["loss"],
            "stage2_val_loss":hist2.history["val_loss"],
        })
        model = model_ft

    elapsed = time.time() - t0
    print(f"\n[OK] {model_name} trained in {elapsed/60:.1f} minutes.")

    # Save final model
    final_path = str(MODELS_DIR / f"{model_name}_final.keras")
    model.save(final_path)
    print(f"[OK] Model saved → {final_path}")

    # Save history
    hist_path = MODELS_DIR / f"{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


# ── Train All Models ───────────────────────────────────────────────────────────
def train_all(model_names: list = None) -> dict:
    """
    Train all specified models and return a combined history dict.

    Args:
        model_names: List of model names to train. Defaults to all 4.

    Returns:
        dict mapping model_name → training history.
    """
    if model_names is None:
        model_names = list(MODEL_BUILDERS.keys())

    all_histories = {}
    for name in model_names:
        train_ds, val_ds, test_ds = load_datasets(name)
        history = train_model(name, train_ds, val_ds)
        all_histories[name] = history

    print("\n" + "=" * 60)
    print("  ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Models saved to: {MODELS_DIR}")

    # Save combined summary
    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_histories, f, indent=2)

    return all_histories


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Brain Tumour Classification Models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_BUILDERS.keys()),
        choices=list(MODEL_BUILDERS.keys()),
        help="Models to train (default: all)",
    )
    args = parser.parse_args()

    print("\n[INFO] Starting training pipeline…")
    print(f"[INFO] Models to train: {args.models}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] Stage 1 epochs: {STAGE1_EPOCHS} | Stage 2 epochs: {STAGE2_EPOCHS}")
    print(f"[INFO] Early stopping patience: {EARLY_STOP_PATIENCE}\n")

    train_all(args.models)
