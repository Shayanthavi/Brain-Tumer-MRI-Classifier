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

Bug Fixes Applied:
  [BUG-1] FIXED: Removed CosineDecay LearningRateSchedule from Stage 2.
          CosineDecay + ReduceLROnPlateau is incompatible – ReduceLROnPlateau
          calls optimizer.lr.assign() which raises TypeError when LR is a
          Schedule object. Stage 2 now uses a fixed float LR (1e-5).
  [BUG-2] FIXED: Augmented images are now clipped to [0.0, 1.0] before
          model-specific preprocessing. RandomBrightness can push values
          slightly above 1.0, causing silent training instability.
  [BUG-3] FIXED: Stage 2 now loads the complete Stage 1 saved model via
          keras.models.load_model(), then unfreezes the top layers. The
          previous load_weights() on a fresh differently-configured model
          risked silent weight mismatches.
  [BUG-7] FIXED: Stage 1 epochs increased from 20→30, patience from 5→8
          to allow the head to properly converge before Stage 2 fine-tuning.
"""

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

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
BATCH_SIZE   = 32           # Proposal: 32 or 64 depending on GPU memory
# [BUG-7 FIX] Increased Stage 1 epochs from 20→30 and patience from 5→8
# so the classification head can properly converge before Stage 2.
STAGE1_EPOCHS = 30          # Head training (was 20)
STAGE2_EPOCHS = 30          # Fine-tuning
CNN_EPOCHS   = 80           # Custom CNN needs more epochs to converge from scratch
STAGE1_LR    = 1e-3
# [BUG-1 FIX] STAGE2_LR must be a plain float, NOT a LearningRateSchedule.
# ReduceLROnPlateau (used in callbacks) calls optimizer.lr.assign(new_lr),
# which raises TypeError if the optimizer was built with a Schedule object.
STAGE2_LR    = 1e-5         # Fixed float – no Schedule (was CosineDecay)
CNN_LR       = 5e-4         # More conservative LR for training CNN from scratch
# [BUG-7 FIX] Increased patience values for better convergence
EARLY_STOP_PATIENCE = 8     # Stage 1 & 2 patience (was 5)
CNN_EARLY_STOP_PATIENCE = 12  # Custom CNN needs more patience (was 10)
NUM_CLASSES  = 4

CLASS_NAMES  = ["glioma", "meningioma", "notumor", "pituitary"]


# ── Dataset Loaders ────────────────────────────────────────────────────────────
def _make_dataset(split: str, model_name: str, augment: bool = False) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from the processed images for a given split.

    Args:
        split:      'train', 'val', or 'test'.
        model_name: Model name for correct preprocessing function.
        augment:    Apply augmentation if True (training only).

    Returns:
        Batched and prefetched tf.data.Dataset.

    Pipeline order:
        1. Load images (uint8 0-255) → cast to float32 → divide by 255 → [0,1]
        2. (Training only) Apply augmentation in [0,1] space
        3. [BUG-2 FIX] Clip to [0.0, 1.0] to handle brightness overflow
        4. Apply model-specific preprocessing:
           - custom_cnn: identity (already [0,1])
           - VGG16/ResNet50/EfficientNet: rescale back to [0,255] then
             apply the backbone's channel-mean subtraction
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

    # Step 1: Normalize to [0, 1]
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Step 2: Augmentation (training only)
    if augment:
        aug_model = build_augmentation_pipeline()
        ds = ds.map(
            lambda x, y: (aug_model(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # [BUG-2 FIX] Step 3: Clip to [0, 1] — RandomBrightness can produce
    # values slightly outside [0, 1], causing numerical instability.
    ds = ds.map(
        lambda x, y: (tf.clip_by_value(x, 0.0, 1.0), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Step 4: Model-specific preprocessing
    if model_name == "custom_cnn":
        # Custom CNN uses [0, 1] normalized inputs directly
        pass
    else:
        # Transfer learning models expect their backbone's preprocessing
        # (channel-mean subtraction). Rescale back to [0, 255] range first.
        ds = ds.map(
            lambda x, y: (preprocess_fn(x * 255.0), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(tf.data.AUTOTUNE)


def _compute_class_weights(train_ds: tf.data.Dataset) -> dict:
    """
    Compute balanced class weights from the training dataset.

    Iterates once through the unbatched dataset to collect all labels,
    then uses sklearn's compute_class_weight with 'balanced' strategy.
    This gives higher weight to under-represented classes (Glioma,
    Meningioma) to counteract class imbalance.

    Returns:
        dict mapping class index (int) → weight (float).
    """
    print("[INFO] Computing class weights (single dataset pass)...")
    y_true_all = []
    for _, y_b in train_ds.unbatch():
        y_true_all.append(int(np.argmax(y_b.numpy())))

    y_arr = np.array(y_true_all)
    unique_classes = np.unique(y_arr)
    weights_arr = compute_class_weight("balanced", classes=unique_classes, y=y_arr)
    class_weights = {int(c): float(w) for c, w in zip(unique_classes, weights_arr)}

    print(f"[INFO] Class weights: { {CLASS_NAMES[k]: f'{v:.3f}' for k, v in class_weights.items()} }")
    return class_weights


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
def _make_callbacks(model_name: str, stage: int, patience: int = EARLY_STOP_PATIENCE,
                    use_reduce_lr: bool = True) -> list:
    """
    Create training callbacks.

    [BUG-1 FIX] ReduceLROnPlateau is only included when use_reduce_lr=True,
    which requires the optimizer to use a plain float learning rate (not a
    LearningRateSchedule). Stage 2 now always uses a plain float LR, so
    ReduceLROnPlateau is safe to use for both stages.

    Includes:
      - EarlyStopping (monitors val_loss for stable convergence)
      - ModelCheckpoint (save best weights by val_loss)
      - ReduceLROnPlateau (reduce LR on plateau — only with float LR)
      - TensorBoard logging
    """
    weights_path = str(MODELS_DIR / f"{model_name}_stage{stage}_best.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min",
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / model_name / f"stage{stage}"),
            histogram_freq=0,
        ),
    ]

    # [BUG-1 FIX] Only add ReduceLROnPlateau when using a plain float LR.
    # When optimizer is built with a LearningRateSchedule, ReduceLROnPlateau
    # calls optimizer.lr.assign() which raises:
    #   TypeError: This optimizer was created with a LearningRateSchedule
    #   object as its learning_rate constructor argument, hence its learning
    #   rate is not settable.
    # Since Stage 2 now uses a fixed float LR, this is always safe.
    if use_reduce_lr:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
                mode="min",
            )
        )

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

    # ── Compute class weights once ─────────────────────────────────────────
    class_weights = _compute_class_weights(train_ds)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"[INFO] Stage 1: Training classification head…")
    model = get_model(model_name, fine_tune=False)

    # Use different hyperparameters for Custom CNN vs Transfer Learning
    if model_name == "custom_cnn":
        lr      = CNN_LR
        epochs  = CNN_EPOCHS
        patience = CNN_EARLY_STOP_PATIENCE
    else:
        lr      = STAGE1_LR
        epochs  = STAGE1_EPOCHS
        patience = EARLY_STOP_PATIENCE

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    print(f"[INFO] LR={lr}, Epochs={epochs}, Patience={patience}")
    # [BUG-1 FIX] use_reduce_lr=True is safe because Stage 1 uses plain float LR
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=_make_callbacks(model_name, stage=1, patience=patience, use_reduce_lr=True),
        class_weight=class_weights,
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

        # [BUG-3 FIX] Load the complete Stage 1 saved model rather than
        # building a fresh model and calling load_weights().
        # Building fresh + load_weights() on a differently-configured model
        # (different layer.trainable flags) can cause silent weight mismatches
        # because the model graph structure differs between stage1 and stage2.
        # Loading the full model preserves the exact architecture + weights,
        # then we unfreeze the desired layers before recompiling.
        best_weights_path = str(MODELS_DIR / f"{model_name}_stage1_best.keras")

        if Path(best_weights_path).exists():
            print(f"[INFO] Loading Stage 1 best model from: {best_weights_path}")
            model_ft = keras.models.load_model(best_weights_path)
        else:
            print("[WARN] Stage 1 checkpoint not found. Using Stage 1 final weights.")
            model_ft = model  # Fallback: use the Stage 1 model in memory

        # Unfreeze backbone top layers for fine-tuning
        # We unfreeze the backbone sub-model (the first layer that is a Model)
        _unfreeze_top_layers(model_ft, model_name)

        # [BUG-1 FIX] Stage 2 uses a PLAIN FLOAT learning rate, NOT a
        # LearningRateSchedule. This is mandatory for ReduceLROnPlateau to
        # work. CosineDecay was removed because it creates an unset-able LR
        # that crashes ReduceLROnPlateau with TypeError.
        model_ft.compile(
            optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LR),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )

        # [BUG-1 FIX] use_reduce_lr=True is now safe — STAGE2_LR is a float
        hist2 = model_ft.fit(
            train_ds,
            validation_data=val_ds,
            epochs=STAGE2_EPOCHS,
            callbacks=_make_callbacks(model_name, stage=2, use_reduce_lr=True),
            class_weight=class_weights,
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

    # Compare stage losses and revert to Stage 1 if it was better
    if is_transfer:
        best_stage1_loss = min(hist1.history["val_loss"])
        best_stage2_loss = min(hist2.history["val_loss"])
        if best_stage1_loss < best_stage2_loss:
            print("[INFO] Stage 1 had lower val_loss. Reverting to Stage 1 best model.")
            model = keras.models.load_model(str(MODELS_DIR / f"{model_name}_stage1_best.keras"))

    # Save final model
    final_path = str(MODELS_DIR / f"{model_name}_final.keras")
    model.save(final_path)
    print(f"[OK] Model saved → {final_path}")

    # Save history
    hist_path = MODELS_DIR / f"{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


def _unfreeze_top_layers(model: tf.keras.Model, model_name: str) -> None:
    """
    Unfreeze the top layers of the backbone for Stage 2 fine-tuning.

    This modifies the model in-place. We find the backbone sub-model
    (the layer whose type is a Model/Functional) and unfreeze its
    top N layers according to model-specific strategy.

    Args:
        model:      The loaded Stage 1 model.
        model_name: Name of the model to determine unfreeze strategy.
    """
    # Find the backbone layer (it is a sub-model embedded in the functional graph)
    backbone = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(getattr(layer, 'layers', [])) > 10:
            backbone = layer
            break

    if backbone is None:
        print("[WARN] Could not find backbone sub-model. Skipping layer unfreeze.")
        return

    # Freeze all backbone layers first, then selectively unfreeze top N
    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze strategy per model
    if model_name == "vgg16":
        # Unfreeze last 8 layers (block4_conv3, block5_conv1/2/3 + their BN)
        n_unfreeze = 8
    elif model_name == "resnet50":
        # Unfreeze last 33 layers (full conv5 block: conv5_block1/2/3)
        n_unfreeze = 33
    elif model_name == "efficientnet":
        # Unfreeze last 50 layers (top MBConv blocks)
        n_unfreeze = 50
    else:
        n_unfreeze = 0

    if n_unfreeze > 0:
        for layer in backbone.layers[-n_unfreeze:]:
            # Skip BatchNormalization in frozen backbone — keep BN statistics stable
            # This prevents catastrophic forgetting from BN mean/variance shifts
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    total_count = len(backbone.layers)
    print(f"[INFO] Backbone layers unfrozen: {trainable_count}/{total_count} (top {n_unfreeze})")


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
