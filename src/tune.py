"""
PHASE 8b – HYPERPARAMETER TUNING (KerasTuner · Hyperband)
========================================================
Script: src/tune.py

Implements the Proposal's stated intent (Section 4.4) that the **15% validation
set is used to tune hyperparameters** — which the plain training script only
*monitors* (early stopping / LR reduction). This module performs an actual
search over the most impactful hyperparameters using KerasTuner's Hyperband
algorithm and the project's existing validation split.

Search space (per model):
  • dense_units_1   ∈ {256, 512, 1024}     classification-head width
  • dense_units_2   ∈ {128, 256}
  • dropout_1       ∈ [0.2 … 0.6]          head regularisation
  • dropout_2       ∈ [0.1 … 0.5]
  • learning_rate   ∈ {1e-3, 5e-4, 1e-4}
  • l2              ∈ {1e-5, 1e-4, 1e-3}   (transfer-learning heads only)

Why these: the convolutional feature extractors (custom conv stack / ImageNet
backbones) are kept fixed — they are the expensive, well-understood part. The
head capacity, regularisation and learning rate are what actually move the
val-accuracy needle on a small medical dataset, and they are cheap to search.

Workflow:
  1. Hyperband search over the space above (frozen backbone for transfer models)
     optimising val_accuracy on the existing 70/15/15 split.
  2. Save the best hyperparameters → models/{model}_best_hp.json
  3. Final training of the best configuration, reusing the exact two-stage
     strategy from src/train.py (Stage 1 head + Stage 2 fine-tune for transfer,
     single long run for the Custom CNN). The resulting model is saved as
     models/{model}_final.keras + models/{model}_history.json — i.e. the SAME
     artefact layout that src/evaluate.py and src/app.py already consume, so the
     tuned model flows through the rest of the pipeline with no other changes.

Run on a GPU/HPC node — Hyperband trains many candidate models and is slow on CPU.

Usage:
    python -m src.tune                         # tune all four models
    python -m src.tune --models custom_cnn     # tune one model
    python -m src.tune --max-epochs 30 --factor 3
    python -m src.tune --no-final-train        # search only, skip final retrain
    python -m src.tune --overwrite             # discard any previous search state
"""

import argparse
import json
import time

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0

from src.models import INPUT_SHAPE, NUM_CLASSES
from src.train import (
    BASE_DIR, MODELS_DIR, CLASS_NAMES,
    STAGE1_EPOCHS, STAGE2_EPOCHS, STAGE2_LR, EARLY_STOP_PATIENCE,
    CNN_EPOCHS, CNN_EARLY_STOP_PATIENCE,
    _make_dataset, _compute_class_weights, _make_callbacks, _unfreeze_top_layers,
)

# ── Constants ──────────────────────────────────────────────────────────────────
TUNING_DIR = BASE_DIR / "tuning"
TUNING_DIR.mkdir(parents=True, exist_ok=True)

_BACKBONES = {
    "vgg16":        VGG16,
    "resnet50":     ResNet50,
    "efficientnet": EfficientNetB0,
}

MODEL_NAMES = ["custom_cnn", "vgg16", "resnet50", "efficientnet"]


# ── Hypermodel building blocks ───────────────────────────────────────────────────
def _build_head(x, hp, use_l2: bool):
    """Tunable classification head shared by all models."""
    init = "he_normal"
    reg = keras.regularizers.l2(hp.Choice("l2", [1e-5, 1e-4, 1e-3])) if use_l2 else None

    units1 = hp.Choice("dense_units_1", [256, 512, 1024])
    units2 = hp.Choice("dense_units_2", [128, 256])
    drop1  = hp.Float("dropout_1", 0.2, 0.6, step=0.1)
    drop2  = hp.Float("dropout_2", 0.1, 0.5, step=0.1)

    x = layers.Dense(units1, activation="relu", kernel_initializer=init,
                     kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Dropout(drop1, name="dropout1")(x)

    x = layers.Dense(units2, activation="relu", kernel_initializer=init,
                     kernel_regularizer=reg, name="fc2")(x)
    x = layers.BatchNormalization(name="bn_fc2")(x)
    x = layers.Dropout(drop2, name="dropout2")(x)

    return layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)


def _build_custom_cnn(hp) -> keras.Model:
    """Custom CNN with the fixed conv stack from models.py + a tunable head."""
    init = "he_normal"
    inputs = keras.Input(shape=INPUT_SHAPE, name="input")
    x = inputs
    for i, filters in enumerate([32, 64, 128, 256, 512], start=1):
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          kernel_initializer=init, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool{i}")(x)

    # 1x1 channel reduction before flatten (same trick as models.py)
    x = layers.Conv2D(64, (1, 1), padding="same", kernel_initializer=init, name="conv_reduce")(x)
    x = layers.BatchNormalization(name="bn_reduce")(x)
    x = layers.Activation("relu", name="relu_reduce")(x)
    x = layers.Flatten(name="flatten")(x)

    outputs = _build_head(x, hp, use_l2=False)
    return keras.Model(inputs, outputs, name="CustomCNN_tuned")


def _build_transfer(hp, model_name: str) -> keras.Model:
    """Transfer-learning model with a FROZEN backbone + tunable head (Stage-1 style)."""
    base = _BACKBONES[model_name](
        weights="imagenet", include_top=False, input_shape=INPUT_SHAPE,
    )
    for layer in base.layers:
        layer.trainable = False

    inputs = keras.Input(shape=INPUT_SHAPE, name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = _build_head(x, hp, use_l2=True)
    return keras.Model(inputs, outputs, name=f"{model_name}_tuned")


def build_model(hp, model_name: str) -> keras.Model:
    """KerasTuner hypermodel: build + compile a candidate for `model_name`."""
    if model_name == "custom_cnn":
        model = _build_custom_cnn(hp)
    elif model_name in _BACKBONES:
        model = _build_transfer(hp, model_name)
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


# ── Final training of the best configuration ─────────────────────────────────────
def _final_train(model_name, best_hp, train_ds, val_ds, class_weights) -> dict:
    """
    Retrain the best hyperparameter configuration using the project's standard
    two-stage strategy, saving artefacts in the same layout as src/train.py.
    """
    print(f"\n[INFO] Final training of best '{model_name}' configuration…")
    model = build_model(best_hp, model_name)

    if model_name == "custom_cnn":
        epochs, patience = CNN_EPOCHS, CNN_EARLY_STOP_PATIENCE
    else:
        epochs, patience = STAGE1_EPOCHS, EARLY_STOP_PATIENCE

    hist1 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=_make_callbacks(model_name, stage=1, patience=patience, use_reduce_lr=True),
        class_weight=class_weights, verbose=1,
    )
    history = {
        "stage1_acc":      hist1.history["accuracy"],
        "stage1_val_acc":  hist1.history["val_accuracy"],
        "stage1_loss":     hist1.history["loss"],
        "stage1_val_loss": hist1.history["val_loss"],
    }

    # Stage 2 fine-tuning for transfer-learning models (mirrors src/train.py)
    if model_name != "custom_cnn":
        stage1_ckpt = MODELS_DIR / f"{model_name}_stage1_best.keras"
        if stage1_ckpt.exists():
            model = keras.models.load_model(str(stage1_ckpt))
        _unfreeze_top_layers(model, model_name)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LR),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )
        hist2 = model.fit(
            train_ds, validation_data=val_ds, epochs=STAGE2_EPOCHS,
            callbacks=_make_callbacks(model_name, stage=2, use_reduce_lr=True),
            class_weight=class_weights, verbose=1,
        )
        history.update({
            "stage2_acc":      hist2.history["accuracy"],
            "stage2_val_acc":  hist2.history["val_accuracy"],
            "stage2_loss":     hist2.history["loss"],
            "stage2_val_loss": hist2.history["val_loss"],
        })

    final_path = MODELS_DIR / f"{model_name}_final.keras"
    model.save(str(final_path))
    with open(MODELS_DIR / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Tuned model saved → {final_path}")
    return history


# ── Tune a single model ──────────────────────────────────────────────────────────
def tune_model(model_name, max_epochs=30, factor=3,
               objective="val_accuracy", final_train=True, overwrite=False) -> dict:
    """
    Run a Hyperband search for `model_name`, save the best hyperparameters, and
    (optionally) retrain + save the best configuration.

    Returns the dict of best hyperparameter values.
    """
    print(f"\n{'='*60}\n  HYPERPARAMETER TUNING: {model_name.upper()}\n{'='*60}")
    t0 = time.time()

    train_ds = _make_dataset("train", model_name, augment=True)
    val_ds   = _make_dataset("val",   model_name, augment=False)
    class_weights = _compute_class_weights(train_ds)

    direction = "max" if "acc" in objective else "min"
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model(hp, model_name),
        objective=kt.Objective(objective, direction),
        max_epochs=max_epochs,
        factor=factor,
        directory=str(TUNING_DIR),
        project_name=model_name,
        overwrite=overwrite,
    )

    print("\n[INFO] Search space:")
    tuner.search_space_summary()

    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, mode="min",
    )
    tuner.search(
        train_ds, validation_data=val_ds,
        epochs=max_epochs, class_weight=class_weights,
        callbacks=[early], verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\n[OK] Best hyperparameters for {model_name}:")
    for k, v in best_hp.values.items():
        print(f"     {k:16}: {v}")

    hp_path = MODELS_DIR / f"{model_name}_best_hp.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(hp_path, "w") as f:
        json.dump(best_hp.values, f, indent=2)
    print(f"[OK] Best hyperparameters saved → {hp_path}")

    if final_train:
        _final_train(model_name, best_hp, train_ds, val_ds, class_weights)

    print(f"[OK] {model_name} tuning finished in {(time.time()-t0)/60:.1f} min.")
    return best_hp.values


# ── Tune all models ──────────────────────────────────────────────────────────────
def tune_all(model_names=None, max_epochs=30, factor=3,
             objective="val_accuracy", final_train=True, overwrite=False) -> dict:
    """Run Hyperband tuning for every requested model."""
    if model_names is None:
        model_names = MODEL_NAMES

    best = {}
    for name in model_names:
        best[name] = tune_model(
            name, max_epochs=max_epochs, factor=factor,
            objective=objective, final_train=final_train, overwrite=overwrite,
        )

    summary_path = MODELS_DIR / "tuning_best_hp_summary.json"
    with open(summary_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n[OK] All tuning complete. Summary → {summary_path}")
    return best


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning (KerasTuner Hyperband)")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES,
                        help="Models to tune (default: all)")
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="Hyperband max epochs per candidate (default: 30)")
    parser.add_argument("--factor", type=int, default=3,
                        help="Hyperband reduction factor (default: 3)")
    parser.add_argument("--objective", default="val_accuracy",
                        help="Search objective (default: val_accuracy)")
    parser.add_argument("--no-final-train", action="store_true",
                        help="Only search; do not retrain/save the best model")
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard any previous search state and start fresh")
    args = parser.parse_args()

    print("\n[INFO] Starting hyperparameter tuning…")
    print(f"[INFO] Models: {args.models}")
    print(f"[INFO] Hyperband max_epochs={args.max_epochs}, factor={args.factor}, objective={args.objective}\n")

    tune_all(
        args.models,
        max_epochs=args.max_epochs,
        factor=args.factor,
        objective=args.objective,
        final_train=not args.no_final_train,
        overwrite=args.overwrite,
    )
