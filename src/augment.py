"""
PHASE 6 – DATA AUGMENTATION
==============================
Script: src/augment.py

Implements all augmentation techniques required by the Proposal:
  - Rotation           (±30°)
  - Horizontal Flip
  - Brightness Adjustment
  - Contrast Adjustment
  - Zoom
  - Shift (width & height)

Each technique is implemented as a Keras RandomLayer and combined
into a single Sequential augmentation model used during training.

Why Augmentation?
  Deep learning models overfit when data is limited.
  Augmentation artificially expands the training set by creating
  plausible variations of existing images. This makes the model
  more robust to unseen scanner settings, patient positioning, and
  imaging artifacts.

  Reference: Proposal (Section 4.2 – Phase 2: Data Augmentation)
             Sajjad et al. [8] in Proposal: "Accuracy improved
             significantly with augmentation compared to no augmentation."
"""

import tensorflow as tf
from tensorflow import keras

# ── Individual Augmentation Layers ─────────────────────────────────────────────
#
# 1. ROTATION (±30°)
#    Why: Brain MRI scans may be acquired at slightly different angles
#         due to patient head positioning. Rotation invariance allows the
#         model to correctly classify tumours regardless of orientation.
#
# 2. HORIZONTAL FLIP
#    Why: A tumour on the left side of the brain should be classified
#         the same as its mirrored counterpart on the right side (type
#         classification is location-agnostic for Glioma/Meningioma).
#         Doubles effective dataset size.
#
# 3. ZOOM (0.8×–1.2×)
#    Why: Tumours vary greatly in size. Training on zoomed images
#         ensures the model generalises to small, medium, and large
#         tumour presentations.
#
# 4. WIDTH/HEIGHT SHIFT (±10%)
#    Why: MRI scans are not always perfectly centred. Small translations
#         mimic real scan variation, preventing the model from relying
#         on absolute spatial position.
#
# 5. BRIGHTNESS ADJUSTMENT (±30%)
#    Why: Different MRI scanners produce images with varying intensity
#         levels. Brightness variation simulates different scanner
#         settings and contrast agent concentrations.
#
# 6. CONTRAST ADJUSTMENT (0.6–1.4)
#    Why: Contrast in MRI images depends on pulse sequence parameters
#         (T1/T2 weighting). Training on varied contrast images improves
#         robustness across scanner protocols.


def build_augmentation_pipeline(
    rotation_factor: float = 0.085,   # ≈ ±30° as fraction of 360°
    zoom_factor: float = 0.15,
    shift_factor: float = 0.10,
    brightness_factor: float = 0.30,
    contrast_lower: float = 0.70,
    contrast_upper: float = 1.30,
) -> keras.Sequential:
    """
    Build and return a Keras Sequential data augmentation model.

    This model is applied ONLY during training (training=True), not
    during validation or inference.

    Returns:
        keras.Sequential augmentation model.
    """
    augmentation = keras.Sequential(
        [
            # ── Geometric Augmentations ────────────────────────────────────────
            keras.layers.RandomRotation(
                factor=rotation_factor,
                fill_mode="nearest",
                name="random_rotation",
            ),
            keras.layers.RandomFlip(
                mode="horizontal",
                name="random_flip",
            ),
            keras.layers.RandomZoom(
                height_factor=zoom_factor,
                width_factor=zoom_factor,
                fill_mode="nearest",
                name="random_zoom",
            ),
            keras.layers.RandomTranslation(
                height_factor=shift_factor,
                width_factor=shift_factor,
                fill_mode="nearest",
                name="random_shift",
            ),
            # ── Photometric Augmentations ──────────────────────────────────────
            keras.layers.RandomBrightness(
                factor=brightness_factor,
                value_range=(0.0, 1.0),
                name="random_brightness",
            ),
            keras.layers.RandomContrast(
                factor=(contrast_upper - contrast_lower) / 2,
                name="random_contrast",
            ),
        ],
        name="augmentation_pipeline",
    )
    return augmentation


# ── Convenience wrapper ────────────────────────────────────────────────────────
def augment_batch(batch: tf.Tensor, augmentation_model: keras.Sequential) -> tf.Tensor:
    """
    Apply augmentation to a batch of images during training.

    Args:
        batch:             Float32 tensor (batch, H, W, 3) in [0, 1].
        augmentation_model: The Sequential augmentation model.

    Returns:
        Augmented float32 tensor.
    """
    return augmentation_model(batch, training=True)


if __name__ == "__main__":
    aug = build_augmentation_pipeline()
    aug.summary()
    print("\n[OK] Augmentation pipeline built successfully.")
