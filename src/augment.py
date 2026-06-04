"""
PHASE 6 – DATA AUGMENTATION
==============================
Script: src/augment.py

Implements all augmentation techniques required by the Proposal:
  - Rotation           (±15° — tuned down from ±20°)
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

Augmentation Tuning Notes:
  [BUG-6] Aggressive augmentation hurts classification of visually
  ambiguous classes (Glioma, Meningioma). Large rotations, zoom, and
  brightness changes can distort the subtle intensity gradients and
  boundary signatures that distinguish these classes.

  Key changes vs original:
    - Rotation: ±20° → ±15°  (0.055 → 0.042 of 360°)
      Justification: MRI head scans are typically close to upright.
      ±15° covers real acquisition variability without destroying
      orientation-specific features that differentiate tumor types.
    - Zoom: 0.15 → 0.12
      Justification: Extreme zoom changes apparent tumor size too
      aggressively; ±12% is sufficient to cover real size variation.
    - Brightness: 0.30 → 0.20
      Justification: MRI intensity is scanner-dependent but rarely
      varies by >20% in practice. Excessive brightness augmentation
      can wash out Glioma's characteristic hyperintense regions.
    - Contrast: (0.7–1.3) → (0.8–1.2)
      Justification: Reduced range to preserve T1/T2 contrast ratios
      that carry diagnostic information for Meningioma classification.
"""

import tensorflow as tf
from tensorflow import keras

# ── Individual Augmentation Layers ─────────────────────────────────────────────
#
# 1. ROTATION (±15°)
#    Why: Brain MRI scans may be acquired at slightly different angles
#         due to patient head positioning. Rotation invariance allows the
#         model to correctly classify tumours regardless of orientation.
#         Tuned from ±20° to ±15° to preserve orientation-based features
#         that are diagnostic for Glioma vs Meningioma.
#
# 2. HORIZONTAL FLIP
#    Why: A tumour on the left side of the brain should be classified
#         the same as its mirrored counterpart on the right side (type
#         classification is location-agnostic for Glioma/Meningioma).
#         Doubles effective dataset size for left/right symmetry cases.
#
# 3. ZOOM (±12%)
#    Why: Tumours vary greatly in size. Training on zoomed images
#         ensures the model generalises to small, medium, and large
#         tumour presentations. Tuned from ±15% to ±12% to prevent
#         over-zooming that removes context around the tumour boundary.
#
# 4. WIDTH/HEIGHT SHIFT (±10%)
#    Why: MRI scans are not always perfectly centred. Small translations
#         mimic real scan variation, preventing the model from relying
#         on absolute spatial position.
#
# 5. BRIGHTNESS ADJUSTMENT (±20%)
#    Why: Different MRI scanners produce images with varying intensity
#         levels. Brightness variation simulates different scanner
#         settings and contrast agent concentrations.
#         Tuned from ±30% to ±20% to preserve hyperintense regions.
#
# 6. CONTRAST ADJUSTMENT (0.8–1.2)
#    Why: Contrast in MRI images depends on pulse sequence parameters
#         (T1/T2 weighting). Training on varied contrast images improves
#         robustness across scanner protocols. Tuned from (0.7–1.3)
#         to (0.8–1.2) to preserve T1/T2 contrast ratios.


def build_augmentation_pipeline(
    rotation_factor: float = 0.042,   # ≈ ±15° as fraction of 360° (was 0.055 = ±20°)
    zoom_factor: float = 0.12,         # ±12% (was 0.15 = ±15%)
    shift_factor: float = 0.10,        # ±10% (unchanged)
    brightness_factor: float = 0.20,   # ±20% (was 0.30 = ±30%)
    contrast_lower: float = 0.80,      # 0.80–1.20 (was 0.70–1.30)
    contrast_upper: float = 1.20,
) -> keras.Sequential:
    """
    Build and return a Keras Sequential data augmentation model.

    This model is applied ONLY during training (training=True), not
    during validation or inference.

    Important: The output of this pipeline may contain values slightly
    outside [0.0, 1.0] due to RandomBrightness. The training pipeline
    in train.py clips values to [0.0, 1.0] after augmentation.

    Returns:
        keras.Sequential augmentation model.
    """
    augmentation = keras.Sequential(
        [
            # ── Geometric Augmentations ────────────────────────────────────────
            keras.layers.RandomRotation(
                factor=rotation_factor,
                fill_mode="constant",
                fill_value=0.0,
                name="random_rotation",
            ),
            keras.layers.RandomFlip(
                mode="horizontal",
                name="random_flip",
            ),
            keras.layers.RandomZoom(
                height_factor=zoom_factor,
                width_factor=zoom_factor,
                fill_mode="constant",
                fill_value=0.0,
                name="random_zoom",
            ),
            keras.layers.RandomTranslation(
                height_factor=shift_factor,
                width_factor=shift_factor,
                fill_mode="constant",
                fill_value=0.0,
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
        Augmented float32 tensor (may have values slightly outside [0, 1]
        due to RandomBrightness — clip in the calling code).
    """
    return augmentation_model(batch, training=True)


if __name__ == "__main__":
    aug = build_augmentation_pipeline()
    aug.summary()
    print("\n[OK] Augmentation pipeline built successfully.")
    print("[INFO] Rotation:   ±15° (fraction: 0.042)")
    print("[INFO] Zoom:       ±12%")
    print("[INFO] Shift:      ±10%")
    print("[INFO] Brightness: ±20%")
    print("[INFO] Contrast:   0.80–1.20")
