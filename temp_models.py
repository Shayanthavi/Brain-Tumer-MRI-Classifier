"""
PHASE 7 – MODEL IMPLEMENTATION
================================
Script: src/models.py

Implements all four models required by the Proposal (Section 4.3):

  Model 1 – Custom CNN
    ─ Lightweight architecture inspired by Badzˇa et al. [10]
    ─ 4 convolutional blocks with increasing filters
    ─ Dropout for regularisation

  Model 2 – VGG16 Transfer Learning
    ─ ImageNet-pretrained backbone, frozen during Stage 1
    ─ Last 4 layers unfrozen during Stage 2 fine-tuning
    ─ Custom classification head (GAP → Dense → Softmax)

  Model 3 – ResNet50 Transfer Learning
    ─ ImageNet-pretrained backbone (He et al. [11])
    ─ Last residual block unfrozen during Stage 2
    ─ Custom classification head

  Model 4 – EfficientNetB0 Transfer Learning
    ─ ImageNet-pretrained backbone (Tan & Le [12])
    ─ Optimised accuracy-vs-efficiency trade-off
    ─ Custom classification head

Conference Paper Relation:
  The Conference Paper (Parisot et al.) uses Gentle AdaBoost with
  hand-crafted features. We replace that with end-to-end CNNs that
  learn features automatically, which the Proposal recommends.
  The paper's concept of spatial feature extraction (Gabor textures,
  symmetry features) is reflected in our multi-scale convolutional
  feature extractors in the Custom CNN and Transfer Learning heads.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
NUM_CLASSES = 4
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


def get_preprocess_input(model_name: str):
  """Return the correct preprocessing function for a model name."""
  if model_name == "custom_cnn":
    return lambda images: tf.cast(images, tf.float32) / 255.0
  if model_name == "vgg16":
    return lambda images: vgg16_preprocess_input(tf.cast(images, tf.float32))
  if model_name == "resnet50":
    return lambda images: resnet50_preprocess_input(tf.cast(images, tf.float32))
  if model_name == "efficientnet":
    return lambda images: efficientnet_preprocess_input(tf.cast(images, tf.float32))
  raise ValueError(f"Unknown model name '{model_name}'")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 – CUSTOM CNN
# ══════════════════════════════════════════════════════════════════════════════
def build_custom_cnn(num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Build a lightweight custom CNN for brain tumour classification.

    Architecture:
      INPUT (224×224×3)
       ↓
      [Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)] × 1   → 112×112×32
       ↓
      [Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2)] × 1   → 56×56×64
       ↓
      [Conv2D(128, 3×3) → BN → ReLU → MaxPool(2×2)] × 1  → 28×28×128
       ↓
      [Conv2D(256, 3×3) → BN → ReLU → MaxPool(2×2)] × 1  → 14×14×256
       ↓
      GlobalAveragePooling2D                                → 256
       ↓
      Dense(512, ReLU) → Dropout(0.5)                      → 512
       ↓
      Dense(num_classes, Softmax)                           → 4

    Why included:
      Provides a baseline comparison. The Proposal explicitly requires
      a Custom CNN as Approach A (Section 4.3.1). Inspired by Badzˇa
      et al. [10] who showed that simpler networks can be effective.

    Relation to Conference Paper:
      Unlike the AdaBoost + hand-crafted features approach, this CNN
      learns spatial features (edges, textures, tumour shapes) end-to-end.
    """
    inputs = keras.Input(shape=INPUT_SHAPE, name="input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="CustomCNN")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 – VGG16 TRANSFER LEARNING
# ══════════════════════════════════════════════════════════════════════════════
def build_vgg16(num_classes: int = NUM_CLASSES, fine_tune: bool = False) -> keras.Model:
    """
    Build a VGG16-based transfer learning model.

    Architecture:
      VGG16 backbone (ImageNet weights, include_top=False)
       ↓
      GlobalAveragePooling2D
       ↓
      Dense(512, ReLU) → BatchNorm → Dropout(0.5)
       ↓
      Dense(256, ReLU) → Dropout(0.3)
       ↓
      Dense(num_classes, Softmax)

    Fine-tuning strategy (Proposal Section 4.3.2):
      Stage 1: All VGG16 layers frozen. Train head only. LR = 1e-3.
      Stage 2: Last 4 VGG16 layers unfrozen. Fine-tune. LR = 1e-5.

    Why included:
      The Proposal explicitly lists VGG16 as one of the required Transfer
      Learning models (Section 4.3.2). Reference: Swati et al. [2] achieved
      94.82% accuracy with VGG19-based transfer learning.

    Relation to Conference Paper:
      Replaces AdaBoost's hand-crafted Gabor features with automatically
      learned hierarchical feature maps via VGG16 convolutions.
    """
    base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )

    # Stage 1: freeze all backbone layers
    for layer in base.layers:
        layer.trainable = False

    # Stage 2: unfreeze last 4 layers for fine-tuning
    if fine_tune:
        for layer in base.layers[-4:]:
            layer.trainable = True

    inputs = keras.Input(shape=INPUT_SHAPE, name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="VGG16_TransferLearning")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 – RESNET50 TRANSFER LEARNING
# ══════════════════════════════════════════════════════════════════════════════
def build_resnet50(num_classes: int = NUM_CLASSES, fine_tune: bool = False) -> keras.Model:
    """
    Build a ResNet50-based transfer learning model.

    Architecture:
      ResNet50 backbone (ImageNet weights, include_top=False)
       ↓
      GlobalAveragePooling2D
       ↓
      Dense(512, ReLU) → BatchNorm → Dropout(0.5)
       ↓
      Dense(256, ReLU) → Dropout(0.3)
       ↓
      Dense(num_classes, Softmax)

    Fine-tuning strategy:
      Stage 1: Freeze all ResNet50 layers. Train head. LR = 1e-3.
      Stage 2: Unfreeze last residual block (≈17 layers). LR = 1e-5.

    Why included:
      The Proposal explicitly requires ResNet50 (Section 4.3.2, Gap 1).
      ResNet50 introduced residual connections (He et al. [11]) that allow
      training very deep networks without vanishing gradients. More
      computationally efficient than VGG16 with better performance.

    Relation to Conference Paper:
      ResNet50's residual blocks capture multi-scale spatial features
      similar in spirit to the multi-scale Gabor features in the paper,
      but learned end-to-end rather than hand-crafted.
    """
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )

    for layer in base.layers:
        layer.trainable = False

    if fine_tune:
        # Unfreeze last 17 layers (last ResNet block: conv5_block*)
        for layer in base.layers[-17:]:
            layer.trainable = True

    inputs = keras.Input(shape=INPUT_SHAPE, name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ResNet50_TransferLearning")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4 – EFFICIENTNETB0 TRANSFER LEARNING
# ══════════════════════════════════════════════════════════════════════════════
def build_efficientnet(num_classes: int = NUM_CLASSES, fine_tune: bool = False) -> keras.Model:
    """
    Build an EfficientNetB0-based transfer learning model.

    Architecture:
      EfficientNetB0 backbone (ImageNet weights, include_top=False)
       ↓
      GlobalAveragePooling2D
       ↓
      Dense(512, ReLU) → BatchNorm → Dropout(0.5)
       ↓
      Dense(256, ReLU) → Dropout(0.3)
       ↓
      Dense(num_classes, Softmax)

    Fine-tuning strategy:
      Stage 1: Freeze all EfficientNetB0 layers. Train head. LR = 1e-3.
      Stage 2: Unfreeze last 20 layers (top blocks). LR = 1e-5.

    Why included:
      The Proposal explicitly requires EfficientNet (Section 4.3.2, Gap 1).
      EfficientNet (Tan & Le [12]) optimises model scaling in depth, width,
      and resolution simultaneously. Achieves top accuracy with far fewer
      parameters than VGG16 or ResNet50 → more efficient.

    Relation to Conference Paper:
      Addresses Gap 1 from the Proposal: "High-accuracy models like VGG19
      are heavy. There is a need for efficient architectures like EfficientNet."
    """
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )

    for layer in base.layers:
        layer.trainable = False

    if fine_tune:
        for layer in base.layers[-20:]:
            layer.trainable = True

    inputs = keras.Input(shape=INPUT_SHAPE, name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB0_TransferLearning")
    return model


# ── Model Registry ─────────────────────────────────────────────────────────────
MODEL_BUILDERS = {
    "custom_cnn":  build_custom_cnn,
    "vgg16":       build_vgg16,
    "resnet50":    build_resnet50,
    "efficientnet": build_efficientnet,
}

def get_model(name: str, fine_tune: bool = False) -> keras.Model:
    """
    Return a compiled model by name.

    Args:
        name:      One of 'custom_cnn', 'vgg16', 'resnet50', 'efficientnet'.
        fine_tune: If True, build with Stage-2 layers unfrozen.

    Returns:
        Uncompiled keras.Model.
    """
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_BUILDERS.keys())}")
    if name == "custom_cnn":
        return MODEL_BUILDERS[name]()
    return MODEL_BUILDERS[name](fine_tune=fine_tune)


# ── Print Summary ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for model_name in MODEL_BUILDERS:
        print(f"\n{'='*60}")
        print(f"  {model_name.upper()}")
        print("=" * 60)
        m = get_model(model_name)
        m.summary()
        trainable = sum(tf.size(w).numpy() for w in m.trainable_weights)
        total     = sum(tf.size(w).numpy() for w in m.weights)
        print(f"  Trainable params: {trainable:,}")
        print(f"  Total params:     {total:,}")
