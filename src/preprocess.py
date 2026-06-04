"""
PHASE 5 – IMAGE PREPROCESSING
================================
Script: src/preprocess.py

Implements every preprocessing step required by the Proposal:
  1. Image loading
  2. Grayscale handling
  3. Image resizing (224x224)
  4. Gaussian filtering  → smooth high-frequency noise
  5. Median filtering    → remove impulse noise, preserve edges
  6. Skull stripping     → contour-based brain extraction via OpenCV
  7. Normalization       → Min-Max scaling to [0,1]

Reference:
  - Proposal Section 4.1 (Phase 1: Data Acquisition and Preprocessing)
  - Conference Paper: Intensity normalization using Median/IQR
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
RAW_DIR      = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

IMG_SIZE     = 224          # Proposal: resize to 224×224
CHANNELS     = 3            # Replicate grayscale 3× for pre-trained models

# Normalised class names (must match folder names in raw dataset)
CLASS_MAP = {
    "glioma":     "glioma",
    "meningioma": "meningioma",
    "pituitary":  "pituitary",
    "notumor":    "notumor",
}
CLASSES = list(CLASS_MAP.keys())

# Train / Val / Test split ratios (Proposal Section 4.4)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # remainder after train+val


# ── Step 1: Image Loading ──────────────────────────────────────────────────────
def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    Why:  Proposal mandates MRI image loading as first step.
    From: Proposal (Section 4.1, Step 1).

    Returns:
        BGR image as uint8 ndarray.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img


# ── Step 2: Grayscale Handling ─────────────────────────────────────────────────
def to_grayscale_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale, then replicate to 3 channels.

    Why:  MRI images are grayscale. Pre-trained CNN backbones (VGG16,
          ResNet50, EfficientNet) expect 3-channel input. Replicating
          the single grayscale channel three times preserves intensity
          information while satisfying the model's input shape.
    From: Proposal (Section 4.1, Step 2).

    Returns:
        3-channel ndarray (H, W, 3) in grayscale.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.merge([gray, gray, gray])   # Replicate to 3 channels
    return rgb


# ── Step 3: Image Resizing ─────────────────────────────────────────────────────
def resize_image(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    Resize the image to (size × size).

    Why:  All CNN backbones require a fixed spatial dimension. The
          Proposal specifies 224×224 to match ImageNet-trained models.
    From: Proposal (Section 4.1, Step 2).

    Returns:
        Resized ndarray (size, size, C).
    """
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ── Step 4: Gaussian Filtering ────────────────────────────────────────────────
def apply_gaussian_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to smooth high-frequency noise.

    Why:  MRI scans often contain Gaussian noise from the acquisition
          process. Gaussian filtering smooths the image and reduces the
          impact of random pixel fluctuations, helping the model focus on
          structural features rather than noise.
    From: Proposal (Section 4.1, Step 3 – Gaussian Filter).

    Args:
        kernel_size: Size of the Gaussian kernel (must be odd).

    Returns:
        Smoothed ndarray.
    """
    ksize = (kernel_size, kernel_size)
    return cv2.GaussianBlur(img, ksize, sigmaX=0)


# ── Step 5: Median Filtering ──────────────────────────────────────────────────
def apply_median_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Median filter to remove impulse/salt-and-pepper noise.

    Why:  Median filtering is superior to Gaussian filtering for impulse
          noise because it replaces each pixel with the median of its
          neighbourhood, preserving sharp edges (including tumor boundaries).
          Preserving edges is crucial for accurate tumor demarcation.
          The Conference Paper also uses intensity normalisation with
          Median statistics to handle inhomogeneity in MRI images.
    From: Proposal (Section 4.1, Step 3 – Median Filter).
          Conference Paper (Experimental Setup, Preprocessing).

    Returns:
        Filtered ndarray.
    """
    return cv2.medianBlur(img, kernel_size)


# ── Step 6: Skull Stripping ───────────────────────────────────────────────────
def skull_strip(img: np.ndarray) -> np.ndarray:
    """
    Remove the skull and non-brain tissues using a contour-based approach.

    Algorithm:
      1. Convert to single-channel grayscale.
      2. Apply Otsu's thresholding to create binary mask.
      3. Apply morphological closing to fill holes in the mask.
      4. Find contours; keep the largest (= brain region).
      5. Draw a filled mask for the largest contour.
      6. Apply the mask to the original image.
      7. Crop to bounding box of the brain region.
      8. Resize back to IMG_SIZE × IMG_SIZE.

    Why:  The skull and surrounding non-brain tissue do not contribute
          to tumour classification and introduce irrelevant features.
          Removing them allows the CNN to focus exclusively on brain
          parenchyma, improving generalisation.
    From: Proposal (Section 4.1, Step 4 – Skull Stripping).

    Returns:
        Skull-stripped ndarray (IMG_SIZE, IMG_SIZE, C).
    """
    # Work on a grayscale copy for thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Otsu's threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours on the closed mask
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No contours found – return resized original
        return resize_image(img, IMG_SIZE)

    # Keep the largest contour (brain region)
    largest = max(contours, key=cv2.contourArea)

    # Build a filled mask from the largest contour
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)

    # Apply mask to the original (3-channel) image
    if len(img.shape) == 3:
        mask3 = cv2.merge([mask, mask, mask])
        stripped = cv2.bitwise_and(img, mask3)
    else:
        stripped = cv2.bitwise_and(img, mask)

    # Crop to bounding box of the brain
    x, y, w, h = cv2.boundingRect(largest)
    if w > 0 and h > 0:
        stripped = stripped[y:y + h, x:x + w]

        # Pad to square to preserve aspect ratio
        diff = abs(w - h)
        top, bottom, left, right = 0, 0, 0, 0
        if w > h:
            top = diff // 2
            bottom = diff - top
        else:
            left = diff // 2
            right = diff - left
        stripped = cv2.copyMakeBorder(stripped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Resize back to target size
    stripped = resize_image(stripped, IMG_SIZE)
    return stripped


# ── Step 7: Normalisation ─────────────────────────────────────────────────────
def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Apply Min-Max normalisation to scale pixel values to [0.0, 1.0].

    Why:  Neural network training is numerically more stable when inputs
          are in a small, consistent range. Min-Max normalisation maps
          pixel values from [0, 255] to [0.0, 1.0], speeding convergence
          and preventing exploding/vanishing gradients.
          The Conference Paper also uses intensity normalisation (Median
          + IQR) as part of its preprocessing pipeline.
    From: Proposal (Section 4.1, Step 5 – Normalization).
          Conference Paper (Experimental Setup, Preprocessing).

    Returns:
        Float32 ndarray in range [0.0, 1.0].
    """
    img_f = img.astype(np.float32)
    min_v, max_v = img_f.min(), img_f.max()
    if max_v - min_v > 0:
        img_f = (img_f - min_v) / (max_v - min_v)
    else:
        img_f = np.zeros_like(img_f, dtype=np.float32)
    return img_f


# ── Full Preprocessing Pipeline ────────────────────────────────────────────────
def preprocess_image(image_path: str, return_stages: bool = False):
    """
    Execute the complete preprocessing pipeline on a single image.

    Pipeline:
      Load → Grayscale (3-ch) → Resize → Gaussian Filter
           → Median Filter → Skull Strip → Normalize

    Args:
        image_path:    Path to the raw MRI image.
        return_stages: If True, return a dict of intermediate stages
                       (used by the Streamlit UI for visualisation).

    Returns:
        Normalised float32 ndarray (IMG_SIZE, IMG_SIZE, 3) if
        return_stages is False, else a dict with all intermediate arrays.
    """
    raw   = load_image(image_path)
    gray3 = to_grayscale_rgb(raw)
    resized = resize_image(gray3, IMG_SIZE)
    gauss   = apply_gaussian_filter(resized, kernel_size=3)
    median  = apply_median_filter(gauss, kernel_size=3)
    stripped = skull_strip(median)
    norm     = normalize_image(stripped)

    if return_stages:
        return {
            "raw":      raw,
            "gray3":    gray3,
            "resized":  resized,
            "gaussian": gauss,
            "median":   median,
            "stripped": stripped,
            "normalized": norm,
        }
    return norm


# ── Dataset Preparation ────────────────────────────────────────────────────────
def _collect_paths_and_labels(raw_dir: Path):
    """
    Scan raw_dir/Training folder and collect (path, class_name) pairs.
    Falls back to scanning raw_dir directly if no Training subfolder.
    """
    training_dir = raw_dir / "Training"
    if not training_dir.exists():
        training_dir = raw_dir

    paths, labels = [], []
    for cls_folder in sorted(training_dir.iterdir()):
        if not cls_folder.is_dir():
            continue
        cls_name = cls_folder.name.lower()
        if cls_name not in CLASS_MAP:
            continue
        for img_file in cls_folder.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                paths.append(str(img_file))
                labels.append(CLASS_MAP[cls_name])

    # Also collect Testing folder images (we re-split ourselves)
    testing_dir = raw_dir / "Testing"
    if testing_dir.exists():
        for cls_folder in sorted(testing_dir.iterdir()):
            if not cls_folder.is_dir():
                continue
            cls_name = cls_folder.name.lower()
            if cls_name not in CLASS_MAP:
                continue
            for img_file in cls_folder.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    paths.append(str(img_file))
                    labels.append(CLASS_MAP[cls_name])

    return paths, labels


def prepare_dataset(force: bool = False) -> Path:
    """
    Preprocess all raw images and organise them into
    data/processed/{train,val,test}/{class} folders.

    Split: 70% train, 15% val, 15% test  (Proposal Section 4.4)

    Args:
        force: Re-process even if processed directory already exists.

    Returns:
        Path to the processed dataset root.
    """
    if PROCESSED_DIR.exists() and any(PROCESSED_DIR.iterdir()) and not force:
        print(f"[INFO] Processed dataset already exists at: {PROCESSED_DIR}")
        print("[INFO] Pass force=True to re-process.")
        return PROCESSED_DIR

    if force and PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Collecting image paths from raw dataset…")
    paths, labels = _collect_paths_and_labels(RAW_DIR)

    if not paths:
        raise FileNotFoundError(
            f"No images found under {RAW_DIR}. "
            "Please run src/download.py first."
        )

    print(f"[INFO] Total images found: {len(paths)}")

    # ── Stratified split ───────────────────────────────────────────────────────
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels,
        train_size=TRAIN_RATIO,
        stratify=labels,
        random_state=42,
    )
    # Second split: val vs test  (50/50 of the remaining 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,     # 50% of 30% = 15%
        stratify=y_temp,
        random_state=42,
    )

    print(f"[INFO] Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    splits = {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }

    # ── Process and save ───────────────────────────────────────────────────────
    for split_name, (img_paths, img_labels) in splits.items():
        print(f"\n[INFO] Processing '{split_name}' split ({len(img_paths)} images)…")
        for img_path, label in tqdm(zip(img_paths, img_labels), total=len(img_paths)):
            save_dir = PROCESSED_DIR / split_name / label
            save_dir.mkdir(parents=True, exist_ok=True)

            filename = Path(img_path).name
            save_path = save_dir / filename

            try:
                processed = preprocess_image(img_path)
                # Save as PNG to preserve precision
                save_arr = (processed * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(save_path), save_arr)
            except Exception as e:
                print(f"\n[WARNING] Skipping {img_path}: {e}")

    print(f"\n[OK] Processed dataset saved to: {PROCESSED_DIR}")
    _print_split_summary()
    return PROCESSED_DIR


def _print_split_summary():
    """Print a table of image counts per split and class."""
    print("\n" + "=" * 55)
    print("  PROCESSED DATASET SUMMARY")
    print("=" * 55)
    for split in ["train", "val", "test"]:
        split_dir = PROCESSED_DIR / split
        if not split_dir.exists():
            continue
        total = 0
        print(f"\n  [{split.upper()}]")
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                n = len(list(cls_dir.glob("*")))
                print(f"    {cls_dir.name:<15}: {n:>5} images")
                total += n
        print(f"    {'TOTAL':<15}: {total:>5} images")
    print("=" * 55)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prepare_dataset()
