"""
PHASE 3 – DATASET ACQUISITION (Kaggle API)
=========================================
Script: src/download.py

Downloads the Brain Tumor MRI Dataset from the official Kaggle dataset
using the Kaggle API. This implementation intentionally avoids any Hugging
Face mirrors or caches and follows the Proposal requirement to use the
Kaggle Brain Tumor MRI Dataset as the canonical source.

Requirements:
 - The `kaggle` CLI package must be installed and configured. Place
   your `kaggle.json` in `~/.kaggle/kaggle.json` or run `kaggle configure`.

Behavior:
 - Uses Kaggle API only.
 - Downloads and extracts the dataset archive.
 - Organizes classes into: `glioma`, `meningioma`, `pituitary`, `notumor`.
 - Handles safe re-downloads on Windows (removes existing raw folder when
   `force=True`).
"""

import os
import shutil
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

# Official Kaggle dataset id. This mirrors the Proposal's intended source.
# If this dataset id changes, update the constant below.
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"

REQUIRED_CLASSES = {"glioma", "meningioma", "pituitary", "notumor"}


def _rmtree_win_safe(path: Path):
    """Remove tree with Windows permission retries."""
    def _onerror(func, path_str, exc_info):
        try:
            os.chmod(path_str, 0o777)
            func(path_str)
        except Exception:
            pass

    shutil.rmtree(path, onerror=_onerror)


def download_dataset(force: bool = False) -> Path:
    """
    Download the Kaggle Brain Tumor MRI Dataset using the Kaggle API.

    Args:
        force: If True, remove any existing `data/raw` and re-download.

    Returns:
        Path to the raw dataset folder.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError(
            "The 'kaggle' package is required. Install with 'pip install kaggle' "
            "and configure your API token (place kaggle.json in ~/.kaggle/)."
        ) from e

    if RAW_DIR.exists() and any(RAW_DIR.iterdir()) and not force:
        print(f"[INFO] Dataset already exists at: {RAW_DIR}")
        print("[INFO] Pass force=True to re-download.")
        return RAW_DIR

    if force and RAW_DIR.exists():
        print("[INFO] Removing existing dataset for re-download…")
        try:
            _rmtree_win_safe(RAW_DIR)
        except Exception:
            shutil.rmtree(RAW_DIR, ignore_errors=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"[INFO] Downloading '{KAGGLE_DATASET}' from Kaggle…")
    print("[INFO] This may take several minutes depending on your connection speed.")

    # Download and unzip into a temporary folder inside RAW_DIR
    target_dir = RAW_DIR / "_kaggle_download"
    if target_dir.exists():
        _rmtree_win_safe(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        # dataset_download_files will produce a single zip file and optionally
        # extract depending on the 'unzip' flag. We use unzip=True to extract
        # directly into our target directory.
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(target_dir),
            unzip=True,
            quiet=False,
        )
    except Exception as e:
        # Clean up partial download
        print(f"[ERROR] Kaggle download failed: {e}")
        try:
            _rmtree_win_safe(target_dir)
        except Exception:
            pass
        raise

    # The extraction may create subfolders; discover the top-level folder
    extracted_root = None
    for child in sorted(target_dir.iterdir()):
        # Skip hidden/system files
        if child.name.startswith("."):
            continue
        # If file (images zipped), keep target_dir as root
        if child.is_dir():
            extracted_root = child
            break

    if extracted_root is None:
        extracted_root = target_dir

    print(f"[INFO] Organising downloaded dataset from: {extracted_root}")

    # Move/organise class folders into RAW_DIR (flattening if necessary)
    moved_any = False
    for root, dirs, files in os.walk(extracted_root):
        for d in dirs:
            name = d.lower()
            if name in REQUIRED_CLASSES:
                src = Path(root) / d
                dest = RAW_DIR / d
                if dest.exists():
                    _rmtree_win_safe(dest)
                shutil.move(str(src), str(dest))
                print(f"[OK] Moved class folder: {d}")
                moved_any = True

    # If we didn't find class folders, try a common layout: Training/Testing
    if not moved_any:
        # Look for Training or training folder
        candidates = [extracted_root / "Training", extracted_root / "training", extracted_root]
        found = False
        for cand in candidates:
            if cand.exists():
                for cls in REQUIRED_CLASSES:
                    # Some datasets use 'no_tumor' or 'No' variants; we perform a case-insensitive search
                    matched = None
                    for sub in cand.iterdir():
                        if sub.is_dir() and sub.name.lower().replace(" ", "") in [cls, cls.replace('notumor','no_tumor'), 'no_tumor']:
                            matched = sub
                            break
                    if matched:
                        dest = RAW_DIR / matched.name
                        if dest.exists():
                            _rmtree_win_safe(dest)
                        shutil.move(str(matched), str(dest))
                        print(f"[OK] Organized class folder: {matched.name}")
                        found = True
                if found:
                    break

    # Final verification: ensure required class folders exist under RAW_DIR
    present = {d.name.lower() for d in RAW_DIR.iterdir() if d.is_dir()}
    missing = REQUIRED_CLASSES.difference(present)
    if missing:
        print(f"[WARNING] After extraction, missing class folders: {sorted(missing)}")
        print("[INFO] Listing top-level folders in raw: ")
        for d in sorted(RAW_DIR.iterdir()):
            print("   ", d.name)
    else:
        print(f"[OK] All required classes found: {sorted(REQUIRED_CLASSES)}")

    # Remove temporary download folder if it still exists
    try:
        if target_dir.exists():
            _rmtree_win_safe(target_dir)
    except Exception:
        pass

    print(f"\n[OK] Dataset ready at: {RAW_DIR}\n")
    return RAW_DIR


def verify_dataset(raw_dir: Optional[Path] = None) -> bool:
    """
    Verify that the downloaded dataset has the required class folders.
    """
    if raw_dir is None:
        raw_dir = RAW_DIR
    print("[INFO] Verifying dataset structure…")
    if not raw_dir.exists():
        print(f"[ERROR] Raw directory not found: {raw_dir}")
        return False

    found_classes = {d.name.lower() for d in raw_dir.iterdir() if d.is_dir()}
    missing = REQUIRED_CLASSES.difference(found_classes)
    if missing:
        print(f"[ERROR] Missing class folders: {sorted(missing)}")
        print(f"[INFO] Found folders: {sorted(found_classes)}")
        return False

    # Print counts
    total = 0
    for cls in sorted(found_classes):
        cls_dir = raw_dir / cls
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        print(f"    {cls:<15} → {len(imgs):>5} images")
        total += len(imgs)
    print(f"\n[OK] Total images in raw: {total}\n")
    return True


def print_dataset_summary(raw_dir: Optional[Path] = None):
    if raw_dir is None:
        raw_dir = RAW_DIR
    print("=" * 55)
    print(" BRAIN TUMOR MRI DATASET SUMMARY (Kaggle source)")
    print("=" * 55)

    if not raw_dir.exists():
        print(f"[INFO] Raw directory not found: {raw_dir}")
        return

    for cls_dir in sorted(raw_dir.iterdir()):
        if cls_dir.is_dir():
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
            print(f"    {cls_dir.name:<15}: {len(imgs):>5} images")

    print("=" * 55)


if __name__ == "__main__":
    raw = download_dataset()
    ok = verify_dataset(raw)
    print_dataset_summary(raw)
