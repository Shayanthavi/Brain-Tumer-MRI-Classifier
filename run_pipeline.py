"""
run_pipeline.py
================
Master pipeline script – runs all phases in order:
  1. Download dataset
  2. Preprocess & prepare dataset
  3. Train all models
  4. Evaluate all models

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-download    (if already downloaded)
    python run_pipeline.py --skip-preprocess  (if already preprocessed)
    python run_pipeline.py --models custom_cnn vgg16  (train subset)
"""

import argparse
import sys
import time
from pathlib import Path

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Brain Tumor Classification – Full Pipeline"
)
parser.add_argument("--skip-download",    action="store_true", help="Skip dataset download")
parser.add_argument("--skip-preprocess",  action="store_true", help="Skip preprocessing")
parser.add_argument("--skip-train",       action="store_true", help="Skip training")
parser.add_argument("--skip-evaluate",    action="store_true", help="Skip evaluation")
parser.add_argument("--force-download",   action="store_true", help="Force re-download")
parser.add_argument("--force-preprocess", action="store_true", help="Force re-preprocessing")
parser.add_argument(
    "--models",
    nargs="+",
    default=["custom_cnn", "vgg16", "resnet50", "efficientnet"],
    choices=["custom_cnn", "vgg16", "resnet50", "efficientnet"],
    help="Models to train and evaluate",
)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Banner ─────────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════╗
║  BRAIN TUMOR DETECTION & CLASSIFICATION PIPELINE         ║
║  University of Ruhuna – EE7204 / EC7205                  ║
║  Deep Learning & Image Processing Mini-Project           ║
╚══════════════════════════════════════════════════════════╝
""")

t_start = time.time()

# ── PHASE 3: Download ──────────────────────────────────────────────────────────
if not args.skip_download:
    print("\n" + "─" * 55)
    print(" PHASE 3: DATASET ACQUISITION")
    print("─" * 55)
    from src.download import download_dataset, verify_dataset, print_dataset_summary
    raw = download_dataset(force=args.force_download)
    verify_dataset(raw)
    print_dataset_summary(raw)
else:
    print("[SKIP] Dataset download phase skipped.")

# ── PHASE 5: Preprocess ────────────────────────────────────────────────────────
if not args.skip_preprocess:
    print("\n" + "─" * 55)
    print(" PHASE 5: IMAGE PREPROCESSING")
    print("─" * 55)
    from src.preprocess import prepare_dataset
    prepare_dataset(force=args.force_preprocess)
else:
    print("[SKIP] Preprocessing phase skipped.")

# ── PHASE 8: Train ─────────────────────────────────────────────────────────────
if not args.skip_train:
    print("\n" + "─" * 55)
    print(f" PHASE 8: TRAINING – {args.models}")
    print("─" * 55)
    from src.train import train_all
    train_all(args.models)
else:
    print("[SKIP] Training phase skipped.")

# ── PHASE 9: Evaluate ──────────────────────────────────────────────────────────
if not args.skip_evaluate:
    print("\n" + "─" * 55)
    print(f" PHASE 9: EVALUATION – {args.models}")
    print("─" * 55)
    from src.evaluate import evaluate_all
    df = evaluate_all(args.models)
    if not df.empty:
        print("\nFinal Comparison Table:")
        print(df.to_string(index=False))
else:
    print("[SKIP] Evaluation phase skipped.")

# ── Done ───────────────────────────────────────────────────────────────────────
elapsed = time.time() - t_start
print(f"""
╔══════════════════════════════════════════════════════════╗
║  PIPELINE COMPLETE  – Total time: {elapsed/60:.1f} minutes           
╟──────────────────────────────────────────────────────────╢
║  Next steps:                                             ║
║    1. Review reports/ directory for evaluation charts    ║
║    2. Launch UI: streamlit run src/app.py                ║
╚══════════════════════════════════════════════════════════╝
""")
