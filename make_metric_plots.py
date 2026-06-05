"""
make_metric_plots.py
====================
Generates per-class metric plots (pure matplotlib) from the authoritative
evaluation artefacts produced by src/evaluate.py:
    reports/classification_reports.json
    reports/model_comparison.csv

Outputs (reports/):
    per_class_f1_comparison.png       grouped bars: F1 per class, all models
    per_class_metrics_<best>.png      precision/recall/F1 for the best model
    overall_metrics_comparison.png    accuracy/precision/recall/F1 per model
"""
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

reports = json.load(open(REPORTS / "classification_reports.json"))
rows = list(csv.DictReader(open(REPORTS / "model_comparison.csv")))
models = [r["Model"] for r in rows]                       # already sorted by accuracy
best = models[0]

# ── 1. Per-class F1 across all models (grouped bars) ─────────────────────────────
x = np.arange(len(CLASSES))
w = 0.8 / len(models)
colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
fig, ax = plt.subplots(figsize=(11, 6))
for i, m in enumerate(models):
    f1s = [reports[m][c]["f1-score"] * 100 for c in CLASSES]
    bars = ax.bar(x + i * w - 0.4 + w / 2, f1s, w, label=m, color=colors[i % len(colors)])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES])
ax.set_ylabel("F1-Score (%)"); ax.set_ylim(0, 105)
ax.set_title("Per-Class F1-Score by Model", fontweight="bold")
ax.legend(title="Model"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(REPORTS / "per_class_f1_comparison.png", dpi=150); plt.close(fig)
print("[OK] reports/per_class_f1_comparison.png")

# ── 2. Precision / Recall / F1 per class for the best model ───────────────────────
metrics = ["precision", "recall", "f1-score"]
mcolors = ["#59a14f", "#edc948", "#b07aa1"]
fig, ax = plt.subplots(figsize=(10, 6))
w2 = 0.25
for i, met in enumerate(metrics):
    vals = [reports[best][c][met] * 100 for c in CLASSES]
    bars = ax.bar(x + (i - 1) * w2, vals, w2, label=met.capitalize(), color=mcolors[i])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES])
ax.set_ylabel("Score (%)"); ax.set_ylim(0, 105)
ax.set_title(f"Per-Class Precision / Recall / F1 — {best.upper()} (best model)", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(REPORTS / f"per_class_metrics_{best}.png", dpi=150); plt.close(fig)
print(f"[OK] reports/per_class_metrics_{best}.png")

# ── 3. Overall metrics per model (grouped bars) ──────────────────────────────────
omets = ["Accuracy", "Precision", "Recall", "F1-Score"]
fig, ax = plt.subplots(figsize=(11, 6))
xm = np.arange(len(models))
w3 = 0.2
for i, met in enumerate(omets):
    vals = [float(r[met]) * 100 for r in rows]
    bars = ax.bar(xm + (i - 1.5) * w3, vals, w3, label=met, color=colors[i])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(xm); ax.set_xticklabels(models)
ax.set_ylabel("Score (%)"); ax.set_ylim(0, 105)
ax.set_title("Overall Weighted Metrics by Model (test set)", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(REPORTS / "overall_metrics_comparison.png", dpi=150); plt.close(fig)
print("[OK] reports/overall_metrics_comparison.png")

print(f"\nBest model: {best}  (acc={float(rows[0]['Accuracy'])*100:.2f}%)")
