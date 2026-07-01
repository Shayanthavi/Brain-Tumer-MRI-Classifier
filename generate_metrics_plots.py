"""
generate_metrics_plots.py
==========================
Regenerate ALL evaluation-metric plots with pure **matplotlib**, straight from
the saved evaluation artefacts — no trained models, no TensorFlow required:

    reports/model_comparison.csv          (weighted acc/precision/recall/F1 per model)
    reports/classification_reports.json   (per-class precision/recall/F1/support)

This is handy for reports, slides and viva/demo prep: you can rebuild every
chart in a second without re-running training or evaluation.

Outputs (written to reports/):
    overall_metrics_comparison.png   grouped bars: acc/precision/recall/F1 per model
    accuracy_leaderboard.png         horizontal ranked bars (best model on top)
    per_class_f1_comparison.png      grouped bars: F1 per class, every model
    per_class_metrics_<best>.png     precision/recall/F1 per class for the best model
    per_class_f1_heatmap.png         matplotlib heatmap: models (rows) x classes (cols)

Note: Confusion matrices and training curves are NOT regenerated here — those
need the trained models / saved history (`python -m src.evaluate`). Everything
in THIS script is reconstructable from the two artefact files above.

Usage:
    python generate_metrics_plots.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend – just saves PNGs, no window
import matplotlib.pyplot as plt

# ── Paths & constants ────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
REPORTS  = ROOT / "reports"
CLASSES  = ["glioma", "meningioma", "notumor", "pituitary"]
METRICS  = ["Accuracy", "Precision", "Recall", "F1-Score"]

# One stable colour per model and per metric (colour-blind friendly Tableau set)
MODEL_COLORS  = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
METRIC_COLORS = ["#59a14f", "#edc948", "#b07aa1", "#ff9da7"]

REPORTS.mkdir(parents=True, exist_ok=True)


# ── Load the saved evaluation artefacts ──────────────────────────────────────────
def load_artifacts():
    """Read model_comparison.csv and classification_reports.json.

    Returns:
        rows    : list of dict rows from the CSV (already sorted best→worst)
        models  : list of model names in that order
        reports : dict {model: sklearn-style classification report dict}
    """
    csv_path  = REPORTS / "model_comparison.csv"
    json_path = REPORTS / "classification_reports.json"

    if not csv_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing evaluation artefacts.\n"
            f"  expected: {csv_path}\n"
            f"            {json_path}\n"
            f"Run `python -m src.evaluate` first to produce them."
        )

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    with open(json_path) as f:
        reports = json.load(f)

    models = [r["Model"] for r in rows]        # CSV is pre-sorted by accuracy
    return rows, models, reports


def _bar_labels(ax, bars, fmt="{:.1f}", dy=0.6, size=8):
    """Write the numeric value on top of each bar."""
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom", fontsize=size)


# ── 1. Overall weighted metrics, grouped by model ────────────────────────────────
def plot_overall_metrics(rows, models):
    x  = np.arange(len(models))
    w  = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, metric in enumerate(METRICS):
        vals = [float(r[metric]) * 100 for r in rows]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=metric, color=METRIC_COLORS[i])
        _bar_labels(ax, bars, fmt="{:.1f}", dy=0.5, size=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Overall Weighted Metrics by Model (held-out test set)", fontweight="bold")
    ax.legend(ncol=4, loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = REPORTS / "overall_metrics_comparison.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")


# ── 2. Accuracy leaderboard (horizontal, best on top) ────────────────────────────
def plot_accuracy_leaderboard(rows, models):
    accs = [float(r["Accuracy"]) * 100 for r in rows]
    # reverse so the highest accuracy sits at the TOP of a horizontal bar chart
    order = list(reversed(range(len(models))))
    names = [models[i].upper() for i in order]
    vals  = [accs[i] for i in order]
    cols  = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, vals, color=cols, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.2f}%",
                va="center", ha="left", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("Model Accuracy Leaderboard", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = REPORTS / "accuracy_leaderboard.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")


# ── 3. Per-class F1 for every model (grouped bars) ───────────────────────────────
def plot_per_class_f1(models, reports):
    x = np.arange(len(CLASSES))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, m in enumerate(models):
        f1s  = [reports[m][c]["f1-score"] * 100 for c in CLASSES]
        bars = ax.bar(x + i * w - 0.4 + w / 2, f1s, w,
                      label=m.upper(), color=MODEL_COLORS[i % len(MODEL_COLORS)])
        _bar_labels(ax, bars, fmt="{:.0f}", dy=0.6, size=7)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylabel("F1-Score (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Per-Class F1-Score by Model", fontweight="bold")
    ax.legend(title="Model", ncol=len(models))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = REPORTS / "per_class_f1_comparison.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")


# ── 4. Precision / Recall / F1 per class, for the BEST model ──────────────────────
def plot_best_model_per_class(best, reports):
    metrics = ["precision", "recall", "f1-score"]
    x  = np.arange(len(CLASSES))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, met in enumerate(metrics):
        vals = [reports[best][c][met] * 100 for c in CLASSES]
        bars = ax.bar(x + (i - 1) * w, vals, w,
                      label=met.capitalize(), color=METRIC_COLORS[i])
        _bar_labels(ax, bars, fmt="{:.0f}", dy=0.6, size=7)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.set_title(f"Per-Class Precision / Recall / F1 — {best.upper()} (best model)",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = REPORTS / f"per_class_metrics_{best}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")


# ── 5. Per-class F1 heatmap (matplotlib imshow) ──────────────────────────────────
def plot_f1_heatmap(models, reports):
    # rows = models (best at top), cols = classes
    grid = np.array([[reports[m][c]["f1-score"] * 100 for c in CLASSES] for m in models])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.upper() for m in models])

    # Annotate each cell; pick black/white text for contrast against the fill
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            val = grid[r, c]
            ax.text(c, r, f"{val:.0f}", ha="center", va="center",
                    color="white" if val > 60 else "black", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("F1-Score (%)")
    ax.set_title("Per-Class F1-Score Heatmap (model × class)", fontweight="bold")
    fig.tight_layout()
    out = REPORTS / "per_class_f1_heatmap.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    rows, models, reports = load_artifacts()
    best = models[0]                       # CSV sorted best→worst by accuracy

    print(f"[INFO] Models found : {', '.join(models)}")
    print(f"[INFO] Best model   : {best} "
          f"(acc={float(rows[0]['Accuracy'])*100:.2f}%)\n")

    plot_overall_metrics(rows, models)
    plot_accuracy_leaderboard(rows, models)
    plot_per_class_f1(models, reports)
    plot_best_model_per_class(best, reports)
    plot_f1_heatmap(models, reports)

    print(f"\n[DONE] 5 plots (re)generated in {REPORTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
