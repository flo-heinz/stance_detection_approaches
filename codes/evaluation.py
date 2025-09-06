"""
This script was generated with the assistance of ChatGPT based on the following
kinds of prompts/instructions:

1) Goal
   - "Evaluate stance prediction results from multiple models and compare both
      classification and regression performance."

2) Inputs & outputs
   - "Read several prediction JSON files (gold + predicted). Save a plaintext
      evaluation summary and PNG figures to dedicated folders."

3) Metrics
   - "Report per-class classification metrics (precision/recall/F1), overall
      Accuracy, Macro-F1, Micro-F1, Weighted-F1; include a confusion matrix."
   - "Compute regression metrics on stance scores: MAE, MSE, Pearson r, Spearman ρ."

4) Plots
   - "Create bar charts for Accuracy/Macro-F1/Micro-F1 on a fixed [0,1] scale."
   - "Plot Weighted-F1 with auto zoom when no ylim is given."
   - "Produce a grouped bar plot of per-class F1 across models (fixed [0,1])."
   - "Combine MAE↓/MSE↓/Pearson↑/Spearman↑ into one grouped plot with global zoom."

5) Usability
   - "Annotate y-axis labels with arrows (↑/↓) to indicate direction of improvement."
   - "Auto-create output folders if they don't exist and print concise progress logs."

6) Robustness
   - "Gracefully skip missing prediction files; avoid crashes when some classes
      are absent (zero_division=0). Keep plots readable with dynamic y-limits."
"""

import os
import json
import io
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    f1_score,
)
from scipy.stats import pearsonr, spearmanr

# file lists and output locations
PREDICTION_FILES = [
    "NLP-Predictions_mistral_zero_shot.json",
    "NLP-Predictions_mistral_chain_of_stance.json",
    "NLP-Predictions_SciBERT-regression.json",
    "NLP-Predictions_mistral_few_shot_10.json",
    "NLP-Predictions_mistral_few_shot_30.json",
    "NLP-Predictions_mistral_few_shot_50.json",
]

# fixed class ordering used throughout reports and plots
STANCE_CATEGORIES = [
    "Strongly Contra", "Contra", "Neutral", "Pro", "Strongly Pro", "Irrelevant"
]

OUTPUT_TEXT_DIR = "outputs"
OUTPUT_FIG_DIR = "figures"
OUTPUT_EVAL_FILE = os.path.join(OUTPUT_TEXT_DIR, "evaluation_results.txt")

os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)


# map continuous stance score to a discrete category for evaluation
def score_to_category(score: float) -> str:
    if score is None or abs(score) < 1e-9:
        return "Irrelevant"
    if score <= -0.75:
        return "Strongly Contra"
    if score <= -0.25:
        return "Contra"
    if score < 0.25:
        return "Neutral"
    if score < 0.75:
        return "Pro"
    return "Strongly Pro"


# save current matplotlib figure and close
def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# dynamic y-limits for readability (unless hard bounds provided)
def _apply_zoom_ylim(values: List[float], hard_bounds: Tuple[float, float] | None = None):
    if hard_bounds is not None:
        plt.ylim(*hard_bounds)
        return
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isfinite(vmin) and np.isfinite(vmax):
        if vmin == vmax:
            pad = 0.05 if vmax == 0 else abs(vmax) * 0.1
            plt.ylim(vmin - pad, vmax + pad)
        else:
            span = vmax - vmin
            pad = max(0.02, span * 0.1)
            plt.ylim(vmin - pad, vmax + pad)


# single-metric bar plot across models (auto-zoom if no ylim)
def plot_model_bars(
    model_names: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    outpath: str,
    ylim: Tuple[float, float] | None = None,
):
    plt.figure(figsize=(8.8, 5.0))
    x = np.arange(len(model_names))
    plt.bar(x, values)
    plt.xticks(x, model_names, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    _apply_zoom_ylim(values, hard_bounds=ylim)
    _savefig(outpath)


# grouped bars (e.g., per-class F1 across models)
def plot_grouped_bars(
    x_labels: List[str],
    series: Dict[str, List[float]],
    title: str,
    ylabel: str,
    outpath: str,
):
    n_groups = len(x_labels)
    legends = list(series.keys())
    n_series = len(legends)

    x = np.arange(n_groups)
    width = 0.8 / max(n_series, 1)

    plt.figure(figsize=(max(10, n_groups * 1.3), 5.4))
    for i, name in enumerate(legends):
        plt.bar(x + (i - (n_series - 1) / 2) * width, series[name], width, label=name)

    plt.xticks(x, x_labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)  # per-class F1 on common [0,1] scale
    plt.legend()
    _savefig(outpath)


# evaluate a single predictions file (returns printable text, summary dict, and per-class F1)
def evaluate_file(file_path: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    method_name = os.path.splitext(os.path.basename(file_path))[0]
    out = io.StringIO()
    out.write(f"\n{'='*60}\nEvaluation: {method_name}\n")

    # unpack gold/pred
    gold_scores = [r["gold_stance"] for r in results]
    pred_scores = [r["predicted_stance_score"] for r in results]
    gold_labels = [score_to_category(s) for s in gold_scores]
    pred_labels = [r["predicted_stance_category"] for r in results]

    # classification report
    rep_dict = classification_report(
        gold_labels, pred_labels, labels=STANCE_CATEGORIES,
        digits=3, zero_division=0, output_dict=True
    )
    rep_text = classification_report(
        gold_labels, pred_labels, labels=STANCE_CATEGORIES,
        digits=3, zero_division=0
    )
    out.write("\nClassification Report:\n")
    out.write(rep_text + "\n")

    # aggregated classification metrics
    acc = accuracy_score(gold_labels, pred_labels)
    macro_f1 = rep_dict["macro avg"]["f1-score"]
    weighted_f1 = rep_dict["weighted avg"]["f1-score"]
    micro_f1 = f1_score(gold_labels, pred_labels, labels=STANCE_CATEGORIES, average="micro", zero_division=0)

    out.write(f"Accuracy ↑: {acc:.3f}\n")
    out.write(f"F1 (macro) ↑: {macro_f1:.3f}\n")
    out.write(f"F1 (micro) ↑: {micro_f1:.3f}\n")
    out.write(f"F1 (weighted) ↑: {weighted_f1:.3f}\n")

    # confusion matrix
    cm = confusion_matrix(gold_labels, pred_labels, labels=STANCE_CATEGORIES)
    out.write("\nConfusion Matrix (rows = gold, cols = predicted):\n")
    out.write("\t" + "\t".join(STANCE_CATEGORIES) + "\n")
    for label, row in zip(STANCE_CATEGORIES, cm):
        out.write(f"{label}\t" + "\t".join(map(str, row)) + "\n")

    # regression metrics on continuous scores
    mae = mean_absolute_error(gold_scores, pred_scores)
    mse = mean_squared_error(gold_scores, pred_scores)
    pearson_corr = pearsonr(gold_scores, pred_scores)[0]
    spearman_corr = spearmanr(gold_scores, pred_scores)[0]

    out.write("\nRegression Metrics:\n")
    out.write(f"MAE ↓: {mae:.3f}\n")
    out.write(f"MSE ↓: {mse:.3f}\n")
    out.write(f"Pearson r ↑: {pearson_corr:.3f}\n")
    out.write(f"Spearman ρ ↑: {spearman_corr:.3f}\n")

    per_class_f1 = {cls: rep_dict.get(cls, {}).get("f1-score", 0.0) for cls in STANCE_CATEGORIES}

    summary = {
        "model": method_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "mae": float(mae),
        "mse": float(mse),
        "pearson": float(pearson_corr),
        "spearman": float(spearman_corr),
    }

    return out.getvalue(), summary, per_class_f1


# combined grouped plot for MAE↓/MSE↓/Pearson↑/Spearman↑ with global y-zoom
def plot_regression_metrics_all_in_one(
    short_names: List[str],
    summaries: List[Dict[str, Any]],
    outpath: str
):
    metrics = [("MAE ↓", "mae"), ("MSE ↓", "mse"), ("Pearson r ↑", "pearson"), ("Spearman ρ ↑", "spearman")]
    x = np.arange(len(metrics))
    n_models = len(short_names)
    width = 0.8 / max(n_models, 1)

    # collect values to set a sensible y-range (allow negatives for correlations)
    all_vals = [s[key] for s in summaries for _, key in metrics]

    plt.figure(figsize=(10.5, 6.0))
    for i, model_name in enumerate(short_names):
        vals = [summaries[i][key] for _, key in metrics]
        plt.bar(x + (i - (n_models - 1) / 2) * width, vals, width, label=model_name)

    plt.xticks(x, [lab for lab, _ in metrics])
    plt.ylabel("Metric value")
    plt.title("Regression Metrics Comparison Across Models (↓ lower is better, ↑ higher is better)")
    _apply_zoom_ylim(all_vals, hard_bounds=None)
    plt.legend()
    _savefig(outpath)


def main():
    # collect text reports, summaries, and per-class F1 across all models
    all_text_blocks: List[str] = []
    summaries: List[Dict[str, Any]] = []
    per_model_class_f1: Dict[str, Dict[str, float]] = {}

    for path in PREDICTION_FILES:
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        text, summary, per_cls = evaluate_file(path)
        print(text)
        all_text_blocks.append(text)
        summaries.append(summary)
        short = summary["model"].replace("NLP-Predictions_", "")
        per_model_class_f1[short] = per_cls

    # save the combined textual evaluation
    with open(OUTPUT_EVAL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text_blocks))
    print(f"\nAll evaluations saved to: {OUTPUT_EVAL_FILE}")

    if not summaries:
        print("No summaries to plot.")
        return

    short_names = [s["model"].replace("NLP-Predictions_", "") for s in summaries]

    # accuracy / macro-f1 / micro-f1 on fixed [0,1] scales
    plot_model_bars(short_names, [s["accuracy"] for s in summaries], "Accuracy ↑",
                    "Model Accuracy Comparison", os.path.join(OUTPUT_FIG_DIR, "accuracy.png"), ylim=(0, 1))
    plot_model_bars(short_names, [s["macro_f1"] for s in summaries], "F1 (macro) ↑",
                    "Model Macro-F1 Comparison", os.path.join(OUTPUT_FIG_DIR, "f1_macro.png"), ylim=(0, 1))
    plot_model_bars(short_names, [s["micro_f1"] for s in summaries], "F1 (micro) ↑",
                    "Model Micro-F1 Comparison", os.path.join(OUTPUT_FIG_DIR, "f1_micro.png"), ylim=(0, 1))

    # weighted-f1 with auto-zoom for clearer differences
    plot_model_bars(short_names, [s["weighted_f1"] for s in summaries], "F1 (weighted) ↑",
                    "Model Weighted-F1 Comparison", os.path.join(OUTPUT_FIG_DIR, "f1_weighted.png"), ylim=None)

    # per-class F1 grouped bars (fixed [0,1])
    series = {model: [f1s.get(cls, 0.0) for cls in STANCE_CATEGORIES] for model, f1s in per_model_class_f1.items()}
    plot_grouped_bars(STANCE_CATEGORIES, series, "Per-class F1 across models", "F1-score ↑",
                      os.path.join(OUTPUT_FIG_DIR, "per_class_f1.png"))

    # regression metrics (global zoom; supports negatives)
    plot_regression_metrics_all_in_one(short_names, summaries, os.path.join(OUTPUT_FIG_DIR, "regression_metrics.png"))

    print(f"Figures saved to: {OUTPUT_FIG_DIR}")


if __name__ == "__main__":
    main()
