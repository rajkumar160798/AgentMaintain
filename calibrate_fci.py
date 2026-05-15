"""
calibrate_fci.py
================
Compute Drift Evidence Score (DES) and Fault Concentration Index (FCI) for every
decision recorded in traces.json, then plot the FCI distributions by ground-truth
class to empirically calibrate the EGLR routing thresholds.

Run BEFORE the EGLR ablation evaluation:
    python calibrate_fci.py

Outputs
-------
plots/fci_calibration.png   — FCI distribution plot with suggested thresholds
plots/fci_roc.png           — FCI threshold sweep (precision/recall vs threshold)
results/fci_calibration.json — percentile statistics and suggested thresholds
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from agent_tools import compute_drift_evidence_score, compute_fault_concentration_index
from experiment_config import PLOT_CONFIG

TRACES_PATH = "traces.json"
OUTPUT_DIR = "plots"
RESULTS_DIR = "results"


def load_traces(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_fci_for_traces(traces: list) -> pd.DataFrame:
    records = []
    for trace in traces:
        p_values = trace.get("p_values")
        shap_values = trace.get("shap_values")
        if not p_values or not shap_values:
            continue

        des = compute_drift_evidence_score(p_values, shap_values)
        fci = compute_fault_concentration_index(des)

        records.append({
            "model":        trace.get("model", "unknown"),
            "cycle":        trace.get("cycle"),
            "action":       trace.get("action"),
            "ground_truth": trace.get("ground_truth", "unknown"),
            "fci":          fci,
            "des_max":      max(des.values()) if des else 0.0,
            "des_sum":      sum(des.values()) if des else 0.0,
            "top_sensor":   max(des, key=des.get) if des else None,
        })
    return pd.DataFrame(records)


def suggest_thresholds(df: pd.DataFrame) -> dict:
    """
    Find the FCI thresholds that best separate sensor_failure (high FCI)
    from operational_drift (low FCI) using a precision-recall sweep.
    """
    fault_df  = df[df["ground_truth"] == "sensor_failure"]["fci"].values
    drift_df  = df[df["ground_truth"] == "operational_drift"]["fci"].values
    normal_df = df[df["ground_truth"] == "normal"]["fci"].values

    stats = {}
    for name, arr in [("sensor_failure", fault_df), ("operational_drift", drift_df), ("normal", normal_df)]:
        if len(arr) == 0:
            continue
        stats[name] = {
            "n":    int(len(arr)),
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "p10":  float(np.percentile(arr, 10)),
            "p25":  float(np.percentile(arr, 25)),
            "p50":  float(np.percentile(arr, 50)),
            "p75":  float(np.percentile(arr, 75)),
            "p90":  float(np.percentile(arr, 90)),
        }

    # Use p25 of sensor_failure as high threshold candidate (conservative),
    # and p75 of operational_drift as low threshold candidate.
    high_candidate = float(np.percentile(fault_df, 25)) if len(fault_df) > 0 else 0.6
    low_candidate  = float(np.percentile(drift_df, 75)) if len(drift_df) > 0 else 0.15

    # Clamp so they don't overlap
    if high_candidate <= low_candidate:
        high_candidate = low_candidate + 0.1

    return {
        "distributions": stats,
        "suggested_fci_high_threshold": round(high_candidate, 3),
        "suggested_fci_low_threshold":  round(low_candidate, 3),
    }


def plot_fci_distributions(df: pd.DataFrame, high_t: float, low_t: float, out_dir: str):
    sns.set_theme(style=PLOT_CONFIG.get("style", "whitegrid"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: KDE per class ---
    ax = axes[0]
    palette = {
        "sensor_failure":    "#e74c3c",
        "operational_drift": "#2ecc71",
        "normal":            "#3498db",
        "unknown":           "#95a5a6",
    }
    for gt_class, color in palette.items():
        sub = df[df["ground_truth"] == gt_class]["fci"]
        if len(sub) < 2:
            continue
        sns.kdeplot(sub, ax=ax, label=gt_class, color=color, fill=True, alpha=0.3)
        ax.axvline(sub.median(), color=color, linestyle="--", linewidth=1.0)

    ax.axvline(high_t, color="black", linestyle="-", linewidth=2.0,
               label=f"fci_high = {high_t:.3f}")
    ax.axvline(low_t,  color="gray",  linestyle="-", linewidth=2.0,
               label=f"fci_low = {low_t:.3f}")
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                     low_t, high_t, alpha=0.08, color="orange", label="LLM invocation band")
    ax.set_xlabel("Fault Concentration Index (FCI)")
    ax.set_ylabel("Density")
    ax.set_title("FCI Distribution by Ground-Truth Class\n(dashed = class median)")
    ax.legend(fontsize=8)

    # --- Right: Boxplot per class ---
    ax2 = axes[1]
    plot_df = df[df["ground_truth"].isin(palette.keys())].copy()
    order = [c for c in ["sensor_failure", "operational_drift", "normal"] if c in plot_df["ground_truth"].values]
    sns.boxplot(data=plot_df, x="ground_truth", y="fci", hue="ground_truth",
                order=order, palette=palette, legend=False, ax=ax2)
    ax2.axhline(high_t, color="black", linestyle="--", linewidth=1.5,
                label=f"fci_high = {high_t:.3f}")
    ax2.axhline(low_t,  color="gray",  linestyle="--", linewidth=1.5,
                label=f"fci_low = {low_t:.3f}")
    ax2.set_title("FCI Box Plot by Ground-Truth Class")
    ax2.set_xlabel("Ground-Truth Label")
    ax2.set_ylabel("FCI")
    ax2.legend(fontsize=8)

    plt.suptitle(
        "Drift Evidence Score — Fault Concentration Index (FCI) Calibration\n"
        "Orange band = ambiguous region requiring LLM invocation",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fci_calibration.png")
    plt.savefig(out_path, dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved: {out_path}")


def plot_fci_threshold_sweep(df: pd.DataFrame, out_dir: str):
    """
    Sweep FCI threshold (used as a classifier for sensor_failure) and
    plot precision and recall to help select τ_high.
    """
    fault_mask = (df["ground_truth"] == "sensor_failure").astype(int)
    if fault_mask.sum() == 0:
        return

    fci_vals = df["fci"].values
    thresholds = np.linspace(0.0, 1.0, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        pred = (fci_vals >= t).astype(int)
        tp = ((pred == 1) & (fault_mask == 1)).sum()
        fp = ((pred == 1) & (fault_mask == 0)).sum()
        fn = ((pred == 0) & (fault_mask == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    best_idx = int(np.argmax(f1s))
    best_t   = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#e74c3c")
    ax.plot(thresholds, recalls,    label="Recall",    color="#2ecc71")
    ax.plot(thresholds, f1s,        label="F1",        color="#3498db", linewidth=2)
    ax.axvline(best_t, color="black", linestyle="--",
               label=f"Best F1 threshold = {best_t:.3f} (F1={f1s[best_idx]:.3f})")
    ax.set_xlabel("FCI Threshold (fci_high candidate)")
    ax.set_ylabel("Score")
    ax.set_title("FCI Threshold Sweep for Sensor-Failure Detection\n(positive class: sensor_failure)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fci_roc.png")
    plt.savefig(out_path, dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches="tight")
    plt.close()
    print(f"FCI threshold sweep saved: {out_path}")
    return best_t


if __name__ == "__main__":
    if not os.path.exists(TRACES_PATH):
        print(f"ERROR: {TRACES_PATH} not found. Run evaluator.py first.")
        exit(1)

    print("Loading traces...")
    traces = load_traces(TRACES_PATH)
    print(f"Loaded {len(traces)} trace entries.")

    print("Computing DES and FCI for each trace...")
    df = compute_fci_for_traces(traces)
    print(f"Computed FCI for {len(df)} entries.")
    print(df.groupby("ground_truth")["fci"].describe().round(3))

    calibration = suggest_thresholds(df)
    high_t = calibration["suggested_fci_high_threshold"]
    low_t  = calibration["suggested_fci_low_threshold"]

    print("\nSuggested thresholds:")
    print(f"  fci_high (-> ISSUE_REPLACEMENT_TICKET) = {high_t}")
    print(f"  fci_low  (-> RETRAIN_MODEL)             = {low_t}")

    # Plots
    plot_fci_distributions(df, high_t, low_t, OUTPUT_DIR)
    best_t = plot_fci_threshold_sweep(df, OUTPUT_DIR)
    if best_t is not None:
        calibration["best_f1_threshold"] = round(float(best_t), 3)
        print(f"  Best F1 threshold for fault detection = {best_t:.3f}")

    print("\nNOTE: If sensor_failure and operational_drift FCI distributions overlap,")
    print("EGLR bypass will be rare. Report this as an empirical finding.")
    print("Update EGLR_CONFIG thresholds with values from results/fci_calibration.json")

    # Save calibration results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "fci_calibration.json")
    with open(out_json, "w") as f:
        json.dump(calibration, f, indent=4)
    print(f"\nCalibration results saved to {out_json}")
    print("\nNext step: update EGLR_CONFIG in experiment_config.py with these thresholds,")
    print("then run:  python evaluator.py --eglr")
