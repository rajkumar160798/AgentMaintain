import time
import json
import os
import argparse
import pynvml
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, bootstrap as scipy_bootstrap
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from graph_builder import build_graph
from experiment_config import (
    MODEL_CONFIGS, EVALUATION_CONFIG, PLOT_CONFIG,
    STREAMING_CONFIG, SENSITIVITY_CONFIGS, EGLR_CONFIG,
)
from monitor import StreamingMonitor


# ---------------------------------------------------------------------------
# VRAM Tracker
# ---------------------------------------------------------------------------

class VRAMTracker:
    def __init__(self):
        self.tracking = False
        self.max_vram = 0
        self.thread = None
        self.handle = None
        self.enabled = True
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print("Failed to init pynvml or no GPU found, disabling VRAM tracking:", e)
            self.enabled = False

    def _track(self):
        self.max_vram = 0
        while self.tracking:
            if self.handle and self.enabled:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                used_mb = info.used / 1024 / 1024
                if used_mb > self.max_vram:
                    self.max_vram = used_mb
            time.sleep(0.05)

    def start(self):
        if not self.enabled:
            return
        self.tracking = True
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.enabled:
            return 0.0
        self.tracking = False
        if self.thread:
            self.thread.join()
        return self.max_vram


# ---------------------------------------------------------------------------
# Bootstrap confidence interval helper
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list, stat_fn=np.mean, n_boot: int = 2000,
                 confidence: float = 0.95, rng_seed: int = 42) -> tuple:
    """Return (lower, upper) bootstrap CI for stat_fn applied to values."""
    if len(values) < 2:
        v = stat_fn(values) if values else 0.0
        return (float(v), float(v))
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(rng_seed)
    boot_stats = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# McNemar test between two binary decision arrays
# ---------------------------------------------------------------------------

def mcnemar_test(correct_a: list, correct_b: list) -> dict:
    """
    McNemar's test for paired binary outcomes.
    Returns chi2 statistic, p-value, and n01/n10 counts.
    """
    n01 = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)
    n10 = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    contingency = [[sum(1 for a, b in zip(correct_a, correct_b) if not a and not b),
                    n01],
                   [n10,
                    sum(1 for a, b in zip(correct_a, correct_b) if a and b)]]
    if (n01 + n10) == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": 0, "n10": 0}
    chi2, p_val, _, _ = chi2_contingency(contingency, correction=True)
    return {"chi2": float(chi2), "p_value": float(p_val), "n01": n01, "n10": n10}


# ---------------------------------------------------------------------------
# KS-detection sensitivity analysis (model-free — runs only the monitor layer)
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(output_dir: str = "plots"):
    """
    Sweeps reference/current window sizes to assess KS detection rate vs
    false-positive rate without invoking any LLM inference.
    """
    print("\n========== SENSITIVITY ANALYSIS (KS Window Sizes) ==========")
    records = []
    for cfg in SENSITIVITY_CONFIGS:
        ref_w = cfg["reference_window_size"]
        cur_w = cfg["current_window_size"]
        label = f"ref={ref_w}/cur={cur_w}"

        monitor = StreamingMonitor(
            data_path="data/CMAPSSData/train_FD001.txt",
            reference_window_size=ref_w,
            current_window_size=cur_w,
            p_value_threshold=STREAMING_CONFIG["p_value_threshold"],
            multiple_testing_correction=STREAMING_CONFIG["multiple_testing_correction"],
            step_size=STREAMING_CONFIG["step_size"],
        )

        detections, total = 0, 0
        while True:
            result = monitor.detect_drift()
            if result["status"] == "end_of_stream":
                break
            total += 1
            if result["drift_detected"]:
                detections += 1
            if total >= 200:
                break

        detection_rate = detections / total if total > 0 else 0.0
        records.append({
            "config": label,
            "ref_window": ref_w,
            "cur_window": cur_w,
            "detection_rate": detection_rate,
            "total_steps": total,
        })
        print(f"  {label}: detection_rate={detection_rate:.3f} over {total} steps")

    sens_df = pd.DataFrame(records)
    if not sens_df.empty:
        plt.figure(figsize=PLOT_CONFIG.get("figsize", (10, 6)))
        ax = sns.barplot(data=sens_df, x="config", y="detection_rate", palette="Blues_d")
        plt.title("KS Drift Detection Rate vs Window Configuration\n(all 200 baseline steps, no fault injection)")
        plt.ylabel("Detection Rate (fraction of steps flagged)")
        plt.xlabel("Window Configuration")
        plt.ylim(0, 1.05)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=9, padding=2)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sensitivity_window.png",
                    dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches="tight")
        plt.close()
        print(f"Sensitivity plot saved to {output_dir}/sensitivity_window.png")

    return sens_df


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    max_loops: int = 100,
    step_size: int = 20,
    output_dir: str = "plots",
    eglr_mode: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    traces = []
    tracker = VRAMTracker()
    monitor_refs = []

    for config_name, models_list in MODEL_CONFIGS:
        print(f"\n==========================================")
        print(f"Starting Evaluation for Setup: {config_name}")
        print(f"Using models: {models_list}")
        print(f"==========================================")

        app, monitor = build_graph(models=models_list, step_size=step_size, eglr_mode=eglr_mode)
        monitor_refs.append((config_name, monitor))

        loop_count = 0

        while loop_count < max_loops:
            initial_state = {
                "current_data": {},
                "p_values": {},
                "drift_detected": False,
                "shap_values": None,
                "action_decision": None,
                "messages": [],
                "reasoning": None,
                "llm_latency": None,
                "current_idx": None,
                "manual_content": None,
                "consensus": None,
                "current_fault_type": None,
                "monitor_latency_s": None,
                "plan_latency_s": None,
                "shap_latency_s": None,
                "consensus_confidence": None,
                "winning_model": None,
            }

            tracker.start()
            result = app.invoke(initial_state)
            peak_vram = tracker.stop()

            current_idx = result.get("current_idx")

            if result.get("messages") and "End of data stream reached." in result["messages"]:
                print("Stream ended.")
                break

            # ---------------------------------------------------------------
            # CRITICAL: ground truth must be sampled BEFORE injecting the new
            # fault — otherwise the cycle that triggers a fault is labelled as
            # a fault even though the model ran on pre-fault data.
            # ---------------------------------------------------------------
            ground_truth = monitor.get_ground_truth_label(current_idx) if current_idx else "normal"

            if current_idx:
                monitor.run_fault_schedule(current_idx)

            action = result.get("action_decision")
            reasoning = result.get("reasoning", "")
            latency = result.get("llm_latency", 0.0)

            if action:
                is_sensor_failure = ground_truth == "sensor_failure"

                if ground_truth == "sensor_failure":
                    correct = (action == "ISSUE_REPLACEMENT_TICKET")
                elif ground_truth == "operational_drift":
                    correct = (action == "RETRAIN_MODEL")
                else:
                    correct = (action == "RETRAIN_MODEL")

                tokens_est = len(reasoning.split()) if reasoning else 0

                metrics = {
                    "model": config_name,
                    "cycle": current_idx,
                    "action": action,
                    "latency_s": latency,
                    "peak_vram_mb": peak_vram,
                    "token_count": tokens_est,
                    "is_sensor_failure": is_sensor_failure,
                    "correct": correct,
                    "ground_truth": ground_truth,
                    "monitor_latency_s": result.get("monitor_latency_s", 0.0),
                    "plan_latency_s": result.get("plan_latency_s", 0.0),
                    "shap_latency_s": result.get("shap_latency_s", 0.0),
                    "llm_latency_s": latency,
                    "consensus_confidence": result.get("consensus_confidence", None),
                    "winning_model": result.get("winning_model", None),
                    "fci": result.get("fci", None),
                    "llm_bypassed": result.get("llm_bypassed", False),
                }
                all_metrics.append(metrics)

                trace = {
                    "model": config_name,
                    "cycle": current_idx,
                    "p_values": result.get("p_values"),
                    "shap_values": result.get("shap_values"),
                    "reasoning": reasoning,
                    "action": action,
                    "ground_truth": ground_truth,
                }
                traces.append(trace)

                print(
                    f"[{config_name}] Cycle {current_idx} | GT: {ground_truth} "
                    f"| Action: {action} | Correct: {correct}"
                )

            loop_count += 1
            time.sleep(0.1)

    # -----------------------------------------------------------------------
    # Persist raw data
    # -----------------------------------------------------------------------
    with open("traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=4)

    df = pd.DataFrame(all_metrics)
    df.to_csv("evaluation_metrics.csv", index=False)

    all_fault_logs = []
    for config_name, monitor_ref in monitor_refs:
        for event in monitor_ref.fault_log:
            all_fault_logs.append({**event, "config": config_name})

    with open("fault_log.json", "w", encoding="utf-8") as f:
        json.dump(all_fault_logs, f, indent=4)
    print("Fault log saved to fault_log.json")

    if df.empty:
        print("No actions were executed; no plots generated.")
        return

    # -----------------------------------------------------------------------
    # Statistical analysis
    # -----------------------------------------------------------------------
    sns.set_theme(style=PLOT_CONFIG.get("style", "whitegrid"))

    summary = {}
    for config_name, _ in MODEL_CONFIGS:
        model_df = df[df["model"] == config_name]
        if model_df.empty:
            continue

        correct_vals = model_df["correct"].tolist()
        lat_vals = model_df["latency_s"].tolist()

        ecr_lo, ecr_hi = bootstrap_ci(correct_vals, np.mean)
        lat_lo, lat_hi = bootstrap_ci(lat_vals, np.mean)

        fault_df = model_df[model_df["is_sensor_failure"] == True]
        norm_df = model_df[model_df["is_sensor_failure"] == False]

        fault_ci = bootstrap_ci(fault_df["correct"].tolist(), np.mean) if not fault_df.empty else (None, None)
        norm_ci = bootstrap_ci(norm_df["correct"].tolist(), np.mean) if not norm_df.empty else (None, None)

        y_true = (model_df["is_sensor_failure"] == True).astype(int).tolist()
        y_pred = (model_df["action"] == "ISSUE_REPLACEMENT_TICKET").astype(int).tolist()

        if len(set(y_true)) >= 2:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            prec = rec = f1 = 0.0

        summary[config_name] = {
            "overall_ecr": round(float(model_df["correct"].mean()), 4),
            "overall_ecr_ci_95": [round(ecr_lo, 4), round(ecr_hi, 4)],
            "fault_ecr": round(float(fault_df["correct"].mean()), 4) if not fault_df.empty else None,
            "fault_ecr_ci_95": [round(fault_ci[0], 4), round(fault_ci[1], 4)] if fault_ci[0] is not None else None,
            "normal_ecr": round(float(norm_df["correct"].mean()), 4) if not norm_df.empty else None,
            "normal_ecr_ci_95": [round(norm_ci[0], 4), round(norm_ci[1], 4)] if norm_ci[0] is not None else None,
            "avg_latency_s": round(float(model_df["latency_s"].mean()), 4),
            "avg_latency_ci_95": [round(lat_lo, 4), round(lat_hi, 4)],
            "p95_latency_s": round(float(model_df["latency_s"].quantile(0.95)), 4),
            "avg_vram_mb": round(float(model_df["peak_vram_mb"].mean()), 2),
            "avg_token_count": round(float(model_df["token_count"].mean()), 1),
            "total_decisions": int(len(model_df)),
            "fault_decisions": int(model_df["is_sensor_failure"].sum()),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "llm_bypass_rate": round(float(model_df["llm_bypassed"].mean()), 4) if "llm_bypassed" in model_df else 0.0,
            "bypassed_ecr": round(float(model_df[model_df["llm_bypassed"] == True]["correct"].mean()), 4) if "llm_bypassed" in model_df and model_df["llm_bypassed"].any() else None,
            "llm_invoked_ecr": round(float(model_df[model_df["llm_bypassed"] == False]["correct"].mean()), 4) if "llm_bypassed" in model_df and (~model_df["llm_bypassed"]).any() else None,
        }

    # Cohen's Kappa + McNemar pairwise
    config_names = [c for c, _ in MODEL_CONFIGS]
    pairwise = {}
    for i in range(len(config_names)):
        for j in range(i + 1, len(config_names)):
            a, b = config_names[i], config_names[j]
            df_a = df[df["model"] == a]
            df_b = df[df["model"] == b]
            if df_a.empty or df_b.empty:
                continue
            merged = pd.merge(df_a, df_b, on="cycle", suffixes=("_a", "_b"))
            if merged.empty:
                continue
            kappa = cohen_kappa_score(
                merged["action_a"] == "ISSUE_REPLACEMENT_TICKET",
                merged["action_b"] == "ISSUE_REPLACEMENT_TICKET",
            )
            mc = mcnemar_test(
                merged["correct_a"].tolist(),
                merged["correct_b"].tolist(),
            )
            key = f"{a}_vs_{b}"
            pairwise[key] = {
                "cohens_kappa": round(float(kappa), 4),
                "mcnemar_chi2": round(mc["chi2"], 4),
                "mcnemar_p_value": round(mc["p_value"], 4),
                "n01_a_wrong_b_right": mc["n01"],
                "n10_a_right_b_wrong": mc["n10"],
            }

    summary["pairwise_statistics"] = pairwise

    # Legacy key for backwards compatibility with older analysis scripts
    if "qwen2.5:7b_only_vs_phi3.5_only" in pairwise:
        summary["cohens_kappa_qwen_vs_phi"] = pairwise["qwen2.5:7b_only_vs_phi3.5_only"]["cohens_kappa"]

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Summary saved to results/summary.json")

    # -----------------------------------------------------------------------
    # Plot 1: Latency vs ECR with bootstrap CI error bars
    # -----------------------------------------------------------------------
    ecr_data = []
    for config_name, _ in MODEL_CONFIGS:
        if config_name not in summary:
            continue
        s = summary[config_name]
        ecr_data.append({
            "model": config_name,
            "latency_s": s["avg_latency_s"],
            "ecr": s["overall_ecr"],
            "ecr_err": (s["overall_ecr"] - s["overall_ecr_ci_95"][0]),
        })
    ecr_df = pd.DataFrame(ecr_data)

    fig, ax = plt.subplots(figsize=PLOT_CONFIG.get("figsize", (8, 6)))
    for _, row in ecr_df.iterrows():
        ax.errorbar(row["latency_s"], row["ecr"], yerr=row["ecr_err"],
                    fmt="o", markersize=12, capsize=5, label=row["model"])
    ax.set_title("Latency vs Error Correction Rate (ECR)\nwith 95% Bootstrap Confidence Intervals")
    ax.set_xlabel("Average Latency (seconds)")
    ax.set_ylabel("Error Correction Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_vs_ecr.png", dpi=PLOT_CONFIG.get("dpi", 150))
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 2: VRAM utilisation
    # -----------------------------------------------------------------------
    plt.figure(figsize=PLOT_CONFIG.get("figsize", (8, 6)))
    sns.barplot(data=df, x="model", y="peak_vram_mb", hue="model", legend=False)
    plt.title("Peak VRAM Utilization during Inference")
    plt.ylabel("Peak VRAM (MB)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vram_utilization.png", dpi=PLOT_CONFIG.get("dpi", 150))
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 3: ECR on injected hard faults
    # -----------------------------------------------------------------------
    faults_df = df[df["is_sensor_failure"] == True]
    if not faults_df.empty:
        plt.figure(figsize=PLOT_CONFIG.get("figsize", (8, 6)))
        sns.barplot(data=faults_df, x="model", y="correct", hue="model", legend=False)
        plt.title("Accuracy (ECR) on Injected Hard Faults Only")
        plt.ylabel("Accuracy Rate")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/injection_accuracy.png", dpi=PLOT_CONFIG.get("dpi", 150))
        plt.close()

    # -----------------------------------------------------------------------
    # Plot 4: Decision timeline
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(len(MODEL_CONFIGS), 1,
                             figsize=(14, 4 * len(MODEL_CONFIGS)), sharex=True)
    if len(MODEL_CONFIGS) == 1:
        axes = [axes]
    action_color_map = PLOT_CONFIG.get(
        "action_colors",
        {"ISSUE_REPLACEMENT_TICKET": "#e74c3c", "RETRAIN_MODEL": "#2ecc71"},
    )

    for ax, (config_name, _) in zip(axes, MODEL_CONFIGS):
        model_df = df[df["model"] == config_name].copy().sort_values("cycle")
        if model_df.empty:
            continue
        for _, row in model_df.iterrows():
            color = action_color_map.get(row["action"], "#95a5a6")
            marker = "o" if row["correct"] else "X"
            ax.scatter(row["cycle"], 0.5, color=color, marker=marker, s=120, zorder=3)

        linestyles = {"sensor_failure": "--", "operational_drift": "-."}
        offsets = {"sensor_failure": 0, "operational_drift": 2}

        for event in all_fault_logs:
            if event["config"] == config_name:
                x_val = event["cycle"] + offsets.get(event["fault_type"], 0)
                ls = linestyles.get(event["fault_type"], "--")
                ax.axvline(x=x_val, color="orange", linestyle=ls, alpha=0.7, linewidth=1.5)
                ax.text(x_val, 0.85,
                        f"{event['sensor_id']}\n{event['fault_type'][:6]}",
                        fontsize=8, ha="center", color="darkorange",
                        transform=ax.get_xaxis_transform(), clip_on=False)

        legend_elements = [
            Patch(facecolor="#e74c3c", label="ISSUE_REPLACEMENT_TICKET"),
            Patch(facecolor="#2ecc71", label="RETRAIN_MODEL"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Correct"),
            Line2D([0], [0], marker="X", color="w", markerfacecolor="gray", markersize=8, label="Incorrect"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
        ax.set_yticks([])
        ax.set_ylabel(config_name, fontsize=10, rotation=0, labelpad=90, ha="right", va="center")
        ax.set_title(f"Decision Timeline: {config_name}", fontsize=11)

    axes[-1].set_xlabel("Cycle Index", fontsize=12)
    fig.suptitle(
        "Triage Decision Timeline per Model Configuration\n(Orange dashed = Fault Injection Events)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/decision_timeline.png",
                dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 5: Precision / Recall / F1
    # -----------------------------------------------------------------------
    records = []
    for config_name, _ in MODEL_CONFIGS:
        if config_name not in summary:
            continue
        s = summary[config_name]
        records.append({"model": config_name, "metric": "Precision", "value": s["precision"]})
        records.append({"model": config_name, "metric": "Recall",    "value": s["recall"]})
        records.append({"model": config_name, "metric": "F1-Score",  "value": s["f1_score"]})

    metrics_df = pd.DataFrame(records)
    if not metrics_df.empty:
        plt.figure(figsize=PLOT_CONFIG.get("figsize", (10, 6)))
        ax = sns.barplot(data=metrics_df, x="model", y="value", hue="metric", palette="Set2")
        plt.title("Precision, Recall, F1-Score per Model\n(Positive Class: ISSUE_REPLACEMENT_TICKET)")
        plt.ylim(0, 1.15)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=9, padding=2)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/precision_recall_f1.png",
                    dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches="tight")
        plt.close()

    # -----------------------------------------------------------------------
    # Plot 6: Per-node latency breakdown (stacked bar)
    # -----------------------------------------------------------------------
    latency_cols = ["monitor_latency_s", "plan_latency_s", "shap_latency_s", "llm_latency_s"]
    latency_labels = ["KS-Test Monitor", "RAG Plan", "SHAP Diagnosis", "LLM Inference"]
    latency_df = df[latency_cols + ["model"]].copy()
    latency_df[latency_cols] = latency_df[latency_cols].fillna(0)
    latency_df = latency_df[latency_df[latency_cols].sum(axis=1) > 0]

    if not latency_df.empty:
        avg_latency = latency_df.groupby("model")[latency_cols].mean()
        fig, ax = plt.subplots(figsize=PLOT_CONFIG.get("figsize", (10, 6)))
        bottom = pd.Series([0.0] * len(avg_latency), index=avg_latency.index)
        colors = ["#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
        for col, label, color in zip(latency_cols, latency_labels, colors):
            ax.bar(avg_latency.index, avg_latency[col], bottom=bottom,
                   label=label, color=color, alpha=0.85)
            bottom += avg_latency[col]
        ax.set_title("Average Per-Node Latency Breakdown by Model Configuration")
        ax.set_yscale("log")
        ax.set_ylabel("Average Latency (seconds, log scale)")
        ax.legend(title="Pipeline Node", loc="upper left")
        for i, (idx, row) in enumerate(avg_latency.iterrows()):
            total = row[latency_cols].sum()
            ax.text(i, total + 0.05, f"{total:.2f}s", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_breakdown.png",
                    dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches="tight")
        plt.close()

    # -----------------------------------------------------------------------
    # Plot 7: Cohen's Kappa and McNemar p-values heatmap
    # -----------------------------------------------------------------------
    if pairwise:
        pair_keys = list(pairwise.keys())
        kappas = [pairwise[k]["cohens_kappa"] for k in pair_keys]
        pvals = [pairwise[k]["mcnemar_p_value"] for k in pair_keys]

        fig, axes2 = plt.subplots(1, 2, figsize=(12, max(4, len(pair_keys) * 1.2)))
        short_keys = [k.replace("qwen2.5:7b_only", "Qwen")
                       .replace("phi3.5_only", "Phi")
                       .replace("consensus_qwen_phi", "Consensus") for k in pair_keys]

        axes2[0].barh(short_keys, kappas, color="#3498db", alpha=0.8)
        axes2[0].set_title("Cohen's Kappa\n(Model Agreement)")
        axes2[0].set_xlabel("Kappa")
        axes2[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)

        bar_colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in pvals]
        axes2[1].barh(short_keys, pvals, color=bar_colors, alpha=0.8)
        axes2[1].axvline(0.05, color="red", linestyle="--", linewidth=1, label="α=0.05")
        axes2[1].set_title("McNemar Test p-value\n(red = significant difference)")
        axes2[1].set_xlabel("p-value")
        axes2[1].legend(fontsize=8)

        plt.suptitle("Pairwise Statistical Comparison of Model Configurations", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pairwise_statistics.png",
                    dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches="tight")
        plt.close()
        print(f"Pairwise statistics plot saved.")

    # -----------------------------------------------------------------------
    # Plot 8: ECR breakdown by ground-truth class (grouped bar)
    # -----------------------------------------------------------------------
    gt_records = []
    for config_name, _ in MODEL_CONFIGS:
        model_df = df[df["model"] == config_name]
        for gt_class in ["sensor_failure", "operational_drift", "normal"]:
            sub = model_df[model_df["ground_truth"] == gt_class]
            if sub.empty:
                continue
            gt_records.append({
                "model": config_name,
                "ground_truth": gt_class,
                "ecr": float(sub["correct"].mean()),
                "n": len(sub),
            })

    gt_df = pd.DataFrame(gt_records)
    if not gt_df.empty:
        plt.figure(figsize=PLOT_CONFIG.get("figsize", (12, 6)))
        ax = sns.barplot(data=gt_df, x="model", y="ecr", hue="ground_truth", palette="Set1")
        plt.title("ECR Breakdown by Ground-Truth Class per Model")
        plt.ylabel("Error Correction Rate")
        plt.ylim(0, 1.15)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=9, padding=2)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ecr_by_class.png",
                    dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches="tight")
        plt.close()
        print("ECR-by-class plot saved.")

    # -----------------------------------------------------------------------
    # Sensitivity analysis (KS only — fast, model-free)
    # -----------------------------------------------------------------------
    run_sensitivity_analysis(output_dir=output_dir)

    print("\n=== Evaluation complete. All outputs written. ===")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentMaintain Ablation Evaluator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only 5 cycles per config to test the pipeline end-to-end")
    parser.add_argument("--max-loops", type=int, default=100,
                        help="Maximum loops per model config (default: 100)")
    parser.add_argument("--step-size", type=int, default=20,
                        help="Data stream step size (default: 20)")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Output directory for plots")
    parser.add_argument("--eglr", action="store_true",
                        help="Enable Evidence-Gated LLM Routing (EGLR) — DES-FCI algorithmic pre-router")
    args = parser.parse_args()

    run_ablation(
        max_loops=5 if args.dry_run else args.max_loops,
        step_size=args.step_size,
        output_dir=args.output_dir,
        eglr_mode=args.eglr,
    )
