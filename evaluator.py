import time
import json
import os
import argparse
import pynvml
import threading
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from graph_builder import build_graph
from experiment_config import MODEL_CONFIGS, EVALUATION_CONFIG, PLOT_CONFIG

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
        if not self.enabled: return
        self.tracking = True
        self.thread = threading.Thread(target=self._track)
        self.thread.start()
        
    def stop(self):
        if not self.enabled: return 0.0
        self.tracking = False
        if self.thread:
            self.thread.join()
        return self.max_vram

def run_ablation(max_loops=100, step_size=20, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = []
    traces = []
    tracker = VRAMTracker()
    monitor_refs = []
    
    for config_name, models_list in MODEL_CONFIGS:
        print(f"==========================================")
        print(f"Starting Evaluation for Setup: {config_name}")
        print(f"Using models: {models_list}")
        print(f"==========================================")
        
        # Reset graph & monitor for fresh state
        app, monitor = build_graph(models=models_list, step_size=step_size)
        monitor_refs.append((config_name, monitor))
        
        loop_count = 0
        
        while loop_count < max_loops:
            # Inject faults dynamically via schedule BEFORE invoke
            monitor.run_fault_schedule(monitor.current_idx)

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
            
            if result.get("messages") and "End of data stream reached." in result["messages"]:
                print("Stream ended.")
                break
                
            action = result.get("action_decision")
            reasoning = result.get("reasoning", "")
            latency = result.get("llm_latency", 0.0)
            
            if action:
                # Calculate True/False Correctness based on Ground Truth from Fault Log
                ground_truth = monitor.get_ground_truth_label(result.get("current_idx"))
                fault_in_this_cycle = ground_truth == "sensor_failure"
                
                if ground_truth == "sensor_failure":
                    correct = (action == "ISSUE_REPLACEMENT_TICKET")
                elif ground_truth == "operational_drift":
                    correct = (action == "RETRAIN_MODEL")
                else:
                    correct = (action == "RETRAIN_MODEL") # conservative default for normal operation

                tokens_est = len(reasoning.split()) if reasoning else 0
                
                metrics = {
                    "model": config_name,
                    "cycle": result.get("current_idx"),
                    "action": action,
                    "latency_s": latency,
                    "peak_vram_mb": peak_vram,
                    "token_count": tokens_est,
                    "fault_injected": fault_in_this_cycle,
                    "correct": correct,
                    "monitor_latency_s": result.get("monitor_latency_s", 0.0),
                    "plan_latency_s":    result.get("plan_latency_s", 0.0),
                    "shap_latency_s":    result.get("shap_latency_s", 0.0),
                    "llm_latency_s":     latency,
                    "consensus_confidence": result.get("consensus_confidence", None),
                    "winning_model": result.get("winning_model", None),
                }
                all_metrics.append(metrics)
                
                trace = {
                    "model": config_name,
                    "cycle": result.get("current_idx"),
                    "p_values": result.get("p_values"),
                    "shap_values": result.get("shap_values"),
                    "reasoning": reasoning,
                    "action": action
                }
                traces.append(trace)
                
                print(f"[{config_name}] Cycle {result.get('current_idx')} | GT: {ground_truth} | Action: {action} | Correct: {correct}")
                
            loop_count += 1
            time.sleep(0.1)
            
    # Export Traces and CSV
    with open("traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=4)
        
    df = pd.DataFrame(all_metrics)
    df.to_csv("evaluation_metrics.csv", index=False)
    
    # Export Fault Log
    all_fault_logs = []
    for config_name, monitor_ref in monitor_refs:
        for event in monitor_ref.fault_log:
            event["config"] = config_name
            all_fault_logs.append(event)

    with open("fault_log.json", "w", encoding="utf-8") as f:
        json.dump(all_fault_logs, f, indent=4)
    print("Fault log saved to fault_log.json")

    # Generate Plots
    sns.set_theme(style=PLOT_CONFIG.get("style", "whitegrid"))
    
    if len(df) > 0:
        # Plot 1: Latency vs. Accuracy (ECR)
        ecr_df = df.groupby('model')['correct'].mean().reset_index()
        ecr_df.rename(columns={'correct': 'ecr'}, inplace=True)
        avg_lat = df.groupby('model')['latency_s'].mean().reset_index()
        summary = pd.merge(ecr_df, avg_lat, on='model')
        
        plt.figure(figsize=PLOT_CONFIG.get("figsize", (8, 6)))
        sns.scatterplot(data=summary, x='latency_s', y='ecr', hue='model', s=200)
        plt.title('Latency vs Error Correction Rate (ECR)')
        plt.xlabel('Average Latency (seconds)')
        plt.ylabel('Error Correction Rate')
        plt.savefig(f'{output_dir}/latency_vs_ecr.png', dpi=PLOT_CONFIG.get("dpi", 150))
        plt.close()
        
        # Plot 2: VRAM
        plt.figure(figsize=PLOT_CONFIG.get("figsize", (8, 6)))
        sns.barplot(data=df, x='model', y='peak_vram_mb', hue='model')
        plt.title('Peak VRAM Utilization during Inference')
        plt.ylabel('Peak VRAM (MB)')
        plt.savefig(f'{output_dir}/vram_utilization.png', dpi=PLOT_CONFIG.get("dpi", 150))
        plt.close()

        # Plot 3: ECR on Point Anomalies (Injected hard faults)
        faults_df = df[df['fault_injected'] == True]
        if not faults_df.empty:
            plt.figure(figsize=PLOT_CONFIG.get("figsize", (8, 6)))
            sns.barplot(data=faults_df, x='model', y='correct', hue='model')
            plt.title('Accuracy (ECR) specifically on Injected Hard Faults')
            plt.ylabel('Accuracy Rate')
            plt.savefig(f'{output_dir}/injection_accuracy.png', dpi=PLOT_CONFIG.get("dpi", 150))
            plt.close()

        # ------------ NEW PLOTS ------------

        # Plot 4: Triage Decision Timeline
        fig, axes = plt.subplots(len(MODEL_CONFIGS), 1, figsize=(14, 4 * len(MODEL_CONFIGS)), sharex=True)
        if len(MODEL_CONFIGS) == 1: axes = [axes]
        action_color_map = PLOT_CONFIG.get("action_colors", {"ISSUE_REPLACEMENT_TICKET": "#e74c3c", "RETRAIN_MODEL": "#2ecc71"})

        for ax, (config_name, _) in zip(axes, MODEL_CONFIGS):
            model_df = df[df['model'] == config_name].copy()
            model_df = model_df.sort_values('cycle')
            if model_df.empty: continue
            
            for _, row in model_df.iterrows():
                color = action_color_map.get(row['action'], '#95a5a6')
                marker = 'o' if row['correct'] else 'X'
                ax.scatter(row['cycle'], 0.5, color=color, marker=marker, s=120, zorder=3)
            
            if os.path.exists("fault_log.json"):
                with open("fault_log.json") as f: fl = json.load(f)
                for event in fl:
                    if event["config"] == config_name:
                        ax.axvline(x=event['cycle'], color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
                        ax.text(event['cycle'], 0.85, f"{event['sensor_id']}\n{event['fault_type'][:6]}", 
                                fontsize=7, ha='center', color='darkorange', transform=ax.get_xaxis_transform())
            
            legend_elements = [
                Patch(facecolor='#e74c3c', label='ISSUE_REPLACEMENT_TICKET'),
                Patch(facecolor='#2ecc71', label='RETRAIN_MODEL'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Correct'),
                Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', markersize=8, label='Incorrect'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
            ax.set_yticks([])
            ax.set_ylabel(config_name, fontsize=10, rotation=0, labelpad=90, ha='right', va='center')
            ax.set_title(f"Decision Timeline: {config_name}", fontsize=11)

        axes[-1].set_xlabel("Cycle Index", fontsize=12)
        fig.suptitle("Triage Decision Timeline per Model Configuration\n(Orange dashed = Fault Injection Events)", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/decision_timeline.png', dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches='tight')
        plt.close()

        # Plot 5: Precision, Recall, F1
        records = []
        for config_name, _ in MODEL_CONFIGS:
            model_df = df[df['model'] == config_name].copy()
            if model_df.empty: continue
            
            y_true = (model_df['fault_injected'] == True).astype(int).tolist()
            y_pred = (model_df['action'] == 'ISSUE_REPLACEMENT_TICKET').astype(int).tolist()
            
            if len(set(y_true)) < 2:
                p, r, f_score = 0.0, 0.0, 0.0
            else:
                p = precision_score(y_true, y_pred, zero_division=0)
                r = recall_score(y_true, y_pred, zero_division=0)
                f_score = f1_score(y_true, y_pred, zero_division=0)
            
            records.append({"model": config_name, "metric": "Precision", "value": p})
            records.append({"model": config_name, "metric": "Recall",    "value": r})
            records.append({"model": config_name, "metric": "F1-Score",  "value": f_score})

        metrics_df = pd.DataFrame(records)
        if not metrics_df.empty:
            plt.figure(figsize=PLOT_CONFIG.get("figsize", (10, 6)))
            ax = sns.barplot(data=metrics_df, x='model', y='value', hue='metric', palette='Set2')
            plt.title('Precision, Recall, F1-Score per Model\n(Positive Class: ISSUE_REPLACEMENT_TICKET)')
            plt.ylim(0, 1.15)
            for container in ax.containers: ax.bar_label(container, fmt='%.2f', fontsize=9, padding=2)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/precision_recall_f1.png', dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches='tight')
            plt.close()

        # Plot 6: Per-Node Latency Breakdown
        latency_cols = ['monitor_latency_s', 'plan_latency_s', 'shap_latency_s', 'llm_latency_s']
        latency_labels = ['KS-Test Monitor', 'RAG Plan', 'SHAP Diagnosis', 'LLM Inference']
        latency_df = df[latency_cols + ['model']].copy()
        
        # Replace NaNs with 0 to prevent summation errors
        latency_df[latency_cols] = latency_df[latency_cols].fillna(0)
        latency_df = latency_df[latency_df[latency_cols].sum(axis=1) > 0]

        if not latency_df.empty:
            avg_latency = latency_df.groupby('model')[latency_cols].mean()
            fig, ax = plt.subplots(figsize=PLOT_CONFIG.get("figsize", (10, 6)))
            bottom = pd.Series([0.0] * len(avg_latency), index=avg_latency.index)
            colors = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c']
            
            for col, label, color in zip(latency_cols, latency_labels, colors):
                ax.bar(avg_latency.index, avg_latency[col], bottom=bottom, label=label, color=color, alpha=0.85)
                bottom += avg_latency[col]
            
            ax.set_title('Average Per-Node Latency Breakdown by Model Configuration')
            ax.set_ylabel('Average Latency (seconds)')
            ax.legend(title='Pipeline Node', loc='upper left')
            
            for i, (idx, row) in enumerate(avg_latency.iterrows()):
                total = row[latency_cols].sum()
                ax.text(i, total + 0.05, f'{total:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/latency_breakdown.png', dpi=PLOT_CONFIG.get("dpi", 150), bbox_inches='tight')
            plt.close()

        # Generate summary.json
        summary = {}
        for config_name, _ in MODEL_CONFIGS:
            model_df = df[df['model'] == config_name]
            if model_df.empty: continue
            summary[config_name] = {
                "overall_ecr":        round(float(model_df['correct'].mean()), 4),
                "fault_ecr":          round(float(model_df[model_df['fault_injected'] == True]['correct'].mean()), 4) if not model_df[model_df['fault_injected'] == True].empty else None,
                "normal_ecr":         round(float(model_df[model_df['fault_injected'] == False]['correct'].mean()), 4) if not model_df[model_df['fault_injected'] == False].empty else None,
                "avg_latency_s":      round(float(model_df['latency_s'].mean()), 4),
                "p95_latency_s":      round(float(model_df['latency_s'].quantile(0.95)), 4) if not model_df['latency_s'].empty else None,
                "avg_vram_mb":        round(float(model_df['peak_vram_mb'].mean()), 2),
                "avg_token_count":    round(float(model_df['token_count'].mean()), 1),
                "total_decisions":    int(len(model_df)),
                "fault_decisions":    int(model_df['fault_injected'].sum()),
            }

        os.makedirs("results", exist_ok=True)
        with open("results/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        print("Summary saved to results/summary.json")

    else:
        print("No actions were executed; no plots generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentMaintain Ablation Evaluator")
    parser.add_argument("--dry-run", action="store_true", help="Run only 5 cycles per config to test the pipeline end-to-end without full evaluation")
    parser.add_argument("--max-loops", type=int, default=100, help="Maximum loops per model config (default: 100)")
    parser.add_argument("--step-size", type=int, default=20, help="Data stream step size (default: 20)")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()
    
    run_ablation(
        max_loops=5 if args.dry_run else args.max_loops,
        step_size=args.step_size,
        output_dir=args.output_dir
    )
