import time
import json
import os
import pynvml
import threading
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from graph_builder import build_graph

class VRAMTracker:
    def __init__(self):
        self.tracking = False
        self.max_vram = 0
        self.thread = None
        self.handle = None
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print("Failed to init pynvml or no GPU found:", e)
    
    def _track(self):
        self.max_vram = 0
        while self.tracking:
            if self.handle:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                used_mb = info.used / 1024 / 1024
                if used_mb > self.max_vram:
                    self.max_vram = used_mb
            time.sleep(0.05) # high frequency sampling
            
    def start(self):
        self.tracking = True
        self.thread = threading.Thread(target=self._track)
        self.thread.start()
        
    def stop(self):
        self.tracking = False
        if self.thread:
            self.thread.join()
        return self.max_vram

def run_ablation():
    # Ablation study: test models individually, then together as a consensus system
    model_configs = [
        ("qwen2.5:7b_only", ["qwen2.5:7b"]),
        ("phi3.5_only", ["phi3.5"]),
        ("consensus_qwen_phi", ["qwen2.5:7b", "phi3.5"])
    ]
    max_loops = 100
    step_size = 20
    
    os.makedirs("plots", exist_ok=True)
    
    all_metrics = []
    traces = []
    
    tracker = VRAMTracker()
    
    for config_name, models_list in model_configs:
        print(f"==========================================")
        print(f"Starting Evaluation for Setup: {config_name}")
        print(f"Using models: {models_list}")
        print(f"==========================================")
        
        # Reset graph & monitor for fresh state
        app, monitor = build_graph(models=models_list, step_size=step_size)
        
        loop_count = 0
        
        while loop_count < max_loops:
            # Inject fault on cycle 100 boundaries
            # Current idx starts at 80, so it hits 100, 200, 300 etc
            if monitor.current_idx and int(monitor.current_idx) > 0 and (int(monitor.current_idx) % 100 == 0):
                monitor.inject_fault(sensor_id="sensor_14", severity=10.0)
                fault_in_this_cycle = True
            else:
                fault_in_this_cycle = False

            initial_state = {
                "current_data": {},
                "p_values": {},
                "drift_detected": False,
                "shap_values": None,
                "action_decision": None,
                "messages": [],
                "reasoning": None,
                "llm_latency": None,
                "current_idx": None
            }
            
            # Start VRAM tracking exactly during the invoke block
            tracker.start()
            
            # Run graph
            result = app.invoke(initial_state)
            
            # Stop VRAM tracking
            peak_vram = tracker.stop()
            
            if result.get("messages") and "End of data stream reached." in result["messages"]:
                print("Stream ended.")
                break
                
            action = result.get("action_decision")
            reasoning = result.get("reasoning", "")
            latency = result.get("llm_latency", 0.0)
            
            if action:
                # Compute ECR (Error Correction Rate correctness)
                if fault_in_this_cycle:
                    correct = (action == "ISSUE_REPLACEMENT_TICKET")
                else:
                    # For natural drift, RETRAIN_MODEL is typically correct triage
                    correct = (action == "RETRAIN_MODEL")
                    
                tokens_est = len(reasoning.split()) if reasoning else 0
                
                metrics = {
                    "model": config_name,
                    "cycle": result.get("current_idx"),
                    "action": action,
                    "latency_s": latency,
                    "peak_vram_mb": peak_vram,
                    "token_count": tokens_est,
                    "fault_injected": fault_in_this_cycle,
                    "correct": correct
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
                
                print(f"[{config_name}] Cycle {result.get('current_idx')} | Action: {action} | Latency: {latency:.2f}s | Correct: {correct} | VRAM Peak: {peak_vram:.1f} MB")
                
            loop_count += 1
            time.sleep(0.1)  # Brief pause between loops
            
    # Export Traces JSON
    with open("traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=4)
        
    df = pd.DataFrame(all_metrics)
    df.to_csv("evaluation_metrics.csv", index=False)
    
    # Generate Plots for paper
    sns.set_theme(style="whitegrid")
    
    if len(df) > 0:
        # Plot 1: Latency vs. Accuracy (ECR)
        ecr_df = df.groupby('model')['correct'].mean().reset_index()
        ecr_df.rename(columns={'correct': 'ecr'}, inplace=True)
        avg_lat = df.groupby('model')['latency_s'].mean().reset_index()
        
        summary = pd.merge(ecr_df, avg_lat, on='model')
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=summary, x='latency_s', y='ecr', hue='model', s=200)
        plt.title('Latency vs Error Correction Rate (ECR)', fontsize=14)
        plt.xlabel('Average Latency (seconds)')
        plt.ylabel('Error Correction Rate')
        plt.savefig('plots/latency_vs_ecr.png')
        plt.close()
        
        # Plot 2: VRAM Utilization Comparison
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x='model', y='peak_vram_mb', hue='model', errorbar='sd')
        plt.title('Peak VRAM Utilization during Inference', fontsize=14)
        plt.ylabel('Peak VRAM (MB)')
        plt.savefig('plots/vram_utilization.png')
        plt.close()

        # Plot 3: ECR specifically under Point Anomalies (Injected faults)
        faults_df = df[df['fault_injected'] == True]
        if not faults_df.empty:
            plt.figure(figsize=(8, 6))
            sns.barplot(data=faults_df, x='model', y='correct', hue='model', errorbar=None)
            plt.title('Accuracy (ECR) specifically on Injected Hard Faults', fontsize=14)
            plt.ylabel('Accuracy Rate')
            plt.savefig('plots/injection_accuracy.png')
            plt.close()

        print("Evaluations completed. Traces saved to traces.json. Plots saved to plots/ directory.")
    else:
        print("No actions were executed; no plots generated.")

if __name__ == "__main__":
    run_ablation()
