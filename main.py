import time
from graph_builder import build_graph

def main():
    print("Starting AgentMaintain Orchestrator...")
    print("Monitoring C-MAPSS data stream for feature drift...")
    print("--------------------------------------------------")
    
    app, monitor = build_graph(models=["qwen2.5:7b", "phi3.5"], step_size=20)
    
    loop_count = 0
    max_loops = 100 
    
    while loop_count < max_loops:
        # Faults will be injected after invoke in the standard simulation,
        # but for main.py (simple testing loop) we can inject them as we advance.

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
        
        result = app.invoke(initial_state)
        
        # After invoke, monitor's stream has advanced. We inject faults on the matching current cycle.
        if result.get("current_idx"):
            monitor.run_fault_schedule(result.get("current_idx"))
        
        if result.get("messages") and "End of data stream reached." in result["messages"]:
            print("Data stream ended. Shutting down.")
            break
            
        action = result.get("action_decision")
        if action:
            print(f">>> FINAL TRIAGE ACTION EXECUTED: {action}")
            print("--------------------------------------------------")
        
        loop_count += 1
        time.sleep(0.1) # Faster simulation for skip-step
        
    print("AgentMaintain Simulation Finished.")

if __name__ == "__main__":
    main()
