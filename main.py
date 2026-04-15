import time
from graph_builder import build_graph

def main():
    print("Starting AgentMaintain Orchestrator...")
    print("Monitoring C-MAPSS data stream for feature drift...")
    print("--------------------------------------------------")
    
    app, monitor = build_graph(model_name="qwen2.5:7b", step_size=20)
    
    loop_count = 0
    max_loops = 100 
    
    while loop_count < max_loops:
        # Check current index to inject fault every 100 baseline cycles
        # Note: Depending on where current_idx is initialized, we handle safety.
        if monitor.current_idx and int(monitor.current_idx) > 0 and (int(monitor.current_idx) % 100 == 0):
            monitor.inject_fault(sensor_id="sensor_14", severity=10.0)

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
        
        result = app.invoke(initial_state)
        
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
