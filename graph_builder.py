from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import json
import time

from agent_tools import diagnose_with_shap, TriageAction
from monitor import StreamingMonitor

# Define the State for LangGraph
class AgentState(TypedDict):
    current_data: Dict[str, float]
    p_values: Dict[str, float]
    drift_detected: bool
    shap_values: Optional[Dict[str, float]]
    action_decision: Optional[str]
    reasoning: Optional[str]
    messages: list
    llm_latency: Optional[float]
    current_idx: Optional[int]

def build_graph(model_name="qwen2.5:7b", step_size=20):
    # Initialize the LLM
    llm = ChatOllama(
        model=model_name,
        temperature=0
    )
    
    # Bind the tool for structured output
    structured_llm = llm.with_structured_output(TriageAction)
    
    # Initialize the Monitor
    monitor = StreamingMonitor(data_path="data/CMAPSSData/train_FD001.txt", step_size=step_size)
    
    def monitor_data_node(state: AgentState):
        """Monitors data stream and detects drift."""
        print("---NODE: monitor_data---")
        
        result = monitor.detect_drift()
        
        if result["status"] == "end_of_stream":
            return {"drift_detected": False, "messages": ["End of data stream reached."]}
            
        drift_detected = result["drift_detected"]
        p_values = result.get("p_values", {})
        current_data = result.get("current_data", {})
        
        if drift_detected:
            print(f"Drift detected at cycle {result['current_idx']}.")
        else:
            print(f"No drift at cycle {result['current_idx']}.")
            
        return {
            "drift_detected": drift_detected,
            "p_values": p_values,
            "current_data": current_data,
            "shap_values": None,
            "action_decision": None,
            "reasoning": None,
            "current_idx": result["current_idx"]
        }
    
    def diagnose_drift_node(state: AgentState):
        """Diagnoses the drift using SHAP Values."""
        print("---NODE: diagnose_drift---")
        
        current_data = state["current_data"]
        
        # Call the SHAP tool
        shap_vals = diagnose_with_shap.invoke({"sensor_data": current_data})
        
        print("SHAP Diagnosis completed.")
        return {"shap_values": shap_vals}
    
    def execute_action_node(state: AgentState):
        """Uses LLM to evaluate p-values and SHAP, and decides on an action."""
        print("---NODE: execute_action---")
        
        p_values = state["p_values"]
        shap_values = state["shap_values"]
        
        prompt = f"""
        You are an Expert AI Engineer specializing in Agentic MLOps and Industrial Predictive Maintenance.
        Your task is to triage a diagnosed drift event based on the following indicators:
        
        1. Statistical P-Values from KS-Test:
        {json.dumps(p_values, indent=2)}
        
        2. SHAP Feature Importances (Diagnostic):
        {json.dumps(shap_values, indent=2)}
        
        Triage Logic Constraints:
        - If SHAP indicates a single sensor is a massive outlier (e.g., highly dominant importance while others are small and p-value is low), output: ACTION: ISSUE_REPLACEMENT_TICKET.
        - If SHAP indicates a global shift across multiple sensors (multiple features show importance and low p-values), output: ACTION: RETRAIN_MODEL.
        
        Analyze the situation and output your reasoning and Action.
        """
        
        t0 = time.time()
        result = structured_llm.invoke(prompt)
        t1 = time.time()
        latency = t1 - t0
        
        # Safely print reasoning bypassing Windows console charmap limits
        safe_reasoning = str(result.reasoning).encode('ascii', errors='ignore').decode('ascii') if result.reasoning else ""
        print(f"Agent Reasoning: {safe_reasoning}")
        print(f"Agent Action: {result.action}")
        print(f"LLM Latency: {latency:.2f} seconds")
        
        return {
            "action_decision": result.action,
            "reasoning": result.reasoning,
            "llm_latency": latency
        }
    
    def route_drift(state: AgentState):
        """Routes the state based on whether drift was detected."""
        if state.get("drift_detected", False):
            return "diagnose_drift"
        return END
    
    # Build the Graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("monitor_data", monitor_data_node)
    workflow.add_node("diagnose_drift", diagnose_drift_node)
    workflow.add_node("execute_action", execute_action_node)
    
    # Set the entrypoint
    workflow.set_entry_point("monitor_data")
    
    # Add edges
    workflow.add_conditional_edges(
        "monitor_data",
        route_drift,
        {
            "diagnose_drift": "diagnose_drift",
            END: END
        }
    )
    workflow.add_edge("diagnose_drift", "execute_action")
    workflow.add_edge("execute_action", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app, monitor
