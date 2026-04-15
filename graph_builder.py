from typing import TypedDict, Dict, Optional, List, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import json
import time

from agent_tools import diagnose_with_shap, TriageAction, retrieve_maintenance_manuals
from monitor import StreamingMonitor
from experiment_config import EVALUATION_CONFIG

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
    manual_content: Optional[str]
    consensus: Optional[str]
    current_fault_type: Optional[str]
    monitor_latency_s: Optional[float]
    plan_latency_s: Optional[float]
    shap_latency_s: Optional[float]
    consensus_confidence: Optional[float]
    winning_model: Optional[str]


def build_graph(models: Optional[List[str]] = None, step_size: int = 20):
    if models is None:
        models = ["qwen2.5:7b", "llama3.1:8b"]

    structured_llms = {}
    for model_name in models:
        try:
            llm = ChatOllama(model=model_name, temperature=0)
            structured_llms[model_name] = llm.with_structured_output(TriageAction)
        except Exception as exc:
            print(f"Warning: Skipping model '{model_name}' because it could not be loaded: {exc}")

    if not structured_llms:
        raise RuntimeError(
            "No supported Ollama models were available. "
            "Install at least one model and update the model list in graph_builder.py."
        )

    monitor = StreamingMonitor(data_path="data/CMAPSSData/train_FD001.txt", step_size=step_size)

    def monitor_data_node(state: AgentState):
        """Monitors the stream and detects drift."""
        start_t = time.perf_counter()
        print("---NODE: monitor_data---")

        result = monitor.detect_drift()
        elapsed = time.perf_counter() - start_t

        if result["status"] == "end_of_stream":
            return {"drift_detected": False, "messages": ["End of data stream reached."], "monitor_latency_s": elapsed}

        drift_detected = result["drift_detected"]
        p_values = result.get("p_values", {})
        current_data = result.get("current_data", {})
        current_fault_type = result.get("current_fault_type")

        if drift_detected:
            print(f"Drift detected at cycle {result['current_idx']} (fault type: {current_fault_type}).")
        else:
            print(f"No drift at cycle {result['current_idx']}.")

        return {
            "drift_detected": drift_detected,
            "p_values": p_values,
            "current_data": current_data,
            "shap_values": None,
            "action_decision": None,
            "reasoning": None,
            "current_idx": result["current_idx"],
            "manual_content": None,
            "consensus": None,
            "current_fault_type": current_fault_type,
            "monitor_latency_s": elapsed
        }

    def plan_node(state: AgentState):
        """Fetches relevant maintenance guidance before diagnosis."""
        start_t = time.perf_counter()
        print("---NODE: plan---")
        manual_content = retrieve_maintenance_manuals.invoke({"query": "sensor failure vs operational drift"})
        elapsed = time.perf_counter() - start_t
        return {"manual_content": manual_content, "plan_latency_s": elapsed}

    def diagnose_drift_node(state: AgentState):
        """Diagnoses the drift using SHAP values."""
        start_t = time.perf_counter()
        print("---NODE: diagnose_drift---")

        current_data = state["current_data"]
        shap_vals = diagnose_with_shap.invoke({"sensor_data": current_data})

        elapsed = time.perf_counter() - start_t
        print("SHAP Diagnosis completed.")
        return {"shap_values": shap_vals, "shap_latency_s": elapsed}

    def run_consensus(llms: Dict[str, Any], prompt: str):
        decisions = []
        reasonings = []
        confidences = []
        model_names = []
        total_latency = 0.0

        for model_name, structured_llm in llms.items():
            success = False
            for attempt in range(EVALUATION_CONFIG["llm_retries"]):
                try:
                    start_t = time.perf_counter()
                    result = structured_llm.invoke(prompt)
                    latency = time.perf_counter() - start_t
                    total_latency += latency
                    decisions.append(result.action)
                    reasonings.append(f"[{model_name}] {result.reasoning}")
                    confidences.append(result.confidence)
                    model_names.append(model_name)
                    print(f"{model_name} -> action: {result.action} | confidence: {result.confidence:.2f} | latency: {latency:.2f}s")
                    success = True
                    break
                except Exception as exc:
                    print(f"Warning: Retry {attempt+1}/{EVALUATION_CONFIG['llm_retries']} for '{model_name}': {exc}")
                    time.sleep(EVALUATION_CONFIG["retry_backoff_base"] ** attempt)
            if not success:
                print(f"Error: Skipping model '{model_name}' after {EVALUATION_CONFIG['llm_retries']} failures.")

        if not decisions:
            raise RuntimeError("All model invocations failed during consensus.")

        vote_counts = {}
        for action in decisions:
            vote_counts[action] = vote_counts.get(action, 0) + 1

        consensus_action = max(vote_counts, key=vote_counts.get)
        winning_model = None
        consensus_confidence = None

        if len(decisions) > 1 and vote_counts[consensus_action] == 1:
            max_conf_idx = confidences.index(max(confidences))
            if confidences.count(max(confidences)) > 1:
                consensus_action = "RETRAIN_MODEL"
                reasonings.append("Consensus fallback to RETRAIN_MODEL due to identical confidences.")
            else:
                consensus_action = decisions[max_conf_idx]
                winning_model = model_names[max_conf_idx]
                consensus_confidence = confidences[max_conf_idx]
                reasonings.append(f"Consensus resolved by confidence: {winning_model} ({consensus_confidence:.2f}) -> {consensus_action}")
        else:
            best_conf = -1
            for idx, action in enumerate(decisions):
                if action == consensus_action and confidences[idx] > best_conf:
                    best_conf = confidences[idx]
                    winning_model = model_names[idx]
                    consensus_confidence = confidences[idx]

        return consensus_action, reasonings, total_latency / len(decisions), consensus_confidence, winning_model

    def execute_action_node(state: AgentState):
        """Uses multi-model consensus to choose a final action."""
        print("---NODE: execute_action---")

        p_values = state["p_values"]
        shap_values = state["shap_values"]
        manual_content = state.get("manual_content", "")
        current_fault_type = state.get("current_fault_type", "unknown")

        prompt = f"""
You are an Expert AI Engineer specializing in Agentic MLOps and Industrial Predictive Maintenance.
A primary goal is to distinguish between sensor hardware failure and operational/environmental drift.

Context:
- Fault type hint: {current_fault_type}
- Statistical P-Values from KS-test:
{json.dumps(p_values, indent=2)}

- SHAP feature importances:
{json.dumps(shap_values, indent=2)}

Maintenance guidance:
{manual_content}

Triage Rules:
- If one or a few sensors dominate SHAP importance and their p-values are low, choose ACTION: ISSUE_REPLACEMENT_TICKET.
- If many sensors show low p-values and distributed SHAP importance, choose ACTION: RETRAIN_MODEL.
- Prefer the consensus action from multiple models.

Reply with concise reasoning and one of these exact actions: ISSUE_REPLACEMENT_TICKET or RETRAIN_MODEL.
"""

        consensus_action, reasonings, avg_latency, consensus_confidence, winning_model = run_consensus(structured_llms, prompt)
        final_reasoning = "\n---\n".join(reasonings)

        return {
            "action_decision": consensus_action,
            "reasoning": final_reasoning,
            "llm_latency": avg_latency,
            "consensus": consensus_action,
            "consensus_confidence": consensus_confidence,
            "winning_model": winning_model
        }

    def route_drift(state: AgentState):
        if state.get("drift_detected", False):
            return "plan"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("monitor_data", monitor_data_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("diagnose_drift", diagnose_drift_node)
    workflow.add_node("execute_action", execute_action_node)

    workflow.set_entry_point("monitor_data")
    workflow.add_conditional_edges(
        "monitor_data",
        route_drift,
        {"plan": "plan", END: END}
    )
    workflow.add_edge("plan", "diagnose_drift")
    workflow.add_edge("diagnose_drift", "execute_action")
    workflow.add_edge("execute_action", END)

    app = workflow.compile()
    return app, monitor
