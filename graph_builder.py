from typing import TypedDict, Dict, Optional, List, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import json
import time

from agent_tools import (
    diagnose_with_shap,
    TriageAction,
    retrieve_maintenance_manuals,
    compute_drift_evidence_score,
    compute_fault_concentration_index,
)
from monitor import StreamingMonitor
from experiment_config import EVALUATION_CONFIG, EGLR_CONFIG


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
    # --- EGLR additions ---
    des_scores: Optional[Dict[str, float]]
    fci: Optional[float]
    llm_bypassed: Optional[bool]


def build_graph(
    models: Optional[List[str]] = None,
    step_size: int = 20,
    eglr_mode: bool = False,
):
    """
    Build the AgentMaintain LangGraph workflow.

    Parameters
    ----------
    models : list of Ollama model names to use for consensus.
    step_size : streaming window step size in cycles.
    eglr_mode : if True, insert a DES-FCI node after SHAP diagnosis and
                use Evidence-Gated LLM Routing (EGLR) to bypass LLM inference
                when the Fault Concentration Index is unambiguous.
    """
    if models is None:
        models = ["qwen2.5:7b", "phi3.5"]

    structured_llms = {}
    for model_name in models:
        try:
            llm = ChatOllama(model=model_name, temperature=0)
            structured_llms[model_name] = llm.with_structured_output(TriageAction)
        except Exception as exc:
            print(f"Warning: Skipping model '{model_name}': {exc}")

    if not structured_llms:
        raise RuntimeError(
            "No supported Ollama models were available. "
            "Install at least one model and update the model list."
        )

    monitor = StreamingMonitor(
        data_path="data/CMAPSSData/train_FD001.txt", step_size=step_size
    )

    # ------------------------------------------------------------------
    # Node definitions
    # ------------------------------------------------------------------

    def monitor_data_node(state: AgentState):
        start_t = time.perf_counter()
        print("---NODE: monitor_data---")
        result = monitor.detect_drift()
        elapsed = time.perf_counter() - start_t

        if result["status"] == "end_of_stream":
            return {
                "drift_detected": False,
                "messages": ["End of data stream reached."],
                "monitor_latency_s": elapsed,
            }

        drift_detected = result["drift_detected"]
        if drift_detected:
            print(f"Drift detected at cycle {result['current_idx']} (fault type: {result.get('current_fault_type')}).")
        else:
            print(f"No drift at cycle {result['current_idx']}.")

        return {
            "drift_detected": drift_detected,
            "p_values": result.get("p_values", {}),
            "current_data": result.get("current_data", {}),
            "shap_values": None,
            "action_decision": None,
            "reasoning": None,
            "current_idx": result["current_idx"],
            "manual_content": None,
            "consensus": None,
            "current_fault_type": result.get("current_fault_type"),
            "monitor_latency_s": elapsed,
            "des_scores": None,
            "fci": None,
            "llm_bypassed": False,
        }

    def plan_node(state: AgentState):
        start_t = time.perf_counter()
        print("---NODE: plan---")
        manual_content = retrieve_maintenance_manuals.invoke(
            {"query": "sensor failure vs operational drift"}
        )
        elapsed = time.perf_counter() - start_t
        return {"manual_content": manual_content, "plan_latency_s": elapsed}

    def diagnose_drift_node(state: AgentState):
        start_t = time.perf_counter()
        print("---NODE: diagnose_drift---")
        shap_vals = diagnose_with_shap.invoke({"sensor_data": state["current_data"]})
        elapsed = time.perf_counter() - start_t
        print("SHAP Diagnosis completed.")
        return {"shap_values": shap_vals, "shap_latency_s": elapsed}

    # --- EGLR-specific nodes -------------------------------------------

    def compute_des_fci_node(state: AgentState):
        """
        Compute the Drift Evidence Score (DES) and Fault Concentration Index (FCI).
        This node sits between SHAP diagnosis and the LLM execution node.
        DES_i = -ln(p_i) * |phi_i|   (joint KS-SHAP signal)
        FCI   = Gini(DES)             (scalar triage signal)
        """
        print("---NODE: compute_des_fci---")
        des = compute_drift_evidence_score(
            state["p_values"], state["shap_values"]
        )
        fci = compute_fault_concentration_index(des)
        print(f"FCI = {fci:.4f}")
        return {"des_scores": des, "fci": fci}

    def direct_fault_node(state: AgentState):
        """EGLR fast-path: concentrated evidence → sensor failure, no LLM call."""
        fci = state.get("fci", 1.0)
        msg = (
            f"[EGLR-BYPASS] FCI={fci:.4f} >= {EGLR_CONFIG['fci_high_threshold']} "
            f"(concentrated evidence) → algorithmic decision: ISSUE_REPLACEMENT_TICKET"
        )
        print(f"---NODE: direct_fault (EGLR bypass)--- {msg}")
        return {
            "action_decision": "ISSUE_REPLACEMENT_TICKET",
            "reasoning": msg,
            "llm_latency": 0.0,
            "consensus": "ISSUE_REPLACEMENT_TICKET",
            "consensus_confidence": fci,
            "winning_model": "EGLR-ALGORITHMIC",
            "llm_bypassed": True,
        }

    def direct_drift_node(state: AgentState):
        """EGLR fast-path: distributed evidence → operational drift, no LLM call."""
        fci = state.get("fci", 0.0)
        msg = (
            f"[EGLR-BYPASS] FCI={fci:.4f} <= {EGLR_CONFIG['fci_low_threshold']} "
            f"(distributed evidence) → algorithmic decision: RETRAIN_MODEL"
        )
        print(f"---NODE: direct_drift (EGLR bypass)--- {msg}")
        return {
            "action_decision": "RETRAIN_MODEL",
            "reasoning": msg,
            "llm_latency": 0.0,
            "consensus": "RETRAIN_MODEL",
            "consensus_confidence": 1.0 - fci,
            "winning_model": "EGLR-ALGORITHMIC",
            "llm_bypassed": True,
        }

    # --- LLM consensus node -------------------------------------------

    def run_consensus(llms: Dict[str, Any], prompt: str):
        decisions, reasonings, confidences, model_names = [], [], [], []
        total_latency = 0.0

        for model_name, structured_llm in llms.items():
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
                    print(
                        f"{model_name} -> action: {result.action} "
                        f"| confidence: {result.confidence:.2f} | latency: {latency:.2f}s"
                    )
                    break
                except Exception as exc:
                    print(f"Warning: Retry {attempt+1}/{EVALUATION_CONFIG['llm_retries']} for '{model_name}': {exc}")
                    time.sleep(EVALUATION_CONFIG["retry_backoff_base"] ** attempt)
            else:
                print(f"Error: Skipping model '{model_name}' after all retries.")

        if not decisions:
            raise RuntimeError("All model invocations failed during consensus.")

        vote_counts = {}
        for action in decisions:
            vote_counts[action] = vote_counts.get(action, 0) + 1

        consensus_action = max(vote_counts, key=vote_counts.get)
        winning_model = None
        consensus_confidence = None

        if len(decisions) > 1 and vote_counts[consensus_action] == 1:
            max_conf = max(confidences)
            if confidences.count(max_conf) > 1:
                consensus_action = "RETRAIN_MODEL"
                reasonings.append("Consensus fallback (identical confidences) → RETRAIN_MODEL.")
            else:
                idx = confidences.index(max_conf)
                consensus_action = decisions[idx]
                winning_model = model_names[idx]
                consensus_confidence = confidences[idx]
                reasonings.append(
                    f"Tie resolved by confidence: {winning_model} "
                    f"({consensus_confidence:.2f}) → {consensus_action}"
                )
        else:
            best_conf = -1.0
            for idx, action in enumerate(decisions):
                if action == consensus_action and confidences[idx] > best_conf:
                    best_conf = confidences[idx]
                    winning_model = model_names[idx]
                    consensus_confidence = confidences[idx]

        return (
            consensus_action,
            reasonings,
            total_latency / len(decisions),
            consensus_confidence,
            winning_model,
        )

    def execute_action_node(state: AgentState):
        print("---NODE: execute_action---")
        p_values = state["p_values"]
        shap_values = state["shap_values"]
        manual_content = state.get("manual_content", "")
        current_fault_type = state.get("current_fault_type", "unknown")

        # Include FCI in the prompt when EGLR mode is active (ambiguous band)
        fci_hint = ""
        if state.get("fci") is not None:
            fci_hint = (
                f"\nFault Concentration Index (FCI): {state['fci']:.4f}  "
                f"[0=distributed/drift, 1=concentrated/fault] — ambiguous band, LLM invoked."
            )

        prompt = f"""
You are an Expert AI Engineer specializing in Agentic MLOps and Industrial Predictive Maintenance.
A primary goal is to distinguish between sensor hardware failure and operational/environmental drift.

Context:
- Fault type hint: {current_fault_type}{fci_hint}
- Statistical P-Values from KS-test:
{json.dumps(p_values, indent=2)}

- SHAP feature importances (RUL-grounded):
{json.dumps(shap_values, indent=2)}

Maintenance guidance:
{manual_content}

Triage Rules:
- If one or a few sensors dominate SHAP importance AND their p-values are low → ISSUE_REPLACEMENT_TICKET.
- If many sensors show low p-values with distributed SHAP importance → RETRAIN_MODEL.

Reply with concise reasoning and EXACTLY one of: ISSUE_REPLACEMENT_TICKET or RETRAIN_MODEL.
"""

        consensus_action, reasonings, avg_latency, consensus_confidence, winning_model = (
            run_consensus(structured_llms, prompt)
        )
        final_reasoning = "\n---\n".join(reasonings)

        return {
            "action_decision": consensus_action,
            "reasoning": final_reasoning,
            "llm_latency": avg_latency,
            "consensus": consensus_action,
            "consensus_confidence": consensus_confidence,
            "winning_model": winning_model,
            "llm_bypassed": False,
        }

    # ------------------------------------------------------------------
    # Routing functions
    # ------------------------------------------------------------------

    def route_drift(state: AgentState):
        if state.get("drift_detected", False):
            return "plan"
        return END

    def route_eglr(state: AgentState):
        """Evidence-Gated LLM Routing: use FCI to decide whether to bypass LLM."""
        fci = state.get("fci", 0.5)
        hi = EGLR_CONFIG["fci_high_threshold"]
        lo = EGLR_CONFIG["fci_low_threshold"]
        if fci >= hi:
            return "direct_fault"
        if fci <= lo:
            return "direct_drift"
        return "execute_action"

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    workflow = StateGraph(AgentState)
    workflow.add_node("monitor_data",  monitor_data_node)
    workflow.add_node("plan",          plan_node)
    workflow.add_node("diagnose_drift", diagnose_drift_node)
    workflow.add_node("execute_action", execute_action_node)

    workflow.set_entry_point("monitor_data")
    workflow.add_conditional_edges(
        "monitor_data", route_drift, {"plan": "plan", END: END}
    )
    workflow.add_edge("plan", "diagnose_drift")

    if eglr_mode:
        workflow.add_node("compute_des_fci", compute_des_fci_node)
        workflow.add_node("direct_fault",    direct_fault_node)
        workflow.add_node("direct_drift",    direct_drift_node)

        workflow.add_edge("diagnose_drift", "compute_des_fci")
        workflow.add_conditional_edges(
            "compute_des_fci",
            route_eglr,
            {
                "direct_fault":   "direct_fault",
                "direct_drift":   "direct_drift",
                "execute_action": "execute_action",
            },
        )
        workflow.add_edge("direct_fault",   END)
        workflow.add_edge("direct_drift",   END)
    else:
        workflow.add_edge("diagnose_drift", "execute_action")

    workflow.add_edge("execute_action", END)

    app = workflow.compile()
    return app, monitor
