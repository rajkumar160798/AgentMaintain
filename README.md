# AgentMaintain

AgentMaintain is a prototype for autonomous drift triage and root-cause analysis using multi-model consensus. The project demonstrates how an agent can go beyond a binary retrain trigger and decide whether an observed anomaly indicates a sensor failure or an operational/environmental shift.

## Project Overview

This repository implements a practical version of the "Agentic Drift-Triage" concept:

- `monitor.py` simulates a streaming C-MAPSS sensor feed and detects drift using KS-tests.
- `agent_tools.py` provides SHAP explainability and a retrieval-style maintenance manual tool.
- `graph_builder.py` builds a Plan-Diagnose-Act workflow with multi-model SLM consensus.
- `main.py` runs a simulation that injects synthetic sensor faults and routes decisions through the agent.
- `monitor.py` also contains evaluation utilities for tracking performance, latency, VRAM, and action traces.

## What This Repo Adds

- Multi-model consensus by querying multiple LLM backends and selecting the action with majority agreement.
- RAG-inspired maintenance guidance through a local retrieval tool with sensor failure and operational drift descriptions.
- Sensor failure injection and operational drift injection support in the streaming monitor.
- A more explicit Plan-Diagnose-Act cycle in the graph workflow.

## Repository Structure

- `main.py` - primary simulation entrypoint for the AgentMaintain orchestrator.
- `agent_tools.py` - SHAP diagnosis plus retrieval-based maintenance guidance supporting the agent.
- `monitor.py` - streaming monitor implementation with fault injection and drift detection.
- `graph_builder.py` - builds a LangGraph workflow with multi-model consensus.
- `evaluator.py` - evaluation metric collection and plotting utilities.
- `requirements.txt` - Python dependencies.
- `data/CMAPSSData/` - local dataset files (ignored by Git via `.gitignore`).
- `plots/` - generated evaluation plots.
- `traces.json` - serialized execution traces.
- `evaluation_metrics.csv` - collected evaluation metrics.

## Getting Started

### Prerequisites

- Python 3.10+ recommended
- A working Ollama installation with at least one of the supported models (`qwen2.5:7b`, `phi3.5`, or similar)
- The default code attempts both `qwen2.5:7b` and `phi3.5`, but missing models are skipped automatically.
- The `data/CMAPSSData/` folder must contain the C-MAPSS dataset files, especially `train_FD001.txt`.

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Recommended Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

### Run the Main Simulation

```powershell
python main.py
```

This starts the orchestrator, streams the C-MAPSS dataset, injects synthetic faults periodically, and prints the agent's final triage actions.

### Run Evaluation / Ablation

```powershell
python monitor.py
```

This evaluates multiple model backends, saves action traces to `traces.json`, writes metrics to `evaluation_metrics.csv`, and generates plots under `plots/`.

## Configuration

- `graph_builder.py` defaults to multi-model consensus and uses `data/CMAPSSData/train_FD001.txt`.
- `monitor.py` supports both `inject_sensor_failure()` and `inject_operational_drift()`.
- `.gitignore` includes `data/` so your local dataset files remain out of source control.

## Notes

- The SHAP diagnostic tool is trained on a lightweight RandomForest and serves as a proxy explainability module.
- The workflow now includes a planning phase that retrieves relevant maintenance guidance before diagnosis.
- Consensus logic prefers majority action; if models disagree strongly, the fallback is `RETRAIN_MODEL`.

## Troubleshooting

- If `train_FD001.txt` is not found, verify the `data/CMAPSSData/` folder and file list.
- If Ollama model loading fails, confirm the model names and installation.
- If consensus does not behave as expected, inspect `graph_builder.py` and the contained prompt guidance.

## License

This repository is provided as-is for research and experimentation.
