# AgentMaintain

AgentMaintain is an experimental Python project for simulating an agentic predictive maintenance pipeline using the NASA C-MAPSS turbofan engine dataset. It combines streaming drift detection, SHAP explainability, structured agent decision-making, and lightweight evaluation instrumentation.

## Project Overview

This repository demonstrates an end-to-end maintenance agent workflow:

- `monitor.py` simulates streaming sensor telemetry from the C-MAPSS dataset and detects drift using the Kolmogorov-Smirnov test.
- `agent_tools.py` defines a SHAP diagnostic tool with a dummy RandomForest model to identify which sensors are driving the anomaly.
- `graph_builder.py` constructs a state graph pipeline where the agent monitors data, diagnoses drift, and uses an LLM to decide between actionable responses.
- `main.py` runs a short simulation that steps through the stream, injects faults periodically, and prints final triage actions.
- `monitor.py` also contains evaluation utilities to compare multiple model backends and track performance metrics, VRAM, and traces.

## Key Concepts

- Streaming drift detection: continuous data windows are compared against a reference baseline using KS-tests.
- SHAP explainability: feature importance is computed for the current sensor vector to support diagnostic reasoning.
- Agent orchestration: a LangGraph state graph orchestrates monitoring, diagnosis, and action execution.
- Structured output: LLM responses are constrained to a defined schema (`TriageAction`) for predictable actions.

## Repository Structure

- `main.py` - primary simulation entrypoint for the AgentMaintain orchestrator.
- `agent_tools.py` - SHAP diagnostic tool and dummy model training helper.
- `monitor.py` - `StreamingMonitor` implementation and ablation/evaluation runner.
- `graph_builder.py` - graph workflow assembly that wires monitoring, diagnosis, and LLM triage.
- `evaluator.py` - helper for evaluation metrics and result summarization.
- `agent_tools.py` - tool definitions used in the agent workflow.
- `requirements.txt` - Python dependencies.
- `data/CMAPSSData/` - local dataset files (ignored by Git via `.gitignore`).
- `plots/` - generated evaluation plots.
- `traces.json` - serialized execution traces.
- `evaluation_metrics.csv` - collected performance metrics.

## Getting Started

### Prerequisites

- Python 3.10+ recommended
- GPU support is optional, but `pynvml` is used for VRAM tracking if available.
- The `data/CMAPSSData/` folder must contain the C-MAPSS dataset files, especially `train_FD001.txt`.

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Recommended Environment

Use a virtual environment to isolate dependencies:

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

This starts the orchestrator, loads the C-MAPSS stream, injects synthetic faults at regular intervals, and prints the agent's triage decisions.

### Run Evaluation / Ablation

```powershell
python monitor.py
```

This executes the `run_ablation()` path, evaluates multiple models, records metrics to `evaluation_metrics.csv`, saves traces to `traces.json`, and writes plots under `plots/`.

## Configuration

- `graph_builder.py` uses `data/CMAPSSData/train_FD001.txt` by default.
- `monitor.py` uses the same default dataset path and supports changing `step_size`, `max_loops`, and fault injection logic.
- `.gitignore` now includes `data/` so local dataset files are not committed.

## Notes

- The SHAP diagnostic tool uses a dummy `RandomForestRegressor` trained on the first training file and a placeholder target (`time_in_cycles`).
- The agent action logic in `graph_builder.py` is intentionally simplified for demo purposes:
  - `ISSUE_REPLACEMENT_TICKET` for localized sensor faults
  - `RETRAIN_MODEL` for broad drift across multiple sensors
- The LLM backend is configured via `langchain_ollama.ChatOllama` and requires the appropriate model to be available.

## Troubleshooting

- If `train_FD001.txt` is not found, verify the `data/CMAPSSData/` path and file names.
- If the LLM backend fails, confirm the `ChatOllama` model name and local Ollama installation.
- If VRAM tracking fails, the code falls back gracefully when `pynvml` cannot initialize.

## License

This repository is provided as-is for research and experimentation.
