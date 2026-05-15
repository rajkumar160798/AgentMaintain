# experiment_config.py — single source of truth for all experiment hyperparameters

import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODEL_CONFIGS = [
    ("qwen2.5:7b_only",    ["qwen2.5:7b"]),
    ("phi3.5_only",        ["phi3.5"]),
    ("consensus_qwen_phi", ["qwen2.5:7b", "phi3.5"]),
]

STREAMING_CONFIG = {
    "reference_window_size":      50,
    "current_window_size":        30,
    "p_value_threshold":          0.05,
    "multiple_testing_correction": True,   # Bonferroni correction across 21 sensors
    "step_size":                  20,
}

# Fault schedule for the 100-loop ablation (step=20, starting idx=80).
# Trigger cycles span the full evaluation range (≈80–2080) to ensure
# balanced coverage of sensor-failure, operational-drift, and normal states.
FAULT_SCHEDULE = [
    # --- Phase 1: sudden catastrophic failure (sensor_14) ---
    {"trigger_cycle": 100,  "sensor_id": "sensor_14", "fault_type": "sensor_failure",    "severity":   10.0},
    # --- Phase 2: gradual operational drift (sensor_7) ---
    {"trigger_cycle": 300,  "sensor_id": "sensor_7",  "fault_type": "operational_drift", "drift_rate":  0.15},
    # --- Phase 3: second hard fault at moderate severity (sensor_11) ---
    {"trigger_cycle": 500,  "sensor_id": "sensor_11", "fault_type": "sensor_failure",    "severity":    7.0},
    # --- Phase 4: slow environmental drift on a secondary sensor ---
    {"trigger_cycle": 700,  "sensor_id": "sensor_9",  "fault_type": "operational_drift", "drift_rate":  0.05},
    # --- Phase 5: severe instantaneous failure on a rarely-faulted sensor ---
    {"trigger_cycle": 900,  "sensor_id": "sensor_3",  "fault_type": "sensor_failure",    "severity":    8.5},
    # --- Phase 6: subtle drift test (low rate to challenge the KS detector) ---
    {"trigger_cycle": 1200, "sensor_id": "sensor_21", "fault_type": "operational_drift", "drift_rate":  0.02},
    # --- Phase 7: high-severity fault on a temperature sensor ---
    {"trigger_cycle": 1500, "sensor_id": "sensor_12", "fault_type": "sensor_failure",    "severity":   12.0},
]

EVALUATION_CONFIG = {
    "max_loops":          100,
    "llm_retries":        3,
    "retry_backoff_base": 2,   # exponential back-off: 2^attempt seconds
}

PLOT_CONFIG = {
    "dpi":     150,
    "figsize": (10, 6),
    "style":   "whitegrid",
    "action_colors": {
        "ISSUE_REPLACEMENT_TICKET": "#e74c3c",
        "RETRAIN_MODEL":            "#2ecc71",
    },
}

# Evidence-Gated LLM Routing (EGLR) thresholds
# FCI >= fci_high_threshold  →  direct ISSUE_REPLACEMENT_TICKET (no LLM)
# FCI <= fci_low_threshold   →  direct RETRAIN_MODEL (no LLM)
# Otherwise                  →  invoke LLM consensus
# Thresholds are calibrated from the empirical FCI distributions in calibrate_fci.py
EGLR_CONFIG = {
    "fci_high_threshold": 0.60,
    "fci_low_threshold":  0.15,
}

# Window-size grid for the model-free KS sensitivity analysis
SENSITIVITY_CONFIGS = [
    {"reference_window_size": 25,  "current_window_size": 15},
    {"reference_window_size": 50,  "current_window_size": 30},  # default
    {"reference_window_size": 75,  "current_window_size": 45},
    {"reference_window_size": 100, "current_window_size": 50},
]
