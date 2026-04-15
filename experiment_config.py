# experiment_config.py
# Single source of truth for all experiment hyperparameters

MODEL_CONFIGS = [
    ("qwen2.5:7b_only",    ["qwen2.5:7b"]),
    ("phi3.5_only",        ["phi3.5"]),
    ("consensus_qwen_phi", ["qwen2.5:7b", "phi3.5"]),
]

STREAMING_CONFIG = {
    "reference_window_size": 50,
    "current_window_size":   30,
    "p_value_threshold":     0.05,
    "multiple_testing_correction": True,  # Bonferroni
    "step_size":             20,
}

FAULT_SCHEDULE = [
    {"trigger_cycle": 100, "sensor_id": "sensor_14", "fault_type": "sensor_failure",    "severity": 10.0},
    {"trigger_cycle": 300, "sensor_id": "sensor_7",  "fault_type": "operational_drift", "drift_rate": 0.15},
    {"trigger_cycle": 500, "sensor_id": "sensor_11", "fault_type": "sensor_failure",    "severity": 7.0},
]

EVALUATION_CONFIG = {
    "max_loops":    100,
    "llm_retries":  3,
    "retry_backoff_base": 2,  # seconds, exponential: 2^attempt
}

PLOT_CONFIG = {
    "dpi":     150,
    "figsize": (10, 6),
    "style":   "whitegrid",
    "action_colors": {
        "ISSUE_REPLACEMENT_TICKET": "#e74c3c",
        "RETRAIN_MODEL":            "#2ecc71",
    }
}
