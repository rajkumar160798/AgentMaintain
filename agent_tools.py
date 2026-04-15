import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Define Pydantic Schema for SLM structured output
class TriageAction(BaseModel):
    reasoning: str = Field(description="The agent's reasoning for why a specific action was chosen based on p-value and SHAP features.")
    action: str = Field(description="The determined action. Must be either 'ISSUE_REPLACEMENT_TICKET' or 'RETRAIN_MODEL'.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Model's self-assessed confidence in this triage action, between 0.0 and 1.0")

class DummyModelCache:
    _model = None
    _explainer = None
    _features = [f'sensor_{i}' for i in range(1, 22)]

    @classmethod
    def get_explainer(cls, data_path: str):
        if cls._explainer is None:
            print("Training dummy RandomForest for SHAP explainability...")
            # Load a small subset of data to train the dummy model
            columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + cls._features
            df = pd.read_csv(data_path, sep=r'\s+', header=None, names=columns)
            # Use a dummy target: remaining useful life or just time_in_cycles for simplicity
            X = df[cls._features]
            y = df['time_in_cycles']
            
            # Train a quick random forest
            cls._model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
            cls._model.fit(X, y)
            
            # Initialize TreeExplainer
            cls._explainer = shap.TreeExplainer(cls._model)
            print("Dummy model trained and SHAP explainer initialized.")
            
        return cls._explainer, cls._features

# Define global path for data to use in the tool.
DATA_PATH = "data/CMAPSSData/train_FD001.txt"

MAINTENANCE_MANUALS = {
    "sensor_failure": (
        "Sensor failure often appears as a localized anomaly in one or a few sensors. "
        "Common signals include a stuck-at-constant output, sudden jumps to an extreme value, "
        "or one sensor diverging while others remain stable. "
        "A recommended response is to inspect and replace the affected sensor rather than retraining the model."
    ),
    "operational_drift": (
        "Operational drift typically affects many sensors at once and emerges as a gradual shift in baseline readings. "
        "The signal may evolve over time due to changing engine conditions, load, or environment. "
        "This is usually best addressed by retraining or updating the predictive model on the new operating distribution."
    ),
    "general": (
        "When triaging drift, compare statistical p-values across sensors with SHAP feature importance. "
        "A dominant sensor importance combined with a low p-value strongly suggests hardware failure. "
        "Distributed importance across many sensors suggests a systemic shift in the environment or process."
    )
}

@tool
def retrieve_maintenance_manuals(query: str) -> str:
    """
    Returns a short maintenance-style summary for the requested failure mode.
    """
    normalized = query.lower()
    if "sensor" in normalized and "failure" in normalized:
        return MAINTENANCE_MANUALS["sensor_failure"]
    if "operational" in normalized or "drift" in normalized:
        return MAINTENANCE_MANUALS["operational_drift"]
    return MAINTENANCE_MANUALS["general"]

@tool
def diagnose_with_shap(sensor_data: dict) -> dict:
    """
    Applies SHAP TreeExplainer to the current sensor readings to determine feature importance.
    This helps the agent understand WHICH sensors are contributing most to the prediction,
    acting as a diagnostic for drift.
    
    Args:
        sensor_data (dict): Dictionary mapping sensor names (e.g., 'sensor_1') to current float values.

    Returns:
        dict: A dictionary mapping sensor names to their SHAP importance values. 
              Higher absolute values indicate higher anomaly contribution.
    """
    explainer, features = DummyModelCache.get_explainer(DATA_PATH)
    
    # Extract only the 21 sensor features in the correct order
    input_array = np.array([[sensor_data.get(f, 0.0) for f in features]])
    
    # Calculate SHAP values for the single row
    shap_values = explainer.shap_values(input_array)
    
    # If the model output is multidimensional, take the first dimension
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    shap_values = shap_values[0] # Get the first (and only) row
    
    # Map back to feature names
    importance = {features[i]: float(shap_values[i]) for i in range(len(features))}
    
    # Sort by absolute importance for convenience
    importance = dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))
    
    return importance
