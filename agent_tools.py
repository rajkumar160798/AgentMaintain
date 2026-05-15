import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal


class TriageAction(BaseModel):
    reasoning: str = Field(description="The agent's reasoning for why a specific action was chosen based on p-value and SHAP features.")
    action: Literal["ISSUE_REPLACEMENT_TICKET", "RETRAIN_MODEL"] = Field(
        description="The determined action. Must be exactly 'ISSUE_REPLACEMENT_TICKET' or 'RETRAIN_MODEL'."
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Model's self-assessed confidence in this triage action, between 0.0 and 1.0")


class RULSHAPExplainer:
    """
    Trains a RandomForest to predict Remaining Useful Life (RUL) from sensor readings,
    then uses TreeSHAP to attribute each sensor's contribution to the RUL prediction.
    RUL is computed per engine unit as: max_cycles_in_unit - current_cycle.
    High-magnitude SHAP values indicate sensors that most influence degradation prediction.
    """
    _model = None
    _explainer = None
    _features = [f'sensor_{i}' for i in range(1, 22)]

    @classmethod
    def get_explainer(cls, data_path: str):
        if cls._explainer is None:
            print("Training RUL-based RandomForest for SHAP explainability...")
            columns = (
                ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
                + cls._features
            )
            df = pd.read_csv(data_path, sep=r'\s+', header=None, names=columns)

            # Compute proper RUL: for each engine unit the remaining life at cycle t is
            # (max cycle observed for that unit) - t.
            max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
            df['rul'] = max_cycles - df['time_in_cycles']

            X = df[cls._features]
            y = df['rul']

            cls._model = RandomForestRegressor(
                n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
            )
            cls._model.fit(X, y)
            cls._explainer = shap.TreeExplainer(cls._model)
            print("RUL model trained and SHAP explainer initialised.")

        return cls._explainer, cls._features


DATA_PATH = "data/CMAPSSData/train_FD001.txt"


# ---------------------------------------------------------------------------
# Novel Algorithm: Drift Evidence Score (DES) and Fault Concentration Index (FCI)
# ---------------------------------------------------------------------------

def compute_drift_evidence_score(
    p_values: dict, shap_values: dict, epsilon: float = 1e-300
) -> dict:
    """
    Drift Evidence Score (DES) — joint KS-SHAP sensor signal.

    E_i = -ln(p_i^KS) * |phi_i^SHAP|

    A sensor receives a high DES only when it is BOTH statistically divergent
    (low KS p-value) AND influential on RUL prediction (large |SHAP|).
    This eliminates two categories of false signal:
      - sensors that drift but have no RUL impact (phi ≈ 0 → E ≈ 0)
      - sensors that are important but not currently diverging (p ≈ 1 → E ≈ 0)

    Returns:
        dict: sensor → DES value (non-negative float)
    """
    scores = {}
    for sensor, p_val in p_values.items():
        p_clamped = max(float(p_val), epsilon)
        phi_abs = abs(float(shap_values.get(sensor, 0.0)))
        scores[sensor] = -np.log(p_clamped) * phi_abs
    return scores


def compute_fault_concentration_index(des_scores: dict) -> float:
    """
    Fault Concentration Index (FCI) — Gini coefficient of the DES vector.

    FCI = (2 * sum(i * E_(i)) / (n * sum(E_i))) - (n+1)/n
    where E_(i) are DES values sorted in ascending order.

    FCI ≈ 1.0  →  evidence concentrated in one/few sensors  →  hardware fault
    FCI ≈ 0.0  →  evidence distributed across many sensors  →  operational drift

    If all DES values are zero (no drift detected), returns 0.5 (neutral/unknown).
    """
    values = np.array(sorted(des_scores.values()), dtype=float)
    total = values.sum()
    if total == 0.0:
        return 0.5  # neutral: no evidence either way
    n = len(values)
    index = np.arange(1, n + 1, dtype=float)
    gini = (2.0 * np.dot(index, values)) / (n * total) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))

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
    ),
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
    The underlying model predicts Remaining Useful Life (RUL), so SHAP values indicate which
    sensors are most responsible for the predicted degradation state.
    Higher absolute SHAP values indicate stronger contribution to the anomaly.

    Args:
        sensor_data (dict): Dictionary mapping sensor names (e.g. 'sensor_1') to current float values.

    Returns:
        dict: Sensor names mapped to SHAP importance values, sorted by absolute magnitude.
    """
    explainer, features = RULSHAPExplainer.get_explainer(DATA_PATH)

    input_array = np.array([[sensor_data.get(f, 0.0) for f in features]])
    shap_values = explainer.shap_values(input_array)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = shap_values[0]

    importance = {features[i]: float(shap_values[i]) for i in range(len(features))}
    importance = dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))
    return importance
