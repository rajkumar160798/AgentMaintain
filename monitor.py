import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os

class StreamingMonitor:
    def __init__(self, data_path: str, reference_window_size: int = 50, current_window_size: int = 30, p_value_threshold: float = 0.05, step_size: int = 1):
        """
        Simulates streaming data from C-MAPSS dataset and monitors for statistical drift
        using the Kolmogorov-Smirnov test.
        """
        self.data_path = data_path
        self.reference_window_size = reference_window_size
        self.current_window_size = current_window_size
        self.p_value_threshold = p_value_threshold
        self.step_size = step_size
        
        # C-MAPSS column names
        self.columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                       [f'sensor_{i}' for i in range(1, 22)]
        
        self.data = self._load_data()
        self.current_idx = self.reference_window_size + self.current_window_size
        self.max_idx = len(self.data)
        self.current_fault_type = None
        
        # Sensor columns to monitor (all 21)
        self.sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        
    def _load_data(self) -> pd.DataFrame:
        """Loads the C-MAPSS dataset and returns it as a DataFrame."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        df = pd.read_csv(self.data_path, sep=r'\s+', header=None, names=self.columns)
        return df

    def get_next_batch(self):
        """Advances the stream by one step and returns the current reference and current windows."""
        if self.current_idx >= self.max_idx:
            # We reached the end of the stream
            return None, None
            
        ref_start = self.current_idx - self.current_window_size - self.reference_window_size
        ref_end = self.current_idx - self.current_window_size
        reference_window = self.data.iloc[ref_start:ref_end]
        
        curr_start = self.current_idx - self.current_window_size
        curr_end = self.current_idx
        current_window = self.data.iloc[curr_start:curr_end]
        
        self.current_idx += self.step_size
        
        return reference_window, current_window

    def inject_sensor_failure(self, sensor_id: str, severity: float = 5.0):
        """
        Injects a stuck-at-constant sensor failure into the streaming data for the remainder of the stream.
        """
        mean_val = self.data[sensor_id].mean()
        std_val = self.data[sensor_id].std()
        fault_value = mean_val + (severity * std_val) if std_val > 0 else mean_val + 50.0
        
        print(f"--- SENSOR FAILURE INJECTED on {sensor_id} from cycle {self.current_idx} onwards (val: {fault_value:.2f}) ---")
        self.data.loc[self.current_idx:, sensor_id] = fault_value
        self.current_fault_type = "sensor_failure"

    def inject_operational_drift(self, sensor_id: str, drift_rate: float = 0.1):
        """
        Injects a gradual operational drift in a specific sensor signal.
        """
        start = int(self.current_idx)
        drift_len = len(self.data) - start
        drift_values = drift_rate * np.arange(drift_len)
        self.data.loc[start:, sensor_id] = self.data.loc[start:, sensor_id].values + drift_values
        self.current_fault_type = "operational_drift"
        print(f"--- OPERATIONAL DRIFT INJECTED on {sensor_id} starting at cycle {start} with rate {drift_rate:.3f} ---")

    def inject_fault(self, sensor_id: str, severity: float = 5.0):
        """Compatibility wrapper for backwards compatibility with earlier fault injection logic."""
        self.inject_sensor_failure(sensor_id=sensor_id, severity=severity)

    def detect_drift(self):
        """
        Runs KS-test on all 21 sensors between reference and current windows.
        Returns drift status dictionary.
        """
        reference_window, current_window = self.get_next_batch()
        
        if reference_window is None or current_window is None:
            return {"status": "end_of_stream"}
            
        drift_detected = False
        sensor_p_values = {}
        
        for sensor in self.sensor_cols:
            ref_data = reference_window[sensor].values
            curr_data = current_window[sensor].values
            
            if len(np.unique(ref_data)) == 1 and len(np.unique(curr_data)) == 1 and ref_data[0] == curr_data[0]:
                p_val = 1.0
            else:
                _, p_val = ks_2samp(ref_data, curr_data)
                
            sensor_p_values[sensor] = p_val
            
            if p_val < self.p_value_threshold:
                drift_detected = True
                
        return {
            "status": "active",
            "drift_detected": drift_detected,
            "p_values": sensor_p_values,
            "current_idx": self.current_idx,
            "current_data": current_window.iloc[-1].to_dict(),
            "current_fault_type": self.current_fault_type
        }
