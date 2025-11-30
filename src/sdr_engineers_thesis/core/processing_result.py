from dataclasses import dataclass
import numpy as np

@dataclass
class ProcessingResult:
    raw_psd: np.ndarray
    display_psd: np.ndarray
    time_samples: np.ndarray
    channel_power: float
    sweep_time_ms: float
