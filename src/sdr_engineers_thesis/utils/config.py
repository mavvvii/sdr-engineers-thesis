from dataclasses import dataclass, field
from typing import List

# ==================== SPRZĘT ====================
@dataclass
class SDRSettings:
    center_freq: float = 99.8e6
    sample_rates: List[int] = field(default_factory=lambda: [2, 4, 8, 10])  # POPRAWIONE
    sample_rate: int = None 
    gain: int = 30
    lna_gain: int = 16
    vga_gain: int = 16

    def __post_init__(self):
        if self.sample_rate is None:
            self.sample_rate = int(self.sample_rates[2] * 1e6)

# ==================== PRZETWARZANIE ====================
@dataclass  # DODAJ decorator
class ProcessingSettings:
    fft_size: int = 8192
    num_rows: int = 256
    overlap_ratio: float = 0.1
    time_plot_samples: int = 1024
    fft_averages: int = 1
    waterfall_speed: int = 2

# ==================== RESZTA KLAS (już poprawione) ====================
@dataclass
class OptimizationSettings:
    use_fastfft: bool = False
    use_numba: bool = False  
    use_overlap: bool = True

@dataclass
class MeasurementSettings:
    channel_start_freq: float = 99.7e6
    channel_end_freq: float = 100.0e6
    
    @property
    def measurement_bw(self) -> float:
        return self.channel_end_freq - self.channel_start_freq

@dataclass
class DisplaySettings:
    freq_plot_range: tuple = (-120, 30)
    time_plot_range: tuple = (-1.1, 1.1)
    waterfall_range: tuple = (-100, 0)
    max_hold_enabled: bool = False
    current_fps: float = 0.0
    measured_power: float = 0.0

@dataclass
class GUISettings:
    window_size: tuple = (1600, 1000)
    theme: str = "dark"
    show_grid: bool = True
    auto_range_enabled: bool = True

@dataclass
class AppConfig:
    hardware: SDRSettings = field(default_factory=SDRSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    measurement: MeasurementSettings = field(default_factory=MeasurementSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    gui: GUISettings = field(default_factory=GUISettings)

default_config = AppConfig()