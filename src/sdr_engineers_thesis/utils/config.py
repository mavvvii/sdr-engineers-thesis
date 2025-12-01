"""Application configuration dataclasses and default configuration.

This module defines dataclasses for hardware, processing and GUI
configuration and exposes a `default_config` instance.
"""

from dataclasses import dataclass, field
from typing import List


# ==================== SPRZĘT ====================
@dataclass
class SDRSettings:
    """Hardware settings for the SDR device.

    Attributes:
        center_freq: Default center frequency in Hz.
        sample_rates: List of sample rate options in MS/s.
        span: Currently selected sample rate in Hz.
        gain: Default LNA gain in dB.
        lna_gain: LNA gain value.
        vga_gain: VGA gain value.
    """

    center_freq: float = 2442e6
    sample_rates: List[int] = field(default_factory=lambda: [2, 4, 8, 10])  # POPRAWIONE
    span: int | None = 60e6
    gain: int = 16
    lna_gain: int = 32
    vga_gain: int = 16
    # Test mode: if True, do not attempt to open real HackRF hardware but
    # instead read IQ samples from `test_iq_file` (raw interleaved int8 I,Q)
    test_mode: bool = False
    test_iq_file: str | None = None

    def __post_init__(self):
        """Ensure a default span is selected if not provided."""
        if self.span is None:
            self.span = int(self.sample_rates[2] * 1e6)


# ==================== PRZETWARZANIE ====================
@dataclass  # DODAJ decorator
class ProcessingSettings:
    """Processing-related settings such as FFT size and waterfall rows."""

    fft_size: int = 8192
    num_rows: int = 256
    overlap_ratio: float = 0.1
    time_plot_samples: int = 1024
    fft_averages: int = 1
    waterfall_speed: int = 2


# ==================== RESZTA KLAS (już poprawione) ====================
@dataclass
class OptimizationSettings:
    """Flags enabling optional optimizations (fastfft, numba, overlap)."""

    use_fastfft: bool = True
    use_numba: bool = True
    use_overlap: bool = True


@dataclass
class MeasurementSettings:
    """Settings for channel power measurement ranges."""

    channel_start_freq: float = 2412e6
    channel_end_freq: float = 2480e6

    @property
    def measurement_bw(self) -> float:
        """Return the measurement bandwidth in Hz."""
        return self.channel_end_freq - self.channel_start_freq


@dataclass
class DisplaySettings:
    """UI display default ranges and metrics."""

    freq_plot_range: tuple = (-120, 30)
    time_plot_range: tuple = (-1.1, 1.1)
    waterfall_range: tuple = (-100, 0)
    max_hold_enabled: bool = False
    measured_power: float = 0.0


@dataclass
class GUISettings:
    """Settings for the GUI appearance and defaults."""

    window_size: tuple = (1600, 1000)
    theme: str = "dark"
    show_grid: bool = True
    auto_range_enabled: bool = True


@dataclass
class AppConfig:
    """Root configuration composed of sub-configuration dataclasses."""

    hardware: SDRSettings = field(default_factory=SDRSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    measurement: MeasurementSettings = field(default_factory=MeasurementSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    gui: GUISettings = field(default_factory=GUISettings)


default_config: AppConfig = AppConfig()
