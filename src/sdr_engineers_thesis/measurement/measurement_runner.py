"""Utilities to automate linearity and K(f) calibration sweeps.

This script reuses the existing HackRF + SignalProcessor pipeline,
without the GUI. You can call the helpers from your own runner or
extend `GeneratorController` to talk to your RF generator (SCPI, etc.).
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

from sdr_engineers_thesis.core.hackrf_device import HackRFDevice
from sdr_engineers_thesis.core.signal_processor import SignalProcessor
from sdr_engineers_thesis.utils.config import default_config

# HackRF limitation per window
MAX_DEVICE_SPAN_HZ = 20_000_000


# --- Generator control ---------------------------------------------------- #
class GeneratorController:
    """Override these methods to control your generator."""

    def set_freq(self, freq_hz: float) -> None:  # pragma: no cover - hardware stub
        raise NotImplementedError

    def set_power(self, power_dbm: float) -> None:  # pragma: no cover - hardware stub
        raise NotImplementedError

    def rf_on(self) -> None:  # pragma: no cover - hardware stub
        raise NotImplementedError

    def rf_off(self) -> None:  # pragma: no cover - hardware stub
        raise NotImplementedError


class DummyGenerator(GeneratorController):
    """Fallback that only logs actions (useful for dry runs)."""

    def __init__(self) -> None:
        self.freq_hz = None
        self.power_dbm = None
        self.enabled = False

    def set_freq(self, freq_hz: float) -> None:
        self.freq_hz = freq_hz
        print(f"[DummyGen] freq -> {freq_hz/1e6:.3f} MHz")

    def set_power(self, power_dbm: float) -> None:
        self.power_dbm = power_dbm
        print(f"[DummyGen] power -> {power_dbm:.1f} dBm")

    def rf_on(self) -> None:
        self.enabled = True
        print("[DummyGen] RF ON")

    def rf_off(self) -> None:
        self.enabled = False
        print("[DummyGen] RF OFF")


# --- SDR measurement helpers --------------------------------------------- #
@dataclass
class SDRMeasurementSession:
    """Single-window SDR capture for automation (no GUI, no threads)."""

    span_hz: int
    center_hz: float
    fft_size: int
    lna_gain: int
    vga_gain: int

    def __post_init__(self) -> None:
        self.span_hz = int(max(100, min(MAX_DEVICE_SPAN_HZ, self.span_hz)))
        self.fft_size = int(max(128, self.fft_size))

        self.device = HackRFDevice(
            span=self.span_hz,
            center_freq=self.center_hz,
            gain=self.lna_gain,
        )
        self.device.set_vga_gain(self.vga_gain)

        self.processor = SignalProcessor(
            fft_size=self.fft_size,
            num_rows=default_config.processing.num_rows,
            time_plot_samples=default_config.processing.time_plot_samples,
            sample_rate=self.span_hz,
            calibration_file=None,  # raw measurements; loader stays unused here
        )
        self.processor.set_center_freq_khz(int(self.center_hz / 1e3))
        self.processor.set_sample_rate(self.span_hz)

    def _freq_axis(self, center_hz: float) -> np.ndarray:
        freq_resolution = self.span_hz / self.fft_size
        offsets = (np.arange(self.fft_size) - self.fft_size / 2.0) * freq_resolution
        return center_hz + offsets

    def measure_single_bin(self, target_freq_hz: float, avg_frames: int = 3) -> Tuple[float, float]:
        """
        Capture PSD around target_freq_hz and return (psd_dbm, actual_bin_freq_hz).
        """
        self.device.set_span(self.span_hz)
        self.device.set_center_freq(target_freq_hz)
        self.processor.set_center_freq_khz(int(target_freq_hz / 1e3))
        self.processor.set_sample_rate(self.span_hz)

        psd_accum = []
        for _ in range(max(1, int(avg_frames))):
            samples = self.device.read(self.fft_size)
            result = self.processor.process(samples, max_hold_mode=False)
            psd_accum.append(result.raw_psd)

        psd_mean = np.mean(psd_accum, axis=0)
        freq_axis = self._freq_axis(target_freq_hz)
        idx = int(np.argmin(np.abs(freq_axis - target_freq_hz)))
        return float(psd_mean[idx]), float(freq_axis[idx])


# --- Sweep runners ------------------------------------------------------- #
def run_linearity_sweep(
    generator: GeneratorController,
    session: SDRMeasurementSession,
    target_freq_hz: float,
    power_steps_dbm: Sequence[float],
    repeats: int,
    cable_loss_db: float = 0.0,
    settle_s: float = 0.05,
    avg_frames: int = 3,
    outfile: Path | str = Path("measurement/linearity_auto.csv"),
    simulate: bool = False,
    sim_offset_db: float = -3.0,
    sim_noise_db: float = 1.0,
) -> Path:
    """
    Sweep Pin levels at one frequency.

    CSV kolumny: Pin_dBm, Pout_dBm, G_vga_dB, G_lna_dB, CableLoss_dB, Freq_hz, ts.
    """
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _ in range(max(1, int(repeats))):
        for pin_dbm in random.sample(list(power_steps_dbm), len(power_steps_dbm)):
            pgen_dbm = pin_dbm + cable_loss_db  # generator setting
            generator.set_freq(target_freq_hz)
            generator.set_power(pgen_dbm)
            generator.rf_on()
            time.sleep(settle_s)
            if simulate:
                noise = random.gauss(0.0, sim_noise_db)
                psdr_dbm = pin_dbm + sim_offset_db + noise
                bin_freq = target_freq_hz
            else:
                psdr_dbm, bin_freq = session.measure_single_bin(target_freq_hz, avg_frames=avg_frames)
            rows.append(
                {
                    "Pin_dBm": pin_dbm,
                    "Pout_dBm": psdr_dbm,
                    "G_vga_dB": session.vga_gain,
                    "G_lna_dB": session.lna_gain,
                    "CableLoss_dB": cable_loss_db,
                    "Freq_hz": bin_freq,
                    "ts": time.time(),
                }
            )
    generator.rf_off()

    with outfile.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return outfile


def run_kof_sweep(
    generator: GeneratorController,
    session: SDRMeasurementSession,
    freqs_hz: Iterable[float],
    ref_pin_dbm: float,
    repeats: int,
    cable_loss_db: float = 0.0,
    settle_s: float = 0.05,
    avg_frames: int = 3,
    outfile: Path | str = Path("measurement/kof_auto.csv"),
    simulate: bool = False,
    sim_offset_db: float = -3.0,
    sim_noise_db: float = 1.0,
) -> Path:
    """
    Sweep frequency przy stałym Pin.

    CSV kolumny: F_hz, Pin_dBm, Pout_DBm, Kof_db, CableLoss_dB, ts.
    """
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    freq_list = list(freqs_hz)
    if not freq_list:
        raise ValueError("freqs_hz cannot be empty")

    rows = []
    pgen_dbm = ref_pin_dbm + cable_loss_db
    generator.set_power(pgen_dbm)
    for _ in range(max(1, int(repeats))):
        for freq_hz in random.sample(freq_list, len(freq_list)):
            generator.set_freq(freq_hz)
            generator.rf_on()
            time.sleep(settle_s)
            if simulate:
                noise = random.gauss(0.0, sim_noise_db)
                psdr_dbm = ref_pin_dbm + sim_offset_db + noise
                bin_freq = freq_hz
            else:
                psdr_dbm, bin_freq = session.measure_single_bin(freq_hz, avg_frames=avg_frames)
            rows.append(
                {
                    "F_hz": bin_freq,
                    "Pin_dBm": ref_pin_dbm,
                    "Pout_DBm": psdr_dbm,
                    "Kof_db": ref_pin_dbm - psdr_dbm,
                    "CableLoss_dB": cable_loss_db,
                    "ts": time.time(),
                }
            )
    generator.rf_off()

    with outfile.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return outfile


# --- Example usage (dry run) --------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover - manual runner
    gen = DummyGenerator()
    span = min(MAX_DEVICE_SPAN_HZ, int(default_config.hardware.span or 2_000_000))
    session = SDRMeasurementSession(
        span_hz=span,
        center_hz=3.5e9,
        fft_size=default_config.processing.fft_size,
        lna_gain=default_config.hardware.lna_gain,
        vga_gain=default_config.hardware.vga_gain,
    )

    # Quick dry-run to verify wiring (dummy generator prints actions)
    run_linearity_sweep(
        generator=gen,
        session=session,
        target_freq_hz=3.5e9,
        power_steps_dbm=[-65, -60, -55, -50],
        repeats=1,
        outfile=Path("measurement/_linearity_dryrun.csv"),
        simulate=True,
    )

    run_kof_sweep(
        generator=gen,
        session=session,
        freqs_hz=[3.4e9, 3.5e9, 3.6e9],
        ref_pin_dbm=-30,
        repeats=1,
        outfile=Path("measurement/_kof_dryrun.csv"),
        simulate=True,
    )
