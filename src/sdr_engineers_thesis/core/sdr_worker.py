"""Core SDR worker: capture/process samples and emit Qt signals."""

# pylint: disable=R0902, R0904, R0915

import threading
from pathlib import Path
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from sdr_engineers_thesis.core.hackrf_device import HackRFDevice
from sdr_engineers_thesis.core.signal_processor import SignalProcessor
from sdr_engineers_thesis.utils.config import default_config

# Stałe konfiguracyjne
NUM_ROWS = default_config.processing.num_rows
OVERLAP_RATIO = default_config.processing.overlap_ratio
MAX_DEVICE_SPAN_HZ = 20_000_000  # HackRF One limitation


SAMPLE_RATES = default_config.hardware.sample_rates
TIME_PLOT_SAMPLES = default_config.processing.time_plot_samples

# Flagi optymalizacji
USE_OVERLAP = default_config.optimization.use_overlap


@dataclass
class WindowPlan:
    center_hz: float
    span_hz: int
    fft_size: int
    overlap: int
    processor: SignalProcessor



class SDRWorker(QObject):
    """Worker: pobiera próbki z HackRF, przetwarza i emituje sygnały Qt."""

    time_plot_update = pyqtSignal(object)
    freq_plot_update = pyqtSignal(object)
    waterfall_plot_update = pyqtSignal(object)
    status_update = pyqtSignal(str)
    sweep_time_update = pyqtSignal(float)
    channel_power_update = pyqtSignal(float)
    end_of_run = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.lna_gain = default_config.hardware.lna_gain
        self.vga_gain = default_config.hardware.vga_gain
        self.gain = self.vga_gain  # backward compatibility with UI naming
        self.span = default_config.hardware.span
        self.center_freq = default_config.hardware.center_freq / 1e3
        self.fft_size = default_config.processing.fft_size

        self.max_hold_mode = False
        self.running = True
        self.waterfall_speed = 2
        self.waterfall_counter = 0
        self.windows: List[WindowPlan] = []
        self.last_samples: List[Optional[np.ndarray]] = []
        self.waterfall_buffer: Optional[np.ndarray] = None

        self.channel_start_freq = default_config.measurement.channel_start_freq
        self.channel_end_freq = default_config.measurement.channel_end_freq
        self.channel_power = 0.0

        # Opcjonalny plik kalibracyjny dla przeliczeń mocy
        self.calibration_file = str(
            Path(__file__).resolve().parent.parent / "measurement" / "far_measurement.txt"
        )

        self.sample_queue = Queue(maxsize=4)
        self.processing_thread: Optional[threading.Thread] = None
        self.processing_running = True
        self.frame_counter = 0

        self._refresh_window_plan()
        # init device on first window
        first_window = self.windows[0]
        self.device = HackRFDevice(
            first_window.span_hz,
            first_window.center_hz,
            self.lna_gain,
            self.vga_gain,
        )

        self.start_processing_thread()
        self.update_channel_indices()

    def start_processing_thread(self):
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

    def processing_loop(self):
        while self.processing_running:
            try:
                sample_batch = self.sample_queue.get(timeout=0.01)
                if sample_batch:
                    self.process_samples_ultra_fast(sample_batch)
            except Exception:  # pylint: disable=broad-except
                continue

    def _window_fft_size(self, window_span: int) -> int:
        """Scale FFT size per okno, zachowując RBW ~ span/fft_size."""
        scaled = int(max(128, (self.fft_size * window_span) / max(1, self.span)))
        # najbliższa potęga dwójki dla szybkości
        return 1 << (scaled - 1).bit_length()

    def _refresh_window_plan(self):
        """Split span na okna <=20 MHz i odśwież bufory."""
        self.windows = []
        start_freq = self.center_freq * 1e3 - self.span / 2
        remaining = self.span
        current_start = start_freq
        while remaining > 0:
            window_span = int(min(MAX_DEVICE_SPAN_HZ, remaining))
            center = current_start + window_span / 2
            fft_size = self._window_fft_size(window_span)
            print(fft_size)
            overlap = int(fft_size * OVERLAP_RATIO)
            processor = SignalProcessor(
                fft_size,
                NUM_ROWS,
                TIME_PLOT_SAMPLES,
                window_span,
                calibration_file=self.calibration_file,
            )
            processor.set_center_freq_khz(int(center / 1e3))
            processor.set_channel_range(self.channel_start_freq, self.channel_end_freq)
            self.windows.append(WindowPlan(center_hz=center, span_hz=window_span, fft_size=fft_size, overlap=overlap, processor=processor))
            current_start += window_span
            remaining -= window_span

        total_bins = sum(w.fft_size for w in self.windows)
        self.waterfall_buffer = np.full((NUM_ROWS, total_bins), -120.0, dtype=np.float32)
        self.last_samples = [None] * len(self.windows)

    def _apply_window_to_device(self, window: WindowPlan):
        """Ustaw parametry urządzenia dla danego okna."""
        try:
            self.device.set_span(window.span_hz)
            self.device.set_center_freq(window.center_hz)
        except Exception:  # pylint: disable=broad-except
            pass

    def update_channel_indices(self):
        for window in self.windows:
            window.processor.set_center_freq_khz(int(window.center_hz / 1e3))
            window.processor.set_sample_rate(window.span_hz)
            window.processor.set_channel_range(self.channel_start_freq, self.channel_end_freq)

    def set_channel_freq_range(self, start_freq, end_freq):
        self.channel_start_freq = start_freq
        self.channel_end_freq = end_freq
        self.update_channel_indices()

    def set_fft_averages(self, n: int):
        """Set number of FFT averages (kept for UI compatibility)."""
        self.fft_averages = max(1, int(n))

    def update_center_freq(self, val_khz):
        self.center_freq = int(val_khz)
        self._refresh_window_plan()
        if self.device is not None and self.windows:
            self._apply_window_to_device(self.windows[0])
        self.update_channel_indices()
        self.status_update.emit(f"Frequency: {self.center_freq} kHz")

    def update_gain(self, val):
        self.gain = val
        self.vga_gain = val
        if self.device is not None:
            self.device.set_vga_gain(int(val))

    def update_sample_rate(self, index):
        new_sr = int(SAMPLE_RATES[index] * 1e6)
        self.update_span_value(new_sr)

    #DONE
    def update_span_value(self, span_value: int):
        #SPAN RANGE 100-100MHz
        span_value = int(max(100, min(100_000_000, int(span_value))))
        self.span = span_value

        self._refresh_window_plan()
        try:
            if self.device is not None and self.windows:
                self._apply_window_to_device(self.windows[0])
        except Exception:  # pylint: disable=broad-except
            pass
        self.update_channel_indices()
        try:
            self.status_update.emit(f"Sample rate set: {self.span} Hz")
        except Exception:  # pylint: disable=broad-except
            pass

    def set_fft_size(self, fft_size: int):
        fft_size = int(max(128, min(10_000_000, fft_size)))
        if fft_size == self.fft_size:
            return
        self.fft_size = fft_size
        self._refresh_window_plan()
        if self.device is not None and self.windows:
            self._apply_window_to_device(self.windows[0])
        self.update_channel_indices()

    def set_waterfall_speed(self, speed):
        self.waterfall_speed = max(1, speed)

    def stop(self):
        self.running = False
        self.processing_running = False

    def process_samples_ultra_fast(self, samples_batch: List[np.ndarray]):
        combined_psd = []
        combined_time = None
        combined_power = 0.0
        sweep_time_ms = 0.0

        raw_psd_blocks = []
        for window, samples in zip(self.windows, samples_batch):
            result = window.processor.process(samples, self.max_hold_mode)
            combined_psd.append(result.display_psd)
            raw_psd_blocks.append(result.raw_psd)
            combined_power = max(combined_power, result.channel_power)
            sweep_time_ms = max(sweep_time_ms, result.sweep_time_ms)
            if combined_time is None:
                combined_time = result.time_samples

        full_psd = np.concatenate(combined_psd) if combined_psd else np.array([], dtype=np.float32)
        raw_psd_full = np.concatenate(raw_psd_blocks) if raw_psd_blocks else np.array([], dtype=np.float32)

        self.channel_power = combined_power
        self.channel_power_update.emit(self.channel_power)

        self.freq_plot_update.emit((raw_psd_full, full_psd))

        if self.frame_counter % 4 == 0 and combined_time is not None:
            self.time_plot_update.emit(combined_time)

        self.waterfall_counter += 1
        if self.waterfall_counter >= self.waterfall_speed and self.waterfall_buffer is not None:
            self.waterfall_buffer[:-1] = self.waterfall_buffer[1:]
            if full_psd.size > 0:
                self.waterfall_buffer[-1, :] = -120.0
                self.waterfall_buffer[-1, : full_psd.size] = full_psd
            self.waterfall_plot_update.emit(self.waterfall_buffer)
            self.waterfall_counter = 0

        self.sweep_time_update.emit(sweep_time_ms)

        self.frame_counter += 1

    def run(self):
        if not self.running:
            return

        try:
            if self.device is None or self.device.device is None:
                try:
                    self.status_update.emit("SDR device not initialized.")
                except Exception:  # pylint: disable=broad-except
                    pass
                self.end_of_run.emit()
                return

            sample_batch: List[np.ndarray] = []
            for idx, window in enumerate(self.windows):
                self._apply_window_to_device(window)
                overlap = window.overlap if USE_OVERLAP else 0
                read_size = max(1, window.fft_size - overlap)
                new_samples = self.device.read(read_size)
                if USE_OVERLAP and self.last_samples[idx] is not None and overlap > 0:
                    samples = np.concatenate((self.last_samples[idx][-overlap:], new_samples))
                else:
                    samples = new_samples
                self.last_samples[idx] = samples
                sample_batch.append(samples)

            if sample_batch and not self.sample_queue.full():
                self.sample_queue.put(sample_batch)

        except Exception as e:  # pylint: disable=broad-except
            self.status_update.emit(f"Read error: {str(e)}")

        self.end_of_run.emit()
