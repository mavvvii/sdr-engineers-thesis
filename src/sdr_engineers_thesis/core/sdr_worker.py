"""Core SDR worker: capture and process samples from HackRF and emit Qt signals.

This module implements the SDRWorker QObject which runs the capture and
processing pipeline. It also contains optional numba/fastfft accelerated
functions.
"""

#  pylint: disable=R0902, R0904, R0915

import sys
import threading
from queue import Queue
from typing import Optional, Tuple

import numpy as np
from hackrf import HackRF
from PyQt6.QtCore import QElapsedTimer, QObject, pyqtSignal

from sdr_engineers_thesis.utils.config import default_config

# USTAWIENIA DLA MAKSYMALNEJ WYDANOŚCI
FFT_SIZE = default_config.processing.fft_size  # DUŻE FFT
NUM_ROWS = default_config.processing.num_rows  # MNIEJSZY WATERFALL
OVERLAP_RATIO = default_config.processing.overlap_ratio  # MNIEJSZE NAKŁADANIE

CENTER_FREQ = default_config.hardware.center_freq
SAMPLE_RATES = default_config.hardware.sample_rates  # MNIEJ OPCJI
SAMPLE_RATE = default_config.hardware.sample_rate
TIME_PLOT_SAMPLES = default_config.processing.time_plot_samples
GAIN = default_config.hardware.gain

# OPTYMALIZACJE
USE_FASTFFT = default_config.optimization.use_fastfft
USE_NUMBA = default_config.optimization.use_numba
USE_OVERLAP = default_config.optimization.use_overlap

# SPRÓBUJ ZAIMPORTOWAĆ OPTYMALIZACJE
try:
    from sdr_engineers_thesis.cpp_modules import fastfft  # type: ignore[attr-defined]

    USE_FASTFFT = True
    print("✓ Using ULTRA FAST fastfft (FFTW3 with caching)")
except ImportError as e:
    print("✗ fastfft not available:", e)
    USE_FASTFFT = False

# NUMBA FUNCTIONS - defined conditionally
if USE_NUMBA:
    try:
        from numba import jit

        # PRE-KOMPILACJA WSZYSTKICH FUNKCJI
        @jit(nopython=True, fastmath=True, cache=True, nogil=True)
        def numba_fft_shift_abs_log(x, fft_size_val):
            """Return PSD in dB computed with FFT, shifted to center.

            Args:
                x: input complex samples.
                fft_size_val: FFT length used for normalization.

            Returns:
                Array of PSD values in dB.
            """
            fft_data = np.fft.fft(x)
            fft_shifted = np.fft.fftshift(fft_data)
            return 10.0 * np.log10((np.abs(fft_shifted) ** 2) / float(fft_size_val) + 1e-12)

        @jit(nopython=True, fastmath=True, cache=True, nogil=True)
        def apply_window_numba(samples, window):
            """Apply window to samples element-wise.

            This is a tiny helper kept for numba compilation.
            """
            return samples * window

        @jit(nopython=True, fastmath=True, cache=True, nogil=True)
        def calculate_channel_power_numba(psd, start_idx, end_idx):
            """Calculate linear power from PSD between indices [start_idx, end_idx)."""
            total = 0.0
            for i in range(start_idx, end_idx):
                total += 10.0 ** (psd[i] / 10.0)
            return total

        @jit(nopython=True, fastmath=True, cache=True, nogil=True)
        def moving_average_numba(current, previous, alpha):
            """Compute single-step exponential moving average (numba)."""
            return previous * alpha + current * (1.0 - alpha)

        @jit(nopython=True, fastmath=True, cache=True, nogil=True)
        def max_hold_numba(current, previous):
            """Compute element-wise max hold between current and previous arrays."""
            for i, curr_val in enumerate(current):
                if curr_val > previous[i]:
                    previous[i] = curr_val
            return previous

        # PRE-KOMPILACJA
        print("Pre-kompilowanie funkcji numba...")
        test_data = np.random.randn(FFT_SIZE).astype(np.complex128)
        test_window = np.hanning(FFT_SIZE)
        test_psd = np.random.randn(FFT_SIZE).astype(np.float32)
        numba_fft_shift_abs_log(test_data, FFT_SIZE)
        apply_window_numba(test_data, test_window)
        calculate_channel_power_numba(test_psd, 0, 100)
        moving_average_numba(test_psd, test_psd, 0.8)
        max_hold_numba(test_psd, test_psd)
        print("✓ Pre-kompilacja zakończona")

    except ImportError as e:
        print("✗ numba not available:", e)
        USE_NUMBA = False
else:
    # Fallback functions when numba is not available
    def apply_window_numba(samples, window):
        """Apply window to samples element-wise (fallback)."""
        return samples * window

    def calculate_channel_power_numba(psd, start_idx, end_idx):
        """Calculate linear power from PSD between indices [start_idx, end_idx) (fallback)."""
        return np.sum(10.0 ** (psd[start_idx:end_idx] / 10.0))

    def moving_average_numba(current, previous, alpha):
        """Compute single-step exponential moving average (fallback)."""
        return previous * alpha + current * (1.0 - alpha)

    def max_hold_numba(current, previous):
        """Compute element-wise max hold between current and previous arrays (fallback)."""
        return np.maximum(previous, current)


# INICJALIZACJA HACKRF
SDR_DEVICE = None
try:
    SDR_DEVICE = HackRF()
    SDR_DEVICE.sample_rate = SAMPLE_RATE
    SDR_DEVICE.center_freq = CENTER_FREQ
    SDR_DEVICE.set_lna_gain(GAIN)
    SDR_DEVICE.set_vga_gain(16)
    print("✓ HackRF initialized successfully")
except (OSError, IOError, ValueError, AttributeError) as e:
    print(f"✗ HackRF initialization failed: {e}")
    sys.exit(1)


class SDRWorker(QObject):
    """SDR data capture and processing worker (runs in a background thread).

    This QObject exposes Qt signals for UI updates and runs a capture loop that
    enqueues frames for processing.
    """

    time_plot_update = pyqtSignal(object)
    freq_plot_update = pyqtSignal(object)
    waterfall_plot_update = pyqtSignal(object)
    status_update = pyqtSignal(str)
    performance_update = pyqtSignal(float)
    channel_power_update = pyqtSignal(float)
    end_of_run = pyqtSignal()

    def __init__(self):
        """Create SDRWorker and start background processing thread."""
        super().__init__()

        # Core SDR settings
        self.gain = GAIN
        self.sample_rate = SAMPLE_RATE
        self.freq_khz = int(CENTER_FREQ / 1e3)

        # Processing state
        self.max_hold_mode = False
        self.running = True
        self.fft_averages = 1
        self.waterfall_speed = 2  # Domyslnie co 2 klatki
        self.waterfall_counter = 0
        self.overlap_samples = int(FFT_SIZE * OVERLAP_RATIO)
        self.last_samples: Optional[np.ndarray] = None

        # Channel power measurement
        self.channel_start_freq = 99.7e6
        self.channel_end_freq = 100.0e6
        self.channel_power = 0.0
        self.channel_indices: Optional[Tuple[int, int]] = None

        # Sweep settings
        self.sweep_time_seconds = 0.0  # Sweep time in seconds (0 = disabled)

        # Multi-threading
        self.sample_queue = Queue(maxsize=4)  # MNIEJSZA KOLEJKA
        self.processing_thread: Optional[threading.Thread] = None
        self.processing_running = True

        # Performance statistics
        self.processing_times = np.zeros(10, dtype=np.float32)
        self.time_index = 0
        self.frame_counter = 0

        self.psd_avg = None
        self.psd_max = None

        # Initialize data buffers and arrays
        self._initialize_data_arrays()

        self.start_processing_thread()
        self.update_channel_indices()  # Oblicz indeksy kanału na start

    def _initialize_data_arrays(self):
        """Initialize all data arrays to avoid too many instance attributes in __init__."""
        # PRE-ALOKACJA WSZYSTKICH TABLIC
        self.spectrogram = np.full((NUM_ROWS, FFT_SIZE), -120.0, dtype=np.float32)
        self.psd_avg = np.full(FFT_SIZE, -120.0, dtype=np.float32)
        self.psd_max = np.full(FFT_SIZE, -120.0, dtype=np.float32)
        self.window = np.hanning(FFT_SIZE).astype(np.float32)
        self.spectrum_history = np.zeros((2, FFT_SIZE), dtype=np.float32)  # Tylko 2 ramki historii

        # OPTYMALIZACJE
        self.fft_buffer = np.zeros(FFT_SIZE, dtype=np.complex128)
        self.psd_buffer = np.zeros(FFT_SIZE, dtype=np.float32)

    def start_processing_thread(self):
        """Start the background processing thread."""
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

    def processing_loop(self):
        """Thread loop that consumes sample_queue and processes frames."""
        while self.processing_running:
            try:
                samples = self.sample_queue.get(timeout=0.01)  # KRÓTSZY TIMEOUT
                if samples is not None:
                    self.process_samples_ultra_fast(samples)
            except Exception:  # pylint: disable=broad-except
                # timeout or queue empty; continue looping
                continue

    def update_channel_indices(self):
        """Oblicz i cache'uj indeksy kanału - WYWOŁUJ TYLKO GDY ZMIENIASZ CZĘSTOTLIWOŚĆ."""
        freq_resolution = self.sample_rate / FFT_SIZE
        center_freq_hz = self.freq_khz * 1e3

        start_offset = self.channel_start_freq - center_freq_hz
        end_offset = self.channel_end_freq - center_freq_hz

        start_idx = int(FFT_SIZE / 2 + start_offset / freq_resolution)
        end_idx = int(FFT_SIZE / 2 + end_offset / freq_resolution)

        # Zabezpieczenie zakresu
        start_idx = max(0, min(FFT_SIZE - 1, start_idx))
        end_idx = max(0, min(FFT_SIZE - 1, end_idx))

        if start_idx >= end_idx:
            start_idx, end_idx = 0, 0

        self.channel_indices = (start_idx, end_idx)

    def set_channel_freq_range(self, start_freq, end_freq):
        """Set the frequency range (Hz) for channel power measurement.

        Args:
            start_freq: start frequency in Hz.
            end_freq: end frequency in Hz.
        """
        self.channel_start_freq = start_freq
        self.channel_end_freq = end_freq
        self.update_channel_indices()

    def update_freq(self, val_khz):
        """Update the tuned frequency (kHz) and refresh channel indices."""
        self.freq_khz = int(val_khz)
        if SDR_DEVICE is not None:
            SDR_DEVICE.center_freq = int(self.freq_khz * 1e3)
        self.update_channel_indices()  # Odśwież indeksy przy zmianie częstotliwości
        self.status_update.emit(f"Frequency: {self.freq_khz} kHz")

    def update_gain(self, val):
        """Update receiver gain (LNA)."""
        self.gain = val
        if SDR_DEVICE is not None:
            SDR_DEVICE.set_lna_gain(int(val))

    def update_sample_rate(self, index):
        """Update sample rate by index into sample_rates list."""
        new_sr = int(SAMPLE_RATES[index] * 1e6)
        self.sample_rate = new_sr
        if SDR_DEVICE is not None:
            SDR_DEVICE.sample_rate = new_sr
        self.update_channel_indices()  # Odśwież indeksy przy zmianie sample rate

    def update_sample_rate_value(self, sr_hz: int):
        """Set sample rate directly (sr_hz in Hz). Enforces sane bounds and updates SDR."""
        # enforce bounds 100 Hz .. 100 MHz
        sr_hz = int(max(100, min(100_000_000, int(sr_hz))))
        self.sample_rate = sr_hz
        try:
            if SDR_DEVICE is not None:
                SDR_DEVICE.sample_rate = sr_hz
        except Exception:  # pylint: disable=broad-except
            pass
        # recalc channel indices after changing sample rate
        self.update_channel_indices()
        try:
            self.status_update.emit(f"Sample rate set: {self.sample_rate} Hz")
        except Exception:  # pylint: disable=broad-except
            pass

    def set_fft_averages(self, n):
        """Set the number of FFT averages used in smoothing."""
        self.fft_averages = max(1, n)

    def set_waterfall_speed(self, speed):
        """Set waterfall update speed in frames."""
        self.waterfall_speed = max(1, speed)

    def stop(self):
        """Stop processing and exit background threads gracefully."""
        self.running = False
        self.processing_running = False

    def calculate_channel_power_fast(self, psd):
        """Compute channel power (dBm) using cached channel indices.

        Returns 0.0 if indices are invalid.
        """
        if self.channel_indices is None or self.channel_indices[0] >= self.channel_indices[1]:
            return 0.0

        start_idx, end_idx = self.channel_indices

        freq_resolution = self.sample_rate / FFT_SIZE

        if USE_NUMBA:
            power_linear = calculate_channel_power_numba(psd, start_idx, end_idx)
        else:
            # Szybsza wersja bez numba
            power_linear = np.sum(10.0 ** (psd[start_idx:end_idx] / 10.0))

        # return 10.0 * np.log10(power_linear) + 30.0
        power_watts = power_linear * freq_resolution
        power_dbm = 10.0 * np.log10(power_watts / 0.001 + 1e-12)

        return power_dbm

    def process_samples_ultra_fast(self, samples):
        """Process a frame of complex samples and emit plot/update signals."""
        timer = QElapsedTimer()
        timer.start()

        # 1. KONWERSJA I USUWANIE DC
        samples = np.asarray(samples, dtype=np.complex64)
        samples -= np.mean(samples)

        # 2. OKIENKOWANIE
        if USE_NUMBA:
            samples_windowed = apply_window_numba(samples.astype(np.complex128), self.window)
        else:
            samples_windowed = samples.astype(np.complex128) * self.window

        # 3. FFT - NAJSZYBSZA MOŻLIWA ŚCIEŻKA
        if USE_FASTFFT:
            # ULTRA FAST - bezpośrednie wywołanie z cachingiem
            fft_data = fastfft.fft(samples_windowed)
            fft_shifted = np.fft.fftshift(fft_data)
            psd = 10.0 * np.log10((np.abs(fft_shifted) ** 2) / float(FFT_SIZE) + 1e-12)
        elif USE_NUMBA:
            # FAST - numba JIT
            psd = numba_fft_shift_abs_log(samples_windowed, FFT_SIZE)
        else:
            # SLOW - numpy fallback
            fft_data = np.fft.fft(samples_windowed)
            fft_shifted = np.fft.fftshift(fft_data)
            psd = 10.0 * np.log10((np.abs(fft_shifted) ** 2) / float(FFT_SIZE) + 1e-12)

        psd = psd.astype(np.float32)

        # 4. UŚREDNIANIE/MAX HOLD
        if self.max_hold_mode:
            if USE_NUMBA:
                self.psd_max = max_hold_numba(psd, self.psd_max)
                display_psd = self.psd_max
            else:
                self.psd_max = np.maximum(self.psd_max, psd)
                display_psd = self.psd_max
        else:
            if USE_NUMBA:
                self.psd_avg = moving_average_numba(psd, self.psd_avg, 0.7)
                display_psd = self.psd_avg
            else:
                self.psd_avg = self.psd_avg * 0.7 + psd * 0.3
                display_psd = self.psd_avg

        # 5. POMIAR MOCY - ULTRA FAST
        self.channel_power = self.calculate_channel_power_fast(display_psd)
        self.channel_power_update.emit(self.channel_power)

        # 6. EMITUJ WYNIKI
        self.freq_plot_update.emit(display_psd)

        # 7. WYKRES CZASU CO 4 KLATKI
        if self.frame_counter % 4 == 0:
            self.time_plot_update.emit(samples[:TIME_PLOT_SAMPLES])

        # 8. WATERFALL CO waterfal_speed KLATEK
        self.waterfall_counter += 1
        if self.waterfall_counter >= self.waterfall_speed:
            # SZYBSZE WATERFALL - unikaj np.roll
            self.spectrogram[:-1] = self.spectrogram[1:]
            self.spectrogram[-1] = display_psd
            self.waterfall_plot_update.emit(self.spectrogram)
            self.waterfall_counter = 0

        # 9. STATYSTYKI WYDANOŚCI
        processing_time = timer.elapsed()
        self.processing_times[self.time_index] = processing_time
        self.time_index = (self.time_index + 1) % len(self.processing_times)

        avg_time = np.mean(self.processing_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        self.performance_update.emit(fps)

        self.frame_counter += 1

    def run(self):
        """Read samples from the SDR device and enqueue them for processing.

        Handles overlap reads and emits status updates on errors.
        """
        if not self.running:
            return

        try:
            # ODCZYT PRÓBEK Z NAKŁADANIEM
            if SDR_DEVICE is None:
                # Device not initialized; emit status and stop this run.
                try:
                    self.status_update.emit("SDR device not initialized.")
                except Exception:  # pylint: disable=broad-except
                    pass
                self.end_of_run.emit()
                return

            if USE_OVERLAP and self.last_samples is not None:
                new_samples = SDR_DEVICE.read_samples(FFT_SIZE - self.overlap_samples)
                samples = np.concatenate((self.last_samples[-self.overlap_samples :], new_samples))
            else:
                samples = SDR_DEVICE.read_samples(FFT_SIZE)

            self.last_samples = samples

            # DODAJ DO KOLEJKI (NON-BLOCKING)
            if not self.sample_queue.full():
                self.sample_queue.put(samples)

        except Exception as e:  # pylint: disable=broad-except
            self.status_update.emit(f"Read error: {str(e)}")

        self.end_of_run.emit()
