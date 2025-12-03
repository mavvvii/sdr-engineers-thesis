import numpy as np
from PyQt6.QtCore import QElapsedTimer
from typing import Optional, Tuple

from sdr_engineers_thesis.utils.config import default_config
from sdr_engineers_thesis.core.processing_result import ProcessingResult

CENTER_FREQ = default_config.hardware.center_freq
USE_FASTFFT = default_config.optimization.use_fastfft
USE_NUMBA = default_config.optimization.use_numba
DEFAULT_FFT_SIZE = default_config.processing.fft_size

# fastfft (opcjonalnie)
try:
    from sdr_engineers_thesis.cpp_modules import fastfft  # type: ignore[attr-defined]

    USE_FASTFFT = True
    print("✓ Using ULTRA FAST fastfft (FFTW3 with caching)")
except ImportError as e:
    print("✗ fastfft not available:", e)
    USE_FASTFFT = False

# Funkcje numba (opcjonalnie)
if USE_NUMBA:
    try:
        from numba import jit

        @jit(nopython=True, fastmath=True, cache=True, nogil=True)
        def numba_fft_post(fft_shifted, fft_size_val):
            """
            Część przyspieszana przez numba: |FFT|^2 -> dB.
            Oczekuje już przeshiftowanej tablicy FFT (fft_shifted).
            Brak np.fft.* wewnątrz, więc numba to ładnie łyka.
            """
            return 10.0 * np.log10(
                (np.abs(fft_shifted) ** 2) / float(fft_size_val) + 1e-12
            )

        def numba_fft_shift_abs_log(x, fft_size_val):
            """
            Wrapper: FFT i fftshift w zwykłym NumPy,
            numba przyspiesza tylko post-processing.
            Ta funkcja NIE jest dekorowana @jit.
            """
            fft_data = np.fft.fft(x)
            fft_shifted = np.fft.fftshift(fft_data)
            return numba_fft_post(fft_shifted, fft_size_val)

        @jit(nopython=True, fastmath=True, cache=True, nogil=True)
        def apply_window_numba(samples, window):
            return samples * window

        @jit(nopython=True, fastmath=True, cache=True, nogil=True)
        def calculate_channel_power_numba(psd, start_idx, end_idx):
            total = 0.0
            for i in range(start_idx, end_idx):
                total += 10.0 ** (psd[i] / 10.0)
            return total

        @jit(nopython=True, fastmath=True, cache=True, nogil=True)
        def moving_average_numba(current, previous, alpha):
            return previous * alpha + current * (1.0 - alpha)

        @jit(nopython=True, fastmath=True, cache=True, nogil=True)
        def max_hold_numba(current, previous):
            for i, curr_val in enumerate(current):
                if curr_val > previous[i]:
                    previous[i] = curr_val
            return previous

        # Prekompilacja
        print("Pre-kompilowanie funkcji numba...")
        test_data = np.random.randn(DEFAULT_FFT_SIZE).astype(np.complex128)
        test_window = np.hanning(DEFAULT_FFT_SIZE)
        test_psd = np.random.randn(DEFAULT_FFT_SIZE).astype(np.float32)

        # wywołujemy wrapper – w środku skompiluje się numba_fft_post
        numba_fft_shift_abs_log(test_data, DEFAULT_FFT_SIZE)
        apply_window_numba(test_data, test_window)
        calculate_channel_power_numba(test_psd, 0, 100)
        moving_average_numba(test_psd, test_psd, 0.8)
        max_hold_numba(test_psd, test_psd)
        print("✓ Pre-kompilacja zakończona")
    except ImportError as e:
        print("✗ numba not available:", e)
        USE_NUMBA = False
else:

    def apply_window_numba(samples, window):
        return samples * window

    def calculate_channel_power_numba(psd, start_idx, end_idx):
        return np.sum(10.0 ** (psd[start_idx:end_idx] / 10.0))

    def moving_average_numba(current, previous, alpha):
        return previous * alpha + current * (1.0 - alpha)

    def max_hold_numba(current, previous):
        return np.maximum(previous, current)

class SignalProcessor:
    """Przetwarzanie DSP: FFT, uśrednianie/max-hold, moc kanału."""

    def __init__(
        self,
        fft_size: int,
        num_rows: int,
        time_plot_samples: int,
        sample_rate: int,
        calibration_file: Optional[str] = None,
    ):
        self.fft_size = fft_size
        self.num_rows = num_rows
        self.time_plot_samples = time_plot_samples

        self.spectrogram = np.full((num_rows, fft_size), -120.0, dtype=np.float32)
        self.psd_avg = np.full(fft_size, -120.0, dtype=np.float32)
        self.psd_max = np.full(fft_size, -120.0, dtype=np.float32)
        self.window = np.hanning(fft_size).astype(np.float32)

        self.processing_times = np.zeros(10, dtype=np.float32)
        self.time_index = 0

        self.sample_rate = sample_rate
        self.freq_khz = int(CENTER_FREQ / 1e3)
        self.channel_start_freq = default_config.measurement.channel_start_freq
        self.channel_end_freq = default_config.measurement.channel_end_freq
        self.channel_indices: Optional[Tuple[int, int]] = None
        self.update_channel_indices()

        # Kalibracja (opcjonalna)
        self.calib_freqs: Optional[np.ndarray] = None
        self.calib_offsets: Optional[np.ndarray] = None
        self._load_calibration(calibration_file)

    def update_channel_indices(self) -> None:
        freq_resolution = self.sample_rate / self.fft_size
        center_freq_hz = self.freq_khz * 1e3

        start_offset = self.channel_start_freq - center_freq_hz
        end_offset = self.channel_end_freq - center_freq_hz

        start_idx = int(self.fft_size / 2 + start_offset / freq_resolution)
        end_idx = int(self.fft_size / 2 + end_offset / freq_resolution)

        start_idx = max(0, min(self.fft_size - 1, start_idx))
        end_idx = max(0, min(self.fft_size - 1, end_idx))

        if start_idx >= end_idx:
            start_idx, end_idx = 0, 0

        self.channel_indices = (start_idx, end_idx)

    def _load_calibration(self, filepath: Optional[str]) -> None:
        """
        Wczytuje plik kalibracyjny w formacie (MHz, dBm):
        f_MHz dbm
        100.0 -1.2
        110.0 -0.8

        f_MHz - częstotliwość w MHz,
        dbm   - poprawka [dB], dodawana do channel_power.
        """
        self.calib_freqs = None
        self.calib_offsets = None

        if not filepath:
            return

        freqs: list[float] = []
        offsets: list[float] = []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    if parts[0].lower() in ("f", "freq", "frequency"):
                        continue
                    try:
                        # konwertujemy z MHz na Hz w wewnętrznym przechowywaniu
                        freq_val = float(parts[0]) * 1e6
                        offset_db = float(parts[1])
                    except ValueError:
                        continue
                    freqs.append(freq_val)
                    offsets.append(offset_db)

            if freqs:
                freqs_arr = np.array(freqs, dtype=float)
                offsets_arr = np.array(offsets, dtype=float)
                order = np.argsort(freqs_arr)
                self.calib_freqs = freqs_arr[order]
                self.calib_offsets = offsets_arr[order]
                print(f"✓ Loaded calibration from {filepath} ({len(freqs_arr)} points)")
            else:
                print(f"⚠ Calibration file {filepath} has no valid data")

        except OSError as e:
            print(f"⚠ Could not read calibration file {filepath}: {e}")
            self.calib_freqs = None
            self.calib_offsets = None

    def _get_calibration_offset(self, freq_hz: float) -> float:
        """
        Zwraca poprawkę kalibracyjną [dB] dla danej częstotliwości [Hz].
        Interpolacja liniowa między punktami z pliku; brak pliku -> 0.0.
        """
        if self.calib_freqs is None or self.calib_offsets is None:
            return 0.0
        return float(np.interp(freq_hz, self.calib_freqs, self.calib_offsets))

    def _freq_axis_hz(self) -> np.ndarray:
        """Zwraca wektor częstotliwości (Hz) odpowiadający binom FFT po fftshift."""
        bin_width = self.sample_rate / self.fft_size
        start_freq = self.freq_khz * 1e3 - (self.sample_rate / 2.0)
        return start_freq + bin_width * np.arange(self.fft_size)

    def _smooth_curve(self, curve: np.ndarray, kernel_hz: float = 200_000.0) -> np.ndarray:
        """
        Wygładza krzywą konwolucją z oknem Hanninga o zadanej szerokości w Hz,
        żeby wyeliminować pojedyncze piki po interpolacji.
        """
        bin_width = self.sample_rate / self.fft_size
        kernel_bins = max(1, int(kernel_hz / bin_width))
        if kernel_bins < 3:
            return curve
        if kernel_bins % 2 == 0:
            kernel_bins += 1  # symetryczne okno
        window = np.hanning(kernel_bins).astype(np.float32)
        window /= window.sum()
        return np.convolve(curve, window, mode="same")

    def _apply_calibration_to_psd(self, psd: np.ndarray) -> np.ndarray:
        """
        Interpoluje poprawkę kalibracyjną po całym widmie, żeby uniknąć
        pojedynczych pików w miejscach punktów pomiarowych.
        """
        if self.calib_freqs is None or self.calib_offsets is None:
            return psd

        freq_axis = self._freq_axis_hz()

        # Liniowa interpolacja korekty, z wartościami skrajnymi stałymi poza zakresem.
        offset_curve = np.interp(
            freq_axis,
            self.calib_freqs,
            self.calib_offsets,
            left=self.calib_offsets[0],
            right=self.calib_offsets[-1],
        ).astype(np.float32)

        smoothed_offset = self._smooth_curve(offset_curve)

        # Przywróć wartości dokładnie w punktach kalibracyjnych,
        # żeby nie tracić poziomu na anchorach po wygładzaniu.
        freq_axis_hz = freq_axis
        for freq_val, offset_db in zip(self.calib_freqs, self.calib_offsets):
            idx = int(np.argmin(np.abs(freq_axis_hz - freq_val)))
            smoothed_offset[idx] = float(offset_db)

        return psd + smoothed_offset

    def set_center_freq_khz(self, freq_khz: int) -> None:
        self.freq_khz = freq_khz
        self.update_channel_indices()

    def set_sample_rate(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.update_channel_indices()

    def set_channel_range(self, start_freq: float, end_freq: float) -> None:
        self.channel_start_freq = start_freq
        self.channel_end_freq = end_freq
        self.update_channel_indices()

    def _calculate_channel_power(self, psd: np.ndarray) -> float:
        if self.channel_indices is None or self.channel_indices[0] >= self.channel_indices[1]:
            return 0.0

        start_idx, end_idx = self.channel_indices
        freq_resolution = self.sample_rate / self.fft_size

        if USE_NUMBA:
            power_linear = calculate_channel_power_numba(psd, start_idx, end_idx)
        else:
            power_linear = np.sum(10.0 ** (psd[start_idx:end_idx] / 10.0))

        power_watts = power_linear * freq_resolution
        channel_power_dbm = 10.0 * np.log10(power_watts / 0.001 + 1e-12)
        return channel_power_dbm

    def _record_processing_time(self, processing_time_ms: float) -> float:
        self.processing_times[self.time_index] = processing_time_ms
        self.time_index = (self.time_index + 1) % len(self.processing_times)
        return float(np.mean(self.processing_times))

    #pomocnicza
    def _freq_to_bin(self, freq_hz: float) -> int:
        freq_resolution = self.sample_rate / self.fft_size  # Hz na bin
        center_freq_hz = self.freq_khz * 1e3

        # przesunięcie od częstotliwości środkowej
        offset_hz = freq_hz - center_freq_hz

        # indeks po fftshift: 0..N-1, środek w N/2
        bin_float = self.fft_size / 2 + offset_hz / freq_resolution
        bin_idx = int(round(bin_float))

        # zabezpieczenie na zakres
        bin_idx = max(0, min(self.fft_size - 1, bin_idx))
        return bin_idx

    def _compute_psd(self, samples_windowed: np.ndarray) -> np.ndarray:
        if USE_FASTFFT:
            fft_data = fastfft.fft(samples_windowed)
            fft_shifted = np.fft.fftshift(fft_data)
            psd = 10.0 * np.log10((np.abs(fft_shifted) ** 2) / float(self.fft_size) + 1e-12)

            import datetime
            now = datetime.datetime.now().replace(microsecond=0)
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            target_freq_hz = 3000.021e6
            bin_idx = self._freq_to_bin(target_freq_hz)
            psd_at_freq = psd[bin_idx]

            with open("measurement/measurement_AMPLITUDE_3400_3800_10mhz-step.txt", "a") as f:
                f.write(f'{timestamp};Power_dbm:{max(psd)}\n')

        elif USE_NUMBA:
            psd = numba_fft_shift_abs_log(samples_windowed, self.fft_size)
        else:
            fft_data = np.fft.fft(samples_windowed)
            fft_shifted = np.fft.fftshift(fft_data)
            psd = 10.0 * np.log10((np.abs(fft_shifted) ** 2) / float(self.fft_size) + 1e-12)
        return psd.astype(np.float32)

    def _apply_window(self, samples: np.ndarray) -> np.ndarray:
        if USE_NUMBA:
            return apply_window_numba(samples.astype(np.complex128), self.window)
        return samples.astype(np.complex128) * self.window

    def update_waterfall(self, display_psd: np.ndarray) -> np.ndarray:
        self.spectrogram[:-1] = self.spectrogram[1:]
        self.spectrogram[-1] = display_psd
        return self.spectrogram

    def process(self, samples: np.ndarray, max_hold_mode: bool) -> ProcessingResult:
        timer = QElapsedTimer()
        timer.start()

        samples = np.asarray(samples, dtype=np.complex64)
        samples -= np.mean(samples)

        samples_windowed = self._apply_window(samples)
        psd = self._compute_psd(samples_windowed)
        psd = self._apply_calibration_to_psd(psd)

        # zawsze aktualizujemy max_hold niezależnie od trybu, by warstwa max miała dane
        if USE_NUMBA:
            self.psd_max = max_hold_numba(psd, self.psd_max)
        else:
            self.psd_max = np.maximum(self.psd_max, psd)

        if max_hold_mode:
            display_psd = self.psd_max
        else:
            if USE_NUMBA:
                self.psd_avg = moving_average_numba(psd, self.psd_avg, 0.7)
            else:
                self.psd_avg = self.psd_avg * 0.7 + psd * 0.3
            display_psd = self.psd_avg

        channel_power = self._calculate_channel_power(display_psd)

        processing_time = timer.elapsed()
        sweep_time_ms = self._record_processing_time(processing_time)

        time_samples = samples[: self.time_plot_samples]

        return ProcessingResult(
            raw_psd=psd,
            display_psd=display_psd,
            time_samples=time_samples,
            channel_power=channel_power,
            sweep_time_ms=sweep_time_ms,
        )
