from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer, QElapsedTimer
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGridLayout, QWidget, QSlider, 
                            QLabel, QVBoxLayout, QPushButton, QComboBox, QCheckBox,
                            QGroupBox, QSpinBox, QDoubleSpinBox, QProgressBar, QHBoxLayout)
from PyQt6.QtGui import QAction, QKeySequence
import pyqtgraph as pg
import numpy as np
import time
import signal
import sys
import os
import threading
from queue import Queue

from hackrf import HackRF

from sdr_engineers_thesis.utils.config import default_config

# USTAWIENIA DLA MAKSYMALNEJ WYDANOŚCI
fft_size = default_config.processing.fft_size  # DUŻE FFT
num_rows = default_config.processing.num_rows   # MNIEJSZY WATERFALL
overlap_ratio = default_config.processing.overlap_ratio  # MNIEJSZE NAKŁADANIE

center_freq = default_config.hardware.center_freq
sample_rates = default_config.hardware.sample_rates  # MNIEJ OPCJI
sample_rate = default_config.hardware.sample_rate
time_plot_samples = default_config.processing.time_plot_samples
gain = default_config.hardware.gain

# OPTYMALIZACJE
use_fastfft = default_config.optimization.use_fastfft
use_numba = default_config.optimization.use_numba
use_overlap = default_config.optimization.use_overlap

# SPRÓBUJ ZAIMPORTOWAĆ OPTYMALIZACJE
try:
    from sdr_engineers_thesis.cpp_modules import fastfft
    use_fastfft = True
    print("✓ Using ULTRA FAST fastfft (FFTW3 with caching)")
except Exception as e:
    print("✗ fastfft not available:", e)

try:
    from numba import jit, float32, float64, complex64, complex128
    use_numba = True
    print("✓ Using numba JIT compilation")
    
    # PRE-KOMPILACJA WSZYSTKICH FUNKCJI
    @jit(nopython=True, fastmath=True, cache=True, nogil=True)
    def numba_fft_shift_abs_log(x, fft_size):
        fft_data = np.fft.fft(x)
        fft_shifted = np.fft.fftshift(fft_data)
        return 10.0 * np.log10((np.abs(fft_shifted)**2) / float(fft_size) + 1e-12)
    
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
        for i in range(len(current)):
            if current[i] > previous[i]:
                previous[i] = current[i]
        return previous
    
    # PRE-KOMPILACJA
    print("Pre-kompilowanie funkcji numba...")
    test_data = np.random.randn(fft_size).astype(np.complex128)
    test_window = np.hanning(fft_size)
    test_psd = np.random.randn(fft_size).astype(np.float32)
    numba_fft_shift_abs_log(test_data, fft_size)
    apply_window_numba(test_data, test_window)
    calculate_channel_power_numba(test_psd, 0, 100)
    moving_average_numba(test_psd, test_psd, 0.8)
    max_hold_numba(test_psd, test_psd)
    print("✓ Pre-kompilacja zakończona")
    
except Exception as e:
    print("✗ numba not available:", e)
    use_numba = False

# INICJALIZACJA HACKRF
sdr = None
try:
    sdr = HackRF()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.set_lna_gain(gain)
    sdr.set_vga_gain(16)
    print("✓ HackRF initialized successfully")
except Exception as e:
    print(f"✗ HackRF initialization failed: {e}")
    sys.exit(1)

class SDRWorker(QObject):
    time_plot_update = pyqtSignal(object)
    freq_plot_update = pyqtSignal(object)
    waterfall_plot_update = pyqtSignal(object)
    status_update = pyqtSignal(str)
    performance_update = pyqtSignal(float)
    channel_power_update = pyqtSignal(float)
    end_of_run = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.gain = gain
        self.sample_rate = sample_rate
        self.freq_khz = int(center_freq / 1e3)
        
        # PRE-ALOKACJA WSZYSTKICH TABLIC
        self.spectrogram = np.full((num_rows, fft_size), -120.0, dtype=np.float32)
        self.PSD_avg = np.full(fft_size, -120.0, dtype=np.float32)
        self.PSD_max = np.full(fft_size, -120.0, dtype=np.float32)
        self.window = np.hanning(fft_size).astype(np.float32)
        self.spectrum_history = np.zeros((2, fft_size), dtype=np.float32)  # Tylko 2 ramki historii
        
        self.max_hold_mode = False
        self.running = True
        self.fft_averages = 1
        self.waterfall_speed = 2  # Domyslnie co 2 klatki
        self.waterfall_counter = 0
        self.overlap_samples = int(fft_size * overlap_ratio)
        self.last_samples = None
        
        # POMIAR MOCY
        self.channel_start_freq = 99.7e6
        self.channel_end_freq = 100.0e6
        self.channel_power = 0.0
        self.channel_indices = None  # Cache indeksów kanału
        
        # WIELOWĄTKOWOŚĆ
        self.sample_queue = Queue(maxsize=4)  # MNIEJSZA KOLEJKA
        self.processing_thread = None
        self.processing_running = True
        
        # STATYSTYKI
        self.processing_times = np.zeros(10, dtype=np.float32)
        self.time_index = 0
        self.frame_counter = 0
        
        # OPTYMALIZACJE
        self.fft_buffer = np.zeros(fft_size, dtype=np.complex128)
        self.psd_buffer = np.zeros(fft_size, dtype=np.float32)
        
        self.start_processing_thread()
        self.update_channel_indices()  # Oblicz indeksy kanału na start

    def start_processing_thread(self):
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        
    def processing_loop(self):
        while self.processing_running:
            try:
                samples = self.sample_queue.get(timeout=0.01)  # KRÓTSZY TIMEOUT
                if samples is not None:
                    self.process_samples_ultra_fast(samples)
            except:
                continue

    def update_channel_indices(self):
        """Oblicz i cache'uj indeksy kanału - WYWOŁUJ TYLKO GDY ZMIENIASZ CZĘSTOTLIWOŚĆ"""
        freq_resolution = self.sample_rate / fft_size
        center_freq_hz = self.freq_khz * 1e3
        
        start_offset = (self.channel_start_freq - center_freq_hz)
        end_offset = (self.channel_end_freq - center_freq_hz)
        
        start_idx = int(fft_size/2 + start_offset / freq_resolution)
        end_idx = int(fft_size/2 + end_offset / freq_resolution)
        
        # Zabezpieczenie zakresu
        start_idx = max(0, min(fft_size-1, start_idx))
        end_idx = max(0, min(fft_size-1, end_idx))
        
        if start_idx >= end_idx:
            start_idx, end_idx = 0, 0
            
        self.channel_indices = (start_idx, end_idx)

    def set_channel_freq_range(self, start_freq, end_freq):
        self.channel_start_freq = start_freq
        self.channel_end_freq = end_freq
        self.update_channel_indices()

    def update_freq(self, val_khz):
        self.freq_khz = int(val_khz)
        sdr.center_freq = int(self.freq_khz * 1e3)
        self.update_channel_indices()  # Odśwież indeksy przy zmianie częstotliwości
        self.status_update.emit(f"Frequency: {self.freq_khz} kHz")

    def update_gain(self, val):
        self.gain = val
        sdr.set_lna_gain(int(val))

    def update_sample_rate(self, index):
        new_sr = int(sample_rates[index] * 1e6)
        self.sample_rate = new_sr
        sdr.sample_rate = new_sr
        self.update_channel_indices()  # Odśwież indeksy przy zmianie sample rate

    def set_fft_averages(self, n):
        self.fft_averages = max(1, n)

    def set_waterfall_speed(self, speed):
        self.waterfall_speed = max(1, speed)

    def stop(self):
        self.running = False
        self.processing_running = False

    def calculate_channel_power_fast(self, psd):
        #Do przekminy
        """ULTRA SZYBKI pomiar mocy - używaj cache'owanych indeksów"""
        if self.channel_indices is None or self.channel_indices[0] >= self.channel_indices[1]:
            return 0.0
            
        start_idx, end_idx = self.channel_indices
        
        freq_resolution = self.sample_rate / fft_size

        if use_numba:
            power_linear = calculate_channel_power_numba(psd, start_idx, end_idx)
        else:
            # Szybsza wersja bez numba
            power_linear = np.sum(10.0 ** (psd[start_idx:end_idx] / 10.0))
        
        # return 10.0 * np.log10(power_linear) + 30.0
        power_watts = power_linear * freq_resolution
        power_dbm = 10.0 * np.log10(power_watts / 0.001 + 1e-12)
    
        return power_dbm

    def process_samples_ultra_fast(self, samples):
        """ULTRA SZYBKIE PRZETWARZANIE - MINIMALNE KOPIOWANIE"""
        timer = QElapsedTimer()
        timer.start()
        
        # 1. KONWERSJA I USUWANIE DC
        samples = np.asarray(samples, dtype=np.complex64)
        samples -= np.mean(samples)
        
        # 2. OKIENKOWANIE
        if use_numba:
            samples_windowed = apply_window_numba(samples.astype(np.complex128), self.window)
        else:
            samples_windowed = samples.astype(np.complex128) * self.window
        
        # 3. FFT - NAJSZYBSZA MOŻLIWA ŚCIEŻKA
        if use_fastfft:
            # ULTRA FAST - bezpośrednie wywołanie z cachingiem
            fft_data = fastfft.fft(samples_windowed)
            fft_shifted = np.fft.fftshift(fft_data)
            psd = 10.0 * np.log10((np.abs(fft_shifted)**2) / float(fft_size) + 1e-12)
        elif use_numba:
            # FAST - numba JIT
            psd = numba_fft_shift_abs_log(samples_windowed, fft_size)
        else:
            # SLOW - numpy fallback
            fft_data = np.fft.fft(samples_windowed)
            fft_shifted = np.fft.fftshift(fft_data)
            psd = 10.0 * np.log10((np.abs(fft_shifted)**2) / float(fft_size) + 1e-12)
        
        psd = psd.astype(np.float32)
        
        # 4. UŚREDNIANIE/MAX HOLD
        if self.max_hold_mode:
            if use_numba:
                self.PSD_max = max_hold_numba(psd, self.PSD_max)
                display_psd = self.PSD_max
            else:
                self.PSD_max = np.maximum(self.PSD_max, psd)
                display_psd = self.PSD_max
        else:
            if use_numba:
                self.PSD_avg = moving_average_numba(psd, self.PSD_avg, 0.7)
                display_psd = self.PSD_avg
            else:
                self.PSD_avg = self.PSD_avg * 0.7 + psd * 0.3
                display_psd = self.PSD_avg
        
        # 5. POMIAR MOCY - ULTRA FAST
        self.channel_power = self.calculate_channel_power_fast(display_psd)
        print(len(display_psd))
        self.channel_power_update.emit(self.channel_power)
        
        # 6. EMITUJ WYNIKI
        self.freq_plot_update.emit(display_psd)
        
        # 7. WYKRES CZASU CO 4 KLATKI
        if self.frame_counter % 4 == 0:
            self.time_plot_update.emit(samples[:time_plot_samples])
        
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
        if not self.running:
            return
            
        try:
            # ODCZYT PRÓBEK Z NAKŁADANIEM
            if use_overlap and self.last_samples is not None:
                new_samples = sdr.read_samples(fft_size - self.overlap_samples)
                samples = np.concatenate((self.last_samples[-self.overlap_samples:], new_samples))
            else:
                samples = sdr.read_samples(fft_size)
            
            self.last_samples = samples
            
            # DODAJ DO KOLEJKI (NON-BLOCKING)
            if not self.sample_queue.full():
                self.sample_queue.put(samples)
            
        except Exception as e:
            self.status_update.emit(f"Read error: {str(e)}")
        
        self.end_of_run.emit()