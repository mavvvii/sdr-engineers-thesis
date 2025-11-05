from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer, QElapsedTimer
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGridLayout, QWidget, QSlider, 
                            QLabel, QVBoxLayout, QPushButton, QComboBox, QCheckBox,
                            QGroupBox, QSpinBox, QDoubleSpinBox, QProgressBar, QHBoxLayout)
from PyQt6.QtGui import QAction, QKeySequence
import pyqtgraph as pg
import numpy as np

from sdr_engineers_thesis.core.sdr_worker import SDRWorker
from sdr_engineers_thesis.utils.config import default_config


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced PySDR Spectrum Analyzer - HackRF")
        self.setMinimumSize(QSize(1600, 1000))

        self.spectrogram_min = -100.0
        self.spectrogram_max = 0.0
        self.current_fps = 0
        self.channel_power = 0.0

        self.setup_ui()
        self.setup_menu()
        self.setup_worker()

    def setup_ui(self):
        layout = QGridLayout()

        # Worker & thread
        self.sdr_thread = QThread()
        self.worker = SDRWorker()
        self.worker.moveToThread(self.sdr_thread)

        # Time plot
        time_group = QGroupBox("Time Domain")
        time_layout = QVBoxLayout()
        self.time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Sample'})
        self.time_plot.setYRange(-1.1, 1.1)
        self.time_plot.setXRange(0, default_config.processing.time_plot_samples)
        self.time_curve_i = self.time_plot.plot(pen='y', name='I')
        self.time_curve_q = self.time_plot.plot(pen='r', name='Q')
        time_layout.addWidget(self.time_plot)
        time_group.setLayout(time_layout)
        layout.addWidget(time_group, 0, 0)

        # Frequency plot
        freq_group = QGroupBox("Frequency Domain")
        freq_layout = QVBoxLayout()
        self.freq_plot = pg.PlotWidget(labels={'left': 'PSD (dB)', 'bottom': 'Frequency [MHz]'})
        self.freq_curve = self.freq_plot.plot(pen='g')
        self.freq_plot.setYRange(-120, 30)
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Dodaj region do zaznaczania kanału
        self.channel_region = pg.LinearRegionItem([99.7, 100.0])
        self.channel_region.setZValue(10)
        self.channel_region.sigRegionChanged.connect(self.on_channel_region_changed)
        self.freq_plot.addItem(self.channel_region)
        
        freq_layout.addWidget(self.freq_plot)
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group, 1, 0)

        # Waterfall
        waterfall_group = QGroupBox("Waterfall")
        waterfall_layout = QVBoxLayout()
        self.waterfall_plot = pg.PlotWidget(labels={'left': 'Time', 'bottom': 'Frequency [MHz]'})
        self.imageitem = pg.ImageItem(axisOrder='row-major')
        self.waterfall_plot.addItem(self.imageitem)
        
        cmap = pg.colormap.get('viridis')
        self.imageitem.setLookupTable(cmap.getLookupTable(0.0, 1.0, 1024))
        self.imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
        
        self.colorbar = pg.ColorBarItem(colorMap=cmap, values=(self.spectrogram_min, self.spectrogram_max))
        self.colorbar.setImageItem(self.imageitem)
        
        waterfall_layout.addWidget(self.waterfall_plot)
        waterfall_group.setLayout(waterfall_layout)
        layout.addWidget(waterfall_group, 2, 0)

        # Controls panel
        controls_widget = self.create_controls_panel()
        layout.addWidget(controls_widget, 0, 1, 3, 1)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def create_controls_panel(self):
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()

        # Status i wydajność
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Ready")
        self.performance_label = QLabel("FPS: 0.0")
        self.channel_power_label = QLabel("Channel Power: -- dBm")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.performance_label)
        status_layout.addWidget(self.channel_power_label)
        status_group.setLayout(status_layout)
        controls_layout.addWidget(status_group)

        # Ustawienia częstotliwości
        freq_group = QGroupBox("Frequency Settings")
        freq_layout = QVBoxLayout()

        self.freq_label = QLabel(f"Frequency: {default_config.hardware.center_freq/1e6:.3f} MHz")
        freq_layout.addWidget(self.freq_label)
        
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(1_000, 6_000_000)
        self.freq_slider.setValue(int(default_config.hardware.center_freq / 1e3))
        self.freq_slider.sliderMoved.connect(self.on_freq_slider_moved)
        freq_layout.addWidget(self.freq_slider)
        
        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setRange(1.0, 6000.0)
        self.freq_spinbox.setValue(default_config.hardware.center_freq / 1e6)
        self.freq_spinbox.setSuffix(" MHz")
        self.freq_spinbox.valueChanged.connect(self.on_freq_spinbox_changed)
        freq_layout.addWidget(self.freq_spinbox)
        
        freq_group.setLayout(freq_layout)
        controls_layout.addWidget(freq_group)

        # Ustawienia wzmocnienia
        gain_group = QGroupBox("Gain Settings")
        gain_layout = QVBoxLayout()

        self.gain_label = QLabel(f"Gain: {default_config.hardware.gain} dB")
        gain_layout.addWidget(self.gain_label)
        
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, 40)
        self.gain_slider.setValue(default_config.hardware.gain)
        self.gain_slider.sliderMoved.connect(self.on_gain_slider_moved)
        gain_layout.addWidget(self.gain_slider)
        
        gain_group.setLayout(gain_layout)
        controls_layout.addWidget(gain_group)

        # Ustawienia sample rate
        sr_group = QGroupBox("Sample Rate")
        sr_layout = QVBoxLayout()
        
        self.sr_combo = QComboBox()
        self.sr_combo.addItems([f"{x} MHz" for x in default_config.hardware.sample_rates])
        self.sr_combo.setCurrentIndex(3)
        self.sr_combo.currentIndexChanged.connect(self.on_sr_changed)
        sr_layout.addWidget(self.sr_combo)
        
        sr_group.setLayout(sr_layout)
        controls_layout.addWidget(sr_group)

        # Pomiar mocy w kanale
        power_group = QGroupBox("Channel Power Measurement")
        power_layout = QVBoxLayout()
        
        power_range_layout = QHBoxLayout()
        self.channel_start_spin = QDoubleSpinBox()
        self.channel_start_spin.setRange(1.0, 6000.0)
        self.channel_start_spin.setValue(99.7)
        self.channel_start_spin.setSuffix(" MHz")
        self.channel_start_spin.valueChanged.connect(self.on_channel_range_changed)
        
        self.channel_end_spin = QDoubleSpinBox()
        self.channel_end_spin.setRange(1.0, 6000.0)
        self.channel_end_spin.setValue(100.0)
        self.channel_end_spin.setSuffix(" MHz")
        self.channel_end_spin.valueChanged.connect(self.on_channel_range_changed)
        
        power_range_layout.addWidget(QLabel("From:"))
        power_range_layout.addWidget(self.channel_start_spin)
        power_range_layout.addWidget(QLabel("To:"))
        power_range_layout.addWidget(self.channel_end_spin)
        
        self.channel_bw_label = QLabel("Bandwidth: 0.3 MHz")
        self.measured_power_label = QLabel("Power: -- dBm")
        
        power_layout.addLayout(power_range_layout)
        power_layout.addWidget(self.channel_bw_label)
        power_layout.addWidget(self.measured_power_label)
        
        power_group.setLayout(power_layout)
        controls_layout.addWidget(power_group)

        # Ustawienia FFT
        fft_group = QGroupBox("FFT Settings")
        fft_layout = QVBoxLayout()
        
        self.fft_avg_spin = QSpinBox()
        self.fft_avg_spin.setRange(1, 16)
        self.fft_avg_spin.setValue(1)
        self.fft_avg_spin.valueChanged.connect(self.worker.set_fft_averages)
        fft_layout.addWidget(QLabel("FFT Averages:"))
        fft_layout.addWidget(self.fft_avg_spin)
        
        self.waterfall_speed_spin = QSpinBox()
        self.waterfall_speed_spin.setRange(1, 10)
        self.waterfall_speed_spin.setValue(1)
        self.waterfall_speed_spin.valueChanged.connect(self.worker.set_waterfall_speed)
        fft_layout.addWidget(QLabel("Waterfall Speed:"))
        fft_layout.addWidget(self.waterfall_speed_spin)
        
        fft_group.setLayout(fft_layout)
        controls_layout.addWidget(fft_group)

        # Tryby wyświetlania
        mode_group = QGroupBox("Display Modes")
        mode_layout = QVBoxLayout()
        
        self.mode_btn = QPushButton("Mode: Real-time")
        self.mode_btn.clicked.connect(self.toggle_mode)
        mode_layout.addWidget(self.mode_btn)
        
        self.max_hold_btn = QPushButton("Clear Max Hold")
        self.max_hold_btn.clicked.connect(self.clear_max_hold)
        mode_layout.addWidget(self.max_hold_btn)
        
        mode_group.setLayout(mode_layout)
        controls_layout.addWidget(mode_group)

        # Waterfall controls
        wf_group = QGroupBox("Waterfall Controls")
        wf_layout = QVBoxLayout()
        
        self.auto_range_btn = QPushButton("Auto Range (-2σ .. +2σ)")
        self.auto_range_btn.clicked.connect(self.auto_range)
        wf_layout.addWidget(self.auto_range_btn)
        
        self.wf_min_spin = QDoubleSpinBox()
        self.wf_min_spin.setRange(-200, 200)
        self.wf_min_spin.setValue(self.spectrogram_min)
        self.wf_min_spin.valueChanged.connect(self.update_waterfall_levels)
        wf_layout.addWidget(QLabel("Waterfall Min:"))
        wf_layout.addWidget(self.wf_min_spin)
        
        self.wf_max_spin = QDoubleSpinBox()
        self.wf_max_spin.setRange(-200, 200)
        self.wf_max_spin.setValue(self.spectrogram_max)
        self.wf_max_spin.valueChanged.connect(self.update_waterfall_levels)
        wf_layout.addWidget(QLabel("Waterfall Max:"))
        wf_layout.addWidget(self.wf_max_spin)
        
        wf_group.setLayout(wf_layout)
        controls_layout.addWidget(wf_group)

        controls_layout.addStretch()
        controls_widget.setLayout(controls_layout)
        return controls_widget

    def setup_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('File')
        exit_action = QAction('Exit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu('View')
        fullscreen_action = QAction('Fullscreen', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

    def setup_worker(self):
        # Connect signals
        self.worker.time_plot_update.connect(self.on_time_update)
        self.worker.freq_plot_update.connect(self.on_freq_update)
        self.worker.waterfall_plot_update.connect(self.on_waterfall_update)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.performance_update.connect(self.on_performance_update)
        self.worker.channel_power_update.connect(self.on_channel_power_update)
        self.worker.end_of_run.connect(self.schedule_worker)

        # Start thread & worker
        self.sdr_thread.started.connect(self.worker.run)
        self.sdr_thread.start()

    def on_freq_slider_moved(self, value):
        freq_mhz = value / 1e3
        self.freq_label.setText(f"Frequency: {freq_mhz:.3f} MHz")
        self.freq_spinbox.setValue(freq_mhz)
        self.worker.update_freq(value)

    def on_freq_spinbox_changed(self, value):
        value_khz = int(value * 1000)
        self.freq_slider.setValue(value_khz)
        self.worker.update_freq(value_khz)

    def on_gain_slider_moved(self, value):
        self.gain_label.setText(f"Gain: {value} dB")
        self.worker.update_gain(value)

    def on_sr_changed(self, index):
        self.worker.update_sample_rate(index)

    def on_channel_region_changed(self):
        """Aktualizuj zakres kanału gdy region jest przesuwany"""
        start, end = self.channel_region.getRegion()
        self.channel_start_spin.setValue(start)
        self.channel_end_spin.setValue(end)
        self.on_channel_range_changed()

    def on_channel_range_changed(self):
        """Aktualizuj zakres kanału gdy spinboxy są zmieniane"""
        start_freq = self.channel_start_spin.value() * 1e6
        end_freq = self.channel_end_spin.value() * 1e6
        bandwidth = (end_freq - start_freq) / 1e6
        
        self.channel_bw_label.setText(f"Bandwidth: {bandwidth:.3f} MHz")
        self.worker.set_channel_freq_range(start_freq, end_freq)
        
        # Zaktualizuj region na wykresie
        self.channel_region.setRegion([self.channel_start_spin.value(), self.channel_end_spin.value()])

    def on_channel_power_update(self, power):
        """Aktualizuj wyświetlaną moc kanału"""
        self.channel_power = power
        self.measured_power_label.setText(f"Power: {power:.2f} dBm")
        self.channel_power_label.setText(f"Channel Power: {power:.2f} dBm")

    def toggle_mode(self):
        self.worker.max_hold_mode = not self.worker.max_hold_mode
        self.mode_btn.setText("Mode: Max hold" if self.worker.max_hold_mode else "Mode: Real-time")

    def clear_max_hold(self):
        self.worker.PSD_max = -120.0 * np.ones(default_config.processing.fft_size)

    def auto_range(self):
        if hasattr(self.worker, 'spectrogram'):
            sigma = np.std(self.worker.spectrogram)
            mean = np.mean(self.worker.spectrogram)
            self.spectrogram_min = mean - 2*sigma
            self.spectrogram_max = mean + 2*sigma
            self.wf_min_spin.setValue(self.spectrogram_min)
            self.wf_max_spin.setValue(self.spectrogram_max)
            self.imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))

    def update_waterfall_levels(self):
        self.spectrogram_min = self.wf_min_spin.value()
        self.spectrogram_max = self.wf_max_spin.value()
        self.imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
        self.colorbar.setLevels((self.spectrogram_min, self.spectrogram_max))

    def on_time_update(self, samples):
        self.time_curve_i.setData(samples.real)
        self.time_curve_q.setData(samples.imag)

    def on_freq_update(self, psd):
        center_mhz = self.worker.freq_khz / 1e3
        f = np.linspace(center_mhz - (self.worker.sample_rate/2)/1e6,
                        center_mhz + (self.worker.sample_rate/2)/1e6,
                        len(psd))
        self.freq_curve.setData(f, psd)

    def on_waterfall_update(self, spectrogram):
        self.imageitem.setImage(spectrogram, autoLevels=False)

    def on_status_update(self, message):
        self.status_label.setText(message)

    def on_performance_update(self, fps):
        self.current_fps = fps
        self.performance_label.setText(f"FPS: {fps:.1f}")

    def schedule_worker(self):
        QTimer.singleShot(0, self.worker.run)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, event):
        self.worker.stop()
        self.sdr_thread.quit()
        self.sdr_thread.wait(1000)
        event.accept()