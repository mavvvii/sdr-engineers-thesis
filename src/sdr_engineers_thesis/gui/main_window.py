"""Main window UI for the PySDR spectrum analyzer.

This module provides the `MainWindow` QMainWindow which hosts plots and
controls for interacting with the SDR worker.
"""

#  pylint: disable=R0902, R0904, R0915
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSize, Qt, QThread, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sdr_engineers_thesis.core.sdr_worker import SDRWorker
from sdr_engineers_thesis.utils.config import default_config


class MainWindow(QMainWindow):
    """Application main window hosting plots and controls."""

    def __init__(self):
        """Initialize UI, menu and background worker thread."""
        super().__init__()
        self.setWindowTitle("Advanced PySDR Spectrum Analyzer - HackRF")
        self.setMinimumSize(QSize(1600, 1000))

        # Initialize core attributes
        self.spectrogram_min = -100.0
        self.spectrogram_max = 0.0
        self.current_fps = 0
        self.channel_power = 0.0

        self.time_plot = None
        self.time_curve_i = None
        self.time_curve_q = None
        self.freq_plot = None
        self.freq_curve = None
        self.channel_region = None
        self.waterfall_plot = None
        self.imageitem = None
        self.colorbar = None

        # Initialize all UI components in __init__
        self._initialize_ui_components()

        self.setup_ui()
        self.setup_menu()
        self.setup_worker()

    def _initialize_ui_components(self):
        """Initialize all UI components to avoid attribute-defined-outside-init."""
        # Plot items
        self.time_plot = None
        self.time_curve_i = None
        self.time_curve_q = None
        self.freq_plot = None
        self.freq_curve = None
        self.channel_region = None
        self.waterfall_plot = None
        self.imageitem = None
        self.colorbar = None

        # Status labels
        self.status_label = QLabel("Ready")
        self.performance_label = QLabel("FPS: 0.0")
        self.channel_power_label = QLabel("Channel Power: -- dBm")

        # Frequency controls
        self.freq_label = QLabel(f"Frequency: {default_config.hardware.center_freq / 1e6:.3f} MHz")
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_spinbox = QDoubleSpinBox()

        # Gain controls
        self.gain_label = QLabel(f"Gain: {default_config.hardware.gain} dB")
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)

        # SPAN/RBW/Sweep controls
        self.span_spin = QDoubleSpinBox()
        self.span_unit = QComboBox()
        self.rbw_spin = QDoubleSpinBox()
        self.rbw_unit = QComboBox()
        self.sweep_spin = QDoubleSpinBox()
        self.sweep_unit = QComboBox()

        # Channel power measurement
        self.channel_start_spin = QDoubleSpinBox()
        self.channel_end_spin = QDoubleSpinBox()
        self.channel_bw_label = QLabel("Bandwidth: 0.3 MHz")
        self.measured_power_label = QLabel("Power: -- dBm")

        # FFT settings
        self.fft_avg_spin = QSpinBox()
        self.waterfall_speed_spin = QSpinBox()

        # Display modes
        self.mode_btn = QPushButton("Mode: Real-time")
        self.max_hold_btn = QPushButton("Clear Max Hold")

        # Waterfall controls
        self.auto_range_btn = QPushButton("Auto Range (-2σ .. +2σ)")
        self.wf_min_spin = QDoubleSpinBox()
        self.wf_max_spin = QDoubleSpinBox()

        # Thread and worker
        self.sdr_thread = QThread()
        self.worker = SDRWorker()

    def setup_ui(self):
        """Create and arrange all UI widgets and plots."""
        layout = QGridLayout()

        # Worker setup
        self.worker.moveToThread(self.sdr_thread)

        # Create plot groups
        self._create_time_plot(layout)
        self._create_frequency_plot(layout)
        self._create_waterfall_plot(layout)

        # Controls panel
        controls_widget = self.create_controls_panel()
        layout.addWidget(controls_widget, 0, 1, 3, 1)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _create_time_plot(self, layout):
        """Create time domain plot."""
        time_group = QGroupBox("Time Domain")
        time_layout = QVBoxLayout()
        self.time_plot = pg.PlotWidget(labels={"left": "Amplitude", "bottom": "Sample"})
        self.time_plot.setYRange(-1.1, 1.1)
        self.time_plot.setXRange(0, default_config.processing.time_plot_samples)
        self.time_curve_i = self.time_plot.plot(pen="y", name="I")
        self.time_curve_q = self.time_plot.plot(pen="r", name="Q")
        time_layout.addWidget(self.time_plot)
        time_group.setLayout(time_layout)
        layout.addWidget(time_group, 0, 0)

    def _create_frequency_plot(self, layout):
        """Create frequency domain plot."""
        freq_group = QGroupBox("Frequency Domain")
        freq_layout = QVBoxLayout()
        self.freq_plot = pg.PlotWidget(labels={"left": "PSD (dB)", "bottom": "Frequency [MHz]"})
        self.freq_curve = self.freq_plot.plot(pen="g")
        self.freq_plot.setYRange(-120, 30)
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)

        # Add channel region
        self.channel_region = pg.LinearRegionItem([99.7, 100.0])
        self.channel_region.setZValue(10)
        self.channel_region.sigRegionChanged.connect(self.on_channel_region_changed)
        self.freq_plot.addItem(self.channel_region)

        freq_layout.addWidget(self.freq_plot)
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group, 1, 0)

    def _create_waterfall_plot(self, layout):
        """Create waterfall plot."""
        waterfall_group = QGroupBox("Waterfall")
        waterfall_layout = QVBoxLayout()
        self.waterfall_plot = pg.PlotWidget(labels={"left": "Time", "bottom": "Frequency [MHz]"})
        self.imageitem = pg.ImageItem(axisOrder="row-major")
        self.waterfall_plot.addItem(self.imageitem)

        cmap = pg.colormap.get("viridis")
        self.imageitem.setLookupTable(cmap.getLookupTable(0.0, 1.0, 1024))
        self.imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))

        self.colorbar = pg.ColorBarItem(
            colorMap=cmap, values=(self.spectrogram_min, self.spectrogram_max)
        )
        self.colorbar.setImageItem(self.imageitem)

        waterfall_layout.addWidget(self.waterfall_plot)
        waterfall_group.setLayout(waterfall_layout)
        layout.addWidget(waterfall_group, 2, 0)

    def create_controls_panel(self):
        """Create the right-hand controls panel widget and return it."""
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()

        # Add control groups
        self._create_status_group(controls_layout)
        self._create_frequency_group(controls_layout)
        self._create_gain_group(controls_layout)
        self._create_span_group(controls_layout)
        self._create_power_group(controls_layout)
        self._create_fft_group(controls_layout)
        self._create_mode_group(controls_layout)
        self._create_waterfall_group(controls_layout)

        controls_layout.addStretch()
        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_status_group(self, parent_layout):
        """Create status group."""
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.performance_label)
        status_layout.addWidget(self.channel_power_label)
        status_group.setLayout(status_layout)
        parent_layout.addWidget(status_group)

    def _create_frequency_group(self, parent_layout):
        """Create frequency settings group."""
        freq_group = QGroupBox("Frequency Settings")
        freq_layout = QVBoxLayout()

        freq_layout.addWidget(self.freq_label)

        self.freq_slider.setRange(1_000, 6_000_000)
        self.freq_slider.setValue(int(default_config.hardware.center_freq / 1e3))
        self.freq_slider.sliderMoved.connect(self.on_freq_slider_moved)
        freq_layout.addWidget(self.freq_slider)

        self.freq_spinbox.setRange(1.0, 6000.0)
        self.freq_spinbox.setValue(default_config.hardware.center_freq / 1e6)
        self.freq_spinbox.setSuffix(" MHz")
        self.freq_spinbox.valueChanged.connect(self.on_freq_spinbox_changed)
        freq_layout.addWidget(self.freq_spinbox)

        freq_group.setLayout(freq_layout)
        parent_layout.addWidget(freq_group)

    def _create_gain_group(self, parent_layout):
        """Create gain settings group."""
        gain_group = QGroupBox("Gain Settings")
        gain_layout = QVBoxLayout()

        gain_layout.addWidget(self.gain_label)

        self.gain_slider.setRange(0, 40)
        self.gain_slider.setValue(default_config.hardware.gain)
        self.gain_slider.sliderMoved.connect(self.on_gain_slider_moved)
        gain_layout.addWidget(self.gain_slider)

        gain_group.setLayout(gain_layout)
        parent_layout.addWidget(gain_group)

    def _create_span_group(self, parent_layout):
        """Create SPAN/RBW/Sweep group."""
        span_group = QGroupBox("SPAN / RBW / Sweep")
        span_layout = QVBoxLayout()

        # SPAN controls
        span_hbox = QHBoxLayout()
        self.span_spin.setRange(0.0001, 100000.0)
        self.span_spin.setDecimals(6)
        self.span_spin.setValue(default_config.hardware.sample_rate / 1e6)
        self.span_unit.addItems(["Hz", "kHz", "MHz"])
        self.span_unit.setCurrentText("MHz")
        self.span_spin.valueChanged.connect(self.on_span_changed)
        self.span_unit.currentTextChanged.connect(self.on_span_changed)
        span_hbox.addWidget(QLabel("SPAN:"))
        span_hbox.addWidget(self.span_spin)
        span_hbox.addWidget(self.span_unit)
        span_layout.addLayout(span_hbox)

        # RBW controls
        rbw_hbox = QHBoxLayout()
        self.rbw_spin.setRange(0.000001, 100000.0)
        self.rbw_spin.setDecimals(6)
        init_rbw_hz = default_config.hardware.sample_rate / default_config.processing.fft_size
        self.rbw_spin.setValue(init_rbw_hz / 1e3)
        self.rbw_unit.addItems(["Hz", "kHz", "MHz"])
        self.rbw_unit.setCurrentText("kHz")
        self.rbw_spin.valueChanged.connect(self.on_rbw_changed)
        self.rbw_unit.currentTextChanged.connect(self.on_rbw_changed)
        rbw_hbox.addWidget(QLabel("RBW:"))
        rbw_hbox.addWidget(self.rbw_spin)
        rbw_hbox.addWidget(self.rbw_unit)
        span_layout.addLayout(rbw_hbox)

        # Sweep controls
        sweep_hbox = QHBoxLayout()
        self.sweep_spin.setRange(0.0, 3600.0)
        self.sweep_spin.setDecimals(3)
        self.sweep_spin.setValue(0.0)
        self.sweep_unit.addItems(["ms", "s", "min"])
        self.sweep_unit.setCurrentText("s")
        self.sweep_spin.valueChanged.connect(self.on_sweep_time_changed)
        self.sweep_unit.currentTextChanged.connect(self.on_sweep_time_changed)
        sweep_hbox.addWidget(QLabel("Sweep Time:"))
        sweep_hbox.addWidget(self.sweep_spin)
        sweep_hbox.addWidget(self.sweep_unit)
        span_layout.addLayout(sweep_hbox)

        span_group.setLayout(span_layout)
        parent_layout.addWidget(span_group)

    def _create_power_group(self, parent_layout):
        """Create channel power measurement group."""
        power_group = QGroupBox("Channel Power Measurement")
        power_layout = QVBoxLayout()

        power_range_layout = QHBoxLayout()
        self.channel_start_spin.setRange(1.0, 6000.0)
        self.channel_start_spin.setValue(99.7)
        self.channel_start_spin.setSuffix(" MHz")
        self.channel_start_spin.valueChanged.connect(self.on_channel_range_changed)

        self.channel_end_spin.setRange(1.0, 6000.0)
        self.channel_end_spin.setValue(100.0)
        self.channel_end_spin.setSuffix(" MHz")
        self.channel_end_spin.valueChanged.connect(self.on_channel_range_changed)

        power_range_layout.addWidget(QLabel("From:"))
        power_range_layout.addWidget(self.channel_start_spin)
        power_range_layout.addWidget(QLabel("To:"))
        power_range_layout.addWidget(self.channel_end_spin)

        power_layout.addLayout(power_range_layout)
        power_layout.addWidget(self.channel_bw_label)
        power_layout.addWidget(self.measured_power_label)

        power_group.setLayout(power_layout)
        parent_layout.addWidget(power_group)

    def _create_fft_group(self, parent_layout):
        """Create FFT settings group."""
        fft_group = QGroupBox("FFT Settings")
        fft_layout = QVBoxLayout()

        self.fft_avg_spin.setRange(1, 16)
        self.fft_avg_spin.setValue(1)
        self.fft_avg_spin.valueChanged.connect(self.worker.set_fft_averages)
        fft_layout.addWidget(QLabel("FFT Averages:"))
        fft_layout.addWidget(self.fft_avg_spin)

        self.waterfall_speed_spin.setRange(1, 10)
        self.waterfall_speed_spin.setValue(1)
        self.waterfall_speed_spin.valueChanged.connect(self.worker.set_waterfall_speed)
        fft_layout.addWidget(QLabel("Waterfall Speed:"))
        fft_layout.addWidget(self.waterfall_speed_spin)

        fft_group.setLayout(fft_layout)
        parent_layout.addWidget(fft_group)

    def _create_mode_group(self, parent_layout):
        """Create display modes group."""
        mode_group = QGroupBox("Display Modes")
        mode_layout = QVBoxLayout()

        self.mode_btn.clicked.connect(self.toggle_mode)
        mode_layout.addWidget(self.mode_btn)

        self.max_hold_btn.clicked.connect(self.clear_max_hold)
        mode_layout.addWidget(self.max_hold_btn)

        mode_group.setLayout(mode_layout)
        parent_layout.addWidget(mode_group)

    def _create_waterfall_group(self, parent_layout):
        """Create waterfall controls group."""
        wf_group = QGroupBox("Waterfall Controls")
        wf_layout = QVBoxLayout()

        self.auto_range_btn.clicked.connect(self.auto_range)
        wf_layout.addWidget(self.auto_range_btn)

        self.wf_min_spin.setRange(-200, 200)
        self.wf_min_spin.setValue(self.spectrogram_min)
        self.wf_min_spin.valueChanged.connect(self.update_waterfall_levels)
        wf_layout.addWidget(QLabel("Waterfall Min:"))
        wf_layout.addWidget(self.wf_min_spin)

        self.wf_max_spin.setRange(-200, 200)
        self.wf_max_spin.setValue(self.spectrogram_max)
        self.wf_max_spin.valueChanged.connect(self.update_waterfall_levels)
        wf_layout.addWidget(QLabel("Waterfall Max:"))
        wf_layout.addWidget(self.wf_max_spin)

        wf_group.setLayout(wf_layout)
        parent_layout.addWidget(wf_group)

    def setup_menu(self):
        """Create application menu bar with File/View actions."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("View")
        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

    def setup_worker(self):
        """Wire up worker signals and start worker thread."""
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
        """Handle frequency slider movement and update UI/worker."""
        freq_mhz = value / 1e3
        self.freq_label.setText(f"Frequency: {freq_mhz:.3f} MHz")
        self.freq_spinbox.setValue(freq_mhz)
        self.worker.update_freq(value)

    def on_freq_spinbox_changed(self, value):
        """Handle frequency spinbox change and sync slider/worker."""
        value_khz = int(value * 1000)
        self.freq_slider.setValue(value_khz)
        self.worker.update_freq(value_khz)

    def on_gain_slider_moved(self, value):
        """Handle gain slider changes and update worker."""
        self.gain_label.setText(f"Gain: {value} dB")
        self.worker.update_gain(value)

    def _unit_multiplier(self, unit_text: str) -> float:
        """Get multiplier for frequency units."""
        if unit_text == "Hz":
            return 1.0
        if unit_text == "kHz":
            return 1e3
        if unit_text == "MHz":
            return 1e6
        return 1.0

    def on_span_changed(self, _=None):
        """Compute displayed span in Hz, clamp to valid range and update worker."""
        # compute span in Hz from spin + unit
        span_val = self.span_spin.value()
        span_hz = span_val * self._unit_multiplier(self.span_unit.currentText())
        # clamp to [100 Hz, 100 MHz]
        span_hz = max(100.0, min(100_000_000.0, span_hz))
        # update worker (sample_rate = span)
        try:
            self.worker.update_sample_rate_value(int(span_hz))
        except Exception:  # pylint: disable=broad-except
            pass
        # update RBW display
        self._update_rbw_display(span_hz)

    def _update_rbw_display(self, span_hz):
        """Update RBW display based on span."""
        rbw_hz = span_hz / default_config.processing.fft_size
        # choose a friendly unit for rbw display
        if rbw_hz >= 1e6:
            self.rbw_unit.setCurrentText("MHz")
            self.rbw_spin.setValue(rbw_hz / 1e6)
        elif rbw_hz >= 1e3:
            self.rbw_unit.setCurrentText("kHz")
            self.rbw_spin.setValue(rbw_hz / 1e3)
        else:
            self.rbw_unit.setCurrentText("Hz")
            self.rbw_spin.setValue(rbw_hz)

    def on_rbw_changed(self, _=None):
        """Handle RBW control changes and update sample rate/span accordingly."""
        rbw_val = self.rbw_spin.value()
        rbw_hz = rbw_val * self._unit_multiplier(self.rbw_unit.currentText())
        # desired sample rate = rbw * fft_size
        desired_sr = rbw_hz * default_config.processing.fft_size
        # clamp to [100 Hz, 100 MHz]
        desired_sr = max(100.0, min(100_000_000.0, desired_sr))
        try:
            self.worker.update_sample_rate_value(int(desired_sr))
        except Exception:  # pylint: disable=broad-except
            pass
        # update SPAN display
        self._update_span_display(desired_sr)

    def _update_span_display(self, desired_sr):
        """Update SPAN display based on desired sample rate."""
        if desired_sr >= 1e6:
            self.span_unit.setCurrentText("MHz")
            self.span_spin.setValue(desired_sr / 1e6)
        elif desired_sr >= 1e3:
            self.span_unit.setCurrentText("kHz")
            self.span_spin.setValue(desired_sr / 1e3)
        else:
            self.span_unit.setCurrentText("Hz")
            self.span_spin.setValue(desired_sr)

    def on_sweep_time_changed(self, _=None):
        """Convert sweep time UI inputs into seconds and store on the worker."""
        val = self.sweep_spin.value()
        unit = self.sweep_unit.currentText()
        if unit == "ms":
            seconds = val / 1000.0
        elif unit == "s":
            seconds = val
        elif unit == "min":
            seconds = val * 60.0
        else:
            seconds = val
        # store on worker for later use
        try:
            self.worker.sweep_time_seconds = float(seconds)
        except Exception:  # pylint: disable=broad-except
            pass

    def on_channel_region_changed(self):
        """Update channel range when region is moved."""
        start, end = self.channel_region.getRegion()
        self.channel_start_spin.setValue(start)
        self.channel_end_spin.setValue(end)
        self.on_channel_range_changed()

    def on_channel_range_changed(self):
        """Update channel range when spinboxes are changed."""
        start_freq = self.channel_start_spin.value() * 1e6
        end_freq = self.channel_end_spin.value() * 1e6
        bandwidth = (end_freq - start_freq) / 1e6

        self.channel_bw_label.setText(f"Bandwidth: {bandwidth:.3f} MHz")
        self.worker.set_channel_freq_range(start_freq, end_freq)

        # Update region on plot
        self.channel_region.setRegion(
            [self.channel_start_spin.value(), self.channel_end_spin.value()]
        )

    def on_channel_power_update(self, power):
        """Update displayed channel power."""
        self.channel_power = power
        self.measured_power_label.setText(f"Power: {power:.2f} dBm")
        self.channel_power_label.setText(f"Channel Power: {power:.2f} dBm")

    def toggle_mode(self):
        """Toggle between real-time and max-hold display modes."""
        self.worker.max_hold_mode = not self.worker.max_hold_mode
        self.mode_btn.setText("Mode: Max hold" if self.worker.max_hold_mode else "Mode: Real-time")

    def clear_max_hold(self):
        """Clear max-hold buffer to minimum values."""
        self.worker.psd_max = -120.0 * np.ones(default_config.processing.fft_size)

    def auto_range(self):
        """Auto-range waterfall levels based on spectrogram statistics."""
        if hasattr(self.worker, "spectrogram"):
            sigma = np.std(self.worker.spectrogram)
            mean = np.mean(self.worker.spectrogram)
            self.spectrogram_min = mean - 2 * sigma
            self.spectrogram_max = mean + 2 * sigma
            self.wf_min_spin.setValue(self.spectrogram_min)
            self.wf_max_spin.setValue(self.spectrogram_max)
            self.imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))

    def update_waterfall_levels(self):
        """Apply user-specified waterfall min/max levels to the display."""
        self.spectrogram_min = self.wf_min_spin.value()
        self.spectrogram_max = self.wf_max_spin.value()
        self.imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
        self.colorbar.setLevels((self.spectrogram_min, self.spectrogram_max))

    def on_time_update(self, samples):
        """Receive time-domain samples and update traces."""
        self.time_curve_i.setData(samples.real)
        self.time_curve_q.setData(samples.imag)

    def on_freq_update(self, psd):
        """Update frequency-domain plot with new PSD data."""
        center_mhz = self.worker.freq_khz / 1e3
        f = np.linspace(
            center_mhz - (self.worker.sample_rate / 2) / 1e6,
            center_mhz + (self.worker.sample_rate / 2) / 1e6,
            len(psd),
        )
        self.freq_curve.setData(f, psd)

    def on_waterfall_update(self, spectrogram):
        """Update waterfall image with new spectrogram data."""
        self.imageitem.setImage(spectrogram, autoLevels=False)

    def on_status_update(self, message):
        """Display status messages from the worker in the UI."""
        self.status_label.setText(message)

    def on_performance_update(self, fps):
        """Update FPS display in the UI."""
        self.current_fps = fps
        self.performance_label.setText(f"FPS: {fps:.1f}")

    def schedule_worker(self):
        """Schedule worker.run to be called by the Qt event loop."""
        QTimer.singleShot(0, self.worker.run)

    def toggle_fullscreen(self):
        """Toggle full screen mode on/off."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def close_event(self, event):
        """Handle window close: stop worker and wait for thread shutdown."""
        self.worker.stop()
        self.sdr_thread.quit()
        self.sdr_thread.wait(1000)
        event.accept()

    # Qt override - must keep original name
    def closeEvent(self, event):  # pylint: disable=invalid-name
        """Qt close event handler."""
        self.close_event(event)
