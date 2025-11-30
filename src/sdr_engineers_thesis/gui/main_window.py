"""Main window UI for the PySDR spectrum analyzer.

This module provides the `MainWindow` QMainWindow which hosts plots and
controls for interacting with the SDR worker.
"""

#  pylint: disable=R0902, R0904, R0915
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSize, Qt, QThread, QTimer, QRectF
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QToolButton,
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

import math

from sdr_engineers_thesis.core.sdr_worker import SDRWorker
from sdr_engineers_thesis.utils.config import default_config


class MainWindow(QMainWindow):
    """Application main window hosting plots and controls."""

    RBW_FFT_MAX = 262_144  # safety cap for huge FFT sizes to avoid OOM
    MAX_DEVICE_SPAN_HZ = 20_000_000  # HackRF One limitation per capture

    def __init__(self):
        """Initialize UI, menu and background worker thread."""
        super().__init__()
        self.setWindowTitle("Advanced PySDR Spectrum Analyzer - HackRF")
        self.setMinimumSize(QSize(1600, 1000))

        # Initialize core attributes
        self.spectrogram_min = -100.0
        self.spectrogram_max = 0.0
        self.current_sweep_ms = 0.0
        self.channel_power = 0.0

        self.time_plot = None
        self.time_curve_i = None
        self.time_curve_q = None
        self.freq_plot = None
        self.freq_curve_current = None
        self.freq_curve_avg = None
        self.freq_curve_max = None
        self.marker_line_v = None
        self.marker_line_h = None
        self.marker_label = None
        self.marker_freq_mhz = None
        self.marker_power_dbm = None
        self.last_freq_axis = None
        self.last_psd_current = None
        self.last_psd_avg = None
        self.last_psd_max = None
        self.channel_region = None
        self.waterfall_plot = None
        self.imageitem = None
        self.colorbar = None
        self.last_waterfall_data = None
        self.wf_cmap_combo = None

        # Initialize all UI components in __init__
        self._initialize_ui_components()

        self.setup_ui()
        self.setup_menu()
        self.setup_worker()
        self._tune_layout_spacing()

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
        self.performance_label = QLabel("Sweep time: -- ms")
        self.channel_power_label = QLabel("Channel Power: -- dBm")

        # Frequency controls (center)
        center_mhz = default_config.hardware.center_freq / 1e6
        self.freq_label = QLabel(f"Center: {center_mhz:.3f} MHz")
        self.center_freq_spin = QDoubleSpinBox()
        self.center_freq_spin.setSingleStep(0.1)
        self.center_freq_spin.setKeyboardTracking(False)

        # Gain controls (VGA)
        self.gain_label = QLabel(f"VGA Gain: {default_config.hardware.vga_gain} dB")
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)

        # SPAN/RBW controls
        self.span_spin = QDoubleSpinBox()
        self.span_spin.setKeyboardTracking(False)
        self.span_unit = QComboBox()
        self.rbw_spin = QDoubleSpinBox()
        self.rbw_spin.setKeyboardTracking(False)
        self.rbw_unit = QComboBox()

        # Channel power measurement
        self.channel_start_spin = QDoubleSpinBox()
        self.channel_end_spin = QDoubleSpinBox()
        self.channel_bw_label = QLabel("Bandwidth: 0.3 MHz")
        self.measured_power_label = QLabel("Power: -- dBm")

        # FFT settings
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
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 1)

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

    def _tune_layout_spacing(self):
        """Lightweight visual tuning for a cleaner UI."""
        self.setStyleSheet(
            """
            QGroupBox { font-weight: bold; }
            QLabel { font-size: 12px; }
            QDoubleSpinBox, QSpinBox { min-height: 26px; }
            QPushButton { min-height: 28px; }
            """
        )

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
        self.freq_curve_current = self.freq_plot.plot(pen=pg.mkPen("g", width=1.5))
        self.freq_curve_avg = self.freq_plot.plot(pen=pg.mkPen("y", width=1.2))
        self.freq_curve_max = self.freq_plot.plot(pen=pg.mkPen("r", width=1.0))
        self.marker_line_v = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", style=Qt.PenStyle.DotLine))
        self.marker_line_h = pg.InfiniteLine(angle=0, pen=pg.mkPen("w", style=Qt.PenStyle.DotLine))
        self.marker_label = pg.TextItem(color="w", anchor=(0, 1))
        self.freq_plot.addItem(self.marker_line_v)
        self.freq_plot.addItem(self.marker_line_h)
        self.freq_plot.addItem(self.marker_label)
        self.freq_plot.setYRange(-120, 30)
        # Keep Y axis fixed instead of auto-rescaling to current trace
        self.freq_plot.enableAutoRange(axis="y", enable=False)
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
        self._update_waterfall_freq_range()

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
        self._create_display_group(controls_layout)
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

        self.center_freq_spin.setRange(1.0, 6000.0)
        self.center_freq_spin.setDecimals(6)
        self.center_freq_spin.setSuffix(" MHz")

        center = default_config.hardware.center_freq / 1e6
        self.center_freq_spin.setValue(center)
        self._update_freq_label(center)

        self.center_freq_spin.valueChanged.connect(self.on_center_changed)

        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center:"))
        center_layout.addWidget(self.center_freq_spin)
        freq_layout.addLayout(center_layout)

        freq_group.setLayout(freq_layout)
        parent_layout.addWidget(freq_group)
        # klik na wykresie centrumje widok i ustawia marker
        self.freq_plot.scene().sigMouseClicked.connect(self._handle_freq_click)

    def _create_gain_group(self, parent_layout):
        """Create gain settings group."""
        gain_group = QGroupBox("Gain Settings")
        gain_layout = QVBoxLayout()

        gain_layout.addWidget(self.gain_label)

        self.gain_slider.setRange(0, 62)
        self.gain_slider.setValue(default_config.hardware.vga_gain)
        self.gain_slider.setSingleStep(2)
        self.gain_slider.setPageStep(2)
        self.gain_slider.sliderMoved.connect(self.on_gain_slider_moved)
        gain_layout.addWidget(self.gain_slider)

        gain_group.setLayout(gain_layout)
        parent_layout.addWidget(gain_group)

    def _create_span_group(self, parent_layout):
        """Create SPAN/RBW group."""
        span_group = QGroupBox("SPAN / RBW")
        span_layout = QVBoxLayout()

        # SPAN controls
        span_hbox = QHBoxLayout()
        self.span_spin.setRange(1, 100000.0)
        self.span_spin.setDecimals(0)
        self.span_spin.setSingleStep(1)
        self.span_spin.setValue(int(default_config.hardware.span / 1e6))
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
        self.rbw_label = QLabel("RBW: --")
        rbw_hbox.addWidget(self.rbw_label)
        self.rbw_down_btn = QToolButton()
        self.rbw_down_btn.setText("RBW -")
        self.rbw_up_btn = QToolButton()
        self.rbw_up_btn.setText("RBW +")
        # RBW + => większe RBW (mniejsza FFT), RBW - => mniejsze RBW (większa FFT)
        self.rbw_up_btn.clicked.connect(lambda: self._rbw_step(+1))
        self.rbw_down_btn.clicked.connect(lambda: self._rbw_step(-1))
        rbw_hbox.addWidget(self.rbw_down_btn)
        rbw_hbox.addWidget(self.rbw_up_btn)
        span_layout.addLayout(rbw_hbox)

        # startowa wartość RBW
        self._update_rbw_display(default_config.hardware.span)

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

    def _create_display_group(self, parent_layout):
        """Create display settings group (layers, modes, marker)."""
        disp_group = QGroupBox("Display Settings")
        disp_layout = QVBoxLayout()

        # Layer visibility checkboxes
        self.show_current_cb = QCheckBox("Show Current (green)")
        self.show_current_cb.setChecked(True)
        self.show_current_cb.stateChanged.connect(self._toggle_current_curve)
        disp_layout.addWidget(self.show_current_cb)

        self.show_avg_cb = QCheckBox("Show Average (yellow)")
        self.show_avg_cb.setChecked(True)
        self.show_avg_cb.stateChanged.connect(self._toggle_avg_curve)
        disp_layout.addWidget(self.show_avg_cb)

        self.show_max_cb = QCheckBox("Show Max Hold (red)")
        self.show_max_cb.setChecked(True)
        self.show_max_cb.stateChanged.connect(self._toggle_max_curve)
        disp_layout.addWidget(self.show_max_cb)

        self.max_hold_btn.clicked.connect(self.clear_max_hold)
        disp_layout.addWidget(self.max_hold_btn)

        # Marker toggle
        self.marker_cb = QCheckBox("Show marker")
        self.marker_cb.setChecked(True)
        self.marker_cb.stateChanged.connect(self._toggle_marker)
        disp_layout.addWidget(self.marker_cb)

        disp_group.setLayout(disp_layout)
        parent_layout.addWidget(disp_group)

    def _create_waterfall_group(self, parent_layout):
        """Create waterfall controls group."""
        wf_group = QGroupBox("Waterfall Controls")
        wf_layout = QVBoxLayout()

        self.auto_range_btn.clicked.connect(self.auto_range)
        wf_layout.addWidget(self.auto_range_btn)

        # Colormap selector
        self.wf_cmap_combo = QComboBox()
        self.wf_cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "turbo"])
        self.wf_cmap_combo.setCurrentText("viridis")
        self.wf_cmap_combo.currentTextChanged.connect(self._apply_waterfall_cmap)
        wf_layout.addWidget(QLabel("Waterfall Colormap:"))
        wf_layout.addWidget(self.wf_cmap_combo)

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
        self.worker.sweep_time_update.connect(self.on_sweep_time_update)
        self.worker.channel_power_update.connect(self.on_channel_power_update)
        self.worker.end_of_run.connect(self.schedule_worker)

        # Start thread & worker
        self.sdr_thread.started.connect(self.worker.run)
        self.sdr_thread.start()

    def on_center_changed(self, _=None):
        """Update center/span when center frequency changes."""
        center_mhz = self.center_freq_spin.value()
        span_hz = self.span_spin.value() * self._unit_multiplier(self.span_unit.currentText())
        span_hz = float(max(100.0, min(100_000_000.0, span_hz)))
        start_mhz = center_mhz - span_hz / 2e6
        stop_mhz = center_mhz + span_hz / 2e6
        center_hz = center_mhz * 1e6
        self._apply_span_and_rbw(span_hz, center_hz, start_mhz, stop_mhz)
        self.freq_plot.setXRange(start_mhz, stop_mhz, padding=0)
        self._update_waterfall_freq_range(start_mhz, stop_mhz)

    def on_gain_slider_moved(self, value):
        """Handle gain slider changes and update worker."""
        even_val = int(value) - int(value) % 2
        if even_val != value:
            self.gain_slider.blockSignals(True)
            self.gain_slider.setValue(even_val)
            self.gain_slider.blockSignals(False)
        self.gain_label.setText(f"VGA Gain: {even_val} dB")
        self.worker.update_gain(even_val)

    def _unit_multiplier(self, unit_text: str) -> float:
        """Get multiplier for frequency units."""
        if unit_text == "Hz":
            return 1.0
        if unit_text == "kHz":
            return 1e3
        if unit_text == "MHz":
            return 1e6
        return 1.0

    def _update_freq_label(self, center_mhz: float) -> None:
        """Refresh center label text."""
        self.freq_label.setText(f"Center: {center_mhz:.3f} MHz")

    def _recenter_frequency_view(self, _event=None) -> None:
        """Center frequency plot x-range to current start/stop."""
        center_mhz = self.center_freq_spin.value()
        span_hz = self.span_spin.value() * self._unit_multiplier(self.span_unit.currentText())
        span_hz = float(max(100.0, min(100_000_000.0, span_hz)))
        start_mhz = center_mhz - span_hz / 2e6
        stop_mhz = center_mhz + span_hz / 2e6
        self.freq_plot.setXRange(start_mhz, stop_mhz, padding=0)
        self._update_freq_label(center_mhz)
        self._update_waterfall_freq_range(start_mhz, stop_mhz)

    def _handle_freq_click(self, event):
        """On click: center view and set marker at clicked x, current y."""
        if self.freq_plot.sceneBoundingRect().contains(event.scenePos()):
            mouse_point = self.freq_plot.plotItem.vb.mapSceneToView(event.scenePos())
            freq_mhz = mouse_point.x()
            power_dbm = mouse_point.y()
            voltage_text = self._dbm_to_voltage_text(power_dbm)
            # update marker position
            self.marker_freq_mhz = freq_mhz
            self.marker_power_dbm = power_dbm
            if self.marker_line_v:
                self.marker_line_v.setValue(freq_mhz)
            if self.marker_line_h:
                self.marker_line_h.setValue(power_dbm)
            if self.marker_label:
                self.marker_label.setText(
                    f"f={freq_mhz:.3f} MHz, P={power_dbm:.2f} dBm, {voltage_text}"
                )
                self.marker_label.setPos(freq_mhz, power_dbm)
            # recenter view around click
            self._recenter_frequency_view()

    def _apply_span_and_rbw(self, span_hz: float, center_hz: float, start_mhz: float, stop_mhz: float) -> None:
        """Apply span/sample rate and RBW-derived FFT size to the worker."""
        # clamp span for device limits (całość widma, okna dzielone do 20 MHz)
        span_hz = float(max(100.0, min(100_000_000.0, span_hz)))

        rbw_val = self.rbw_spin.value()
        rbw_hz = rbw_val * self._unit_multiplier(self.rbw_unit.currentText())
        # derive FFT size so RBW ~= rbw_hz (span/fft_size), allow large sizes
        desired_fft = int(math.ceil(span_hz / max(rbw_hz, 1e-9)))
        fft_pow = 1 << (int(desired_fft) - 1).bit_length()  # power-of-two for speed
        desired_fft = max(128, min(self.RBW_FFT_MAX, fft_pow))

        try:
            self.worker.update_span_value(int(span_hz))
            self.worker.set_fft_size(desired_fft)
            self.worker.update_center_freq(int(center_hz / 1e3))
        except Exception:  # pylint: disable=broad-except
            pass

        # refresh UI labels/spins
        self._update_freq_label(center_hz / 1e6)
        self._update_span_display(span_hz)
        self._update_rbw_display(span_hz)
        self.freq_plot.setXRange(start_mhz, stop_mhz, padding=0)
        self._update_waterfall_freq_range(start_mhz, stop_mhz)

    def on_span_changed(self, _=None):
        """Compute displayed span in Hz, clamp to valid range and update worker."""
        # compute span in Hz from spin + unit
        span_val = self.span_spin.value()
        span_hz = span_val * self._unit_multiplier(self.span_unit.currentText())
        # clamp to [100 Hz, 100 MHz]
        span_hz = max(100.0, min(100_000_000.0, span_hz))
        center_mhz = self.center_freq_spin.value()
        start_mhz = center_mhz - span_hz / 2e6
        stop_mhz = center_mhz + span_hz / 2e6
        center_hz = center_mhz * 1e6
        self._apply_span_and_rbw(span_hz, center_hz, start_mhz, stop_mhz)
        self.freq_plot.setXRange(start_mhz, stop_mhz, padding=0)
        self._update_waterfall_freq_range(start_mhz, stop_mhz)

    def _update_rbw_display(self, span_hz):
        """Update RBW display based on span."""
        if not hasattr(self, "rbw_label"):
            return
        # efektywny RBW liczony na pojedynczym oknie (<=20 MHz)
        windows = getattr(self.worker, "windows", [])
        if windows:
            fft_size = windows[0].fft_size
            window_span = windows[0].span_hz
        else:
            fft_size = default_config.processing.fft_size
            window_span = min(span_hz, self.MAX_DEVICE_SPAN_HZ)
        rbw_hz = window_span / float(max(1, fft_size))
        # choose a friendly unit for rbw display
        if rbw_hz >= 1e6:
            text_val, unit = rbw_hz / 1e6, "MHz"
        elif rbw_hz >= 1e3:
            text_val, unit = rbw_hz / 1e3, "kHz"
        else:
            text_val, unit = rbw_hz, "Hz"
        self.rbw_label.setText(f"RBW: {text_val:.2f} {unit}")

    def on_rbw_changed(self, _=None):
        """Handle RBW control changes (RBW drives effective sample rate)."""
        rbw_val = self.rbw_spin.value()
        rbw_hz = rbw_val * self._unit_multiplier(self.rbw_unit.currentText())
        center_mhz = self.center_freq_spin.value()
        span_hz = max(100.0, min(100_000_000.0, self.span_spin.value() * self._unit_multiplier(self.span_unit.currentText())))
        start_mhz = center_mhz - span_hz / 2e6
        stop_mhz = center_mhz + span_hz / 2e6

        # derive FFT size so RBW ~= rbw_hz (span/fft_size)
        desired_fft = int(math.ceil(span_hz / max(rbw_hz, 1e-9)))
        fft_pow = 1 << (int(desired_fft) - 1).bit_length()  # power-of-two for speed
        desired_fft = max(128, min(self.RBW_FFT_MAX, fft_pow))

        span_hz = min(span_hz, 100_000_000.0)
        try:
            self.worker.update_span_value(int(span_hz))
            self.worker.set_fft_size(desired_fft)
        except Exception:  # pylint: disable=broad-except
            pass

        self._update_freq_label(center_mhz)
        self._update_span_display(span_hz)
        center_hz = center_mhz * 1e6
        self.worker.update_center_freq(int(center_hz / 1e3))
        self._update_rbw_display(span_hz)

        # Faktyczne RBW po uwzględnieniu ograniczeń FFT (span/fft_size okna)
        windows = getattr(self.worker, "windows", [])
        if windows:
            fft_size_actual = windows[0].fft_size
            window_span = windows[0].span_hz
        else:
            fft_size_actual = desired_fft
            window_span = min(span_hz, self.MAX_DEVICE_SPAN_HZ)
        actual_rbw = window_span / float(max(1, fft_size_actual))
        readable = actual_rbw
        if readable >= 1e6:
            text_val, unit = readable / 1e6, "MHz"
        elif readable >= 1e3:
            text_val, unit = readable / 1e3, "kHz"
        else:
            text_val, unit = readable, "Hz"
        self.status_label.setText(f"RBW set to {text_val:.3f} {unit}")
        self.rbw_label.setText(f"RBW: {text_val:.2f} {unit}")

    def _rbw_step(self, direction: int):
        """Adjust RBW via FFT size steps (direction>0 => większe RBW = mniejsze FFT)."""
        windows = getattr(self.worker, "windows", [])
        current_fft = windows[0].fft_size if windows else getattr(self.worker, "fft_size", 8192)
        if direction > 0:
            new_fft = max(128, current_fft // 2)
        else:
            new_fft = min(self.RBW_FFT_MAX, current_fft * 2)
        if new_fft == current_fft:
            return
        # ustaw FFT i zaktualizuj widoki
        try:
            self.worker.set_fft_size(new_fft)
        except Exception:
            return
        span_hz = getattr(self.worker, "span", 0)
        self._update_rbw_display(span_hz)
        self._recenter_frequency_view()
        self.status_label.setText(f"FFT size set to {new_fft}")

    def _update_span_display(self, desired_sr):
        """Update SPAN display based on desired sample rate."""
        self.span_spin.blockSignals(True)
        self.span_unit.blockSignals(True)
        if desired_sr >= 1e6:
            self.span_unit.setCurrentText("MHz")
            self.span_spin.setValue(int(round(desired_sr / 1e6)))
        elif desired_sr >= 1e3:
            self.span_unit.setCurrentText("kHz")
            self.span_spin.setValue(int(round(desired_sr / 1e3)))
        else:
            self.span_unit.setCurrentText("Hz")
            self.span_spin.setValue(int(round(desired_sr)))
        self.span_unit.blockSignals(False)
        self.span_spin.blockSignals(False)
        # zaktualizuj RBW label i oś waterfall
        self._update_rbw_display(desired_sr)
        self._update_waterfall_freq_range()

    def _update_waterfall_freq_range(self, start_mhz: float | None = None, stop_mhz: float | None = None):
        """Align waterfall X-range and image rect to current span."""
        if start_mhz is None or stop_mhz is None:
            center_mhz = self.center_freq_spin.value()
            span_hz = self.span_spin.value() * self._unit_multiplier(self.span_unit.currentText())
            span_hz = float(max(100.0, min(100_000_000.0, span_hz)))
            start_mhz = center_mhz - span_hz / 2e6
            stop_mhz = center_mhz + span_hz / 2e6
        width_mhz = stop_mhz - start_mhz
        if self.waterfall_plot:
            self.waterfall_plot.setXRange(start_mhz, stop_mhz, padding=0)
        # setRect tylko gdy znamy rozmiar obrazu
        if self.imageitem and getattr(self.imageitem, "image", None) is not None:
            height_rows = self.imageitem.image.shape[0]
            self.imageitem.setRect(QRectF(start_mhz, 0, width_mhz, height_rows))
            self.imageitem.update()

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

    def _toggle_current_curve(self, state):
        if self.freq_curve_current:
            enable = state != 0
            if enable and self.last_freq_axis is not None and self.last_psd_current is not None:
                n = min(len(self.last_freq_axis), len(self.last_psd_current))
                self.freq_curve_current.setData(self.last_freq_axis[:n], self.last_psd_current[:n])
            self.freq_curve_current.setVisible(enable)

    def _toggle_avg_curve(self, state):
        if self.freq_curve_avg:
            enable = state != 0
            if enable and self.last_freq_axis is not None and self.last_psd_avg is not None:
                n = min(len(self.last_freq_axis), len(self.last_psd_avg))
                self.freq_curve_avg.setData(self.last_freq_axis[:n], self.last_psd_avg[:n])
            self.freq_curve_avg.setVisible(enable)

    def _toggle_max_curve(self, state):
        if self.freq_curve_max:
            enable = state != 0
            if enable and self.last_freq_axis is not None and self.last_psd_max is not None:
                n = min(len(self.last_freq_axis), len(self.last_psd_max))
                self.freq_curve_max.setData(self.last_freq_axis[:n], self.last_psd_max[:n])
            self.freq_curve_max.setVisible(enable)

    def _toggle_marker(self, state):
        visible = state != 0
        if self.marker_line_v:
            self.marker_line_v.setVisible(visible)
        if self.marker_line_h:
            self.marker_line_h.setVisible(visible)
        if self.marker_label:
            self.marker_label.setVisible(visible)

    def toggle_mode(self):
        """Toggle between real-time and max-hold display modes."""
        self.worker.max_hold_mode = not self.worker.max_hold_mode
        self.mode_btn.setText("Mode: Max hold" if self.worker.max_hold_mode else "Mode: Real-time")

    def clear_max_hold(self):
        """Clear max-hold buffer to minimum values."""
        for window in getattr(self.worker, "windows", []):
            if hasattr(window.processor, "psd_max"):
                window.processor.psd_max = -120.0 * np.ones_like(window.processor.psd_max)

    def auto_range(self):
        """Auto-range waterfall levels based on spectrogram statistics."""
        spectrogram = self.last_waterfall_data
        if spectrogram is not None:
            # dostosuj do aktualnej mocy: min/max z danych (z lekkim marginesem)
            vmin = float(np.min(spectrogram))
            vmax = float(np.max(spectrogram))
            margin = 1.0
            self.spectrogram_min = vmin - margin
            self.spectrogram_max = vmax + margin
            self.wf_min_spin.setValue(self.spectrogram_min)
            self.wf_max_spin.setValue(self.spectrogram_max)
            self.imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
            if self.colorbar:
                self.colorbar.setLevels((self.spectrogram_min, self.spectrogram_max))

    def update_waterfall_levels(self):
        """Apply user-specified waterfall min/max levels to the display."""
        self.spectrogram_min = self.wf_min_spin.value()
        self.spectrogram_max = self.wf_max_spin.value()
        self.imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
        self.colorbar.setLevels((self.spectrogram_min, self.spectrogram_max))

    def _apply_waterfall_cmap(self, cmap_name: str):
        """Change waterfall colormap and update colorbar."""
        try:
            cmap = pg.colormap.get(cmap_name)
        except KeyError:
            return
        self.imageitem.setLookupTable(cmap.getLookupTable(0.0, 1.0, 1024))
        if self.colorbar:
            self.colorbar.setColorMap(cmap)

    def on_time_update(self, samples):
        """Receive time-domain samples and update traces."""
        self.time_curve_i.setData(samples.real)
        self.time_curve_q.setData(samples.imag)

    def on_freq_update(self, psd):
        """Update frequency-domain plot with new PSD data."""
        # Unpack raw (current) and averaged PSD
        if isinstance(psd, tuple):
            raw_psd, avg_psd = psd
        else:
            raw_psd, avg_psd = psd, None

        psd_len = len(raw_psd) if raw_psd is not None else len(avg_psd) if avg_psd is not None else 0
        center_mhz = self.worker.center_freq / 1e3
        f = np.linspace(
            center_mhz - (self.worker.span / 2) / 1e6,
            center_mhz + (self.worker.span / 2) / 1e6,
            psd_len,
        )

        current_psd = raw_psd if raw_psd is not None else avg_psd

        if current_psd is not None and psd_len > 0:
            self.freq_curve_current.setData(f, current_psd)
            self.last_freq_axis = f
            self.last_psd_current = current_psd

        # build concatenated avg/max to match current length
        windows = getattr(self.worker, "windows", [])
        if windows:
            max_concat = []
            if avg_psd is not None and psd_len > 0:
                self.freq_curve_avg.setData(f, avg_psd)
                self.last_psd_avg = avg_psd
            else:
                avg_concat = []
                for w in windows:
                    proc = getattr(w, "processor", None)
                    if proc is None:
                        continue
                    if hasattr(proc, "psd_avg"):
                        avg_concat.append(proc.psd_avg)
                    if hasattr(proc, "psd_max"):
                        max_concat.append(proc.psd_max)
                if avg_concat:
                    avg_full = np.concatenate(avg_concat)
                    avg_full = avg_full[: len(f)]
                    self.freq_curve_avg.setData(f[: len(avg_full)], avg_full)
                    self.last_psd_avg = avg_full
                if max_concat:
                    max_full = np.concatenate(max_concat)
                    max_full = max_full[: len(f)]
                    self.freq_curve_max.setData(f[: len(max_full)], max_full)
                    self.last_psd_max = max_full
            # Max hold from processors
            if not max_concat:
                for w in windows:
                    proc = getattr(w, "processor", None)
                    if proc is None or not hasattr(proc, "psd_max"):
                        continue
                    max_concat.append(proc.psd_max)
                if max_concat:
                    max_full = np.concatenate(max_concat)
                    max_full = max_full[: len(f)]
                    self.freq_curve_max.setData(f[: len(max_full)], max_full)
                    self.last_psd_max = max_full
        else:
            self.last_psd_avg = None
            self.last_psd_max = None

        # marker: jeśli ustawiony, odczytaj aktualną wartość PSD w pobliżu i zaktualizuj opis
        if self.marker_freq_mhz is not None and current_psd is not None and psd_len > 0:
            idx = int(np.argmin(np.abs(f - self.marker_freq_mhz)))
            idx = max(0, min(psd_len - 1, idx))
            power_val = current_psd[idx]
            self.marker_power_dbm = power_val
            voltage_text = self._dbm_to_voltage_text(power_val)
            if self.marker_line_v:
                self.marker_line_v.setValue(self.marker_freq_mhz)
            if self.marker_line_h:
                self.marker_line_h.setValue(power_val)
            if self.marker_label:
                self.marker_label.setText(
                    f"f={self.marker_freq_mhz:.3f} MHz, P={power_val:.2f} dBm, {voltage_text}"
                )
                self.marker_label.setPos(self.marker_freq_mhz, power_val)

    def on_waterfall_update(self, spectrogram):
        """Update waterfall image with new spectrogram data."""
        self.imageitem.setImage(spectrogram, autoLevels=False)
        self.last_waterfall_data = spectrogram
        # zsynchronizuj prostokąt z aktualnym spanem po wstawieniu danych
        center_mhz = self.worker.center_freq / 1e3
        span_hz = self.worker.span
        start_mhz = center_mhz - span_hz / 2e6
        stop_mhz = center_mhz + span_hz / 2e6
        width_mhz = stop_mhz - start_mhz
        if self.imageitem and getattr(self.imageitem, "image", None) is not None:
            self.imageitem.setRect(QRectF(start_mhz, 0, width_mhz, spectrogram.shape[0]))
            self.imageitem.update()

    def on_status_update(self, message):
        """Display status messages from the worker in the UI."""
        self.status_label.setText(message)

    def on_sweep_time_update(self, sweep_ms):
        """Update sweep time display in the UI."""
        self.current_sweep_ms = sweep_ms
        self.performance_label.setText(f"Sweep time: {sweep_ms:.1f} ms")

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

    def _dbm_to_voltage_text(self, power_dbm: float, impedance_ohm: float = 50.0) -> str:
        """Convert dBm to Vrms for a given impedance (best-effort, uncalibrated)."""
        # power_dbm -> watts
        power_w = 1e-3 * (10.0 ** (power_dbm / 10.0))
        # Vrms = sqrt(P * R)
        v_rms = math.sqrt(power_w * impedance_ohm)
        return f"V≈{v_rms*1e3:.2f} mVrms"
