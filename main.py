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






def apply_environment_optimizations():
    """ULTRA OPTYMALIZACJE ŚRODOWISKA"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'

from sdr_engineers_thesis.gui.main_window import MainWindow

if __name__ == "__main__":
    apply_environment_optimizations()
    
    app = QApplication([])
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.exec()