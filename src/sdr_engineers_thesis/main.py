"""Main entry point for the SDR Engineer's Thesis application."""

import os
import signal

from PyQt6.QtWidgets import QApplication

from sdr_engineers_thesis.gui.main_window import MainWindow


def apply_environment_optimizations() -> None:
    """
    Apply environment variable settings to optimize performance.

    Args:
        None
    Attributes:
        OMP_NUM_THREADS: Limits the number of threads used by OpenMP to 1.
        OPENBLAS_NUM_THREADS: Limits the number of threads used by OpenBLAS to 1.
        MKL_NUM_THREADS: Limits the number of threads used by Intel MKL to 1.
        NUMBA_NUM_THREADS: Limits the number of threads used by Numba to 1.

    Returns:
        None

    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"


if __name__ == "__main__":
    apply_environment_optimizations()

    app: QApplication = QApplication([])
    app.setStyle("Fusion")

    window: MainWindow = MainWindow()
    window.show()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.exec()
