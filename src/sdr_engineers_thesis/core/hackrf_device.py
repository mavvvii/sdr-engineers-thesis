"""HackRF device initialization and control module."""

import numpy as np
from hackrf import HackRF


class HackRFInitializationError(Exception):
    """Custom exception for HackRF initialization errors."""


class HackRFDevice(HackRF):
    """Initialization and control of HackRF device.

    Attributes:
        device: Instance of the HackRF device.

    Methods:
        set_center_freq: Set the center frequency of the device.
        set_lna_gain: Set the LNA gain of the device.
        set_vga_gain: Set the VGA gain of the device.
        set_span: Set the sample rate (span) of the device.
        read: Read samples from the device.
    """

    def __init__(self, span: int, center_freq: float, lna_gain: int, vga_gain: int) -> None:
        """Initialize HackRF device with specified settings.

        Args:
            span: Sample rate in samples per second.
            center_freq: Center frequency in Hz.
            lna_gain: LNA gain in dB.
            vga_gain: VGA gain in dB.

        Returns:
            None
        """
        try:
            self.device: HackRF = HackRF()
            self.set_span(span)
            self.set_center_freq(center_freq)
            self.set_lna_gain(lna_gain)
            self.set_vga_gain(vga_gain)
            self.device.disable_amp()
            print("✓ HackRF initialized successfully")
        except (OSError, IOError, ValueError, AttributeError) as e:
            raise HackRFInitializationError(f"Failed to initialize HackRF device: {e}") from e

    def set_center_freq(self, freq_hz: float) -> None:
        """Set the center frequency of the HackRF device.

        Args:
            freq_hz: Center frequency in Hz.

        Returns:
            None

        Raises:
            ValueError: If the frequency is out of valid range.
        """
        try:
            self.device.center_freq = freq_hz
        except ValueError as e:
            raise ValueError(f"Invalid center frequency: {freq_hz}") from e

    def set_lna_gain(self, gain_db: int) -> None:
        """Set the LNA gain of the HackRF device.

        Args:
            gain_db: LNA gain in dB.

        Returns:
            None

        Raises:
            ValueError: If the gain is out of valid range.
        """
        try:
            self.device.set_lna_gain(gain_db)
        except ValueError as e:
            raise ValueError(f"Invalid LNA gain: {gain_db}") from e

    def set_vga_gain(self, gain_db: int) -> None:
        """Set the VGA gain of the HackRF device.

        Args:
            gain_db: VGA gain in dB.

        Returns:
            None

        Raises:
            ValueError: If the gain is out of valid range.
        """
        try:
            self.device.set_vga_gain(gain_db)
        except ValueError as e:
            raise ValueError(f"Invalid VGA gain: {gain_db}") from e

    def set_span(self, sample_rate: int) -> None:
        """Set the sample rate (span) of the HackRF device.

        Args:
            sample_rate: Sample rate in samples per second.

        Returns:
            None

        Raises:
            ValueError: If the sample rate is out of valid range.
        """
        try:
            self.device.sample_rate = int(sample_rate)
        except ValueError as e:
            raise ValueError(f"Invalid sample rate: {sample_rate}") from e

    def read(self, num_samples: int) -> np.ndarray:
        """Read samples from the HackRF device.

        Args:
            num_samples: Number of samples to read.
        Returns:
            Numpy array of complex samples.
        """
        return self.device.read_samples(num_samples)
