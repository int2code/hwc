"""OS-specific hardware access strategies for Audio devices."""

from pathlib import Path
from abc import ABC, abstractmethod


class OSHardwareStrategy(ABC):  # pylint: disable=too-few-public-methods
    """Abstract interface for OS-dependent audio hardware operations."""

    @abstractmethod
    def find_assigned_drive(
        self, device_name: str | None, serial_number: str
    ) -> Path | None:
        """Find assigned drive by serial number from audio device config.

        :param device_name: Optional name of the audio device.
        :param serial_number: Mass storage serial number of the audio device.
        :return: Path to the mounted drive or ``None`` if not found.
        """

    @abstractmethod
    def get_audio_device_name(
        self, serial_number: str, config_device_name: str, timeout_s: float = 5
    ) -> str:
        """Resolve the audio device name for the given USB serial number.

        On platforms where the OS assigns stable names (Windows), returns
        ``config_device_name`` directly.  On platforms where the name must
        be discovered at runtime (Linux/ALSA), queries the system and
        returns the real card identifier.

        :param serial_number: USB serial number of the audio device.
        :param config_device_name: Name from device config; returned directly
                                   on Windows; unused on Linux.
        :param timeout_s: Maximum seconds to wait for the device to appear
                          (Linux only; ignored on Windows).
        :return: Audio device name suitable for passing to PyAudio / sounddevice.
        :raises RuntimeError: (Linux only) if ``/sys/class/sound`` is absent,
                              if no card matching ``serial_number`` is found
                              within ``timeout_s``, or if the matched card's
                              ``id`` file cannot be read.
        """
