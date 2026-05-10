"""Windows-specific hardware strategy."""

import logging
import gc
from pathlib import Path

import pythoncom
import wmi  # pylint: disable=import-error
from tenacity import retry, stop_after_attempt, wait_incrementing

from .base_strategy import (  # pylint: disable=no-name-in-module
    OSHardwareStrategy,
)

logger = logging.getLogger(__name__)


class WindowsHardwareStrategy(
    OSHardwareStrategy
):  # pylint: disable=too-few-public-methods
    """Windows implementation for locating and mounting device mass storage."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_incrementing(start=0.5, increment=0.5),
        reraise=True,
        before_sleep=lambda rs: logger.warning(
            "find_assigned_drive attempt %d failed: %s. Retrying...",
            rs.attempt_number,
            rs.outcome.exception(),
        ),
    )
    def find_assigned_drive(self, device_name: str | None, serial_number: str) -> Path:
        """Find assigned drive by MSC serial number.

        Using WMI get disk drive and mapping to disk partition and logical
        disk. From those associations read assigned drive letter like `E:`.

        disk drive <--> "Antecedent" disk drive to disk partition "Dependent" <-->
        "Antecedent" logical disk to partition "Dependent" -> Device ID

        :param device_name: Optional name of the audio device.
        :param serial_number: Mass storage serial number of the audio device.
        :return: Drive path for the matched device (for example, ``E:``).
        """
        pythoncom.CoInitialize()  # pylint: disable=no-member

        c = None
        disk_drive = None
        disk_drive_to_partition = None
        logical_disk_to_partition = None

        try:
            c = wmi.WMI()
            msc_serial_number = serial_number
            disk_drive = c.Win32_DiskDrive(SerialNumber=msc_serial_number)

            if disk_drive:
                for disk_drive_to_partition in c.Win32_DiskDriveToDiskPartition():
                    if disk_drive_to_partition.Antecedent == disk_drive[0]:
                        break
                else:
                    raise RuntimeError(
                        f"Could not find partition assigned to drive {disk_drive=}"
                    )

                for logical_disk_to_partition in c.Win32_LogicalDiskToPartition():
                    if (
                        logical_disk_to_partition.Antecedent
                        == disk_drive_to_partition.Dependent
                    ):
                        drive_letter = str(logical_disk_to_partition.Dependent.DeviceID)
                        logger.info(
                            "Found drive letter '%s' for %s with serial number '%s'.",
                            drive_letter,
                            device_name if device_name else "device",
                            serial_number,
                        )
                        return Path(drive_letter)

                raise RuntimeError(
                    f"Could not find logical disk assigned to "
                    f"partition {disk_drive_to_partition=}"
                )

            raise RuntimeError(f"Could not find disk drive with {msc_serial_number=}")

        finally:
            c = None
            disk_drive = None
            disk_drive_to_partition = None
            logical_disk_to_partition = None

            gc.collect()
            pythoncom.CoUninitialize()  # pylint: disable=no-member

    def _mount_drive(self, device_name: str | None, serial_number: str) -> Path:
        """Mount the USB drive identified by serial number.

        :param device_name: Optional name of the audio device.
        :param serial_number: Mass storage serial number of the audio device.
        :return: Path to the mounted drive.
        """

    # pylint: disable=unused-argument
    def get_audio_device_name(
        self, serial_number: str, config_device_name: str, timeout_s: float = 5
    ) -> str:
        """Return the audio device name from config.

        On Windows, PyAudio device names are stable and match what is specified
        in ``config_device_name``, so no hardware lookup is needed.

        :param serial_number: USB serial number (unused on Windows).
        :param config_device_name: Expected device name to return directly.
        :param timeout_s: Unused on Windows; kept for interface compatibility.
        :return: ``config_device_name`` unchanged.
        """
        return config_device_name
