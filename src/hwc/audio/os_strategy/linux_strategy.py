"""Linux-specific implementation of the OSHardwareStrategy for audio devices."""

import os
import re
import time
import logging
import subprocess
from pathlib import Path

import psutil
from base_strategy import (  # pylint: disable=no-name-in-module
    OSHardwareStrategy,
)

logger = logging.getLogger(__name__)


class LinuxHardwareStrategy(
    OSHardwareStrategy
):  # pylint: disable=too-few-public-methods
    """Linux-specific implementation of the OSHardwareStrategy."""

    def _get_block_device_by_serial(self, serial_number: str, timeout_s=5) -> Path:
        """Find block device path by serial number.

        :param serial_number: Serial number of the USB device.
        :param timeout_s: Timeout in seconds.
        :return: Path to the block device if found, None otherwise.
        """
        disk_by_id_path = Path("/dev/disk/by-id")
        if not disk_by_id_path.exists():
            raise RuntimeError(
                "/dev/disk/by-id/ does not exist. Cannot find block devices by serial number."
            )

        start_time = time.time()
        while (time.time() - start_time) < timeout_s:
            for device_link in disk_by_id_path.iterdir():
                device_name = device_link.name.lower()
                if (
                    device_name.startswith("usb-")
                    and serial_number.lower() in device_name
                ):
                    if "-part" in device_name:
                        continue
                    block_device = device_link.resolve()
                    logger.info(
                        "Found block device for serial '%s': %s",
                        serial_number,
                        block_device,
                    )
                    return block_device
            time.sleep(0.5)
        raise RuntimeError(
            f"Block device with serial number '{serial_number}' not found."
        )

    def _get_mount_point(self, block_device: Path) -> Path | None:
        """Get mount point for a block device by reading /proc/mounts.

        :param block_device: Path to the block device (e.g., /dev/sdb).
        :return: Mount point path if mounted, None otherwise.
        """
        for partition in psutil.disk_partitions(all=True):
            if partition.device == str(block_device):
                mount_point = Path(partition.mountpoint)
                logger.info(
                    "Block device '%s' is already mounted at %s.",
                    block_device,
                    mount_point,
                )
                return mount_point

        return None

    def _mount_drive(self, device_name: str | None, serial_number: str) -> Path:
        """Mount the USB drive identified by serial number.

        :param device_name: Optional name of the audio device, used for logging and mount point naming.
        :param serial_number: Serial number of the USB device.
        :return: Path to the mount point if successful, None otherwise.
        """

        block_device = self._get_block_device_by_serial(serial_number)

        safe_name = device_name if device_name else "device"
        mount_point = Path(f"/tmp/a2bridge_{os.getuid()}/{safe_name}_{serial_number}")

        # Is block device already mounted?
        existing_mount = self._get_mount_point(block_device)
        if existing_mount == mount_point:
            try:
                if any(mount_point.iterdir()):
                    logger.info(
                        "Block device '%s' is correctly mounted and healthy at %s.",
                        block_device,
                        mount_point,
                    )
                    return mount_point
                logger.warning(
                    "Mount at %s is empty (dead session). Tearing down...",
                    mount_point,
                )
            except OSError as e:
                logger.warning(
                    "Mount at %s is unreadable (%s). Tearing down...",
                    mount_point,
                    e,
                )
            # Force unmount if corrupted mount (handles stale mounts and ensures clean state)
            subprocess.run(
                ["sudo", "-n", "umount", "-l", str(mount_point)],
                capture_output=True,
                check=True,
            )
        elif existing_mount:
            logger.info(
                "Block device '%s' was auto-mounted at %s. Unmounting...",
                block_device,
                existing_mount,
            )
            subprocess.run(
                ["sudo", "-n", "umount", "-l", str(existing_mount)],
                capture_output=True,
                check=True,
            )

        # Ensure new mount point is clean before mounting
        if mount_point.exists() and os.path.ismount(mount_point):
            logger.warning(
                "Target mount point %s is locked by a stale mount. Forcing unmount...",
                mount_point,
            )
            subprocess.run(
                ["sudo", "-n", "umount", "-l", str(mount_point)],
                capture_output=True,
                check=True,
            )

        # Mount the block device to the target mount point with appropriate permissions
        mount_point.mkdir(parents=True, exist_ok=True)

        uid = os.getuid()
        gid = os.getgid()
        mount_options = f"uid={uid},gid={gid},sync"  # Use sync to ensure immediate write-through for test reliability

        logger.info("Mounting %s to %s...", block_device, mount_point)
        mount_cmd = [
            "sudo",
            "-n",
            "mount",
            "-o",
            mount_options,
            str(block_device),
            str(mount_point),
        ]

        try:
            subprocess.run(mount_cmd, check=True, capture_output=True, text=True)
            logger.info("Mounting successful at %s!", mount_point)
        except subprocess.CalledProcessError as e:
            logger.error("Mounting failed! Stderr: %s", e.stderr)
            raise RuntimeError(
                f"Failed to mount {block_device} to {mount_point}"
            ) from e

        return mount_point

    def find_assigned_drive(
        self, device_name: str | None, serial_number: str
    ) -> Path | None:
        """Find the mount point of the drive assigned to the audio device.

        This function identifies the drive assigned to the audio device by
        matching the serial number, then returns the mount point where files
        can be copied to.

        :param device_name: Optional name of the audio device.
        :param serial_number: Serial number of the audio device.
        :return: Path to the mount point if found and mounted, None otherwise.
        """
        if not serial_number:
            logger.warning("No serial number provided for drive search")
            return None

        mount_point = self._mount_drive(device_name, serial_number)
        logger.info("Found mount point for serial '%s': %s", serial_number, mount_point)
        return mount_point

    def _get_full_alsa_device_name(self, card_num: int) -> str:
        """Construct the full PortAudio/sounddevice device name for an ALSA card.

        Assembles the name from two ``/proc/asound`` sources so the result
        matches exactly what sounddevice / PyAudio report, for example
        ``"XAudio A2B Interface: USB Audio (hw:1,0)"``.

        :param card_num: ALSA card number (e.g. ``1`` for ``card1``).
        :return: Full device name string.
        :raises RuntimeError: if the card long name or PCM name cannot be read.
        """
        # ── card long name ────────────────────────────────────────────────────
        # /proc/asound/cards format:
        #   " N [short_id       ]: type - Long Card Name"
        cards_file = Path("/proc/asound/cards")
        card_long_name = None
        if cards_file.exists():
            for line in cards_file.read_text(encoding="utf-8").splitlines():
                if re.match(rf"^\s*{card_num}\s+\[", line) and " - " in line:
                    card_long_name = line.split(" - ", 1)[1].strip()
                    break

        if not card_long_name:
            raise RuntimeError(
                f"Could not read long name for ALSA card {card_num} "
                f"from /proc/asound/cards."
            )

        # ── PCM device name ───────────────────────────────────────────────────
        # Try playback first, then capture (USB audio exposes both on device 0).
        # /proc/asound/cardN/pcm0p/info  →  name: USB Audio
        pcm_name = None
        for pcm_dir in ("pcm0p", "pcm0c"):
            info_file = Path(f"/proc/asound/card{card_num}/{pcm_dir}/info")
            if info_file.exists():
                for line in info_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("name:"):
                        pcm_name = line.split(":", 1)[1].strip()
                        break
            if pcm_name:
                break

        if not pcm_name:
            raise RuntimeError(
                f"Could not read PCM device name for ALSA card {card_num} "
                f"from /proc/asound/card{card_num}/pcm0p(c)/info."
            )

        return f"{card_long_name}: {pcm_name} (hw:{card_num},0)"

    # pylint: disable=unused-argument, too-many-nested-blocks
    def get_audio_device_name(
        self, serial_number: str, config_device_name: str, timeout_s: float = 5
    ) -> str:
        """Find the full PortAudio/sounddevice device name for a USB audio device
        by its serial number.

        Iterates over ``/sys/class/sound/cardN`` entries (control-device nodes
        only, excluding PCM sub-devices such as ``card0D0``).  For each card the
        sysfs path is walked upward until a ``serial`` attribute file is found.
        When the serial matches, the full device name is assembled from
        ``/proc/asound`` so it matches exactly what sounddevice reports, e.g.
        ``"XAudio A2B Interface: USB Audio (hw:1,0)"``.

        The search is retried every 0.5 s until ``timeout_s`` is reached, to
        allow time for the kernel to register the device after it is plugged in.

        :param serial_number: USB serial number of the audio device.
        :param config_device_name: Unused on Linux; kept for interface compatibility.
        :param timeout_s: Maximum seconds to wait for the device to appear.
        :return: Full PortAudio/sounddevice device name.
        :raises RuntimeError: if ``/sys/class/sound`` does not exist, if no card
                              matching ``serial_number`` is found within ``timeout_s``,
                              or if the ``/proc/asound`` files cannot be read.
        """
        sound_class = Path("/sys/class/sound")
        if not sound_class.exists():
            raise RuntimeError(
                "Cannot find audio device by serial: /sys/class/sound does not exist."
            )

        start_time = time.time()
        while (time.time() - start_time) < timeout_s:
            for card_link in sound_class.iterdir():
                if not re.fullmatch(r"card\d+", card_link.name):
                    continue

                real_path = card_link.resolve()
                current = real_path.parent

                while current != current.parent:
                    serial_file = current / "serial"
                    if serial_file.exists():
                        try:
                            found_serial = serial_file.read_text(
                                encoding="utf-8"
                            ).strip()
                            if found_serial.upper() == serial_number.upper():
                                card_num = int(
                                    re.search(r"card(\d+)", card_link.name).group(1)
                                )
                                name = self._get_full_alsa_device_name(card_num)
                                logger.info(
                                    "Resolved audio device name '%s' for serial '%s'.",
                                    name,
                                    serial_number,
                                )
                                return name
                        except OSError:
                            pass  # unreadable serial file — continue searching
                    current = current.parent

            time.sleep(0.5)

        raise RuntimeError(
            f"Audio device with serial '{serial_number}' not found in "
            f"/sys/class/sound within {timeout_s} s."
        )
