"""Audio playback and capture control with device management.

Provides :class:`AudioControl` for playing and capturing audio via sounddevice.
Recording uses blocking ``sd.rec()`` / ``sd.playrec()``; looping playback uses a
callback ``sd.OutputStream`` so both can run concurrently on different devices.

Returned capture arrays are always shaped ``(n_samples, channels)`` — the native
sounddevice / interleaved layout.
"""

import sys
import logging
import time
from pathlib import Path
from typing import Optional, Union, Literal

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

if sys.platform == "win32":
    from .os_strategy.windows_strategy import WindowsHardwareStrategy
else:
    from .os_strategy.linux_strategy import LinuxHardwareStrategy

logger = logging.getLogger(__name__)

#: PortAudio status flags, in the bit order :class:`StreamStats` packs them.
FLAG_NAMES = (
    "input_underflow",
    "input_overflow",
    "output_underflow",
    "output_overflow",
    "priming_output",
)

#: Individually recorded abnormal callbacks per stream.  Past this the run is
#: broken in a way the aggregates already describe, so the ring stops growing
#: and a stream lasting hours cannot exhaust memory.
MAX_STATUS_EVENTS = 4096

#: A callback slower than this is worth recording even without a status flag:
#: it is the likely cause of the underflow reported by the *next* one.
SLOW_CALLBACK_S = 0.05


def _describe_flags(bits: int) -> str:
    """Render a packed status word as readable flag names.

    :param bits: packed flags as stored by :class:`StreamStats`
    :returns: comma-separated flag names, or ``"none"``
    """
    names = [n for i, n in enumerate(FLAG_NAMES) if bits & (1 << i)]
    return ", ".join(names) if names else "none"


class StreamStats:
    """Accumulate PortAudio callback statistics for one stream.

    Updated from the audio thread, so every method called there does nothing but
    write numbers into preallocated storage: no logging, no allocation, no locks.
    Formatting happens in :meth:`summary`, after the stream has stopped — logging
    from inside a callback would cause the very underflow being measured.

    :param label: name used in log lines, to tell concurrent streams apart
    """

    __slots__ = (
        "label",
        "calls",
        "frames",
        "flags_seen",
        "flag_counts",
        "dur_sum",
        "dur_max",
        "gap_max",
        "headroom_min",
        "n_events",
        "_last_end",
        "_ev_call",
        "_ev_flags",
        "_ev_dur",
        "_ev_head",
    )

    def __init__(self, label: str = "stream") -> None:
        self.label = label
        self.calls = 0
        self.frames = 0
        self.flags_seen = 0
        self.flag_counts = [0] * len(FLAG_NAMES)
        self.dur_sum = 0.0
        self.dur_max = 0.0
        self.gap_max = 0.0
        self.headroom_min = float("inf")
        self.n_events = 0
        self._last_end = 0.0
        self._ev_call = np.zeros(MAX_STATUS_EVENTS, dtype=np.int64)
        self._ev_flags = np.zeros(MAX_STATUS_EVENTS, dtype=np.int32)
        self._ev_dur = np.zeros(MAX_STATUS_EVENTS, dtype=np.float64)
        self._ev_head = np.zeros(MAX_STATUS_EVENTS, dtype=np.float64)

    def record(
        self, frames: int, status, headroom: float, t0: float, t1: float
    ) -> None:
        """Record one callback.  **Audio thread only.**

        :param frames: frames this callback handled
        :param status: the ``sounddevice.CallbackFlags`` PortAudio passed
        :param headroom: seconds between now and when this buffer reaches the
            DAC — the margin that must reach zero before the device starves
        :param t0: ``perf_counter`` before the callback body
        :param t1: ``perf_counter`` after it
        """
        dur = t1 - t0
        self.calls += 1
        self.frames += frames
        self.dur_sum += dur
        if dur > self.dur_max:
            self.dur_max = dur
        if self._last_end:
            gap = t0 - self._last_end
            if gap > self.gap_max:
                self.gap_max = gap
        self._last_end = t1
        if 0.0 < headroom < self.headroom_min:
            self.headroom_min = headroom

        bits = 0
        if status:
            for i, name in enumerate(FLAG_NAMES):
                if getattr(status, name):
                    bits |= 1 << i
                    self.flag_counts[i] += 1
            self.flags_seen |= bits

        if bits or dur > SLOW_CALLBACK_S:
            i = self.n_events
            if i < MAX_STATUS_EVENTS:
                self._ev_call[i] = self.calls
                self._ev_flags[i] = bits
                self._ev_dur[i] = dur
                self._ev_head[i] = headroom
            self.n_events = i + 1

    @property
    def had_underflow(self) -> bool:
        """Whether an output underflow was reported at any point.

        :returns: ``True`` if this machine failed to feed the device in time
        """
        return bool(self.flag_counts[FLAG_NAMES.index("output_underflow")])

    def summary(self) -> None:
        """Log aggregates and every recorded abnormal callback."""
        if self.calls == 0:
            logger.info("[%s] no callbacks recorded", self.label)
            return
        logger.info(
            "[%s] %d callbacks, %d frames; callback duration mean=%.3f ms "
            "max=%.3f ms; longest gap between callbacks=%.1f ms; min DAC "
            "headroom=%.1f ms",
            self.label,
            self.calls,
            self.frames,
            self.dur_sum / self.calls * 1e3,
            self.dur_max * 1e3,
            self.gap_max * 1e3,
            (
                self.headroom_min * 1e3
                if self.headroom_min != float("inf")
                else float("nan")
            ),
        )
        for name, count in zip(FLAG_NAMES, self.flag_counts):
            logger.log(
                logging.WARNING if count else logging.DEBUG,
                "[%s]   %-17s %d",
                self.label,
                name,
                count,
            )
        if not self.flags_seen and self.n_events == 0:
            logger.info(
                "[%s] clean: PortAudio reported no xruns, so any gap in the "
                "audio originated outside this machine",
                self.label,
            )
            return
        if self.n_events > MAX_STATUS_EVENTS:
            logger.warning(
                "[%s] %d abnormal callbacks; only the first %d were recorded",
                self.label,
                self.n_events,
                MAX_STATUS_EVENTS,
            )
        for i in range(min(self.n_events, MAX_STATUS_EVENTS)):
            logger.warning(
                "[%s]   callback #%d: %s (duration %.3f ms, DAC headroom %.1f ms)",
                self.label,
                int(self._ev_call[i]),
                _describe_flags(int(self._ev_flags[i])),
                self._ev_dur[i] * 1e3,
                self._ev_head[i] * 1e3,
            )


class AudioControl:
    """Audio in/out control backed by sounddevice.

    Recording uses ``sd.rec()`` (input only) or ``sd.playrec()`` (simultaneous
    capture + playback on the same device).  Looping playback uses a callback
    ``sd.OutputStream`` so it does not conflict with ``sd.rec()`` on a separate
    device.
    """

    def __init__(
        self, audio_in_selector: Optional[dict], audio_out_selector: Optional[dict]
    ) -> None:
        """Initialise audio control with input/output device selectors.

        :param audio_in_selector: sounddevice device property filters for the
            input device.  Supported keys: ``name`` (substring), ``hostapi``
            (index), ``hostApiName`` (resolved to ``hostapi`` index).
        :param audio_out_selector: same as ``audio_in_selector`` for the output device.
        """
        self._audio_in_selector = audio_in_selector
        self._audio_out_selector = audio_out_selector

        self._playing_audio = False
        self._out_stream: Optional[sd.OutputStream] = None

        self._play_data: Optional[np.ndarray] = (
            None  # WAV data loaded into memory for looping playback; shared by the callback.
        )
        self._play_pos: int = 0  # current frame index in _play_data
        self._out_stats = StreamStats("playback")
        self._recording: Optional[np.ndarray] = (
            None  # Buffer written by sd.rec()/sd.playrec().
        )

    def __del__(self):
        self.close_resources()

    def close_resources(self) -> None:
        """Close stream handlers."""
        self._playing_audio = False
        if self._out_stream:
            self._out_stream.stop()
            self._out_stream.close()
            self._out_stream = None
            try:
                self._out_stats.summary()
            except Exception:  # pylint:disable=broad-except
                # Reachable from __del__, where logging may already be torn
                # down; a lost summary must not break interpreter shutdown.
                pass

    def _log_stream_open(self, stream, device_index: int, direction: str) -> None:
        """Log what PortAudio granted for a freshly opened stream.

        Requested settings are hints: PortAudio substitutes what the device can
        actually do, and the granted blocksize and latency are what determine
        how much slack there is before an underflow.  Logging them is the only
        way to know after the fact which was in force.

        :param stream: the started stream
        :param device_index: sounddevice device index it was opened on
        :param direction: ``"output"`` or ``"input"``, used in the log line
        """
        logger.info("sounddevice %s, %s", sd.__version__, sd.get_portaudio_version()[1])
        try:
            info = sd.query_devices(device_index)
            api = sd.query_hostapis(info["hostapi"])
            logger.info(
                "%s device[%s] %r via %s | max in/out ch=%s/%s | device latency "
                "low=%.4f s high=%.4f s | default sr=%s",
                direction,
                device_index,
                info["name"],
                api["name"],
                info["max_input_channels"],
                info["max_output_channels"],
                info[f"default_low_{direction}_latency"],
                info[f"default_high_{direction}_latency"],
                info["default_samplerate"],
            )
        except Exception:  # pylint:disable=broad-except
            logger.warning("Could not query %s device info", direction, exc_info=True)
        try:
            logger.info(
                "%s stream granted: samplerate=%s blocksize=%s latency=%s "
                "channels=%s dtype=%s samplesize=%s",
                direction,
                stream.samplerate,
                stream.blocksize,
                stream.latency,
                stream.channels,
                stream.dtype,
                stream.samplesize,
            )
            if stream.blocksize == 0:
                logger.info(
                    "blocksize 0 = the driver picks per callback, so frame "
                    "counts vary between callbacks"
                )
        except Exception:  # pylint:disable=broad-except
            logger.warning("Could not read granted stream settings", exc_info=True)

    @property
    def out_stats(self) -> StreamStats:
        """Callback statistics for the current or last playback session.

        :returns: the live :class:`StreamStats`; ``had_underflow`` is the bit
            that says whether this machine starved the device
        """
        return self._out_stats

    def _find_host_api_index(self, host_api_name: str) -> int:
        """Return the sounddevice host-API index matching *host_api_name*.

        :param host_api_name: exact host API name (e.g. ``"Windows WDM-KS"``)
        :returns: matching host API index
        :raises RuntimeError: if no host API with that name exists
        """
        for i, api in enumerate(sd.query_hostapis()):
            logger.debug("Checking host API %s: %s", i, api)
            if api["name"] == host_api_name:
                logger.debug("Found host API index %s: %s", i, api)
                return i
        raise RuntimeError(f"Could not find host API with name {host_api_name}")

    def _get_audio_device_name(self, serial_number: str, name: str | None) -> str:
        """Resolve the audio device name for the given USB serial number.

        :param serial_number: Serial number of the audio device.
        :param name: Optional name of the audio device.
        :return: Resolved audio device name.
        """
        os_hardware = (
            WindowsHardwareStrategy()
            if sys.platform == "win32"
            else LinuxHardwareStrategy()
        )
        return os_hardware.get_audio_device_name(serial_number, name)

    def _find_audio_device_index(self, **kwargs) -> int:
        """Return the index of the first audio device matching *kwargs*.

        ``name`` is matched as a substring; ``hostApiName`` is resolved to a
        ``hostapi`` index; all other keys use exact match.

        :param kwargs: device property filters (e.g. ``name="USB Audio"``,
            ``hostApiName="Windows WDM-KS"``, ``max_input_channels=2``)
        :returns: index of the first matching device
        :raises RuntimeError: if no device matches
        """
        if "serial" in kwargs:
            kwargs["name"] = self._get_audio_device_name(
                kwargs.pop("serial"), kwargs.get("name")
            )
        if "hostApiName" in kwargs:
            kwargs["hostapi"] = self._find_host_api_index(kwargs.pop("hostApiName"))

        for i, info in enumerate(sd.query_devices()):
            logger.debug("Checking device %s: %s", i, info)
            for k, v in kwargs.items():
                if k == "name":
                    if v not in info.get(k, ""):
                        break
                elif info.get(k) != v:
                    break
            else:
                logger.debug("Found audio device index %s: %s", i, info)
                return i

        msg = "".join(f"{k}={v}, " for k, v in kwargs.items())
        raise RuntimeError(f"Could not find associated audio device {msg}")

    @staticmethod
    def _load_wav(audio_file: str) -> tuple[np.ndarray, int]:
        """Load a WAV file into a numpy array.

        24-bit samples are sign-extended to ``int32`` (no native int24 dtype).

        :param audio_file: path to WAV file
        :returns: ``(data, sample_rate)`` — *data* shaped ``(n_frames, channels)``
        """
        rate, data = wavfile.read(audio_file)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data, rate

    @staticmethod
    def _tile_to_frames(data: np.ndarray, frames: int) -> np.ndarray:
        """Repeat *data* along axis 0 until it is exactly *frames* rows long.

        :param data: source array shaped ``(n, channels)``
        :param frames: desired output frame count
        :returns: array shaped ``(frames, channels)``
        """
        repeats = (frames + len(data) - 1) // len(data)
        return np.tile(data, (repeats, 1))[:frames]

    def _stream_out_callback(
        self, outdata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Fill *outdata* from ``_play_data``, wrapping around at EOF.

        Runs on PortAudio's audio thread.  ``status`` carries the over-/underflow
        flags for this buffer and is recorded rather than discarded: without it
        a dropout in the captured audio cannot be attributed to this machine or
        cleared of it.  Recording is arithmetic only — see :class:`StreamStats`.

        :param outdata: buffer to fill, shaped ``(frames, channels)``
        :param frames: frames requested
        :param time_info: PortAudio timestamps for this buffer
        :param status: over-/underflow flags for this buffer
        """
        t0 = time.perf_counter()
        if not self._playing_audio:
            # Still record the final callback before unwinding the stream.
            self._out_stats.record(frames, status, 0.0, t0, time.perf_counter())
            raise sd.CallbackStop()

        play_data_len = len(self._play_data)
        end = self._play_pos + frames
        if end <= play_data_len:
            outdata[:] = self._play_data[self._play_pos : end]
        else:
            # wrap around
            first = play_data_len - self._play_pos
            outdata[:first] = self._play_data[self._play_pos :]
            remainder = frames - first
            outdata[first:] = self._play_data[:remainder]
            end = remainder
        self._play_pos = end % play_data_len

        try:
            headroom = time_info.outputBufferDacTime - time_info.currentTime
        except AttributeError:  # pragma: no cover - host API without timestamps
            headroom = 0.0
        self._out_stats.record(frames, status, headroom, t0, time.perf_counter())

    # pylint: disable=protected-access
    def reinit_sounddevice(self) -> None:
        """Force PortAudio to re-enumerate all audio devices.

        After a DUT config reload the device re-enumerates on the USB bus,
        making sounddevice's cached device list stale — this call refreshes it.
        """
        sd._terminate()
        sd._initialize()

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def play_data(
        self,
        data: np.ndarray,
        samplerate: int,
        channels: Optional[int] = None,
        blocksize: int = 0,
        latency: Literal["high", "low"] | float = "high",
        reinit: bool = False,
    ) -> None:
        """Play a numpy array on the output device in gapless looping mode.

        Loops automatically at EOF; call :meth:`stop_audio` to stop.
        If *channels* differs from ``data.shape[1]``, channel 0 is broadcast
        to all output channels.  Multiple calls while already playing are ignored.

        :param data: audio samples shaped ``(n_frames, src_channels)``
        :param samplerate: sample rate in Hz
        :param channels: output channel count; ``None`` uses *data*'s channel count
        :param blocksize: PortAudio frames per callback buffer (``0`` = driver default)
        :param latency: PortAudio latency hint — ``'low'``, ``'high'``, or seconds
        :param reinit: re-init PortAudio before opening stream (see :meth:`reinit_sounddevice`)
        :raises RuntimeError: if no output selector is configured
        """
        if not self._audio_out_selector:
            raise RuntimeError(
                "Selected audio device does not support audio out stream"
            )
        if self._playing_audio:
            logger.info("Audio is already running!")
            return

        self.close_resources()
        if reinit:
            self.reinit_sounddevice()

        if data.ndim == 1:  #
            data = data.reshape(-1, 1)

        if channels is not None and data.shape[1] != channels:
            logger.info(
                "Broadcasting %d source channel(s) → %d output channel(s).",
                data.shape[1],
                channels,
            )
            data = np.tile(data[:, :1], (1, channels))

        self._play_data = data
        self._play_pos = 0
        self._out_stats = StreamStats("playback")  # per-session counters
        channels = self._play_data.shape[1]
        dtype = self._play_data.dtype
        self._playing_audio = True

        out_idx = self._find_audio_device_index(**self._audio_out_selector)
        self._out_stream = sd.OutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            device=out_idx,
            callback=self._stream_out_callback,
            blocksize=blocksize,
            latency=latency,
        )
        self._out_stream.start()
        self._log_stream_open(self._out_stream, out_idx, "output")
        logger.info("Playback started on output device.")

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def play_audio(
        self,
        audio_file: str,
        channels: Optional[int] = None,
        blocksize: int = 0,
        latency: Literal["high", "low"] | float = "high",
        reinit: bool = False,
    ) -> tuple[np.ndarray, int]:
        """Load a WAV file and play it in gapless looping mode.

        Convenience wrapper around :meth:`play_data`.

        :param audio_file: path to WAV file
        :param channels: output channel count (``None`` → WAV channel count)
        :param blocksize: see :meth:`play_data`
        :param latency: see :meth:`play_data`
        :param reinit: see :meth:`play_data`
        :returns: ``(data, samplerate)`` — *data* shaped ``(n_frames, channels)``
                  for multi-channel WAVs or ``(n_frames,)`` for mono
        :raises RuntimeError: if no output selector is configured
        """
        data, samplerate = self._load_wav(audio_file)
        logger.info("Loaded %s for playback.", Path(audio_file).name)
        self.play_data(
            data,
            samplerate,
            channels=channels,
            blocksize=blocksize,
            latency=latency,
            reinit=reinit,
        )
        return data, samplerate

    def stop_audio(self) -> None:
        """Stop looping playback and close the output stream.

        Sets ``_playing_audio = False`` so the next callback invocation raises
        :class:`~sounddevice.CallbackStop`, letting PortAudio drain the stream
        cleanly.  The method then busy-waits until the stream is no longer
        active before calling ``stop()``/``close()``, avoiding a race between
        the Python thread and the PortAudio audio thread.

        Safe to call even if audio is not currently playing.
        """
        if self._playing_audio:
            self._playing_audio = False
            try:
                while self._out_stream and self._out_stream.active:
                    time.sleep(0.1)
            except Exception:  # pylint:disable=broad-except
                pass
            if self._out_stream:
                try:
                    logger.info(
                        "Output stream cpu_load at stop: %.3f",
                        self._out_stream.cpu_load,
                    )
                except Exception:  # pylint:disable=broad-except
                    pass
                self._out_stream.stop()
                self._out_stream.close()
                self._out_stream = None
            self._out_stats.summary()

    def start_recording(  # pylint:disable=too-many-arguments, too-many-positional-arguments
        self,
        duration_s: float,
        channels: int,
        sample_rate: int,
        dtype: Union[str, np.dtype] = "int16",
        play_file: Optional[str] = None,
        blocksize: int = 0,
        latency: Union[str, float] = "high",
    ) -> None:
        """Start a non-blocking recording; returns immediately.

        Uses ``sd.rec()`` (input only) or ``sd.playrec()`` when *play_file* is
        given.  Call :meth:`stop_recording` to block until completion.

        :param duration_s: recording duration in seconds
        :param channels: number of input channels to capture
        :param sample_rate: sample rate in Hz
        :param dtype: numpy dtype for captured samples (default ``'int16'``)
        :param play_file: WAV file to play simultaneously; looped if shorter than *duration_s*
        :param blocksize: PortAudio frames per callback buffer (``0`` = driver default)
        :param latency: PortAudio latency hint — ``'low'``, ``'high'``, or seconds
        :raises RuntimeError: if no input selector is configured
        :raises RuntimeError: if *play_file* given but no output selector configured
        :raises RuntimeError: if a recording is already in progress
        """
        if not self._audio_in_selector:
            raise RuntimeError("No audio input selector configured.")
        if self._recording is not None:
            raise RuntimeError(
                "A recording is already in progress. Call stop_recording() first."
            )

        frames = int(duration_s * sample_rate)
        in_idx = self._find_audio_device_index(**self._audio_in_selector)
        try:
            in_info = sd.query_devices(in_idx)
            logger.info(
                "input device[%s] %r via %s | max in ch=%s | device latency "
                "low=%.4f s high=%.4f s | requested blocksize=%s latency=%s",
                in_idx,
                in_info["name"],
                sd.query_hostapis(in_info["hostapi"])["name"],
                in_info["max_input_channels"],
                in_info["default_low_input_latency"],
                in_info["default_high_input_latency"],
                blocksize,
                latency,
            )
        except Exception:  # pylint:disable=broad-except
            logger.warning("Could not query input device info", exc_info=True)

        if play_file is not None:
            if not self._audio_out_selector:
                raise RuntimeError(
                    "play_file requires audio_out_selector to be configured."
                )
            out_idx = self._find_audio_device_index(**self._audio_out_selector)
            out_data, _ = self._load_wav(play_file)
            out_data = self._tile_to_frames(out_data, frames)

            logger.info(
                "playrec start: %.1f s, %d ch in @ %d Hz, file=%s",
                duration_s,
                channels,
                sample_rate,
                play_file,
            )
            self._recording = sd.playrec(
                out_data,
                samplerate=sample_rate,
                input_mapping=list(range(1, channels + 1)),
                dtype=dtype,
                device=(in_idx, out_idx),
                blocksize=blocksize,
                latency=latency,
            )
        else:
            logger.info(
                "rec start: %.1f s, %d ch @ %d Hz", duration_s, channels, sample_rate
            )
            self._recording = sd.rec(
                frames,
                samplerate=sample_rate,
                channels=channels,
                dtype=dtype,
                device=in_idx,
                blocksize=blocksize,
                latency=latency,
            )

    def stop_recording(self) -> np.ndarray:
        """Block until the current recording finishes and return the samples.

        :returns: numpy array shaped ``(n_samples, channels)``
        :raises RuntimeError: if no recording is in progress
        """
        if self._recording is None:
            raise RuntimeError(
                "No recording in progress. Call start_recording() first."
            )
        sd.wait()
        # rec()/playrec() run their callback inside sounddevice, so the flags
        # are only reachable afterwards through get_status().  Same question as
        # for playback: did this machine keep up, or did the audio arrive
        # already damaged?
        try:
            status = sd.get_status()
        except RuntimeError:
            status = None  # nothing finished yet; nothing to report
        if status:
            flags = [n for n in FLAG_NAMES if getattr(status, n, False)]
            logger.warning(
                "Recording finished with PortAudio status flags: %s",
                ", ".join(flags) or str(status),
            )
        else:
            logger.info("Recording finished with no PortAudio xruns reported.")
        samples = self._recording.copy()
        self._recording = None
        return samples

    def record_audio(  # pylint:disable=too-many-arguments, too-many-positional-arguments
        self,
        duration_s: float,
        channels: int,
        samplerate: int,
        dtype: Union[str, np.dtype] = "int16",
        play_file: Optional[str] = None,
        blocksize: int = 0,
        latency: Union[str, float] = "high",
    ) -> np.ndarray:
        """Record audio for a fixed duration and return per-channel samples.

        Blocking convenience wrapper around :meth:`start_recording` +
        :meth:`stop_recording`.

        :param duration_s: recording duration in seconds
        :param channels: number of input channels to capture
        :param samplerate: sample rate in Hz
        :param dtype: numpy dtype for captured samples (default ``'int16'``)
        :param play_file: WAV file to play simultaneously (see :meth:`start_recording`)
        :param blocksize: PortAudio frames per callback buffer (``0`` = driver default)
        :param latency: PortAudio latency hint — ``'low'``, ``'high'``, or seconds
        :returns: numpy array shaped ``(n_samples, channels)``
        """
        self.start_recording(
            duration_s,
            channels,
            samplerate,
            dtype,
            play_file,
            blocksize=blocksize,
            latency=latency,
        )
        return self.stop_recording()
