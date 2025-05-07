"""Provide signals support for AVT7510 relays module board.

Example configuration:

.. code-block:: python

    from hwc.common import Signals, DOSignal
    from hwc.avt_5710.signals_avt_5710 import SignalEnginAvt5710, SignalPropertiesAvt5710


    class PowerBoard(Signals):
        dev1_power = DOSignal(
            hardware_properties=[
                SignalPropertiesAvt5710(relay_no=1, active_state=False),
            ]
        )
        dev2_power = DOSignal(
            hardware_properties=[
                SignalPropertiesAvt5710(relay_no=2, active_state=False),
            ]
        )

        dev1_pi = DOSignal(
            hardware_properties=[
                SignalPropertiesAvt5710(relay_no=5, active_state=True),
            ]
        )
        dev2_pi = DOSignal(
            hardware_properties=[
                SignalPropertiesAvt5710(relay_no=6, active_state=True),
            ]
        )


    avt_engine = SignalEnginAvt5710("COM5")

    power_board = PowerBoard(avt_engine)
    power_board.read_states()
    print(power_board.dev1_power.state)

    power_board.dev1_power.state = False

    power_board.write_states()

    print(power_board.dev1_power.state)


"""
import time
from dataclasses import dataclass
from typing import List, Literal

import serial
from retry import retry

from hwc.common import SignalProperties, SignalsEngine

# pylint notice similarity between AVT and waveshare implementations
# but there's no reason to extract it to base class or make it a template method
# pylint:disable=R0801

@dataclass
class SignalPropertiesAvt5710(SignalProperties):
    """Signal Properties specific for a AVT5710 relays module board."""

    relay_no: int
    active_state: Literal[True, False]


class SignalEnginAvt5710(SignalsEngine):
    """Signals engin implementation for AVT5710.

    The board is controlled via a serial COM port. Transmission parameters: 19200 8N1.
    Commands are sent as ASCII characters and always begin with the start character `Esc`(0x1B)
    and finish with the end character `Enter` (0x0D).

    More information: https://serwis.avt.pl/manuals/AVT5710.pdf

    """

    _SIGNAL_PROPERTY_TYPE = SignalPropertiesAvt5710
    _START_CHR = b"\x1B"  # Esc
    _STOP_CHR = b"\x0D"  # Enter
    _SET_CMD = b"S"
    _RST_CMD = b"C"
    _COM_BAUD_RATE = 19200
    _READ_K_STATE = b"L"
    _READ_ALL_STATE = b"R"

    def __init__(self, com_port: str) -> None:
        super().__init__()
        self._avt_serial = self._init_serial_port(com_port)

    @retry(serial.serialutil.SerialException, tries=5, delay=1)
    def _init_serial_port(self, com_port: str):
        return serial.Serial(com_port, self._COM_BAUD_RATE, timeout=1)

    @retry((RuntimeError, serial.serialutil.SerialException), tries=10, delay=1)
    def _read_states(self) -> int:
        with self._avt_serial as com_port:
            cmd = self._START_CHR + self._READ_ALL_STATE + self._STOP_CHR
            com_port.write(cmd)
            state = com_port.read(4)

        if (
            bytes([state[0]]) == self._READ_ALL_STATE
            and bytes([state[3]]) == self._STOP_CHR
        ):
            state_as_bytes = bytes.fromhex(chr(state[1]) + chr(state[2]))
            return int.from_bytes(state_as_bytes)

        raise RuntimeError(
            f"Can not read relays state of AVT5710 module. Received response: {state}."
        )

    @retry((RuntimeError, serial.serialutil.SerialException), tries=10, delay=1)
    def _set_relay_states(self, relay_no: int, state: bool) -> None:
        relay_no_ascii = ord(str(relay_no)).to_bytes()
        ctrl_cmd = self._SET_CMD if state else self._RST_CMD
        with self._avt_serial as com_port:
            cmd = self._START_CHR + ctrl_cmd + relay_no_ascii + self._STOP_CHR
            try:
                com_port.write(cmd)
            except TimeoutError as e:
                raise RuntimeError(
                    f"Can not write relay: {relay_no} state to AVT5710 module."
                ) from e
            time.sleep(0.1)

    def read_states(self) -> None:
        """Read all signal (relay) states from the AVT5710 board"""
        relays_state = self._read_states()
        self._update_signals_state(relays_state, self._signal_members)

    def _set_relays_states(self) -> None:
        signals_to_update = [
            i for i in self._signal_members if not i.__state__ == i.__new_state__
        ]
        for signal in signals_to_update:
            try:
                relay_no = signal.get_hw_property_by_type(
                    self._SIGNAL_PROPERTY_TYPE
                ).relay_no
            except ValueError:
                continue
            active_state = signal.get_hw_property_by_type(
                self._SIGNAL_PROPERTY_TYPE
            ).active_state
            new_state = (
                signal.__new_state__ if active_state else not signal.__new_state__
            )
            self._set_relay_states(relay_no, new_state)

    def _update_signals_state(
        self, current_status: int, signals: List["Signal"]
    ) -> None:
        for signal in signals:
            try:
                relay_no = signal.get_hw_property_by_type(
                    self._SIGNAL_PROPERTY_TYPE
                ).relay_no
            except ValueError:
                continue
            active_state = signal.get_hw_property_by_type(
                self._SIGNAL_PROPERTY_TYPE
            ).active_state

            state = bool(current_status & (2 ** (relay_no - 1)))
            state = state if active_state else not state
            signal.__state__ = state

    def write_states(self) -> None:
        """Set all updated signal (relay) states to AVT5710 boards."""
        self._set_relays_states()
        self.read_states()
