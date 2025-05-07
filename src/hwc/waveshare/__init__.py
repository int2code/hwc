"""Provide signals support for `WaveShare Modbus Poe ETH Relay` module.

Example configuration:

.. code-block:: python

    import time

    from hwc.common import Signals, DOSignal
    from hwc.waveshare import (
        SignalEnginWaveShareEthMb, SignalPropertiesWaveShareEthMb)


    class PowerBoard(Signals):
        dut_1_power = DOSignal(
            hardware_properties=[
                SignalPropertiesWaveShareEthMb(relay_no=1, active_state=True),
            ]
        )
        dut_2_power = DOSignal(
            hardware_properties=[
                SignalPropertiesWaveShareEthMb(relay_no=2, active_state=False),
            ]
        )


    engine = SignalEnginWaveShareEthMb('192.168.254.167')

    power_board = PowerBoard(engine)
    power_board.read_states()
    print(power_board.dut_1_power.state)
    print(power_board.dut_2_power.state)

    power_board.dut_1_power.state = True
    power_board.dut_2_power.state = False

    power_board.write_states()
    time.sleep(1)

    print(power_board.dut_1_power.state)
    print(power_board.dut_2_power.state)

    power_board.dut_1_power.state = False
    power_board.dut_2_power.state = True

    power_board.write_states()

    print(power_board.dut_1_power.state)
    print(power_board.dut_2_power.state)


"""

from dataclasses import dataclass
from typing import List, Literal

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import (
    ModbusException,
    ModbusIOException,
)
from pymodbus.framer.base import FramerType
from retry import retry

from hwc.common import SignalProperties, SignalsEngine


@dataclass
class SignalPropertiesWaveShareEthMb(SignalProperties):
    """Signal Properties specific for a `WaveShare Modbus Poe ETH Relay` module board."""

    relay_no: int
    active_state: Literal[True, False]


class SignalEnginWaveShareEthMb(SignalsEngine):
    """Signals engin implementation for WaveShare Modbus Poe ETH Relay.

    More information: https://www.waveshare.com/wiki/Modbus_POE_ETH_Relay

    """

    _SIGNAL_PROPERTY_TYPE = SignalPropertiesWaveShareEthMb

    def __init__(self, host_ip: str, port=4196, slave=1, number_of_relays: int = 30) -> None:
        super().__init__()
        self._slave = slave
        self._number_of_relays = number_of_relays
        self._modbus = ModbusTcpClient(
            host=host_ip,
            port=port,
            framer=FramerType.RTU,
            reconnect_delay=1,
            retries=3,
        )

    @retry(exceptions=ModbusException, tries=3, delay=1)
    def read_states(self) -> None:
        """Read all signal (relay) states from the board."""
        with self._modbus:
            relays_state = self._modbus.read_coils(
                address=0, count=self._number_of_relays, slave=self._slave
            )
        if isinstance(relays_state, ModbusIOException):
            raise relays_state

        self._update_signals_state(relays_state.bits, self._signal_members)

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
            with self._modbus:
                self._modbus.write_coil(
                    address=relay_no - 1, value=new_state, slave=self._slave
                )

    def _update_signals_state(
        self, current_status: List[bool], signals: List["Signal"]
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

            state = current_status[relay_no - 1]
            state = state if active_state else not state
            signal.__state__ = state

    @retry(exceptions=ModbusException, tries=3, delay=1)
    def write_states(self) -> None:
        """Set all updated signal (relay) states to boards."""
        self._set_relays_states()
        self.read_states()
