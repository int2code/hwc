"""Waveshare Modbus RTU 8-Channel Analog Output (DAC) Engine.

This module provides an interface to control the Waveshare 8-Channel DAC B
using the Modbus RTU protocol over a serial connection.
https://www.waveshare.com/wiki/Modbus_RTU_Analog_Output_8CH

.. code-block:: python

    from hwc.common import Signals
    from hwc.common._base import AOSignal
    from hwc.waveshare.analog_output import (
        SignalPropertiesWSAO8Ch,
        SignalEnginWSAO8Ch,
    )


    class AOSignals(Signals):
        '''Collection of Analog Output Signals for Waveshare 8 Channel AO Device.'''

        ao1 = AOSignal(
            hardware_properties=[SignalPropertiesWSAO8Ch(device_id=1, channel_no=1)],
            immediate_update=True,
        )

        ao2 = AOSignal(
            hardware_properties=[SignalPropertiesWSAO8Ch(device_id=1, channel_no=2)],
            immediate_update=True,
        )


    engine = SignalEnginWSAO8Ch(com_port="COM3")


    ao_signals = AOSignals(engine)

    # if immediate_update is True, there is no need to call
    # `ao_signals.write_states()` and `ao_signals.read_states()`
    print(ao_signals.ao1.state)
    ao_signals.ao1.state = 1.5
    print(ao_signals.ao1.state)
    ao_signals.ao2.state = 2.5
    print(ao_signals.ao2.state)

"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pymodbus import FramerType, ModbusException
from pymodbus.client.serial import ModbusSerialClient
from retry import retry

from ..common import SignalProperties, SignalsEngine


@dataclass
class SignalPropertiesWSAO8Ch(SignalProperties):
    """Signal Properties specific for a `WaveShare Modbus Poe ETH Relay` module board."""

    device_id: int
    channel_no: int
    symbolic_range: Tuple[float, float] = (0.0, 10.0)
    physical_range: Tuple[int, int] = (0, 10000)

    def encode_value(self, symbolic_value: float) -> int:
        """
        Encodes a symbolic value into a physical DAC value.

        :param symbolic_value: The symbolic value to encode.
        :return: The encoded physical DAC value.
        :raises ValueError: If the symbolic value is out of the allowable range.
        """
        physical_min, physical_max = self.physical_range
        symbolic_min, symbolic_max = self.symbolic_range

        if not symbolic_min <= symbolic_value <= symbolic_max:
            raise ValueError(
                f"Symbolic value {symbolic_value} "
                f"out of range {symbolic_min}-{symbolic_max}"
            )

        physical_value = (
            (symbolic_value - symbolic_min)
            * (physical_max - physical_min)
            / (symbolic_max - symbolic_min)
        ) + physical_min
        return int(round(physical_value))

    def decode_value(self, physical_value: int) -> float:
        """
        Decodes a physical DAC value into a symbolic value.

        :param physical_value: The physical DAC value to decode.
        :return: The decoded symbolic value.
        :raises ValueError: If the physical value is out of the allowable range.
        """
        physical_min, physical_max = self.physical_range
        symbolic_min, symbolic_max = self.symbolic_range

        if not physical_min <= physical_value <= physical_max:
            raise ValueError(
                f"Physical value {physical_value} "
                f"out of range {physical_min}-{physical_max}"
            )

        symbolic_value = (
            (physical_value - physical_min)
            * (symbolic_max - symbolic_min)
            / (physical_max - physical_min)
        ) + symbolic_min
        return symbolic_value


class SignalEnginWSAO8Ch(SignalsEngine):
    """DAC Engine for Waveshare 8-Channel DAC.
    This class provides an interface to control the Waveshare 8-Channel DAC B
    using the Modbus RTU protocol over a serial connection.

    https://www.waveshare.com/wiki/Modbus_RTU_Analog_Output_8CH

    """

    _SIGNAL_PROPERTY_TYPE = SignalPropertiesWSAO8Ch

    _CHANNELS_NUMBER = 8  # Number of DAC channels
    _BAUDRATE = 9600  # Serial communication baud rate
    _STARTING_ADDRESS = 0x0000  # Starting address for DAC channels

    def __init__(self, com_port: str):
        """
        Initializes the DAC engine.

        :param com_port: The COM port to which the DAC is connected.
        """
        super().__init__()
        self._values: Dict[int, List[Optional[int]]] = defaultdict(
            lambda: [None] * self._CHANNELS_NUMBER
        )
        self._com_port = com_port
        self._modbus = ModbusSerialClient(
            framer=FramerType.RTU,
            port=self._com_port,
            baudrate=self._BAUDRATE,
            timeout=1,
        )

    @retry(exceptions=ModbusException, tries=3, delay=1)
    def read_states(self) -> None:
        """Read all signal (relay) states from the board."""
        self._update_signals_state()

    def _set_relays_states(self) -> None:
        signals_to_update = defaultdict(lambda: [0] * self._CHANNELS_NUMBER)
        for member in self._signal_members:
            properties = member.get_hw_property_by_type(self._SIGNAL_PROPERTY_TYPE)
            idx = properties.channel_no - 1
            signals_to_update[properties.device_id][idx] = properties.encode_value(
                member.__new_state__
            )

        with self._modbus:
            for device_id, states in signals_to_update.items():
                response = self._modbus.write_registers(
                    address=self._STARTING_ADDRESS,
                    values=states,
                    device_id=device_id,
                )
                self._update_state_on_response(device_id, response)

    def _update_signals_state(
        self,
    ) -> None:
        dev_ids = set()
        for member in self._signal_members:
            properties = member.get_hw_property_by_type(self._SIGNAL_PROPERTY_TYPE)
            dev_ids.add(properties.device_id)

        with self._modbus:
            for device_id in dev_ids:
                response = self._modbus.read_holding_registers(
                    address=self._STARTING_ADDRESS,
                    count=self._CHANNELS_NUMBER,
                    device_id=device_id,
                )
                self._update_state_on_response(device_id, response)

    def _update_state_on_response(self, device_id, response):
        for i, value in enumerate(response.registers):
            for member in self._signal_members:
                properties = member.get_hw_property_by_type(self._SIGNAL_PROPERTY_TYPE)
                if properties.device_id == device_id and properties.channel_no == i + 1:
                    member.__state__ = properties.decode_value(value)

    @retry(exceptions=ModbusException, tries=3, delay=1)
    def write_states(self) -> None:
        """Set all updated signal (relay) states to boards."""
        self._set_relays_states()
