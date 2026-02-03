"""Provide base classes and types for hardware signal representation."""

from abc import ABC, abstractmethod
from random import seed
from typing import List, TypeVar


class SignalsEngine(ABC):
    """Abstract class for Signals' Engine."""

    def __init__(self):
        self._signal_members = None

    def associate_signal_members(self, signal_members: List["Signal"]):
        """Assign signal members to engine instance.

        :param signal_members: members of signals class

        """
        self._signal_members = signal_members
        for signal in self._signal_members:
            signal.__read_states__ = self.read_states
            signal.__write_states__ = self.write_states

    @abstractmethod
    def read_states(self) -> None:
        """Read all signal states using given engine."""

    @abstractmethod
    def write_states(self) -> None:
        """Set all updated signal states using given engine."""


SignalsEnginBase = TypeVar("SignalsEnginBase", bound=SignalsEngine)


class SignalProperties:  # pylint: disable=too-few-public-methods
    """Signal Properties base type."""


SignalPropertiesBase = TypeVar("SignalPropertiesBase", bound=SignalProperties)


class Signal:
    """Base Signal representation."""

    def __init__(
        self,
        hardware_properties: List[SignalPropertiesBase],
        immediate_update: bool = False,
    ):
        self.immediate_update = immediate_update
        self._hardware_properties = hardware_properties
        self._state = None
        self._new_state = None
        self.__name__ = ""
        self.__read_states__ = lambda: None
        self.__write_states__ = lambda: None

    def __str__(self) -> str:
        return f"[{self.__class__.__name__}] {self.__name__}.state: {self._state}"

    def get_hw_property_by_type(
        self, hw_property_type: type["SignalProperties"]
    ) -> "SignalProperties":
        """Return hardware device property by given property type.

        :param hw_property_type: type of properties to be returned
        :returns: hardware device signal property

        """
        for hw_property in self._hardware_properties:
            if isinstance(hw_property, hw_property_type):
                return hw_property
        raise ValueError(
            f"There is no `{hw_property_type}` defined in hardware_properties."
        )

    @property
    def __state__(self):
        return self._state

    @__state__.setter
    def __state__(self, value):
        self._state = self._new_state = value

    @property
    def __new_state__(self):
        return self._new_state

    @__new_state__.setter
    def __new_state__(self, value):
        self._new_state = value


class Signals:
    """Base Signals container."""

    def __init__(self, engine: SignalsEngine) -> None:
        self.__signal_members__: List[Signal] = []
        self._get_signal_members()
        self._engine = engine
        self._engine.associate_signal_members(self.__signal_members__)
        self._next_index = 0

    def _get_signal_members(self) -> None:
        for member_name in dir(self.__class__):
            member = getattr(self, member_name)
            if isinstance(member, Signal):
                member.__name__ = member_name
                self.__signal_members__.append(member)

    def read_states(self) -> None:
        """Read all signal states using given engine."""
        self._engine.read_states()

    def write_states(self) -> None:
        """Set all updated signal states using given engine."""
        self._engine.write_states()

    def set_engine(self, engine: SignalsEngine) -> None:
        """Set engine instance.
        :param engine: engine instance to be set"""
        self._engine = engine
        self._engine.associate_signal_members(self.__signal_members__)

    def __iter__(self):
        self._next_index = 0
        return self

    def __next__(self):
        if self._next_index < len(self.__signal_members__):
            value = self.__signal_members__[self._next_index]
            self._next_index += 1
            return value
        raise StopIteration

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DISignal(Signal):
    """Digital input signal representation."""

    @property
    def state(self):
        """State of Digital Input signal."""
        if self.immediate_update:
            self.__read_states__()
        return bool(self.__state__)


class DOSignal(Signal):
    """Digital output signal representation."""

    @property
    def state(self) -> bool:
        """State of Digital Output signal."""
        if self.immediate_update:
            self.__read_states__()
        if not self.__state__ == self.__new_state__:
            raise RuntimeError("A new state has not been sent to the device.")
        return bool(self.__state__)

    @state.setter
    def state(self, value: bool) -> None:
        """Set state of Digital Output signal.

        :param value: new value of digital output signal

        """
        self.__new_state__ = value
        if self.immediate_update:
            self.__write_states__()


class AOSignal(Signal):
    """Digital output signal representation."""

    @property
    def value(self) -> float:
        """State of Digital Output signal."""
        if self.immediate_update:
            self.__read_states__()
        if not self.__state__ == self.__new_state__:
            raise RuntimeError("A new state has not been sent to the device.")
        return self.__state__

    @value.setter
    def value(self, value: float) -> None:
        """Set state of Digital Output signal.

        :param value: new value of digital output signal

        """
        self.__new_state__ = value
        if self.immediate_update:
            self.__write_states__()
