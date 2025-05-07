"""Hardware control module - dedicated to controlling real hardware through a unified API."""
import time
from typing import List

from ._base import Signal, Signals, SignalProperties, DISignal, DOSignal, SignalsEngine


def signal_power_cycle(relays_board: Signals, do_signal_names: List[str], cutoff_time_s=0.25):
    """Switch off, wait, switch on DO signal by given name.

    :param relays_board: signal instance
    :param do_signal_names: names of the signal on which the power cycle will be performed
    :param cutoff_time_s: time how long the signal will be off

    """
    relays_board.read_states()
    for do_signal_name in do_signal_names:
        signal = getattr(relays_board, do_signal_name)
        signal.state = False
    relays_board.write_states()
    time.sleep(cutoff_time_s)
    for do_signal_name in do_signal_names:
        signal = getattr(relays_board, do_signal_name)
        signal.state = True
    relays_board.write_states()
