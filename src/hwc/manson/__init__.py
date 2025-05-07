"""Hardware control for Manson PSU."""
import serial


class MansonPSU:
    """Control Manson Power Supply Unit.

    :param port: Serial port (e.g., "COM3" or "/dev/ttyUSB0")
    :param baudrate: Communication speed, typically 9600 for Manson PSU
    :param timeout: Timeout for serial communication in seconds

    """
    def __init__(self, port: str, baudrate: int = 9600, timeout: int = 1):
        """Initialize the serial connection to the Manson power supply unit."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = self.init_connection()

    def init_connection(self):
        """Open the serial connection."""
        return serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )

    def disconnect(self):
        """Close the serial connection."""
        if self._serial and self._serial.is_open:
            self._serial.close()

    def send_command(self, command: str) -> str:
        """Send a command to the PSU and return the response.

        :param command: command to send
        :return: response from the PSU
        """
        command = command.strip() + '\r'
        with self._serial as connection:
            connection.write(command.encode('ascii'))
            response = connection.readline().decode('ascii').strip()

        return response

    def set_voltage(self, voltage: float) -> bool:
        """Set the output voltage.

        :param voltage: desired voltage
        :return: PSU response as boolean
        """
        if voltage <= 0 or voltage >=100:
            raise ValueError

        command = f"VOLT{int(voltage * 10):03}"
        response = self.send_command(command)
        return response != ""

    def set_current(self, current: float) -> bool:
        """Set the output current.

        :param current: desired current
        :return: PSU response as boolean
        """
        if current < 0 or current >= 10:
            raise ValueError

        command = f"CURR{int(current * 100):03}"
        response = self.send_command(command)
        return response != ""

    def get_voltage(self) -> float:
        """Get the current output voltage.

        :return: output voltage
        """
        response = self.send_command("GETD")
        if response != '':
            return int(response[0:4]) / float(100)
        return False

    def get_current(self) -> float:
        """Get the current output current.

        :return: output current
        """
        response = self.send_command("GETD")
        if response != '':
            return int(response[4:8]) / float(100)
        return False

    def set_output(self, state: bool) -> bool:
        """Turn the output on or off.

        :param state: True to turn on, False to turn off
        :return: PSU response as boolean
        """
        on, off = "SOUT0", "SOUT1"
        command = on if state else off
        response = self.send_command(command)
        return response != ""
