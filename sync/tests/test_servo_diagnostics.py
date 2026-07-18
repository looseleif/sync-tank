import io
import sys
import types
import unittest
from unittest import mock

from servo_diagnostics import PCA9685ServoDriver, RecordingServoDriver, SafeServoDiagnostic, _LinuxPCA9685Device


class SafeServoDiagnosticTests(unittest.TestCase):
    def test_safe_diagnostic_moves_all_channels_in_small_window(self):
        driver = RecordingServoDriver(channel_count=4)
        diagnostic = SafeServoDiagnostic(driver=driver, channel_count=4)

        result = diagnostic.run(pause_seconds=0.0)

        self.assertEqual(result[0]["angles"], [90, 90, 90, 90])
        self.assertEqual(result[1]["angles"], [100, 100, 100, 100])
        self.assertEqual(result[2]["angles"], [80, 80, 80, 80])
        self.assertEqual(result[3]["angles"], [90, 90, 90, 90])
        self.assertEqual(driver.commands[-1], (3, 90))

    def test_pca9685_driver_supports_legacy_adafruit_module_api(self):
        fake_module = types.ModuleType("Adafruit_PCA9685")

        class FakePCA9685:
            def __init__(self, address=0x40, busnum=None):
                self.address = address
                self.busnum = busnum

            def set_pwm_freq(self, frequency_hz):
                self.frequency_hz = frequency_hz

            def set_pwm(self, channel, on, off):
                self.channel = channel
                self.on = on
                self.off = off

        fake_module.PCA9685 = FakePCA9685

        with mock.patch.dict(
            sys.modules,
            {"Adafruit_PCA9685": fake_module, "adafruit_pca9685": None},
        ):
            driver = PCA9685ServoDriver(channel_count=2, busnum=1, frequency_hz=60)

        self.assertEqual(driver.pwm.pwm.address, 0x40)
        self.assertEqual(driver.pwm.pwm.busnum, 1)
        self.assertEqual(driver.pwm.frequency_hz, 60)

    def test_pca9685_driver_supports_circuitpython_channel_api(self):
        fake_module = types.ModuleType("adafruit_pca9685")

        class FakeChannel:
            def __init__(self):
                self.duty_cycle = None

        class FakePCA9685:
            def __init__(self, i2c, address=0x40):
                self.i2c = i2c
                self.address = address
                self.channels = [FakeChannel() for _ in range(16)]

        fake_module.PCA9685 = FakePCA9685

        fake_busio = types.ModuleType("busio")
        fake_board = types.ModuleType("board")
        fake_board.SCL = object()
        fake_board.SDA = object()
        fake_busio.I2C = lambda scl, sda: (scl, sda)

        with mock.patch.dict(
            sys.modules,
            {"adafruit_pca9685": fake_module, "Adafruit_PCA9685": None, "busio": fake_busio, "board": fake_board},
        ):
            driver = PCA9685ServoDriver(channel_count=2, frequency_hz=50)

        driver.set_angle(1, 90)

        self.assertEqual(driver.pwm.pwm.frequency, 50)
        self.assertEqual(driver.pwm.pwm.channels[1].duty_cycle, 375 << 4)

    def test_linux_pca9685_uses_led_register_offsets(self):
        device = _LinuxPCA9685Device.__new__(_LinuxPCA9685Device)
        device.address = 0x40
        device.busnum = 1
        device.frequency_hz = 50
        written = []

        def fake_write_bytes(register, data):
            written.append((register, list(data)))

        device._write_bytes = fake_write_bytes  # type: ignore[assignment]
        device.set_pwm(3, 0, 2048)

        self.assertEqual(written[0][0], 0x06 + 3 * 4)
        self.assertEqual(written[0][1], [0, 0, 0x00, 0x08])

    def test_linux_pca9685_initialization_uses_standard_sequence(self):
        device = _LinuxPCA9685Device.__new__(_LinuxPCA9685Device)
        device.address = 0x40
        device.busnum = 1
        device.frequency_hz = 50
        device._mode1 = 0x00
        device._prescale = 0xFE
        device._mode2 = 0x01
        device._auto_increment = 0x20
        device._restart = 0x80
        device._sleep = 0x10
        device._outdrv = 0x04
        writes = []
        device._write_byte = lambda register, value: writes.append((register, value))  # type: ignore[assignment]

        device._init_device()

        self.assertEqual(writes[0], (0x00, 0x10))
        self.assertEqual(writes[1], (0xFE, 121))
        self.assertEqual(writes[2], (0x01, 0x04))
        self.assertEqual(writes[3], (0x00, 0x20))
        self.assertEqual(writes[4], (0x00, 0x80 | 0x20))

    def test_single_channel_diagnostic_targets_one_output(self):
        driver = RecordingServoDriver(channel_count=4)
        diagnostic = SafeServoDiagnostic(driver=driver, channel_count=4, channel=2)

        diagnostic.run(pause_seconds=0.0)

        self.assertEqual([cmd[0] for cmd in driver.commands], [2, 2, 2, 2])

    def test_angle_command_sets_one_channel_once(self):
        driver = RecordingServoDriver(channel_count=4)

        with mock.patch("servo_diagnostics.PCA9685ServoDriver", return_value=driver):
            with mock.patch.object(sys, "argv", ["servo_diagnostics.py", "--channel", "2", "--angle", "95"]):
                with mock.patch("sys.stdout", new=io.StringIO()):
                    from servo_diagnostics import main

                    result = main()

        self.assertEqual(result, 0)
        self.assertEqual(driver.commands, [(2, 95)])


if __name__ == "__main__":
    unittest.main()
