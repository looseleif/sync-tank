#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import List, Optional, Sequence


class ServoDriver:
    def __init__(self, channel_count: int = 16) -> None:
        if channel_count <= 0:
            raise ValueError("channel_count must be positive")
        self.channel_count = channel_count

    def set_angle(self, channel: int, angle: int) -> None:
        raise NotImplementedError

    def set_all(self, angle: int) -> None:
        for channel in range(self.channel_count):
            self.set_angle(channel, angle)

    def set_angles(self, angles: Sequence[int]) -> None:
        if len(angles) != self.channel_count:
            raise ValueError(f"expected {self.channel_count} angles, received {len(angles)}")
        for channel, angle in enumerate(angles):
            self.set_angle(channel, angle)


class RecordingServoDriver(ServoDriver):
    def __init__(self, channel_count: int = 16) -> None:
        super().__init__(channel_count=channel_count)
        self.commands: List[tuple[int, int]] = []

    def set_angle(self, channel: int, angle: int) -> None:
        self.commands.append((channel, angle))


class PCA9685ServoDriver(ServoDriver):
    def __init__(
        self,
        channel_count: int = 16,
        address: int = 0x40,
        busnum: Optional[int] = None,
        frequency_hz: int = 50,
    ) -> None:
        super().__init__(channel_count=channel_count)
        self.address = address
        self.busnum = busnum
        self.frequency_hz = frequency_hz
        self.pwm = None

        module = None
        module_name = None
        for module_name in ("Adafruit_PCA9685", "adafruit_pca9685"):
            try:
                module = __import__(module_name, fromlist=["PCA9685"])
                break
            except ImportError:
                continue

        if module is None:
            raise RuntimeError(
                "PCA9685 support is not installed. Install it with 'pip install adafruit-circuitpython-pca9685'"
            )

        if module_name == "Adafruit_PCA9685":
            self.pwm = _PCA9685Adapter(module.PCA9685(address=address, busnum=busnum), frequency_hz)
            return

        try:
            import busio
            import board

            if busnum is None:
                i2c = busio.I2C(board.SCL, board.SDA)
            else:
                i2c = busio.I2C(getattr(board, f"SCL{busnum}"), getattr(board, f"SDA{busnum}"))
            pwm = module.PCA9685(i2c, address=address)
            self.pwm = _PCA9685Adapter(pwm, frequency_hz)
        except Exception:
            self._init_with_linux_i2c(module)

    def _init_with_linux_i2c(self, module) -> None:
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - environment dependency
            raise RuntimeError("Linux I2C access is not available on this system") from exc

        i2c_bus = self._discover_i2c_bus()
        i2c_path = f"/dev/i2c-{i2c_bus}"
        if not os.path.exists(i2c_path):
            raise RuntimeError(f"I2C device {i2c_path} is not available")

        fd = os.open(i2c_path, os.O_RDWR)
        fcntl.ioctl(fd, 0x0703, self.address)
        os.close(fd)
        self.pwm = _LinuxPCA9685Device(self.address, i2c_bus, frequency_hz=self.frequency_hz)

    def _discover_i2c_bus(self) -> int:
        if self.busnum is not None:
            return self.busnum
        for candidate in range(0, 20):
            path = f"/dev/i2c-{candidate}"
            if os.path.exists(path):
                return candidate
        raise RuntimeError("No accessible I2C bus device was found")

    def _angle_to_pulse(self, angle: int) -> int:
        if not 0 <= angle <= 180:
            raise ValueError("angle must be between 0 and 180 degrees")
        return int(150 + (angle / 180.0) * 450)

    def set_angle(self, channel: int, angle: int) -> None:
        if channel < 0 or channel >= self.channel_count:
            raise ValueError(f"channel {channel} out of range")
        if self.pwm is None:
            raise RuntimeError("PCA9685 driver is not initialized")
        pulse = self._angle_to_pulse(angle)
        self.pwm.set_pwm(channel, 0, pulse)


class _PCA9685Adapter:
    def __init__(self, pwm, frequency_hz: int) -> None:
        self.pwm = pwm
        self.frequency_hz = frequency_hz
        if hasattr(pwm, "set_pwm_freq"):
            pwm.set_pwm_freq(frequency_hz)
        else:
            pwm.frequency = frequency_hz

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        if hasattr(self.pwm, "set_pwm"):
            self.pwm.set_pwm(channel, on, off)
            return
        if on != 0:
            raise ValueError("CircuitPython PCA9685 adapter only supports on=0")
        self.pwm.channels[channel].duty_cycle = off << 4


class _LinuxPCA9685Device:
    def __init__(self, address: int, busnum: int, frequency_hz: int = 50) -> None:
        self.address = address
        self.busnum = busnum
        self.frequency_hz = frequency_hz
        self._mode1 = 0x00
        self._prescale = 0xFE
        self._mode2 = 0x01
        self._sleep = 0x10
        self._auto_increment = 0x20
        self._restart = 0x80
        self._outdrv = 0x04
        self._init_device()

    def _init_device(self) -> None:
        prescale = self._calculate_prescale(self.frequency_hz)
        self._write_byte(self._mode1, self._sleep)
        time.sleep(0.005)
        self._write_byte(self._prescale, prescale)
        self._write_byte(self._mode2, self._outdrv)
        self._write_byte(self._mode1, self._auto_increment)
        time.sleep(0.005)
        self._write_byte(self._mode1, self._restart | self._auto_increment)

    def _calculate_prescale(self, frequency_hz: int) -> int:
        return int(round(25000000.0 / (4096.0 * frequency_hz)) - 1)

    def _write_byte(self, register: int, value: int) -> None:
        fd = os.open(f"/dev/i2c-{self.busnum}", os.O_RDWR)
        try:
            import fcntl

            fcntl.ioctl(fd, 0x0703, self.address)
            os.write(fd, bytes([register, value]))
        finally:
            os.close(fd)

    def set_pwm_freq(self, frequency_hz: int) -> None:
        self.frequency_hz = frequency_hz
        self._init_device()

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        if not 0 <= channel <= 15:
            raise ValueError("channel must be between 0 and 15")
        register = 0x06 + channel * 4
        self._write_bytes(register, [on & 0xFF, (on >> 8) & 0xFF, off & 0xFF, (off >> 8) & 0xFF])

    def _write_bytes(self, register: int, data: bytes | List[int]) -> None:
        fd = os.open(f"/dev/i2c-{self.busnum}", os.O_RDWR)
        try:
            import fcntl

            fcntl.ioctl(fd, 0x0703, self.address)
            os.write(fd, bytes([register, *data]))
        finally:
            os.close(fd)


class SafeServoDiagnostic:
    def __init__(
        self,
        driver: ServoDriver,
        channel_count: int = 16,
        center_angle: int = 90,
        step_angle: int = 10,
        channel: Optional[int] = None,
    ) -> None:
        self.driver = driver
        self.channel_count = channel_count
        self.center_angle = center_angle
        self.step_angle = step_angle
        self.channel = channel

    def run(self, pause_seconds: float = 0.2) -> List[dict]:
        """Run a low-amplitude verification loop across all configured channels."""
        sequence = [
            self.center_angle,
            self.center_angle + self.step_angle,
            self.center_angle - self.step_angle,
            self.center_angle,
        ]
        results: List[dict] = []

        for angle in sequence:
            if self.channel is None:
                self.driver.set_all(angle)
            else:
                self.driver.set_angle(self.channel, angle)
            if pause_seconds > 0:
                time.sleep(pause_seconds)
            results.append({"angle": angle, "angles": [angle] * self.channel_count})

        return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a safe PCA9685 servo verification loop")
    parser.add_argument("--channel-count", type=int, default=16, help="number of servo channels to exercise")
    parser.add_argument("--pause-seconds", type=float, default=0.25, help="pause between each diagnostic step")
    parser.add_argument("--center-angle", type=int, default=90, help="default midpoint angle")
    parser.add_argument("--step-angle", type=int, default=10, help="small angular change for each step")
    parser.add_argument("--i2c-address", type=int, default=0x40, help="PCA9685 I2C address")
    parser.add_argument("--busnum", type=int, default=None, help="optional I2C bus number")
    parser.add_argument("--frequency-hz", type=int, default=50, help="PWM frequency in Hz")
    parser.add_argument("--probe", action="store_true", help="print the computed pulse values for a single channel without sending them")
    parser.add_argument("--probe-channel", type=int, default=None, help="channel to probe or target when running the diagnostic")
    parser.add_argument("--channel", type=int, default=None, help="channel to drive; if omitted, all channels are driven")
    parser.add_argument("--angle", type=int, default=None, help="set the selected channel to one angle instead of running the diagnostic")
    parser.add_argument("--all-channels", action="store_true", help="run the diagnostic for every physical channel in sequence")
    parser.add_argument("--bank-size", type=int, default=4, help="number of channels per bank when using --all-channels")
    parser.add_argument("--bank", type=int, default=None, help="only run channels in the specified bank (0-based)")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        driver = PCA9685ServoDriver(
            channel_count=args.channel_count,
            address=args.i2c_address,
            busnum=args.busnum,
            frequency_hz=args.frequency_hz,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1

    def run_for_channel(channel: int) -> List[dict]:
        diagnostic = SafeServoDiagnostic(
            driver=driver,
            channel_count=args.channel_count,
            center_angle=args.center_angle,
            step_angle=args.step_angle,
            channel=channel,
        )
        return diagnostic.run(pause_seconds=args.pause_seconds)

    if args.angle is not None:
        channel = args.channel if args.channel is not None else args.probe_channel
        if channel is None:
            parser.error("--angle requires --channel")
        driver.set_angle(channel, args.angle)
        print(json.dumps({"channel": channel, "angle": args.angle}, indent=2))
        return 0

    if args.probe:
        probe_channel = args.channel if args.channel is not None else args.probe_channel
        probe_channel = 0 if probe_channel is None else probe_channel
        for angle in (args.center_angle, args.center_angle + args.step_angle, args.center_angle - args.step_angle, args.center_angle):
            pulse = driver._angle_to_pulse(angle)
            print(f"probe ch={probe_channel} angle={angle} pulse={pulse}")
        return 0

    if args.all_channels:
        channels = list(range(args.channel_count))
        if args.bank is not None:
            start = args.bank * args.bank_size
            end = min(start + args.bank_size, args.channel_count)
            channels = list(range(start, end))
        for channel in channels:
            print(f"=== channel {channel} ===")
            results = run_for_channel(channel)
            print(json.dumps(results, indent=2))
        return 0

    channel = args.channel if args.channel is not None else args.probe_channel
    if channel is None:
        diagnostic = SafeServoDiagnostic(
            driver=driver,
            channel_count=args.channel_count,
            center_angle=args.center_angle,
            step_angle=args.step_angle,
        )
        results = diagnostic.run(pause_seconds=args.pause_seconds)
    else:
        results = run_for_channel(channel)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
