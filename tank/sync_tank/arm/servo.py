from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from threading import Event, Lock, Thread, current_thread
from time import monotonic, sleep
from typing import Any, Protocol


@dataclass
class ServoConfig:
    id: str
    name: str
    gpio_pin: int | None
    channel: int | None
    min_angle: float
    max_angle: float
    neutral_angle: float
    min_pulse_width: float = 0.0005
    max_pulse_width: float = 0.0025

    @classmethod
    def from_dict(cls, servo_id: str, data: dict[str, Any]) -> "ServoConfig":
        return cls(
            id=servo_id,
            name=str(data.get("name", servo_id)),
            gpio_pin=int(data["gpio_pin"]) if data.get("gpio_pin") is not None else None,
            channel=int(data["channel"]) if data.get("channel") is not None else None,
            min_angle=float(data.get("min_angle", 0)),
            max_angle=float(data.get("max_angle", 180)),
            neutral_angle=float(data.get("neutral_angle", 90)),
            min_pulse_width=float(data.get("min_pulse_width", 0.0005)),
            max_pulse_width=float(data.get("max_pulse_width", 0.0025)),
        )

    def clamp(self, angle: float) -> float:
        return max(self.min_angle, min(self.max_angle, float(angle)))

    def contains(self, angle: float) -> bool:
        return self.min_angle <= float(angle) <= self.max_angle


class ServoDriver(Protocol):
    def set_angle(self, servo_id: str, angle: float) -> None: ...
    def disable(self, servo_id: str) -> None: ...
    def stop_all(self) -> None: ...


class MockServoDriver:
    def __init__(self, servos: dict[str, ServoConfig]):
        self.positions = {servo_id: servo.neutral_angle for servo_id, servo in servos.items()}
        self.disabled: set[str] = set()

    def set_angle(self, servo_id: str, angle: float) -> None:
        self.positions[servo_id] = angle
        self.disabled.discard(servo_id)

    def disable(self, servo_id: str) -> None:
        self.disabled.add(servo_id)

    def stop_all(self) -> None:
        self.disabled.update(self.positions)


class GpioZeroServoDriver:
    def __init__(self, servos: dict[str, ServoConfig]):
        from gpiozero import AngularServo
        from gpiozero.pins.lgpio import LGPIOFactory

        factory = LGPIOFactory()
        self._devices = {
            servo_id: AngularServo(
                int(servo.gpio_pin),
                min_angle=servo.min_angle,
                max_angle=servo.max_angle,
                initial_angle=servo.neutral_angle,
                min_pulse_width=servo.min_pulse_width,
                max_pulse_width=servo.max_pulse_width,
                pin_factory=factory,
            )
            for servo_id, servo in servos.items()
            if servo.gpio_pin is not None
        }

    def set_angle(self, servo_id: str, angle: float) -> None:
        self._devices[servo_id].angle = angle

    def disable(self, servo_id: str) -> None:
        self._devices[servo_id].detach()

    def stop_all(self) -> None:
        for device in self._devices.values():
            device.detach()


class PCA9685ServoDriver:
    def __init__(self, servos: dict[str, ServoConfig], config: dict[str, Any]):
        self.servos = servos
        self.address = int(str(config.get("address", "0x40")), 0)
        self.frequency_hz = int(config.get("frequency_hz", 50))
        self.bus = int(config.get("bus", 1))
        self.min_tick = int(config.get("min_tick", 150))
        self.max_tick = int(config.get("max_tick", 600))
        self._backend_name = ""
        self._backend = self._open_backend()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def _open_backend(self) -> Any:
        try:
            import Adafruit_PCA9685

            pwm = Adafruit_PCA9685.PCA9685(address=self.address, busnum=self.bus)
            pwm.set_pwm_freq(self.frequency_hz)
            self._backend_name = "pca9685_adafruit_legacy"
            return ("legacy", pwm)
        except Exception:
            pass

        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685

            i2c = busio.I2C(board.SCL, board.SDA)
            pca = PCA9685(i2c, address=self.address)
            pca.frequency = self.frequency_hz
            self._backend_name = "pca9685_circuitpython"
            return ("circuitpython", pca)
        except Exception:
            pass

        try:
            return ("linux_i2c", _LinuxI2CPCA9685(self.bus, self.address, self.frequency_hz))
        except Exception as exc:
            raise RuntimeError(f"PCA9685 unavailable at 0x{self.address:02x} on i2c-{self.bus}: {exc}") from exc

    def set_angle(self, servo_id: str, angle: float) -> None:
        servo = self.servos[servo_id]
        if servo.channel is None:
            raise KeyError(f"Servo {servo_id} has no PCA9685 channel configured")
        tick = self._angle_to_tick(angle)
        kind, device = self._backend
        if kind == "legacy":
            device.set_pwm(servo.channel, 0, tick)
        elif kind == "circuitpython":
            device.channels[servo.channel].duty_cycle = int((tick / 4095) * 0xFFFF)
        else:
            device.set_pwm(servo.channel, 0, tick)

    def disable(self, servo_id: str) -> None:
        servo = self.servos[servo_id]
        if servo.channel is None:
            return
        kind, device = self._backend
        if kind == "legacy":
            device.set_pwm(servo.channel, 0, 0)
        elif kind == "circuitpython":
            device.channels[servo.channel].duty_cycle = 0
        else:
            device.set_pwm(servo.channel, 0, 0)

    def stop_all(self) -> None:
        for servo_id in self.servos:
            self.disable(servo_id)

    def _angle_to_tick(self, angle: float) -> int:
        angle = max(0.0, min(180.0, float(angle)))
        return round(self.min_tick + ((self.max_tick - self.min_tick) * (angle / 180.0)))


class _LinuxI2CPCA9685:
    MODE1 = 0x00
    MODE2 = 0x01
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus_number: int, address: int, frequency_hz: int):
        try:
            import smbus2 as smbus_module
        except Exception:
            import smbus as smbus_module

        self.bus = smbus_module.SMBus(bus_number)
        self.address = address
        self._write(self.MODE1, 0x00)
        self._write(self.MODE2, 0x04)
        self.set_pwm_freq(frequency_hz)

    def set_pwm_freq(self, frequency_hz: int) -> None:
        prescale = round(25_000_000.0 / (4096 * int(frequency_hz)) - 1)
        old_mode = self._read(self.MODE1)
        sleep_mode = (old_mode & 0x7F) | 0x10
        self._write(self.MODE1, sleep_mode)
        self._write(self.PRESCALE, prescale)
        self._write(self.MODE1, old_mode)
        sleep(0.005)
        self._write(self.MODE1, old_mode | 0x80)

    def set_pwm(self, channel: int, on_tick: int, off_tick: int) -> None:
        register = self.LED0_ON_L + 4 * int(channel)
        values = [
            int(on_tick) & 0xFF,
            int(on_tick) >> 8,
            int(off_tick) & 0xFF,
            int(off_tick) >> 8,
        ]
        for offset, value in enumerate(values):
            self._write(register + offset, value)

    def _write(self, register: int, value: int) -> None:
        self.bus.write_byte_data(self.address, register, value)

    def _read(self, register: int) -> int:
        return self.bus.read_byte_data(self.address, register)


class ArmController:
    def __init__(self, config: dict[str, Any]):
        self.disable_after_move = bool(config.get("disable_pwm_after_move", False))
        self.movement_delay = float(config.get("movement_delay_seconds", 0.35))
        self.pca9685 = config.get("pca9685") or {}
        self.devices = config.get("devices") or {}
        self.servos = {
            servo_id: ServoConfig.from_dict(servo_id, servo_data)
            for servo_id, servo_data in (config.get("servos") or {}).items()
        }
        self.positions = {servo_id: servo.neutral_angle for servo_id, servo in self.servos.items()}
        self._idle_lock = Lock()
        self._idle_stop = Event()
        self._idle_thread: Thread | None = None
        self._idle_state: dict[str, Any] = {"active": False}
        self.driver_status = "mock"
        self.driver: ServoDriver = MockServoDriver(self.servos)

        backend = str(config.get("backend", "gpio")).lower()
        if backend == "gpio":
            try:
                self.driver = GpioZeroServoDriver(self.servos)
                self.driver_status = "gpio"
            except Exception as exc:
                self.driver_status = f"mock_gpio_unavailable: {exc}"
        elif backend in {"pca9685", "pca"}:
            try:
                self.driver = PCA9685ServoDriver(self.servos, self.pca9685)
                backend_name = getattr(self.driver, "backend_name", "pca9685")
                self.driver_status = backend_name or "pca9685"
            except Exception as exc:
                self.driver_status = f"mock_pca9685_unavailable: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "driver": self.driver_status,
            "address": f"0x{int(str(self.pca9685.get('address', '0x40')), 0):02x}" if self.pca9685 else None,
            "frequency_hz": int(self.pca9685.get("frequency_hz", 50)) if self.pca9685 else None,
            "disable_pwm_after_move": self.disable_after_move,
            "idle": dict(self._idle_state),
            "servos": {
                servo_id: {
                    "id": servo_id,
                    "name": servo.name,
                    "gpio_pin": servo.gpio_pin,
                    "channel": servo.channel,
                    "min_angle": servo.min_angle,
                    "max_angle": servo.max_angle,
                    "neutral_angle": servo.neutral_angle,
                    "angle": self.positions[servo_id],
                }
                for servo_id, servo in self.servos.items()
            },
            "devices": self.devices,
        }

    def set_angle(self, servo_id: str, angle: float, *, reject_out_of_range: bool = False, stop_idle: bool = True) -> dict[str, Any]:
        if stop_idle:
            self.stop_idle(reason="manual_servo")
        if servo_id not in self.servos:
            raise KeyError(f"Unknown servo: {servo_id}")

        servo = self.servos[servo_id]
        if reject_out_of_range and not servo.contains(angle):
            raise ValueError(f"{servo_id} angle {angle} outside safe range {servo.min_angle}-{servo.max_angle}")
        clamped = servo.clamp(angle)
        self.driver.set_angle(servo_id, clamped)
        self.positions[servo_id] = clamped
        if self.disable_after_move:
            sleep(self.movement_delay)
            self.driver.disable(servo_id)
        return {"servo_id": servo_id, "angle": clamped, "requested_angle": float(angle), "clamped": clamped != float(angle)}

    def set_channel(self, channel: int, angle: float, *, device_id: str | None = None, joint: str | None = None) -> dict[str, Any]:
        self.stop_idle(reason="manual_channel")
        for servo_id, servo in self.servos.items():
            if servo.channel == int(channel):
                result = self.set_angle(servo_id, angle, reject_out_of_range=True, stop_idle=False)
                result.update({"channel": int(channel), "device_id": device_id, "joint": joint})
                return result
        raise KeyError(f"Unknown PCA9685 channel: {channel}")

    def set_device_pose(self, device_id: str, pose: dict[str, float]) -> dict[str, Any]:
        self.stop_idle(reason="manual_pose")
        device = self.devices.get(device_id)
        if not device:
            raise KeyError(f"Unknown device: {device_id}")
        joints = device.get("joints") or {}
        applied = {}
        for joint, angle in pose.items():
            servo_id = joints.get(joint)
            if not servo_id:
                raise KeyError(f"Unknown joint for {device_id}: {joint}")
            applied[joint] = self.set_angle(str(servo_id), float(angle), reject_out_of_range=True, stop_idle=False)
        return {"device_id": device_id, "device_type": device.get("type"), "pose": applied, "arm": self.status()}

    def start_idle_scan(
        self,
        device_id: str = "reeflex-001",
        *,
        center: dict[str, float] | None = None,
        amplitude: float = 8.0,
        period_seconds: float = 9.0,
        step_seconds: float = 0.35,
    ) -> dict[str, Any]:
        device = self.devices.get(device_id)
        if not device:
            raise KeyError(f"Unknown device: {device_id}")
        joints = device.get("joints") or {}
        required = ("base", "shoulder", "elbow")
        for joint in required:
            if joint not in joints:
                raise KeyError(f"{device_id} does not expose {joint}")

        amplitude = max(1.0, min(float(amplitude), 18.0))
        period_seconds = max(3.0, float(period_seconds))
        step_seconds = max(0.15, float(step_seconds))
        center = center or {}
        neutral = {
            joint: float(center.get(joint, self.servos[str(joints[joint])].neutral_angle))
            for joint in required
        }

        self.stop_idle(reason="restart_idle")
        self._idle_stop = Event()
        self._idle_state = {
            "active": True,
            "device_id": device_id,
            "mode": "small_arc_circle",
            "amplitude": amplitude,
            "period_seconds": period_seconds,
            "step_seconds": step_seconds,
            "center": neutral,
            "started_at_monotonic": monotonic(),
            "last_reason": "started",
        }
        self._idle_thread = Thread(
            target=self._idle_loop,
            args=(device_id, neutral, amplitude, period_seconds, step_seconds),
            daemon=True,
        )
        self._idle_thread.start()
        return {"status": "idle_started", "idle": dict(self._idle_state), "arm": self.status()}

    def stop_idle(self, reason: str = "stopped") -> dict[str, Any]:
        with self._idle_lock:
            thread = self._idle_thread
            active = bool(thread and thread.is_alive())
            self._idle_stop.set()
            self._idle_state = {**self._idle_state, "active": False, "last_reason": reason}
            self._idle_thread = None
        if active and thread and thread is not current_thread():
            thread.join(timeout=1)
        return {"status": "idle_stopped", "idle": dict(self._idle_state)}

    def _idle_loop(self, device_id: str, center: dict[str, float], amplitude: float, period_seconds: float, step_seconds: float) -> None:
        joints = self.devices[device_id].get("joints") or {}
        start = monotonic()
        while not self._idle_stop.is_set():
            phase = ((monotonic() - start) / period_seconds) * 2 * pi
            pose = {
                "base": center["base"] + amplitude * sin(phase),
                "shoulder": center["shoulder"] + amplitude * cos(phase),
                "elbow": center["elbow"] + (amplitude * 0.55) * sin(phase + (pi / 2)),
            }
            for joint, angle in pose.items():
                servo_id = str(joints[joint])
                try:
                    self.set_angle(servo_id, angle, reject_out_of_range=False, stop_idle=False)
                except Exception:
                    self._idle_stop.set()
                    self._idle_state = {**self._idle_state, "active": False, "last_reason": f"idle_error_{joint}"}
                    break
            self._idle_state = {**self._idle_state, "active": not self._idle_stop.is_set(), "last_pose": pose}
            self._idle_stop.wait(step_seconds)

    def stop(self) -> dict[str, str]:
        self.stop_idle(reason="arm_stop")
        self.driver.stop_all()
        return {"status": "stopped"}
