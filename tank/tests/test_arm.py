import unittest
from time import sleep

from sync_tank.arm.servo import ArmController


class TestArm(unittest.TestCase):
    def test_servo_angle_is_clamped(self):
        arm = ArmController(
            {
                "backend": "mock",
                "servos": {
                    "servo_1": {
                        "name": "Base",
                        "gpio_pin": 17,
                        "min_angle": 10,
                        "max_angle": 120,
                        "neutral_angle": 60,
                    }
                },
            }
        )

        result = arm.set_angle("servo_1", 999)

        assert result["angle"] == 120
        assert arm.status()["servos"]["servo_1"]["angle"] == 120


def test_servo_angle_is_clamped():
    arm = ArmController(
        {
            "backend": "mock",
            "servos": {
                "servo_1": {
                    "name": "Base",
                    "gpio_pin": 17,
                    "min_angle": 10,
                    "max_angle": 120,
                    "neutral_angle": 60,
                }
            },
        }
    )

    result = arm.set_angle("servo_1", 999)

    assert result["angle"] == 120
    assert arm.status()["servos"]["servo_1"]["angle"] == 120


def test_strict_servo_angle_rejects_out_of_range():
    arm = ArmController(
        {
            "backend": "mock",
            "servos": {
                "reeflex_base": {
                    "name": "REEFLEX Base",
                    "channel": 2,
                    "min_angle": 20,
                    "max_angle": 160,
                    "neutral_angle": 90,
                }
            },
        }
    )

    try:
        arm.set_angle("reeflex_base", 999, reject_out_of_range=True)
    except ValueError as exc:
        assert "outside safe range" in str(exc)
    else:
        raise AssertionError("Expected out-of-range angle to be rejected")


def test_device_pose_maps_semantic_joint_to_channel_servo():
    arm = ArmController(
        {
            "backend": "mock",
            "servos": {
                "lighthouse_pan": {
                    "name": "Lighthouse Pan",
                    "channel": 0,
                    "min_angle": 20,
                    "max_angle": 160,
                    "neutral_angle": 90,
                },
                "lighthouse_tilt": {
                    "name": "Lighthouse Tilt",
                    "channel": 1,
                    "min_angle": 45,
                    "max_angle": 125,
                    "neutral_angle": 90,
                },
            },
            "devices": {
                "lighthouse-001": {
                    "type": "lighthouse",
                    "joints": {"pan": "lighthouse_pan", "tilt": "lighthouse_tilt"},
                }
            },
        }
    )

    result = arm.set_device_pose("lighthouse-001", {"pan": 100, "tilt": 80})

    assert result["device_type"] == "lighthouse"
    assert arm.status()["servos"]["lighthouse_pan"]["angle"] == 100
    assert arm.status()["servos"]["lighthouse_tilt"]["angle"] == 80


def test_channel_command_maps_to_configured_servo():
    arm = ArmController(
        {
            "backend": "mock",
            "servos": {
                "reeflex_base": {
                    "name": "REEFLEX Base",
                    "channel": 2,
                    "min_angle": 20,
                    "max_angle": 160,
                    "neutral_angle": 90,
                }
            },
        }
    )

    result = arm.set_channel(2, 95, device_id="reeflex-001", joint="base")

    assert result["servo_id"] == "reeflex_base"
    assert result["channel"] == 2
    assert result["angle"] == 95


def test_raydar_survey_uses_safe_step_and_dwell_waypoints():
    arm = ArmController(
        {
            "backend": "mock",
            "servos": {
                "lighthouse_pan": {"channel": 1, "min_angle": 20, "max_angle": 160, "neutral_angle": 90},
                "lighthouse_tilt": {"channel": 0, "min_angle": 45, "max_angle": 125, "neutral_angle": 90},
            },
            "devices": {
                "lighthouse-001": {
                    "type": "lighthouse",
                    "joints": {"pan": "lighthouse_pan", "tilt": "lighthouse_tilt"},
                }
            },
        }
    )

    result = arm.start_lighthouse_survey(
        pan_amplitude=99,
        tilt_amplitude=99,
        dwell_seconds=0.5,
        waypoint_count=99,
    )
    sleep(0.02)
    status = arm.status()
    arm.stop_idle(reason="test_complete")

    assert result["status"] == "survey_started"
    assert status["idle"]["mode"] == "raydar_step_dwell"
    assert status["idle"]["pan_amplitude"] == 25
    assert status["idle"]["tilt_amplitude"] == 10
    assert status["idle"]["waypoint_count"] == 24
    assert 20 <= status["servos"]["lighthouse_pan"]["angle"] <= 160
    assert 45 <= status["servos"]["lighthouse_tilt"]["angle"] <= 125
