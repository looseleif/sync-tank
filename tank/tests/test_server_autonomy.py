from sync_tank.server import _start_configured_autonomy


class FakeArm:
    def __init__(self, driver_status="pca9685"):
        self.driver_status = driver_status
        self._idle_state = {"active": False}
        self.calls = []

    def start_lighthouse_survey(self, device_id, **options):
        self.calls.append(("raydar", device_id, options))
        self._idle_state = {"active": True, "mode": "raydar_step_dwell"}
        return {"status": "survey_started"}

    def start_idle_scan(self, device_id, **options):
        self.calls.append(("reeflex", device_id, options))
        self._idle_state = {"active": True, "mode": "small_arc_circle"}
        return {"status": "idle_started"}


def test_autonomy_reports_raydar_start():
    arm = FakeArm()

    result = _start_configured_autonomy(
        arm,
        {"raydar": {"autostart": True, "device_id": "lighthouse-001", "waypoint_count": 12}},
    )

    assert result == {"status": "started", "mode": "raydar", "result": "survey_started"}
    assert arm.calls[0][0:2] == ("raydar", "lighthouse-001")


def test_autonomy_reports_reeflex_start():
    arm = FakeArm()

    result = _start_configured_autonomy(
        arm,
        {"reeflex": {"autostart": True, "device_id": "reeflex-001", "amplitude": 6}},
    )

    assert result == {"status": "started", "mode": "reeflex", "result": "idle_started"}
    assert arm.calls[0][0:2] == ("reeflex", "reeflex-001")


def test_autonomy_explains_unavailable_hardware():
    arm = FakeArm("mock_pca9685_unavailable: no device")

    result = _start_configured_autonomy(arm, {"raydar": {"autostart": True}})

    assert result["reason"] == "pca9685_unavailable"
    assert arm._idle_state["last_reason"] == "pca9685_unavailable"
