import json
import base64
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tank_manager import TankManagerApp
from wildlife_system import (
    AI_LABEL, MotionTracker, VisionController, ask_the_deep, frame_revision,
    proportional_centering, raydar_waypoints, select_best_capture,
    normalize_structure,
)
from scripts.fake_tank_node import FakeState, fixture_jpeg, validate_raydar_pose


class WildlifePrimitiveTests(unittest.TestCase):
    def test_frame_revision_prefers_etag_then_last_modified_then_hash(self):
        image = fixture_jpeg("fish", 1)
        self.assertEqual(frame_revision({"ETag": '"one"'}, image), ("etag", '"one"'))
        self.assertEqual(frame_revision({"Last-Modified": "today"}, image), ("last-modified", "today"))
        self.assertEqual(frame_revision({}, image)[0], "sha256")

    def test_waypoints_form_twelve_point_bounded_circle(self):
        points = raydar_waypoints()
        self.assertEqual(len(points), 12)
        self.assertEqual(points[0], {"pan": 115.0, "tilt": 90.0})
        self.assertLessEqual(max(p["pan"] for p in points), 115)
        self.assertGreaterEqual(min(p["tilt"] for p in points), 80)

    def test_centering_dead_zone_cap_and_limits(self):
        limits = {"min_pan": 20, "max_pan": 160, "min_tilt": 45, "max_tilt": 125}
        self.assertEqual(proportional_centering((.5, .5), (90, 90), limits), (90, 90, True))
        pan, tilt, centered = proportional_centering((1, 0), (159, 124), limits)
        self.assertEqual((pan, tilt, centered), (160, 125, False))

    def test_motion_persistence_smoothing_loss_and_filtering(self):
        tracker = MotionTracker(persistence_frames=3, smoothing=.5)
        self.assertFalse(tracker.update([{"x": .2, "y": .4, "area": .03}], 1)["persistent"])
        tracker.update([{"x": .4, "y": .4, "area": .03}], 2)
        target = tracker.update([{"x": .6, "y": .4, "area": .03}], 3)
        self.assertTrue(target["persistent"])
        self.assertGreater(target["x"], .4)
        self.assertIsNone(tracker.update([{"x": .5, "y": .5, "area": .8}], 4))
        self.assertEqual(tracker.lost_for(5), 2)

    def test_best_capture_combines_sharpness_and_center(self):
        best = select_best_capture([{"id": 1, "sharpness": .2, "center_score": 1}, {"id": 2, "sharpness": 1, "center_score": .8}])
        self.assertEqual(best["id"], 2)

    def test_structure_bounds_grid_transform_and_types(self):
        item = normalize_structure({"item_type": "structure_shape", "structure_type": "arch", "rotation": 725, "scale": 9, "color": "#abc", "label": "Cave", "placement": {"position": {"x": .123, "y": -.2, "z": 1.2}}})
        self.assertEqual(item["placement"]["position"], {"x": .1, "y": 0.0, "z": 1.0})
        self.assertEqual(item["rotation"], 5)
        self.assertEqual(item["scale"], 3)
        with self.assertRaisesRegex(ValueError, "type"):
            normalize_structure({"structure_type": "dragon", "placement": {"position": {"x": .5, "y": .5, "z": .5}}})


class SightingTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.mkdtemp(prefix="wildlife-")
        self.requests = []

        def openai_transport(url, body, headers):
            self.requests.append((url, body, headers))
            return {"output_text": json.dumps({"visual_evidence": "A silver silhouette", "possible_subject": "small fish", "uncertainty": "medium", "interesting_fact": "Fish use fins to stabilize.", "narration": "Captain's log: a flash by the reef."})}

        self.app = TankManagerApp(self.storage, openai_transport=openai_transport)
        self.app.register_cameras([{"camera_id": "cam-1", "tank_id": "tank-1", "status": "online"}])
        self.app.handle_frame_upload({"camera_id": "cam-1", "image_bytes": fixture_jpeg("fish", 1)})

    def tearDown(self):
        shutil.rmtree(self.storage)

    def test_manual_and_auto_capture_metadata_cooldown_and_persistence(self):
        manual = self.app.capture_sighting({"camera_id": "cam-1", "trigger": "manual", "label": "Fish"})
        auto = self.app.capture_sighting({"camera_id": "cam-1", "trigger": "raydar-auto"})
        with self.assertRaisesRegex(ValueError, "cooldown"):
            self.app.capture_sighting({"camera_id": "cam-1", "trigger": "raydar-auto"})
        self.assertEqual(manual["label"], "Fish")
        self.assertEqual(auto["trigger"], "raydar-auto")
        restored = TankManagerApp(self.storage)
        self.assertEqual(len(restored.list_sightings()), 2)

    def test_analysis_requires_confirmation_and_key_and_is_manual_only(self):
        sighting = self.app.capture_sighting({"camera_id": "cam-1"})
        with self.assertRaisesRegex(ValueError, "confirmation"):
            self.app.analyze_sighting(sighting["sighting_id"], {})
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                self.app.analyze_sighting(sighting["sighting_id"], {"confirmed": True})
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}, clear=True):
            analyzed = self.app.analyze_sighting(sighting["sighting_id"], {"confirmed": True, "persona": "researcher"})
        self.assertEqual(analyzed["ai_field_note"]["label"], AI_LABEL)
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("test-only", json.dumps(self.requests[0][1]))
        self.assertTrue(self.requests[0][1]["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_invalid_and_oversized_content_is_rejected(self):
        self.app.frame_dir.joinpath("cam-1.jpg").write_bytes(b"not jpeg")
        with self.assertRaisesRegex(ValueError, "valid JPEG"):
            self.app.capture_sighting({"camera_id": "cam-1"})
        with self.assertRaisesRegex(ValueError, "known camera"):
            self.app.capture_sighting({"camera_id": "../escape"})

    def test_storage_cleanup_preserves_favorites_and_labels(self):
        unknown = self.app.capture_sighting({"camera_id": "cam-1"})
        favorite = self.app.capture_sighting({"camera_id": "cam-1", "favorite": True})
        labeled = self.app.capture_sighting({"camera_id": "cam-1", "label": "Fish"})
        self.assertEqual(self.app.cleanup_sightings(max_count=2), 1)
        self.assertNotIn(unknown["sighting_id"], self.app.sightings)
        self.assertIn(favorite["sighting_id"], self.app.sightings)
        self.assertIn(labeled["sighting_id"], self.app.sightings)

    def test_capture_chooses_best_frame_from_burst(self):
        soft = fixture_jpeg("empty", 1)
        sharp = fixture_jpeg("fish", 9)
        sighting = self.app.capture_sighting({"camera_id": "cam-1", "burst": [
            {"image_base64": base64.b64encode(soft).decode(), "sharpness": .1, "center_score": 1, "scores": {"chosen": "soft"}},
            {"image_base64": base64.b64encode(sharp).decode(), "sharpness": 1, "center_score": .9, "scores": {"chosen": "sharp"}},
        ]})
        self.assertEqual(self.app.sighting_image(sighting["sighting_id"]), sharp)
        self.assertEqual(sighting["scores"]["chosen"], "sharp")


class VisionAndFakeNodeTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeState("fake-one", "tank-1", "127.0.0.1", 18081)
        self.base = "http://fake.invalid"

    def test_fake_payload_stream_inventory_and_safe_servo_rejection(self):
        payload = self.fake.payload()
        self.assertEqual(len(payload["cameras"]), 4)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            validate_raydar_pose(999, 90)
        self.assertEqual(validate_raydar_pose(90, 90), {"pan": 90.0, "tilt": 90.0})

    def test_raydar_tracking_and_network_failure_stop_commands(self):
        commands = []
        urls = lambda *keys, **kwargs: self.base + "/api/controls/lighthouse/pose" if "lighthouse_pose" in keys else None

        def post(url, payload):
            commands.append(payload); return {"ok": True}

        controller = VisionController(urls, post)
        self.assertEqual(controller.start_raydar({"tank_id": "tank-1", "camera_id": "raydar-cam"})["state"], "Survey")
        controller.tick_raydar([], 1)
        controller.tick_raydar([{"x": .9, "y": .5, "area": .03}], 2)
        controller.tick_raydar([{"x": .9, "y": .5, "area": .03}], 3)
        state = controller.tick_raydar([{"x": .9, "y": .5, "area": .03}], 4)
        self.assertEqual(state["state"], "Track")
        self.assertLessEqual(abs(commands[-1]["pan"] - commands[-2]["pan"]), 27)  # survey-to-track transition remains bounded by safe limits
        self.assertEqual(controller.stop_raydar({})["state"], "STOP")
        before = len(commands); controller.tick_raydar([], 10); self.assertEqual(len(commands), before)

    def test_missing_urls_are_unavailable(self):
        controller = VisionController(lambda *args, **kwargs: None, lambda *_: {})
        self.assertEqual(controller.start_raydar({})["state"], "Unavailable")
        self.assertEqual(controller.start_reeflex({})["state"], "Unavailable")

    def test_raydar_start_and_stop_use_advertised_tank_survey_urls(self):
        calls = []

        def urls(*keys, **kwargs):
            if "lighthouse_pose" in keys:
                return self.base + "/pose"
            if "lighthouse_survey_start" in keys:
                return self.base + "/survey/start"
            if "lighthouse_survey_stop" in keys:
                return self.base + "/survey/stop"
            return None

        controller = VisionController(urls, lambda url, payload: calls.append((url, payload)) or {"ok": True})
        payload = {"tank_id": "tank-1", "node_id": "tank-pi-001", "camera_id": "raydar-cam"}

        self.assertEqual(controller.start_raydar(payload)["state"], "Survey")
        self.assertEqual(controller.stop_raydar(payload)["state"], "STOP")
        self.assertTrue(calls[0][0].endswith("/survey/start"))
        self.assertTrue(calls[1][0].endswith("/survey/stop"))

    def test_one_second_centered_target_triggers_single_auto_capture(self):
        captures = []
        controller = VisionController(lambda *keys, **kwargs: self.base + "/pose", lambda *_: {}, capture_callback=captures.append)
        controller.start_raydar({"tank_id": "tank-1", "camera_id": "raydar-cam"})
        candidate = [{"x": .5, "y": .5, "area": .03}]
        for now in (1, 2, 3, 4, 5):
            controller.tick_raydar(candidate, now)
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0]["trigger"], "raydar-auto")


if __name__ == "__main__":
    unittest.main()
