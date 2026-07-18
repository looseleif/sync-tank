import unittest
from unittest import mock

from scripts.feed_observer_agent import FeedObserver


class FeedObserverTests(unittest.TestCase):
    def test_forward_to_deep_link_uses_display_urls(self):
        observer = FeedObserver(
            "http://127.0.0.1:8765",
            "/tmp/sync-tank-observer-test",
            threshold=0.045,
            burst_frames=5,
            deep_link_base="http://REDACTED_PRIVATE_IP:8765",
            display_base="http://SYNC_WIRED_IP:8765",
        )
        sent = []

        def fake_post(url, payload, timeout_seconds=3.0):
            sent.append((url, payload))
            return True

        with mock.patch("scripts.feed_observer_agent.post_json", side_effect=fake_post):
            ok = observer.forward_to_deep_link(
                {
                    "observation_id": "obs-1",
                    "camera_id": "usb_0",
                    "frame_urls": ["/observer_events/obs-1/frame-1.jpg"],
                    "event_type": "motion",
                    "focus_region": {"x": 0.625, "y": 0.375, "width": 0.2, "height": 0.2, "confidence": 0.7},
                },
                {"camera_id": "usb_0", "camera_type": "endoscope_cam", "stream_url": "http://edge/stream"},
            )

        self.assertTrue(ok)
        self.assertEqual(sent[0][0], "http://REDACTED_PRIVATE_IP:8765/api/observations/register")
        self.assertEqual(sent[0][1]["frame_urls"][0], "http://SYNC_WIRED_IP:8765/observer_events/obs-1/frame-1.jpg")
        self.assertTrue(sent[0][1]["classifier_request"]["requested"])
        self.assertEqual(sent[0][1]["classifier_request"]["focus_region"]["x"], 0.625)


if __name__ == "__main__":
    unittest.main()
