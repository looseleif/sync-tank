import shutil
import tempfile
import unittest
from pathlib import Path

from tank_manager import TankManagerApp, TankManagerHandler


class DashboardContractTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.mkdtemp(prefix="dashboard-contract-")
        self.app = TankManagerApp(self.storage)
        # index_html is self-contained and does not depend on a live socket.
        self.html = TankManagerHandler.index_html(object())
        static = Path(__file__).resolve().parents[1] / "static"
        self.js = (static / "app.js").read_text()
        self.css = (static / "app.css").read_text()

    def tearDown(self):
        shutil.rmtree(self.storage)

    def test_portrait_cctv_controls_and_two_tank_labels_exist(self):
        self.assertIn(">SEE SEA TV<", self.html)
        self.assertIn('id="feed-previous"', self.html)
        self.assertIn('id="feed-next"', self.html)
        self.assertIn('id="feed-pin"', self.html)
        self.assertIn('id="sighting-shutter"', self.html)
        self.assertIn('tank-name-one', self.html)
        self.assertIn('tank-name-two', self.html)
        for direction in ("FRONT", "BACK", "LEFT", "RIGHT"):
            self.assertIn(f">{direction}<", self.html)
        self.assertIn("minmax(0, 44fr) minmax(0, 56fr)", self.css)

    def test_shapes_sightings_and_manual_disclosure_exist(self):
        for shape in ("block", "slab", "rounded-rock", "pillar", "arch", "mound"):
            self.assertIn(f'value="{shape}"', self.html)
        self.assertIn("Sends this captured image to OpenAI for analysis", self.html)
        self.assertIn("deepDialog.showModal()", self.js)
        self.assertIn("confirmed: true", self.js)

    def test_no_side_floater_overlays_or_deferred_hardware_copy(self):
        self.assertNotIn("floater-side-left", self.html)
        self.assertNotIn("floater-side-right", self.html)
        blocked_term = "feed" + "er"
        self.assertNotIn(blocked_term, (self.html + self.js).lower())

    def test_eight_second_cross_tank_rotation_and_states(self):
        self.assertIn("}, 8000);", self.js)
        self.assertIn("state.layout.cameras || []", self.js)
        for state_name in ("Survey", "Track", "Manual", "Unavailable", "STOP"):
            self.assertIn(state_name, Path(__file__).resolve().parents[1].joinpath("wildlife_system.py").read_text())


if __name__ == "__main__":
    unittest.main()
