"""Render the real panel texture to catch mirroring and stale landmark transforms.

Run with MUJOCO_GL=egl on a Linux machine without a display.
"""

import unittest

import mujoco
import numpy as np

from track_a_mobile_robot.main import load_model
from track_a_mobile_robot.perception import Perception, expected_range_bearing
from track_a_mobile_robot.scene_config import SceneConfig, SpawnPose, load_config


class StationCameraTests(unittest.TestCase):
    def test_symmetric_distant_view_never_emits_nonfinite_pose(self):
        dock = load_config().charger
        config = SceneConfig(SpawnPose(dock.x - 2.9, dock.y, 0), dock)
        model = load_model(config=config)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with Perception(model) as perception:
            for observation in perception.update(data).markers:
                self.assertTrue(np.all(np.isfinite(observation.T_camera_marker)))
                self.assertTrue(np.isfinite(observation.reprojection_error_px))

    def test_marker_detects_at_default_and_rotated_station(self):
        dock = load_config().charger
        cases = (
            SceneConfig(SpawnPose(dock.x - 1.2, dock.y, 0), dock),
            SceneConfig(SpawnPose(0.0, -0.2, 90), SpawnPose(0.0, 1.0, -90)),
        )
        for config in cases:
            with self.subTest(config=config):
                model = load_model(config=config)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                with Perception(model) as perception:
                    observations = perception.update(data).markers
                    self.assertEqual([observation.marker_id for observation in observations], [0])
                    observed = observations[0]
                    expected = expected_range_bearing(config.robot.pose_world, marker_map=perception.marker_map)
                    self.assertLess(abs(observed.range_m-expected[0]), 0.025)
                    self.assertLess(abs(observed.bearing_rad-expected[1]), 0.01)
                    self.assertTrue(np.all(np.isfinite(observed.T_base_marker)))


if __name__ == "__main__":
    unittest.main()
