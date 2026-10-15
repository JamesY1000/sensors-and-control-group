"""Regression checks for configured poses, reset behaviour and landmark placement."""

import math
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from track_a_mobile_robot.evaluation import read_true_pose
from track_a_mobile_robot.main import load_model
from track_a_mobile_robot.perception import expected_range_bearing, marker_map_from_model
from track_a_mobile_robot.scene_config import SceneConfig, SpawnPose, load_config


class ConfigTests(unittest.TestCase):
    def test_read_custom_config_and_reject_invalid_settings(self):
        valid = "[robot]\nx=0.3\ny=-0.7\nyaw_deg=30\n[charger]\nx=4\ny=0\nyaw_deg=135\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.toml"
            path.write_text(valid)
            config = load_config(path)
            self.assertAlmostEqual(config.robot.pose_world[2], math.pi / 6)
            self.assertEqual(config.charger.x, 4)
            for invalid in (
                valid.replace("yaw_deg=30", "yaw=30"),
                valid.replace("x=0.3", "x=nan"),
                valid.replace("x=0.3", "x=true"),
                valid.replace("[charger]", "[chargre]"),
                valid + "extra=1\n",
            ):
                with self.subTest(invalid=invalid):
                    path.write_text(invalid)
                    with self.assertRaises(ValueError):
                        load_config(path)


class ScenePlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SceneConfig(SpawnPose(0.3, -0.7, 30), SpawnPose(4.0, 0.0, 135))
        cls.model = load_model(config=cls.config)

    def test_robot_pose_and_viewer_reset_use_configuration(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        np.testing.assert_allclose(read_true_pose(data), self.config.robot.pose_world, atol=1e-12)
        data.ctrl[:] = 2
        for _ in range(100):
            mujoco.mj_step(self.model, data)
        self.assertTrue(np.all(np.isfinite(data.qpos)))
        self.assertFalse(any(data.warning.number))
        mujoco.mj_resetData(self.model, data)
        mujoco.mj_forward(self.model, data)
        np.testing.assert_allclose(read_true_pose(data), self.config.robot.pose_world, atol=1e-12)

    def test_marker_moves_and_rotates_with_charger_without_robot_truth(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        marker_map = marker_map_from_model(self.model)
        marker = data.site("aruco_marker_0")
        np.testing.assert_allclose(marker_map[0][:3, 3], marker.xpos, atol=1e-12)
        np.testing.assert_allclose(marker_map[0][:3, :3], marker.xmat.reshape(3, 3), atol=1e-12)
        station = data.body("charging_station")
        np.testing.assert_allclose(station.xpos[:2], [4.0, 0.0], atol=1e-12)
        yaw = math.radians(135)
        np.testing.assert_allclose(marker_map[0][:3, 2], [math.cos(yaw), math.sin(yaw), 0], atol=1e-12)
        x, y, yaw = self.config.robot.pose_world
        dx, dy = marker.xpos[:2] - [x, y]
        expected = [math.hypot(dx, dy), math.atan2(math.sin(math.atan2(dy, dx)-yaw), math.cos(math.atan2(dy, dx)-yaw))]
        np.testing.assert_allclose(expected_range_bearing((x, y, yaw), marker_map=marker_map), expected)
        # Moving the free robot cannot change a surveyed stationary landmark.
        data.qpos[:2] += [0.5, 0.1]
        mujoco.mj_forward(self.model, data)
        np.testing.assert_array_equal(marker_map_from_model(self.model)[0], marker_map[0])

    def test_landmark_on_moving_body_is_rejected(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body><freejoint/><geom type="sphere" size="0.1"/>
          <site name="aruco_marker_0"/></body></worldbody></mujoco>''')
        with self.assertRaises(ValueError):
            marker_map_from_model(model)


if __name__ == "__main__":
    unittest.main()
