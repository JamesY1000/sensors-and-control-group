"""Navigation consumes the configured scene's scans and fixed landmark map."""

import math
import unittest

import mujoco
import numpy as np

from track_a_mobile_robot.control import LQIController, velocity_command_to_wheel_speeds
from track_a_mobile_robot.estimation import StateEstimate
from track_a_mobile_robot.main import load_model
from track_a_mobile_robot.navigation import Navigator
from track_a_mobile_robot.perception import (
    MarkerObservation,
    PerceptionOutput,
    lidar_points_in_base,
    marker_map_from_model,
)
from track_a_mobile_robot.scene_config import SceneConfig, SpawnPose
from track_a_mobile_robot.sensing import CameraReading, read_lidar


def scan_output(model, *, markers=None, time=0.0):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scan = read_lidar(data)
    return PerceptionOutput(
        time=time,
        lidar=scan,
        lidar_points_base=lidar_points_in_base(scan),
        camera=CameraReading(time=time, rgb=np.zeros((1, 1, 3), dtype=np.uint8)),
        markers=[] if markers is None else markers,
        new_camera_frame=markers is not None,
    )


class NavigationIntegrationTests(unittest.TestCase):
    def test_default_corner_scan_locates_axis_aligned_room_and_search_goals(self):
        config = SceneConfig(SpawnPose(-0.5, -2.5, 45.0), SpawnPose(6.9, 0.0, 180.0))
        model = load_model(config=config)
        navigator = Navigator(marker_map=marker_map_from_model(model), verbose=False)
        state = StateEstimate(0.0, config.robot.pose_world, np.zeros((3, 3)))
        reference = navigator.trajectory_reference(state, scan_output(model))

        # The current corner spawn faces 45 degrees; the room remains aligned
        # with world X, with goals 0.85 m inside its x=-1 and x=7 walls.
        np.testing.assert_allclose(navigator._room.axis, [1.0, 0.0])
        np.testing.assert_allclose(navigator._room.near_center, [-0.15, 0.0], atol=0.01)
        np.testing.assert_allclose(navigator._room.far_center, [6.15, 0.0], atol=0.01)
        np.testing.assert_allclose(navigator._goal[:2], [6.15, 0.0], atol=0.01)
        navigator._toward_far_end = False
        np.testing.assert_allclose(navigator._end_goal()[:2], [-0.15, 0.0], atol=0.01)
        command = LQIController().compute_velocity_command(state, reference)
        wheel_speeds = velocity_command_to_wheel_speeds(command)
        self.assertTrue(np.all(np.isfinite(wheel_speeds)))
        self.assertLessEqual(max(abs(value) for value in wheel_speeds), 7.88)

    def test_west_facing_corner_spawn_reverses_axis_without_rotating_room(self):
        config = SceneConfig(SpawnPose(6.5, 2.5, 225.0), SpawnPose(6.9, 0.0, 180.0))
        model = load_model(config=config)
        navigator = Navigator(marker_map=marker_map_from_model(model), verbose=False)
        state = StateEstimate(0.0, config.robot.pose_world, np.zeros((3, 3)))
        navigator.trajectory_reference(state, scan_output(model))
        np.testing.assert_allclose(navigator._room.axis, [-1.0, 0.0])
        np.testing.assert_allclose(navigator._room.near_center, [6.15, 0.0], atol=0.01)
        np.testing.assert_allclose(navigator._room.far_center, [-0.15, 0.0], atol=0.01)

    def test_marker_approach_uses_copied_transformed_landmarks(self):
        config = SceneConfig(SpawnPose(-0.5, -2.5, 45.0), SpawnPose(4.0, 0.0, 135.0))
        model = load_model(config=config)
        transform = marker_map_from_model(model)[0]
        # A non-default ID verifies confirmation also uses the supplied map.
        supplied_map = {7: transform.copy()}
        navigator = Navigator(marker_map=supplied_map, verbose=False)
        supplied_map[7][:] = np.nan
        observation = MarkerObservation(
            time=0.0, marker_id=7, range_m=4.0, bearing_rad=0.0,
            T_camera_marker=np.eye(4), T_base_marker=np.eye(4),
            reprojection_error_px=0.0, ambiguity_ratio=0.0, side_px=100.0,
            corners_px=np.zeros((4, 2)),
        )
        for frame in range(3):
            time = frame * 0.1
            state = StateEstimate(time, config.robot.pose_world, np.zeros((3, 3)))
            navigator.trajectory_reference(
                state, scan_output(model, markers=[observation], time=time),
            )
        self.assertEqual(navigator.phase, "TO_STAGING")
        heading = math.radians(-45.0)
        normal = np.array([math.cos(math.radians(135)), math.sin(math.radians(135))])
        marker_xy = transform[:2, 3]
        np.testing.assert_allclose(navigator._staging_pose, [*(marker_xy + 0.5 * normal), heading])
        np.testing.assert_allclose(navigator._final_pose, [*(marker_xy + 0.2 * normal), heading])


if __name__ == "__main__":
    unittest.main()
