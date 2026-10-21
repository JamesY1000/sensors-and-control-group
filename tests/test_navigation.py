"""Behaviour checks for planning, docking and estimated-state reference tracking.

Run with: python -m unittest discover -s tests -v
These use synthetic sensor observations; rendered MuJoCo trials are separate.
"""

from dataclasses import replace
import math
import unittest

import numpy as np

from track_a_mobile_robot.control import LQIController, velocity_command_to_wheel_speeds
from track_a_mobile_robot.estimation import StateEstimate
from track_a_mobile_robot.navigation import Navigator, TrajectoryReference, plan_waypoints, segment_is_clear
from track_a_mobile_robot.perception import CameraPoseObservation, PerceptionOutput
from track_a_mobile_robot.scene_config import load_config
from track_a_mobile_robot.sensing import CameraReading, LidarReading
from track_a_mobile_robot.state_model import wrap_angle


def observations(time, pose, obstacles=(), *, camera=True, fresh=True, scan_time=None, too_close=False):
    """Express known synthetic points as a robot-frame LiDAR scan."""
    points = np.asarray(obstacles, dtype=float).reshape(-1, 2)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    local = (points - np.asarray(pose[:2])) @ np.array([[c, -s], [s, c]])
    ranges = tuple(np.linalg.norm(local, axis=1)) if not too_close else (math.nan,)
    lidar = LidarReading(time if scan_time is None else scan_time, (), ranges, 0.16, 8.0)
    observation = CameraPoseObservation(time, 0, tuple(pose), np.eye(3) * .001) if camera and fresh else None
    return PerceptionOutput(time, lidar, local, CameraReading(time, np.empty((0, 0, 3))),
                            [], fresh, observation)


def state(time, pose):
    return StateEstimate(time, tuple(pose), np.eye(3) * .001)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_config().navigation
        self.nav = Navigator((2, 0, math.pi), self.settings)

    def step(self, time, pose, **kwargs):
        return self.nav.trajectory_reference(state(time, pose), observations(time, pose, **kwargs))

    def test_direct_route_and_clear_detour(self):
        np.testing.assert_allclose(plan_waypoints((0, 0), (2, 0), [], .27, .1), [[2, 0]])
        wall = np.column_stack((np.ones(15), np.linspace(-.7, .7, 15)))
        route = plan_waypoints((0, 0), (2, 0), wall, .27, .1)
        self.assertGreater(len(route), 1)
        previous = np.zeros(2)
        for waypoint in route:
            self.assertTrue(segment_is_clear(previous, waypoint, wall, .27))
            previous = waypoint
        np.testing.assert_allclose(route[-1], [2, 0])

    def test_blocked_goal_and_enclosed_start_have_no_route(self):
        self.assertEqual(plan_waypoints((0, 0), (2, 0), [(2, 0)], .27, .1), [])
        angles = np.linspace(0, 2 * math.pi, 100)
        ring = np.column_stack((np.cos(angles), np.sin(angles)))
        self.assertEqual(plan_waypoints((0, 0), (2, 0), ring, .27, .1), [])

    def test_endpoint_connections_preserve_clearance_near_grid_boundaries(self):
        obstacles = np.array([(0, .28), (1, 0), (2, .28)])
        for start, goal in (((0, 0), (2, 0)), ((2, 0), (0, 0))):
            route = plan_waypoints(start, goal, obstacles, .27, .1)
            self.assertTrue(route)
            previous = start
            for point in route:
                self.assertTrue(segment_is_clear(previous, point, obstacles, .27))
                previous = point
            np.testing.assert_allclose(route[-1], goal)

    def test_optional_planning_margin_does_not_trap_a_safe_start(self):
        obstacles = np.array([(0, .28), (.7, 0)])
        pose = (0, 0, 0)
        reference = self.step(0, pose, obstacles=obstacles)
        self.assertFalse(reference.stop_requested)
        previous = pose[:2]
        for point in self.nav.waypoints:
            self.assertTrue(segment_is_clear(previous, point, obstacles, self.settings.clearance_m))
            previous = point
        # The actual safety clearance still stops the robot.
        self.assertTrue(self.step(.2, pose, obstacles=[(0, .26)]).stop_requested)

    def test_new_scan_replans_a_blocked_segment(self):
        pose = (0, 0, 0)
        self.step(0, pose)
        np.testing.assert_allclose(self.nav.waypoints, [self.nav.approach_pose[:2]])
        wall = np.column_stack((np.full(11, .7), np.linspace(-.5, .5, 11)))
        reference = self.step(.2, pose, obstacles=wall)
        self.assertFalse(reference.stop_requested)
        self.assertGreater(len(self.nav.waypoints), 1)
        self.assertTrue(segment_is_clear(pose[:2], self.nav.waypoints[0], wall, .27))

    def test_blocked_approach_target_is_reported(self):
        reference = self.step(0, (0, 0, 0), obstacles=[self.nav.approach_pose[:2]])
        self.assertTrue(reference.stop_requested)
        self.assertIn("Approach target obstructed", self.nav.status)

    def test_configured_dock_orientation_sets_targets(self):
        nav = Navigator((3, 4, math.pi / 2), self.settings)
        np.testing.assert_allclose(nav.approach_pose, [3, 4.5, -math.pi / 2])
        np.testing.assert_allclose(nav.docking_pose, [3, 4.3, -math.pi / 2])

    def test_camera_confirmations_must_be_new_and_reliable(self):
        pose = self.nav.approach_pose
        self.assertTrue(self.step(0, pose).stop_requested)
        for time in (.02, .04, .06):
            self.assertTrue(self.step(time, pose, fresh=False).stop_requested)
        self.assertEqual(self.nav.phase, "ALIGN")
        self.step(.08, pose, camera=False)
        self.step(.10, pose)
        self.step(.12, pose)
        self.assertEqual(self.nav.phase, "ALIGN")
        self.step(.14, pose)
        self.assertEqual(self.nav.phase, "DOCK")

    def test_marker_loss_stops_and_resume_does_not_retrace_line(self):
        pose = self.nav.approach_pose
        for time in (0, .02, .04):
            self.step(time, pose)
        midway = np.array([1.65, 0, 0])
        self.assertTrue(self.step(.6, midway, camera=False).stop_requested)
        reference = self.step(.62, midway)
        self.assertFalse(reference.stop_requested)
        self.assertGreaterEqual(reference.pose_world[0], midway[0])
        self.assertEqual(self.nav.phase, "DOCK")

    def test_obstacle_and_stale_scan_stop_motion(self):
        pose = (0, 0, 0)
        self.assertTrue(self.step(0, pose, obstacles=[(.2, 0)]).stop_requested)
        self.assertTrue(self.step(1, pose, scan_time=0).stop_requested)
        self.assertTrue(self.step(2, pose, too_close=True).stop_requested)
        self.assertFalse(self.step(3, pose).stop_requested)
        reference = self.nav.trajectory_reference(state(4, pose), observations(4, pose), stop_requested=True)
        self.assertTrue(reference.stop_requested)

    def test_final_position_and_heading_are_both_required(self):
        for time in (0, .02, .04):
            self.step(time, self.nav.approach_pose)
        reference = self.step(.06, (1.8, .12, 0))
        self.assertNotEqual(self.nav.phase, "DONE")
        reference = self.step(.08, (1.7, 0, .2))
        self.assertTrue(reference.rotate_only)
        self.assertNotEqual(self.nav.phase, "DONE")
        self.assertTrue(self.step(.1, self.nav.docking_pose).stop_requested)
        self.assertEqual(self.nav.phase, "DONE")
        self.assertTrue(self.step(.12, self.nav.docking_pose, camera=False).stop_requested)

    def test_in_place_heading_control_and_wheel_limits(self):
        pose = (0, 0, math.pi / 2)
        self.step(0, pose)
        reference = self.step(.02, pose)
        command = LQIController().compute_velocity_command(state(.02, pose), reference)
        self.assertEqual(command.v_mps, 0)
        self.assertLess(command.omega_rad_s, 0)
        self.assertLessEqual(max(map(abs, velocity_command_to_wheel_speeds(command))), 7.88)

    def test_heading_integral_builds_when_turn_stalls_and_resets_on_stop(self):
        controller = LQIController()
        pose = (0, 0, 0)
        commands = []
        for index in range(50):
            time = index * .02
            reference = TrajectoryReference(time, (0, 0, .2), 0, 0, rotate_only=True)
            command = controller.compute_velocity_command(state(time, pose), reference)
            commands.append(command.omega_rad_s)
            velocity_command_to_wheel_speeds(command)
        self.assertGreater(commands[-1], commands[0])
        stop = TrajectoryReference(1, pose, 0, 0, stop_requested=True)
        self.assertEqual(controller.compute_velocity_command(state(1, pose), stop).omega_rad_s, 0)
        restart = TrajectoryReference(1.02, (0, 0, .2), 0, 0, rotate_only=True)
        self.assertAlmostEqual(controller.compute_velocity_command(state(1.02, pose), restart).omega_rad_s,
                               commands[0])

    def test_closed_loop_waypoints_reach_dock(self):
        # Exercise the real reference generator and controller around a blocking wall.
        wall = np.column_stack((np.full(13, .7), np.linspace(-.6, .6, 13)))
        pose = np.array([0., 0., 0.])
        controller = LQIController()
        phases = set()
        for index in range(6000):
            time = index * .02
            reference = self.step(time, pose, obstacles=wall)
            phases.add(self.nav.phase)
            command = controller.compute_velocity_command(state(time, pose), reference)
            velocity_command_to_wheel_speeds(command)
            self.assertTrue(segment_is_clear(pose[:2], pose[:2], wall, .27))
            if self.nav.phase == "DONE":
                break
            midpoint = pose[2] + .5 * .02 * command.omega_rad_s
            pose[:2] += .02 * command.v_mps * np.array([math.cos(midpoint), math.sin(midpoint)])
            pose[2] = wrap_angle(pose[2] + .02 * command.omega_rad_s)
        self.assertEqual(self.nav.phase, "DONE", (pose, self.nav.status))
        self.assertIn("APPROACH", phases)
        self.assertIn("DOCK", phases)
        self.assertLessEqual(np.linalg.norm(pose[:2] - self.nav.docking_pose[:2]), .03)

    def test_invalid_settings_and_time_are_rejected(self):
        for changes in ({"clearance_m": -1}, {"camera_confirmations": 1.5},
                        {"camera_confirmations": True}, {"dock_distance_m": .6}):
            with self.assertRaises(ValueError):
                replace(self.settings, **changes)
        self.step(0, (0, 0, 0))
        with self.assertRaises(ValueError):
            self.step(0, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
