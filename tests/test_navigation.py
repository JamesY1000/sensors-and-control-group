"""Exploration, local avoidance, trajectory tracking, docking and display checks."""

from dataclasses import replace
from contextlib import nullcontext
import math
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import mujoco

from track_a_mobile_robot.control import MotionController, driving_gain, lqi_model, velocity_command_to_wheel_speeds
from track_a_mobile_robot.display import OverlayPublisher, SensorDisplay
from track_a_mobile_robot.estimation import StateEstimate
from track_a_mobile_robot.navigation import (
    BoundaryFollower, MotionTrajectory, MotionRequest, Navigator, TrajectoryReference,
    corridor_is_clear, grouped_scan, motion_is_clear, path_is_clear,
)
from track_a_mobile_robot.perception import DockObservation, PerceptionOutput
from track_a_mobile_robot.scene_config import load_config
from track_a_mobile_robot.sensing import CameraReading, LidarReading
from track_a_mobile_robot.state_model import wrap_angle


def observations(time, pose, obstacles=(), *, camera=True, fresh=True, scan_time=None, too_close=False):
    """Generate synthetic sensor measurements; navigation never receives generator truth."""
    points = np.asarray(obstacles, dtype=float).reshape(-1, 2)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    local = (points - np.asarray(pose[:2])) @ np.array([[c, -s], [s, c]])
    ranges = tuple(np.linalg.norm(local, axis=1)) if not too_close else (math.nan,)
    lidar = LidarReading(time if scan_time is None else scan_time, (), ranges, .16, 8.)
    estimate = state(time, pose)
    observation = DockObservation(time, 0, estimate.dock_pose_base, np.eye(3) * .001) if camera and fresh else None
    return PerceptionOutput(time, lidar, local, CameraReading(time, np.empty((0, 0, 3))), [], fresh, observation)


def state(time, pose):
    return StateEstimate(time, tuple(pose), np.eye(3) * .001, (2, 0, math.pi), np.eye(3) * .001)


def navigation_settings():
    # These synthetic poses assume fixed offsets; user tuning of the demo YAML
    # should not silently change the geometry of a unit-test scenario.
    return replace(load_config().navigation, coarse_distance_m=1.5,
                   approach_distance_m=.5, dock_distance_m=.3, clearance_m=.27)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.settings = navigation_settings()
        self.nav = Navigator(self.settings)

    def step(self, time, pose, **kwargs):
        return self.nav.motion_request(state(time, pose), observations(time, pose, **kwargs))

    def test_first_marker_starts_reference_immediately(self):
        request = self.step(0, (0, 0, 0))
        self.assertEqual(self.nav.phase, 'APPROACH')
        self.assertFalse(request.visual_servo)
        self.assertEqual(request.dock_offset_m, self.settings.coarse_distance_m)
        self.assertEqual(request.speed_limit_mps, self.settings.drive_speed_mps)
        self.assertFalse(hasattr(self.nav, 'waypoints'))
        self.assertIsNotNone(request.reference)

    def test_direct_approach_keeps_one_full_reference_past_one_metre(self):
        estimate = replace(state(0, (0,0,0)), dock_pose_local=(6.5,0,math.pi))
        request = self.nav.motion_request(estimate, observations(0, estimate.pose_local))
        self.assertEqual(request.reference.kind, 'DRIVE')
        np.testing.assert_allclose(self.nav.trajectory_path[-1,:2], (5,0), atol=1e-12)
        segment = request.reference.segment_id
        # Follow ideal reference samples through several metres without restarting.
        duration = self.nav._trajectory._duration
        for time in np.linspace(.02, duration*.6, 400):
            pose = self.nav._trajectory._pose_at(3*(time/duration)**2-2*(time/duration)**3)
            estimate = replace(estimate, time=time, pose_local=tuple(pose))
            request = self.nav.motion_request(estimate, observations(time, pose))
            self.assertEqual(request.mode, 'APPROACH')
            self.assertEqual(request.reference.segment_id, segment)
            self.assertGreater(request.reference.v_mps, 0)
            self.assertEqual(request.reference.omega_rad_s, 0)
        self.assertGreater(estimate.pose_local[0], 3.)

    def test_distant_blocker_shortens_reference_without_starting_boundary_following(self):
        estimate = replace(state(0, (0,0,0)), dock_pose_local=(6.5,0,math.pi))
        self.nav.motion_request(estimate, observations(0, estimate.pose_local))
        estimate = replace(estimate, time=.02)
        request = self.nav.motion_request(estimate,
            observations(.02, estimate.pose_local, obstacles=[(3.,0.)]))
        self.assertEqual(self.nav.phase, 'APPROACH')
        self.assertFalse(self.nav._detour.active)
        self.assertEqual(self.nav._detour.side, 0)
        np.testing.assert_allclose(self.nav.trajectory_path[-1,:2], (2.,0.), atol=1e-12)
        self.assertFalse(request.visual_servo)
        # Removing the obstruction restores the full target without a side choice.
        for time in (.04, .46):
            estimate = replace(estimate, time=time)
            request = self.nav.motion_request(estimate, observations(time, estimate.pose_local))
        self.assertEqual(request.mode, 'APPROACH')
        np.testing.assert_allclose(self.nav.trajectory_path[-1,:2], (5,0), atol=1e-12)

    def test_boundary_following_starts_only_near_a_previously_distant_blocker(self):
        dock = (6.5,0,math.pi)
        for time, pose in ((0., (0.,0.,0.)), (1., (2.,0.,0.))):
            estimate = replace(state(time, pose), dock_pose_local=dock)
            self.nav.motion_request(estimate, observations(time, pose, obstacles=[(3.,0.)]))
            self.assertEqual(self.nav.phase, 'APPROACH' if time == 0 else 'AVOID')
        self.assertTrue(self.nav._detour.active)
        self.assertIn(self.nav._detour.side, (-1,1))

    def test_obstacle_beyond_direct_goal_does_not_trigger_detour(self):
        estimate = replace(state(0, (0,0,0)), dock_pose_local=(6.5,0,math.pi))
        request = self.nav.motion_request(estimate, observations(0, estimate.pose_local, obstacles=[(6.,0.)]))
        self.assertEqual(request.mode, 'APPROACH')
        np.testing.assert_allclose(self.nav.trajectory_path[-1,:2], (5,0), atol=1e-12)

    def test_estimated_dock_orientation_sets_display_targets(self):
        estimate = replace(state(0, (0, 0, 0)), dock_pose_local=(3, 4, math.pi / 2))
        self.nav.motion_request(estimate, observations(0, (0, 0, 0)))
        np.testing.assert_allclose(self.nav.approach_pose, [3, 4.5, -math.pi / 2])
        np.testing.assert_allclose(self.nav.docking_pose, [3, 4.3, -math.pi / 2])

    def test_stored_estimate_allows_handover_at_coarse_offset(self):
        request = self.step(0, (.5, 0, 0), camera=False)
        self.assertEqual(self.nav.phase, 'ALIGN')
        self.assertEqual(request.dock_offset_m, .5)
        self.assertEqual(request.speed_limit_mps, self.settings.pbvs_speed_mps)

    def test_marker_loss_never_times_out_in_any_travel_phase(self):
        for pose, mode in (((0,0,0), 'APPROACH'), ((1.2,0,0), 'ALIGN'), ((1.5,0,0), 'DOCK')):
            self.nav = Navigator(self.settings)
            if mode != 'APPROACH':
                self.step(-.02, (.5,0,0))
            self.step(0, pose)
            for time in (.7, 21., 300., 10000.):
                request = self.step(time, pose, camera=False)
                self.assertEqual(request.mode, mode)
                self.assertFalse(request.stop_requested)

    def test_prediction_can_start_pbvs_without_fresh_camera(self):
        self.step(0, (0,0,0))
        request = self.step(300, (.5,0,0), camera=False)
        self.assertEqual(request.mode, 'ALIGN')
        self.assertTrue(request.visual_servo)

    def test_both_final_position_and_heading_required(self):
        self.step(-.02, (.5,0,0))
        for t in (0, .02, .04):
            self.step(t, (1.5, 0, 0))
        self.assertTrue(self.step(.06, (1.6, .02, 0)).visual_servo)
        self.assertTrue(self.step(.08, (1.7, 0, .2)).visual_servo)
        self.assertNotEqual(self.nav.phase, 'DONE')
        self.assertTrue(self.step(.1, (1.7, 0, 0)).stop_requested)
        self.assertEqual(self.nav.phase, 'DONE')
        self.assertTrue(self.step(.12, (1.7, 0, 0), camera=False).stop_requested)

    def test_stale_scan_and_close_hits_stop(self):
        self.assertTrue(self.step(1, (0, 0, 0), scan_time=0).stop_requested)
        self.assertTrue(self.step(2, (0, 0, 0), too_close=True).stop_requested)
        self.assertTrue(self.step(3, (0, 0, 0), obstacles=[(.2, 0)]).stop_requested)

    def test_motion_guard_covers_curves_reverse_and_footprint(self):
        self.assertTrue(motion_is_clear([], .15, .4, .27))
        self.assertFalse(motion_is_clear([(.35, 0)], .15, 0, .27))
        self.assertFalse(motion_is_clear([(-.35, 0)], -.15, 0, .27))
        self.assertFalse(motion_is_clear([(.1, .26)], .15, 1, .27))
        self.assertFalse(motion_is_clear([(.2, 0)], 0, .4, .27))
        self.assertTrue(motion_is_clear([(.7, .7)], .15, .4, .27))

    def test_blocked_tracking_waits_without_detour_and_resumes(self):
        request = self.step(0, (0, 0, 0))
        blocked = observations(0, (0, 0, 0), obstacles=[(.35, 0)])
        self.assertFalse(self.nav.allow_motion(request, .15, 0, blocked))
        self.assertIn('blocked', self.nav.status)
        self.assertEqual(self.nav.phase, 'APPROACH')
        self.assertTrue(self.nav.allow_motion(request, .15, 0, observations(0, (0, 0, 0))))

    def test_closed_loop_hybrid_reaches_both_offsets(self):
        # Several starts exercise the entire approach, not just the final 20 cm.
        for start in ((-2, .7, 0), (0, -.6, .2), (1.6, 0, 0)):
            self.nav = Navigator(self.settings)
            controller = MotionController(self.settings)
            pose = np.array(start, dtype=float)
            phases = set()
            for i in range(7000):
                time = i * .02
                request = self.step(time, pose)
                phases.add(self.nav.phase)
                command = controller.compute_velocity_command(state(time, pose), request)
                velocity_command_to_wheel_speeds(command)
                self.assertTrue(self.nav.allow_motion(request, command.v_mps, command.omega_rad_s,
                                                     observations(time, pose)) or request.stop_requested)
                if self.nav.phase == 'DONE':
                    break
                midpoint = pose[2] + .01 * command.omega_rad_s
                pose[:2] += .02 * command.v_mps * np.array([math.cos(midpoint), math.sin(midpoint)])
                pose[2] = wrap_angle(pose[2] + .02 * command.omega_rad_s)
            self.assertEqual(self.nav.phase, 'DONE', (start, pose, self.nav.status))
            if np.linalg.norm(np.array(start[:2]) - [2,0]) > self.settings.coarse_distance_m + self.settings.coarse_tolerance_m:
                self.assertIn('APPROACH', phases)
            self.assertIn('ALIGN', phases)
            self.assertIn('DOCK', phases)

    def test_invalid_time_settings_and_request_are_rejected(self):
        for changes in ({'clearance_m': -1}, {'search_spin_s': True},
                        {'lidar_timeout_s': 0}, {'dock_distance_m': .6}):
            with self.assertRaises(ValueError):
                replace(self.settings, **changes)
        with self.assertRaises(ValueError):
            MotionRequest(0, 'APPROACH')
        self.step(0, (0, 0, 0))
        with self.assertRaises(ValueError):
            self.step(0, (0, 0, 0))


class AvoidanceTests(unittest.TestCase):
    def setUp(self):
        self.settings = navigation_settings()

    def test_corridor_includes_body_width_endcaps_and_reverse(self):
        self.assertFalse(corridor_is_clear([[.5, .2]], [1, 0], .27))
        self.assertFalse(corridor_is_clear([[1.2, 0]], [1, 0], .27))
        self.assertFalse(corridor_is_clear([[-.4, 0]], [-.5, 0], .27))
        self.assertTrue(corridor_is_clear([[.5, .4]], [1, 0], .27))
        self.assertFalse(corridor_is_clear([[math.nan, 0]], [1, 0], .27))

    def test_follows_blocking_obstacle_instead_of_nearer_side_wall(self):
        follower = BoundaryFollower(self.settings)
        points = np.array([[.9, 0], [0, -.5], [-.3, -.5]])
        for time in (0., .1):
            mode, _ = follower.update(time, [0,0,0], [3,0,0], points, 8)
            self.assertEqual(mode, 'avoid')
            np.testing.assert_allclose(follower.anchor, [.9,0])

    def test_grouping_closes_small_leg_gaps_without_enlarging_walls(self):
        legs = [[.7, -.35], [.7, .35]]
        self.assertTrue(corridor_is_clear(legs, [1., 0], .27))
        self.assertFalse(corridor_is_clear(grouped_scan(legs, .8), [1., 0], .27))
        wide = [[.7, -.6], [.7, .6]]
        self.assertTrue(corridor_is_clear(grouped_scan(wide, .8), [1., 0], .27))
        wall = [[x, .45] for x in np.linspace(-1, 2, 40)]
        self.assertTrue(corridor_is_clear(grouped_scan(wall, .8), [1., 0], .27))

    def test_no_progress_and_return_to_hit_point_stop_boundary_motion(self):
        follower = BoundaryFollower(self.settings)
        follower.update(0, [0,0,0], [3,0,0], [[.7,0]], 8)
        self.assertEqual(follower.update(31, [0,0,0], [3,0,0], [[.7,0]], 8)[0], 'blocked')
        follower.reset()
        for time, xy in enumerate(((0,0),(.5,0),(1,0),(1,1),(0,1),(0,0))):
            result, _ = follower.update(time, [*xy,0], [3,0,0], [[.7,0]], 8, forced=True)
        self.assertEqual(result, 'blocked')
        self.assertTrue(follower.failed)

    def test_side_and_target_persist_then_clear_corridor_resumes(self):
        detour = BoundaryFollower(self.settings)
        points = np.array([[.7, y] for y in np.linspace(-.3, .3, 20)])
        mode, target = detour.update(0, [0, 0, 0], [2, 0, 0], points, 8)
        self.assertEqual(mode, 'avoid')
        side = detour.side
        for t in (.02, .2, .4):
            again, same = detour.update(t, [0, 0, 0], [2, 0, 0], points, 8)
            self.assertEqual(detour.side, side)
            np.testing.assert_array_equal(same, target)
        self.assertEqual(detour.update(.6, [0, 0, 0], [2, 0, 0], [], 8)[0], 'avoid')
        self.assertEqual(detour.update(1.01, [0, 0, 0], [2, 0, 0], [], 8)[0], 'clear')
        self.assertFalse(detour.active)

    def test_boundary_side_persists_until_timeout(self):
        detour = BoundaryFollower(replace(self.settings, detour_timeout_s=2.))
        points = [[.7, 0]]
        detour.update(0, [0, 0, 0], [2, 0, 0], points, 8)
        side = detour.side
        detour.update(1.1, [0, 0, 0], [2, 0, 0], points, 8)
        self.assertEqual(detour.side, side)
        self.assertEqual(detour.update(2.1, [0, 0, 0], [2, 0, 0], points, 8)[0], 'blocked')
        self.assertEqual(detour.update(20, [0, 0, 0], [2, 0, 0], points, 8)[0], 'blocked')
        detour.update(21, [1.2, 0, 0], [2, 0, 0], [], 8)
        self.assertEqual(detour.update(21.5, [1.2, 0, 0], [2, 0, 0], [], 8)[0], 'clear')

    def test_nearby_leg_gaps_trigger_boundary_following(self):
        legs = np.array([[.7, -.35], [.7, .35], [1.4, -.35], [1.4, .35]])
        # Ordinary body clearance fits this gap; grouping clearance closes it.
        self.assertTrue(corridor_is_clear(legs, [2., 0], self.settings.clearance_m))
        follower = BoundaryFollower(self.settings)
        mode, target = follower.update(0, [0, 0, 0], [2., 0, 0], legs, 8)
        self.assertEqual(mode, 'avoid')
        self.assertTrue(corridor_is_clear(legs, target[:2], self.settings.boundary_join_gap_m/2))

    def test_robot_can_move_out_of_extra_grouping_margin(self):
        follower = BoundaryFollower(self.settings)
        points = np.array([[.35, 0.]])
        mode, target = follower.update(0, [0, 0, 0], [2., 0, 0], points, 8)
        self.assertEqual(mode, 'avoid')
        self.assertTrue(corridor_is_clear(points, target[:2], self.settings.clearance_m))
        self.assertGreater(np.linalg.norm(points[0] - target[:2]), .35)

    def test_occlusion_does_not_interrupt_boundary_and_final_docking_does_not_detour(self):
        nav = Navigator(self.settings)
        self.assertEqual(nav.motion_request(state(0, (0, 0, 0)),
            observations(0, (0, 0, 0), obstacles=[(.7, 0)])).mode, 'AVOID')
        self.assertEqual(nav.motion_request(state(4, (0, 0, 0)),
            observations(4, (0, 0, 0), obstacles=[(.7, 0)], camera=False)).mode, 'AVOID')
        self.assertEqual(nav.motion_request(state(21, (.2, 0, 0)),
            observations(21, (.2, 0, 0), obstacles=[(.7, 0)], camera=False)).mode, 'AVOID')
        self.assertEqual(nav.motion_request(state(52, (.2, 0, 0)),
            observations(52, (.2, 0, 0), obstacles=[(.7, 0)], camera=False)).mode, 'STOP')
        nav = Navigator(self.settings)
        nav.motion_request(state(-.02, (.5,0,0)), observations(-.02, (.5,0,0)))
        for t in (0, .02, .04):
            request = nav.motion_request(state(t, (1.5, 0, 0)), observations(t, (1.5, 0, 0)))
        self.assertEqual(nav.phase, 'DOCK')
        self.assertFalse(nav.allow_motion(request, .06, 0, observations(.04, (1.5, 0, 0), obstacles=[(1.82, 0)])))
        self.assertEqual(nav.phase, 'DOCK')

    def test_exhaustion_stays_stopped_even_if_camera_is_lost(self):
        nav = Navigator(replace(self.settings, detour_timeout_s=2.))
        nav.motion_request(state(0, (0, 0, 0)), observations(0, (0, 0, 0), obstacles=[(.7, 0)]))
        for time in (3, 30):
            request = nav.motion_request(state(time, (0, 0, 0)),
                observations(time, (0, 0, 0), obstacles=[(.7, 0)], camera=False))
            self.assertTrue(request.stop_requested)
            self.assertIn('exhausted', nav.status)

    def test_occlusion_keeps_lqi_tracking_the_travel_target(self):
        nav, controller = Navigator(self.settings), MotionController(self.settings)
        pose = (0,0,0)
        nav.motion_request(state(0, pose), observations(0, pose))
        for time in (30., 30.2, 30.22):
            request = nav.motion_request(state(time, pose), observations(time, pose, camera=False))
            command = controller.compute_velocity_command(state(time, pose), request)
            self.assertEqual(request.mode, 'APPROACH')
            self.assertEqual(request.reference.kind, 'DRIVE')
        self.assertGreater(command.v_mps, 0)

    def test_alignment_and_final_handover_continue_during_occlusion(self):
        nav = Navigator(self.settings)
        nav.motion_request(state(-.02, (.5,0,0)), observations(-.02, (.5,0,0)))
        pose = (1.5,0,.4)
        nav.motion_request(state(0, pose), observations(0, pose))
        request = nav.motion_request(state(300, pose), observations(300, pose, camera=False))
        self.assertEqual(request.mode, 'ALIGN')
        aligned = (1.5,0,0)
        request = nav.motion_request(state(301, aligned), observations(301, aligned, camera=False))
        self.assertEqual(request.mode, 'DOCK')
        final = (1.7,0,0)
        request = nav.motion_request(state(302, final), observations(302, final, camera=False))
        self.assertTrue(request.stop_requested)
        self.assertEqual(nav.phase, 'DONE')

    def test_lqi_passes_obstacles_and_rejoins_approach(self):
        scenes = [np.array([[.7, -.35], [.7, .35], [1.4, -.35], [1.4, .35]]),
                  np.array([[.7, y] for y in np.linspace(-.25, .25, 20)]),
                  np.array([[x, y] for x, low, high in ((.65, -.4, .1), (1.1, -.1, .4))
                            for y in np.linspace(low, high, 20)])]
        for obstacles in scenes:
            nav, controller = Navigator(self.settings), MotionController(self.settings)
            pose = np.zeros(3)
            phases = set()
            for i in range(10000):
                time = i * .02
                output = observations(time, pose, obstacles=obstacles)
                # Keep the required stand-off region beyond the obstacle cluster.
                # The former 1.5 m target was inside its inflated rear boundary.
                estimate = replace(state(time, pose), dock_pose_local=(3.5, 0., math.pi))
                request = nav.motion_request(estimate, output)
                phases.add(nav.phase)
                command = controller.compute_velocity_command(estimate, request)
                if nav.allow_motion(request, command.v_mps, command.omega_rad_s, output):
                    midpoint = pose[2] + .01 * command.omega_rad_s
                    pose[:2] += .02 * command.v_mps * np.array([math.cos(midpoint), math.sin(midpoint)])
                    pose[2] = wrap_angle(pose[2] + .02 * command.omega_rad_s)
                else:
                    controller.reset()
                self.assertGreaterEqual(np.linalg.norm(obstacles - pose[:2], axis=1).min(), self.settings.clearance_m)
                if nav.phase == 'DONE':
                    break
            self.assertIn('AVOID', phases)
            self.assertEqual(nav.phase, 'DONE', (pose, nav.status))


class LQITests(unittest.TestCase):
    def setUp(self):
        self.settings = navigation_settings()
        self.controller = MotionController(self.settings)

    def request(self, time, pose, v=.06, kind='DRIVE', direction=1):
        return MotionRequest(time, 'APPROACH', .06, 1.5, reference=TrajectoryReference(
            time, tuple(pose), v, 0., kind, 1, direction))

    def test_augmented_design_is_stable_forward_and_reverse(self):
        for speed in (-.12, -.02, .02, .05, .12):
            A, B, Q, R = lqi_model(speed)
            K = driving_gain(speed)
            self.assertTrue(np.all(np.linalg.eigvals(A - B @ K).real < 0))
            self.assertGreater(np.linalg.norm(K[:, 3]), 0)
            # Independent numerical integration of the linear error model.
            error = np.array([.1, .05, .1, 0.])
            for _ in range(20000):
                error += .01 * (A - B @ K) @ error
            self.assertLess(np.linalg.norm(error), 1e-5)

    def test_reference_feedforward_and_feedback_signs(self):
        target = (0., 0., 0.)
        request = self.request(0, target)
        feedforward = self.controller.compute_velocity_command(state(0, target), request)
        self.assertAlmostEqual(feedforward.v_mps, .06)
        self.assertAlmostEqual(feedforward.omega_rad_s, 0)
        ahead = self.controller.compute_velocity_command(state(0, (.02, 0, 0)), request)
        self.assertLess(ahead.v_mps, feedforward.v_mps)
        left = self.controller.compute_velocity_command(state(0, (0, .03, 0)), request)
        self.assertLess(left.omega_rad_s, 0)
        reverse = self.request(0, target, v=-.04, direction=-1)
        left_reverse = self.controller.compute_velocity_command(state(0, (0, .03, 0)), reverse)
        self.assertGreater(left_reverse.omega_rad_s, 0)

    def test_local_frame_does_not_change_command(self):
        pose = np.array([1., .04, -.1])
        reference = np.array([1.01, 0, 0])
        angle = 2.1
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        def transform(value):
            return (*tuple(rotation @ value[:2] + [12, -7]), wrap_angle(value[2] + angle))
        original = state(0, pose)
        a = self.controller.compute_velocity_command(original, self.request(0, reference))
        self.controller.reset()
        shifted = replace(original, pose_local=transform(pose), dock_pose_local=transform(original.dock_pose_local))
        b = self.controller.compute_velocity_command(shifted, self.request(0, transform(reference)))
        np.testing.assert_allclose([a.v_mps, a.omega_rad_s], [b.v_mps, b.omega_rad_s], atol=1e-12)

    def test_integral_accumulates_and_stop_clears_it(self):
        values = []
        for i in range(50):
            t = i * .02
            values.append(self.controller.compute_velocity_command(state(t, (-.01, 0, 0)),
                          self.request(t, (0, 0, 0), v=0)).v_mps)
        self.assertGreater(values[-1], values[0])
        self.controller.compute_velocity_command(state(1, (0, 0, 0)), MotionRequest(1, 'STOP'))
        restarted = self.controller.compute_velocity_command(state(1.02, (-.01, 0, 0)),
                                                              self.request(1.02, (0, 0, 0), v=0))
        self.assertAlmostEqual(restarted.v_mps, values[0])

    def test_heading_wrap_turn_and_stale_reference(self):
        command = self.controller.compute_velocity_command(state(0, (0, 0, math.pi - .02)),
                   self.request(0, (0, 0, -math.pi + .02), v=0, kind='TURN'))
        self.assertEqual(command.v_mps, 0)
        self.assertGreater(command.omega_rad_s, 0)
        with self.assertRaises(ValueError):
            self.controller.compute_velocity_command(state(1, (0, 0, 0)),
                replace(self.request(0, (0, 0, 0)), time=1))

    def test_trajectory_has_bounded_consistent_velocities(self):
        for target in ((1., 0., 0.), (-.5, 0., 0.), (0., 0., math.pi / 2)):
            generator = MotionTrajectory(self.settings)
            pose = np.zeros(3)
            previous = generator.sample(0, pose, target, .15)
            for i in range(1, 1000):
                time = i * .01
                reference = generator.sample(time, pose, target, .15)
                if reference.segment_id != previous.segment_id:
                    break
                self.assertLessEqual(abs(reference.v_mps), .15)
                self.assertLessEqual(abs(reference.omega_rad_s), self.settings.turn_speed_rad_s)
                # Numerical derivative agrees with nonholonomic reference velocity.
                velocity = (np.array(reference.pose_local[:2]) - previous.pose_local[:2]) / .01
                average_v = (reference.v_mps + previous.v_mps) / 2
                heading = reference.pose_local[2]
                np.testing.assert_allclose(velocity, average_v * np.array([math.cos(heading), math.sin(heading)]), atol=1e-5)
                pose = np.array(reference.pose_local)
                previous = reference

    def test_arc_is_tangent_and_has_consistent_linear_and_angular_velocity(self):
        for target in ((1., .4, 0.), (1., -.4, 0.), (-1., .4, 0.)):
            trajectory = MotionTrajectory(self.settings)
            initial = trajectory.sample(0, np.zeros(3), target, .15, max_distance=2.)
            self.assertEqual(initial.kind, 'DRIVE')
            self.assertAlmostEqual(initial.pose_local[2], 0.)
            np.testing.assert_allclose(trajectory._pose_at(1)[:2], target[:2], atol=1e-12)
            previous = initial
            dt = trajectory._duration/1000
            for i in range(1, 1000):
                reference = trajectory.sample(i*dt, previous.pose_local, target, .15, max_distance=2.)
                self.assertEqual(reference.segment_id, initial.segment_id)
                velocity = (np.array(reference.pose_local[:2])-previous.pose_local[:2])/dt
                heading = previous.pose_local[2]+wrap_angle(reference.pose_local[2]-previous.pose_local[2])/2
                speed = (reference.v_mps+previous.v_mps)/2
                np.testing.assert_allclose(velocity, speed*np.array([math.cos(heading), math.sin(heading)]), atol=1e-6)
                self.assertAlmostEqual(wrap_angle(reference.pose_local[2]-previous.pose_local[2])/dt,
                                       (reference.omega_rad_s+previous.omega_rad_s)/2, places=6)
                self.assertLessEqual(abs(reference.v_mps), .15)
                self.assertLessEqual(abs(reference.omega_rad_s), self.settings.turn_speed_rad_s)
                previous = reference

    def test_curve_check_detects_obstacle_outside_straight_chord(self):
        trajectory = MotionTrajectory(self.settings)
        trajectory.sample(0, np.zeros(3), (1., .5, 0.), .15, max_distance=2.)
        obstacle = np.array([[.55,.11]])
        self.assertTrue(corridor_is_clear(obstacle, [1.,.5], .05))
        self.assertFalse(path_is_clear(obstacle, trajectory.path()[:,:2], .05))

    def test_blocked_arc_falls_back_to_a_checked_spot_turn(self):
        nav = Navigator(self.settings)
        nav.phase = 'AVOID'
        estimate = replace(state(0, (0,0,0)), dock_pose_local=(2.5,.5,math.pi))
        request = nav._tracking_request(estimate, None, self.settings.detour_speed_mps,
            observations(0, (0,0,0), obstacles=[(.5,-.14)]), target=np.array([1.,.5,0.]))
        self.assertEqual(request.mode, 'AVOID')
        self.assertEqual(request.reference.kind, 'TURN')
        self.assertTrue(path_is_clear(np.array([[.5,-.14]]), nav.trajectory_path[:,:2], self.settings.clearance_m))

    def test_reference_horizon_and_curved_lqi_design(self):
        trajectory = MotionTrajectory(self.settings)
        trajectory.sample(0, np.zeros(3), (5.,2.,0.), .15, max_distance=1.)
        self.assertLessEqual(np.linalg.norm(trajectory.path()[:,:2],axis=1).max(), 1.+1e-9)
        for turn in (-.4,.4):
            A,B,_,_ = lqi_model(.15,turn)
            self.assertEqual(A[0,1],turn)
            self.assertEqual(A[1,0],-turn)
            self.assertTrue(np.all(np.linalg.eigvals(A-B@driving_gain(.15,turn)).real<0))

    def test_direct_reference_turns_then_reaches_full_target_on_a_line(self):
        trajectory = MotionTrajectory(self.settings)
        target = np.array([4.,3.,0.])
        turn = trajectory.sample(0, (0,0,0), target, .15, allow_curve=False)
        self.assertEqual(turn.kind, 'TURN')
        time = trajectory._duration+.02
        # A small residual heading error must not turn the drive into an arc.
        pose = (0.,0.,math.atan2(3,4)-.01)
        drive = trajectory.sample(time, pose, target, .15, allow_curve=False)
        self.assertEqual(drive.kind, 'DRIVE')
        path = trajectory.path()
        np.testing.assert_allclose(path[-1,:2], target[:2], atol=1e-12)
        np.testing.assert_allclose(path[:,1], .75*path[:,0], atol=1e-12)
        np.testing.assert_allclose(path[:,2], math.atan2(3,4), atol=1e-12)

    def test_stop_and_estimate_jump_restart_reference(self):
        nav = Navigator(self.settings)
        initial = nav.motion_request(state(0, (0, 0, 0)), observations(0, (0, 0, 0)))
        moving = nav.motion_request(state(1, (.05, 0, 0)), observations(1, (.05, 0, 0)))
        self.assertGreater(moving.reference.v_mps, 0)
        self.assertFalse(nav.allow_motion(moving, .15, 0, observations(1, (.05, 0, 0), obstacles=[(.4, 0)])))
        resumed = nav.motion_request(state(10, (.05, 0, 0)), observations(10, (.05, 0, 0)))
        self.assertGreater(resumed.reference.segment_id, initial.reference.segment_id)
        np.testing.assert_allclose(resumed.reference.pose_local, (.05, 0, 0), atol=1e-12)
        self.assertEqual(resumed.reference.v_mps, 0)
        jumped = nav.motion_request(state(10.02, (.15, .15, .2)), observations(10.02, (.15, .15, .2)))
        np.testing.assert_allclose(jumped.reference.pose_local, (.15, .15, .2), atol=1e-12)

    def test_hybrid_closed_loop_with_velocity_deadband_and_turn_bias(self):
        nav = Navigator(self.settings)
        pose = np.array([0., -.2, .1])
        for i in range(6500):
            time = i * .02
            request = nav.motion_request(state(time, pose), observations(time, pose))
            command = self.controller.compute_velocity_command(state(time, pose), request)
            velocity_command_to_wheel_speeds(command)
            if nav.phase == 'DONE':
                break
            v = math.copysign(max(0, abs(command.v_mps) - .012), command.v_mps)
            omega = math.copysign(max(0, abs(command.omega_rad_s) - .035), command.omega_rad_s)
            midpoint = pose[2] + .01 * omega
            pose[:2] += .02 * v * np.array([math.cos(midpoint), math.sin(midpoint)])
            pose[2] = wrap_angle(pose[2] + .02 * omega)
        self.assertEqual(nav.phase, 'DONE', (pose, nav.status))
        self.assertLessEqual(np.linalg.norm(pose[:2] - [1.7, 0]), .03)
        self.assertLessEqual(abs(pose[2]), math.radians(3))


class PBVSTests(unittest.TestCase):
    def setUp(self):
        self.settings = navigation_settings()
        self.controller = MotionController(self.settings)

    def test_handover_ignores_heading_and_discards_trajectory(self):
        nav = Navigator(self.settings)
        pose = (.4, .15, .7)
        for t in (0, .02, .04):
            request = nav.motion_request(state(t, pose), observations(t, pose))
        self.assertEqual(request.mode, 'ALIGN')
        self.assertTrue(request.visual_servo)
        self.assertIsNone(request.reference)
        self.assertIsNone(nav.trajectory_preview)
        command = self.controller.compute_velocity_command(state(.04, pose), request)
        self.assertLess(command.omega_rad_s, 0)

    def test_side_and_too_close_arrivals_must_first_reach_coarse_offset(self):
        # All are within the old 1.75 m dock-centred region, but outside the
        # circle around the required stand-off point at (0.5, 0).
        for pose in ((1.8,1.,0.), (1.5,0.,0.), (.5,.26,0.)):
            nav = Navigator(self.settings)
            request = nav.motion_request(state(0, pose), observations(0, pose))
            self.assertEqual(request.mode, 'APPROACH')
            self.assertFalse(request.visual_servo)
            np.testing.assert_allclose(nav.coarse_pose[:2], (.5,0.), atol=1e-12)
            reached = (.5,.24,1.2)
            request = nav.motion_request(state(1, reached), observations(1, reached))
            self.assertEqual(request.mode, 'ALIGN')
            self.assertIsNone(request.reference)

    def test_stand_off_gate_follows_rotated_dock_normal(self):
        dock = (2.,3.,math.pi/2)
        # Dock normal is +Y, so the coarse target is (2, 4.5), not along world X.
        nav = Navigator(self.settings)
        for time, pose, expected in ((0., (3.,3.2,0.), 'APPROACH'),
                                     (1., (2.2,4.5,-.7), 'ALIGN')):
            estimate = replace(state(time, pose), dock_pose_local=dock)
            request = nav.motion_request(estimate, observations(time, pose))
            self.assertEqual(request.mode, expected)
            np.testing.assert_allclose(nav.coarse_pose[:2], (2.,4.5), atol=1e-12)

    def test_direct_velocities_reverse_and_heading_only(self):
        request = MotionRequest(0, 'ALIGN', .1, .5)
        ahead = self.controller.compute_velocity_command(state(0, (1.4, 0, 0)), request)
        behind = self.controller.compute_velocity_command(state(0, (1.6, 0, 0)), request)
        turn = self.controller.compute_velocity_command(state(0, (1.5, 0, .3)), request)
        self.assertGreater(ahead.v_mps, 0)
        self.assertLess(behind.v_mps, 0)
        self.assertAlmostEqual(behind.omega_rad_s, 0)
        self.assertEqual(turn.v_mps, 0)
        self.assertLess(turn.omega_rad_s, 0)
        for command in (ahead, behind, turn):
            velocity_command_to_wheel_speeds(command)
            self.assertLessEqual(abs(command.v_mps), .1)
            self.assertLessEqual(abs(command.omega_rad_s), self.settings.turn_speed_rad_s)

    def test_relative_feedback_is_invariant_to_odometry_frame(self):
        original = state(0, (.8, .15, .2))
        angle = 1.7
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        def transform(pose):
            return (*tuple(rotation @ pose[:2] + [8, -3]), wrap_angle(pose[2] + angle))
        shifted = replace(original, pose_local=transform(original.pose_local),
                          dock_pose_local=transform(original.dock_pose_local))
        request = MotionRequest(0, 'ALIGN', .1, .5)
        a = self.controller.compute_velocity_command(original, request)
        b = self.controller.compute_velocity_command(shifted, request)
        np.testing.assert_allclose([a.v_mps, a.omega_rad_s], [b.v_mps, b.omega_rad_s], atol=1e-12)

    def test_alignment_corrects_position_drift_without_camera_timeout(self):
        nav = Navigator(self.settings)
        nav.motion_request(state(-.02, (.5,0,0)), observations(-.02, (.5,0,0)))
        for t in (0, .02, .04):
            nav.motion_request(state(t, (1.5, 0, .3)), observations(t, (1.5, 0, .3)))
        pose = (1.465, 0, 0)
        request = nav.motion_request(state(.1, pose), observations(.1, pose))
        self.assertEqual(request.mode, 'ALIGN')
        self.assertIsNone(request.reference)
        self.assertGreater(self.controller.compute_velocity_command(state(.1, pose), request).v_mps, 0)
        request = nav.motion_request(state(300, pose), observations(300, pose, camera=False))
        self.assertFalse(request.stop_requested)
        self.assertGreater(self.controller.compute_velocity_command(state(300, pose), request).v_mps, 0)

    def test_handover_does_not_bypass_obstacle_clearance(self):
        nav = Navigator(self.settings)
        for t in (0, .02, .04):
            pose = (.4, 0, 0)
            request = nav.motion_request(state(t, pose), observations(t, pose, obstacles=[(1., 0)]))
        self.assertFalse(request.visual_servo)
        self.assertEqual(nav.phase, 'AVOID')

    def test_pbvs_settles_with_low_speed_friction(self):
        # Proportional speed alone can stall outside the 3 cm tolerance.
        pose = np.array([1.35, .04, .1])
        for i in range(4000):
            t = i*.02
            request = MotionRequest(t, 'ALIGN', .1, .5)
            command = self.controller.compute_velocity_command(state(t, pose), request)
            v = math.copysign(max(0., abs(command.v_mps) - .025), command.v_mps)
            omega = math.copysign(max(0., abs(command.omega_rad_s) - .035), command.omega_rad_s)
            heading = pose[2] + .01*omega
            pose[:2] += .02*v*np.array([math.cos(heading), math.sin(heading)])
            pose[2] = wrap_angle(pose[2] + .02*omega)
            if np.linalg.norm(pose[:2] - [1.5, 0]) <= .03 and abs(pose[2]) <= math.radians(3):
                break
        self.assertLessEqual(np.linalg.norm(pose[:2] - [1.5, 0]), .03)
        self.assertLessEqual(abs(pose[2]), math.radians(3))

    def test_heading_settles_despite_wheel_motor_deadband(self):
        # kv=0.1 and wheel frictionloss=0.1 imply about 1 rad/s of wheel
        # deadband, or 0.23 rad/s for an in-place turn. Check both directions.
        for initial_heading in (-.2, .2):
            self.controller.reset()
            pose = np.array([1.48, 0., initial_heading])
            for i in range(300):
                command = self.controller.compute_velocity_command(
                    state(i*.02, pose), MotionRequest(i*.02, 'ALIGN', .1, .5))
                self.assertEqual(command.v_mps, 0.)
                self.assertLessEqual(abs(command.omega_rad_s), self.settings.turn_speed_rad_s)
                omega = math.copysign(max(0., abs(command.omega_rad_s)-.24), command.omega_rad_s)
                pose[2] = wrap_angle(pose[2] + .02*omega)
            self.assertLessEqual(abs(pose[2]), math.radians(self.settings.heading_tolerance_deg))


class MarkerAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.settings = navigation_settings()
        self.nav = Navigator(self.settings)
        self.controller = MotionController(self.settings)

    def test_timed_exploration_works_without_pose_feedback_or_references(self):
        spin = self.settings.search_spin_s
        for time, expected in ((0,'SCAN'), (spin/2,'SCAN'), (spin+.02,'EXPLORE'),
                               (spin+self.settings.search_drive_s+.04,'SCAN')):
            estimate = StateEstimate(time, (0,0,0), np.zeros((3,3)))
            output = observations(time, (0,0,0), camera=False)
            request = self.nav.motion_request(estimate, output)
            command = self.controller.compute_velocity_command(estimate, request)
            self.assertEqual(request.mode, expected)
            self.assertIsNone(request.reference)
            self.assertTrue(self.nav.allow_motion(request, command.v_mps, command.omega_rad_s, output))
            self.assertEqual(command.v_mps, 0 if expected=='SCAN' else self.settings.drive_speed_mps)
            self.assertEqual(command.omega_rad_s, self.settings.turn_speed_rad_s if expected=='SCAN' else 0)
        self.assertIsNone(self.nav.trajectory_preview)

    def test_blocked_opening_keeps_turning_and_new_obstacle_stops_drive(self):
        def step(time, obstacles=()):
            estimate = StateEstimate(time, (0,0,0), np.zeros((3,3)))
            output = observations(time, (0,0,0), camera=False, obstacles=obstacles)
            return self.nav.motion_request(estimate, output), output
        step(0)
        request, _ = step(self.settings.search_spin_s+.02, [(.5,0)])
        self.assertEqual(request.mode, 'SCAN')
        request, _ = step(self.settings.search_spin_s+.04)
        self.assertEqual(request.mode, 'EXPLORE')
        output = observations(request.time, (0,0,0), camera=False, obstacles=[(.35,0)])
        self.assertFalse(self.nav.allow_motion(request, .15, 0, output))
        self.assertEqual(self.nav.phase, 'SCAN')

    def test_stale_lidar_stops_exploration_and_pauses_its_timer(self):
        for time in (0., 1., 10.):
            estimate = StateEstimate(time, (0,0,0), np.zeros((3,3)))
            request = self.nav.motion_request(estimate, observations(time, (0,0,0), camera=False, scan_time=-1))
            self.assertTrue(request.stop_requested)
        self.assertEqual(self.nav._search_remaining, self.settings.search_spin_s)

    def test_stale_lidar_still_stops_after_camera_loss(self):
        nav = Navigator(self.settings)
        nav.motion_request(state(0, (0,0,0)), observations(0, (0,0,0)))
        request = nav.motion_request(state(300, (0,0,0)),
            observations(300, (0,0,0), camera=False, scan_time=299))
        self.assertTrue(request.stop_requested)
        self.assertIn('LiDAR', nav.status)

    def test_first_detection_starts_approach(self):
        for acquire_time in (0., 30.):
            self.nav = Navigator(self.settings)
            if acquire_time:
                estimate = StateEstimate(0, (0, 0, 0), np.zeros((3, 3)))
                self.nav.motion_request(estimate, observations(0, (0, 0, 0), camera=False))
            request = self.nav.motion_request(state(acquire_time, (0, 0, 0)),
                                              observations(acquire_time, (0, 0, 0)))
            self.assertEqual(request.mode, 'APPROACH')
            self.assertIsNotNone(request.reference)

    def test_returning_camera_updates_target_without_restarting_navigation(self):
        self.nav.motion_request(state(0, (0,0,0)), observations(0, (0,0,0)))
        request = self.nav.motion_request(state(300, (.1,0,0)), observations(300, (.1,0,0), camera=False))
        self.assertEqual(request.mode, 'APPROACH')
        # This state represents a later accepted EKF correction, in the same frame.
        estimate = replace(state(301, (.12,0,0)), dock_pose_local=(2.1,.02,math.pi))
        request = self.nav.motion_request(estimate, observations(301, (.12,0,0)))
        self.assertEqual(request.mode, 'APPROACH')
        np.testing.assert_allclose(self.nav.dock_pose, estimate.dock_pose_local)
        self.assertFalse(request.stop_requested)

    def test_factory_needs_no_spawn_pose_or_marker_map(self):
        from track_a_mobile_robot import main
        model = SimpleNamespace(sensor_noise=np.array([.001, .001]),
                                sensor=lambda name: SimpleNamespace(id=int('right' in name)))
        with patch.object(main, 'Perception', return_value=Mock()):
            estimator, _, _ = main.create_demo(model, load_config().estimation)
        self.assertFalse(estimator.initialised)
        self.assertFalse(hasattr(estimator, 'marker_map'))


class DisplayTests(unittest.TestCase):
    def test_world_preview_is_invariant_to_the_local_odometry_frame(self):
        # Changing the odometry origin must not move the relative plan in the GUI.
        model = mujoco.MjModel.from_xml_string('<mujoco/>')
        window = SimpleNamespace(user_scn=mujoco.MjvScene(model, maxgeom=200), lock=nullcontext)
        body = SimpleNamespace(xpos=np.array([10., 20., 0.]),
                               xmat=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]))
        data = SimpleNamespace(body=lambda name: body)
        nav = SimpleNamespace(settings=navigation_settings(), phase='APPROACH',
                              trajectory_preview=np.array([[0.,0.,0.],[1.,0.,0.],[.5,0.,0.]]))
        display = SensorDisplay()
        display._draw_scene(window, data, state(0, (0,0,0)), nav)
        positions = np.array([g.pos.copy() for g in window.user_scn.geoms[:window.user_scn.ngeom]])
        np.testing.assert_allclose(positions[49], (10,20.5,.025), atol=1e-6)
        angle = .7
        rotation = np.array([[math.cos(angle),-math.sin(angle)], [math.sin(angle),math.cos(angle)]])
        preview = nav.trajectory_preview.copy()
        preview[:,:2] = preview[:,:2] @ rotation.T + [3,4]
        preview[:,2] += angle
        nav.trajectory_preview = preview
        display._draw_scene(window, data, state(0, (3,4,angle)), nav)
        shifted = np.array([g.pos.copy() for g in window.user_scn.geoms[:window.user_scn.ngeom]])
        np.testing.assert_allclose(shifted, positions, atol=1e-6)

    def test_pending_overlay_is_latest_only_while_upload_blocks(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        class SlowViewer:
            calls = []
            def is_running(self):
                return True
            def set_images(self, images):
                self.calls.append(images)
                if len(self.calls) == 1:
                    entered.set()
                    release.wait(2)
                if images == 20:
                    finished.set()
            def set_texts(self, texts):
                pass
        window = SlowViewer()
        publisher = OverlayPublisher(window)
        try:
            publisher.submit(0, '')
            self.assertTrue(entered.wait(1))
            for i in range(1, 21):
                publisher.submit(i, '')
            self.assertEqual(publisher.pending.qsize(), 1)
            self.assertEqual(window.calls, [0])
            release.set()
            self.assertTrue(finished.wait(1))
            self.assertEqual(window.calls, [0, 20])
        finally:
            release.set()
            publisher.close()
        self.assertFalse(publisher.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
