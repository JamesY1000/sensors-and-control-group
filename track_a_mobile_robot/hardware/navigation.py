"""Random initial search, then LQI to an offset pose and direct final PBVS."""

import math
import random
from dataclasses import replace

import numpy as np

from ..navigation import Navigator, MotionRequest, corridor_is_clear, dock_target_pose, grouped_scan, motion_is_clear


class HardwareNavigator(Navigator):
    """Reuse trajectory/avoidance primitives without the simulation's phase sequence."""

    def __init__(self, settings, *, future_tolerance_s=.1, rng=None, obstacle_checks_enabled=True):
        super().__init__(settings)
        self.future_tolerance_s = future_tolerance_s
        self.obstacle_checks_enabled = obstacle_checks_enabled
        self._rng = random.Random() if rng is None else rng
        self._start_search_turn()
        self.status = 'Searching for the first usable dock marker'
        self._following_boundary = False
        self._finishing_pose = False
        self._joining_axis = None
        self._camera_has_marker = False
        self._last_missing_marker_time = -math.inf
        self._last_camera_time = -math.inf
        self._last_visual_pose = None
        self._marker_search_active = False
        self._marker_wait_since = None
        self._marker_search_direction = None

    def observe_camera(self, perception):
        """Record visibility before the runner replaces raw poses with EKF acceptances."""
        if perception.new_camera_frame:
            self._last_camera_time = perception.camera.time
            self._camera_has_marker = perception.dock_observation is not None
            if not self._camera_has_marker:
                self._last_missing_marker_time = perception.camera.time
                self._dock_observation_times.clear()

    def marker_status(self, time):
        age = time - self._last_dock_observation_time
        age_text = f'{max(0., age):.2f}s' if math.isfinite(age) else 'never'
        return (f'Marker: {"detected" if self._camera_has_marker else "NOT detected"}; '
                f'last accepted={age_text}; confirmations={len(self._dock_observation_times)}')

    def _update_visual_evidence(self, time, perception):
        # Accepted images can arrive from fusion after their encoder bracket;
        # they need not coincide with a new camera frame on this control tick.
        observation = perception.dock_observation
        if (observation is not None and self._last_dock_observation_time < observation.time
                and observation.time > self._last_missing_marker_time
                and -self.future_tolerance_s <= time-observation.time <= self.settings.reacquire_observation_age_s):
            self._last_dock_observation_time = observation.time
            self._last_visual_pose = observation.pose_base
            self._dock_observation_times.append(observation.time)
        while (self._dock_observation_times and
               time-self._dock_observation_times[0] > self.settings.reacquire_observation_age_s):
            self._dock_observation_times.popleft()

    def _visual_ready(self):
        return self._camera_has_marker and len(self._dock_observation_times) >= self.settings.reacquire_confirmations

    def _find_marker(self, time, state):
        """Hold, then rotate in place until visual feedback returns; retain the phase."""
        if self._visual_ready():
            self._marker_search_active = False
            self._marker_wait_since = None
            self._marker_search_direction = None
            self._trajectory.reset()
            return None
        self._marker_search_active = True
        prefix = 'PBVS paused' if self.phase == 'DOCK' else 'At offset / marker search'
        if self._camera_has_marker and time-self._last_camera_time <= self.settings.reacquire_observation_age_s:
            # Stop as soon as a usable candidate enters view so it stays in frame
            # while delayed fusion collects the required accepted observations.
            self._marker_wait_since = time
            return self._stop(time, f'{prefix}: marker visible; waiting for accepted detections')
        if self._marker_wait_since is None:
            self._marker_wait_since = time
        if time-self._marker_wait_since < self.settings.reacquire_wait_s:
            return self._stop(time, f'{prefix}: waiting for fresh accepted marker detections')
        if self._marker_search_direction is None:
            # Use position bearing, not the marker's orientation. The fused
            # relative position accounts for encoder motion since its last sighting.
            # Latch once per search so crossing the estimated bearing without a
            # detection continues the sweep instead of reversing back and forth.
            pose = state.dock_pose_base
            if pose is None:
                pose = self._last_visual_pose
            bearing = math.atan2(pose[1], pose[0]) if pose is not None else 0.
            self._marker_search_direction = -1 if bearing < 0. else 1
        self._trajectory.reset()
        direction = 'left' if self._marker_search_direction > 0 else 'right'
        self.status = f'{prefix}: slowly searching {direction} for marker'
        # SCAN selects the existing bounded spot-turn controller, without changing
        # NAVIGATE/DOCK or activating the initial random forward exploration.
        return MotionRequest(time, 'SCAN',
                             turn_rate_rad_s=self._marker_search_direction*.5*self.settings.turn_speed_rad_s)

    def _start_search_turn(self):
        self.phase = 'SCAN'
        self._search_remaining = self._rng.uniform(.5, 1.) * self.settings.search_spin_s
        self._search_turn = self._rng.choice((-1, 1)) * self.settings.turn_speed_rad_s

    def _explore(self, time, perception):
        if self._search_remaining <= 0:
            if self.phase == 'EXPLORE':
                self._start_search_turn()
            else:
                duration = self._rng.uniform(.5, 1.) * self.settings.search_drive_s
                distance = self.settings.drive_speed_mps * duration
                points = grouped_scan(perception.lidar_points_base, self.settings.boundary_join_gap_m)
                if distance + self.settings.clearance_m + .04 < perception.lidar.range_max_m and corridor_is_clear(
                        points, [distance, 0.], self.settings.clearance_m + .04):
                    self.phase = 'EXPLORE'
                    self._search_remaining = duration
                else:
                    self._start_search_turn()
        self._search_active = True
        if self.phase == 'SCAN':
            self.status = 'Random spot turn: searching for first marker'
            return MotionRequest(time, 'SCAN', turn_rate_rad_s=self._search_turn)
        self.status = 'Random short straight drive: searching for first marker'
        return MotionRequest(time, 'EXPLORE', self.settings.drive_speed_mps)

    def allow_motion(self, request, v, omega, perception):
        if not self.obstacle_checks_enabled:
            return not request.stop_requested
        allowed = super().allow_motion(request, v, omega, perception)
        if not allowed and request.mode == 'EXPLORE':
            self._start_search_turn()
        return allowed

    def pause_for_sensors(self, time, reason):
        # A long sensor outage must not consume a detour's stall/time budget.
        # Plan again from the current pose and scan when publication resumes.
        self._detour.reset()
        self._following_boundary = self._motion_blocked = False
        self._finishing_pose = False
        self._marker_wait_since = None
        return self._stop(time, reason)

    def motion_request(self, state, perception, *, stop_requested=False):
        time = float(state.time)
        if not math.isfinite(time) or not np.isfinite(state.pose_local).all():
            raise ValueError('Estimated pose and time must be finite')
        if self._previous_time is not None and time <= self._previous_time:
            raise ValueError('Navigation timestamps must increase; recreate navigation on reset')
        dt = 0. if self._previous_time is None else time - self._previous_time
        self._previous_time = time
        self._update_visual_evidence(time, perception)
        if self._search_active:
            self._search_remaining -= dt
        self._search_active = False
        if stop_requested:
            return self.pause_for_sensors(time, 'Stop requested')
        if self.phase == 'DONE':
            return self._stop(time, 'Docked')

        scan_age = time - perception.lidar.time
        if not math.isfinite(scan_age) or not -self.future_tolerance_s <= scan_age <= self.settings.lidar_timeout_s:
            return self.pause_for_sensors(time, 'Waiting for fresh LiDAR')
        if not self.obstacle_checks_enabled:
            # Empty only the planning view; the runner retains every measured
            # range and point for display/logging. All route/footprint checks
            # then see no obstacles, including the final corridor and NaN hits.
            perception = replace(perception, lidar=replace(perception.lidar, ranges_m=()),
                                 lidar_points_base=np.empty((0, 2)))
        if perception.lidar_too_close or not motion_is_clear(
                perception.lidar_points_base, 0, 0, self.settings.clearance_m):
            return self._stop(time, 'Obstacle inside robot clearance')
        if self._marker_search_active:
            request = self._find_marker(time, state)
            if request is not None:
                return request
        if state.dock_pose_local is None:
            if self.phase in ('SCAN', 'EXPLORE'):
                return self._explore(time, perception)
            return self._find_marker(time, state) or self._stop(time, 'Waiting for dock pose after encoder gap')

        if self.phase in ('SCAN', 'EXPLORE'):
            self.phase = 'NAVIGATE'
            self._trajectory.reset()

        self.dock_pose = np.asarray(state.dock_pose_local)
        self.coarse_pose = dock_target_pose(self.dock_pose, self.settings.coarse_distance_m)
        self.docking_pose = dock_target_pose(self.dock_pose, self.settings.dock_distance_m)
        offset = dock_target_pose(state.dock_pose_base, self.settings.coarse_distance_m)
        final = dock_target_pose(state.dock_pose_base, self.settings.dock_distance_m)
        tolerance = self.settings.position_tolerance_m
        heading_tolerance = math.radians(self.settings.heading_tolerance_deg)
        points = grouped_scan(perception.lidar_points_base, self.settings.boundary_join_gap_m)

        if self.phase == 'NAVIGATE':
            offset_tolerance = getattr(self.settings, 'offset_position_tolerance_m', tolerance)
            offset_heading_tolerance = math.radians(getattr(
                self.settings, 'offset_heading_tolerance_deg', self.settings.heading_tolerance_deg))
            # Join the dock normal 30 cm outside the coarse pose when starting
            # off-axis or too close. The final LQI leg then arrives head-on.
            normal = np.array([math.cos(self.dock_pose[2]), math.sin(self.dock_pose[2])])
            relative = np.asarray(state.pose_local[:2]) - self.dock_pose[:2]
            along = float(relative @ normal)
            across = float(relative @ np.array([-normal[1], normal[0]]))
            if self._joining_axis is None:
                self._joining_axis = (np.linalg.norm(offset[:2]) > offset_tolerance and
                                      (abs(across) > tolerance or along < self.settings.coarse_distance_m-tolerance))
            elif (not self._joining_axis and np.linalg.norm(offset[:2]) > offset_tolerance
                  and abs(across) > 2*tolerance):
                # A detour or pose correction can move us off the entry line.
                self._joining_axis = True
                self._trajectory.reset()
            distance = self.settings.coarse_distance_m + (.3 if self._joining_axis else 0.)
            navigation_pose = dock_target_pose(self.dock_pose, distance)
            navigation_offset = dock_target_pose(state.dock_pose_base, distance)
            # Keep centreline staging precise; relax only the actual PBVS handover.
            arrival_tolerance = tolerance if self._joining_axis else offset_tolerance
            arrival_heading_tolerance = heading_tolerance if self._joining_axis else offset_heading_tolerance
            if np.linalg.norm(navigation_offset[:2]) <= arrival_tolerance:
                # Finish the actual offset POSE with LQI, inside NAVIGATE.
                # Never translate during this turn, even after small EKF corrections.
                self._detour.reset()
                self._following_boundary = self._motion_blocked = False
                if not self._finishing_pose:
                    self._trajectory.reset()
                self._finishing_pose = True
                if abs(navigation_offset[2]) > arrival_heading_tolerance:
                    reference = self._trajectory.generate_reference(
                        time, np.asarray(state.pose_local), navigation_pose, 0., turn_only=True)
                    self.status = f'LQI: facing dock on centreline; heading error={math.degrees(navigation_offset[2]):+.1f} deg'
                    return MotionRequest(time, 'NAVIGATE', reference=reference)
                if self._joining_axis:
                    self._joining_axis = False
                    self._finishing_pose = False
                    return self._stop(time, 'LQI: on dock centreline; starting head-on approach to offset')
                if not self._visual_ready():
                    return self._find_marker(time, state)
                visual_offset = dock_target_pose(self._last_visual_pose, self.settings.coarse_distance_m)
                if np.linalg.norm(visual_offset[:2]) > offset_tolerance or abs(visual_offset[2]) > offset_heading_tolerance:
                    return self._stop(time, 'At estimated offset; camera has not confirmed head-on alignment '
                                      f'({np.linalg.norm(visual_offset[:2]):.3f} m / '
                                      f'{abs(math.degrees(visual_offset[2])):.1f} deg; limits '
                                      f'{offset_tolerance:.3f} m / {math.degrees(offset_heading_tolerance):.1f} deg)')
                if not corridor_is_clear(points, final[:2], self.settings.clearance_m):
                    return self._stop(time, 'At offset pose; final docking corridor blocked')
                self.phase = 'DOCK'
                self._trajectory.reset()
            else:
                if self._finishing_pose:
                    self._trajectory.reset()
                self._finishing_pose = False
                result, target = self._detour.update(
                    time, state.pose_local, navigation_pose, perception.lidar_points_base,
                    perception.lidar.range_max_m, forced=self._motion_blocked)
                self._motion_blocked = False
                following = result not in ('clear', 'approach')
                if following != self._following_boundary:
                    self._trajectory.reset()
                self._following_boundary = following
                if result == 'clear':
                    self.status = ('LQI: joining dock centreline' if self._joining_axis
                                   else 'LQI: head-on approach to offset pose')
                    return self._tracking_request(state, distance,
                                                  self.settings.drive_speed_mps, perception, navigation_pose, direct=True)
                if result == 'approach':
                    self.status = 'LQI: approaching distant obstruction along clear route'
                    return self._tracking_request(state, distance,
                                                  self.settings.drive_speed_mps, perception, target, direct=True)
                if target is None:
                    return self._stop(time, self._detour.reason if result == 'blocked'
                                      else 'Boundary blocked; waiting for clearance')
                if not self._joining_axis:
                    self._joining_axis = True
                    self._detour.reset()
                    return self._stop(time, 'LQI: obstruction on final approach; replanning via dock centreline')
                self.status = f'LQI: avoiding obstacle on {"left" if self._detour.side > 0 else "right"}'
                return self._tracking_request(state, None, self.settings.detour_speed_mps,
                                              perception, target, direct=False)

        # A remembered dock is insufficient for visual docking or declaring DONE.
        if not self._visual_ready():
            return self._find_marker(time, state)
        if np.linalg.norm(final[:2]) <= tolerance and abs(final[2]) <= heading_tolerance:
            visual_final = dock_target_pose(self._last_visual_pose, self.settings.dock_distance_m)
            if np.linalg.norm(visual_final[:2]) > tolerance or abs(visual_final[2]) > heading_tolerance:
                return self._stop(time, 'At estimated dock; waiting for camera confirmation of final pose')
            self.phase = 'DONE'
            return self._stop(time, 'Docked')
        if not corridor_is_clear(points, final[:2], self.settings.clearance_m):
            return self._stop(time, 'PBVS docking corridor blocked')
        self.status = (f'PBVS: docking directly; error={np.linalg.norm(final[:2]):.3f} m, '
                       f'heading={math.degrees(final[2]):+.1f} deg')
        return MotionRequest(time, 'DOCK', self.settings.dock_speed_mps, self.settings.dock_distance_m)
