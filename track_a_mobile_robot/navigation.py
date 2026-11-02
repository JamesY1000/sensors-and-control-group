"""
Our navigator should essentially choose between exploring, approaching and docking.

Currently its inputs are:
- Estimated robot and dock poses (from estimation)
- Robot-frame LiDAR points and scan timestamps (from perception)
- Docking offsets, speeds, clearance and LiDAR timeout settings

It has 4 main stages:
1. Explore:
   - Send a timed turning command while looking for a marker
   - When forward clearance is available, drive briefly and scan again
   - The first usable marker starts odometry and immediately starts the approach
2. Approach:
   - Track a full direct trajectory to the coarse dock region using LQI
   - No final heading is required before handing over using the estimated dock
   - Follow the sensed obstacle boundary when the direct approach is obstructed
3. Reacquire:
   - Turn in place with LQI to face the estimated dock
   - Confirm recent accepted camera poses before allowing PBVS inward travel
   - Try a limited angular sweep, then stop if reacquisition times out
4. Dock:
   - PBVS first reaches and aligns at the approach pose
   - Then PBVS moves to the final offset until position and heading are in tolerance

It outputs:
- A MotionRequest containing the behaviour, reference pose/velocity and speed limit
- Current phase and status for the display

Coarse trajectories use local estimated poses; PBVS uses the relative dock estimate.
There are no grids or global route searches. Boundary following keeps one
passing side until the corridor clears, with a timeout if it cannot progress.
LiDAR also checks the actual command; final docking stops when obstructed.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .estimation import StateEstimate
from .perception import PerceptionOutput
from .scene_config import NavigationConfig
from .state_model import wrap_angle


@dataclass(frozen=True)
class TrajectoryReference:
    """Store a sample of a turning or straight-driving trajectory in local odometry."""

    time: float
    pose_local: tuple[float, float, float]
    v_mps: float
    omega_rad_s: float
    kind: str
    segment_id: int
    direction: int = 1

    def __post_init__(self):
        if self.kind not in ('TURN', 'DRIVE'):
            raise ValueError('Unknown trajectory segment')
        if self.direction not in (-1, 1):
            raise ValueError('Trajectory direction must be -1 or 1')
        if len(self.pose_local) != 3 or not all(math.isfinite(value) for value in
                (self.time, *self.pose_local, self.v_mps, self.omega_rad_s)):
            raise ValueError('Reference pose, time and velocities must be finite')


class MotionTrajectory:
    """Generate spot turns and straight drives for approach and obstacle detours."""

    def __init__(self, settings):
        self.settings = settings
        self._segment_id = 0
        self.reset()

    def reset(self):
        self._goal = None
        self._kind = None
        self._reference = None
        self._progress = 0.

    @property
    def preview(self):
        """Copy the active segment's start, end and current reference for display."""
        if self._reference is None:
            return None
        return np.vstack((self._start_pose, self._start_pose + self._delta,
                          self._reference.pose_local))

    def path(self, remaining=False):
        """Return segment endpoints for continuous line-clearance checks/display."""
        if self._reference is None:
            return None
        return np.array([self._pose_at(self._progress if remaining else 0.),
                         self._pose_at(1.)])

    def _pose_at(self, progress):
        """Interpolate position for a drive or heading for a spot turn."""
        pose = self._start_pose + progress*self._delta
        pose[2] = wrap_angle(pose[2])
        return pose

    def _start_segment(self, time, pose, target, speed, max_distance=math.inf, turn_only=False):
        """Turn towards the target, then drive straight with a speed-limited duration."""
        self._segment_id += 1
        self._start_time = time
        self._start_pose = np.array(pose, dtype=float)
        self._goal = np.array(target, dtype=float)
        delta = target[:2] - pose[:2]
        if turn_only:
            # Reacquisition must never translate, including after EKF corrections.
            delta = np.zeros(2)
        distance = float(np.linalg.norm(delta))
        # Direct travel reaches the full target. Only detours limit the distance.
        # Keep the endpoint fixed while tracking instead of chasing the robot.
        if distance > max_distance:
            delta = delta * max_distance/distance
            distance = max_distance
        heading = float(target[2])
        direction = 1
        if distance > self.settings.position_tolerance_m:
            heading = math.atan2(delta[1], delta[0])
            if (abs(wrap_angle(heading - pose[2])) > math.pi / 2 and
                    abs(wrap_angle(target[2] - pose[2])) < math.pi / 4):
                direction = -1
                heading = wrap_angle(heading + math.pi)
                speed = min(speed, self.settings.dock_speed_mps)
        angle = wrap_angle(heading - pose[2])
        # Align first; LQI removes the small remaining heading error during travel.
        max_bearing = math.radians(self.settings.heading_tolerance_deg)/2
        self._direction = direction
        if distance <= self.settings.position_tolerance_m or abs(angle) > max_bearing:
            self._kind = 'TURN'
            self._delta = np.array([0., 0., angle])
            self._signed_distance = 0.
            self._duration = max(.02, 1.5*abs(angle)/(.8*self.settings.turn_speed_rad_s),
                                 math.sqrt(6*abs(angle)/.8))
        else:
            self._kind = 'DRIVE'
            self._start_pose[2] = heading
            self._signed_distance = direction*distance
            self._delta = np.array([*delta, 0.])
            # Cubic progress starts/stops smoothly. Limit speed and acceleration,
            # reserving 20% of the speed limit for tracking feedback.
            self._duration = max(.02, 1.5*distance/(.8*speed), math.sqrt(6*distance/.15))

    def _needs_new_segment(self, time, pose, target, turn_only):
        """Restart for a changed goal, large tracking error or a completed segment."""
        if self._goal is None:
            return True
        # Camera corrections can move the local target. Small changes are
        # absorbed at the next segment; larger changes restart from the estimate.
        restart = (np.linalg.norm(target[:2] - self._goal[:2]) > .05 or
                   abs(wrap_angle(target[2] - self._goal[2])) > math.radians(5))
        restart |= turn_only and self._kind != 'TURN'
        if self._reference is not None:
            reference = np.asarray(self._reference.pose_local)
            restart |= (np.linalg.norm(pose[:2] - reference[:2]) > .15 or
                        abs(wrap_angle(pose[2] - reference[2])) > math.radians(25))
        if time >= self._start_time + self._duration:
            # Wait for the tracker to settle before advancing to the next
            # segment. Integral action can then overcome low-speed friction.
            end = self._start_pose + self._delta
            settled = (abs(wrap_angle(pose[2] - end[2])) <= math.radians(self.settings.heading_tolerance_deg) / 2
                       if self._kind == 'TURN' else
                       np.linalg.norm(pose[:2] - end[:2]) <= self.settings.position_tolerance_m)
            restart |= settled
            # A stationary differential drive cannot remove lateral error.
            # If the segment ends with lateral error, generate a new segment.
            if self._kind == 'DRIVE':
                c, s = math.cos(end[2]), math.sin(end[2])
                along, across = np.array([[c, s], [-s, c]]) @ (pose[:2] - end[:2])
                restart |= abs(along) <= self.settings.position_tolerance_m and abs(across) > self.settings.position_tolerance_m
        return bool(restart)

    def generate_reference(self, time, pose, target, speed, *,
                           max_distance=math.inf, turn_only=False) -> TrajectoryReference:
        """Return the desired pose and velocities for this control timestamp.

        Keep the active segment until it finishes or needs replacing. Evaluate
        it analytically rather than storing a list of timed trajectory points.
        path() separately supplies points for collision checks and the display.
        """
        pose, target = np.asarray(pose, dtype=float), np.asarray(target, dtype=float)
        if self._needs_new_segment(time, pose, target, turn_only):
            self._start_segment(time, pose, target, speed, max_distance, turn_only)

        # Cubic progress starts and ends at zero speed. Its time derivative
        # converts the segment's distance and heading change into v_ref/omega_ref.
        u = float(np.clip((time - self._start_time) / self._duration, 0, 1))
        progress = 3 * u**2 - 2 * u**3
        rate = 6 * u * (1 - u) / self._duration
        self._progress = progress
        reference = self._pose_at(progress)
        self._reference = TrajectoryReference(time, tuple(reference), self._signed_distance * rate,
                                              self._delta[2] * rate, self._kind, self._segment_id, self._direction)
        return self._reference


@dataclass(frozen=True)
class MotionRequest:
    """Supply a timed exploration command, an LQI reference or a PBVS dock offset."""

    time: float
    mode: str
    speed_limit_mps: float = 0.0
    dock_offset_m: float | None = None
    reference: TrajectoryReference | None = None
    turn_rate_rad_s: float = 0.0

    @property
    def visual_servo(self) -> bool:
        return self.mode in ('ALIGN', 'DOCK')

    @property
    def stop_requested(self) -> bool:
        return self.mode == 'STOP'

    def __post_init__(self) -> None:
        if self.mode not in ('STOP', 'SCAN', 'EXPLORE', 'APPROACH', 'AVOID', 'REACQUIRE', 'ALIGN', 'DOCK'):
            raise ValueError('Unknown motion mode')
        if not all(math.isfinite(v) for v in (self.time, self.speed_limit_mps, self.turn_rate_rad_s)):
            raise ValueError('Motion request values must be finite')
        if self.speed_limit_mps < 0:
            raise ValueError('Speed limit cannot be negative')
        if self.mode in ('APPROACH', 'ALIGN', 'DOCK') and (self.dock_offset_m is None or
                                  not math.isfinite(self.dock_offset_m) or self.dock_offset_m <= 0):
            raise ValueError('Visual docking requires a positive dock offset')


def dock_target_pose(dock_pose, offset: float) -> np.ndarray:
    """Return the offset target in the same frame as the supplied dock pose.

    Phase checks use the robot frame; trajectory generation uses local odometry.
    """
    x, y, yaw = dock_pose
    # Dock +X points out from its face. Place the target along that axis and
    # reverse its heading by pi so the robot faces into the dock.
    return np.array([x + offset * math.cos(yaw), y + offset * math.sin(yaw),
                     wrap_angle(yaw + math.pi)])


def motion_is_clear(points_base: np.ndarray, v: float, omega: float, clearance: float,
                    horizon: float = 1.0) -> bool:
    """Check the swept footprint of the current command, without planning a route."""
    points = np.asarray(points_base, dtype=float).reshape(-1, 2)
    if not np.isfinite(points).all():
        return False
    if not len(points):
        return True
    if abs(omega) < 1e-9 or abs(v) < 1e-9:
        return corridor_is_clear(points, [v * horizon, 0], clearance)
    times = np.linspace(0.0, horizon, 21)
    centres = np.column_stack((v / omega * np.sin(omega * times),
                               v / omega * (1 - np.cos(omega * times))))
    # Half a sample's travel covers the spaces between sampled footprint centres.
    margin = abs(v) * horizon / 40
    distances = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
    return bool(np.all(distances >= clearance + margin))


def path_is_clear(points_base, path_base, clearance):
    """Check the circular footprint continuously along each straight path segment."""
    points, path = np.asarray(points_base), np.asarray(path_base)
    if not np.isfinite(points).all() or not np.isfinite(path).all():
        return False
    if not len(points):
        return True
    starts, edges = path[:-1], np.diff(path, axis=0)
    delta = points[:, None, :] - starts
    fraction = np.clip(np.sum(delta*edges, axis=2) / np.maximum(np.sum(edges**2, axis=1), 1e-12), 0, 1)
    distance = np.linalg.norm(delta - fraction[:, :, None]*edges, axis=2)
    # Retain 1 cm of extra clearance beyond the configured footprint margin.
    return bool(np.all(distance >= clearance + .01))


def corridor_is_clear(points, end, radius):
    """Check a circular robot swept along the segment from the origin to end."""
    points, end = np.asarray(points).reshape(-1, 2), np.asarray(end)
    if not np.isfinite(points).all():
        return False
    length2 = float(end @ end)
    fraction = np.clip(points @ end / length2, 0, 1) if length2 > 1e-12 else np.zeros(len(points))
    offsets = points - fraction[:, None] * end
    return bool(np.all(np.sum(offsets * offsets, axis=1) >= radius * radius))


def grouped_scan(points, join_gap):
    """Connect nearby neighbouring returns, leaving wider openings unfilled.

    These short sampled edges describe a local obstacle boundary, not a grid.
    Robot clearance is applied separately, so joining legs does not enlarge walls.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 2 or not np.isfinite(points).all():
        return points
    points = points[np.argsort(np.arctan2(points[:, 1], points[:, 0]))]
    delta = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(delta, axis=1)
    edges = (lengths > .05) & (lengths <= join_gap)
    # Never bridge opposite bearings through the robot itself.
    edges &= np.sum(points * np.roll(points, -1, axis=0), axis=1) > 0
    additions = [points]
    for start, vector, length in zip(points[edges], delta[edges], lengths[edges]):
        fractions = np.arange(1, math.ceil(length / .05)) / math.ceil(length / .05)
        additions.append(start + fractions[:, None] * vector)
    return np.vstack(additions)


class BoundaryFollower:
    """Choose short targets along one side of a locally connected scan boundary."""

    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.active = self.failed = False
        self.reason = ''
        self.target = self.anchor = None
        self.side = 0
        self._clear_since = None
        self._scan = self._boundary = None

    def exhausted(self, time):
        return self.failed or (self.active and time - self._started > self.settings.detour_timeout_s)

    def update(self, time, pose, goal, points, range_max, *, forced=False):
        pose, goal = np.asarray(pose), np.asarray(goal)
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all():
            return 'waiting', None
        # The scan is slower than control. Reuse its connected edges between scans.
        if self._scan is None or not np.array_equal(points, self._scan):
            self._scan = points.copy()
            self._boundary = grouped_scan(points, self.settings.boundary_join_gap_m)
        boundary = self._boundary
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        delta = rotation.T @ (goal[:2] - pose[:2])
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-9)
        radius = self.settings.clearance_m + .04  # Includes edge-sampling margin.
        lookahead = min(distance, max(0., range_max - radius))
        if not self.active:
            if not len(boundary):
                return 'clear', None
            # Check all observed obstacles along the direct route, including
            # distant blockers. Unseen space still needs checking on later scans.
            if corridor_is_clear(boundary, delta, radius) and not forced:
                return 'clear', None
            along = np.clip(boundary @ direction, 0., distance)
            blocking = boundary[np.linalg.norm(boundary-along[:, None]*direction, axis=1) <= radius]
            contacts = blocking if len(blocking) else boundary
            contact = contacts[np.argmin(np.linalg.norm(contacts, axis=1))]
            # Seeing a distant blockage does not mean we are at its boundary.
            # First track the clear prefix of the direct route, stopping short
            # by enough room to turn. Re-evaluate this from each new scan.
            stand_off = max(1., radius + self.settings.detour_step_m)
            advance = float(contact @ direction) - stand_off
            if not forced and advance > self.settings.position_tolerance_m:
                end = direction * advance
                if corridor_is_clear(boundary, end, radius):
                    return 'approach', np.array([*(pose[:2] + rotation @ end),
                        wrap_angle(pose[2] + math.atan2(direction[1], direction[0]))])
            self.active = True
            self._started = self._progress_time = time
            self._hit = self._progress_pose = pose[:2].copy()
            self._hit_distance = distance
            self._travel = 0.
            # Start on the obstacle blocking the goal corridor, not an unrelated
            # nearer wall beside/behind the robot. Remember that contact locally.
            self.anchor = pose[:2] + rotation @ contact

        # Leave only after sustained clearance and progress beyond the hit point.
        # Removing the obstruction entirely also permits a stopped robot to resume.
        progress = distance < self._hit_distance - .1 or not len(boundary) or self.failed
        if progress and corridor_is_clear(boundary, direction * lookahead, radius + .04) and not forced:
            if self._clear_since is None:
                self._clear_since = time
            if time - self._clear_since >= .4:
                self.reset()
                return 'clear', None
        else:
            self._clear_since = None
        moved = np.linalg.norm(pose[:2] - self._progress_pose)
        if moved >= .1:
            self._travel += moved
            self._progress_pose = pose[:2].copy()
            self._progress_time = time
        looped = self._travel > 2. and np.linalg.norm(pose[:2] - self._hit) < .2
        stalled = time - self._progress_time > self.settings.progress_timeout_s
        self.failed = self.exhausted(time) or looped or stalled
        if self.failed:
            if not self.reason:
                self.reason = ('Boundary loop detected; stopped' if looped else
                               'Boundary stalled; no position progress' if stalled else
                               'Detour time budget exhausted; stopped')
            self.target = None
            return 'blocked', None

        # Hold a target until reached or newly obstructed, rather than replacing it
        # on each noisy scan. Physical clearance still guards every actual command.
        if self.target is not None:
            local = rotation.T @ (self.target[:2] - pose[:2])
            if (self.settings.position_tolerance_m < np.linalg.norm(local) <= self.settings.detour_step_m + .15
                    and corridor_is_clear(boundary, local, radius - .02) and not forced):
                return 'avoid', self.target
            self.target = None
        if not len(boundary):
            return 'waiting', None

        # Advance the contact along a local section of the same boundary. Choosing
        # the globally nearest hit here can switch from furniture to the room wall.
        anchor_base = rotation.T @ (self.anchor - pose[:2])
        contact_distances = np.linalg.norm(boundary-anchor_base, axis=1)
        nearby = boundary[contact_distances <= max(.5, self.settings.detour_step_m + .15)]
        nearest = (nearby[np.argmin(np.linalg.norm(nearby, axis=1))] if len(nearby)
                   else boundary[np.argmin(contact_distances)])
        self.anchor = pose[:2] + rotation @ nearest
        separation = max(float(np.linalg.norm(nearest)), 1e-9)
        normal = nearest / separation
        tangent = np.array([-normal[1], normal[0]])
        candidates = []
        for side in ((self.side,) if self.side else (1, -1)):
            follow = side*tangent + np.clip(2*(separation-radius-.15), -.8, .8)*normal
            bearing = math.atan2(follow[1], follow[0])
            for offset in (0, 15, -15, 30, -30, 60, -60, 90, -90):
                angle = wrap_angle(bearing + math.radians(offset))
                for step in (self.settings.detour_step_m, self.settings.detour_step_m/2):
                    end = step * np.array([math.cos(angle), math.sin(angle)])
                    if corridor_is_clear(boundary, end, radius):
                        score = abs(offset)/90 + .2*np.linalg.norm(delta-end) - step
                        candidates.append((score, side, angle, end))
        if not candidates:
            return 'waiting', None
        _, self.side, angle, end = min(candidates, key=lambda item: item[0])
        self.target = np.array([*(pose[:2] + rotation @ end), wrap_angle(pose[2] + angle)])
        return 'avoid', self.target


class Navigator:
    """Maintain docking phases, generate references and check motion clearance."""

    def __init__(self, settings: NavigationConfig) -> None:
        self.settings = settings
        self.phase = 'SCAN'
        self.status = 'Scanning for a usable dock marker'
        self._search_remaining = settings.search_spin_s
        self._search_active = False
        self.dock_pose = self.coarse_pose = self.approach_pose = self.docking_pose = None
        self._previous_time = None
        self._trajectory = MotionTrajectory(settings)
        self._detour = BoundaryFollower(settings)
        self._motion_blocked = False
        self._path_scan_time = None
        self._path_obstacles = np.empty((0, 2))
        self._dock_observation_times = deque(maxlen=settings.reacquire_confirmations)
        self._last_dock_observation_time = -math.inf
        self._reset_reacquisition()

    def _reset_reacquisition(self):
        self._reacquire_started = None
        self._reacquire_wait_since = None
        self._reacquire_side = 0

    def _observe_dock(self, time, perception):
        """Remember distinct EKF-accepted frames, including just before handover.

        main.update_demo removes rejected measurements before calling navigation.
        Camera frames arrive more slowly than control ticks; missing a new frame
        on one tick must not erase otherwise recent confirmation evidence.
        """
        observation = perception.dock_observation
        max_age = self.settings.reacquire_observation_age_s
        if (perception.new_camera_frame and observation is not None and
                self._last_dock_observation_time < observation.time <= time and
                time - observation.time <= max_age):
            self._dock_observation_times.append(observation.time)
            self._last_dock_observation_time = observation.time
        while self._dock_observation_times and time - self._dock_observation_times[0] > max_age:
            self._dock_observation_times.popleft()

    def _reacquire_request(self, state, bearing):
        """Face the dock, wait for vision, then sweep within a bounded angle/time."""
        time = state.time
        if time - self._reacquire_started >= self.settings.reacquire_timeout_s:
            return self._stop(time, 'Marker reacquisition timed out; waiting for accepted camera poses')
        # Accepted observations during a sweep bring the camera back towards the
        # corrected dock bearing. Keep the EKF and its local frame throughout.
        if self._dock_observation_times and self._reacquire_side:
            self._reacquire_side = 0
            self._reacquire_wait_since = None
            self._trajectory.reset()
        sweep = math.radians(self.settings.reacquire_sweep_deg)
        error = wrap_angle(bearing + self._reacquire_side * sweep)
        if abs(error) <= math.radians(self.settings.heading_tolerance_deg):
            self._trajectory.reset()
            if self._reacquire_wait_since is None:
                self._reacquire_wait_since = time
            if time - self._reacquire_wait_since < self.settings.reacquire_wait_s:
                self.status = (f'Reacquiring dock: holding for camera '
                               f'({len(self._dock_observation_times)}/{self.settings.reacquire_confirmations} recent poses)')
                return MotionRequest(time, 'REACQUIRE')
            self._reacquire_side = -1 if self._reacquire_side > 0 else 1
            self._reacquire_wait_since = None
            error = wrap_angle(bearing + self._reacquire_side * sweep)
        else:
            self._reacquire_wait_since = None
        target = np.array([*state.pose_local[:2], wrap_angle(state.pose_local[2] + error)])
        reference = self._trajectory.generate_reference(time, state.pose_local, target, 0., turn_only=True)
        self.status = ('LQI: turning to face the estimated dock' if not self._reacquire_side else
                       f'LQI: reacquisition sweep ({self.settings.reacquire_sweep_deg:g} deg each side)')
        return MotionRequest(time, 'REACQUIRE', reference=reference)

    def _explore(self, time, perception):
        """Alternate timed wheel commands; LiDAR decides whether forward travel is clear."""
        if self._search_remaining <= 0:
            if self.phase == 'EXPLORE':
                self.phase = 'SCAN'
                self._search_remaining = self.settings.search_spin_s
            else:
                # Check the intended short drive before starting it. Keep turning
                # until a clear opening faces forward; no pose or visited map is used.
                points = grouped_scan(perception.lidar_points_base, self.settings.boundary_join_gap_m)
                distance = self.settings.drive_speed_mps * self.settings.search_drive_s
                if distance + self.settings.clearance_m < perception.lidar.range_max_m and corridor_is_clear(
                        points, [distance, 0], self.settings.clearance_m + .04):
                    self.phase = 'EXPLORE'
                    self._search_remaining = self.settings.search_drive_s
        self._search_active = True
        if self.phase == 'SCAN':
            self.status = 'Timed scan for dock; no odometry yet'
            return MotionRequest(time, 'SCAN', turn_rate_rad_s=self.settings.turn_speed_rad_s)
        self.status = 'Short forward exploration command; no odometry yet'
        return MotionRequest(time, 'EXPLORE', self.settings.drive_speed_mps)

    @property
    def detour_pose(self):
        return self._detour.target

    @property
    def trajectory_preview(self):
        return self._trajectory.preview

    @property
    def trajectory_path(self):
        return self._trajectory.path()

    def _tracking_request(self, state, offset, speed, perception, target=None):
        direct = self.phase == 'APPROACH'
        target = dock_target_pose(state.dock_pose_local, offset) if target is None else target
        if perception.lidar.time != self._path_scan_time:
            self._path_scan_time = perception.lidar.time
            self._path_obstacles = grouped_scan(perception.lidar_points_base, self.settings.boundary_join_gap_m)
        horizon = (math.inf if direct else
                   min(1., perception.lidar.range_max_m - self.settings.clearance_m - .1))
        if horizon <= 0:
            return self._stop(state.time, 'Insufficient LiDAR range for a trajectory')
        pose = np.asarray(state.pose_local)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        # Approach and detours both turn then drive straight. Check the entire
        # remaining segment, including the connection from the estimated pose.
        reference = self._trajectory.generate_reference(
            state.time, pose, target, speed, max_distance=horizon)
        path = np.vstack((pose[:2], self._trajectory.path(remaining=True)[:, :2]))
        path_base = (path - pose[:2]) @ rotation
        if path_is_clear(self._path_obstacles, path_base, self.settings.clearance_m):
            return MotionRequest(state.time, self.phase, speed, offset, reference=reference)
        self._motion_blocked = True
        self._detour.target = None
        return self._stop(state.time, 'Trajectory blocked by LiDAR; choosing a detour')

    def _stop(self, time, reason):
        self.status = reason
        self._search_active = False
        self._trajectory.reset()
        return MotionRequest(time, 'STOP')

    def motion_request(self, state: StateEstimate, perception: PerceptionOutput,
                       *, stop_requested: bool = False) -> MotionRequest:
        """Choose the next behaviour using estimated poses and fresh observations."""
        time = float(state.time)
        if not math.isfinite(time) or not np.isfinite(state.pose_local).all():
            raise ValueError('Estimated pose and time must be finite')
        if self._previous_time is not None and time <= self._previous_time:
            raise ValueError('Navigation timestamps must increase; recreate navigation on reset')
        dt = 0. if self._previous_time is None else time - self._previous_time
        self._previous_time = time
        self._observe_dock(time, perception)
        if self._search_active:
            self._search_remaining -= dt
        self._search_active = False
        if stop_requested or self.phase == 'DONE':
            return self._stop(time, 'Stop requested' if stop_requested else 'Docked')

        # Acquisition starts approach, even on the very first update.
        # Both trajectory targets and phase checks come from the joint estimate.
        if state.dock_pose_local is not None:
            if self.dock_pose is None:
                self.phase = 'APPROACH'
                self._trajectory.reset()
                self._search_active = False
            self.dock_pose = np.asarray(state.dock_pose_local)
            self.coarse_pose = dock_target_pose(self.dock_pose, self.settings.coarse_distance_m)
            self.approach_pose = dock_target_pose(self.dock_pose, self.settings.approach_distance_m)
            self.docking_pose = dock_target_pose(self.dock_pose, self.settings.dock_distance_m)

        scan_age = time - perception.lidar.time
        if not math.isfinite(scan_age) or not -1e-9 <= scan_age <= self.settings.lidar_timeout_s:
            return self._stop(time, 'Waiting for fresh LiDAR')
        if perception.lidar_too_close or not motion_is_clear(
                perception.lidar_points_base, 0, 0, self.settings.clearance_m):
            return self._stop(time, 'Obstacle inside robot clearance')
        if state.dock_pose_local is None:
            return self._explore(time, perception)
        # Obstacle recovery retains its finite time budget.
        # Once exhausted, only a newly clear corridor (or reset) releases the stop.
        if self._detour.exhausted(time):
            result, _ = self._detour.update(time, state.pose_local, self.coarse_pose,
                perception.lidar_points_base, perception.lidar.range_max_m)
            self.phase = 'APPROACH' if result == 'clear' else 'AVOID'
            if result != 'clear':
                return self._stop(time, self._detour.reason)
        # Coarse travel continues from encoder prediction during occlusion.
        # REACQUIRE confirms vision once before PBVS; ALIGN/DOCK can subsequently
        # bridge marker loss using the same estimate, without resetting odometry.

        dock_base = state.dock_pose_base
        coarse = dock_target_pose(dock_base, self.settings.coarse_distance_m)
        approach = dock_target_pose(dock_base, self.settings.approach_distance_m)
        final = dock_target_pose(dock_base, self.settings.dock_distance_m)
        # Reach the actual stand-off point along the dock normal before PBVS.
        # Distance to the dock alone would admit close sideways approaches.
        # Arrival heading stays free; REACQUIRE faces the dock before PBVS.
        front = -np.dot(dock_base[:2], [math.cos(dock_base[2]), math.sin(dock_base[2])])
        at_coarse = front > 0 and np.linalg.norm(coarse[:2]) <= self.settings.coarse_tolerance_m
        approach_clear = corridor_is_clear(perception.lidar_points_base, approach[:2], self.settings.clearance_m)
        if self.phase in ('APPROACH', 'AVOID') and at_coarse and approach_clear:
            self.phase = 'REACQUIRE'
            self._reset_reacquisition()
            self._reacquire_started = time
            self._trajectory.reset()
            self._detour.reset()
            self._motion_blocked = False

        if self.phase == 'REACQUIRE':
            if not at_coarse or not approach_clear:
                # A camera correction or new obstacle can invalidate the handover
                # region. Replan using the corrected estimate before moving in.
                self.phase = 'APPROACH'
                self._reset_reacquisition()
                self._trajectory.reset()
            else:
                bearing = math.atan2(dock_base[1], dock_base[0])
                confirmed = len(self._dock_observation_times) == self.settings.reacquire_confirmations
                if confirmed and abs(bearing) <= math.radians(self.settings.reacquire_heading_tolerance_deg):
                    self.phase = 'ALIGN'
                    self._reset_reacquisition()
                    self._trajectory.reset()
                else:
                    return self._reacquire_request(state, bearing)

        if self.phase in ('APPROACH', 'AVOID'):
            # Only after reaching the stand-off region may avoidance aim towards
            # the PBVS offset. Side arrivals must first reach the coarse target.
            goal = self.approach_pose if at_coarse else self.coarse_pose
            result, target = self._detour.update(time, state.pose_local, goal,
                perception.lidar_points_base, perception.lidar.range_max_m, forced=self._motion_blocked)
            self._motion_blocked = False
            if result == 'approach':
                self.phase = 'APPROACH'
                self.status = 'LQI: approaching distant obstruction along the clear route'
                return self._tracking_request(state, self.settings.coarse_distance_m,
                    self.settings.drive_speed_mps, perception, target)
            if result != 'clear':
                self.phase = 'AVOID'
                if target is None:
                    reason = (self._detour.reason if result == 'blocked'
                              else 'Boundary blocked; waiting for clearance')
                    return self._stop(time, reason)
                self.status = f'Following boundary {"left" if self._detour.side > 0 else "right"} with LQI'
                return self._tracking_request(state, None, self.settings.detour_speed_mps, perception, target)
            if self.phase == 'AVOID':
                self._trajectory.reset()
                self.phase = 'APPROACH'

        tolerance = self.settings.position_tolerance_m
        heading_tolerance = math.radians(self.settings.heading_tolerance_deg)
        if self.phase == 'ALIGN':
            if np.linalg.norm(approach[:2]) <= tolerance and abs(approach[2]) <= heading_tolerance:
                self.phase = 'DOCK'
            else:
                # Report target-relative errors; odometry heading alone does not
                # show how much alignment remains with the estimated dock.
                self.status = (f'PBVS approach: error={np.linalg.norm(approach[:2]):.3f} m, '
                               f'heading={math.degrees(approach[2]):+.1f} deg')
                return MotionRequest(time, 'ALIGN', self.settings.pbvs_speed_mps,
                                     self.settings.approach_distance_m)

        if self.phase == 'DOCK':
            if np.linalg.norm(final[:2]) <= tolerance and abs(final[2]) <= heading_tolerance:
                self.phase = 'DONE'
                return self._stop(time, 'Docked')
            # Restore the approach alignment if a pose correction reveals a large error.
            x, y, yaw = dock_base
            lateral = -math.sin(yaw) * x + math.cos(yaw) * y
            if np.linalg.norm(final[:2]) > tolerance and (
                    abs(lateral) > max(.08, 2 * tolerance) or abs(final[2]) > math.radians(35)):
                self.phase = 'ALIGN'
                return self._stop(time, 'Dock alignment lost; returning to PBVS approach')
            self.status = (f'PBVS docking: error={np.linalg.norm(final[:2]):.3f} m, '
                           f'heading={math.degrees(final[2]):+.1f} deg')
            return MotionRequest(time, 'DOCK', self.settings.dock_speed_mps, self.settings.dock_distance_m)

        self.status = 'LQI: approaching the coarse dock region (free arrival heading)'
        return self._tracking_request(state, self.settings.coarse_distance_m, self.settings.drive_speed_mps, perception)

    def allow_motion(self, request: MotionRequest, v: float, omega: float,
                     perception: PerceptionOutput) -> bool:
        """Apply the LiDAR stop layer to the actual command, including curved/reverse motion."""
        if request.stop_requested:
            return False
        if motion_is_clear(perception.lidar_points_base, v, omega, self.settings.clearance_m):
            return True
        # Restart from the current estimate when clearance returns. The reference
        # must not continue along its trajectory while the robot is held stopped.
        self._trajectory.reset()
        self._search_active = False
        if request.mode == 'EXPLORE':
            self.phase = 'SCAN'
            self._search_remaining = self.settings.search_spin_s
        if request.mode in ('APPROACH', 'AVOID'):
            self._motion_blocked = True
            self._detour.target = None
        self.status = 'Motion blocked by LiDAR; waiting for a clear command'
        return False
