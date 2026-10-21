"""
Our navigator should guide the estimated robot pose towards the known charging dock.

It has 3 main stages:
1. Approach: plan waypoints around LiDAR obstacles to the dock approach pose
2. Align: face the dock and wait for reliable camera pose observations
3. Dock: follow a slow straight approach, then stop within pose tolerances

Navigation generates desired poses and velocities over time. Control uses the
estimated state to track these references and calculate wheel commands.
The dock pose is a configured map value; no live simulator pose is used here.
"""

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .estimation import StateEstimate
from .perception import PerceptionOutput, lidar_points_to_world
from .scene_config import NavigationConfig
from .state_model import wrap_angle


@dataclass(frozen=True)
class TrajectoryReference:
    """Desired world pose and velocities at one timestamp, passed to control."""

    time: float
    pose_world: tuple[float, float, float]
    v_ref_mps: float
    omega_ref_rad_s: float
    stop_requested: bool = False
    segment_id: int = 0 # A new segment resets the controller's integral error
    rotate_only: bool = False

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose_world, dtype=float)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("Reference pose must contain three finite values")
        if not all(math.isfinite(value) for value in (self.time, self.v_ref_mps, self.omega_ref_rad_s)):
            raise ValueError("Reference time and velocities must be finite")
        if not isinstance(self.stop_requested, bool) or not isinstance(self.rotate_only, bool):
            raise TypeError("Reference mode flags must be bools")
        if self.rotate_only and self.v_ref_mps != 0:
            raise ValueError("Rotation references must have zero forward speed")
        object.__setattr__(self, "pose_world", tuple(float(value) for value in pose))


def segment_is_clear(start, end, obstacles: np.ndarray, clearance: float) -> bool:
    """Check the robot's clearance along a line, including both endpoints."""
    if not len(obstacles):
        return True
    start, end = np.asarray(start), np.asarray(end)
    direction = end - start
    length_squared = float(direction @ direction)
    fractions = np.clip((obstacles - start) @ direction / max(length_squared, 1e-12), 0, 1)
    closest = start + fractions[:, None] * direction
    return bool(np.all(np.linalg.norm(obstacles - closest, axis=1) >= clearance))


def plan_waypoints(start, goal, obstacles, clearance: float, resolution: float) -> list[np.ndarray]:
    """Find a route through the latest observed points, then remove unnecessary bends.

    A* searches a grid with obstacles expanded by robot clearance. Unknown space
    is provisionally traversable; incoming scans must recheck the route as we move.
    An empty route means blocked, not permission to drive straight through.
    """
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 2)
    if start.shape != (2,) or goal.shape != (2,):
        raise ValueError("Planner start and goal must be XY positions")
    if not np.all(np.isfinite(np.concatenate((start, goal, obstacles.ravel())))):
        raise ValueError("Planner positions must be finite")
    if not all(math.isfinite(value) and value > 0 for value in (clearance, resolution)):
        raise ValueError("Planner clearance and resolution must be positive")
    if segment_is_clear(start, goal, obstacles, clearance):
        return [goal.copy()]
    if not all(segment_is_clear(p, p, obstacles, clearance) for p in (start, goal)):
        return []

    # Grid bounds come from the goal and scan, without assuming a particular room.
    points = np.vstack((start, goal, obstacles))
    origin = points.min(axis=0) - 2 * clearance
    shape = np.ceil((points.max(axis=0) + 2 * clearance - origin) / resolution).astype(int) + 1
    if np.prod(shape) > 100_000:
        return [] # Keep a bad or distant goal from creating an unbounded search
    blocked = np.zeros(tuple(shape), dtype=bool)
    radius = clearance + resolution / math.sqrt(2)
    for point in obstacles:
        lower = np.maximum(0, np.floor((point - radius - origin) / resolution).astype(int))
        upper = np.minimum(shape, np.ceil((point + radius - origin) / resolution).astype(int) + 1)
        xs = origin[0] + np.arange(lower[0], upper[0]) * resolution
        ys = origin[1] + np.arange(lower[1], upper[1]) * resolution
        blocked[lower[0]:upper[0], lower[1]:upper[1]] |= (
            (xs[:, None] - point[0])**2 + (ys[None, :] - point[1])**2 <= radius**2
        )
    first = tuple(np.rint((start - origin) / resolution).astype(int))
    last = tuple(np.rint((goal - origin) / resolution).astype(int))
    if blocked[first] or blocked[last]:
        return []
    queue = [(math.dist(first, last), first)]
    costs, parents = {first: 0.0}, {}
    visited = set()
    while queue:
        _, cell = heapq.heappop(queue)
        if cell in visited:
            continue
        if cell == last:
            route = [goal.copy()]
            while cell != first:
                route.append(origin + np.array(cell) * resolution)
                cell = parents[cell]
            route.reverse()
            # Connect as many grid steps as possible with each clear straight segment.
            waypoints, current = [], start
            while route:
                index = next((i for i in range(len(route) - 1, -1, -1)
                              if segment_is_clear(current, route[i], obstacles, clearance)), None)
                if index is None:
                    return []
                current = route[index]
                waypoints.append(current)
                route = route[index + 1:]
            return waypoints
        visited.add(cell)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            neighbour = cell[0] + dx, cell[1] + dy
            if not (0 <= neighbour[0] < shape[0] and 0 <= neighbour[1] < shape[1]):
                continue
            if blocked[neighbour] or (dx and dy and (blocked[cell[0] + dx, cell[1]] or blocked[cell[0], cell[1] + dy])):
                continue # Do not cut diagonally between blocked cells
            cost = costs[cell] + math.hypot(dx, dy)
            if cost < costs.get(neighbour, math.inf):
                costs[neighbour], parents[neighbour] = cost, cell
                heapq.heappush(queue, (cost + math.dist(neighbour, last), neighbour))
    return []


class Navigator:
    """Maintain the docking stage, planned waypoints and current reference."""

    def __init__(self, dock_pose_world, settings: NavigationConfig) -> None:
        self.settings = settings
        x, y, yaw = dock_pose_world
        if not np.all(np.isfinite((x, y, yaw))):
            raise ValueError("Dock pose must be finite")
        outward = np.array([math.cos(yaw), math.sin(yaw)])
        heading = wrap_angle(yaw + math.pi)
        self.approach_pose = np.r_[np.array([x, y]) + settings.approach_distance_m * outward, heading]
        self.docking_pose = np.r_[np.array([x, y]) + settings.dock_distance_m * outward, heading]
        self.phase = "APPROACH"
        self.status = "Planning approach"
        self.waypoints: list[np.ndarray] = []
        self._obstacles = np.empty((0, 2))
        self._scan_time = None
        self._camera_time = None
        self._last_good_camera_time = -math.inf
        self._good_frames = 0
        self._previous_time = None
        self._segment_id = 0
        self._reset_reference()

    def _reset_reference(self) -> None:
        self._segment_start = None
        self._segment_goal = None
        self._heading_reference = None
        self._progress = 0.0
        self._speed = 0.0
        self._turning = True

    def _reference(self, time, pose, v=0.0, omega=0.0, *, stop=False, rotate=False):
        return TrajectoryReference(time, tuple(pose), v, omega, stop, self._segment_id, rotate)

    def _stop(self, time, pose, reason):
        self.status = reason
        self._reset_reference()
        return self._reference(time, pose, stop=True)

    def _change_phase(self, phase):
        self.phase = phase
        self._reset_reference()

    def _rotate(self, time, pose, heading, dt):
        # Advance a heading reference at the configured turn rate, with zero translation.
        if self._heading_reference is None:
            self._heading_reference = pose[2]
            self._segment_id += 1
        change = np.clip(wrap_angle(heading - self._heading_reference),
                         -self.settings.turn_speed_rad_s * dt, self.settings.turn_speed_rad_s * dt)
        self._heading_reference = wrap_angle(self._heading_reference + change)
        return self._reference(time, (*pose[:2], self._heading_reference),
                               omega=float(change / dt) if dt else 0.0, rotate=True)

    def _follow(self, time, pose, goal, speed_limit, dt, start=None):
        # Each straight segment starts with alignment. We stop at bends for this demo.
        if self._segment_goal is None or not np.array_equal(goal, self._segment_goal):
            self._reset_reference()
            self._segment_start = np.array(pose[:2] if start is None else start, copy=True)
            self._segment_goal = goal.copy()
            self._segment_id += 1
            # Resume the final line at current progress after a camera-loss pause.
            if start is not None:
                line = goal - self._segment_start
                length = float(np.linalg.norm(line))
                if length > 0:
                    self._progress = float(np.clip((pose[:2] - start) @ (line / length), 0, length))
        displacement = goal - self._segment_start
        length = float(np.linalg.norm(displacement))
        if length < 1e-9:
            return self._stop(time, pose, "Segment complete")
        direction = displacement / length
        heading = math.atan2(direction[1], direction[0])
        error = abs(wrap_angle(heading - pose[2]))
        if self._turning:
            if error > math.radians(self.settings.heading_tolerance_deg):
                self.status = "Turning towards next waypoint"
                return self._rotate(time, pose, heading, dt)
            self._turning = False
            self._segment_id += 1
            dt = 0.0 # Begin forward motion on the following update

        # Accelerate and brake the reference, keeping it close enough for tracking.
        remaining = max(0.0, length - self._progress)
        desired_speed = min(speed_limit, math.sqrt(2 * self.settings.acceleration_mps2 * remaining))
        speed = min(desired_speed, self._speed + self.settings.acceleration_mps2 * dt)
        along = float((pose[:2] - self._segment_start) @ direction)
        limit = max(self._progress, min(length, max(0.0, along) + self.settings.reference_lead_m))
        progress = min(limit, self._progress + 0.5 * (self._speed + speed) * dt)
        velocity = (progress - self._progress) / dt if dt else 0.0
        self._progress, self._speed = progress, speed
        reference_xy = self._segment_start + progress * direction
        self.status = "Following approach path" if self.phase == "APPROACH" else "Docking slowly"
        return self._reference(time, (*reference_xy, heading), velocity)

    def trajectory_reference(self, state_estimate: StateEstimate, perception: PerceptionOutput,
                             *, stop_requested: bool = False) -> TrajectoryReference:
        """Plan from estimated state and sensor observations, then return one reference."""
        pose = np.asarray(state_estimate.pose_world, dtype=float)
        time = float(state_estimate.time)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)) or not math.isfinite(time):
            raise ValueError("Estimated pose and time must be finite")
        if self._previous_time is not None and time <= self._previous_time:
            raise ValueError("Navigation timestamps must increase; recreate the navigator on reset")
        dt = 0.0 if self._previous_time is None else time - self._previous_time
        self._previous_time = time
        settings = self.settings
        tolerance = settings.position_tolerance_m
        heading_tolerance = math.radians(settings.heading_tolerance_deg)

        # Remember fresh reliable camera observations; cached images are not confirmations.
        if perception.new_camera_frame and perception.camera.time != self._camera_time:
            self._camera_time = perception.camera.time
            observation = perception.pose_observation
            if observation is not None and 0 <= time - observation.time <= settings.camera_timeout_s:
                self._last_good_camera_time = observation.time
                self._good_frames += 1
            else:
                self._good_frames = 0
        camera_recent = time - self._last_good_camera_time <= settings.camera_timeout_s

        if stop_requested or self.phase == "DONE":
            return self._stop(time, pose, "Stop requested" if stop_requested else "Docked")
        scan_age = time - perception.lidar.time
        if not math.isfinite(scan_age) or not -1e-9 <= scan_age <= settings.lidar_timeout_s:
            return self._stop(time, pose, "Waiting for fresh LiDAR")
        if perception.lidar_too_close:
            return self._stop(time, pose, "Obstacle inside LiDAR minimum range")
        new_scan = perception.lidar.time != self._scan_time
        if new_scan:
            self._obstacles = lidar_points_to_world(perception.lidar_points_base, pose)
            self._scan_time = perception.lidar.time
        if not segment_is_clear(pose[:2], pose[:2], self._obstacles, settings.clearance_m):
            return self._stop(time, pose, "Obstacle inside robot clearance")

        if self.phase == "APPROACH":
            if np.linalg.norm(pose[:2] - self.approach_pose[:2]) <= tolerance:
                self.waypoints.clear()
                self._change_phase("ALIGN")
            else:
                if self.waypoints and np.linalg.norm(pose[:2] - self.waypoints[0]) <= tolerance:
                    self.waypoints.pop(0)
                    self._reset_reference()
                blocked = bool(self.waypoints) and not segment_is_clear(
                    pose[:2], self.waypoints[0], self._obstacles, settings.clearance_m)
                if (not self.waypoints or blocked) and new_scan:
                    # Leave half a grid cell beyond the stop clearance so small
                    # scan changes do not continually replace a boundary-hugging path.
                    self.waypoints = plan_waypoints(pose[:2], self.approach_pose[:2], self._obstacles,
                                                   settings.clearance_m + settings.grid_resolution_m / 2,
                                                   settings.grid_resolution_m)
                    self._reset_reference()
                if not self.waypoints or (blocked and not new_scan):
                    return self._stop(time, pose, "Approach blocked; waiting for a clear route")
                return self._follow(time, pose, self.waypoints[0], settings.drive_speed_mps, dt)

        if self.phase == "ALIGN":
            if np.linalg.norm(pose[:2] - self.approach_pose[:2]) > 2 * tolerance:
                self._change_phase("APPROACH")
                return self._stop(time, pose, "Returning to approach pose")
            if abs(wrap_angle(self.docking_pose[2] - pose[2])) > heading_tolerance:
                self.status = "Aligning with dock"
                return self._rotate(time, pose, self.docking_pose[2], dt)
            if not camera_recent or self._good_frames < settings.camera_confirmations:
                return self._stop(time, pose, "Waiting for reliable dock observations")
            self._change_phase("DOCK")

        # Keep obstacle checks active during docking. Marker loss pauses the approach.
        if not camera_recent:
            return self._stop(time, pose, "Dock marker lost; waiting for a reliable observation")
        if not segment_is_clear(pose[:2], self.docking_pose[:2], self._obstacles, settings.clearance_m):
            return self._stop(time, pose, "Dock approach obstructed")
        if np.linalg.norm(pose[:2] - self.docking_pose[:2]) <= tolerance:
            if abs(wrap_angle(self.docking_pose[2] - pose[2])) <= heading_tolerance:
                self._change_phase("DONE")
                return self._stop(time, pose, "Docked")
            self.status = "Finishing dock alignment"
            return self._rotate(time, pose, self.docking_pose[2], dt)
        return self._follow(time, pose, self.docking_pose[:2], settings.dock_speed_mps,
                            dt, start=self.approach_pose[:2])
