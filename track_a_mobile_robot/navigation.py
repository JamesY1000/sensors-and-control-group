"""Plan room-search and marker-approach references for feedback control."""
from dataclasses import dataclass
from enum import Enum, auto
import math

import numpy as np
from numpy.typing import ArrayLike

from .estimation import StateEstimate
from .perception import MARKER_MAP, PerceptionOutput, lidar_points_to_world

_APF_STEP_SIZE_M = 0.05
_OBSTACLE_CLEARANCE_M = 0.30
_APF_INFLUENCE_DISTANCE_M = 0.9
_APF_REPULSIVE_GAIN = 0.08
_APF_TANGENTIAL_GAIN = 0.75
# Obstacle points whose tangent is this poorly aligned with the goal direction
# (|cos| below it, i.e. roughly dead ahead) use the committed circulation.
_APF_AMBIGUOUS_ALIGNMENT = 0.5
# Wheel limit is 7.88 rad/s * 0.033 m = 0.26 m/s; 0.22 leaves headroom for steering.
_DRIVE_SPEED_MPS = 0.22
# Large heading changes are driven as forward arcs, never spins on the spot.
# 0.17 m/s on a 0.3 m radius keeps the outer wheel just under its limit.
_ARC_SPEED_MPS = 0.17
_ARC_RADIUS_M = 0.30
_ARC_START_ERROR_RAD = math.radians(60)
_ARC_END_ERROR_RAD = math.radians(20)
# The swept arc must keep this clearance to LiDAR points, else the other
# direction is tried, else the robot backs up straight.
_ARC_CLEARANCE_M = 0.25
_REVERSE_SPEED_MPS = 0.10
# Only the final docking slows down, so the robot stops on the final pose.
_MIN_DOCK_SPEED_MPS = 0.04
_DOCK_DECELERATION_PER_S = 0.8
# The reference point advances along the path at the reference speed but may
# lead the robot by at most this much, so lagging shows up as along-track error
# for the controller to correct without leaving the linearised region.
_MAX_REFERENCE_LEAD_M = 0.15
_STAGING_DISTANCE_M = 0.5
_FINAL_DISTANCE_M = 0.2
_WALL_STANDOFF_M = 0.85
_MIN_SEARCH_DISTANCE_M = 0.5
# Goals passed through at speed count as reached within these distances.
_SEARCH_ARRIVAL_M = 0.30
_STAGING_ARRIVAL_M = 0.20
# Already in front of the marker and roughly facing it: skip staging.
_CORRIDOR_HALF_WIDTH_M = 0.25
_CORRIDOR_HEADING_RAD = math.radians(30)

# Room knowledge: only its size. Where the robot sits in it is measured from
# the first LiDAR scan, assuming the robot starts aligned with the long axis.
ROOM_LENGTH_M = 6.0
ROOM_WIDTH_M = 4.0
_WALL_SECTOR_RAD = math.radians(3)
# Left + right wall distances must add up to the room width within this, or a
# side obstacle is assumed to hide a wall and the robot's lateral offset is unknown.
_WALL_SUM_TOLERANCE_M = 0.3
# Lateral offsets from the room's centre line tried, in order, for each end goal.
_SEARCH_LATERAL_OFFSETS_M = (0.0, 0.9, -0.9)
_MARKER_CONFIRMATIONS = 3
# A goal is abandoned (APF local minimum) if the distance to it has not
# shrunk by _PROGRESS_DISTANCE_M within _STALL_TIMEOUT_S.
_PROGRESS_DISTANCE_M = 0.10
_STALL_TIMEOUT_S = 15.0
_POSITION_TOLERANCE_M = 0.10
_FINAL_POSITION_TOLERANCE_M = 0.03


@dataclass(frozen=True)
class TrajectoryReference:
    """One timestamped desired pose and feed-forward velocity command.

    segment_id changes whenever the navigator starts tracking a new goal or
    phase, so the controller can clear integral error from the previous one.
    """

    time: float
    pose_world: tuple[float, float, float]
    v_ref_mps: float
    omega_ref_rad_s: float
    stop_requested: bool = False
    segment_id: int = 0

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose_world, dtype=float)
        if not math.isfinite(self.time):
            raise ValueError("Trajectory time must be finite.")
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("pose_world must contain three finite values (x, y, heading).")
        if not math.isfinite(self.v_ref_mps) or not math.isfinite(self.omega_ref_rad_s):
            raise ValueError("Reference velocities must be finite.")
        if not isinstance(self.stop_requested, bool):
            raise TypeError("stop_requested must be a bool.")
        if not isinstance(self.segment_id, int) or isinstance(self.segment_id, bool):
            raise TypeError("segment_id must be an int.")
        object.__setattr__(self, "pose_world", tuple(float(value) for value in pose))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _pose_array(pose: ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(pose, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values (x, y, heading).")
    return result


def obstacle_avoidance(
    robot_pose: np.ndarray,
    target_pose: np.ndarray,
    obstacles: ArrayLike,
    circulation: float = 0.0) -> np.ndarray:
    """Choose the next safe waypoint with a reactive artificial potential field.

    Attractive, repulsive, and tangential forces guide each waypoint. The
    caller replans from the next state estimate. Obstacles are
    assumed to be point centers, and the planner keeps
    ``_OBSTACLE_CLEARANCE_M`` between the robot center and each obstacle. A
    start already inside this clearance may leave only along a segment that
    moves monotonically away from it.

    Returns ``target_pose`` when the straight segment is clear, otherwise the
    waypoint one APF step away, with its heading pointing along that step.

    Args:
        robot_pose: The current pose of the robot (x, y, theta).
        target_pose: The desired target pose (x, y, theta).
        obstacles: Obstacle centers, shape (N, 2).
        circulation: +1 to pass obstacles counterclockwise, -1 clockwise, 0
            to always choose per obstacle point from the goal direction. The
            fixed sense is used only for points nearly straight ahead, where
            per-point choices cancel and the waypoint flips with tiny moves.

    Raises:
        ValueError: If a pose or obstacle is invalid, the target is inside the
            obstacle clearance area, or APF cannot find a safe route.
    """
    start = np.asarray(robot_pose, dtype=float)
    goal = np.asarray(target_pose, dtype=float)
    if start.shape != (3,) or goal.shape != (3,):
        raise ValueError("robot_pose and target_pose must each contain (x, y, theta).")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError("robot_pose and target_pose must contain only finite values.")

    obstacle_points = np.asarray(obstacles, dtype=float)
    if obstacle_points.size == 0:
        obstacle_points = np.empty((0, 2), dtype=float)
    if obstacle_points.ndim != 2 or obstacle_points.shape[1] != 2:
        raise ValueError("Each obstacle must contain an (x, y) point.")
    if not np.all(np.isfinite(obstacle_points)):
        raise ValueError("Obstacle points must contain only finite values.")

    def segment_is_clear(point_a: np.ndarray, point_b: np.ndarray) -> bool:
        if not len(obstacle_points):
            return True
        segment = point_b - point_a
        length_squared = float(np.dot(segment, segment))
        offsets = obstacle_points - point_a
        if length_squared == 0:
            closest_points = np.broadcast_to(point_a, obstacle_points.shape)
        else:
            fractions = np.clip(offsets @ segment / length_squared, 0.0, 1.0)
            closest_points = point_a + fractions[:, None] * segment
        distances = np.linalg.norm(obstacle_points - closest_points, axis=1)
        return bool(np.all(distances >= _OBSTACLE_CLEARANCE_M))

    if not segment_is_clear(goal[:2], goal[:2]):
        raise ValueError("The target is inside the obstacle clearance area.")

    if segment_is_clear(start[:2], goal[:2]):
        return goal.copy()
    # From here on there is at least one obstacle (otherwise the line is clear).

    def segment_escapes_clearance(point_a: np.ndarray, point_b: np.ndarray) -> bool:
        """Allow an initially close pose to move only monotonically away."""
        displacement = point_b - point_a
        initial_offsets = point_a - obstacle_points
        initial_distances = np.linalg.norm(initial_offsets, axis=1)
        close_obstacles = initial_distances < _OBSTACLE_CLEARANCE_M
        if not np.any(close_obstacles):
            return segment_is_clear(point_a, point_b)

        length_squared = float(np.dot(displacement, displacement))
        if length_squared == 0:
            closest_points = np.broadcast_to(point_a, obstacle_points.shape)
        else:
            fractions = np.clip(
                (obstacle_points - point_a) @ displacement / length_squared,
                0.0,
                1.0,
            )
            closest_points = point_a + fractions[:, None] * displacement
        segment_distances = np.linalg.norm(obstacle_points - closest_points, axis=1)
        moving_away = np.sum(
            initial_offsets[close_obstacles] * displacement,
            axis=1,
        ) >= -1e-12
        endpoint_distances = np.linalg.norm(
            point_b - obstacle_points[close_obstacles],
            axis=1,
        )
        close_obstacles_escape = (
            np.all(moving_away)
            and np.all(endpoint_distances >= _OBSTACLE_CLEARANCE_M)
        )
        other_obstacles_clear = np.all(
            segment_distances[~close_obstacles] >= _OBSTACLE_CLEARANCE_M
        )
        return bool(close_obstacles_escape and other_obstacles_clear)

    current = start[:2]
    goal_offset = goal[:2] - current
    goal_distance = float(np.linalg.norm(goal_offset))
    attractive_force = goal_offset / goal_distance
    repulsive_force = np.zeros(2)
    offsets = current - obstacle_points
    start_distances = np.linalg.norm(offsets, axis=1)
    start_inside_clearance = bool(
        np.any(start_distances < _OBSTACLE_CLEARANCE_M)
    )

    influenced = (
        (start_distances < _APF_INFLUENCE_DISTANCE_M)
        & (start_distances > 1e-12)
    )
    if np.any(influenced):
        influenced_distances = start_distances[influenced]
        repulsive_magnitudes = (
            _APF_REPULSIVE_GAIN
            * (
                1.0 / influenced_distances
                - 1.0 / _APF_INFLUENCE_DISTANCE_M
            )
            / influenced_distances**2
        )
        obstacle_directions = offsets[influenced] / influenced_distances[:, None]
        tangential_directions = np.column_stack((
            -obstacle_directions[:, 1],
            obstacle_directions[:, 0],
        ))
        tangent_alignment = tangential_directions @ attractive_force
        tangent_signs = np.where(tangent_alignment < 0.0, -1.0, 1.0)
        if circulation:
            tangent_signs = np.where(
                np.abs(tangent_alignment) < _APF_AMBIGUOUS_ALIGNMENT,
                math.copysign(1.0, circulation),
                tangent_signs,
            )
        tangential_directions *= tangent_signs[:, None]
        repulsive_force = np.sum(
            repulsive_magnitudes[:, None] * (
                obstacle_directions
                + _APF_TANGENTIAL_GAIN * tangential_directions
            ),
            axis=0,
        )

    resultant = attractive_force + repulsive_force
    resultant_magnitude = float(np.linalg.norm(resultant))
    if resultant_magnitude <= 1e-9:
        resultant = attractive_force
        resultant_magnitude = 1.0

    step_length = min(_APF_STEP_SIZE_M, goal_distance)
    if start_inside_clearance:
        close_distances = start_distances[
            start_distances < _OBSTACLE_CLEARANCE_M
        ]
        step_length = min(
            goal_distance,
            max(
                step_length,
                float(np.max(_OBSTACLE_CLEARANCE_M - close_distances))
                + _APF_STEP_SIZE_M,
            ),
        )
    direction = resultant / resultant_magnitude
    candidate = current + step_length * direction
    if step_length >= goal_distance:
        candidate = goal[:2].copy()

    def candidate_is_safe(point: np.ndarray) -> bool:
        if start_inside_clearance:
            return segment_escapes_clearance(current, point)
        return segment_is_clear(current, point)

    if not candidate_is_safe(candidate):
        best_candidate = None
        best_progress = -float("inf")
        for turn_degrees in range(-180, 181, 10):
            turn = math.radians(turn_degrees)
            cosine = math.cos(turn)
            sine = math.sin(turn)
            alternative_direction = np.array([
                cosine * direction[0] - sine * direction[1],
                sine * direction[0] + cosine * direction[1],
            ])
            alternative = current + step_length * alternative_direction
            if not candidate_is_safe(alternative):
                continue
            progress = float(np.dot(alternative_direction, attractive_force))
            if progress > best_progress:
                best_candidate = alternative
                best_progress = progress
        if best_candidate is None:
            raise ValueError("APF could not find a safe waypoint toward the target.")
        candidate = best_candidate

    if np.linalg.norm(candidate - goal[:2]) <= 1e-12:
        return goal.copy()
    heading = math.atan2(candidate[1] - current[1], candidate[0] - current[0])
    return np.array([candidate[0], candidate[1], heading])


@dataclass(frozen=True)
class _Room:
    """Room placement in the world frame, from its size and the first scan."""

    axis: np.ndarray         # unit vector along the long axis, toward the far end
    normal: np.ndarray       # unit vector to the left of axis
    near_center: np.ndarray  # search goal at the start end, on the centre line
    far_center: np.ndarray   # search goal at the far end, on the centre line


def _wall_distance(points_base: np.ndarray, bearing_rad: float) -> float | None:
    """Nearest LiDAR return within a narrow sector around a base-frame bearing."""
    if not len(points_base):
        return None
    bearings = np.arctan2(points_base[:, 1], points_base[:, 0])
    in_sector = np.abs(np.arctan2(
        np.sin(bearings - bearing_rad), np.cos(bearings - bearing_rad),
    )) <= _WALL_SECTOR_RAD
    if not np.any(in_sector):
        return None
    return float(np.min(np.linalg.norm(points_base[in_sector], axis=1)))


def _locate_room(estimated: np.ndarray, points_base: np.ndarray) -> _Room:
    """Place the known-size room around the robot using the first LiDAR scan.

    Assumes the robot starts facing along the room's long axis. The rear wall
    fixes how far the far end is; the side walls fix the lateral offset when
    both are visible and consistent with the room width.
    """
    axis = np.array([math.cos(estimated[2]), math.sin(estimated[2])])
    normal = np.array([-axis[1], axis[0]])
    rear = _wall_distance(points_base, math.pi)
    if rear is None or rear >= ROOM_LENGTH_M:
        rear = ROOM_LENGTH_M / 2  # no usable rear wall: assume the middle
    left = _wall_distance(points_base, math.pi / 2)
    right = _wall_distance(points_base, -math.pi / 2)
    lateral = 0.0
    if (
        left is not None and right is not None
        and abs(left + right - ROOM_WIDTH_M) <= _WALL_SUM_TOLERANCE_M
    ):
        lateral = (left - right) / 2
    centre_line = estimated[:2] + lateral * normal
    return _Room(
        axis=axis,
        normal=normal,
        near_center=centre_line + (_WALL_STANDOFF_M - rear) * axis,
        far_center=centre_line + (ROOM_LENGTH_M - rear - _WALL_STANDOFF_M) * axis,
    )


class _Phase(Enum):
    SEARCH = auto()      # drive lengthwise to one end of the room, then the other
    TO_STAGING = auto()  # drive to the point in front of the marker
    FINAL = auto()       # follow the marker's normal line in to the final pose
    DONE = auto()        # stopped at the marker


class Navigator:
    """Generate one trajectory reference per estimate/navigation update.

    The estimator pose and generated references use the world frame. Perception
    supplies LiDAR points in the base frame; the navigator maps them with the
    estimated pose. Detected marker IDs are resolved through the landmark map.

    The mission is a latched state machine: drive lengthwise between the two
    ends of the room until a known marker is seen in several consecutive
    frames, then staging, final approach along the marker normal, and stop.
    Phases only move forward once the marker approach starts. The robot always
    keeps moving: large heading changes are forward arcs, never spins in place.
    """

    def __init__(self, *, verbose: bool = True) -> None:
        """``verbose`` prints a console line whenever the navigation state changes."""
        self._verbose = verbose
        self._reported_state: str | None = None
        self._phase = _Phase.SEARCH
        self._segment_id = 0
        self._segment_start: np.ndarray | None = None
        self._reference_progress = 0.0
        self._follow_time: float | None = None
        self._circulation = 0.0
        self._goal: np.ndarray | None = None
        self._room: _Room | None = None
        self._toward_far_end = True
        self._lateral_index = 0
        self._marker_detections = 0
        self._marker_xy: np.ndarray | None = None
        self._approach_normal: np.ndarray | None = None
        self._staging_pose: np.ndarray | None = None
        self._final_pose: np.ndarray | None = None
        self._final_start: np.ndarray | None = None
        self._detour_goal: np.ndarray | None = None
        self._turn_sign = 0.0
        self._best_goal_distance = math.inf
        self._progress_time: float | None = None

    def trajectory_reference(
        self,
        state_estimate: StateEstimate,
        perception: PerceptionOutput,
        *,
        stop_requested: bool = False,
    ) -> TrajectoryReference:
        """Return a room-search or marker-approach trajectory reference.

        LiDAR obstacle points are transformed from the base frame to the world
        frame using the estimated pose. A detected marker starts marker
        approach only when its ID is present in the known landmark map.
        """
        estimated = _pose_array(state_estimate.pose_world, "state_estimate.pose_world")
        time = float(state_estimate.time)
        if not math.isfinite(time):
            raise ValueError("State estimate time must be finite.")
        if not isinstance(stop_requested, bool):
            raise TypeError("stop_requested must be a bool.")
        if stop_requested:
            return self._reference(time, estimated, 0.0, 0.0, True)

        obstacle_points_base = np.asarray(perception.lidar_points_base, dtype=float)
        if obstacle_points_base.size == 0:
            obstacle_points_base = np.empty((0, 2), dtype=float)
        if obstacle_points_base.ndim != 2 or obstacle_points_base.shape[1] != 2:
            raise ValueError("perception.lidar_points_base must have shape (N, 2).")
        obstacles = np.asarray(
            lidar_points_to_world(obstacle_points_base, estimated),
            dtype=float,
        ).reshape(-1, 2)
        if not np.all(np.isfinite(obstacles)):
            raise ValueError("Perception LiDAR points must be finite.")

        if self._room is None:
            self._room = _locate_room(estimated, obstacle_points_base)

        if self._phase is _Phase.SEARCH:
            marker_id = self._confirmed_marker(perception)
            if marker_id is not None:
                self._start_marker_approach(marker_id)

        if self._phase is _Phase.SEARCH:
            reference = self._search(time, estimated, obstacles)
        elif self._phase is _Phase.TO_STAGING:
            reference = self._to_staging(time, estimated, obstacles)
        elif self._phase is _Phase.FINAL:
            reference = self._final(time, estimated)
        else:
            reference = self._reference(time, estimated, 0.0, 0.0, True)
        self._report_state(time, estimated)
        return reference

    @property
    def phase(self) -> str:
        """Current mission phase name: SEARCH, TO_STAGING, FINAL or DONE."""
        return self._phase.name

    def _state_description(self) -> str:
        def point(xy: np.ndarray) -> str:
            return f"({xy[0]:.2f}, {xy[1]:.2f})"

        if self._phase is _Phase.SEARCH:
            end = "far" if self._toward_far_end else "near"
            return f"searching: driving to the {end} end of the room {point(self._goal)}"
        if self._phase is _Phase.TO_STAGING:
            if self._detour_goal is not None:
                return f"marker approach blocked: detouring via {point(self._detour_goal)}"
            return f"marker seen: driving to staging point {point(self._staging_pose)}"
        if self._phase is _Phase.FINAL:
            return f"docking along the marker normal to {point(self._final_pose)}"
        return "docked at the marker: stopped"

    def _report_state(self, time: float, estimated: np.ndarray) -> None:
        """Print one console line whenever the navigation state changes."""
        if not self._verbose or (self._phase is _Phase.SEARCH and self._goal is None):
            return  # between search goals for one tick; the next goal is reported
        description = self._state_description()
        if description == self._reported_state:
            return
        self._reported_state = description
        print(
            f"Navigator t={time:.2f} s [{self._phase.name}] {description}; "
            f"estimated pose=({estimated[0]:.2f}, {estimated[1]:.2f}, "
            f"{math.degrees(estimated[2]):.0f} deg)",
            flush=True,
        )

    def _reference(
        self,
        time: float,
        pose: np.ndarray,
        v_ref_mps: float,
        omega_ref_rad_s: float,
        stop_requested: bool = False,
    ) -> TrajectoryReference:
        return TrajectoryReference(
            time=time,
            pose_world=tuple(float(value) for value in pose),
            v_ref_mps=v_ref_mps,
            omega_ref_rad_s=omega_ref_rad_s,
            stop_requested=stop_requested,
            segment_id=self._segment_id,
        )

    def _set_phase(self, phase: _Phase) -> None:
        """Enter a phase (or restart the current one with a new goal)."""
        self._phase = phase
        self._segment_start = None
        self._circulation = 0.0
        self._turn_sign = 0.0
        self._segment_id += 1
        self._best_goal_distance = math.inf
        self._progress_time = None

    def _hold(self, time: float, estimated: np.ndarray) -> TrajectoryReference:
        """Stay where the robot is (zero tracking error, zero feed-forward)."""
        return self._reference(time, estimated, 0.0, 0.0)

    def _stalled(self, time: float, goal_distance: float) -> bool:
        """True when the goal distance has not improved for the stall timeout."""
        if (
            self._progress_time is None
            or goal_distance < self._best_goal_distance - _PROGRESS_DISTANCE_M
        ):
            self._best_goal_distance = goal_distance
            self._progress_time = time
        return time - self._progress_time > _STALL_TIMEOUT_S

    def _arc_reference(
        self,
        time: float,
        estimated: np.ndarray,
        desired_heading: float,
        obstacles: np.ndarray | None,
    ) -> TrajectoryReference | None:
        """Turn toward ``desired_heading`` along a forward arc, if the turn is large.

        Returns None when the heading error is small enough for normal path
        following. Once an arc starts, its direction is kept until the error
        drops below ``_ARC_END_ERROR_RAD``, so noisy desired headings near
        +/-pi cannot flip it. The arc is a feed-forward motion (reference on
        the robot itself, so only v and omega feed forward reach the wheels).
        """
        error = _wrap_angle(desired_heading - estimated[2])
        if self._turn_sign != 0.0 and abs(error) <= _ARC_END_ERROR_RAD:
            self._turn_sign = 0.0
            return None
        if self._turn_sign == 0.0:
            if abs(error) <= _ARC_START_ERROR_RAD:
                return None
            self._turn_sign = self._choose_arc_direction(estimated, error, obstacles)
            self._segment_start = None
            self._segment_id += 1
            if self._turn_sign == 0.0:
                # No room to arc either way: back straight out, then try again.
                return self._reference(time, estimated, -_REVERSE_SPEED_MPS, 0.0)
        return self._reference(
            time,
            estimated,
            _ARC_SPEED_MPS,
            self._turn_sign * _ARC_SPEED_MPS / _ARC_RADIUS_M,
        )

    @staticmethod
    def _choose_arc_direction(
        estimated: np.ndarray,
        error: float,
        obstacles: np.ndarray | None,
    ) -> float:
        """Prefer the shorter turn; use the other way if its swept arc is blocked."""
        preferred = 1.0 if error >= 0.0 else -1.0
        for sign, turn in ((preferred, abs(error)), (-preferred, 2 * math.pi - abs(error))):
            if obstacles is None or not len(obstacles):
                return sign
            heading = estimated[2]
            centre = estimated[:2] + sign * _ARC_RADIUS_M * np.array(
                [-math.sin(heading), math.cos(heading)]
            )
            radial = estimated[:2] - centre
            angles = sign * np.linspace(0.0, turn, max(2, int(turn / 0.1)))
            cosines, sines = np.cos(angles), np.sin(angles)
            swept = centre + np.column_stack((
                cosines * radial[0] - sines * radial[1],
                sines * radial[0] + cosines * radial[1],
            ))
            clearance = np.min(np.linalg.norm(
                swept[:, None, :] - obstacles[None, :, :], axis=2,
            ))
            if clearance >= _ARC_CLEARANCE_M:
                return sign
        return 0.0

    def _follow(
        self,
        time: float,
        estimated: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray | None,
        *,
        start: np.ndarray | None = None,
        dock: bool = False,
    ) -> TrajectoryReference:
        """Track a straight segment to ``goal`` at drive speed, detouring when blocked.

        On a clear segment the reference point advances along it at the
        reference speed, never behind the robot and at most
        ``_MAX_REFERENCE_LEAD_M`` ahead, so a robot slower than commanded
        builds along-track error that the controller corrects. ``start`` fixes
        the segment start (else it is where the robot begins the segment);
        ``dock`` slows down to stop on the goal. When APF reports a blocked
        segment, the reference points at the next APF waypoint. Turns larger
        than ``_ARC_START_ERROR_RAD`` are driven as forward arcs.

        Raises:
            ValueError: From ``obstacle_avoidance`` when no safe route exists.
        """
        if obstacles is not None:
            if self._circulation == 0.0:
                self._circulation = self._choose_circulation(estimated, goal, obstacles)
            waypoint = obstacle_avoidance(estimated, goal, obstacles, self._circulation)
            if np.allclose(waypoint[:2], goal[:2]):
                self._circulation = 0.0  # clear line: choose afresh next time
            else:
                self._segment_start = None
                heading = float(waypoint[2])
                arc = self._arc_reference(time, estimated, heading, obstacles)
                if arc is not None:
                    return arc
                return self._reference(
                    time,
                    np.array([estimated[0], estimated[1], heading]),
                    _DRIVE_SPEED_MPS,
                    0.0,
                )

        if self._segment_start is None:
            line_start = start if start is not None else estimated[:2]
            to_goal = goal[:2] - line_start
            arc = self._arc_reference(
                time, estimated, math.atan2(to_goal[1], to_goal[0]), obstacles,
            )
            if arc is not None:
                return arc
            self._start_segment(time, line_start)
        segment = goal[:2] - self._segment_start
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            return self._hold(time, estimated)
        robot_progress = float(np.dot(estimated[:2] - self._segment_start, segment)) / length
        if robot_progress >= length and start is None:
            # Passed the goal without arriving (e.g. lateral offset): restart
            # the segment from here so the robot turns back toward the goal.
            self._segment_start = None
            return self._follow(time, estimated, goal, obstacles, dock=dock)

        elapsed = max(0.0, time - self._follow_time)
        self._follow_time = time
        direction = segment / length
        heading = math.atan2(direction[1], direction[0])
        robot_progress = min(max(robot_progress, 0.0), length)
        speed = _DRIVE_SPEED_MPS
        if dock:
            remaining = length - self._reference_progress
            speed = min(speed, max(_MIN_DOCK_SPEED_MPS, _DOCK_DECELERATION_PER_S * remaining))
        self._reference_progress = max(
            robot_progress,
            min(
                length,
                self._reference_progress + speed * elapsed,
                robot_progress + _MAX_REFERENCE_LEAD_M,
            ),
        )
        if dock and self._reference_progress >= length:
            speed = 0.0  # reference parked on the goal; position error finishes it
        reference_xy = self._segment_start + self._reference_progress * direction
        return self._reference(
            time,
            np.array([reference_xy[0], reference_xy[1], heading]),
            speed,
            0.0,
        )

    @staticmethod
    def _choose_circulation(
        estimated: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> float:
        """Pass nearby obstacles on the side away from where most of them lie.

        Obstacles mostly left of the robot-to-goal line are passed on the
        right, which is counterclockwise around them (+1), and vice versa.
        """
        to_goal = goal[:2] - estimated[:2]
        to_goal = to_goal / max(float(np.linalg.norm(to_goal)), 1e-9)
        offsets = obstacles - estimated[:2]
        distances = np.linalg.norm(offsets, axis=1)
        near = distances < _APF_INFLUENCE_DISTANCE_M
        # Only points within 45 degrees of the goal direction: walls alongside
        # the path should not decide which way round the blocking obstacle goes.
        ahead = offsets @ to_goal > math.cos(math.radians(45)) * distances
        relevant = offsets[near & ahead]
        if not len(relevant):
            return 1.0
        side = float(np.sum(to_goal[0] * relevant[:, 1] - to_goal[1] * relevant[:, 0]))
        return 1.0 if side >= 0.0 else -1.0

    def _start_segment(self, time: float, start_xy: np.ndarray) -> None:
        self._segment_start = np.array(start_xy[:2], dtype=float)
        self._reference_progress = 0.0
        self._follow_time = time
        self._segment_id += 1

    # ------------------------------------------------------------------ search

    def _end_goal(self) -> np.ndarray:
        """Current patrol goal: one end of the room, at the current lateral offset."""
        assert self._room is not None
        if self._toward_far_end:
            centre, facing = self._room.far_center, self._room.axis
        else:
            centre, facing = self._room.near_center, -self._room.axis
        goal_xy = centre + _SEARCH_LATERAL_OFFSETS_M[self._lateral_index] * self._room.normal
        return np.array([goal_xy[0], goal_xy[1], math.atan2(facing[1], facing[0])])

    @staticmethod
    def _detour_goal_from_scan(
        estimated: np.ndarray,
        obstacles: np.ndarray,
        toward_xy: np.ndarray,
    ) -> np.ndarray | None:
        """Pick a reachable LiDAR-derived point to escape a blocked approach.

        Candidates lie along LiDAR rays, short of the hit point by the wall
        standoff and clear of obstacles; the one nearest ``toward_xy`` (the
        staging point) wins, so the detour makes progress instead of leading
        away from the marker.
        """
        if not len(obstacles):
            return None
        offsets = obstacles - estimated[:2]
        ranges = np.linalg.norm(offsets, axis=1)
        distances = ranges - _WALL_STANDOFF_M
        usable = distances >= _MIN_SEARCH_DISTANCE_M
        if not np.any(usable):
            return None
        directions = offsets[usable] / ranges[usable, None]
        goals = estimated[:2] + distances[usable, None] * directions
        clearances = np.min(
            np.linalg.norm(goals[:, None, :] - obstacles[None, :, :], axis=2),
            axis=1,
        )
        goals = goals[clearances >= _OBSTACLE_CLEARANCE_M]
        if not len(goals):
            return None
        goal_xy = goals[int(np.argmin(np.linalg.norm(goals - toward_xy, axis=1)))]
        heading = math.atan2(goal_xy[1] - estimated[1], goal_xy[0] - estimated[0])
        return np.array([goal_xy[0], goal_xy[1], heading])

    def _search(
        self,
        time: float,
        estimated: np.ndarray,
        obstacles: np.ndarray,
    ) -> TrajectoryReference:
        """Drive lengthwise to the far end of the room, U-turn, come back, repeat.

        The camera faces the end wall the robot is driving toward, and the
        U-turn arc at each end sweeps it across the room.
        """
        if self._goal is None:
            self._goal = self._end_goal()
            self._set_phase(_Phase.SEARCH)

        goal_distance = float(np.linalg.norm(self._goal[:2] - estimated[:2]))
        if goal_distance <= _SEARCH_ARRIVAL_M:
            self._toward_far_end = not self._toward_far_end
            self._lateral_index = 0
            self._goal = self._end_goal()
            self._set_phase(_Phase.SEARCH)
            goal_distance = float(np.linalg.norm(self._goal[:2] - estimated[:2]))
        if not self._stalled(time, goal_distance):
            try:
                return self._follow(time, estimated, self._goal, obstacles)
            except ValueError:
                pass
        # End goal blocked (obstacle there) or stuck in an APF local minimum:
        # try the next lateral offset, then give up on this end.
        self._lateral_index += 1
        if self._lateral_index >= len(_SEARCH_LATERAL_OFFSETS_M):
            self._lateral_index = 0
            self._toward_far_end = not self._toward_far_end
        self._goal = None
        self._set_phase(_Phase.SEARCH)
        return self._hold(time, estimated)

    # ---------------------------------------------------------- marker approach

    def _confirmed_marker(self, perception: PerceptionOutput) -> int | None:
        """Return a known marker ID once seen in consecutive camera frames."""
        if not perception.new_camera_frame:
            return None
        marker_id = next(
            (
                observation.marker_id
                for observation in perception.markers
                if observation.marker_id in MARKER_MAP
            ),
            None,
        )
        self._marker_detections = self._marker_detections + 1 if marker_id is not None else 0
        if self._marker_detections >= _MARKER_CONFIRMATIONS:
            return marker_id
        return None

    def _start_marker_approach(self, marker_id: int) -> None:
        """Place staging and final poses along the marker's face normal."""
        marker_transform = MARKER_MAP[marker_id]
        marker_xy = marker_transform[:2, 3]
        normal = marker_transform[:2, 2]  # marker z axis: out of the face
        normal = normal / np.linalg.norm(normal)
        heading = math.atan2(-normal[1], -normal[0])
        staging_xy = marker_xy + _STAGING_DISTANCE_M * normal
        final_xy = marker_xy + _FINAL_DISTANCE_M * normal
        self._marker_xy = marker_xy.copy()
        self._approach_normal = normal
        self._staging_pose = np.array([staging_xy[0], staging_xy[1], heading])
        self._final_pose = np.array([final_xy[0], final_xy[1], heading])
        self._set_phase(_Phase.TO_STAGING)

    def _start_final(self, start_xy: np.ndarray) -> None:
        self._detour_goal = None
        self._set_phase(_Phase.FINAL)
        self._final_start = np.array(start_xy, dtype=float)

    def _corridor_start(self, estimated: np.ndarray) -> np.ndarray | None:
        """Point on the marker normal line if the robot can go straight in from here."""
        assert self._marker_xy is not None and self._approach_normal is not None
        assert self._final_pose is not None
        offset = estimated[:2] - self._marker_xy
        along = float(np.dot(offset, self._approach_normal))
        lateral = abs(float(
            offset[0] * self._approach_normal[1] - offset[1] * self._approach_normal[0]
        ))
        heading_error = abs(_wrap_angle(self._final_pose[2] - estimated[2]))
        if (
            _FINAL_DISTANCE_M + 0.15 <= along <= _STAGING_DISTANCE_M + 0.3
            and lateral <= _CORRIDOR_HALF_WIDTH_M
            and heading_error <= _CORRIDOR_HEADING_RAD
        ):
            return self._marker_xy + along * self._approach_normal
        return None

    def _to_staging(
        self,
        time: float,
        estimated: np.ndarray,
        obstacles: np.ndarray,
    ) -> TrajectoryReference:
        assert self._staging_pose is not None
        corridor_start = self._corridor_start(estimated)
        if corridor_start is not None:
            self._start_final(corridor_start)
            return self._final(time, estimated)
        if np.linalg.norm(self._staging_pose[:2] - estimated[:2]) <= _STAGING_ARRIVAL_M:
            self._start_final(self._staging_pose[:2])
            return self._final(time, estimated)

        if (
            self._detour_goal is not None
            and np.linalg.norm(self._detour_goal[:2] - estimated[:2]) <= _POSITION_TOLERANCE_M
        ):
            self._detour_goal = None
            self._set_phase(_Phase.TO_STAGING)
        target = self._detour_goal if self._detour_goal is not None else self._staging_pose
        target_distance = float(np.linalg.norm(target[:2] - estimated[:2]))
        if not self._stalled(time, target_distance):
            try:
                return self._follow(time, estimated, target, obstacles)
            except ValueError:
                pass
        # Stuck or blocked: detour toward the staging point, then retry it.
        if self._detour_goal is None:
            self._detour_goal = self._detour_goal_from_scan(
                estimated, obstacles, self._staging_pose[:2],
            )
        else:
            self._detour_goal = None
        self._set_phase(_Phase.TO_STAGING)
        return self._hold(time, estimated)

    def _final(self, time: float, estimated: np.ndarray) -> TrajectoryReference:
        """Follow the marker's normal line in and stop on the final pose.

        LiDAR avoidance is off since the target is next to the wall. Heading
        is corrected while driving along the line, not by turning in place.
        """
        assert self._final_pose is not None
        assert self._final_start is not None
        offset = self._final_pose[:2] - estimated[:2]
        axis = np.array([math.cos(self._final_pose[2]), math.sin(self._final_pose[2])])
        if (
            np.linalg.norm(offset) <= _FINAL_POSITION_TOLERANCE_M
            or float(np.dot(offset, axis)) <= 0.0
        ):
            self._set_phase(_Phase.DONE)
            return self._reference(time, estimated, 0.0, 0.0, True)
        return self._follow(
            time, estimated, self._final_pose, None, start=self._final_start, dock=True,
        )
