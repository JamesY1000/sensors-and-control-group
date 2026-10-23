"""
Our controller should approach with LQI, then dock with PBVS using estimated poses.

Currently its inputs are:
- StateEstimate from the encoder/camera EKF
- MotionRequest containing a coarse reference or a PBVS dock offset

It has 3 main stages:
1. For coarse approach/detours, compute pose errors relative to the reference.
2. Select the feedback law for the current phase:
   - LQI adds feedback to reference velocity during coarse approach/detours.
   - Driving uses along-track, cross-track, heading and integrated along-track error.
   - Turning uses heading and integrated heading error, with zero forward speed.
   - PBVS commands velocity directly from the estimated dock-relative pose error.
3. Limit the command and convert it into left/right wheel speeds.

It outputs:
- A VelocityCommand (time, forward velocity, angular velocity)

Exploration uses timed commands before odometry starts. Approach tracks references
using the encoder/camera estimate, never simulator poses. Navigation checks
LiDAR clearance before commands are applied.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.linalg import solve_continuous_are

from .estimation import StateEstimate
from .navigation import MotionRequest, dock_target_pose
from .scene_config import NavigationConfig
from .state_model import wrap_angle

_WHEEL_RADIUS_M = .033
_WHEEL_TRACK_M = .288
_MAX_WHEEL_SPEED_RAD_S = 7.88


@dataclass(frozen=True)
class VelocityCommand:
    """Store forward velocity (m/s) and counter-clockwise yaw rate (rad/s)."""

    time: float
    v_mps: float
    omega_rad_s: float


def limited_command(time, v, omega) -> VelocityCommand:
    """Scale both velocities together to respect wheel speed limits."""
    if not all(math.isfinite(value) for value in (time, v, omega)):
        raise ValueError('Command values must be finite')
    peak = max(abs(v - omega * _WHEEL_TRACK_M / 2), abs(v + omega * _WHEEL_TRACK_M / 2)) / _WHEEL_RADIUS_M
    scale = max(1.0, peak / _MAX_WHEEL_SPEED_RAD_S)
    return VelocityCommand(time, v / scale, omega / scale)


def lqi_model(speed, turn_rate=0.):
    """Linearise driving in the moving reference axes, then add an error integral.

    e = estimated pose - reference pose, rotated into reference heading axes.
    z = [along error, across error, heading error, integral of along error].
    e_x_dot = omega_ref*e_y + delta_v;
    e_y_dot = -omega_ref*e_x + v_ref*e_heading.
    e_heading_dot = delta_omega. The last row integrates e_x.
    """
    A = np.zeros((4, 4))
    A[0, 1] = turn_rate
    A[1, 0] = -turn_rate
    A[1, 2] = speed
    A[3, 0] = 1.
    B = np.array([[1., 0.], [0., 0.], [0., 1.], [0., 0.]])
    Q = np.diag([4., 16., 2., 4.])
    R = np.diag([4., 1.])
    return A, B, Q, R


@lru_cache(maxsize=256)
def driving_gain(speed, turn_rate=0.):
    """Solve the augmented LQR problem once per scheduled reference speed and turn rate."""
    A, B, Q, R = lqi_model(speed, turn_rate)
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


class MotionController:
    """LQI coarse tracking and planar PBVS alignment/docking, with shared limits."""

    def __init__(self, settings: NavigationConfig) -> None:
        self.settings = settings
        # In-place turns cannot control x/y. Design LQI only for heading and its
        # integral instead of solving an uncontrollable stationary pose problem.
        A = np.array([[0., 0.], [1., 0.]])
        B = np.array([[1.], [0.]])
        R = np.array([[1.]])
        P = solve_continuous_are(A, B, np.diag([4., 4.]), R)
        self._turn_gain = np.linalg.solve(R, B.T @ P)[0]
        self.reset()

    def reset(self) -> None:
        self._integral = 0.
        self._previous_time = None
        self._segment = None

    def _pbvs_command(self, state, request):
        """Servo the estimated robot-relative target pose using differential-drive motion.

        rho is target distance, alpha its bearing, and beta the remaining final
        orientation after facing the target. No timed trajectory is generated.
        """
        target = dock_target_pose(state.dock_pose_base, request.dock_offset_m)
        if not np.isfinite(target).all():
            raise ValueError('PBVS requires a finite estimated relative pose')
        rho = float(np.linalg.norm(target[:2]))
        yaw = float(target[2])
        limit = self.settings.turn_speed_rad_s
        if rho <= self.settings.position_tolerance_m:
            # Translation is settled; finish the heading without driving sideways.
            # Wheel friction creates an in-place turn deadband of about 0.23 rad/s
            # in this model. Use 0.30 outside tolerance so alignment does not stall.
            omega = (0. if abs(yaw) <= math.radians(self.settings.heading_tolerance_deg)
                     else math.copysign(max(.30, 1.5*abs(yaw)), yaw))
            return limited_command(state.time, 0., float(np.clip(omega, -limit, limit)))
        alpha = math.atan2(target[1], target[0])
        direction = 1
        if abs(alpha) > math.pi/2 and abs(yaw) < math.pi/4:
            # Back out to the offset when too close, keeping the camera towards the dock.
            direction = -1
            alpha = wrap_angle(alpha - math.pi)
        beta = wrap_angle(yaw - alpha)
        omega = float(np.clip(1.5*alpha - .5*beta, -limit, limit))
        # Turn first for a large bearing error. The robot cannot translate laterally.
        # Keep enough wheel speed to overcome low-speed friction outside tolerance;
        # otherwise proportional feedback can stall just short of the offset.
        speed = min(request.speed_limit_mps, max(.04, .6*rho))
        v = direction*speed*max(0., math.cos(alpha)) if abs(alpha) < math.pi/3 else 0.
        return limited_command(state.time, v, omega)

    def compute_velocity_command(self, state: StateEstimate, request: MotionRequest) -> VelocityCommand:
        time = float(state.time)
        if not math.isfinite(time) or not math.isclose(time, request.time, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Motion request must match the current estimate timestamp')
        if request.stop_requested:
            self.reset()
            return limited_command(time, 0, 0)
        if request.mode in ('SCAN', 'EXPLORE'):
            # Before acquisition there is no pose feedback: just bounded velocity
            # commands converted to wheel speeds. Navigation still checks LiDAR.
            self.reset()
            return limited_command(time, min(request.speed_limit_mps, self.settings.drive_speed_mps),
                                   float(np.clip(request.turn_rate_rad_s,
                                                 -self.settings.turn_speed_rad_s, self.settings.turn_speed_rad_s)))
        if request.visual_servo:
            self.reset()  # Do not carry the LQI integral into a different control law.
            if state.dock_pose_local is None:
                return limited_command(time, 0, 0)
            return self._pbvs_command(state, request)
        reference = request.reference
        if reference is None:
            self.reset()
            return limited_command(time, 0, 0)
        if not np.isfinite(state.pose_local).all():
            raise ValueError('Tracking requires a finite estimated pose')
        if not math.isclose(reference.time, time, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Reference must match the current estimate timestamp')
        segment = (request.mode, reference.segment_id, reference.kind)
        if segment != self._segment:
            self.reset()
            self._segment = segment
        dt = 0. if self._previous_time is None else time - self._previous_time
        if dt < 0:
            raise ValueError('Controller timestamps must be non-decreasing')
        self._previous_time = time

        # Compare poses in the same fixed local frame, then express position
        # error along/across the reference heading. Wrap heading across +/-pi.
        error = np.asarray(state.pose_local) - np.asarray(reference.pose_local)
        c, s = math.cos(reference.pose_local[2]), math.sin(reference.pose_local[2])
        error[:2] = np.array([[c, s], [-s, c]]) @ error[:2]
        error[2] = wrap_angle(error[2])
        turn_limit = self.settings.turn_speed_rad_s
        if reference.kind == 'TURN':
            candidate = float(np.clip(self._integral + dt * error[2], -.8, .8))
            raw = reference.omega_rad_s - self._turn_gain @ [error[2], candidate]
            # Freeze integration at saturation unless it moves the command back
            # towards the allowed range. Stops and new segments clear the integral.
            if abs(raw) <= turn_limit or raw * error[2] > 0:
                self._integral = candidate
            omega = reference.omega_rad_s - self._turn_gain @ [error[2], self._integral]
            return limited_command(time, 0, float(np.clip(omega, -turn_limit, turn_limit)))

        # At the smooth ramp endpoints v_ref is zero. Use a small signed design
        # speed there; full pose stabilisation at rest is handled by navigation's
        # turn/drive switching, not claimed for this linear model.
        direction = reference.direction
        design_speed = direction * max(.02, round(abs(reference.v_mps), 2))
        K = driving_gain(design_speed, round(reference.omega_rad_s, 2))
        speed_limit = min(request.speed_limit_mps, self.settings.dock_speed_mps) if direction < 0 else request.speed_limit_mps
        candidate = float(np.clip(self._integral + dt * error[0], -.15, .15))
        feedforward = np.array([reference.v_mps, reference.omega_rad_s])
        raw = feedforward - K @ np.r_[error, candidate]
        if abs(raw[0]) <= speed_limit or raw[0] * error[0] > 0:
            self._integral = candidate
        velocity = feedforward - K @ np.r_[error, self._integral]
        return limited_command(time, float(np.clip(velocity[0], -speed_limit, speed_limit)),
                               float(np.clip(velocity[1], -turn_limit, turn_limit)))


def velocity_command_to_wheel_speeds(command: VelocityCommand) -> tuple[float, float]:
    """Convert the limited planar command into wheel angular velocities (rad/s)."""
    left = (command.v_mps - command.omega_rad_s * _WHEEL_TRACK_M / 2) / _WHEEL_RADIUS_M
    right = (command.v_mps + command.omega_rad_s * _WHEEL_TRACK_M / 2) / _WHEEL_RADIUS_M
    if not all(math.isfinite(value) for value in (left, right)):
        raise ValueError('Wheel speeds must be finite')
    if max(abs(left), abs(right)) > _MAX_WHEEL_SPEED_RAD_S + 1e-9:
        raise ValueError('Wheel speeds exceed their limits')
    return left, right
