"""Feedback control for tracking a time-stamped planar pose reference."""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .estimation import StateEstimate
from .navigation import TrajectoryReference

_WHEEL_RADIUS_M = 0.033
_WHEEL_TRACK_M = 0.288
_MAX_WHEEL_SPEED_RAD_S = 7.88

_LQI_STATE_WEIGHTS = (4.0, 8.0, 4.0, 0.5, 0.5)
_LQI_INPUT_WEIGHTS = (1.0, 1.0)
_INTEGRAL_ERROR_LIMITS = np.array([1.0, 1.0])
_DEFAULT_CONTROL_PERIOD_S = 0.05
_MIN_DESIGN_SPEED_MPS = 0.05
# Gains are designed on a quantized grid so a continuously varying reference
# reuses cached Riccati solutions instead of solving one every control step.
_DESIGN_SPEED_STEP_MPS = 0.01
_DESIGN_OMEGA_STEP_RAD_S = 0.1
# The robot never turns tighter than this (|omega| <= |v| / radius), so it
# never spins on the spot, where the wheels slip and odometry drifts.
_MIN_TURN_RADIUS_M = 0.15

# Backwards-compatible name for callers using the previous controller API.
MotionReference = TrajectoryReference


@dataclass(frozen=True)
class VelocityCommand:
    """Feasible planar velocity command [time s, v m/s, omega rad/s]."""

    time: float
    v_mps: float
    omega_rad_s: float


class LQIController:
    """Pose feedback controller with trapezoidal integral error action."""

    def __init__(self) -> None:
        self._integral_error = np.zeros(2, dtype=float)
        self._previous_error: np.ndarray | None = None
        self._previous_time: float | None = None
        self._previous_command_saturated = False
        self._segment_id: int | None = None

    def reset(self) -> None:
        """Clear accumulated error, for example when a trajectory stops."""
        self._integral_error.fill(0.0)
        self._previous_error = None
        self._previous_time = None
        self._previous_command_saturated = False
        self._segment_id = None

    def compute_velocity_command(
        self,
        estimated_state: StateEstimate,
        reference: TrajectoryReference,
    ) -> VelocityCommand:
        """Compute a feasible command and update the integral state."""
        if reference.stop_requested:
            self.reset()
            return VelocityCommand(time=float(estimated_state.time), v_mps=0.0, omega_rad_s=0.0)
        if self._segment_id is not None and reference.segment_id != self._segment_id:
            # New goal or phase: error accumulated for the old one no longer applies.
            self.reset()
        self._segment_id = reference.segment_id

        error, time = _pose_error(estimated_state, reference)
        elapsed = (
            time - self._previous_time
            if self._previous_time is not None
            else _DEFAULT_CONTROL_PERIOD_S
        )
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("State estimate time must be finite and non-decreasing.")
        design_period = elapsed if elapsed > 0 else _DEFAULT_CONTROL_PERIOD_S
        gain = _lqi_gain(
            _quantize(reference.v_ref_mps, _DESIGN_SPEED_STEP_MPS),
            _quantize(reference.omega_ref_rad_s, _DESIGN_OMEGA_STEP_RAD_S),
            float(round(design_period, 3)),
        )
        # Below the design speed the lateral error is not controllable by
        # turning in place, so integrating it would only wind up.
        moving = abs(reference.v_ref_mps) >= _MIN_DESIGN_SPEED_MPS
        candidate_integral = self._integral_error.copy()
        if self._previous_time is not None:
            if (
                moving
                and self._previous_error is not None
                and not self._previous_command_saturated
            ):
                candidate_integral += (
                    (self._previous_error[:2] + error[:2])
                    * (0.5 * elapsed)
                )
            candidate_integral = np.clip(
                candidate_integral,
                -_INTEGRAL_ERROR_LIMITS,
                _INTEGRAL_ERROR_LIMITS,
            )

        command, saturated = _command_with_integral(
            time, error, candidate_integral, reference, gain,
        )
        if saturated and not np.array_equal(candidate_integral, self._integral_error):
            # Do not accumulate error while the requested motion is limited.
            candidate_integral = self._integral_error.copy()
            command, saturated = _command_with_integral(
                time,
                error,
                candidate_integral,
                reference,
                gain,
            )

        self._integral_error = candidate_integral
        self._previous_error = error
        self._previous_time = time
        self._previous_command_saturated = saturated
        return command


def velocity_command_to_wheel_speeds(command: VelocityCommand) -> tuple[float, float]:
    """Convert a planar velocity command to left/right wheel speeds in rad/s."""
    left = (command.v_mps - command.omega_rad_s * _WHEEL_TRACK_M / 2) / _WHEEL_RADIUS_M
    right = (command.v_mps + command.omega_rad_s * _WHEEL_TRACK_M / 2) / _WHEEL_RADIUS_M

    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("Velocity command produces non-finite wheel speeds.")
    if max(abs(left), abs(right)) > _MAX_WHEEL_SPEED_RAD_S + 1e-9:
        raise ValueError("Velocity command exceeds the wheel speed limits.")

    return left, right


def _pose_error(estimated_state: StateEstimate, reference: TrajectoryReference) -> tuple[np.ndarray, float]:
    pose = np.asarray(estimated_state.pose_world, dtype=float)

    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError("State estimate pose_world must contain three finite values.")
    time = float(estimated_state.time)
    if not math.isfinite(time):
        raise ValueError("State estimate time must be finite.")

    x, y, heading = (float(value) for value in pose)
    target_x, target_y, target_heading = reference.pose_world
    world_dx = target_x - x
    world_dy = target_y - y
    cos_heading = math.cos(heading)
    sin_heading = math.sin(heading)
    error_x = cos_heading * world_dx + sin_heading * world_dy
    error_y = -sin_heading * world_dx + cos_heading * world_dy
    error_heading = math.atan2(math.sin(target_heading - heading), math.cos(target_heading - heading))
    return np.array([error_x, error_y, error_heading]), time

def _quantize(value: float, step: float) -> float:
    return float(round(value / step) * step)


def _command_with_integral(
    time: float,
    error: np.ndarray,
    integral_error: np.ndarray,
    reference: TrajectoryReference,
    gain: np.ndarray,
) -> tuple[VelocityCommand, bool]:
    """Return the feasible command and whether turn-radius or wheel limits applied."""
    augmented_error = np.concatenate((error, integral_error))
    correction = -gain @ augmented_error
    v_requested = float(
        reference.v_ref_mps * math.cos(float(error[2])) + correction[0]
    )
    omega_requested = float(reference.omega_ref_rad_s + correction[1])
    if not math.isfinite(v_requested) or not math.isfinite(omega_requested):
        raise ValueError("Pose and reference produce a non-finite velocity command.")
    max_omega = abs(v_requested) / _MIN_TURN_RADIUS_M
    turn_limited = abs(omega_requested) > max_omega
    if turn_limited:
        omega_requested = math.copysign(max_omega, omega_requested)

    left_wheel_rad_s = (
        v_requested - omega_requested * _WHEEL_TRACK_M / 2
    ) / _WHEEL_RADIUS_M
    right_wheel_rad_s = (
        v_requested + omega_requested * _WHEEL_TRACK_M / 2
    ) / _WHEEL_RADIUS_M
    if not math.isfinite(left_wheel_rad_s) or not math.isfinite(right_wheel_rad_s):
        raise ValueError("Requested motion produces non-finite wheel speeds.")
    peak_wheel_speed = max(abs(left_wheel_rad_s), abs(right_wheel_rad_s))
    scale = max(1.0, peak_wheel_speed / _MAX_WHEEL_SPEED_RAD_S)

    return (
        VelocityCommand(
            time=time,
            v_mps=v_requested / scale,
            omega_rad_s=omega_requested / scale,
        ),
        turn_limited or peak_wheel_speed > _MAX_WHEEL_SPEED_RAD_S,
    )

@lru_cache(maxsize=256)
def _lqi_gain(
    reference_speed_mps: float,
    reference_omega_rad_s: float,
    sample_period_s: float,
) -> np.ndarray:
    """Solve the discrete LQI Riccati equation for the linearized pose error."""
    if not all(math.isfinite(value) for value in (
        reference_speed_mps,
        reference_omega_rad_s,
        sample_period_s,
    )) or sample_period_s <= 0:
        raise ValueError("Reference motion and sample period must be finite; period must be positive.")

    design_speed = reference_speed_mps
    if abs(design_speed) < _MIN_DESIGN_SPEED_MPS:
        design_speed = math.copysign(
            _MIN_DESIGN_SPEED_MPS,
            design_speed if design_speed else 1.0,
        )

    # Linearized error dynamics: e_dot = A e + B delta_u, where delta_u
    # is the change in commanded (v, omega) around the reference motion.
    continuous_a = np.array([
        [0.0, reference_omega_rad_s, 0.0],
        [-reference_omega_rad_s, 0.0, design_speed],
        [0.0, 0.0, 0.0],
    ])
    continuous_b = np.array([
        [-1.0, 0.0],
        [0.0, 0.0],
        [0.0, -1.0],
    ])
    discrete_a = np.eye(3) + sample_period_s * continuous_a
    discrete_b = sample_period_s * continuous_b

    # Trapezoidal integration of the measured pose error is part of the
    # augmented state used by the discrete-time LQI design.
    integral_selector = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    augmented_a = np.block([
        [discrete_a, np.zeros((3, 2))],
        [
            0.5 * sample_period_s * integral_selector @ (np.eye(3) + discrete_a),
            np.eye(2),
        ],
    ])
    augmented_b = np.vstack((
        discrete_b,
        0.5 * sample_period_s * integral_selector @ discrete_b,
    ))
    state_cost = np.diag(_LQI_STATE_WEIGHTS)
    input_cost = np.diag(_LQI_INPUT_WEIGHTS)
    riccati = _solve_discrete_riccati(augmented_a, augmented_b, state_cost, input_cost)
    gain = np.linalg.solve(
        input_cost + augmented_b.T @ riccati @ augmented_b,
        augmented_b.T @ riccati @ augmented_a,
    )
    closed_loop = augmented_a - augmented_b @ gain
    if np.max(np.abs(np.linalg.eigvals(closed_loop))) >= 1.0:
        raise RuntimeError("Calculated LQI gain does not stabilize the linearized model.")
    return gain


def _solve_discrete_riccati(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Solve the discrete algebraic Riccati equation by structure-preserving doubling.

    Plain fixed-point iteration needs tens of thousands of steps when the
    integrator modes sit near the unit circle (small sample period, large
    reference turn rate); doubling converges quadratically.
    """
    identity = np.eye(a.shape[0])
    a_k = a.copy()
    g_k = b @ np.linalg.solve(r, b.T)
    h_k = q.copy()
    for _ in range(100):
        w = identity + g_k @ h_k
        w_inv_a = np.linalg.solve(w, a_k)
        w_inv_g = np.linalg.solve(w, g_k)
        next_a = a_k @ w_inv_a
        next_g = g_k + a_k @ w_inv_g @ a_k.T
        next_h = h_k + a_k.T @ h_k @ w_inv_a
        next_g = 0.5 * (next_g + next_g.T)
        next_h = 0.5 * (next_h + next_h.T)
        if np.max(np.abs(next_h - h_k)) <= 1e-12 * max(1.0, float(np.max(np.abs(next_h)))):
            return next_h
        a_k, g_k, h_k = next_a, next_g, next_h
    raise RuntimeError("Discrete LQI Riccati doubling did not converge.")
