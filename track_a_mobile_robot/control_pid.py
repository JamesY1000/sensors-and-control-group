"""Nikhil's PID comparison, adapted to the current shared controller interface.

Ported from origin/extension/LQI-vs-PID at
2c0365c55119cce9e89c6df9da572e045bc1ce7f. Gains and feedback laws are unchanged.
PBVS, navigation, command limits and anti-windup remain shared with LQI.
"""
from dataclasses import dataclass
import numpy as np
from .control import MotionController
from .scene_config import NavigationConfig
from .state_model import wrap_angle

@dataclass(frozen=True)
class PIDGains:
    """Gains for one PID loop; the derivative acts on a filtered error rate."""

    kp: float
    ki: float = 0.
    kd: float = 0.


class PIDMotionController(MotionController):
    """PID baseline for comparison with LQI.

    Only the coarse tracking law changes: errors, feedforward, saturation,
    anti-windup and PBVS are inherited so differences come from the feedback.
    Default P/I gains copy the LQI gains at the 0.22 m/s drive speed
    (driving_gain(.22) and the turn gain), frozen instead of scheduled with
    speed, plus a small derivative.
    """

    def __init__(self, settings: NavigationConfig, along=PIDGains(1.73, 1.0, .1),
                 heading=PIDGains(1.94, 0., .1), cross_track_gain=4.0,
                 turn=PIDGains(2.83, 2.0, .05), derivative_tau_s=.1) -> None:
        self._along, self._heading, self._turn = along, heading, turn
        self._cross_track_gain = cross_track_gain
        self._derivative_tau_s = derivative_tau_s
        super().__init__(settings)

    def reset(self) -> None:
        super().reset()
        # Clearing these at stops and segment changes also avoids a derivative kick.
        self._previous_error = None
        self._error_rate = np.zeros(3)

    def _error_derivative(self, dt, error):
        """Finite-difference error rate through a first-order low-pass filter."""
        if self._previous_error is not None and dt > 0:
            raw = (error - self._previous_error) / dt
            raw[2] = wrap_angle(error[2] - self._previous_error[2]) / dt
            blend = dt / (self._derivative_tau_s + dt)
            self._error_rate = self._error_rate + blend * (raw - self._error_rate)
        self._previous_error = error.copy()
        return self._error_rate

    def _turn_lqi_command(self, time, reference, heading_error, dt):
        """Replace the LQI spot-turn law with heading PID; forward velocity stays zero."""
        # The full error was recorded before dispatch; keep its rate history continuous.
        rate = self._error_derivative(dt, np.asarray(self.tracking_error))
        g, turn_limit = self._turn, self.settings.turn_speed_rad_s
        candidate = float(np.clip(self._integral + dt * heading_error, -.8, .8))
        raw = reference.omega_rad_s - (g.kp * heading_error + g.ki * candidate + g.kd * rate[2])
        if abs(raw) <= turn_limit or raw * heading_error > 0:
            self._integral = candidate
        omega = reference.omega_rad_s - (g.kp * heading_error + g.ki * self._integral + g.kd * rate[2])
        return self._limited_command(time, 0, float(np.clip(omega, -turn_limit, turn_limit)))

    def _drive_lqi_command(self, time, reference, error, dt, speed_limit):
        """Replace the LQI driving law with along-track PID and heading/cross-track PD."""
        rate = self._error_derivative(dt, error)
        a, h, direction = self._along, self._heading, reference.direction
        if direction < 0:
            speed_limit = min(speed_limit, self.settings.dock_speed_mps)
        turn_limit = self.settings.turn_speed_rad_s
        candidate = float(np.clip(self._integral + dt * error[0], -.15, .15))
        raw_v = reference.v_mps - (a.kp * error[0] + a.ki * candidate + a.kd * rate[0])
        if abs(raw_v) <= speed_limit or raw_v * error[0] > 0:
            self._integral = candidate
        v = reference.v_mps - (a.kp * error[0] + a.ki * self._integral + a.kd * rate[0])
        # Cross-track error only decays through heading (e_y_dot = v_ref*e_heading),
        # so it feeds the heading loop and flips sign when reversing.
        omega = reference.omega_rad_s - (h.kp * error[2] + h.kd * rate[2]
                                         + self._cross_track_gain * direction * error[1])
        return self._limited_command(time, float(np.clip(v, -speed_limit, speed_limit)),
                               float(np.clip(omega, -turn_limit, turn_limit)))

