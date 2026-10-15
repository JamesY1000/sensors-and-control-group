"""Predict TurtleBot planar pose from measured wheel-angle changes.

Provides nominal wheel geometry and a midpoint wheel-odometry calculation.

Pose is [x, y, yaw] in world-frame metres and radians, referenced to the axle
midpoint projected onto the floor. Robot axes are +X forward and +Y left;
positive yaw is counterclockwise and positive wheel rotation drives forward.

Assumes planar motion, equal wheel radii and rolling without slip. Discrete pose
prediction uses midpoint integration, an approximation for short wheel-motion
increments. Motor dynamics, contact effects and slip are outside this model.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DifferentialDriveModel:
    """Fixed assumed geometry and deterministic motion calculations.

    Defaults match the nominal robot files: the scaled tyre meshes have a
    0.033 m radius, and wheel centres at y = +/-0.144 m give 0.288 m separation.
    Calibrated or deliberately perturbed dimensions can be supplied at setup.
    """

    wheel_radius_m: float = 0.033
    wheel_separation_m: float = 0.288

    def __post_init__(self) -> None:
        # The dataclass-generated constructor calls this automatically once.
        # Both lengths must be usable physical dimensions. Wheel separation b
        # appears in denominators in the turning equations, so cannot be zero.
        for name in ("wheel_radius_m", "wheel_separation_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    def wheel_displacement(
        self,
        delta_left_rad: float,
        delta_right_rad: float,
    ) -> tuple[float, float]:
        """Return signed travel distance [m] and heading change [rad].

        Inputs are changes in accumulated encoder angles, not absolute angles.
        Do not wrap these changes: a wheel can turn through multiple revolutions.
        """
        # r = wheel radius [m], b = separation between wheel centres [m].
        # Assuming rolling without slip: d_left = r * delta_left_rad, and likewise
        # for the right wheel. Positive distances mean forward wheel travel.
        r, b = self.wheel_radius_m, self.wheel_separation_m

        # distance = (d_left + d_right) / 2 [m]: average wheel travel gives
        # the distance travelled by the midpoint of the axle.
        distance = 0.5 * r * (delta_left_rad + delta_right_rad)

        # delta_yaw = (d_right - d_left) / b [rad]. More forward movement
        # on the right turns the robot left, giving positive yaw.
        heading_change = r * (delta_right_rad - delta_left_rad) / b
        return distance, heading_change

    # Used by my estimator (eg. diffdrivemodel.predict_pose(...))
    def predict_pose(
        self,
        pose: np.ndarray,
        delta_left_rad: float,
        delta_right_rad: float,
    ) -> np.ndarray:
        """Return a new pose using the midpoint heading during the increment.

        No dt is needed: wheel-angle changes already describe movement over the
        sampling interval. The supplied pose is left unchanged.
        """
        # First convert the two wheel rotations into one travel distance and turn.
        distance, heading_change = self.wheel_displacement(
            delta_left_rad, delta_right_rad
        )
        x, y, yaw = pose

        # phi = yaw + delta_yaw / 2: approximate the curved motion by travelling
        # the full distance along the heading halfway through the turn.
        middle_heading = yaw + 0.5 * heading_change

        # x_next = x + distance * cos(phi)
        # y_next = y + distance * sin(phi)
        # yaw_next = wrap(yaw + delta_yaw)
        # Project the movement into world axes and return a separate pose array.
        return np.array([
            x + distance * np.cos(middle_heading),
            y + distance * np.sin(middle_heading),
            wrap_angle(yaw + heading_change),
        ])

    def motion_jacobians(
        self,
        pose: np.ndarray,
        delta_left_rad: float,
        delta_right_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pose Jacobian F (3x3) and wheel-increment Jacobian G (3x2).

        Evaluate at the pose before prediction and the current wheel changes.
        Both describe predict_pose: rows are [x (m), y (m), yaw (rad)]. F's
        columns are the previous pose; G's columns are left/right wheel radians.
        Yaw derivatives describe local angle changes across the wrap boundary.
        """
        # Use exactly the same midpoint motion as predict_pose so uncertainty
        # propagation describes the pose calculation the estimator actually uses.
        distance, heading_change = self.wheel_displacement(
            delta_left_rad, delta_right_rad
        )
        middle_heading = pose[2] + 0.5 * heading_change
        cos_heading = np.cos(middle_heading)
        sin_heading = np.sin(middle_heading)

        # For wheel-angle changes L and R:
        # d(distance)/dL = d(distance)/dR = r/2
        # d(delta_yaw)/dL = -r/b; d(delta_yaw)/dR = +r/b.
        # The midpoint heading changes by half as much as delta_yaw, producing
        # the distance * r/(2*b) terms when differentiating x_next and y_next.
        half_radius = 0.5 * self.wheel_radius_m
        heading_per_wheel = self.wheel_radius_m / self.wheel_separation_m
        turn_term = 0.5 * distance * heading_per_wheel

        # F = df/dpose, where f is predict_pose and pose = [x, y, yaw].
        # With d = distance, phi = middle_heading, C = cos(phi), S = sin(phi):
        # F = [[1, 0, -d*S], [0, 1, d*C], [0, 0, 1]].
        # Existing heading uncertainty therefore becomes position uncertainty
        # as the robot moves; existing x/y errors carry through unchanged.
        F = np.array([
            [1.0, 0.0, -distance * sin_heading],
            [0.0, 1.0, distance * cos_heading],
            [0.0, 0.0, 1.0],
        ])

        # G = df/d[L, R]. Define a = r/2, k = r/b, t = distance*k/2:
        # G = [[a*C + t*S, a*C - t*S],
        #      [a*S - t*C, a*S + t*C],
        #      [       -k,         k]].
        # Each wheel affects both distance and heading. The +/- terms account
        # for the opposite turning effect of left and right wheel errors.
        G = np.array([
            [half_radius * cos_heading + turn_term * sin_heading,
             half_radius * cos_heading - turn_term * sin_heading],
            [half_radius * sin_heading - turn_term * cos_heading,
             half_radius * sin_heading + turn_term * cos_heading],
            [-heading_per_wheel, heading_per_wheel],
        ])
        return F, G


def wrap_angle(angle_rad: float) -> float:
    """Return an equivalent robot heading in [-pi, pi)."""
    # wrapped = ((angle + pi) mod (2*pi)) - pi.
    # Shift the interval to [0, 2*pi), remove full revolutions, then shift back.
    # Use this for robot headings, not accumulated wheel angles or increments.
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi
