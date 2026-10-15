"""Compare estimated poses with simulator truth for evaluation only."""

# Record results and compare estimates against simulator truth.

import math

import mujoco

from .state_model import wrap_angle


def read_true_pose(data: mujoco.MjData) -> tuple[float, float, float]:
    """Read the base's planar world pose; never feed this to the estimator."""
    base = data.body("base")
    rotation = base.xmat.reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return float(base.xpos[0]), float(base.xpos[1]), yaw


def pose_error(
    estimated_pose: tuple[float, float, float],
    true_pose: tuple[float, float, float],
) -> tuple[float, float]:
    """Return position error [m] and signed wrapped heading error [rad]."""
    x, y, yaw = estimated_pose
    true_x, true_y, true_yaw = true_pose
    return math.hypot(x - true_x, y - true_y), wrap_angle(yaw - true_yaw)
