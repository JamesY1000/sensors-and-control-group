"""
Our estimator should essentially maintain the estimated robot and dock poses,
and their joint covariance, in a fixed local odometry frame.

It stores:
- Joint state x_hat = [robot x, robot y, robot heading, dock x, dock y, dock heading].T
- Joint covariance P (6 x 6), including uncertainty shared between robot and dock

Currently its inputs are:
- Wheel encoder readings (from sensing)
- Derived dock pose in the current robot frame and its covariance (from perception)

It has 3 stages:
1. Initialisation:
   - Wait for the first usable ArUco pose before starting local odometry
   - Define the robot pose as (0, 0, 0) and use current encoders as the baseline
   - Initialise the unknown dock from the camera measurement and its uncertainty
2. Prediction:
   - Compute wheel angle increments from successive encoder readings
   - Use the diff-drive model to predict the robot pose; the dock is stationary
   - Propagate the joint covariance to account for motion and encoder uncertainty
3. Correction:
   - Compare the measured robot-relative dock pose with the predicted measurement
   - Reject observations that are inconsistent with the predicted uncertainty
   - Correct both poses and their covariance using accepted camera measurements

Without a fresh usable camera pose, the estimate continues from prediction only.
Before acquisition, no odometry is accumulated and dock pose is None.

It outputs:
- A StateEstimate object (time, robot pose/covariance, dock pose/covariance)

Neither the robot's starting world pose nor the dock's world pose is an input.
"""

from dataclasses import dataclass

import numpy as np

from .sensing import WheelEncoderReading
from .state_model import DifferentialDriveModel, wrap_angle


CAMERA_INNOVATION_LIMIT = 11.345 # 99% chi-squared threshold, three components


def create_estimator(settings, model, encoder_noise_std_rad, *, gate_camera=True):
    """Construct the shared EKF from update settings and encoder calibration."""
    process_std = np.array([
        settings.process_position_std_m_per_sqrt_s,
        settings.process_position_std_m_per_sqrt_s,
        np.deg2rad(settings.process_heading_std_deg_per_sqrt_s),
    ])
    return PoseEstimator(model, encoder_noise_std_rad, np.diag(process_std**2), gate_camera=gate_camera)


@dataclass(frozen=True)
class StateEstimate:
    """Store a snapshot for navigation and control, separate from the EKF's working state.

    Before acquisition, pose_local is only the current robot-frame identity.
    After acquisition, both poses use the fixed local frame defined at that instant.
    """

    time: float
    pose_local: tuple[float, float, float]
    covariance: np.ndarray
    dock_pose_local: tuple[float, float, float] | None = None
    dock_covariance: np.ndarray | None = None

    @property
    def dock_pose_base(self) -> tuple[float, float, float] | None:
        """Return the estimated dock pose relative to the current robot for PBVS."""
        if self.dock_pose_local is None:
            return None
        # Both poses share the same local frame. Removing the robot transform gives
        # the relative pose, using the fused estimate rather than a raw camera frame.
        pose, _ = relative_dock_model(np.r_[self.pose_local, self.dock_pose_local])
        return tuple(pose)

    def __post_init__(self) -> None:
        # Copy and protect the arrays so later EKF updates cannot change this result.
        for name in ('covariance', 'dock_covariance'):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, dtype=float, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)


def relative_dock_model(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict dock pose in the robot frame and its Jacobian for the joint state."""
    x, y, heading, dx, dy, dock_heading = state
    # Subtract robot position, then rotate from local odometry into the robot frame.
    # This predicts what the camera should measure from our current joint estimate.
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.array([[c, s], [-s, c]])
    relative_xy = rotation @ np.array([dx - x, dy - y])
    measurement = np.r_[relative_xy, wrap_angle(dock_heading - heading)]
    # H (3 x 6) describes how each state component affects the relative measurement.
    # Changing robot heading rotates the relative XY position as well as its heading.
    H = np.zeros((3, 6))
    H[:2, :2] = -rotation
    H[:2, 2] = [relative_xy[1], -relative_xy[0]]
    H[:2, 3:5] = rotation
    H[2, 2], H[2, 5] = -1.0, 1.0
    return measurement, H


class PoseEstimator:
    """EKF for encoder motion and camera observations of one unknown rigid dock."""

    def __init__(self, model: DifferentialDriveModel, encoder_noise_std_rad,
                 process_noise_covariance_per_s: np.ndarray, *, gate_camera=True) -> None:
        # Store the motion model and noise assumptions used in prediction.
        self._model = model
        self._gate_camera = gate_camera
        self._encoder_noise_std_rad = np.asarray(encoder_noise_std_rad, dtype=float)
        self._process_noise_covariance_per_s = np.array(process_noise_covariance_per_s, copy=True)
        # The first three states belong to the robot; the last three belong to the dock.
        # P's diagonal 3 x 3 blocks store each pose covariance. Its off-diagonal
        # blocks store their cross-covariance (how their estimation errors are related).
        self._state = np.zeros(6)
        self._P = np.zeros((6, 6))
        self._previous_encoders: WheelEncoderReading | None = None
        self._last_camera_time: float | None = None
        # Odometry and the joint dock state both start at the first accepted marker.
        self.initialised = False
        self.camera_correction_accepted = False

    def _estimate(self, time: float) -> StateEstimate:
        # Return each pose's covariance, retaining the full joint P inside the EKF.
        # A missing dock pose tells navigation to continue exploration.
        return StateEstimate(time, tuple(self._state[:3]), self._P[:3, :3],
                             tuple(self._state[3:]) if self.initialised else None,
                             self._P[3:, 3:] if self.initialised else None)

    @staticmethod
    def _measurement(pose, covariance):
        # z = [dock x, dock y, dock heading] in the current robot frame.
        # R describes measurement uncertainty in those same coordinates.
        z, R = np.asarray(pose, dtype=float), np.asarray(covariance, dtype=float)
        if z.shape != (3,) or R.shape != (3, 3) or not np.isfinite(z).all() or not np.isfinite(R).all():
            raise ValueError('Expected a finite relative pose and 3x3 covariance')
        if not np.allclose(R, R.T) or np.min(np.linalg.eigvalsh(R)) <= 0:
            raise ValueError('Camera covariance must be symmetric positive definite')
        return z, R

    def initialise_dock(self, encoders: WheelEncoderReading, dock_pose_base,
                        camera_covariance) -> StateEstimate:
        """Start the local frame at the first accepted marker, excluding earlier travel."""
        if self.initialised:
            raise ValueError('Dock is already initialised; correct subsequent observations')
        z, R = self._measurement(dock_pose_base, camera_covariance)
        # At acquisition the robot defines (0, 0, 0). Its frame and the new local
        # frame coincide, so the measured relative dock pose needs no transformation.
        self._state[:] = 0.
        self._state[3:] = z
        self._state[5] = wrap_angle(z[2])
        # Zero robot covariance defines this local origin, not a known world pose.
        # The unknown dock starts with camera uncertainty; cross-covariance grows
        # through subsequent encoder predictions and camera corrections.
        self._P[:] = 0.
        self._P[3:, 3:] = R
        self._previous_encoders = encoders
        self._last_camera_time = encoders.time
        self.initialised = self.camera_correction_accepted = True
        return self._estimate(encoders.time)

    def predict(self, encoders: WheelEncoderReading) -> StateEstimate:
        """Predict with wheel increments only after the first accepted marker."""
        self.camera_correction_accepted = False
        if not self.initialised:
            # Exploration sends timed commands; do not integrate or store its travel.
            return self._estimate(encoders.time)
        previous = self._previous_encoders
        dt = encoders.time - previous.time
        if dt <= 0:
            raise ValueError('Encoder timestamps must increase')
        # Compute wheel angle increments since the previous prediction.
        delta_left = encoders.left_encoder_rad - previous.left_encoder_rad
        delta_right = encoders.right_encoder_rad - previous.right_encoder_rad
        # F (3 x 3) maps previous robot-pose uncertainty into the next pose.
        # G (3 x 2) maps wheel-increment uncertainty into robot-pose uncertainty.
        # Evaluate both at the previous pose, then predict the new robot pose.
        F, G = self._model.motion_jacobians(self._state[:3], delta_left, delta_right)
        self._state[:3] = self._model.predict_pose(self._state[:3], delta_left, delta_right)
        # Extend F to the joint state. The stationary dock keeps its identity block.
        joint_F = np.eye(6)
        joint_F[:3, :3] = F
        # Each increment subtracts two readings. Assuming independent reading noise,
        # its variance is sigma^2 + sigma^2 = 2 * sigma^2 for each wheel.
        wheel_covariance = np.diag(2.0 * self._encoder_noise_std_rad**2)
        # P_pred = F_joint @ P @ F_joint.T + Q_joint
        # Propagate all blocks, including cross-covariance; add motion noise to robot only.
        self._P = joint_F @ self._P @ joint_F.T
        self._P[:3, :3] += G @ wheel_covariance @ G.T + self._process_noise_covariance_per_s * dt
        # Remove small numerical asymmetry from floating-point calculations.
        self._P = .5 * (self._P + self._P.T)
        self._previous_encoders = encoders
        return self._estimate(encoders.time)

    def correct(self, dock_pose_base, camera_covariance, time: float) -> StateEstimate:
        """Fuse a relative dock pose, retaining robot/dock uncertainty and correlation."""
        if not self.initialised or self._previous_encoders is None:
            raise ValueError('Initialise the dock before correction')
        # Fuse each frame once, at the encoder prediction time it belongs to.
        if not np.isclose(time, self._previous_encoders.time, rtol=0, atol=1e-9):
            raise ValueError('Camera and encoder timestamps must be in sync')
        if time <= self._last_camera_time:
            raise ValueError('Camera measurement has already been used or is older')
        z, R = self._measurement(dock_pose_base, camera_covariance)
        # Innovation = measured relative dock pose - predicted relative dock pose.
        # Wrap the heading difference so crossing +/-pi does not create a large error.
        predicted, H = relative_dock_model(self._state)
        innovation = z - predicted
        innovation[2] = wrap_angle(innovation[2])
        # S is innovation covariance: predicted measurement uncertainty plus camera noise.
        S = H @ self._P @ H.T + R
        # Mark even rejected frames as consumed so they cannot be retried as new data.
        self._last_camera_time = time
        self.camera_correction_accepted = False
        # Gate using the normalised innovation, not a fixed distance/angle threshold.
        # A large disagreement relative to S leaves the predicted state and P unchanged.
        if self._gate_camera and float(innovation @ np.linalg.solve(S, innovation)) > CAMERA_INNOVATION_LIMIT:
            return self._estimate(time)
        # Kalman gain K = P @ H.T @ inv(S), computed with a solve instead of an inverse.
        # Shared uncertainty allows this relative observation to update robot and dock.
        K = np.linalg.solve(S, H @ self._P).T
        self._state += K @ innovation
        self._state[2] = wrap_angle(self._state[2])
        self._state[5] = wrap_angle(self._state[5])
        # Joseph covariance update keeps the two uncertainty contributions explicit:
        # P = (I - K H) @ P @ (I - K H).T + K @ R @ K.T
        I_minus_KH = np.eye(6) - K @ H
        self._P = I_minus_KH @ self._P @ I_minus_KH.T + K @ R @ K.T
        self._P = .5 * (self._P + self._P.T)
        self.camera_correction_accepted = True
        return self._estimate(time)
