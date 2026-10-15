"""Estimate planar robot pose using wheel movement and camera localisation.

PoseEKF predicts from encoder angles and corrects with a camera-derived robot
world pose. Image processing belongs in perception; simulator truth is never
an estimator input. StateEstimate is an output snapshot, not the filter itself.
"""

from dataclasses import dataclass

import numpy as np

from .sensing import WheelEncoderReading
from .state_model import DifferentialDriveModel, wrap_angle


@dataclass(frozen=True)
class StateEstimate:
    """Pose and uncertainty at a simulation timestamp.

    pose_world is (x [m], y [m], yaw [rad]). Covariance uses that same ordering:
    its diagonal contains x/y variances [m²] and yaw variance [rad²].
    """

    time: float
    pose_world: tuple[float, float, float]
    covariance: np.ndarray

    def __post_init__(self) -> None:
        # frozen=True alone does not prevent changes to a NumPy array's contents.
        # Give this result its own array so later filter updates cannot alter it.
        covariance = np.array(self.covariance, dtype=float, copy=True)
        covariance.setflags(write=False)
        # Assign the protected copy during construction of this frozen dataclass.
        object.__setattr__(self, "covariance", covariance)


class PoseEKF:
    """Maintain a planar pose and its 3x3 covariance between sensor updates.

    The known initial pose and initial covariance define the starting estimate.
    The first encoder reading establishes a baseline without moving the pose.
    Recreate the filter after a simulation reset.

    Encoder noise parameters are standard deviations of absolute angle readings
    [rad]. Each wheel increment is assigned variance 2*sigma². Treating these
    increments as independent is an approximation: consecutive differences of
    noisy angles share one reading and therefore have correlated errors.
    process_noise_covariance_per_s adds pose covariance per elapsed second to
    represent unmodelled motion, such as slip. Both noise models need validation.
    """

    def __init__(
        self,
        motion_model: DifferentialDriveModel,
        initial_pose: tuple[float, float, float],
        initial_covariance: np.ndarray,
        *,
        encoder_noise_std_rad: tuple[float, float] = (0.0, 0.0),
        process_noise_covariance_per_s: np.ndarray | None = None,
    ) -> None:
        # State notation below: s = [x, y, yaw], with covariance P = Cov(s).
        # Validate the starting pose before storing any estimator state.
        pose = np.asarray(initial_pose, dtype=float)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("initial_pose must contain three finite values: x, y, yaw")
        # These are angle standard deviations [rad], not variances [rad^2].
        encoder_std = np.asarray(encoder_noise_std_rad, dtype=float)
        if (encoder_std.shape != (2,) or not np.all(np.isfinite(encoder_std))
                or np.any(encoder_std < 0)):
            raise ValueError("encoder_noise_std_rad must contain two finite, non-negative values")

        # Keep our own arrays rather than sharing mutable storage with the caller.
        self._motion_model = motion_model
        self._pose = pose.copy()
        self._pose[2] = wrap_angle(self._pose[2])
        # P_0 describes uncertainty in the supplied starting world pose.
        self._covariance = _validate_covariance(initial_covariance, "initial_covariance")

        # For independent angle errors: Var(phi_k - phi_(k-1)) = 2 * sigma^2.
        # W = diag(2*sigma_left^2, 2*sigma_right^2), in wheel radians squared.
        # This assumes independent left/right errors and approximates successive
        # increments as independent, although adjacent differences share a reading.
        self._wheel_increment_covariance = np.diag(2.0 * encoder_std**2)

        # C is additional pose covariance per second; prediction adds C * dt.
        # This models uncertainty from unmodelled motion without injecting noise.
        self._process_covariance_per_s = _validate_covariance(
            np.zeros((3, 3)) if process_noise_covariance_per_s is None
            else process_noise_covariance_per_s,
            "process_noise_covariance_per_s",
        )
        # No angle difference or camera freshness check is possible until readings
        # arrive. The first predict() call will establish the encoder baseline.
        self._previous_encoder_reading: WheelEncoderReading | None = None
        self._estimate: StateEstimate | None = None
        self._last_camera_time: float | None = None

    # Predict using encoders
    def predict(self, encoders: WheelEncoderReading) -> StateEstimate:
        """Predict from wheel-angle changes and return pose plus covariance.

        F propagates existing pose uncertainty; G maps wheel-increment noise
        into pose uncertainty. Both are evaluated before changing the pose.
        Repeated timestamps return the current estimate without another step.
        """
        # Reject invalid sensor values before they can contaminate pose/covariance.
        values = (encoders.time, encoders.left_encoder_rad, encoders.right_encoder_rad)
        if not np.all(np.isfinite(values)):
            raise ValueError("Encoder timestamp and wheel angles must be finite")

        # With no previous reading, skip movement and retain the initial pose/P.
        previous = self._previous_encoder_reading
        if previous is not None:
            if encoders.time < previous.time:
                raise ValueError("Encoder timestamps must not go backwards")
            if encoders.time == previous.time:
                # A cached reading must not create another motion/noise update.
                # Return the latest snapshot, including any camera correction.
                assert self._estimate is not None
                return self._estimate

            # Accumulated wheel angles must not wrap, even after a full revolution.
            # u_k = [delta_phi_left, delta_phi_right], where
            # delta_phi = phi_k - phi_(k-1) for each wheel.
            delta_left = encoders.left_encoder_rad - previous.left_encoder_rad
            delta_right = encoders.right_encoder_rad - previous.right_encoder_rad
            # Wheel changes already include elapsed movement; dt only scales C.
            dt = encoders.time - previous.time

            # F = df/ds and G = df/du, evaluated before advancing the stored pose.
            # That pose is the latest prediction or camera-corrected estimate.
            F, G = self._motion_model.motion_jacobians(self._pose, delta_left, delta_right)

            # Q_k = G W G^T + C * dt: map wheel-angle uncertainty into pose
            # uncertainty, then add motion-model uncertainty for this interval.
            Q = (G @ self._wheel_increment_covariance @ G.T
                 + self._process_covariance_per_s * dt)

            # Mean prediction: s_k^- = f(s_(k-1)^+, u_k).
            # '-' denotes a prediction; '+' denotes the previous corrected state
            # (or its prediction if no camera correction was available).
            self._pose = self._motion_model.predict_pose(self._pose, delta_left, delta_right)

            # Covariance prediction: P_k^- = F P_(k-1)^+ F^T + Q_k.
            # Existing uncertainty propagates through F; Q adds new uncertainty.
            self._covariance = F @ self._covariance @ F.T + Q

        # Keep the actual reading for the next subtraction, including on the first
        # call. Return a detached result for control, logging or evaluation.
        self._previous_encoder_reading = encoders
        return self._snapshot(float(encoders.time))

    # Correct using camera
    def correct(
        self,
        camera_pose_world: tuple[float, float, float],
        camera_covariance: np.ndarray,
        time: float,
    ) -> StateEstimate:
        """Correct using a fresh, time-aligned camera robot pose and covariance R.

        Perception must supply the robot's axle-centre pose in our world frame,
        not the marker pose relative to the camera. R uses [x, y, yaw] ordering
        and must be positive definite. For this full-pose observation H = I.

        Predict to the image capture time before calling this method. Future or
        delayed images are rejected; buffering/interpolation is not implemented.
        One camera observation is accepted per timestamp, preventing cached
        frames from being fused repeatedly. Skip this call when no pose is found.
        """
        # Correct the estimate at the image's capture time, not its delivery time.
        # This simple filter requires prediction to have reached that same time.
        if self._estimate is None:
            raise ValueError("Call predict with an encoder baseline before camera correction")
        if not np.isfinite(time) or not np.isclose(time, self._estimate.time, rtol=0, atol=1e-9):
            raise ValueError("Camera capture time must match the current prediction time")
        if self._last_camera_time is not None and time <= self._last_camera_time + 1e-9:
            raise ValueError("This camera timestamp has already been used")

        # z = camera-derived ROBOT world pose; perception has already handled
        # marker detection and frame transforms. R = Cov(camera measurement error).
        measurement = np.asarray(camera_pose_world, dtype=float)
        if measurement.shape != (3,) or not np.all(np.isfinite(measurement)):
            raise ValueError("camera_pose_world must contain three finite values: x, y, yaw")
        R = _validate_covariance(camera_covariance, "camera_covariance", positive_definite=True)

        # Innovation: nu = z - h(s^-). Here h(s) = s and H = dh/ds = I,
        # because both the measurement and state are the same world-frame pose.
        innovation = measurement - self._pose
        # Compare headings along the shortest angular difference across +/-pi.
        innovation[2] = wrap_angle(innovation[2])
        P = self._covariance

        # Innovation covariance: S = H P^- H^T + R = P^- + R.
        # It combines uncertainty in the prediction and camera measurement.
        S = P + R

        # Kalman gain: K = P^- H^T S^-1 = P^- S^-1, since H = I.
        # It determines how the innovation changes the estimate using P and R.
        # Solve for K = P @ inv(S) without explicitly forming an inverse.
        K = np.linalg.solve(S.T, P.T).T

        # Mean correction: s^+ = s^- + K nu; keep the resulting yaw in [-pi, pi).
        self._pose = self._pose + K @ innovation
        self._pose[2] = wrap_angle(self._pose[2])

        # Covariance correction (H = I):
        # P^+ = (I - K) P^- (I - K)^T + K R K^T, where I is the 3x3 identity.
        # Joseph form preserves covariance symmetry/positivity numerically.
        I_minus_K = np.eye(3) - K
        self._covariance = I_minus_K @ P @ I_minus_K.T + K @ R @ K.T

        # Mark this frame as consumed only after a successful correction.
        # The next predict() uses the corrected internal pose and covariance.
        self._last_camera_time = self._estimate.time
        return self._snapshot(self._estimate.time)

    def _snapshot(self, time: float) -> StateEstimate:
        """Remove round-off asymmetry and return a detached output snapshot."""
        # P <- (P + P^T) / 2. A covariance should be symmetric; averaging removes
        # tiny floating-point differences rather than adding physical uncertainty.
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        # Use an immutable pose tuple; StateEstimate copies/protects covariance.
        self._estimate = StateEstimate(
            time=time,
            pose_world=tuple(float(value) for value in self._pose),
            covariance=self._covariance,
        )
        return self._estimate


def _validate_covariance(
    value: np.ndarray, name: str, *, positive_definite: bool = False,
) -> np.ndarray:
    """Copy a finite, symmetric covariance and reject invalid uncertainty inputs."""
    # Validate a private copy so normalization cannot modify the caller's array.
    covariance = np.array(value, dtype=float, copy=True)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite 3×3 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    # P = P^T mathematically; average any accepted round-off asymmetry.
    covariance = 0.5 * (covariance + covariance.T)
    if positive_definite:
        # Camera R must have strictly positive variance in every direction:
        # v^T R v > 0 for any nonzero v. Cholesky tests this via R = L L^T.
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name} must be positive definite") from error
    else:
        # Initial/process covariance may have zero-variance directions:
        # v^T P v >= 0, equivalent to all eigenvalues being non-negative.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # Scale the floating-point allowance to the size of this covariance.
        tolerance = 10 * np.finfo(float).eps * np.max(np.abs(eigenvalues))
        if eigenvalues[0] < -tolerance:
            raise ValueError(f"{name} must be positive semidefinite")
        if eigenvalues[0] < 0:
            # A valid singular covariance can acquire tiny negative eigenvalues
            # from round-off. Remove only those within the relative tolerance.
            # Reconstruct P = V diag(max(lambda, 0)) V^T after that small repair.
            covariance = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    return covariance
