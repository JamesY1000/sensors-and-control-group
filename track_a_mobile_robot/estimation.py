"""
Our estimator should essentially maintain the estimated pose and its covariance (x_hat = [x_hat, y_hat, theta_hat].T, and P)

Currently its inputs are:
- Wheel encoder readings (from sensing)
- Derived robot pose from aruco marker detections when available (via RGB camera) (from perception)

It has 2 stages:
- Prediction:
    - Computes wheel angle increments from successive wheel encoder readings
    - Uses the diff-drive model and previous estimated pose to predict the current pose (dead reckoning)
    - Propagate the covariance to account for encoder and motion-model uncertainty
- Correction:
    - Correct the predicted robot pose and covariance using valid and fresh robot pose derived from detected aruco markers (using the RGB camera)
    
The correction stage will only occur if there are valid and fresh aruco marker derived robot poses, otherwise the output estimated 
robot pose will just be the predicted pose using wheel encoders.

It ouputs:
- A fused estimated robot pose as a StateEstimate object (time, pose, covariance)

"""

from dataclasses import dataclass

import numpy as np

from .sensing import WheelEncoderReading
from .state_model import DifferentialDriveModel, wrap_angle


@dataclass(frozen=True)
class StateEstimate:
    """
    Store the state estimate (pose and uncertainty) at a simulation snapshot.
    Store time, robot pose (in the world frame), covariance
    """

    time: float
    pose_world: tuple[float, float, float]
    covariance: np.ndarray

    def __post_init__(self) -> None:
        # Give this result its own array so later filter updates cannot alter it.
        covariance = np.array(self.covariance, dtype=float, copy=True)
        covariance.setflags(write=False)
        # Assign the protected copy during construction of this frozen dataclass.
        object.__setattr__(self, "covariance", covariance)

class PoseEstimator:
    """
    Extended Kalman Filter fuses encoder data (used in the prediction step) with
    aruco marker detections (correction step).
    """
    
    # initialise with the drive model, initial pose and covariance, and noise settings
    def __init__(
        self,
        model: DifferentialDriveModel,
        initial_pose: tuple[float, float, float],
        initial_covariance: np.ndarray,
        encoder_noise_std_rad: tuple[float, float],
        process_noise_covariance_per_s: np.ndarray,
    ) -> None:

        self._model = model # Motion model used for prediction and its Jacobians
        self._pose = np.array(initial_pose, dtype=float, copy=True) # Current estimated pose
        self._pose[2] = wrap_angle(self._pose[2]) # Normalise heading to -pi, pi
        self._P = np.array(initial_covariance, dtype=float, copy=True) # Current uncertainty
        self._encoder_noise_std_rad = np.array(encoder_noise_std_rad, dtype=float, copy=True)
        self._process_noise_covariance_per_s = np.array(process_noise_covariance_per_s, dtype=float, copy=True)
        
        # Set empty first/previous encoder reading until the first one arrives
        self._previous_encoders: WheelEncoderReading | None = None
        
        # Same for camera timestamp
        self._last_camera_time: float | None = None
    
        
        

    # Predict using encoders
    def predict(
        self, 
        encoders: WheelEncoderReading
    ) -> StateEstimate:
        """
        Use wheel encoder readings to predict new robot pose and covariance from current estimate
        """
        
        # Set the previous encoder reading
        previous_encoders = self._previous_encoders
        
        if previous_encoders is not None:
            
            # Get 
            dt = encoders.time - previous_encoders.time
            
            if dt <= 0:
                raise ValueError("Encoder timetamps must increase")
            
            # Get the change in encoders in rads since last reading
            delta_left = encoders.left_encoder_rad - previous_encoders.left_encoder_rad
            delta_right = encoders.right_encoder_rad - previous_encoders.right_encoder_rad
            
            # Get the Jacobians at previous pose estimate
            F, G = self._model.motion_jacobians(self._pose, delta_left, delta_right)
            
            # Predict new robot pose
            self._pose = self._model.predict_pose(self._pose, delta_left, delta_right)
            
            # Compute covariance of wheel increments
            # Each wheel increment subtracts two independent noisy angle readings,
            # so its variance is sigma^2 + sigma^2 = 2*sigma^2
            wheel_increment_covariance = np.diag(2.0 * self._encoder_noise_std_rad **2)
            
            # Propagate existing uncertainty and add the motion uncertainty
            # Should be previous uncertainty + encoder uncertainty + additional model uncertainty 
            # So P_k = F*P_k-1*F.T + G*(wheel covariance)*G.T + Q_rate*dT
            self._P = (F @ self._P @ F.T + G @ wheel_increment_covariance @ G.T + self._process_noise_covariance_per_s * dt)
            
            # Remove small numerical asymmetry from floating-point arithmetic.
            self._P = 0.5 * (self._P + self._P.T)
            
        # Upate previous encoder readings with current readings
        self._previous_encoders = encoders
        
        # Pass a pose tuple to StateEstimate
        pose_world = tuple(float(value) for value in self._pose)
        
        return StateEstimate(
            time=encoders.time,
            pose_world=pose_world,
            covariance=self._P
        )

    # Correct using camera
    def correct(
        self, 
        camera_pose: tuple[float, float, float], 
        camera_covariance: np.ndarray, 
        time: float
    )-> StateEstimate:
        """
        Use camera aruco marker detections to correct predicted pose when available
        """
        
        # Check there is already a prediction at current image timestamp
        if self._previous_encoders is None:
            raise ValueError("Prediction using encoders must complete first")
        
        # Check that the camera timestamp is close to the encoder timestamp
        if not np.isclose(time, self._previous_encoders.time, rtol=0.0, atol=1e-9):
            raise ValueError("Camera and encoder timestamps must be in sync to perform correction")
        
        # Only use new or unused camera measurements
        if self._last_camera_time is not None:
            if time <= self._last_camera_time:
                raise ValueError("Camera measurement has already been used or is older")
        
        # Camera derived robot pose and measurement covariance
        z = np.asarray(camera_pose, dtype=float)
        R = np.asarray(camera_covariance, dtype=float)
        
        # Check that the received pose and covariance are the correct shape
        if z.shape != (3,) or R.shape != (3, 3):
            raise ValueError("Expected a 3-element pose and a 3x3 pose covariance")
        
        # Compute pose innovation (measured - predicted pose)
        innovation = z - self._pose
        innovation[2] = wrap_angle(innovation[2]) # Normalise angle
        
        # Innovation covariance and Kalman Gain
        S = self._P + R
        K = np.linalg.solve(S.T, self._P.T).T
        
        # Correct the estimated pose
        # x_hat = x_hat_prev + K*(z - x_hat_prev) or x_hat = x_hat_prev + K*innovation
        self._pose = self._pose + K @ innovation # x_hat is new or corrected pose
        self._pose[2] = wrap_angle(self._pose[2])
        
        # Use Joseph form here to update covariance, let H = I
        # P = (I-K*H)*P_prev*(I-K*H).T + K*R*K.T
        I_minus_K = np.eye(3) - K
        self._P = (I_minus_K @ self._P @ I_minus_K.T + K @ R @ K.T)
        
        # Remove small floating-point asymmetry.
        self._P = 0.5 * (self._P + self._P.T)
        
        # Update previous camera timestamp with current one
        self._last_camera_time = time
        
        pose_world=tuple(float(value) for value in self._pose)
        
        return StateEstimate(
            time=time,
            pose_world=pose_world,
            covariance=self._P
        )
        