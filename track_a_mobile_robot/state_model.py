"""
Our state model should essentially describe how the robot's pose changes given its wheel movements, using the differential-drive motion model.

It stores:
- Wheel radius (m)
- Wheel separation (m)

Currently its inputs are:
- Previous estimated robot pose (x_hat = [x_hat, y_hat, theta_hat].T)
- Left and right wheel angle increments (rad), computed by the estimator from successive encoder readings

It has 3 main calculations:
- Wheel displacement:
  - Converts each wheel's angle increment into distance travelled
  - Computes the robot's forward travel and heading change
- Pose prediction:
  - Uses the previous estimated pose and wheel displacement to predict the next robot pose
  - Uses the midpoint heading to calculate the change in x and y
  - Normalises the resulting heading to [-pi, pi)
- Motion Jacobians:
  - Computes F, which describes how uncertainty in the previous robot pose affects the predicted pose
  - Computes G, which describes how uncertainty in the wheel angle increments affects the predicted pose
  - These matrices are used by the estimator to propagate the pose covariance
  
It outputs through its methods:
- Forward travel and heading change
- Predicted robot pose as a NumPy array [x, y, theta]
- Jacobian matrices F (3 x 3) and G (3 x 2)

The state model performs these calculations when called. The estimator maintains the current estimated pose and covariance.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DifferentialDriveModel:
    """
    Stores the motion geometry and deterministic motion calculations
    Variables derived from turtlebot URDF
    """

    wheel_radius_m: float = 0.033
    wheel_separation_m: float = 0.288
    
    def wheel_displacement(
        self,
        delta_left_rad: float,
        delta_right_rad: float,
    ) -> tuple[float, float]:
        """
        Compute turtlebot forward travel (m) and heading change (rad) from wheel rotations
        """
        
        # Convert wheel rotations into distance travelled by each wheel and the turtlebot
        left_distance = self.wheel_radius_m * delta_left_rad
        right_distance = self.wheel_radius_m * delta_right_rad
        distance = (left_distance + right_distance) / 2.0
        
        # Compute heading change
        heading_change = (right_distance - left_distance) / self.wheel_separation_m
        
        return distance, heading_change
    
    def predict_pose(
        self,
        pose: np.ndarray,
        delta_left_rad: float,
        delta_right_rad: float,
    ) -> np.ndarray:
        """Predict the next [x, y, heading] using midpoint integration"""
        
        x, y, heading = pose
        
        # Get the turtlebot displacement and heading change (detla_s, delta_theta)
        distance, heading_change = self.wheel_displacement(delta_left_rad, delta_right_rad)
        
        # Approximate the travel direction halfway through the turn
        # theta_mid = theta + (delta_theta / 2)
        midpoint_heading = heading + heading_change / 2.0
        
        # Compute the next pose
        next_x = x + distance * np.cos(midpoint_heading) # x_next = x + delta_s * cos(theta_mid)
        next_y = y + distance * np.sin(midpoint_heading) # y_next = y + delta_s * sin(theta_mid)
        next_heading = wrap_angle(heading + heading_change) # theta_next = theta + delta_theta
        
        next_pose = np.array([next_x, next_y, next_heading], dtype=float)
        
        return next_pose
    
    def motion_jacobians(
        self,
        pose: np.ndarray,
        delta_left_rad: float,
        delta_right_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the jacobians for pose F and wheel increments G
        """
        
        x, y, heading = pose
        
        # Get the turtlebot displacement and heading change (detla_s, delta_theta)
        distance, heading_change = self.wheel_displacement(delta_left_rad, delta_right_rad)
        
        midpoint_heading = heading + heading_change / 2.0
        
        # Compute F and G
        half_radius = self.wheel_radius_m / 2.0
        heading_per_wheel = (self.wheel_radius_m / self.wheel_separation_m)
        turn_term = distance * heading_per_wheel / 2.0
        
        # F - sensitivty from errors in previous pose to affect next pose
        # Columns: previous x, previous y, previous heading
        F = np.array([
            [1.0, 0.0, -distance * np.sin(midpoint_heading)],
            [0.0, 1.0,  distance * np.cos(midpoint_heading)],
            [0.0, 0.0,  1.0],
        ])
        
        # G - sensitivity from errors in wheel movement affect the next pose
        # Columns: left wheel increment, right wheel increment
        G = np.array([
            [
                half_radius * np.cos(midpoint_heading) + turn_term * np.sin(midpoint_heading),
                half_radius * np.cos(midpoint_heading) - turn_term * np.sin(midpoint_heading),
            ],
            [
                half_radius * np.sin(midpoint_heading) - turn_term * np.cos(midpoint_heading),
                half_radius * np.sin(midpoint_heading) + turn_term * np.cos(midpoint_heading),
            ],
            [
                -heading_per_wheel,
                heading_per_wheel,
            ],
        ])
        
        
        return F, G
        

        
        




def wrap_angle(angle_rad: float) -> float:
    """Helper function to normalise angle to [-pi, pi)"""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi
