"""Joint robot/dock UKF, with the same interface and noise model as the EKF.

Sigma-point filtering follows Julier and Uhlmann (2004),
https://doi.org/10.1109/JPROC.2003.823141. Parameters: alpha=1, beta=2,
kappa=0. The non-negative mean weights avoid cancellation from very small alpha.
Wheel-increment uncertainty is augmented during prediction; process drift and
camera covariance are additive. Angle means/residuals respect the circle.
"""
import numpy as np

from .estimation import CAMERA_INNOVATION_LIMIT, PoseEstimator
from .state_model import wrap_angle


def symmetric_psd(matrix):
    matrix = .5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    if values.min() < -1e-9 * max(1., float(np.max(np.abs(values)))):
        raise ValueError('UKF covariance lost positive semidefiniteness')
    if values.min() < 0:
        matrix = (vectors * np.maximum(values, 0)) @ vectors.T
    return matrix


def sigma_points(mean, covariance):
    """Scaled unscented transform, including semidefinite anchored initial states."""
    n = len(mean)
    values, vectors = np.linalg.eigh(symmetric_psd(covariance))
    root = vectors * np.sqrt(n * np.maximum(values, 0))
    points = np.vstack((mean, mean + root.T, mean - root.T))
    wm = np.full(2*n+1, 1/(2*n));wm[0] = 0.
    wc = wm.copy();wc[0] = 2.
    return points, wm, wc


def moments(points, wm, wc, angles):
    mean = wm @ points
    for index in angles:
        mean[index] = np.arctan2(wm @ np.sin(points[:, index]), wm @ np.cos(points[:, index]))
    deviation = points - mean
    deviation[:, list(angles)] = wrap_angle(deviation[:, list(angles)])
    return mean, (deviation.T * wc) @ deviation, deviation


class UnscentedPoseEstimator(PoseEstimator):
    """Uses the existing local-frame initialisation and measurement validation."""

    def predict(self, encoders):
        self.camera_correction_accepted = False
        if not self.initialised:
            return self._estimate(encoders.time)
        previous = self._previous_encoders
        dt = encoders.time - previous.time
        if dt <= 0:
            raise ValueError('Encoder timestamps must increase')
        increments = np.array([encoders.left_encoder_rad-previous.left_encoder_rad,
                               encoders.right_encoder_rad-previous.right_encoder_rad])
        covariance = np.zeros((8,8));covariance[:6,:6] = self._P
        covariance[6:,6:] = np.diag(2*self._encoder_noise_std_rad**2)
        points, wm, wc = sigma_points(np.r_[self._state, 0., 0.], covariance)
        wheels = increments + points[:,6:]
        radius, separation = self._model.wheel_radius_m, self._model.wheel_separation_m
        distance = .5*radius*(wheels[:,0]+wheels[:,1])
        turn = radius*(wheels[:,1]-wheels[:,0])/separation
        propagated = points[:,:6].copy()
        propagated[:,0] += distance*np.cos(points[:,2]+.5*turn)
        propagated[:,1] += distance*np.sin(points[:,2]+.5*turn)
        propagated[:,2] = wrap_angle(points[:,2]+turn)
        self._state, self._P, _ = moments(propagated, wm, wc, (2,5))
        self._P[:3,:3] += self._process_noise_covariance_per_s*dt
        self._P = symmetric_psd(self._P)
        self._previous_encoders = encoders
        return self._estimate(encoders.time)

    def correct(self, dock_pose_base, camera_covariance, time):
        if not self.initialised or self._previous_encoders is None:
            raise ValueError('Initialise the dock before correction')
        if not np.isclose(time, self._previous_encoders.time, rtol=0, atol=1e-9):
            raise ValueError('Camera and encoder timestamps must be in sync')
        if time <= self._last_camera_time:
            raise ValueError('Camera measurement has already been used or is older')
        z, R = self._measurement(dock_pose_base, camera_covariance)
        points, wm, wc = sigma_points(self._state, self._P)
        delta = points[:,3:5] - points[:,:2]
        c, s = np.cos(points[:,2]), np.sin(points[:,2])
        observations = np.column_stack((c*delta[:,0]+s*delta[:,1],
            -s*delta[:,0]+c*delta[:,1], wrap_angle(points[:,5]-points[:,2])))
        prediction, measurement_cov, dz = moments(observations, wm, wc, (2,))
        S = measurement_cov + R
        innovation = z-prediction;innovation[2] = wrap_angle(innovation[2])
        self._last_camera_time = time
        self.camera_correction_accepted = False
        if self._gate_camera and float(innovation @ np.linalg.solve(S, innovation)) > CAMERA_INNOVATION_LIMIT:
            return self._estimate(time)
        dx = points-self._state;dx[:,[2,5]] = wrap_angle(dx[:,[2,5]])
        cross = (dx.T*wc) @ dz
        gain = np.linalg.solve(S, cross.T).T
        self._state += gain @ innovation
        self._state[[2,5]] = wrap_angle(self._state[[2,5]])
        self._P = symmetric_psd(self._P-gain @ S @ gain.T)
        self.camera_correction_accepted = True
        return self._estimate(time)
