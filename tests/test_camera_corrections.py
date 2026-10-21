"""Check physical camera-pose validation and EKF outlier rejection."""

import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from track_a_mobile_robot.estimation import PoseEstimator
from track_a_mobile_robot.perception import (
    CameraPoseObservation, MarkerObservation, PerceptionOutput, invert_transform,
    make_transform, pose_to_transform, select_pose_observation,
)
from track_a_mobile_robot.sensing import CameraReading, LidarReading, WheelEncoderReading
from track_a_mobile_robot.state_model import DifferentialDriveModel


class CameraCorrectionTests(unittest.TestCase):
    def estimator(self, heading=0, variance=.01):
        estimator = PoseEstimator(DifferentialDriveModel(), (0, 0, heading),
                                  np.eye(3) * variance, (0, 0), np.zeros((3, 3)))
        estimator.predict(WheelEncoderReading(0, 0, 0))
        return estimator

    def observation(self, base):
        marker_map = {0: pose_to_transform((4, 0, 0))}
        relative = invert_transform(base) @ marker_map[0]
        marker = MarkerObservation(0, 0, 4, 0, relative, relative, .03, .2, 26,
                                   np.zeros((4, 2)))
        return select_pose_observation([marker], marker_map)

    def test_floor_pose_is_accepted_and_preserved(self):
        pose = (1, -.5, .3)
        observation = self.observation(pose_to_transform(pose))
        self.assertIsNotNone(observation)
        np.testing.assert_allclose(observation.pose_world, pose)
        self.assertTrue(np.all(np.linalg.eigvalsh(observation.covariance) > 0))

    def test_low_reprojection_error_does_not_override_impossible_height_or_tilt(self):
        for height in (.61, .096, -.1):
            base = pose_to_transform((1, 0, 0))
            base[2, 3] = height
            self.assertIsNone(self.observation(base))
        angle = math.radians(20)
        base = make_transform([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)],
                               [0, math.sin(angle), math.cos(angle)]], [1, 0, 0])
        self.assertIsNone(self.observation(base))

    def test_outlier_leaves_pose_and_covariance_unchanged(self):
        estimator = self.estimator()
        result = estimator.correct((2, 2, 1), np.eye(3) * .01, 0)
        self.assertFalse(estimator.camera_correction_accepted)
        np.testing.assert_array_equal(result.pose_world, (0, 0, 0))
        np.testing.assert_array_equal(result.covariance, np.eye(3) * .01)
        with self.assertRaises(ValueError):
            estimator.correct((2, 2, 1), np.eye(3) * .01, 0)
        estimator.predict(WheelEncoderReading(.02, 0, 0))
        result = estimator.correct((.1, 0, 0), np.eye(3) * .01, .02)
        self.assertTrue(estimator.camera_correction_accepted)
        self.assertAlmostEqual(result.pose_world[0], .05)

    def test_gate_wraps_heading_and_respects_prediction_uncertainty(self):
        estimator = self.estimator(math.pi - .01)
        estimator.correct((0, 0, -math.pi + .01), np.eye(3) * .01, 0)
        self.assertTrue(estimator.camera_correction_accepted)
        # A metre discrepancy can be plausible with large prediction uncertainty.
        estimator = self.estimator(variance=1)
        estimator.correct((1, 0, 0), np.eye(3) * .01, 0)
        self.assertTrue(estimator.camera_correction_accepted)

    def test_rejected_measurement_is_not_passed_to_navigation(self):
        from track_a_mobile_robot import main
        estimator = self.estimator()
        observation = CameraPoseObservation(.02, 0, (2, 2, 1), np.eye(3) * .01)
        output = PerceptionOutput(.02, LidarReading(.02, (), (), .16, 8),
                                  np.empty((0, 2)), CameraReading(.02, np.empty((0, 0, 3))),
                                  [], True, observation)
        perception = Mock()
        perception.update.return_value = output
        with patch.object(main, 'read_noisy_wheel_encoders',
                          return_value=WheelEncoderReading(.02, 0, 0)):
            estimate, filtered = main.update_demo(None, None, estimator, perception, None)
        self.assertIsNone(filtered.pose_observation)
        self.assertTrue(filtered.new_camera_frame)
        self.assertIsNotNone(output.pose_observation)
        np.testing.assert_array_equal(estimate.pose_world, (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
