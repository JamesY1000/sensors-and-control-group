"""Sensor-only dock measurements and joint local robot/dock EKF checks."""

import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from track_a_mobile_robot.estimation import PoseEstimator, relative_dock_model
from track_a_mobile_robot.perception import (
    DockObservation, MarkerObservation, PerceptionOutput, marker_mounts,
    pose_to_transform, select_dock_observation,
)
from track_a_mobile_robot.sensing import CameraReading, LidarReading, WheelEncoderReading
from track_a_mobile_robot.state_model import DifferentialDriveModel, wrap_angle


class CameraCorrectionTests(unittest.TestCase):
    def estimator(self):
        return PoseEstimator(DifferentialDriveModel(), (.001, .001), np.eye(3) * .01)

    def marker(self, dock_pose, marker_id=0, height=0, tilt=0):
        base_dock = pose_to_transform(dock_pose)
        base_dock[2, 3] = height
        c, s = np.cos(tilt), np.sin(tilt)
        base_dock[:3, :3] = base_dock[:3, :3] @ np.array([[1,0,0],[0,c,-s],[0,s,c]])
        relative = base_dock @ marker_mounts()[marker_id]
        return MarkerObservation(0, marker_id, np.linalg.norm(relative[:2,3]), 0,
                                 relative, relative, .03, .2, 60, np.zeros((4,2)))

    def test_both_markers_measure_same_relative_dock_without_location_map(self):
        for marker_id in (0, 1):
            observation = select_dock_observation([self.marker((3,.2,math.pi), marker_id)])
            np.testing.assert_allclose(observation.pose_base, (3,.2,-math.pi), atol=1e-12)
            self.assertTrue(np.all(np.linalg.eigvalsh(observation.covariance) > 0))

    def test_relative_floor_and_tilt_checks_reject_bad_poses(self):
        for height, tilt in ((.61,0),(-.1,0),(0,math.radians(20))):
            self.assertIsNone(select_dock_observation([self.marker((3,0,math.pi), height=height, tilt=tilt)]))

    def test_exploration_encoders_are_ignored_and_acquisition_starts_zero(self):
        estimator = self.estimator()
        for time, left, right in ((0,10,10),(1,-20,30),(2,50,60)):
            estimate = estimator.predict(WheelEncoderReading(time,left,right))
            np.testing.assert_array_equal(estimate.pose_local, (0,0,0))
            np.testing.assert_array_equal(estimate.covariance, np.zeros((3,3)))
            self.assertIsNone(estimate.dock_pose_local)
            self.assertIsNone(estimator._previous_encoders)
        R = np.diag([.04,.04,.01])
        estimate = estimator.initialise_dock(WheelEncoderReading(2,50,60), (3,.2,math.pi), R)
        np.testing.assert_array_equal(estimate.pose_local, (0,0,0))
        np.testing.assert_allclose(estimate.dock_pose_local, (3,.2,-math.pi))
        np.testing.assert_array_equal(estimate.dock_covariance, R)
        np.testing.assert_array_equal(estimator._P[:3,3:], np.zeros((3,3)))
        # Only travel after acquisition belongs to the new odometry frame.
        estimate = estimator.predict(WheelEncoderReading(3,51,61))
        np.testing.assert_allclose(estimate.pose_local, (.033,0,0), atol=1e-12)

    def test_later_camera_observation_corrects_without_reinitialising(self):
        estimator = self.estimator()
        R = np.eye(3)*.01
        estimator.initialise_dock(WheelEncoderReading(0,100,100), (3,0,math.pi), R)
        estimator.predict(WheelEncoderReading(1,110,110))
        result = estimator.correct((2.67,0,math.pi), R, 1)
        np.testing.assert_allclose(result.pose_local, (.33,0,0), atol=1e-12)
        with self.assertRaises(ValueError):
            estimator.initialise_dock(WheelEncoderReading(1,110,110), (2.67,0,math.pi), R)

    def test_relative_measurement_jacobian_matches_finite_differences(self):
        state = np.array([.3,.4,.7,3,2,-1.])
        _, H = relative_dock_model(state)
        numerical = np.zeros((3,6))
        for i in range(6):
            delta=np.eye(6)[i]*1e-6
            numerical[:,i]=(relative_dock_model(state+delta)[0]-relative_dock_model(state-delta)[0])/(2e-6)
        np.testing.assert_allclose(H,numerical,atol=1e-8)

    def test_joint_correction_keeps_dock_uncertainty_and_cross_covariance(self):
        estimator=self.estimator()
        R=np.diag([.01,.01,.01])
        estimator.initialise_dock(WheelEncoderReading(0,0,0),(3,0,math.pi),R)
        estimator.predict(WheelEncoderReading(1,3,3))
        estimate=estimator.correct((2.88,.01,math.pi-.01),R,1)
        self.assertTrue(estimator.camera_correction_accepted)
        self.assertGreater(np.linalg.norm(estimator._P[:3,3:]),0)
        self.assertGreater(np.trace(estimate.dock_covariance),0)
        self.assertGreaterEqual(np.linalg.eigvalsh(estimator._P).min(),-1e-12)
        self.assertFalse(estimate.covariance.flags.writeable)

    def test_outlier_rejected_without_changing_joint_prediction(self):
        estimator=self.estimator();R=np.eye(3)*.01
        estimator.initialise_dock(WheelEncoderReading(0,0,0),(3,0,math.pi),R)
        estimator.predict(WheelEncoderReading(.02,0,0))
        before=estimator._state.copy(); covariance=estimator._P.copy()
        estimator.correct((-5,5,0),R,.02)
        self.assertFalse(estimator.camera_correction_accepted)
        np.testing.assert_array_equal(estimator._state,before)
        np.testing.assert_array_equal(estimator._P,covariance)
        with self.assertRaises(ValueError):
            estimator.correct((3,0,math.pi),R,.02)

    def test_heading_wrap_is_respected(self):
        estimator=self.estimator();R=np.eye(3)*.01
        estimator.initialise_dock(WheelEncoderReading(0,0,0),(3,0,math.pi-.01),R)
        estimator.predict(WheelEncoderReading(.02,0,0))
        estimator.correct((3,0,-math.pi+.01),R,.02)
        self.assertTrue(estimator.camera_correction_accepted)

    def test_rejected_measurement_is_not_passed_to_navigation(self):
        from track_a_mobile_robot import main
        estimator=self.estimator();R=np.eye(3)*.01
        estimator.initialise_dock(WheelEncoderReading(0,0,0),(3,0,math.pi),R)
        observation=DockObservation(.02,0,(-5,5,0),R)
        output=PerceptionOutput(.02,LidarReading(.02,(),(),.16,8),np.empty((0,2)),
                                CameraReading(.02,np.empty((0,0,3))),[],True,observation)
        perception=Mock();perception.update.return_value=output
        with patch.object(main,'read_noisy_wheel_encoders',return_value=WheelEncoderReading(.02,0,0)):
            _, filtered=main.update_demo(None,None,estimator,perception,None)
        self.assertIsNone(filtered.dock_observation)
        self.assertIsNotNone(output.dock_observation)


if __name__ == '__main__':
    unittest.main()
