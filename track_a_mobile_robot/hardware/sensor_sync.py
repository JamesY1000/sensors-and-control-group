"""Fuse delayed camera poses at capture time using a bounded encoder history.

The committed EKF stays behind the live encoder stream. Camera corrections are
applied there, then a copy is predicted to the latest encoder reading for control.
As history expires, its encoder motion is committed before it is discarded.
This avoids relabelling old images as current measurements or changing EKF math.
"""

import math
from collections import deque
from copy import deepcopy

from ..sensing import WheelEncoderReading


class SensorFusion:
    def __init__(self, estimator, history_s=1.0, max_encoder_gap_s=.2):
        if not 0 < max_encoder_gap_s <= history_s or not math.isfinite(history_s):
            raise ValueError('Expected 0 < max_encoder_gap_s <= history_s')
        self.estimator = estimator
        self.history_s = history_s
        self.max_encoder_gap_s = max_encoder_gap_s
        self.encoders = deque()
        self.pending = deque(maxlen=32)
        self._committed_time = None
        self._last_observation_time = -math.inf
        self.dropped_observations = 0

    def _at(self, time):
        for i, right in enumerate(self.encoders):
            if right.time == time:
                return right
            if right.time > time:
                if i == 0:
                    return None
                left = self.encoders[i-1]
                if right.time - left.time > self.max_encoder_gap_s:
                    return None
                fraction = (time-left.time)/(right.time-left.time)
                return WheelEncoderReading(
                    time,
                    left.left_encoder_rad + fraction*(right.left_encoder_rad-left.left_encoder_rad),
                    left.right_encoder_rad + fraction*(right.right_encoder_rad-left.right_encoder_rad),
                )
        return None

    def _predict_to(self, estimator, start, end):
        for reading in self.encoders:
            if start < reading.time <= end.time:
                estimator.predict(reading)
                start = reading.time
        if start < end.time:
            estimator.predict(end)

    def update(self, encoders=None, observation=None):
        """Return the latest encoder-time estimate and last accepted new observation.

        Inputs are new samples only. Future camera observations wait for a right
        encoder bracket. Late, duplicate and unbracketed observations are dropped.
        Out-of-order encoder data is an error; a clock/encoder reset needs a new run.
        """
        if encoders is not None:
            if not all(math.isfinite(value) for value in (
                encoders.time, encoders.left_encoder_rad, encoders.right_encoder_rad,
            )):
                raise ValueError('Encoder readings must be finite')
            if self.encoders and encoders.time <= self.encoders[-1].time:
                raise ValueError('Encoder timestamps must increase')
            self.encoders.append(encoders)
            cutoff = encoders.time-self.history_s
            while len(self.encoders) > 2 and (
                self.encoders[1].time < cutoff or len(self.encoders) > 2048
            ):
                boundary = self.encoders[1]
                if self.estimator.initialised and boundary.time > self._committed_time:
                    self._predict_to(self.estimator, self._committed_time, boundary)
                    self._committed_time = boundary.time
                self.encoders.popleft()

        if observation is not None:
            if not math.isfinite(observation.time):
                raise ValueError('Camera timestamp must be finite')
            if observation.time <= self._last_observation_time:
                self.dropped_observations += 1
            else:
                if len(self.pending) == self.pending.maxlen:
                    self.dropped_observations += 1
                self.pending.append(observation)
                self._last_observation_time = observation.time

        if not self.encoders:
            return None, None
        latest = self.encoders[-1]
        accepted = None
        while self.pending and self.pending[0].time <= latest.time:
            item = self.pending.popleft()
            aligned = self._at(item.time)
            if aligned is None or (
                self._committed_time is not None and item.time <= self._committed_time
            ):
                self.dropped_observations += 1
                continue
            if not self.estimator.initialised:
                self.estimator.initialise_dock(aligned, item.pose_base, item.covariance)
            else:
                self._predict_to(self.estimator, self._committed_time, aligned)
                self.estimator.correct(item.pose_base, item.covariance, item.time)
            self._committed_time = item.time
            if self.estimator.camera_correction_accepted:
                accepted = item

        # Preview encoder motion without committing beyond a delayed image's time.
        current = deepcopy(self.estimator)
        if current.initialised:
            self._predict_to(current, self._committed_time, latest)
            estimate = current._estimate(latest.time)
        else:
            estimate = current.predict(latest)
        return estimate, accepted
