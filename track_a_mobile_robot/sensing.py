"""Timestamped sensor readings shared by both HAL backends and autonomy modules."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WheelEncoderReading:
    """Snapshot of wheel angles relative to reset, in radians.

    Angles accumulate through multiple revolutions; they do not wrap at 2π.
    Positive rotation on either wheel drives that side forward.
    Readings can come from simulation or hardware.
    These are measured joint positions, not commanded motor speeds.
    """

    time: float
    left_encoder_rad: float
    right_encoder_rad: float


@dataclass(frozen=True)
class LidarReading:
    """One 2D scan, with angles counterclockwise from the LiDAR's +X axis.

    time is the scan acquisition time, on the shared run clock.
    Positive infinity means no return within range; NaN requires a stop
    (too close or unknown). Tuple fields keep the snapshot immutable.
    """

    time: float
    angles_rad: tuple[float, ...]
    ranges_m: tuple[float, ...]
    range_min_m: float
    range_max_m: float


@dataclass(frozen=True)
class CameraReading:
    """A timestamped RGB image: uint8 pixels indexed [row, column, channel].

    time is the acquisition time on the shared run clock, not the time of a later read.
    rgb is a read-only copy, so later rendering cannot overwrite this snapshot.
    """

    time: float
    rgb: np.ndarray
