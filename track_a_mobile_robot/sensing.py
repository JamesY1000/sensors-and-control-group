"""Read the project's simulated sensors without using the robot's world pose."""

# Read simulated sensors, apply configured measurement noise, and return timestamped readings.

from dataclasses import dataclass

import mujoco
import numpy as np


# Specs from Robotis LDS-02
LIDAR_BEAM_COUNT = 360
LIDAR_RANGE_MIN_M = 0.16
LIDAR_RANGE_MAX_M = 8.0

# Specs from PiCam V2 (used by the actual turtlebot3 waffle pi)
CAMERA_NAME = "pi_camera_v2"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


@dataclass(frozen=True)
class WheelEncoderReading:
    """Snapshot of wheel angles relative to reset, in radians.

    Angles accumulate through multiple revolutions; they do not wrap at 2π.
    Positive rotation on either wheel drives that side forward.
    Readings can be ideal or noisy, depending on the function used below.
    These are measured joint positions, not commanded motor speeds.
    """

    time: float
    left_encoder_rad: float
    right_encoder_rad: float


def read_absolute_wheel_encoders(data: mujoco.MjData) -> WheelEncoderReading:
    """Read ideal, accumulated joint angles, with simulation time in seconds.

    After a batch of mj_step calls, call mj_forward(model, data) first to
    refresh sensors at the final integrated state/time. The launcher does this.
    Copy to Python floats so subsequent simulation steps cannot change a reading.
    """

    return WheelEncoderReading(
        time=float(data.time),
        left_encoder_rad=float(data.sensor("wheel_left_encoder").data[0]),
        right_encoder_rad=float(data.sensor("wheel_right_encoder").data[0]),
    )


def read_noisy_wheel_encoders(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
) -> WheelEncoderReading:
    """Take a new sample with independent, zero-mean Gaussian angle errors.

    Each sensor's XML noise attribute supplies its standard deviation in radians.
    Create rng once outside the sampling loop; seed it for repeatable experiments.
    Call once per intended sensor sample, then share the returned reading with
    all consumers. Each call draws new noise, even at the same simulation time.
    Refresh sensors as described in read_absolute_wheel_encoders before sampling.
    MuJoCo's state and clean sensor values are left untouched.
    """
    encoders_true = read_absolute_wheel_encoders(data)

    left_encoder_id = model.sensor("wheel_left_encoder").id
    right_encoder_id = model.sensor("wheel_right_encoder").id
    left_encoder_std = float(model.sensor_noise[left_encoder_id])
    right_encoder_std = float(model.sensor_noise[right_encoder_id])

    left_encoder_noisy = encoders_true.left_encoder_rad + float(rng.normal(0.0, left_encoder_std))
    right_encoder_noisy = encoders_true.right_encoder_rad + float(rng.normal(0.0, right_encoder_std))

    # Build a new frozen snapshot instead of modifying the ideal reading.
    return WheelEncoderReading(
        time=encoders_true.time,
        left_encoder_rad=left_encoder_noisy,
        right_encoder_rad=right_encoder_noisy,
    )


@dataclass(frozen=True)
class LidarReading:
    """One ideal 2D scan, with angles counterclockwise from the LiDAR's +X axis.

    time is the scan acquisition time, which can be older than data.time.
    Positive infinity means no return within range; NaN means a hit too close
    to measure. Tuple fields keep the snapshot immutable after simulation steps.
    """

    time: float
    angles_rad: tuple[float, ...]
    ranges_m: tuple[float, ...]
    range_min_m: float
    range_max_m: float


def read_lidar(data: mujoco.MjData) -> LidarReading:
    """Copy the latest 5 Hz scan produced by the robot's XML rangefinders.

    Initialise with mj_forward(model, data) before the first read. As with the
    encoders, mj_forward after stepping refreshes sensors at the current state;
    the XML interval still holds LiDAR readings between scan times. This getter
    does not step physics, resample the LiDAR or change MuJoCo's raw readings.
    All beams are measured at one instant; spinning-scan distortion and LiDAR
    noise are not modelled yet.
    """
    raw_ranges = np.array([
        data.sensor(f"lidar_range{i:03d}").data[0]
        for i in range(LIDAR_BEAM_COUNT)
    ])
    ranges = raw_ranges.copy()
    ranges[(raw_ranges < 0) | (raw_ranges > LIDAR_RANGE_MAX_M)] = np.inf
    ranges[(raw_ranges >= 0) & (raw_ranges < LIDAR_RANGE_MIN_M)] = np.nan

    return LidarReading(
        time=float(data.sensor("lidar_time").data[0]),
        angles_rad=tuple(float(angle) for angle in np.deg2rad(np.arange(LIDAR_BEAM_COUNT))),
        ranges_m=tuple(float(distance) for distance in ranges),
        range_min_m=LIDAR_RANGE_MIN_M,
        range_max_m=LIDAR_RANGE_MAX_M,
    )


@dataclass(frozen=True)
class CameraReading:
    """A timestamped RGB image: uint8 pixels indexed [row, column, channel].

    time is the actual simulation time of capture, not the time of a later read.
    rgb is a read-only copy, so later rendering cannot overwrite this snapshot.
    """

    time: float
    rgb: np.ndarray


class RGBCamera:
    """Render the robot's ideal Pi Camera v2 at 30 Hz of simulation time.

    Create once per simulation and use as a context manager to release OpenGL.
    Call read after mj_forward on every physics step. Between frame deadlines it
    returns the same snapshot. Deadlines are rounded up to the next physics step;
    infrequent calls capture only the current frame, never invented past images.
    Rewinding simulation time clears the cached frame on the next read.
    """

    def __init__(self, model: mujoco.MjModel):
        camera = model.camera(CAMERA_NAME)
        # Exact intrinsics for this ideal pinhole renderer, in OpenCV pixel
        # coordinates (integer pixel centres, origin at the top-left pixel).
        focal_px = CAMERA_HEIGHT / (2 * np.tan(np.deg2rad(camera.fovy[0]) / 2))
        self.camera_matrix = np.array([
            [focal_px, 0.0, (CAMERA_WIDTH - 1) / 2],
            [0.0, focal_px, (CAMERA_HEIGHT - 1) / 2],
            [0.0, 0.0, 1.0],
        ])
        self.dist_coeffs = np.zeros(5)  # No lens distortion is simulated.

        # Static mounting from the YAML-configured model, not the robot's live pose.
        # OpenCV optical axes: +X right, +Y down, +Z forward.
        if camera.bodyid[0] != model.body("base").id:
            raise ValueError("The RGB camera must be mounted directly on the base body.")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, camera.quat)
        self.camera_to_base = np.eye(4)
        self.camera_to_base[:3, :3] = rotation.reshape(3, 3) @ np.diag([1, -1, -1])
        self.camera_to_base[:3, 3] = camera.pos
        for calibration in (self.camera_matrix, self.dist_coeffs, self.camera_to_base):
            calibration.setflags(write=False)

        self._scene_option = mujoco.MjvOption()
        self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self._scene_option.sitegroup[:] = 0
        self._renderer = mujoco.Renderer(model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
        self._reading: CameraReading | None = None
        self._frame_slot = -1
        self._last_read_time = -1.0
        self._closed = False

    def read(self, data: mujoco.MjData) -> CameraReading:
        """Return the latest frame, capturing only when a new frame is due."""
        if self._closed:
            raise RuntimeError("Cannot read a closed RGB camera.")
        now = float(data.time)
        frame_slot = int(np.floor(now * CAMERA_FPS + 1e-9))
        rewound = now < self._last_read_time
        self._last_read_time = now
        if self._reading is None or rewound or frame_slot > self._frame_slot:
            self._renderer.update_scene(data, camera=CAMERA_NAME, scene_option=self._scene_option)
            rgb = self._renderer.render().copy()
            rgb.setflags(write=False)
            self._reading = CameraReading(time=now, rgb=rgb)
            self._frame_slot = frame_slot
        return self._reading

    def close(self) -> None:
        """Release the rendering context after the last image has been captured."""
        if not self._closed:
            self._renderer.close()
            self._closed = True

    def __enter__(self) -> "RGBCamera":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
