"""MuJoCo sensor acquisition and wheel actuation behind the robot I/O interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..control import VelocityCommand, velocity_command_to_wheel_speeds
from ..sensing import CameraReading, LidarReading, WheelEncoderReading

if TYPE_CHECKING:
    import mujoco


# Specs from Robotis LDS-02
LIDAR_BEAM_COUNT = 360
LIDAR_RANGE_MIN_M = 0.16
LIDAR_RANGE_MAX_M = 8.0

# Specs from PiCam V2 (used by the actual turtlebot3 waffle pi)
CAMERA_NAME = "pi_camera_v2"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


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
    noise_scale: float = 1.,
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
    left_encoder_std = noise_scale * float(model.sensor_noise[left_encoder_id])
    right_encoder_std = noise_scale * float(model.sensor_noise[right_encoder_id])

    left_encoder_noisy = encoders_true.left_encoder_rad + float(rng.normal(0.0, left_encoder_std))
    right_encoder_noisy = encoders_true.right_encoder_rad + float(rng.normal(0.0, right_encoder_std))

    # Build a new frozen snapshot instead of modifying the ideal reading.
    return WheelEncoderReading(
        time=encoders_true.time,
        left_encoder_rad=left_encoder_noisy,
        right_encoder_rad=right_encoder_noisy,
    )


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


class RGBCamera:
    """Render the robot's ideal Pi Camera v2 at 30 Hz of simulation time.

    Create once per simulation and use as a context manager to release OpenGL.
    Call read after mj_forward on every physics step. Between frame deadlines it
    returns the same snapshot. Deadlines are rounded up to the next physics step;
    infrequent calls capture only the current frame, never invented past images.
    Rewinding simulation time clears the cached frame on the next read.
    """

    def __init__(self, model: mujoco.MjModel):
        import mujoco

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

    def __enter__(self) -> RGBCamera:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def lidar_transform_from_model(model: mujoco.MjModel) -> np.ndarray:
    """Read the fixed LiDAR-to-base transform configured before compilation."""
    # Use the sensor's fixed mount relative to the base, not its changing world pose.
    import mujoco

    mount = model.site("lidar_mount")
    if mount.bodyid[0] != model.body("base").id:
        raise ValueError("The LiDAR mounting site must be on the robot base")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, mount.quat)
    transform = np.eye(4)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = mount.pos
    return transform


class MujocoBackend:
    """Expose simulated sensors and differential-drive commands through the HAL.

    Physics advances only through step(); read() samples the current state.
    Only static sensor calibration is exposed to the autonomy pipeline.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, rng=None, encoder_noise_scale=1.):
        self.model, self.data = model, data
        if not np.isfinite(encoder_noise_scale) or encoder_noise_scale <= 0:
            raise ValueError('Encoder noise scale must be positive and finite')
        self.encoder_noise_scale = float(encoder_noise_scale)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.lidar_to_base = lidar_transform_from_model(model)
        self.encoder_noise_std_rad = tuple(
            float(model.sensor_noise[model.sensor(name).id])
            for name in ("wheel_left_encoder", "wheel_right_encoder")
        )
        self._wheel_ids = [model.actuator(name).id for name in ("wheel_left", "wheel_right")]
        self._wheel_commands = (0.0, 0.0)
        self._camera = RGBCamera(model)
        self.camera_matrix = self._camera.camera_matrix
        self.dist_coeffs = self._camera.dist_coeffs
        self.camera_to_base = self._camera.camera_to_base
        self.encoders = self.camera = self.lidar = None

    def now(self) -> float:
        return float(self.data.time)

    def start(self) -> None:
        import mujoco

        self.stop()
        mujoco.mj_forward(self.model, self.data)

    def read(self) -> WheelEncoderReading | None:
        now = self.now()
        reset = self.encoders is not None and now < self.encoders.time
        new_encoder = None
        # Sampling twice at the same timestamp must not invent fresh encoder noise.
        if self.encoders is None or now != self.encoders.time:
            new_encoder = self.encoders = read_noisy_wheel_encoders(self.model, self.data, self.rng, self.encoder_noise_scale)
        scan_time = float(self.data.sensor("lidar_time").data[0])
        if self.lidar is None or reset or scan_time != self.lidar.time:
            self.lidar = read_lidar(self.data)
        self.camera = self._camera.read(self.data)
        return new_encoder

    def health(self) -> str | None:
        for name in ("encoders", "camera", "lidar"):
            if getattr(self, name) is None:
                return f"Waiting for {name}"
        return None

    def send_command(self, command: VelocityCommand) -> None:
        self._wheel_commands = velocity_command_to_wheel_speeds(command)

    def step(self) -> None:
        """Hold the last command and advance physics by one timestep."""
        import mujoco

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self._wheel_ids] = self._wheel_commands
        mujoco.mj_step(self.model, self.data)

    def stop(self) -> None:
        self._wheel_commands = (0.0, 0.0)
        self.data.ctrl[:] = 0.0

    def close(self) -> None:
        self.stop()
        self._camera.close()
