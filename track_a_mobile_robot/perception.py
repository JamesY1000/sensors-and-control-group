"""
Our perception module should essentially convert camera and LiDAR readings into
measurements that estimation and navigation can use.

Currently its inputs are:
- RGB camera images and LiDAR range readings (from sensing)
- Camera intrinsics, sensor mounting transforms, marker sizes and marker mounts

It has 3 main stages:
1. Marker detection:
   - Detect known ArUco IDs and their image corners
   - Use camera calibration and marker size to solve each marker's 3D pose
2. Dock measurement:
   - Apply the camera and marker mounting transforms to obtain the dock pose
     in the current robot frame
   - Reject ambiguous poses and solutions inconsistent with an upright dock
   - Extract [x, y, heading] and assign measurement covariance for the EKF
3. LiDAR processing:
   - Convert valid ranges into obstacle points in the current robot frame
   - Supply these points for checking clearance during approach and docking

It outputs:
- A PerceptionOutput containing the latest readings and robot-frame LiDAR points
- A fresh DockObservation (time, marker ID, pose_base, covariance), when usable

Sensor noise (for robustness testing, see SensorNoise):
- Presets: SensorNoise.ideal(), .nominal() (the original demo noise, used by
  Perception(noise=True)), .realistic() and .harsh()
- LiDAR: range noise growing with distance, range quantisation, random and
  grazing-angle dropouts
- Camera: lens distortion, motion blur from turning, brightness changes,
  pixel noise and frame latency
- Calibration: a deliberate camera mounting error in the assumed extrinsic
Noise is applied once per sensor sample. It simulates the sensor; its inputs
(rendered image, wheel joint speeds) are never passed to estimation or control.

Perception runs during exploration as well as docking; the first usable marker
initialises the local robot/dock estimate. A detected ID alone is not a usable dock pose.
"""

from collections import deque
from dataclasses import dataclass, fields

import cv2
import mujoco
import numpy as np

from .sensing import CAMERA_HEIGHT, CAMERA_WIDTH, CameraReading, LidarReading, RGBCamera, read_lidar
from .state_model import DifferentialDriveModel, wrap_angle


ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_SIZES_M = {0: 0.40, 1: 0.04} # Black square sizes, excluding the white borders

MIN_MARKER_SIDE_PX = 20.0
MAX_REPROJECTION_ERROR_PX = 2.0
AMBIGUITY_WARNING = 0.5

# Robot and dock origins lie on the same floor, with their Z axes upright.
# Reject 3D marker solutions inconsistent with that model before discarding Z/tilt.
MAX_DOCK_HEIGHT_ERROR_M = 0.05
MAX_DOCK_TILT_RAD = np.deg2rad(10.0)

# Initial noise assumptions for the demo; tune these using measured errors.
LIDAR_RANGE_NOISE_STD_M = 0.01
LIDAR_DROPOUT_PROBABILITY = 0.0
CAMERA_PIXEL_NOISE_STD = 2.0 # Image intensity noise on the 0–255 scale
CAMERA_CORNER_STD_PX = 1.0 # Assumed corner-position uncertainty
CAMERA_TRANSLATION_STD_M = 0.02
CAMERA_HEADING_STD_RAD = np.deg2rad(5.0)


@dataclass(frozen=True)
class SensorNoise:
    """Simulated sensor imperfections, applied once per LiDAR scan or camera frame.

    Every value here is an ASSUMPTION for robustness testing. Replace values
    with datasheet figures or your own measurements and cite them; the presets
    below are a graded set of conditions, not claims about the real sensors.

    LiDAR (the LDS-02 reports one range per degree):
        lidar_range_std_m       range noise sd at zero range
        lidar_range_std_per_m   extra sd per metre of range (noise grows with distance)
        lidar_resolution_m      range quantisation step (0 = continuous)
        lidar_dropout_probability   chance any valid beam returns nothing
        lidar_grazing_dropout   extra dropout chance for a beam hitting a surface
                                edge-on (scaled by how oblique the hit is)
    Camera (Pi Camera v2):
        camera_pixel_std        Gaussian intensity noise, 0-255 scale
        camera_gain_std         per-frame brightness gain sd (lighting/exposure changes)
        camera_distortion       OpenCV (k1, k2, p1, p2, k3) applied to the image.
                                The detector does NOT know these unless calibrated
                                values are passed to Perception(dist_coeffs=...).
        camera_exposure_s       exposure time; turning smears the image horizontally
                                by |omega| * exposure * focal length pixels
        camera_latency_s        delay before a frame reaches perception. Frames are
                                stamped on arrival, as a driver unaware of the delay would.
    Calibration:
        camera_mount_error_m    error in the ASSUMED camera position (base frame)
        camera_mount_yaw_error_deg   error in the ASSUMED camera yaw
    """

    lidar_range_std_m: float = 0.0
    lidar_range_std_per_m: float = 0.0
    lidar_resolution_m: float = 0.0
    lidar_dropout_probability: float = 0.0
    lidar_grazing_dropout: float = 0.0
    camera_pixel_std: float = 0.0
    camera_gain_std: float = 0.0
    camera_distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    camera_exposure_s: float = 0.0
    camera_latency_s: float = 0.0
    camera_mount_error_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_mount_yaw_error_deg: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            values = np.atleast_1d(np.asarray(getattr(self, item.name), dtype=float))
            if not np.isfinite(values).all():
                raise ValueError(f"{item.name} must be finite")
            if item.name not in ("camera_distortion", "camera_mount_error_m",
                                 "camera_mount_yaw_error_deg") and (values < 0).any():
                raise ValueError(f"{item.name} cannot be negative")
        if len(self.camera_distortion) != 5 or len(self.camera_mount_error_m) != 3:
            raise ValueError("camera_distortion needs 5 values and camera_mount_error_m needs 3")
        if self.lidar_dropout_probability + self.lidar_grazing_dropout > 1.0:
            raise ValueError("LiDAR dropout probabilities cannot add up to more than 1")

    @classmethod
    def ideal(cls) -> "SensorNoise":
        """Perfect sensors: a baseline that isolates the algorithms' own errors."""
        return cls()

    @classmethod
    def nominal(cls) -> "SensorNoise":
        """The original demo noise, unchanged. Perception(noise=True) uses this."""
        return cls(lidar_range_std_m=LIDAR_RANGE_NOISE_STD_M,
                   lidar_dropout_probability=LIDAR_DROPOUT_PROBABILITY,
                   camera_pixel_std=CAMERA_PIXEL_NOISE_STD)

    @classmethod
    def realistic(cls) -> "SensorNoise":
        """Plausible imperfections of a real TurtleBot3. Placeholder magnitudes:
        source them (ROBOTIS LDS-02 manual, your own Pi camera calibration)."""
        return cls(lidar_range_std_m=0.01, lidar_range_std_per_m=0.01, lidar_resolution_m=0.001,
                   lidar_dropout_probability=0.01, lidar_grazing_dropout=0.3,
                   camera_pixel_std=3.0, camera_gain_std=0.05,
                   camera_distortion=(0.12, -0.25, 0.001, -0.001, 0.0),
                   camera_exposure_s=1 / 60, camera_latency_s=0.033)

    @classmethod
    def harsh(cls) -> "SensorNoise":
        """Deliberately poor conditions, to find where the pipeline breaks."""
        return cls(lidar_range_std_m=0.02, lidar_range_std_per_m=0.02, lidar_resolution_m=0.005,
                   lidar_dropout_probability=0.05, lidar_grazing_dropout=0.6,
                   camera_pixel_std=6.0, camera_gain_std=0.15,
                   camera_distortion=(0.18, -0.38, 0.002, -0.002, 0.0),
                   camera_exposure_s=1 / 30, camera_latency_s=0.1,
                   camera_mount_error_m=(0.02, 0.0, 0.0), camera_mount_yaw_error_deg=2.0)

    @classmethod
    def preset(cls, name: str) -> "SensorNoise":
        """Look up a preset by name: ideal, nominal, realistic or harsh."""
        presets = {"ideal": cls.ideal, "nominal": cls.nominal,
                   "realistic": cls.realistic, "harsh": cls.harsh}
        if name not in presets:
            raise ValueError(f"Unknown noise preset {name!r}; choose from {sorted(presets)}")
        return presets[name]()

    @classmethod
    def from_dict(cls, values: dict) -> "SensorNoise":
        """Build from a config mapping, e.g. a `noise:` section of a YAML file.
        An optional `preset` key gives the starting values; other keys override them."""
        values = dict(values)
        base = cls.preset(values.pop("preset")) if "preset" in values else cls()
        known = {item.name for item in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown noise settings: {sorted(unknown)}")
        merged = {item.name: getattr(base, item.name) for item in fields(cls)}
        for name, value in values.items():
            merged[name] = tuple(value) if isinstance(value, (list, tuple)) else float(value)
        return cls(**merged)


@dataclass(frozen=True)
class MarkerObservation:
    """Store a marker's measured pose and detection quality.

    Camera axes: x right, y down, z forward.
    Robot base axes: x forward, y left, z up.
    Marker axes: x right, y up, z out of the printed face.
    Range and bearing describe the marker in the robot's horizontal plane.
    """

    time: float
    marker_id: int
    range_m: float
    bearing_rad: float
    T_camera_marker: np.ndarray
    T_base_marker: np.ndarray
    reprojection_error_px: float
    ambiguity_ratio: float
    side_px: float
    corners_px: np.ndarray

    @property
    def orientation_reliable(self) -> bool:
        # Similar errors for the two planar solutions mean uncertain orientation.
        return self.ambiguity_ratio < AMBIGUITY_WARNING


@dataclass(frozen=True)
class DockObservation:
    """Store the measured dock pose in the robot frame and its 3x3 covariance."""

    time: float
    marker_id: int
    pose_base: tuple[float, float, float]
    covariance: np.ndarray

    def __post_init__(self) -> None:
        # Give the result its own protected covariance, as in StateEstimate.
        covariance = np.array(self.covariance, dtype=float, copy=True)
        covariance.setflags(write=False)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class PerceptionOutput:
    """Store the latest sensor readings and any new camera pose."""

    time: float
    lidar: LidarReading
    lidar_points_base: np.ndarray
    camera: CameraReading
    markers: list[MarkerObservation] # New image only; may include ambiguous poses
    new_camera_frame: bool
    dock_observation: DockObservation | None = None

    @property
    def lidar_too_close(self) -> bool:
        # NaN is a below-minimum-range hit, not clear space.
        return bool(np.isnan(self.lidar.ranges_m).any())


def make_transform(rotation, translation) -> np.ndarray:
    """Build a 4x4 transform from a rotation and translation."""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid transform using R.T and -R.T @ t."""
    rotation = transform[:3, :3].T
    return make_transform(rotation, -rotation @ transform[:3, 3])


def pose_to_transform(pose) -> np.ndarray:
    """Convert a planar [x, y, heading] into a homogeneous transform."""
    x, y, heading = pose
    c, s = np.cos(heading), np.sin(heading)
    return make_transform([[c, -s, 0], [s, c, 0], [0, 0, 1]], [x, y, 0])


def transform_to_pose(transform: np.ndarray) -> np.ndarray:
    """Extract planar [x, y, heading] from a 4x4 transform."""
    heading = np.arctan2(transform[1, 0], transform[0, 0])
    return np.array([transform[0, 3], transform[1, 3], wrap_angle(heading)])


def lidar_transform_from_model(model: mujoco.MjModel) -> np.ndarray:
    """Read the fixed LiDAR-to-base transform configured before compilation."""
    # Use the sensor's fixed mount relative to the base, not its changing world pose.
    mount = model.site("lidar_mount")
    if mount.bodyid[0] != model.body("base").id:
        raise ValueError("The LiDAR mounting site must be on the robot base")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, mount.quat)
    return make_transform(rotation, mount.pos)


def lidar_points_in_base(reading: LidarReading, lidar_to_base: np.ndarray) -> np.ndarray:
    """Convert valid ranges into (N, 2) obstacle points in the robot frame."""
    ranges = np.asarray(reading.ranges_m)
    angles = np.asarray(reading.angles_rad)
    valid = np.isfinite(ranges)

    # Polar to Cartesian in the LiDAR frame, then apply its mounting transform.
    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])
    points_lidar = np.column_stack([x, y, np.zeros(valid.sum())])
    points_base = points_lidar @ lidar_to_base[:3, :3].T + lidar_to_base[:3, 3]
    return points_base[:, :2]


def lidar_points_to_local(points_base: np.ndarray, pose) -> np.ndarray:
    """Transform obstacle points using the estimated robot pose at scan time."""
    # Navigation uses obstacles and targets in the same fixed local frame.
    x, y, heading = pose
    c, s = np.cos(heading), np.sin(heading)
    return np.asarray(points_base) @ np.array([[c, s], [-s, c]]) + [x, y]


def grazing_factor(reading: LidarReading, max_neighbour_gap_m: float = 0.3) -> np.ndarray:
    """How edge-on each beam meets the surface: 0 head-on, 1 grazing.

    Estimated from the scan itself: the surface direction at a beam is taken
    from its two neighbouring returns. Beams without two close neighbours
    (edges, isolated returns, no return) get 0, i.e. no grazing penalty.
    """
    ranges = np.asarray(reading.ranges_m, dtype=float)
    angles = np.asarray(reading.angles_rad, dtype=float)
    # Beams without a return become NaN points, so they never count as neighbours.
    finite_ranges = np.where(np.isfinite(ranges), ranges, np.nan)
    points = np.column_stack([finite_ranges * np.cos(angles), finite_ranges * np.sin(angles)])
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    length = np.linalg.norm(tangent, axis=1)
    usable = (np.isfinite(length) & (length > 1e-9) & (length < max_neighbour_gap_m)
              & np.isfinite(finite_ranges))
    factor = np.zeros(ranges.size)
    beam = np.column_stack([np.cos(angles[usable]), np.sin(angles[usable])])
    t = tangent[usable]
    # |cos(incidence)| = |beam x tangent| / |tangent| for a 2D surface direction.
    cos_incidence = np.abs(beam[:, 0] * t[:, 1] - beam[:, 1] * t[:, 0]) / length[usable]
    factor[usable] = 1.0 - np.clip(cos_incidence, 0.0, 1.0)
    return factor


def add_lidar_noise(
    reading: LidarReading,
    rng: np.random.Generator,
    range_std_m: float = LIDAR_RANGE_NOISE_STD_M,
    dropout_probability: float = LIDAR_DROPOUT_PROBABILITY,
    range_std_per_m: float = 0.0,
    resolution_m: float = 0.0,
    grazing_dropout: float = 0.0,
) -> LidarReading:
    """Add range noise, quantisation and dropouts to a new scan.

    Noise sd for a beam at range r is range_std_m + range_std_per_m * r.
    A valid beam is lost with probability dropout_probability plus
    grazing_dropout scaled by how edge-on it hits the surface. With only the
    first two settings, this is the original constant-noise model.
    """
    ranges = np.array(reading.ranges_m, dtype=float)
    valid = np.isfinite(ranges)
    if range_std_per_m > 0:
        std = range_std_m + range_std_per_m * ranges[valid]
        ranges[valid] += std * rng.standard_normal(valid.sum())
    else:
        ranges[valid] += rng.normal(0.0, range_std_m, valid.sum())
    if resolution_m > 0:
        ranges[valid] = np.round(ranges[valid] / resolution_m) * resolution_m
    if dropout_probability > 0 or grazing_dropout > 0:
        probability = dropout_probability + grazing_dropout * grazing_factor(reading)
        ranges[valid & (rng.random(ranges.size) < probability)] = np.inf
    finite = np.isfinite(ranges)
    ranges[finite & (ranges > reading.range_max_m)] = np.inf
    ranges[finite & (ranges < reading.range_min_m)] = np.nan
    return LidarReading(
        time=reading.time,
        angles_rad=reading.angles_rad,
        ranges_m=tuple(float(value) for value in ranges),
        range_min_m=reading.range_min_m,
        range_max_m=reading.range_max_m,
    )


def add_camera_noise(
    frame: CameraReading,
    rng: np.random.Generator,
    pixel_std: float = CAMERA_PIXEL_NOISE_STD,
) -> CameraReading:
    """Add image intensity noise to a new frame."""
    # Single precision is enough for 8-bit pixels and halves the working buffer.
    # Keep independent Gaussian noise for each pixel and colour channel.
    pixels = rng.standard_normal(frame.rgb.shape, dtype=np.float32)
    pixels *= pixel_std
    pixels += frame.rgb
    np.rint(pixels, out=pixels)
    np.clip(pixels, 0, 255, out=pixels)
    rgb = pixels.astype(np.uint8)
    rgb.setflags(write=False)
    return CameraReading(time=frame.time, rgb=rgb)


def distortion_maps(camera_matrix, dist_coeffs, width: int, height: int):
    """Remap tables that make an ideal pinhole image look like a distorted lens.

    For every pixel of the DISTORTED output image, find where that ray lands in
    the ideal image (cv2.undistortPoints), so cv2.remap can sample it there.
    Computed once per run.
    """
    columns, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    distorted = np.stack([columns.ravel(), rows.ravel()], axis=1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(distorted, camera_matrix, np.asarray(dist_coeffs, dtype=float),
                                P=camera_matrix).reshape(height, width, 2)
    return ideal[..., 0].astype(np.float32), ideal[..., 1].astype(np.float32)


class CameraNoiseModel:
    """Turn the renderer's perfect image into a realistic camera frame.

    Order follows the physics: the lens distorts the scene, motion during the
    exposure blurs it, the sensor's gain and noise are applied, and the result
    is quantised to 8 bits. Latency is handled by Perception.
    """

    def __init__(self, noise: SensorNoise, camera_matrix, width: int, height: int) -> None:
        self.noise = noise
        self.focal_px = float(np.asarray(camera_matrix)[0, 0])
        self._maps = None
        if np.any(np.asarray(noise.camera_distortion) != 0):
            self._maps = distortion_maps(camera_matrix, noise.camera_distortion, width, height)

    def blur_length_px(self, yaw_rate_rad_s: float) -> float:
        """Horizontal smear from turning during the exposure, in pixels."""
        return abs(yaw_rate_rad_s) * self.noise.camera_exposure_s * self.focal_px

    def apply(self, frame: CameraReading, rng: np.random.Generator, yaw_rate_rad_s: float = 0.0) -> CameraReading:
        noise = self.noise
        rgb = np.asarray(frame.rgb)
        if self._maps is not None:
            rgb = cv2.remap(rgb, *self._maps, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        blur = int(round(self.blur_length_px(yaw_rate_rad_s)))
        if blur >= 2:
            rgb = cv2.blur(rgb, (blur, 1))
        if noise.camera_gain_std > 0:
            gain = max(0.0, 1.0 + rng.normal(0.0, noise.camera_gain_std))
            rgb = np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        if noise.camera_pixel_std > 0:
            return add_camera_noise(CameraReading(frame.time, rgb), rng, noise.camera_pixel_std)
        rgb = np.array(rgb, dtype=np.uint8, copy=True)
        rgb.setflags(write=False)
        return CameraReading(time=frame.time, rgb=rgb)


def perturb_extrinsic(T_base_camera: np.ndarray, noise: SensorNoise) -> np.ndarray:
    """The camera mounting the robot BELIEVES it has, given a calibration error.

    The true mounting still renders the images; only the detector's assumed
    T_base_camera is shifted and turned about the base's vertical axis.
    """
    yaw = np.deg2rad(noise.camera_mount_yaw_error_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    assumed = np.array(T_base_camera, dtype=float, copy=True)
    assumed[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ assumed[:3, :3]
    assumed[:3, 3] += np.asarray(noise.camera_mount_error_m, dtype=float)
    return assumed


class ArucoLandmarkDetector:
    """Detect markers and estimate their pose using camera calibration."""

    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        camera_to_base,
        marker_sizes_m: dict[int, float] | None = None,
    ) -> None:
        self.camera_matrix = np.asarray(camera_matrix, dtype=float)
        # Intrinsics describe image projection; T_base_camera describes the fixed mount.
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=float)
        self.T_base_camera = np.asarray(camera_to_base, dtype=float)
        self.marker_sizes_m = dict(MARKER_SIZES_M if marker_sizes_m is None else marker_sizes_m)
        if any(not np.isfinite(size) or size <= 0 for size in self.marker_sizes_m.values()):
            raise ValueError("Marker sizes must be positive and finite")

        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, frame: CameraReading) -> list[MarkerObservation]:
        """Return detected markers; an empty list is normal when the dock is hidden."""
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        observations = []
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.ravel()):
                observation = self._measure(frame.time, int(marker_id), marker_corners.reshape(4, 2))
                if observation is not None:
                    observations.append(observation)
        return observations

    def _measure(self, time, marker_id, corners_px) -> MarkerObservation | None:
        """Solve the marker pose and reject small, invalid or poorly fitted detections."""
        marker_size = self.marker_sizes_m.get(marker_id)
        corners_px = np.asarray(corners_px, dtype=float)
        if marker_size is None or corners_px.shape != (4, 2) or not np.isfinite(corners_px).all():
            return None

        side_px = float(np.mean(np.linalg.norm(corners_px - np.roll(corners_px, -1, axis=0), axis=1)))
        # Very small markers do not provide enough corner detail for a stable pose.
        if side_px < MIN_MARKER_SIDE_PX:
            return None

        # IPPE requires these corners: top-left, top-right, bottom-right, bottom-left.
        half_side = marker_size / 2.0
        corners_m = np.array([
            [-half_side, half_side, 0], [half_side, half_side, 0],
            [half_side, -half_side, 0], [-half_side, -half_side, 0],
        ])
        # Match the known square to its image corners. A planar square can produce
        # two plausible poses, so keep the candidates for the ambiguity check below.
        count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            corners_m, corners_px, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if count == 0:
            return None

        # Keep finite solutions with every corner in front of the camera.
        solutions = []
        for error, rvec, tvec in zip(np.asarray(errors).ravel(), rvecs, tvecs):
            if not np.isfinite(error) or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            corner_depths = (corners_m @ rotation.T + tvec.reshape(3))[:, 2]
            if np.all(corner_depths > 0):
                solutions.append((float(error), make_transform(rotation, tvec)))
        # Reprojection error measures how well the projected corners match the image.
        solutions.sort(key=lambda solution: solution[0])
        if not solutions or solutions[0][0] > MAX_REPROJECTION_ERROR_PX:
            return None

        error, T_camera_marker = solutions[0]
        ambiguity = 0.0
        if len(solutions) > 1:
            second_error = solutions[1][0]
            # A tie at zero error is still ambiguous.
            ambiguity = 1.0 if second_error <= 1e-9 else error / second_error

        # Compose camera-to-base calibration with the measured marker-to-camera pose.
        T_base_marker = self.T_base_camera @ T_camera_marker
        x, y = T_base_marker[:2, 3]
        return MarkerObservation(
            time=time, marker_id=marker_id,
            range_m=float(np.hypot(x, y)), bearing_rad=float(np.arctan2(y, x)),
            T_camera_marker=T_camera_marker, T_base_marker=T_base_marker,
            reprojection_error_px=error, ambiguity_ratio=ambiguity,
            side_px=side_px, corners_px=corners_px.copy(),
        )

    def draw(self, frame: CameraReading, observations: list[MarkerObservation]) -> np.ndarray:
        """Return an RGB image with marker outlines and pose quality for the demo."""
        image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        for observation in observations:
            corners = observation.corners_px.astype(np.int32)
            colour = (0, 255, 0) if observation.orientation_reliable else (0, 165, 255)
            cv2.polylines(image, [corners], True, colour, 2)
            rvec, _ = cv2.Rodrigues(observation.T_camera_marker[:3, :3])
            cv2.drawFrameAxes(
                image, self.camera_matrix, self.dist_coeffs, rvec,
                observation.T_camera_marker[:3, 3], self.marker_sizes_m[observation.marker_id] / 2,
            )
            label = f"id {observation.marker_id}  ambiguity {observation.ambiguity_ratio:.2f}"
            origin = (max(5, int(corners[:, 0].min())), max(20, int(corners[:, 1].min()) - 10))
            cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def marker_mounts() -> dict[int, np.ndarray]:
    """Measured marker-to-dock mounting geometry, not any world location.

    Dock +X points out of the charging face; marker +Z points the same way.
    These are physical calibration values, like marker size and camera mounting.
    """
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    return {0: make_transform(rotation, [-.05, 0, .40]),
            1: make_transform(rotation, [-.025, 0, .12])}


def select_dock_observation(markers: list[MarkerObservation]) -> DockObservation | None:
    """Measure the dock in the CURRENT robot frame, without a world marker map."""
    mounts = marker_mounts()
    # Prefer the marker with more image detail. Both IDs describe the same rigid dock,
    # but we supply only one measurement per image rather than double-counting it.
    for marker in sorted(markers, key=lambda item: (-item.side_px, item.marker_id)):
        if not marker.orientation_reliable or marker.marker_id not in mounts:
            continue
        # T_base_dock = T_base_marker @ T_marker_dock
        # The inverse mount removes the marker's offset from the dock origin.
        T_base_dock = marker.T_base_marker @ invert_transform(mounts[marker.marker_id])
        if not np.isfinite(T_base_dock).all():
            continue
        # Check height and tilt before reducing the 3D solution to a planar pose.
        # Otherwise an incorrect tilted solution could look plausible in XY alone.
        tilt = np.arccos(np.clip(T_base_dock[2, 2], -1, 1))
        if abs(T_base_dock[2, 3]) > MAX_DOCK_HEIGHT_ERROR_M or tilt > MAX_DOCK_TILT_RAD:
            continue
        pose = transform_to_pose(T_base_dock)
        # Approximate measurement noise from corner uncertainty and apparent size.
        # The noise floors prevent large markers from being treated as exact poses.
        corner_fraction = CAMERA_CORNER_STD_PX / marker.side_px
        translation_std = max(CAMERA_TRANSLATION_STD_M, marker.range_m * corner_fraction)
        heading_std = max(CAMERA_HEADING_STD_RAD, 2 * corner_fraction)
        # Orientation uncertainty shifts the dock origin by the small mounting offset,
        # not by the full robot-to-marker range (this is a relative measurement).
        offset = T_base_dock[:2, 3] - marker.T_base_marker[:2, 3]
        heading_effect = np.array([-offset[1], offset[0], 1.0])
        covariance = np.diag([translation_std**2, translation_std**2, 0.0])
        # This outer product includes the correlation between position and heading
        # caused by rotating the marker-to-dock offset.
        covariance += heading_std**2 * np.outer(heading_effect, heading_effect)
        return DockObservation(marker.time, marker.marker_id, tuple(pose), covariance)
    return None


class Perception:
    """Read sensors and return obstacle points plus any fresh robot-relative dock pose.

    noise: False for ideal sensors, True for the original demo noise
        (SensorNoise.nominal()), or any SensorNoise for robustness studies.
    rng: noise generator; pass a seeded one for repeatable trials.
    dist_coeffs: lens distortion the detector assumes. Defaults to the
        renderer's (none). Pass calibrated values when testing distortion.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        noise: bool | SensorNoise = False,
        rng: np.random.Generator | None = None,
        dist_coeffs=None,
    ) -> None:
        if noise is True:
            noise = SensorNoise.nominal()
        elif noise is False or noise is None:
            noise = SensorNoise.ideal()
        if not isinstance(noise, SensorNoise):
            raise TypeError("noise must be a bool or a SensorNoise")
        self.noise_settings = noise
        self.noise = noise != SensorNoise.ideal()
        self.lidar_to_base = lidar_transform_from_model(model)
        self.camera = RGBCamera(model)
        self.dist_coeffs = np.asarray(self.camera.dist_coeffs if dist_coeffs is None else dist_coeffs, dtype=float)
        # The detector uses the ASSUMED calibration, which may differ from the truth.
        self.detector = ArucoLandmarkDetector(
            self.camera.camera_matrix, self.dist_coeffs,
            perturb_extrinsic(self.camera.camera_to_base, noise),
        )
        self._camera_noise = CameraNoiseModel(noise, self.camera.camera_matrix, CAMERA_WIDTH, CAMERA_HEIGHT)
        self._drive = DifferentialDriveModel()
        # Velocity indices of the wheel joints: what the motor encoders measure.
        self._wheel_dofs = tuple(int(model.jnt_dofadr[model.joint(name).id])
                                 for name in ("wheel_left", "wheel_right"))
        self.rng = rng if rng is not None else np.random.default_rng()
        self._scan: LidarReading | None = None
        self._points = np.empty((0, 2))
        self._frame: CameraReading | None = None
        self._last_capture_time: float | None = None
        self._in_transit: deque[CameraReading] = deque()
        self._last_update_time: float | None = None

    def _yaw_rate(self, data: mujoco.MjData) -> float:
        """Turning rate from the wheel joint speeds, used only to blur the image."""
        left, right = (float(data.qvel[dof]) for dof in self._wheel_dofs)
        return self._drive.wheel_radius_m * (right - left) / self._drive.wheel_separation_m

    def _lidar_noise(self, scan: LidarReading) -> LidarReading:
        n = self.noise_settings
        if not self.noise:
            return scan
        return add_lidar_noise(scan, self.rng, n.lidar_range_std_m, n.lidar_dropout_probability,
                               range_std_per_m=n.lidar_range_std_per_m,
                               resolution_m=n.lidar_resolution_m,
                               grazing_dropout=n.lidar_grazing_dropout)

    def update(self, data: mujoco.MjData) -> PerceptionOutput:
        """Process each sensor sample once, using its acquisition timestamp."""
        now = float(data.time)
        # Discard cached readings if the simulation clock has been reset.
        if self._last_update_time is not None and now < self._last_update_time:
            self._scan = None
            self._frame = None
            self._last_capture_time = None
            self._in_transit.clear()
        self._last_update_time = now

        # Sensors and control may run at different rates. Cache each scan and apply
        # simulated noise once per acquisition, not again on every control update.
        scan_time = float(data.sensor('lidar_time').data[0])
        if self._scan is None or scan_time != self._scan.time:
            self._scan = self._lidar_noise(read_lidar(data))
            self._points = lidar_points_in_base(self._scan, self.lidar_to_base)

        # Capture: apply the camera imperfections once, when the frame is taken,
        # because motion blur depends on the motion during that exposure.
        frame = self.camera.read(data)
        if self._last_capture_time is None or frame.time != self._last_capture_time:
            self._last_capture_time = frame.time
            if self.noise:
                frame = self._camera_noise.apply(frame, self.rng, self._yaw_rate(data))
            self._in_transit.append(frame)

        # Delivery: a frame reaches perception camera_latency_s after capture and
        # is stamped on arrival. With zero latency it arrives at once, as before.
        arrived = None
        while self._in_transit and self._in_transit[0].time <= now - self.noise_settings.camera_latency_s + 1e-9:
            arrived = self._in_transit.popleft()
        new_frame = arrived is not None
        # Cached images can still be displayed, but must not become repeated EKF
        # corrections. Only new frames yield a pose.
        markers = []
        dock_observation = None
        if new_frame:
            self._frame = arrived if arrived.time == now else CameraReading(time=now, rgb=arrived.rgb)
            markers = self.detector.detect(self._frame)
            dock_observation = select_dock_observation(markers)

        return PerceptionOutput(
            time=now,
            lidar=self._scan,
            lidar_points_base=self._points,
            camera=self._frame,
            markers=markers,
            new_camera_frame=new_frame,
            dock_observation=dock_observation,
        )

    def close(self) -> None:
        """Release the camera renderer when the simulation finishes."""
        self.camera.close()

    def __enter__(self) -> "Perception":
        return self

    def __exit__(self, *exc) -> None:
        self.close()