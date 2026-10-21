"""
Our perception module should convert sensor readings into useful measurements
for estimation and navigation.

Currently its inputs are:
- RGB camera images and camera calibration
- LiDAR scans
- Known marker IDs, physical sizes and installed world poses

It has 2 main calculations:
- Camera pose:
    - Detect known aruco markers and estimate their pose relative to the camera
    - Apply the camera mounting transform to get the marker pose in the robot frame
    - Use the known marker world pose to derive the robot's world pose
    - Select one reliable pose from each new image and give it a covariance
- Obstacle points:
    - Convert valid LiDAR ranges into points in the robot frame
    - Preserve the scan timestamp and flag returns below the minimum range

It outputs:
- A PerceptionOutput containing obstacle points, the latest image and detections
- A CameraPoseObservation (time, marker ID, world pose, covariance), when available

No usable new marker pose means the estimator performs prediction only.
The runtime uses sensor data and fixed map/calibration parameters, never the
robot's true world pose. T_a_b maps coordinates from frame b into frame a.

Offline calibration and evaluation live in scripts/perception_diagnostics.py.
"""

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from .sensing import CameraReading, LidarReading, RGBCamera, read_lidar
from .state_model import wrap_angle


ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_SIZES_M = {0: 0.20, 1: 0.04} # Black square sizes, excluding the white borders

MIN_MARKER_SIDE_PX = 20.0
MAX_REPROJECTION_ERROR_PX = 2.0
AMBIGUITY_WARNING = 0.5

# Initial noise assumptions for the demo; tune these using measured errors.
LIDAR_RANGE_NOISE_STD_M = 0.01
LIDAR_DROPOUT_PROBABILITY = 0.0
CAMERA_PIXEL_NOISE_STD = 2.0 # Image intensity noise on the 0–255 scale
CAMERA_CORNER_STD_PX = 1.0 # Assumed corner-position uncertainty
CAMERA_TRANSLATION_STD_M = 0.02
CAMERA_HEADING_STD_RAD = np.deg2rad(5.0)


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
class CameraPoseObservation:
    """Store a camera-derived robot world pose and its 3x3 covariance."""

    time: float
    marker_id: int
    pose_world: tuple[float, float, float]
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
    pose_observation: CameraPoseObservation | None = None

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
    """Convert a planar [x, y, heading] into T_world_base."""
    x, y, heading = pose
    c, s = np.cos(heading), np.sin(heading)
    return make_transform([[c, -s, 0], [s, c, 0], [0, 0, 1]], [x, y, 0])


def transform_to_pose(transform: np.ndarray) -> np.ndarray:
    """Extract planar [x, y, heading] from a 4x4 transform."""
    heading = np.arctan2(transform[1, 0], transform[0, 0])
    return np.array([transform[0, 3], transform[1, 3], wrap_angle(heading)])


def _fixed_body_transform(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    """Get a fixed body's world transform from its installed geometry."""
    ancestors = []
    while body_id != 0:
        if model.body_jntnum[body_id] or model.body_mocapid[body_id] != -1:
            raise ValueError("Known markers must be fixed to the world")
        ancestors.append(body_id)
        body_id = int(model.body_parentid[body_id])

    transform = np.eye(4)
    for body_id in reversed(ancestors):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.body_quat[body_id])
        transform = transform @ make_transform(rotation, model.body_pos[body_id])
    return transform


def marker_map_from_model(model: mujoco.MjModel) -> dict[int, np.ndarray]:
    """Load the known marker map from fixed aruco_marker_<id> sites.

    This is the declared map prior for our demo, using static model geometry.
    A real robot would receive the surveyed marker poses as configuration.
    """
    marker_map = {}
    prefix = "aruco_marker_"
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        if not name.startswith(prefix) or not name[len(prefix):].isdigit():
            continue
        marker_id = int(name[len(prefix):])
        if marker_id in marker_map:
            raise ValueError(f"Duplicate marker ID {marker_id}")
        T_world_body = _fixed_body_transform(model, int(model.site_bodyid[site_id]))
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.site_quat[site_id])
        marker_map[marker_id] = T_world_body @ make_transform(rotation, model.site_pos[site_id])
    return marker_map


def lidar_transform_from_model(model: mujoco.MjModel) -> np.ndarray:
    """Read the fixed LiDAR-to-base transform configured before compilation."""
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


def lidar_points_to_world(points_base: np.ndarray, pose) -> np.ndarray:
    """Transform obstacle points using the estimated robot pose at scan time."""
    x, y, heading = pose
    c, s = np.cos(heading), np.sin(heading)
    return np.asarray(points_base) @ np.array([[c, s], [-s, c]]) + [x, y]


def add_lidar_noise(
    reading: LidarReading,
    rng: np.random.Generator,
    range_std_m: float = LIDAR_RANGE_NOISE_STD_M,
    dropout_probability: float = LIDAR_DROPOUT_PROBABILITY,
) -> LidarReading:
    """Add Gaussian range noise and optional dropouts to a new scan."""
    ranges = np.array(reading.ranges_m, dtype=float)
    valid = np.isfinite(ranges)
    ranges[valid] += rng.normal(0.0, range_std_m, valid.sum())
    if dropout_probability > 0:
        ranges[valid & (rng.random(ranges.size) < dropout_probability)] = np.inf
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
    pixels = frame.rgb.astype(float) + rng.normal(0.0, pixel_std, frame.rgb.shape)
    rgb = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    rgb.setflags(write=False)
    return CameraReading(time=frame.time, rgb=rgb)


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
        if side_px < MIN_MARKER_SIDE_PX:
            return None

        # IPPE requires these corners: top-left, top-right, bottom-right, bottom-left.
        half_side = marker_size / 2.0
        corners_m = np.array([
            [-half_side, half_side, 0], [half_side, half_side, 0],
            [half_side, -half_side, 0], [-half_side, -half_side, 0],
        ])
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
        solutions.sort(key=lambda solution: solution[0])
        if not solutions or solutions[0][0] > MAX_REPROJECTION_ERROR_PX:
            return None

        error, T_camera_marker = solutions[0]
        ambiguity = 0.0
        if len(solutions) > 1:
            second_error = solutions[1][0]
            # A tie at zero error is still ambiguous.
            ambiguity = 1.0 if second_error <= 1e-9 else error / second_error

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


def robot_pose_from_marker(
    observation: MarkerObservation,
    marker_map: dict[int, np.ndarray],
) -> np.ndarray | None:
    """Derive the robot world pose; the caller must check detection quality."""
    T_world_marker = marker_map.get(observation.marker_id)
    if T_world_marker is None:
        return None
    # T_world_base = T_world_marker @ inverse(T_base_marker)
    return transform_to_pose(T_world_marker @ invert_transform(observation.T_base_marker))


def select_pose_observation(
    markers: list[MarkerObservation],
    marker_map: dict[int, np.ndarray],
) -> CameraPoseObservation | None:
    """Choose one reliable marker pose for the EKF from the current image."""
    # Prefer the largest visible marker. Do not fuse two poses from one image
    # as independent measurements, since they share camera/calibration errors.
    candidates = sorted(markers, key=lambda marker: (-marker.side_px, marker.marker_id))
    for marker in candidates:
        if not marker.orientation_reliable:
            continue
        pose = robot_pose_from_marker(marker, marker_map)
        if pose is None or not np.isfinite(pose).all():
            continue

        # Starting covariance model, not a calibrated accuracy claim.
        # Small image footprints and long ranges increase measurement uncertainty.
        corner_fraction = CAMERA_CORNER_STD_PX / marker.side_px
        translation_std = max(CAMERA_TRANSLATION_STD_M, marker.range_m * corner_fraction)
        heading_std = max(CAMERA_HEADING_STD_RAD, 2.0 * corner_fraction)

        # Heading error also moves the inferred robot position around the marker.
        # d(position)/d(heading) = [dy, -dx] in the world frame.
        dx, dy = marker_map[marker.marker_id][:2, 3] - pose[:2]
        heading_effect = np.array([dy, -dx, 1.0])
        covariance = np.diag([translation_std**2, translation_std**2, 0.0])
        covariance += heading_std**2 * np.outer(heading_effect, heading_effect)

        return CameraPoseObservation(
            time=marker.time,
            marker_id=marker.marker_id,
            pose_world=tuple(float(value) for value in pose),
            covariance=covariance,
        )
    return None


class Perception:
    """Read sensors and return obstacle points plus any fresh camera pose."""

    def __init__(
        self,
        model: mujoco.MjModel,
        noise: bool = False,
        rng: np.random.Generator | None = None,
        marker_map: dict[int, np.ndarray] | None = None,
    ) -> None:
        self.marker_map = marker_map_from_model(model) if marker_map is None else marker_map
        self.lidar_to_base = lidar_transform_from_model(model)
        self.camera = RGBCamera(model)
        self.detector = ArucoLandmarkDetector(
            self.camera.camera_matrix, self.camera.dist_coeffs, self.camera.camera_to_base,
        )
        self.noise = noise
        self.rng = rng if rng is not None else np.random.default_rng()
        self._scan: LidarReading | None = None
        self._frame: CameraReading | None = None
        self._last_update_time: float | None = None

    def update(self, data: mujoco.MjData) -> PerceptionOutput:
        """Process each sensor sample once, using its acquisition timestamp."""
        now = float(data.time)
        if self._last_update_time is not None and now < self._last_update_time:
            self._scan = None
            self._frame = None
        self._last_update_time = now

        scan = read_lidar(data)
        if self._scan is None or scan.time != self._scan.time:
            self._scan = add_lidar_noise(scan, self.rng) if self.noise else scan

        frame = self.camera.read(data)
        new_frame = self._frame is None or frame.time != self._frame.time
        markers = []
        pose_observation = None
        if new_frame:
            self._frame = add_camera_noise(frame, self.rng) if self.noise else frame
            markers = self.detector.detect(self._frame)
            pose_observation = select_pose_observation(markers, self.marker_map)

        return PerceptionOutput(
            time=now,
            lidar=self._scan,
            lidar_points_base=lidar_points_in_base(self._scan, self.lidar_to_base),
            camera=self._frame,
            markers=markers,
            new_camera_frame=new_frame,
            pose_observation=pose_observation,
        )

    def close(self) -> None:
        """Release the camera renderer when the simulation finishes."""
        self.camera.close()

    def __enter__(self) -> "Perception":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
