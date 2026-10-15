"""Perception for Track A: what the robot's sensors tell the rest of the system.

Built on the raw readers in sensing.py. Adapted from the 41014 tutorials:
Week 2 (measurement models, the frame chain p^W = T^WB T^BS p^S, range-bearing
noise), Week 3 (camera calibration, reprojection error, ArUco pose and the
planar ambiguity) and Week 4 (lidar polar to Cartesian, constant range noise).
Tutorial code was rewritten with plain OpenCV instead of the Machine Vision
Toolbox so the project keeps a single OpenCV install.

What teammates use
------------------
    perception = Perception(model)              # once, after load_model()
    out = perception.update(data)               # every control step
    out.lidar_points_base   (N, 2) obstacle points in the robot frame [m]
    out.markers             list of MarkerObservation (empty if none seen)
    out.camera              latest CameraReading (30 Hz)

    MarkerObservation.range_m, .bearing_rad     EKF measurement z
    expected_range_bearing(pose)                EKF measurement model h(x)
    range_bearing_jacobian(pose)                its Jacobian H
    robot_pose_from_marker(observation)         robot [x, y, theta] from the marker
    lidar_points_to_world(points, pose)         scan on the map, given a pose ESTIMATE

Frames (4 x 4 transforms; T_a_b maps coordinates in frame b into frame a)
--------------------------------------------------------------------------
    world   MuJoCo world: x east, y north, z up.
    base    robot base, at the wheel axle centre: x forward, y left, z up.
    lidar   scan centre, at (-0.049, 0, 0.1255) m in base, not rotated.
    camera  OpenCV optical frame at (0.076, 0, 0.103) m in base:
            x right, y down, z forward.
    marker  OpenCV ArUco frame: x right, y up, z out of the marker face.

Ground truth
------------
Everything above the "SELF-TEST" banner uses only sensor data and fixed,
known quantities (calibration, mounting positions, the charger's installed
location). Ground truth is read only in the self-test and the calibration
rig, to score results.

Command line (run from the repository root)
-------------------------------------------
    python3 -m track_a_mobile_robot.perception make-marker
    python3 -m track_a_mobile_robot.perception test [--repeats 20] [--pixel-noise 2]
    python3 -m track_a_mobile_robot.perception calibrate
    python3 -m track_a_mobile_robot.perception sensitivity
On Linux without a display, prefix with MUJOCO_GL=egl.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .sensing import (
    CAMERA_HEIGHT,
    CAMERA_NAME,
    CAMERA_WIDTH,
    CameraReading,
    LidarReading,
    RGBCamera,
    read_lidar,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "perception"
MARKER_IMAGE = REPO_ROOT / "world" / "assets" / "marker_0.png"

# =============================================================================
# 1. Constants: marker, landmark map, sensor mounting, noise
# =============================================================================

ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_SIDE_M = 0.20            # black square of the printed marker

# Landmark map: where the charger's marker is installed in the world. This is
# known in advance (like a surveyed landmark), not robot ground truth, so the
# estimator may use it. Matches site aruco_marker_0 in room_with_charger.xml.
# Rotation columns are the marker's x, y, z axes expressed in the world.
MARKER_MAP = {
    0: np.array([
        [0.0, 0.0, -1.0, 4.9295],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.16],
        [0.0, 0.0, 0.0, 1.0],
    ]),
}

# LiDAR extrinsic, from the <frame> in the robot XML.
LIDAR_TO_BASE_XYZ = np.array([-0.049, 0.0, 0.1255])

# Detection quality gates. Below about 20 px across, the bit pattern is
# unreliable (Week 3 wrap-up); a large reprojection error means the corners
# do not fit a square.
MIN_MARKER_SIDE_PX = 20.0
MAX_REPROJECTION_ERROR_PX = 2.0
# Ambiguity ratio above which the marker's orientation should not be trusted.
AMBIGUITY_WARNING = 0.5

# Default simulated noise. STARTING VALUES, NOT MEASURED: replace with
# datasheet or measured figures and cite the source on the project page.
LIDAR_RANGE_NOISE_STD_M = 0.01      # constant range noise, as in Week 4 Part 5
LIDAR_DROPOUT_PROBABILITY = 0.0     # chance a valid beam returns nothing
CAMERA_PIXEL_NOISE_STD = 2.0        # Gaussian noise on 0-255 pixel values

# Marker corners in the marker frame, in the order IPPE_SQUARE requires:
# top-left, top-right, bottom-right, bottom-left (as in Week 3 Part E).
_H = MARKER_SIDE_M / 2
MARKER_CORNERS_M = np.array([[-_H, _H, 0.0], [_H, _H, 0.0], [_H, -_H, 0.0], [-_H, -_H, 0.0]])


# =============================================================================
# 2. Frame helpers
# =============================================================================

def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    """Build a 4 x 4 rigid transform from a 3 x 3 rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation).reshape(3, 3)
    T[:3, 3] = np.asarray(translation).ravel()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (R^T, -R^T t)."""
    return make_transform(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def pose_to_transform(pose) -> np.ndarray:
    """Planar pose [x, y, theta] to the 4 x 4 transform T_world_base."""
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    return make_transform([[c, -s, 0], [s, c, 0], [0, 0, 1]], [x, y, 0.0])


def transform_to_pose(T: np.ndarray) -> np.ndarray:
    """4 x 4 transform to a planar pose [x, y, theta] (drops height, roll, pitch)."""
    return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])


def wrap_angle(angle):
    """Wrap an angle (or array of angles) to (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


# =============================================================================
# 3. LiDAR output
# =============================================================================

def lidar_points_in_base(reading: LidarReading) -> np.ndarray:
    """Scan as an (N, 2) array of x, y points in the robot base frame.

    Polar to Cartesian in the LiDAR frame (Week 4 Part 5), then shifted by the
    LiDAR mounting offset (Week 2 Part C: p^B = T^BS p^S, here a translation).
    Beams with no return (inf) or too close to measure (nan) are dropped.
    """
    ranges = np.asarray(reading.ranges_m, dtype=float)
    angles = np.asarray(reading.angles_rad, dtype=float)
    valid = np.isfinite(ranges)
    x = ranges[valid] * np.cos(angles[valid]) + LIDAR_TO_BASE_XYZ[0]
    y = ranges[valid] * np.sin(angles[valid]) + LIDAR_TO_BASE_XYZ[1]
    return np.column_stack([x, y])


def lidar_points_to_world(points_base: np.ndarray, pose) -> np.ndarray:
    """Move base-frame points onto the map using a robot pose [x, y, theta].

    Pass the ESTIMATED pose from the EKF, never the simulator's true pose.
    """
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(points_base) @ np.array([[c, s], [-s, c]]) + [x, y]


# =============================================================================
# 4. Sensor noise (for robustness testing)
# =============================================================================

def add_lidar_noise(
    reading: LidarReading,
    rng: np.random.Generator,
    range_std_m: float = LIDAR_RANGE_NOISE_STD_M,
    dropout_probability: float = LIDAR_DROPOUT_PROBABILITY,
) -> LidarReading:
    """New scan with Gaussian range noise and optional dropouts.

    Only beams with a valid return are perturbed. Noisy ranges outside the
    sensor limits are reported the same way as ideal ones (inf or nan).
    """
    ranges = np.asarray(reading.ranges_m, dtype=float).copy()
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
        ranges_m=tuple(float(r) for r in ranges),
        range_min_m=reading.range_min_m,
        range_max_m=reading.range_max_m,
    )


def add_camera_noise(
    frame: CameraReading,
    rng: np.random.Generator,
    pixel_std: float = CAMERA_PIXEL_NOISE_STD,
) -> CameraReading:
    """New frame with additive Gaussian pixel noise, clipped to 0-255."""
    noisy = np.asarray(frame.rgb, dtype=float) + rng.normal(0.0, pixel_std, frame.rgb.shape)
    rgb = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    rgb.setflags(write=False)
    return CameraReading(time=frame.time, rgb=rgb)


# =============================================================================
# 5. ArUco landmark detection and pose
# =============================================================================

@dataclass(frozen=True)
class MarkerObservation:
    """One detected marker.

    range_m, bearing_rad: base origin to marker centre, in the base x-y plane;
        bearing is counterclockwise from straight ahead. This is the EKF's z.
    T_camera_marker: marker pose in the camera frame (from PnP).
    T_base_marker: marker pose in the robot base frame (extrinsic applied).
    reprojection_error_px: RMS corner error of the chosen PnP solution.
    ambiguity_ratio: best solution's error divided by the second planar
        solution's error. Near 1 means both orientations fit the image almost
        equally well (Week 3 planar ambiguity), so the marker's orientation is
        unreliable. Range and bearing are much less affected.
    side_px: mean side length of the marker in the image.
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
        return self.ambiguity_ratio < AMBIGUITY_WARNING


class ArucoLandmarkDetector:
    """Detect ArUco markers and measure their pose, range and bearing.

    camera_matrix, dist_coeffs: intrinsics. camera_to_base: the extrinsic
    T_base_camera. All three come from RGBCamera by default (see Perception).
    """

    def __init__(self, camera_matrix, dist_coeffs, camera_to_base):
        self.camera_matrix = np.asarray(camera_matrix, dtype=float)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=float)
        self.T_base_camera = np.asarray(camera_to_base, dtype=float)
        parameters = cv2.aruco.DetectorParameters()
        # Sub-pixel corners; otherwise corners snap to whole pixels.
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY), parameters)

    def detect(self, frame: CameraReading) -> list[MarkerObservation]:
        """All markers passing the quality gates; an empty list is normal.

        Empty means out of view, too far, blocked or a poor fit. The estimator
        should then run its prediction step only.
        """
        gray = cv2.cvtColor(np.asarray(frame.rgb), cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        observations = []
        for marker_corners, marker_id in zip(corners, ids.ravel()):
            obs = self._measure(frame.time, int(marker_id), marker_corners.reshape(4, 2))
            if obs is not None:
                observations.append(obs)
        return observations

    def _measure(self, time, marker_id, corners_px) -> MarkerObservation | None:
        side_px = float(np.mean(np.linalg.norm(corners_px - np.roll(corners_px, -1, axis=0), axis=1)))
        if side_px < MIN_MARKER_SIDE_PX:
            return None
        # Both planar PnP solutions, sorted by reprojection error.
        count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            MARKER_CORNERS_M, corners_px.astype(np.float64),
            self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if count == 0:
            return None
        errors = np.asarray(errors).ravel()
        if errors[0] > MAX_REPROJECTION_ERROR_PX:
            return None
        ambiguity = float(errors[0] / errors[1]) if count > 1 and errors[1] > 0 else 0.0

        rotation, _ = cv2.Rodrigues(rvecs[0])
        T_camera_marker = make_transform(rotation, tvecs[0])
        T_base_marker = self.T_base_camera @ T_camera_marker       # frame chain
        x, y = T_base_marker[0, 3], T_base_marker[1, 3]
        return MarkerObservation(
            time=time, marker_id=marker_id,
            range_m=float(np.hypot(x, y)), bearing_rad=float(np.arctan2(y, x)),
            T_camera_marker=T_camera_marker, T_base_marker=T_base_marker,
            reprojection_error_px=float(errors[0]), ambiguity_ratio=ambiguity,
            side_px=side_px, corners_px=corners_px,
        )

    def draw(self, frame: CameraReading, observations: list[MarkerObservation]) -> np.ndarray:
        """RGB copy of the frame with outlines, axes (x red, y green, z blue)
        and the range and bearing written on it. For the demo video."""
        image = cv2.cvtColor(np.asarray(frame.rgb), cv2.COLOR_RGB2BGR)   # OpenCV draws in BGR
        for obs in observations:
            cv2.polylines(image, [obs.corners_px.astype(np.int32)], True, (0, 255, 0), 2)
            rvec, _ = cv2.Rodrigues(obs.T_camera_marker[:3, :3])
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec,
                              obs.T_camera_marker[:3, 3], MARKER_SIDE_M / 2)
            text = (f"id {obs.marker_id}  r={obs.range_m:.3f} m  "
                    f"b={np.degrees(obs.bearing_rad):+.1f} deg  amb={obs.ambiguity_ratio:.2f}")
            corner = obs.corners_px.min(axis=0).astype(int)
            origin = (max(5, corner[0] - 120), max(20, corner[1] - 10))
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if not observations:
            cv2.putText(image, "no marker detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def robot_pose_from_marker(observation: MarkerObservation, marker_map: dict = MARKER_MAP) -> np.ndarray | None:
    """Robot pose [x, y, theta] in the world from one marker sighting.

    Frame chain: T_world_base = T_world_marker * inverse(T_base_marker).
    Uses the marker's full pose including orientation, so it inherits the
    planar ambiguity: check observation.orientation_reliable first. For the
    EKF, range and bearing to the known marker are the safer measurement.
    """
    T_world_marker = marker_map.get(observation.marker_id)
    if T_world_marker is None:
        return None
    return transform_to_pose(T_world_marker @ invert_transform(observation.T_base_marker))


def expected_range_bearing(pose, marker_id: int = 0, marker_map: dict = MARKER_MAP) -> np.ndarray:
    """Measurement model h(x): range and bearing expected from pose [x, y, theta].

    The h(x) of the Week 2 measurement equation z = h(x) + b + eta.
    """
    dx, dy = marker_map[marker_id][:2, 3] - np.asarray(pose[:2], dtype=float)
    return np.array([np.hypot(dx, dy), wrap_angle(np.arctan2(dy, dx) - pose[2])])


def range_bearing_jacobian(pose, marker_id: int = 0, marker_map: dict = MARKER_MAP) -> np.ndarray:
    """Jacobian H = dh/dx of expected_range_bearing, a 2 x 3 matrix for the EKF."""
    dx, dy = marker_map[marker_id][:2, 3] - np.asarray(pose[:2], dtype=float)
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / q, -dx / q, -1.0]])


# =============================================================================
# 6. Perception: one call per control step
# =============================================================================

@dataclass(frozen=True)
class PerceptionOutput:
    time: float
    lidar: LidarReading                 # the scan (noisy if noise is enabled)
    lidar_points_base: np.ndarray       # (N, 2) points in the base frame
    camera: CameraReading               # latest frame (noisy if enabled)
    markers: list                       # MarkerObservation, new frames only
    new_camera_frame: bool              # True when the camera produced a new image


class Perception:
    """Everything the estimator, navigation and controller need from sensors.

    model: from main.load_model(). noise: add the simulated sensor noise
    above. rng: noise generator (pass a seeded one for repeatable trials).
    Call close() at the end, or use `with Perception(model) as perception:`.
    """

    def __init__(self, model: mujoco.MjModel, noise: bool = False, rng: np.random.Generator | None = None):
        self.camera = RGBCamera(model)
        self.detector = ArucoLandmarkDetector(
            self.camera.camera_matrix, self.camera.dist_coeffs, self.camera.camera_to_base)
        self.noise = noise
        self.rng = rng if rng is not None else np.random.default_rng()
        self._last_frame_time = None

    def update(self, data: mujoco.MjData) -> PerceptionOutput:
        scan = read_lidar(data)
        frame = self.camera.read(data)
        if self.noise:
            scan = add_lidar_noise(scan, self.rng)
        new_frame = frame.time != self._last_frame_time
        markers = []
        if new_frame:
            self._last_frame_time = frame.time
            if self.noise:
                frame = add_camera_noise(frame, self.rng)
            markers = self.detector.detect(frame)
        return PerceptionOutput(time=float(data.time), lidar=scan,
                                lidar_points_base=lidar_points_in_base(scan),
                                camera=frame, markers=markers, new_camera_frame=new_frame)

    def close(self) -> None:
        self.camera.close()

    def __enter__(self) -> "Perception":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# =============================================================================
# 7. Marker image for the world
# =============================================================================

def make_marker_image(path: Path = MARKER_IMAGE) -> Path:
    """Write the station's ArUco texture: 300 px marker in a 420 px white panel.

    The world XML scales the panel to 0.28 m, so the black square is 0.20 m
    (MARKER_SIDE_M) with a 0.04 m white border. Saved as 3-channel colour so
    MuJoCo loads it the same way everywhere.
    """
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY), 0, 300)
    panel = np.full((420, 420), 255, np.uint8)
    panel[60:360, 60:360] = marker
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)):
        raise RuntimeError(f"Could not write {path}")
    return path


# =============================================================================
# SELF-TEST AND CALIBRATION. Ground truth is read below this line only, to
# score results and to place calibration targets. Nothing above uses it.
# =============================================================================

TEST_POSES = [
    (2.0, 0.0, 0.0),
    (2.8, 0.0, 0.0),
    (3.5, 0.0, 0.0),
    (3.5, 0.4, -0.15),
    (3.5, -0.4, 0.25),
    (4.0, 0.0, 0.0),
    (4.0, 0.3, 0.3),      # marker just outside the 62 deg field of view
    (4.4, 0.0, 0.0),      # marker partly leaves the top of the image
]


def _load_scene(extra_board: bool = False) -> mujoco.MjModel:
    from .main import SCENE, load_model
    if not MARKER_IMAGE.exists():
        make_marker_image()
    if not extra_board:
        return load_model(SCENE)
    # Calibration scene: the project scene plus a free-floating checkerboard.
    texture = OUTPUT_DIR / "checkerboard.png"
    half_w, half_h = _make_checkerboard(texture)
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.add_texture(name="checkerboard", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(texture))
    material = spec.add_material(name="checkerboard", specular=0, shininess=0, reflectance=0)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "checkerboard"
    board = spec.worldbody.add_body(name="calibration_board", mocap=True, pos=[0, 0, -5])
    board.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_w, half_h, 0.001],
                   material="checkerboard", contype=0, conaffinity=0)
    return spec.compile()


def _place_robot(model, data, pose) -> None:
    """Teleport the robot to [x, y, theta] at rest (test set-up only)."""
    mujoco.mj_resetData(model, data)
    x, y, theta = pose
    data.qpos[0:3] = [x, y, 0.0]
    data.qpos[3:7] = [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]
    mujoco.mj_forward(model, data)


class _InstantCamera:
    """Renders the robot camera immediately. RGBCamera.read() follows the
    30 Hz schedule and would return a cached frame after a teleport."""

    def __init__(self, model):
        self._renderer = mujoco.Renderer(model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
        self._option = mujoco.MjvOption()
        self._option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self._option.sitegroup[:] = 0

    def capture(self, data) -> CameraReading:
        self._renderer.update_scene(data, camera=CAMERA_NAME, scene_option=self._option)
        return CameraReading(time=float(data.time), rgb=self._renderer.render().copy())

    def close(self):
        self._renderer.close()


def _true_pose(data) -> np.ndarray:
    """GROUND TRUTH robot pose [x, y, theta]. Evaluation only."""
    return transform_to_pose(make_transform(data.body("base").xmat, data.body("base").xpos))


def _true_range_bearing(data) -> np.ndarray:
    """GROUND TRUTH range and bearing to the marker. Evaluation only."""
    offset = data.body("base").xmat.reshape(3, 3).T @ (data.site("aruco_marker_0").xpos - data.body("base").xpos)
    return np.array([np.hypot(offset[0], offset[1]), np.arctan2(offset[1], offset[0])])


def _sim_intrinsics_and_extrinsic(model):
    with RGBCamera(model) as camera:
        return camera.camera_matrix.copy(), camera.dist_coeffs.copy(), camera.camera_to_base.copy()


# --- test: marker accuracy, LiDAR frame check --------------------------------------------

def run_marker_test(repeats: int, pixel_noise: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    model = _load_scene()
    data = mujoco.MjData(model)
    K, dist, T_base_camera = _sim_intrinsics_and_extrinsic(model)
    detector = ArucoLandmarkDetector(K, dist, T_base_camera)
    camera = _InstantCamera(model)
    folder = OUTPUT_DIR / "marker_test"
    folder.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Marker accuracy: {repeats} noisy images per pose, pixel noise sd {pixel_noise:g} ===")
    print(f"{'pose (x, y, th)':>20} {'true r':>7} {'r bias':>7} {'r sd':>6} {'pred sd':>7} "
          f"{'b bias':>7} {'b sd':>6} {'amb':>5} {'pose err':>8} {'seen':>6}")
    print(f"{'':>20} {'[m]':>7} {'[mm]':>7} {'[mm]':>6} {'[mm]':>7} "
          f"{'[deg]':>7} {'[deg]':>6} {'':>5} {'[mm]':>8} {'':>6}")
    for pose in TEST_POSES:
        _place_robot(model, data, pose)
        truth, true_pose = _true_range_bearing(data), _true_pose(data)
        clean = camera.capture(data)
        r, b, amb, pose_err = [], [], [], []
        for k in range(repeats):
            frame = add_camera_noise(clean, rng, pixel_noise) if pixel_noise > 0 else clean
            obs = [o for o in detector.detect(frame) if o.marker_id == 0]
            if k == 0:
                name = "x{:.1f}_y{:+.1f}_th{:+.2f}.png".format(*pose)
                cv2.imwrite(str(folder / name), cv2.cvtColor(detector.draw(frame, obs), cv2.COLOR_RGB2BGR))
            if obs:
                r.append(obs[0].range_m)
                b.append(obs[0].bearing_rad)
                amb.append(obs[0].ambiguity_ratio)
                pose_err.append(np.linalg.norm(robot_pose_from_marker(obs[0])[:2] - true_pose[:2]))
        label = "({:.1f}, {:+.1f}, {:+.2f})".format(*pose)
        if not r:
            print(f"{label:>20} {truth[0]:7.3f} {'not detected':>30}")
            continue
        r, b = np.array(r), np.array(b)
        # Week 3 wrap-up: sigma_Z ~ Z^2 sigma_px / (f_px s), with an ASSUMED
        # 0.3 px corner noise. Z is taken as the range from the camera.
        predicted = (truth[0] - T_base_camera[0, 3]) ** 2 * 0.3 / (K[0, 0] * MARKER_SIDE_M)
        print(f"{label:>20} {truth[0]:7.3f} {1000 * (r.mean() - truth[0]):+7.1f} {1000 * r.std():6.2f} "
              f"{1000 * predicted:7.2f} {np.degrees(wrap_angle(b - truth[1]).mean()):+7.3f} "
              f"{np.degrees(b.std()):6.3f} {np.mean(amb):5.2f} {1000 * np.mean(pose_err):8.1f} "
              f"{len(r):>2}/{repeats:<3}")
    camera.close()
    print("bias: mean measurement minus truth.  sd: spread over the noisy images.  pred sd: Week 3")
    print("formula with an assumed 0.3 px corner noise.  amb: ambiguity ratio (near 1 = orientation")
    print("unreliable).  pose err: position error of robot_pose_from_marker.")
    print(f"Annotated images: {folder}")


def run_lidar_test(seed: int) -> None:
    rng = np.random.default_rng(seed)
    model = _load_scene()
    data = mujoco.MjData(model)
    _place_robot(model, data, (0.3, 0.6, 0.4))
    points = lidar_points_in_base(add_lidar_noise(read_lidar(data), rng))
    # Evaluation only: put the points on the map with the TRUE pose. If the
    # frame chain is right, they land on the walls and obstacles.
    true_pose = _true_pose(data)
    world = lidar_points_to_world(points, true_pose)

    scale, x0, y0 = 100, -1.3, 2.3
    canvas = np.full((460, 660, 3), 255, np.uint8)
    to_px = lambda x, y: (int((x - x0) * scale), int((y0 - y) * scale))
    station = model.body("charging_station").id
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in (0, station):
            continue
        p = model.geom_pos[g] + (model.body_pos[station] if model.geom_bodyid[g] == station else 0)
        size = model.geom_size[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            cv2.rectangle(canvas, to_px(p[0] - size[0], p[1] + size[1]), to_px(p[0] + size[0], p[1] - size[1]),
                          (150, 150, 150), -1)
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            cv2.circle(canvas, to_px(p[0], p[1]), int(size[0] * scale), (150, 150, 150), -1)
    for px, py in world:
        cv2.circle(canvas, to_px(px, py), 2, (0, 0, 255), -1)
    x, y, th = true_pose
    cv2.circle(canvas, to_px(x, y), 6, (255, 0, 0), -1)
    cv2.line(canvas, to_px(x, y), to_px(x + 0.3 * np.cos(th), y + 0.3 * np.sin(th)), (255, 0, 0), 2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "lidar_on_map.png"
    cv2.imwrite(str(path), canvas)
    print(f"\n=== LiDAR frame check ===\n{len(points)} valid points. Drawn on the map "
          f"(grey obstacles, red scan): {path}")


# --- calibrate: intrinsics (checkerboard) and extrinsic check (marker) -------------------

BOARD_INNER_CORNERS = (9, 6)     # 10 x 7 squares, as in Week 3 Part B
BOARD_SQUARE_M = 0.025
_BOARD_SQUARE_PX = 50


def _make_checkerboard(path: Path) -> tuple[float, float]:
    cols, rows = BOARD_INNER_CORNERS[0] + 1, BOARD_INNER_CORNERS[1] + 1
    s = _BOARD_SQUARE_PX
    image = np.full(((rows + 2) * s, (cols + 2) * s), 255, np.uint8)     # one-square white border
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                image[(r + 1) * s:(r + 2) * s, (c + 1) * s:(c + 2) * s] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    metres_per_px = BOARD_SQUARE_M / s
    return image.shape[1] * metres_per_px / 2, image.shape[0] * metres_per_px / 2


def _rotation_about(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def calibrate_intrinsics(n_views: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Week 3 Parts B and C: checkerboard calibration, scored against the true K."""
    rng = np.random.default_rng(seed)
    model = _load_scene(extra_board=True)
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    model.geom_rgba[floor, 3] = 0.0           # hide the floor so it cannot cut through the board
    model.geom_matid[floor] = -1
    _place_robot(model, data, (0.0, 0.0, 0.0))
    K_true, _, _ = _sim_intrinsics_and_extrinsic(model)
    camera = _InstantCamera(model)
    mocap = model.body("calibration_board").mocapid[0]
    T_world_optical = make_transform(data.cam(CAMERA_NAME).xmat.reshape(3, 3) @ np.diag([1, -1, -1]),
                                     data.cam(CAMERA_NAME).xpos)

    cols, rows = BOARD_INNER_CORNERS
    board_points = np.zeros((cols * rows, 3), np.float32)
    board_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * BOARD_SQUARE_M
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    object_points, image_points = [], []
    for _ in range(3 * n_views):
        if len(image_points) == n_views:
            break
        # Random board pose in front of the camera, facing it, tilted 10-35 deg.
        depth = rng.uniform(0.45, 0.8)
        centre = [rng.uniform(-0.12, 0.12) * depth, rng.uniform(-0.08, 0.08) * depth, depth]
        tilt = _rotation_about(rng.normal(size=3), np.deg2rad(rng.uniform(10, 35)))
        T_world_board = T_world_optical @ make_transform(tilt @ np.diag([1.0, -1.0, -1.0]), centre)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, T_world_board[:3, :3].ravel())
        data.mocap_pos[mocap], data.mocap_quat[mocap] = T_world_board[:3, 3], quat
        mujoco.mj_forward(model, data)
        gray = cv2.cvtColor(camera.capture(data).rgb, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, BOARD_INNER_CORNERS)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        object_points.append(board_points)
        image_points.append(corners)
        if len(image_points) == 1:
            preview = cv2.drawChessboardCorners(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                                                BOARD_INNER_CORNERS, corners, True)
            cv2.imwrite(str(OUTPUT_DIR / "checkerboard_view.png"), preview)
    camera.close()
    if len(image_points) < 5:
        raise RuntimeError(f"Only {len(image_points)} usable checkerboard views.")

    # k3 fixed at zero: poorly constrained for a narrow lens, it otherwise
    # soaks up noise as a large, meaningless value.
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, (CAMERA_WIDTH, CAMERA_HEIGHT), None, None, flags=cv2.CALIB_FIX_K3)
    per_view = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        per_view.append(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=2))))

    print(f"\n=== Intrinsic calibration (Week 3 Parts B and C), {len(image_points)} views ===")
    print(f"RMS reprojection error {rms:.3f} px; per view mean {np.mean(per_view):.3f}, "
          f"worst {np.max(per_view):.3f} px")
    for name, (i, j) in {"fx": (0, 0), "fy": (1, 1), "cx": (0, 2), "cy": (1, 2)}.items():
        print(f"  {name}: calibrated {K[i, j]:7.2f}  true {K_true[i, j]:7.2f}  "
              f"difference {K[i, j] - K_true[i, j]:+5.2f} px")
    print(f"  distortion k1 k2 p1 p2 k3: {np.round(dist.ravel(), 4)} (true: all zero)")
    print(f"Detected corners on the first view: {OUTPUT_DIR / 'checkerboard_view.png'}")
    return K, dist


def check_extrinsic(camera_matrix, dist_coeffs) -> np.ndarray:
    """Estimate T_base_camera from marker views at surveyed robot poses and
    compare it with the mounting pose in the robot XML (Week 2 Part C)."""
    model = _load_scene()
    data = mujoco.MjData(model)
    _, _, T_xml = _sim_intrinsics_and_extrinsic(model)
    # Identity extrinsic, so T_base_marker below is really T_camera_marker.
    detector = ArucoLandmarkDetector(camera_matrix, dist_coeffs, np.eye(4))
    camera = _InstantCamera(model)
    # Close and oblique, which keeps the planar ambiguity small.
    rig_poses = [(4.0, 0.25, -0.35), (4.0, -0.25, 0.35), (3.9, 0.1, -0.2), (3.9, -0.1, 0.2),
                 (4.2, 0.2, -0.45), (4.2, -0.2, 0.45), (3.8, 0.0, 0.0), (4.1, 0.0, 0.15)]
    estimates = []
    for pose in rig_poses:
        _place_robot(model, data, pose)
        obs = [o for o in detector.detect(camera.capture(data)) if o.marker_id == 0]
        if obs:
            T_world_base = make_transform(data.body("base").xmat, data.body("base").xpos)   # rig pose
            estimates.append(invert_transform(T_world_base) @ MARKER_MAP[0]
                             @ invert_transform(obs[0].T_camera_marker))
    camera.close()
    t = np.array([T[:3, 3] for T in estimates])
    U, _, Vt = np.linalg.svd(np.mean([T[:3, :3] for T in estimates], axis=0))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt          # average rotation
    angle = np.degrees(np.arccos(np.clip((np.trace(R.T @ T_xml[:3, :3]) - 1) / 2, -1, 1)))
    print(f"\n=== Extrinsic check from the marker (Week 2 Part C), {len(estimates)} views ===")
    print(f"  camera position, estimated: {np.round(t.mean(axis=0), 4)} m "
          f"(spread {np.round(1000 * t.std(axis=0), 1)} mm)")
    print(f"  camera position, robot XML: {np.round(T_xml[:3, 3], 4)} m")
    print(f"  difference {1000 * np.linalg.norm(t.mean(axis=0) - T_xml[:3, 3]):.1f} mm, {angle:.2f} deg")
    print("  The XML value is used in the pipeline; this is an independent check of it.")
    return make_transform(R, t.mean(axis=0))


def run_sensitivity(seed: int) -> None:
    """How much does an extrinsic error cost? Shift the assumed camera mounting
    and measure the range and bearing error (Week 2 Part C.2 for the camera)."""
    model = _load_scene()
    data = mujoco.MjData(model)
    K, dist, T_xml = _sim_intrinsics_and_extrinsic(model)
    camera = _InstantCamera(model)
    poses = [p for p in TEST_POSES if p not in [(4.0, 0.3, 0.3), (4.4, 0.0, 0.0)]]
    frames = []
    for pose in poses:
        _place_robot(model, data, pose)
        frames.append((camera.capture(data), _true_range_bearing(data)))
    camera.close()

    def errors(T_assumed):
        detector = ArucoLandmarkDetector(K, dist, T_assumed)
        r_err, b_err = [], []
        for frame, truth in frames:
            obs = [o for o in detector.detect(frame) if o.marker_id == 0]
            if obs:
                r_err.append(obs[0].range_m - truth[0])
                b_err.append(wrap_angle(obs[0].bearing_rad - truth[1]))
        return 1000 * np.sqrt(np.mean(np.square(r_err))), np.degrees(np.sqrt(np.mean(np.square(b_err))))

    print("\n=== Extrinsic sensitivity: assumed camera mounting deliberately wrong ===")
    print(f"{'assumed error':>28} {'range rms [mm]':>15} {'bearing rms [deg]':>18}")
    cases = [("none (XML value)", np.zeros(3), 0.0)]
    cases += [(f"forward offset {d * 100:+.0f} cm", np.array([d, 0, 0]), 0.0) for d in (0.02, 0.05, 0.10)]
    cases += [(f"sideways offset {d * 100:+.0f} cm", np.array([0, d, 0]), 0.0) for d in (0.02, 0.05)]
    cases += [(f"yaw {a:+.0f} deg", np.zeros(3), a) for a in (1.0, 3.0, 5.0)]
    for label, shift, yaw in cases:
        T = T_xml.copy()
        T[:3, 3] += shift
        T[:3, :3] = _rotation_about([0, 0, 1], np.deg2rad(yaw)) @ T[:3, :3]
        r_rms, b_rms = errors(T)
        print(f"{label:>28} {r_rms:15.1f} {b_rms:18.2f}")
    print("A forward offset shifts the range almost one for one; a sideways offset or a yaw error")
    print("shifts the bearing. Compare with the sub-pixel intrinsic error from 'calibrate'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Perception utilities and self-test.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("make-marker", help="write world/assets/marker_0.png")
    test = sub.add_parser("test", help="marker accuracy and LiDAR frame check (default)")
    test.add_argument("--repeats", type=int, default=20)
    test.add_argument("--pixel-noise", type=float, default=2.0)
    test.add_argument("--seed", type=int, default=0)
    sub.add_parser("calibrate", help="intrinsic calibration and extrinsic check")
    sub.add_parser("sensitivity", help="effect of a wrong camera extrinsic")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "make-marker":
        print(make_marker_image())
    elif args.command == "calibrate":
        K, dist = calibrate_intrinsics()
        check_extrinsic(K, dist)
    elif args.command == "sensitivity":
        run_sensitivity(seed=0)
    else:
        repeats = getattr(args, "repeats", 20)
        run_marker_test(repeats, getattr(args, "pixel_noise", 2.0), getattr(args, "seed", 0))
        run_lidar_test(getattr(args, "seed", 0))


if __name__ == "__main__":
    main()