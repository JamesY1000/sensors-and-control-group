"""Offline perception checks and camera calibration.

Run from the repository root, for example:
    MUJOCO_GL=egl python -m scripts.perception_diagnostics test --repeats 3
    MUJOCO_GL=egl python -m scripts.perception_diagnostics calibrate

These diagnostics deliberately read simulator truth and move test rigs to score
sensor measurements. They are separate from runtime perception and estimation;
importing this module does not execute a check or create a renderer.
"""

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from scripts.generate_marker import main as generate_markers
from track_a_mobile_robot.main import load_spec
from track_a_mobile_robot.perception import (
    MARKER_SIZES_M,
    ArucoLandmarkDetector,
    MarkerObservation,
    add_camera_noise,
    add_lidar_noise,
    invert_transform,
    lidar_points_in_base,
    lidar_points_to_local,
    lidar_transform_from_model,
    make_transform,
    pose_to_transform,
    transform_to_pose,
)
from track_a_mobile_robot.scene_config import DEFAULT_CONFIG, SceneConfig, load_config
from track_a_mobile_robot.sensing import (
    CAMERA_HEIGHT,
    CAMERA_NAME,
    CAMERA_WIDTH,
    CameraReading,
    RGBCamera,
    read_lidar,
)
from track_a_mobile_robot.state_model import wrap_angle


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "perception"
MARKER_DIR = REPO_ROOT / "world" / "assets"


# --- Evaluation-only truth helpers ---------------------------------------------------
# Runtime perception deliberately has no marker map or world poses (see README),
# so these helpers live here, next to the other code that reads simulator truth.

def _fixed_body_transform(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    """World transform of a body, chained through its parents' fixed poses.

    Valid for bodies whose ancestors have no joints (the dock, furniture).
    Needs only the compiled model, so no simulation state is created.
    """
    transform = np.eye(4)
    while body_id != 0:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.body_quat[body_id])
        transform = make_transform(rotation, model.body_pos[body_id]) @ transform
        body_id = int(model.body_parentid[body_id])
    return transform


def marker_map_from_model(model: mujoco.MjModel) -> dict[int, np.ndarray]:
    """TRUE world pose of each marker, from the hidden aruco_marker_<id> sites.

    The sites use OpenCV's ArUco axes (x right, y up, z out of the face), so these
    transforms can be compared directly with T_camera_marker from the detector.
    """
    markers = {}
    for marker_id in MARKER_SIZES_M:
        name = f"aruco_marker_{marker_id}"
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            continue
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.site_quat[site_id])
        T_body_site = make_transform(rotation, model.site_pos[site_id])
        markers[marker_id] = _fixed_body_transform(model, int(model.site_bodyid[site_id])) @ T_body_site
    if not markers:
        raise ValueError("Scene has no aruco_marker_<id> sites for evaluation")
    return markers


def robot_pose_from_marker(observation: MarkerObservation,
                           marker_map: dict[int, np.ndarray]) -> np.ndarray | None:
    """Robot [x, y, heading] in the world from one marker and its TRUE world pose.

    T_world_base = T_world_marker @ inverse(T_base_marker). Evaluation only: it
    scores the full marker pose, including the orientation affected by planar ambiguity.
    """
    T_world_marker = marker_map.get(observation.marker_id)
    if T_world_marker is None:
        return None
    return transform_to_pose(T_world_marker @ invert_transform(observation.T_base_marker))


def lidar_points_to_world(points_base: np.ndarray, pose) -> np.ndarray:
    """Place base-frame points in the world using a pose (here, the TRUE pose)."""
    return lidar_points_to_local(points_base, pose)


# Robot test poses in the station frame: x points out of the dock, yaw pi
# faces the marker. Transform these with the station so changing its spawn
# does not leave the accuracy and calibration rigs at the old room location.
TEST_POSES = [
    (2.9, 0.0, np.pi),
    (2.1, 0.0, np.pi),
    (1.4, 0.0, np.pi),
    (1.4, -0.4, np.pi - 0.15),
    (1.4, 0.4, np.pi + 0.25),
    (0.9, 0.0, np.pi),
    (0.9, -0.3, np.pi + 0.3),     # marker near/outside the field of view
    (0.5, 0.0, np.pi),            # elevated panel partly leaves the image
]


def _station_test_poses(model, poses):
    """Move evaluation-only robot rigs with the installed charger."""
    T_world_station = _fixed_body_transform(model, model.body("charging_station").id)
    return [transform_to_pose(T_world_station @ pose_to_transform(pose)) for pose in poses]


def _load_scene(extra_board: bool = False, config: SceneConfig | None = None) -> mujoco.MjModel:
    if any(not (MARKER_DIR / f"marker_{marker_id}.png").exists() for marker_id in MARKER_SIZES_M):
        generate_markers()
    spec = load_spec(config=config)
    if not extra_board:
        return spec.compile()
    # Calibration scene: the project scene plus a free-floating checkerboard.
    texture = OUTPUT_DIR / "checkerboard.png"
    half_w, half_h = _make_checkerboard(texture)
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


def _true_range_bearing(data, marker_id: int = 0) -> np.ndarray:
    """GROUND TRUTH range and bearing to the marker. Evaluation only."""
    offset = data.body("base").xmat.reshape(3, 3).T @ (data.site(f"aruco_marker_{marker_id}").xpos - data.body("base").xpos)
    return np.array([np.hypot(offset[0], offset[1]), np.arctan2(offset[1], offset[0])])


def _sim_intrinsics_and_extrinsic(model):
    with RGBCamera(model) as camera:
        return camera.camera_matrix.copy(), camera.dist_coeffs.copy(), camera.camera_to_base.copy()


# --- test: marker accuracy, LiDAR frame check --------------------------------------------

def run_marker_test(repeats: int, pixel_noise: float, seed: int, config: SceneConfig | None = None) -> None:
    """Score both marker sizes against simulator truth, without running the EKF."""
    rng = np.random.default_rng(seed)
    model = _load_scene(config=config)
    marker_map = marker_map_from_model(model)
    marker_ids = sorted(set(MARKER_SIZES_M) & set(marker_map))
    data = mujoco.MjData(model)
    K, dist, T_base_camera = _sim_intrinsics_and_extrinsic(model)
    detector = ArucoLandmarkDetector(K, dist, T_base_camera)
    camera = _InstantCamera(model)
    folder = OUTPUT_DIR / "marker_test"
    folder.mkdir(parents=True, exist_ok=True)

    print(f"Marker accuracy: {repeats} noisy images per pose, pixel noise sd {pixel_noise:g}")
    print("Range bias/sd and position error are in mm; bearing bias/sd and yaw error are in degrees.")
    print(f"{'pose (x, y, yaw)':>23} {'ID':>3} {'range m':>8} {'r bias':>8} {'r sd':>7} "
          f"{'b bias':>8} {'b sd':>7} {'XY err':>8} {'yaw err':>8} {'seen':>8} {'reliable':>9}")
    try:
        for pose in _station_test_poses(model, TEST_POSES):
            _place_robot(model, data, pose)
            true_pose = _true_pose(data)
            truth = {marker_id: _true_range_bearing(data, marker_id) for marker_id in marker_ids}
            clean = camera.capture(data)
            results = {marker_id: [] for marker_id in marker_ids}
            for sample in range(repeats):
                frame = add_camera_noise(clean, rng, pixel_noise) if pixel_noise > 0 else clean
                observations = detector.detect(frame)
                if sample == 0:
                    name = "x{:.1f}_y{:+.1f}_yaw{:+.2f}.png".format(*pose)
                    image = cv2.cvtColor(detector.draw(frame, observations), cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(str(folder / name), image):
                        raise RuntimeError(f"Could not write {folder / name}")
                for observation in observations:
                    marker_id = observation.marker_id
                    if marker_id not in results:
                        continue
                    measured_pose = robot_pose_from_marker(observation, marker_map)
                    if measured_pose is None:
                        continue
                    results[marker_id].append((
                        observation.range_m - truth[marker_id][0],
                        wrap_angle(observation.bearing_rad - truth[marker_id][1]),
                        np.linalg.norm(measured_pose[:2] - true_pose[:2]),
                        abs(wrap_angle(measured_pose[2] - true_pose[2])),
                        observation.orientation_reliable,
                    ))
            label = "({:.1f}, {:+.1f}, {:+.2f})".format(*pose)
            for marker_id in marker_ids:
                samples = np.asarray(results[marker_id], dtype=float)
                if not len(samples):
                    print(f"{label:>23} {marker_id:3d} {truth[marker_id][0]:8.3f}  not detected")
                    continue
                print(f"{label:>23} {marker_id:3d} {truth[marker_id][0]:8.3f} "
                      f"{1000 * samples[:, 0].mean():+8.1f} {1000 * samples[:, 0].std():7.2f} "
                      f"{np.degrees(samples[:, 1].mean()):+8.3f} {np.degrees(samples[:, 1].std()):7.3f} "
                      f"{1000 * samples[:, 2].mean():8.1f} {np.degrees(samples[:, 3].mean()):8.2f} "
                      f"{len(samples):3d}/{repeats:<4d} {int(samples[:, 4].sum()):9d}")
    finally:
        camera.close()
    print("Pose errors include raw ambiguous detections; reliable counts show the orientation check.")
    print("These short static trials do not establish a calibrated pose covariance or docking success.")
    print(f"Annotated images: {folder}")


def run_lidar_test(seed: int, config: SceneConfig | None = None) -> None:
    rng = np.random.default_rng(seed)
    model = _load_scene(config=config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = lidar_points_in_base(add_lidar_noise(read_lidar(data), rng), lidar_transform_from_model(model))
    # Evaluation only: put the points on the map with the TRUE pose. If the
    # frame chain is right, they land on the walls and obstacles.
    true_pose = _true_pose(data)
    world = lidar_points_to_world(points, true_pose)

    # A body is static only when its whole ancestor chain is fixed. This
    # includes attached furniture and the dock, but excludes robot parts.
    fixed_bodies = np.ones(model.nbody, dtype=bool)
    for body in range(1, model.nbody):
        fixed_bodies[body] = (
            fixed_bodies[model.body_parentid[body]]
            and model.body_jntnum[body] == 0
            and model.body_mocapid[body] == -1
        )
    box_corners = {}
    wall_corners = []
    corner_signs = np.array([[x, y, z] for x in (-1, 1)
                             for y in (-1, 1) for z in (-1, 1)])
    for g in range(model.ngeom):
        if not fixed_bodies[model.geom_bodyid[g]] or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        rotation = data.geom_xmat[g].reshape(3, 3)
        corners = (corner_signs * model.geom_size[g]) @ rotation.T + data.geom_xpos[g]
        box_corners[g] = corners
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith("wall_"):
            wall_corners.append(corners)
    if not wall_corners:
        raise ValueError("LiDAR map requires fixed wall_* box geometry to determine room bounds")
    wall_xy = np.concatenate(wall_corners)[:, :2]
    lower, upper = wall_xy.min(axis=0) - 0.2, wall_xy.max(axis=0) + 0.2
    scale, x0, y0 = 100, lower[0], upper[1]
    width, height = np.ceil((upper - lower) * scale).astype(int)
    canvas = np.full((height, width, 3), 255, np.uint8)
    to_px = lambda x, y: (int((x - x0) * scale), int((y0 - y) * scale))
    scan_height = float(data.site("lidar_ray000").xpos[2])
    for g in range(model.ngeom):
        if not fixed_bodies[model.geom_bodyid[g]]:
            continue
        # Plot collision shapes only. Detailed dock meshes are visual assets.
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        p, rotation, size = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            corners = box_corners[g]
            # The room boxes are upright (possibly yawed). Their XY hull is
            # the scan slice only if this height intersects the box, leaving
            # the spaces below table tops and chair seats open on the map.
            if not corners[:, 2].min() <= scan_height <= corners[:, 2].max():
                continue
            pixels = np.array([to_px(x, y) for x, y, _ in corners], np.int32)
            cv2.fillConvexPoly(canvas, cv2.convexHull(pixels), (150, 150, 150))
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half_height = abs(rotation[2, 2]) * size[1] + np.linalg.norm(rotation[2, :2]) * size[0]
            if not p[2] - half_height <= scan_height <= p[2] + half_height:
                continue
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
          f"(grey obstacles at scan height {scan_height:.3f} m, red scan): {path}")


# --- calibrate: intrinsics (checkerboard) and extrinsic check (marker) -------------------

BOARD_INNER_CORNERS = (9, 6)     # 10 x 7 squares
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


def calibrate_intrinsics(n_views: int = 25, seed: int = 0,
                         config: SceneConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate using a checkerboard and compare with the true camera matrix."""
    rng = np.random.default_rng(seed)
    model = _load_scene(extra_board=True, config=config)
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    model.geom_rgba[floor, 3] = 0.0           # hide the floor so it cannot cut through the board
    model.geom_matid[floor] = -1
    mujoco.mj_forward(model, data)
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

    print(f"\n=== Intrinsic calibration, {len(image_points)} views ===")
    print(f"RMS reprojection error {rms:.3f} px; per view mean {np.mean(per_view):.3f}, "
          f"worst {np.max(per_view):.3f} px")
    for name, (i, j) in {"fx": (0, 0), "fy": (1, 1), "cx": (0, 2), "cy": (1, 2)}.items():
        print(f"  {name}: calibrated {K[i, j]:7.2f}  true {K_true[i, j]:7.2f}  "
              f"difference {K[i, j] - K_true[i, j]:+5.2f} px")
    print(f"  distortion k1 k2 p1 p2 k3: {np.round(dist.ravel(), 4)} (true: all zero)")
    print(f"Detected corners on the first view: {OUTPUT_DIR / 'checkerboard_view.png'}")
    return K, dist


def check_extrinsic(camera_matrix, dist_coeffs, config: SceneConfig | None = None) -> np.ndarray:
    """Estimate T_base_camera from marker views at surveyed robot poses and
    compare it with the configured mounting pose."""
    model = _load_scene(config=config)
    marker_map = marker_map_from_model(model)
    data = mujoco.MjData(model)
    _, _, T_configured = _sim_intrinsics_and_extrinsic(model)
    # Identity extrinsic, so T_base_marker below is really T_camera_marker.
    detector = ArucoLandmarkDetector(camera_matrix, dist_coeffs, np.eye(4))
    camera = _InstantCamera(model)
    # As close as the 0.40 m marker stays fully in view (about 1.4 m), and oblique,
    # which keeps the planar ambiguity small. Closer, the marker leaves the image.
    rig_poses = _station_test_poses(model, [
        (1.5, -0.3, np.pi - 0.3), (1.5, 0.3, np.pi + 0.3),
        (1.6, -0.15, np.pi - 0.15), (1.6, 0.15, np.pi + 0.15),
        (1.8, -0.4, np.pi - 0.35), (1.8, 0.4, np.pi + 0.35),
        (1.5, 0.0, np.pi), (2.0, 0.2, np.pi + 0.1),
    ])
    estimates = []
    for pose in rig_poses:
        _place_robot(model, data, pose)
        obs = [o for o in detector.detect(camera.capture(data)) if o.marker_id == 0]
        if obs:
            T_world_base = make_transform(data.body("base").xmat, data.body("base").xpos)   # rig pose
            estimates.append(invert_transform(T_world_base) @ marker_map[0]
                             @ invert_transform(obs[0].T_camera_marker))
    camera.close()
    if not estimates:
        raise RuntimeError("No marker views found for the extrinsic check; check charger placement and visibility.")
    t = np.array([T[:3, 3] for T in estimates])
    U, _, Vt = np.linalg.svd(np.mean([T[:3, :3] for T in estimates], axis=0))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt          # average rotation
    angle = np.degrees(np.arccos(np.clip((np.trace(R.T @ T_configured[:3, :3]) - 1) / 2, -1, 1)))
    print(f"\n=== Extrinsic check from the marker, {len(estimates)} views ===")
    print(f"  camera position, estimated: {np.round(t.mean(axis=0), 4)} m "
          f"(spread {np.round(1000 * t.std(axis=0), 1)} mm)")
    print(f"  camera position, configured: {np.round(T_configured[:3, 3], 4)} m")
    print(f"  difference {1000 * np.linalg.norm(t.mean(axis=0) - T_configured[:3, 3]):.1f} mm, {angle:.2f} deg")
    print("  The configured value is used in the pipeline; this is an independent check of it.")
    return make_transform(R, t.mean(axis=0))


def run_sensitivity(config: SceneConfig | None = None) -> None:
    """How much does an extrinsic error cost? Shift the assumed camera mounting
    and measure the range and bearing error."""
    model = _load_scene(config=config)
    data = mujoco.MjData(model)
    K, dist, T_configured = _sim_intrinsics_and_extrinsic(model)
    camera = _InstantCamera(model)
    poses = _station_test_poses(model, TEST_POSES[:-2])
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
        if not r_err:
            raise RuntimeError("No marker detections available for the extrinsic sensitivity check.")
        return 1000 * np.sqrt(np.mean(np.square(r_err))), np.degrees(np.sqrt(np.mean(np.square(b_err))))

    print("\n=== Extrinsic sensitivity: assumed camera mounting deliberately wrong ===")
    print(f"{'assumed error':>28} {'range rms [mm]':>15} {'bearing rms [deg]':>18}")
    cases = [("none (configured value)", np.zeros(3), 0.0)]
    cases += [(f"forward offset {d * 100:+.0f} cm", np.array([d, 0, 0]), 0.0) for d in (0.02, 0.05, 0.10)]
    cases += [(f"sideways offset {d * 100:+.0f} cm", np.array([0, d, 0]), 0.0) for d in (0.02, 0.05)]
    cases += [(f"yaw {a:+.0f} deg", np.zeros(3), a) for a in (1.0, 3.0, 5.0)]
    for label, shift, yaw in cases:
        T = T_configured.copy()
        T[:3, 3] += shift
        T[:3, :3] = _rotation_about([0, 0, 1], np.deg2rad(yaw)) @ T[:3, :3]
        r_rms, b_rms = errors(T)
        print(f"{label:>28} {r_rms:15.1f} {b_rms:18.2f}")
    print("A forward offset shifts the range almost one for one; a sideways offset or a yaw error")
    print("shifts the bearing. Compare with the sub-pixel intrinsic error from 'calibrate'.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline perception evaluation and camera calibration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Scene and sensor settings (default: config/simulation_config.yml).")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("make-marker", help="Regenerate both dock marker textures.")
    for name, help_text in (("test", "Marker accuracy and LiDAR frame check (default)."),
                            ("marker", "Marker accuracy for both marker sizes.")):
        check = sub.add_parser(name, help=help_text)
        check.add_argument("--repeats", type=int, default=20)
        check.add_argument("--pixel-noise", type=float, default=2.0)
        check.add_argument("--seed", type=int, default=0)
    lidar = sub.add_parser("lidar", help="Draw the LiDAR scan on evaluation-only scene geometry.")
    lidar.add_argument("--seed", type=int, default=0)
    calibration = sub.add_parser("calibrate", help="Checkerboard intrinsics and marker extrinsic check.")
    calibration.add_argument("--views", type=int, default=25)
    calibration.add_argument("--seed", type=int, default=0)
    sub.add_parser("sensitivity", help="Measure the effect of deliberately wrong camera mounting.")
    args = parser.parse_args(argv)

    if args.command == "make-marker":
        generate_markers()
        return
    if getattr(args, "repeats", 20) <= 0:
        parser.error("--repeats must be positive")
    if not np.isfinite(getattr(args, "pixel_noise", 2.0)) or getattr(args, "pixel_noise", 2.0) < 0:
        parser.error("--pixel-noise must be finite and non-negative")
    if getattr(args, "seed", 0) < 0:
        parser.error("--seed must be non-negative")
    if getattr(args, "views", 25) < 5:
        parser.error("--views must be at least 5")
    try:
        config = load_config(args.config)
    except ValueError as error:
        parser.error(str(error))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "calibrate":
        K, dist = calibrate_intrinsics(args.views, args.seed, config=config)
        check_extrinsic(K, dist, config=config)
    elif args.command == "sensitivity":
        run_sensitivity(config=config)
    elif args.command == "lidar":
        run_lidar_test(args.seed, config=config)
    else:
        run_marker_test(getattr(args, "repeats", 20), getattr(args, "pixel_noise", 2.0),
                        getattr(args, "seed", 0), config=config)
        if args.command != "marker":
            run_lidar_test(getattr(args, "seed", 0), config=config)


if __name__ == "__main__":
    main()