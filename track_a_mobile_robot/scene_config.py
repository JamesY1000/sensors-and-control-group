"""Load scene, sensor, estimation and navigation settings from one YAML file."""

from dataclasses import dataclass, fields
import math
from pathlib import Path

import numpy as np
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/simulation_config.yml"


@dataclass(frozen=True)
class SpawnPose:
    """Store a planar world pose in metres and degrees."""

    x: float
    y: float
    yaw_deg: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "yaw_deg"):
            _number(getattr(self, name), name)

    @property
    def pose_world(self) -> tuple[float, float, float]:
        """Return [x, y, heading] with heading in radians."""
        return self.x, self.y, math.radians(self.yaw_deg)

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        """Return MuJoCo's (w, x, y, z) rotation about world Z."""
        half_yaw = math.radians(self.yaw_deg) / 2
        return math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)


@dataclass(frozen=True)
class CameraConfig:
    """Store the optical frame's mounting pose and vertical field of view."""

    position_m: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    fov_y_deg: float

    @property
    def rotation(self) -> np.ndarray:
        """Rotate OpenCV optical coordinates into the robot base frame."""
        roll, pitch, yaw = np.deg2rad(self.rpy_deg)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx


@dataclass(frozen=True)
class LidarConfig:
    """Store the position and yaw of a horizontal LiDAR scan."""

    position_m: tuple[float, float, float]
    yaw_deg: float

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        half_yaw = math.radians(self.yaw_deg) / 2
        return math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)


@dataclass(frozen=True)
class EstimationConfig:
    """Store the update rate, noise seed and process uncertainty assumptions."""

    update_hz: float
    noise_seed: int
    process_position_std_m_per_sqrt_s: float
    process_heading_std_deg_per_sqrt_s: float

    def __post_init__(self) -> None:
        if _number(self.update_hz, "estimation.update_hz") <= 0:
            raise ValueError("estimation.update_hz must be positive")
        if isinstance(self.noise_seed, bool) or not isinstance(self.noise_seed, int) or self.noise_seed < 0:
            raise ValueError("estimation.noise_seed must be a non-negative integer")
        for name in ("process_position_std_m_per_sqrt_s", "process_heading_std_deg_per_sqrt_s"):
            if _number(getattr(self, name), f"estimation.{name}") < 0:
                raise ValueError(f"estimation.{name} must be non-negative")


@dataclass(frozen=True)
class NavigationConfig:
    """Store dock offsets, tracking limits and observation requirements."""

    coarse_distance_m: float
    coarse_tolerance_m: float
    pbvs_speed_mps: float
    approach_distance_m: float
    dock_distance_m: float
    drive_speed_mps: float
    dock_speed_mps: float
    turn_speed_rad_s: float
    clearance_m: float
    position_tolerance_m: float
    heading_tolerance_deg: float
    lidar_timeout_s: float
    search_spin_s: float
    search_drive_s: float
    reacquire_heading_tolerance_deg: float
    reacquire_observation_age_s: float
    reacquire_confirmations: int
    reacquire_wait_s: float
    reacquire_sweep_deg: float
    reacquire_timeout_s: float
    progress_timeout_s: float
    boundary_join_gap_m: float
    detour_step_m: float
    detour_speed_mps: float
    detour_timeout_s: float

    def __post_init__(self) -> None:
        for field in fields(self):
            if _number(getattr(self, field.name), f"navigation.{field.name}") <= 0:
                raise ValueError(f"navigation.{field.name} must be positive")
        if self.coarse_distance_m <= self.approach_distance_m:
            raise ValueError("Coarse distance must exceed PBVS approach distance")
        if self.pbvs_speed_mps > self.drive_speed_mps:
            raise ValueError("PBVS approach speed must not exceed driving speed")
        if self.approach_distance_m <= self.dock_distance_m:
            raise ValueError("Approach distance must exceed docking distance")
        if self.dock_speed_mps > self.drive_speed_mps:
            raise ValueError("Docking speed must not exceed approach speed")
        if self.heading_tolerance_deg >= 90:
            raise ValueError("Heading tolerance must be less than 90 degrees")
        if not self.heading_tolerance_deg <= self.reacquire_heading_tolerance_deg < 90:
            raise ValueError("Reacquisition heading tolerance must be between docking tolerance and 90 degrees")
        if type(self.reacquire_confirmations) is not int or self.reacquire_confirmations < 1:
            raise ValueError("Reacquisition requires at least one camera observation")
        if self.reacquire_sweep_deg >= 90:
            raise ValueError("Reacquisition sweep must be less than 90 degrees on each side")
        if self.reacquire_timeout_s <= self.reacquire_wait_s:
            raise ValueError("Reacquisition timeout must exceed the camera wait")
        if self.detour_speed_mps > self.drive_speed_mps:
            raise ValueError("Detour speed must not exceed approach speed")


@dataclass(frozen=True)
class SceneConfig:
    robot: SpawnPose
    dock: SpawnPose
    camera: CameraConfig
    lidar: LidarConfig
    estimation: EstimationConfig
    navigation: NavigationConfig


def _number(value, name: str) -> float:
    """Reject missing, non-numeric or non-finite parameter values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(values, name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name} must contain three numbers")
    return tuple(_number(value, name) for value in values)


def _section(values: dict, name: str, fields: set[str]) -> dict:
    section = values[name]
    if not isinstance(section, dict) or set(section) != fields:
        raise ValueError(f"{name}: expected exactly {', '.join(sorted(fields))}")
    return section


def load_config(path: str | Path = DEFAULT_CONFIG) -> SceneConfig:
    """Read the YAML settings and catch missing or misspelled fields."""
    path = Path(path)
    try:
        with path.open() as source:
            values = yaml.safe_load(source)
        if not isinstance(values, dict) or set(values) != {"robot", "dock", "camera", "lidar", "estimation", "navigation"}:
            raise ValueError("Expected exactly robot, dock, camera, lidar, estimation and navigation sections")

        robot = SpawnPose(**_section(values, "robot", {"x", "y", "yaw_deg"}))
        dock = SpawnPose(**_section(values, "dock", {"x", "y", "yaw_deg"}))
        camera = _section(values, "camera", {"position_m", "rpy_deg", "fov_y_deg"})
        lidar = _section(values, "lidar", {"position_m", "yaw_deg"})
        estimation = _section(values, "estimation", {
            "update_hz", "noise_seed",
            "process_position_std_m_per_sqrt_s", "process_heading_std_deg_per_sqrt_s",
        })
        fov_y_deg = _number(camera["fov_y_deg"], "camera.fov_y_deg")
        if not 0 < fov_y_deg < 180:
            raise ValueError("camera.fov_y_deg must be between 0 and 180 degrees")

        return SceneConfig(
            robot=robot,
            dock=dock,
            camera=CameraConfig(
                position_m=_vector(camera["position_m"], "camera.position_m"),
                rpy_deg=_vector(camera["rpy_deg"], "camera.rpy_deg"),
                fov_y_deg=fov_y_deg,
            ),
            lidar=LidarConfig(
                position_m=_vector(lidar["position_m"], "lidar.position_m"),
                yaw_deg=_number(lidar["yaw_deg"], "lidar.yaw_deg"),
            ),
            estimation=EstimationConfig(**estimation),
            navigation=NavigationConfig(**_section(values, "navigation", {
                field.name for field in fields(NavigationConfig)
            })),
        )
    except (OSError, yaml.YAMLError, ValueError) as error:
        raise ValueError(f"Cannot read scene config {path}: {error}") from error
