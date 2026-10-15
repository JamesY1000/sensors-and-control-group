"""Starting poses for the robot and fixed dock assembly, in metres and degrees."""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/scene.toml"


@dataclass(frozen=True)
class SpawnPose:
    x: float
    y: float
    yaw_deg: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "yaw_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")

    @property
    def pose_world(self) -> tuple[float, float, float]:
        """Planar pose for estimation, with yaw converted to radians."""
        return self.x, self.y, math.radians(self.yaw_deg)

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        """MuJoCo's (w, x, y, z) quaternion for rotation about world Z."""
        half_yaw = math.radians(self.yaw_deg) / 2
        return math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)


@dataclass(frozen=True)
class SceneConfig:
    robot: SpawnPose
    charger: SpawnPose


def load_config(path: str | Path = DEFAULT_CONFIG) -> SceneConfig:
    """Read a complete scene configuration; catch misspelled settings early."""
    path = Path(path)
    try:
        with path.open("rb") as source:
            values = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Cannot read scene config {path}: {error}") from error

    if set(values) != {"robot", "charger"}:
        raise ValueError(f"{path}: expected exactly [robot] and [charger] sections")
    poses = {}
    for name in ("robot", "charger"):
        section = values[name]
        if not isinstance(section, dict) or set(section) != {"x", "y", "yaw_deg"}:
            raise ValueError(f"{path}: [{name}] must contain exactly x, y and yaw_deg")
        try:
            poses[name] = SpawnPose(**section)
        except ValueError as error:
            raise ValueError(f"{path}: [{name}] {error}") from error
    return SceneConfig(**poses)
