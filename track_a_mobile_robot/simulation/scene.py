"""Build the physical scene from YAML; spawn poses never enter autonomy modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..scene_config import SceneConfig, load_config

if TYPE_CHECKING:
    import mujoco


SCENE = Path(__file__).resolve().parents[2] / "world/room_with_charger.xml"


def load_spec(path: str | Path = SCENE, config: SceneConfig | None = None) -> mujoco.MjSpec:
    """Apply scene poses, sensor mounting and the LiDAR clock before compilation."""
    import mujoco

    config = load_config() if config is None else config
    # Load an editable scene so we can apply the config before compiling it.
    spec = mujoco.MjSpec.from_file(str(path))

    # Set the robot and dock world poses, keeping their existing ground heights.
    for name, pose in (("base", config.robot), ("charging_station", config.dock)):
        body = spec.body(name)
        if body is None:
            raise ValueError(f"Scene is missing body {name!r}")
        body.pos = [pose.x, pose.y, body.pos[2]]
        body.quat = pose.quaternion
        # Use our quaternion instead of any alternative orientation in the XML.
        body.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT

    # Convert the configured OpenCV optical frame into MuJoCo's camera frame.
    # OpenCV looks along +Z with +Y down; MuJoCo looks along -Z with +Y up.
    camera = spec.camera("pi_camera_v2")
    camera.pos = config.camera.position_m
    rotation = config.camera.rotation @ np.diag([1.0, -1.0, -1.0])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    camera.quat = quaternion
    camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    camera.fovy = config.camera.fov_y_deg

    # Move the scan rays and LiDAR housing together, keeping the scan horizontal.
    lidar = spec.frame("lidar_mount")
    lidar.pos = config.lidar.position_m
    lidar.quat = config.lidar.quaternion
    lidar.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT

    # Add after XML parsing to avoid duplication by MuJoCo's beam replication.
    # hal.mujoco.read_lidar() uses this clock for the scan acquisition timestamp.
    first_beam = spec.sensor("lidar_range000")
    spec.add_sensor(
        name="lidar_time",
        type=mujoco.mjtSensor.mjSENS_CLOCK,
        nsample=first_beam.nsample,
        interval=first_beam.interval,
    )
    return spec


def load_model(path: str | Path = SCENE, config: SceneConfig | None = None) -> mujoco.MjModel:
    """Compile the configured scene into a simulation model."""
    return load_spec(path, config).compile()
