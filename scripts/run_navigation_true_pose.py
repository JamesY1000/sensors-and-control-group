"""Run navigation and control on the simulator's TRUE pose instead of the EKF.

For testing navigation/control in isolation from estimation drift. Physics,
wheel actuators, LiDAR and camera perception are the same as main.py; only the
pose given to the Navigator and LQIController comes from ground truth.

Run from the repository root:
    python -m scripts.run_navigation_true_pose                 # MuJoCo viewer
    python -m scripts.run_navigation_true_pose --headless      # no window, stops when docked
On Linux without a display, prefix with MUJOCO_GL=egl.
"""

import argparse
import math
from pathlib import Path
import time as wall_clock

import mujoco
import numpy as np

from track_a_mobile_robot.control import LQIController, velocity_command_to_wheel_speeds
from track_a_mobile_robot.estimation import StateEstimate
from track_a_mobile_robot.evaluation import read_true_pose
from track_a_mobile_robot.main import SCENE, load_model
from track_a_mobile_robot.navigation import Navigator
from track_a_mobile_robot.perception import Perception
from track_a_mobile_robot.scene_config import DEFAULT_CONFIG, load_config

CONTROL_HZ = 50


def true_state(data: mujoco.MjData) -> StateEstimate:
    """Ground-truth pose wrapped as a StateEstimate with zero covariance."""
    return StateEstimate(
        time=float(data.time),
        pose_world=read_true_pose(data),
        covariance=np.zeros((3, 3)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--headless", action="store_true", help="Run without the viewer.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Robot and charger starting poses in TOML (default: config/scene.toml).",
    )
    parser.add_argument(
        "--seconds", type=float, default=300.0,
        help="Headless time limit in simulated seconds (default: 300).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    try:
        config = load_config(args.config)
    except ValueError as error:
        parser.error(str(error))

    model = load_model(SCENE, config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    control_period = 1.0 / CONTROL_HZ

    with Perception(model) as perception:
        state = {"navigator": Navigator(marker_map=perception.marker_map), "controller": LQIController(),
                 "next_control": 0.0, "last_time": -math.inf, "done": False}

        def update() -> None:
            now = float(data.time)
            if now < state["last_time"]:
                # Viewer reset: start the mission again.
                state.update(navigator=Navigator(marker_map=perception.marker_map), controller=LQIController(),
                             next_control=now, done=False)
            state["last_time"] = now
            perception_output = perception.update(data)
            if now + 1e-9 < state["next_control"]:
                return
            estimate = true_state(data)
            reference = state["navigator"].trajectory_reference(estimate, perception_output)
            command = state["controller"].compute_velocity_command(estimate, reference)
            left, right = velocity_command_to_wheel_speeds(command)
            data.ctrl[model.actuator("wheel_left").id] = left
            data.ctrl[model.actuator("wheel_right").id] = right
            state["done"] = reference.stop_requested
            state["next_control"] = (
                math.floor(now / control_period + 1e-9) + 1
            ) * control_period

        if args.headless:
            start = wall_clock.time()
            update()
            while data.time < args.seconds and not state["done"]:
                mujoco.mj_step(model, data)
                update()
            x, y, yaw = read_true_pose(data)
            outcome = "docked" if state["done"] else f"not finished ({state['navigator'].phase})"
            print(
                f"{outcome} at t={data.time:.1f} s sim ({wall_clock.time() - start:.0f} s wall); "
                f"true pose=({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f} deg)",
                flush=True,
            )
            return

        from mujoco import viewer

        previous_callback = mujoco.get_mjcb_control()

        def viewer_update(callback_model, callback_data):
            if previous_callback is not None:
                previous_callback(callback_model, callback_data)
            if callback_model is model and callback_data is data:
                update()

        try:
            mujoco.set_mjcb_control(viewer_update)
            viewer.launch(model, data)
        finally:
            mujoco.set_mjcb_control(previous_callback)


if __name__ == "__main__":
    main()
