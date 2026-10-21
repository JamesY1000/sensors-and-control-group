"""
Our main file should essentially set up the simulation and connect sensing,
perception, state estimation, navigation and control.

It has 3 main stages:

1. Set up the simulation:
    - Load the robot/dock starting poses and sensor settings from the YAML config
    - Apply these settings to the scene and create the MuJoCo model and data
    - Open the viewer, or run without a window in headless mode

2. Set up the estimation objects:
    - Create the differential-drive model, pose estimator and perception objects
    - Initialise the estimated pose from our assumed starting pose in the config
    - Set the initial covariance, process noise and repeatable sensor noise
    - Keep these objects between updates so the estimator retains its state
    - Create navigation and control when autonomous driving is enabled

3. Run the simulation loop:
    - Read wheel encoders and predict the robot pose and covariance
    - Process perception and correct using a valid, fresh camera-derived pose
    - Run these together at the configured update rate, with physics stepping between updates
    - Print the estimated world pose, uncertainty and camera correction count
    - Show the live camera feed with marker outlines and correction status in the viewer
    - Generate a trajectory from the estimated pose and track it with wheel commands
    - Restart estimation if the simulation is reset, and close the camera when finished

If no valid camera pose is available, the estimate will just be the encoder prediction.
Use --navigate to drive autonomously; otherwise the wheels stay stopped for sensor checks.
"""

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
import time

# Use the working XWayland backend on this Linux desktop.
if sys.platform == "linux" and os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

# Keep the offscreen camera separate from the viewer's GLFW context on Linux.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .display import update_camera_preview
from .control import LQIController, velocity_command_to_wheel_speeds
from .estimation import PoseEstimator, StateEstimate
from .navigation import Navigator
from .perception import Perception, PerceptionOutput
from .scene_config import DEFAULT_CONFIG, SceneConfig, load_config
from .sensing import read_noisy_wheel_encoders
from .state_model import DifferentialDriveModel


SCENE = Path(__file__).resolve().parents[1] / "world/room_with_charger.xml"


def load_spec(path: str | Path = SCENE, config: SceneConfig | None = None) -> mujoco.MjSpec:
    """Apply scene poses, sensor mounting and the LiDAR clock before compilation."""
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
    # sensing.read_lidar() uses this clock for the scan acquisition timestamp.
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


def create_demo(
    model: mujoco.MjModel,
    config: SceneConfig,
) -> tuple[PoseEstimator, Perception, np.random.Generator]:
    """Create the estimator, perception and encoder noise generator once per run."""
    settings = config.estimation

    # Starting uncertainty in [x, y, heading]. Convert the heading value to radians.
    initial_std = np.array([
        settings.initial_position_std_m,
        settings.initial_position_std_m,
        np.deg2rad(settings.initial_heading_std_deg),
    ])
    # Additional motion-model uncertainty, expressed per square root of a second.
    # Squaring these values gives the process covariance added per second.
    process_std = np.array([
        settings.process_position_std_m_per_sqrt_s,
        settings.process_position_std_m_per_sqrt_s,
        np.deg2rad(settings.process_heading_std_deg_per_sqrt_s),
    ])
    # Use the same encoder noise levels that sensing applies to the wheel readings.
    encoder_std = tuple(
        float(model.sensor_noise[model.sensor(name).id])
        for name in ("wheel_left_encoder", "wheel_right_encoder")
    )

    # Square the standard deviations to get the diagonal entries of P0 and Q_rate.
    estimator = PoseEstimator(
        model=DifferentialDriveModel(),
        initial_pose=config.robot.pose_world, # Our assumed starting pose from the config
        initial_covariance=np.diag(initial_std**2),
        encoder_noise_std_rad=encoder_std,
        process_noise_covariance_per_s=np.diag(process_std**2),
    )
    # Separate streams keep encoder noise independent of camera/LiDAR sampling.
    encoder_seed, perception_seed = np.random.SeedSequence(settings.noise_seed).spawn(2)
    encoder_rng = np.random.default_rng(encoder_seed)
    perception = Perception(model, noise=True, rng=np.random.default_rng(perception_seed))
    return estimator, perception, encoder_rng


def update_demo(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    estimator: PoseEstimator,
    perception: Perception,
    encoder_rng: np.random.Generator,
) -> tuple[StateEstimate, PerceptionOutput]:
    """Run prediction and any camera correction at one simulation timestamp."""
    # Predict the current pose and covariance using this update's encoder readings.
    encoders = read_noisy_wheel_encoders(model, data, encoder_rng)
    estimate = estimator.predict(encoders)

    # Get LiDAR points, marker detections and any valid new camera-derived robot pose.
    output = perception.update(data)
    observation = output.pose_observation
    # Correct only when perception supplies a fresh pose at this prediction's timestamp.
    if observation is not None:
        estimate = estimator.correct(
            observation.pose_world, observation.covariance, observation.time,
        )

    # Without a camera correction, estimate is still the encoder prediction above.
    # Return both outputs for navigation and control to use in the next stage.
    return estimate, output


def main() -> None:
    # Read the run options from the terminal command.
    parser = argparse.ArgumentParser(description="Run the TurtleBot estimation and docking demo.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--headless", action="store_true", help="Run without a viewer.")
    parser.add_argument("--navigate", action="store_true", help="Enable autonomous approach and docking.")
    parser.add_argument(
        "--seconds", type=float,
        help="Simulation duration; defaults to 2 s headless, or until the viewer closes.",
    )
    args = parser.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite")
    duration = args.seconds
    # Give headless runs a default duration; viewer runs can continue until closed.
    if args.headless and duration is None:
        duration = 2.0

    # Load our settings and build the configured simulation model.
    try:
        config = load_config(args.config)
    except ValueError as error:
        parser.error(str(error))
    model = load_model(config=config)
    if config.estimation.update_hz > 1.0 / model.opt.timestep:
        parser.error("estimation.update_hz cannot exceed the physics update rate")
    # MjData holds the changing simulation state. Refresh sensors before the first read.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Use the same loop in both modes; window will be None when running headless.
    viewer_context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer

        viewer_context = viewer.launch_passive(model, data)

    mode = "autonomous docking" if args.navigate else "stationary sensor demo"
    print(f"Sensing and EKF: {config.estimation.update_hz:g} Hz; {mode}.", flush=True)
    print("Pose is in the world frame; sigma is one standard deviation. Camera count is corrections/frames.", flush=True)
    with viewer_context as window:
        # Create these once so each update uses the previous estimate and sensor history.
        estimator, perception, encoder_rng = create_demo(model, config)
        navigator = Navigator(config.dock.pose_world, config.navigation) if args.navigate else None
        controller = LQIController()
        wheel_ids = [model.actuator(name).id for name in ("wheel_left", "wheel_right")]
        wheel_commands = (0.0, 0.0)
        last_physics_time = float(data.time)
        last_update_slot = -1 # Allow the first update at t = 0
        next_report_time = 0.0
        camera_frames = camera_corrections = 0
        try:
            while window is None or window.is_running():
                # Sensor scheduling uses simulation time; viewer pacing uses wall-clock time.
                now = float(data.time)
                if duration is not None and now >= duration:
                    break
                step_start = time.perf_counter()

                # Viewer reset starts a new estimate and clears cached sensor samples.
                if now < last_physics_time:
                    perception.close()
                    estimator, perception, encoder_rng = create_demo(model, config)
                    navigator = Navigator(config.dock.pose_world, config.navigation) if args.navigate else None
                    controller.reset()
                    wheel_commands = (0.0, 0.0)
                    last_update_slot = -1
                    next_report_time = 0.0
                    camera_frames = camera_corrections = 0
                    print("Simulation reset; estimation, sensor sampling and navigation restarted.", flush=True)

                # At 50 Hz, the slot changes every 0.02 simulated seconds.
                # Update sensing and estimation once per slot so their timestamps match.
                # The small tolerance handles floating-point rounding at a slot boundary.
                update_slot = math.floor(now * config.estimation.update_hz + 1e-9)
                if update_slot != last_update_slot:
                    # Refresh sensors at the current state before reading encoders and the camera.
                    mujoco.mj_forward(model, data)
                    estimate, output = update_demo(model, data, estimator, perception, encoder_rng)
                    last_update_slot = update_slot
                    camera_frames += int(output.new_camera_frame)
                    camera_corrections += int(output.pose_observation is not None)

                    # Navigation supplies the reference; control uses only our estimated pose.
                    if navigator is not None:
                        reference = navigator.trajectory_reference(estimate, output)
                        command = controller.compute_velocity_command(estimate, reference)
                        wheel_commands = velocity_command_to_wheel_speeds(command)

                    # Display the same camera image used by perception, with its detections.
                    if window is not None:
                        update_camera_preview(window, perception, output)

                    # Report once per simulated second, rather than on every estimator update.
                    if now >= next_report_time:
                        x, y, heading = estimate.pose_world
                        # The diagonal of P contains variances; square roots give standard deviations.
                        sigma = np.sqrt(np.diag(estimate.covariance))
                        print(
                            f"t={now:6.2f}s  pose=({x:+.3f} m, {y:+.3f} m, {np.degrees(heading):+.2f} deg)  "
                            f"sigma=({sigma[0]:.3f} m, {sigma[1]:.3f} m, {np.degrees(sigma[2]):.2f} deg)  "
                            f"camera={camera_corrections}/{camera_frames}"
                            + (f"  {navigator.phase}: {navigator.status}" if navigator else ""),
                            flush=True,
                        )
                        next_report_time = now + 1.0

                # Hold the latest command between updates, then advance physics one timestep.
                data.ctrl[:] = 0.0
                data.ctrl[wheel_ids] = wheel_commands
                mujoco.mj_step(model, data)
                # Save the post-step time so a reset during viewer sync is detected next loop.
                last_physics_time = float(data.time)

                if window is not None:
                    window.sync()
                    # Pace the viewer approximately in real time; headless runs do not wait.
                    remaining = model.opt.timestep - (time.perf_counter() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            print("\nSimulation interrupted.", flush=True)
        finally:
            # Stop wheel commands and release the renderer even if the run is interrupted.
            data.ctrl[:] = 0.0
            perception.close()

    print(f"Simulation stopped at t={data.time:.3f}s; camera corrections={camera_corrections}/{camera_frames} frames.")


if __name__ == "__main__":
    main()
