"""
Our main file runs the autonomous docking simulation.

1. Build the simulation using YAML spawn poses and sensor settings. Spawn poses
   only place physical objects; they are not passed to robot algorithms.
2. Create perception, a local EKF, navigation and control.
3. Read sensors, explore using timed commands, acquire a dock marker, track coarse/detour trajectories with LQI, reacquire the marker, then dock using PBVS.
   LiDAR checks motion clearance for both controllers.

The GUI shows camera/map panels and a robot-relative 3D preview. Only display
code uses the simulator robot pose to position drawings in the viewer world.
Exploration and docking start automatically when the simulation launches.
"""

import argparse
from contextlib import nullcontext
from dataclasses import replace
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

import cv2
import mujoco
import numpy as np

from .display import SensorDisplay
from .control import MotionController, VelocityCommand, velocity_command_to_wheel_speeds
from .estimation import PoseEstimator, StateEstimate
from .navigation import Navigator
from .perception import Perception, PerceptionOutput
from .scene_config import DEFAULT_CONFIG, EstimationConfig, SceneConfig, load_config
from .sensing import read_noisy_wheel_encoders
from .state_model import DifferentialDriveModel


VIEWER_UPDATE_HZ = 30.0 # Display updates do not need to run at the physics rate


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
    settings: EstimationConfig,
) -> tuple[PoseEstimator, Perception, np.random.Generator]:
    """Create the estimator, perception and encoder noise generator once per run."""

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

    # Scene spawn poses are never passed to estimation or perception.
    estimator = PoseEstimator(
        model=DifferentialDriveModel(),
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

    # Get LiDAR points, marker detections and any valid new robot-relative dock pose.
    output = perception.update(data)
    observation = output.dock_observation
    # Start odometry at the first usable marker, with the robot pose defined as zero.
    if observation is not None:
        if not estimator.initialised:
            estimate = estimator.initialise_dock(encoders, observation.pose_base, observation.covariance)
        else:
            estimate = estimator.correct(observation.pose_base, observation.covariance, observation.time)
        if not estimator.camera_correction_accepted:
            output = replace(output, dock_observation=None)

    # Without a camera correction, estimate is still the encoder prediction above.
    # Return both outputs for navigation and control to use in the next stage.
    return estimate, output


def main() -> None:
    # Small camera images do not need a worker for every CPU core. Leave room
    # for physics and the viewer while OpenCV processes each frame.
    cv2.setNumThreads(min(4, cv2.getNumThreads()))
    # Read the run options from the terminal command.
    parser = argparse.ArgumentParser(description="Run the TurtleBot estimation and docking demo.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--headless", action="store_true", help="Run without a viewer.")
    parser.add_argument("--verbose", action="store_true", help="Include estimated pose, uncertainty and camera counts in logs.")
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

    print(f"Sensing and EKF: {config.estimation.update_hz:g} Hz; autonomous docking.", flush=True)
    print("Odometry starts at marker acquisition. Logs: camera age and simulation speed; --verbose adds pose diagnostics.", flush=True)
    with viewer_context as window:
        # Create these once so each update uses the previous estimate and sensor history.
        estimator, perception, encoder_rng = create_demo(model, config.estimation)
        navigator = Navigator(config.navigation)
        controller = MotionController(config.navigation)
        wheel_ids = [model.actuator(name).id for name in ("wheel_left", "wheel_right")]
        wheel_commands = (0.0, 0.0)
        display = SensorDisplay() if window is not None else None
        last_physics_time = float(data.time)
        last_update_slot = -1 # Allow the first update at t = 0
        next_report_time = 0.0
        last_report_event = None
        last_camera_time = -np.inf
        camera_frames = camera_corrections = 0
        wall_origin = time.perf_counter()
        sim_origin = float(data.time)
        next_viewer_sync = wall_origin
        try:
            while window is None or window.is_running():
                # Sensor scheduling uses simulation time; viewer pacing uses wall-clock time.
                now = float(data.time)
                if duration is not None and now >= duration:
                    break

                # Viewer reset starts a new estimate and clears cached sensor samples.
                if now < last_physics_time:
                    perception.close()
                    estimator, perception, encoder_rng = create_demo(model, config.estimation)
                    navigator = Navigator(config.navigation)
                    controller.reset()
                    wheel_commands = (0.0, 0.0)
                    if display is not None:
                        display.reset()
                    last_update_slot = -1
                    next_report_time = 0.0
                    last_report_event = None
                    last_camera_time = -np.inf
                    camera_frames = camera_corrections = 0
                    wall_origin = time.perf_counter()
                    sim_origin = now
                    next_viewer_sync = wall_origin
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
                    camera_corrections += int(output.dock_observation is not None)
                    if output.dock_observation is not None:
                        last_camera_time = output.dock_observation.time

                    # LQI tracks coarse/detour references. PBVS then commands
                    # velocities from the EKF's relative dock pose, without a reference.
                    request = navigator.motion_request(estimate, output)
                    command = controller.compute_velocity_command(estimate, request)
                    # Check the actual commanded arc against the latest LiDAR.
                    # Navigation chooses detours; this final check can still stop them.
                    allowed = navigator.allow_motion(request, command.v_mps, command.omega_rad_s, output)
                    if not allowed:
                        controller.reset()
                        command = VelocityCommand(estimate.time, 0.0, 0.0)
                    wheel_commands = velocity_command_to_wheel_speeds(command)

                    # Display the same camera image used by perception, with its detections.
                    if window is not None:
                        display.update(window, perception, estimate, output, navigator, data=data)

                    # Report every two seconds, plus phase/stop changes. Changing
                    # PBVS error values must not cause a new line on every update.
                    phase = navigator.phase
                    status = navigator.status
                    stopped = request.stop_requested or not allowed
                    event = (phase, status if stopped else '')
                    if now >= next_report_time or event != last_report_event:
                        speed = (now - sim_origin) / max(time.perf_counter() - wall_origin, 1e-9)
                        camera = f'{now-last_camera_time:.1f}s ago' if np.isfinite(last_camera_time) else 'searching'
                        line = f'{now:6.1f}s  {phase:<9} | {status} | camera {camera} | sim {speed:.2f}x'
                        if args.verbose:
                            x, y, heading = estimate.pose_local
                            sigma = np.sqrt(np.diag(estimate.covariance))
                            line += (f' | pose=({x:+.3f}, {y:+.3f} m, {np.degrees(heading):+.1f} deg)'
                                     f' sigma=({sigma[0]:.3f}, {sigma[1]:.3f} m, {np.degrees(sigma[2]):.1f} deg)'
                                     f' camera={camera_corrections}/{camera_frames}')
                        print(line, flush=True)
                        last_report_event = event
                        next_report_time = now + 2.0

                # Hold the latest command between updates, then advance physics one timestep.
                data.ctrl[:] = 0.0
                data.ctrl[wheel_ids] = wheel_commands
                mujoco.mj_step(model, data)
                # Save the post-step time so a reset during viewer sync is detected next loop.
                last_physics_time = float(data.time)

                if window is not None:
                    # Sync the viewer at 30 Hz of wall time, not every 500 Hz physics step.
                    if time.perf_counter() >= next_viewer_sync:
                        window.sync()
                        next_viewer_sync = time.perf_counter() + 1.0 / VIEWER_UPDATE_HZ
                    # Pace against elapsed time for the whole run. Expensive sensor updates
                    # consume this budget; catch up instead of adding another sleep after them.
                    remaining = wall_origin + (float(data.time) - sim_origin) - time.perf_counter()
                    if remaining > 0:
                        time.sleep(min(remaining, model.opt.timestep))
        except KeyboardInterrupt:
            print("\nSimulation interrupted.", flush=True)
        finally:
            # Stop wheel commands and release the renderer even if the run is interrupted.
            data.ctrl[:] = 0.0
            perception.close()
            if display is not None:
                display.close()

    print(f"Simulation stopped at t={data.time:.3f}s; camera corrections={camera_corrections}/{camera_frames} frames.")


if __name__ == "__main__":
    main()
