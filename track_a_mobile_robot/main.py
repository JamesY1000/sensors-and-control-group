"""Open the default TurtleBot3 scene: python -m track_a_mobile_robot.main."""

# Set everything up, schedule processing, pass data between components and apply commands to MuJoCo.

import argparse
from contextlib import ExitStack
import math
import os
from pathlib import Path
import sys

# Use XWayland on Linux desktops with DISPLAY available. Native Wayland GLFW
# failed to initialise window decorations on this development machine.
# Set PYGLFW_LIBRARY_VARIANT explicitly if another backend is preferred.
if sys.platform == "linux" and os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import numpy as np

from .estimation import PoseEKF, StateEstimate
from .evaluation import pose_error, read_true_pose
from .scene_config import DEFAULT_CONFIG, SceneConfig, load_config
from .sensing import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    LidarReading,
    RGBCamera,
    read_absolute_wheel_encoders,
    read_lidar,
    read_noisy_wheel_encoders,
)
from .state_model import DifferentialDriveModel


SCENE = Path(__file__).resolve().parents[1] / "world/room_with_charger.xml"
# Chosen encoder sampling rate for pose prediction.
ENCODER_HZ = 50
# Starting assumptions, not calibrated values: [x metres, y metres, yaw radians].
INITIAL_POSE_STD = (0.01, 0.01, math.radians(2.0))
# Motion-model uncertainty growth [m/sqrt(s), m/sqrt(s), rad/sqrt(s)].
PROCESS_NOISE_STD_PER_SQRT_S = (0.01, 0.01, math.radians(5.0))


def load_spec(path: str | Path = SCENE, config: SceneConfig | None = None) -> mujoco.MjSpec:
    """Apply configured starting poses and add the LiDAR sampling clock.

    MuJoCo 3.15 recursively duplicates an unreferenced XML clock during
    replication. Adding it after parsing avoids that model-loading problem.
    The rangefinders and their sampling interval remain defined in robot XML.
    Applying poses before compilation also makes viewer resets restore them.
    """
    config = load_config() if config is None else config
    spec = mujoco.MjSpec.from_file(str(path))
    for name, pose in (("base", config.robot), ("charging_station", config.charger)):
        body = spec.body(name)
        if body is None:
            raise ValueError(f"Scene is missing body {name!r}")
        # Preserve the model's ground height; only planar placement is configurable.
        body.pos = [pose.x, pose.y, body.pos[2]]
        body.quat = pose.quaternion
        body.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    first_beam = spec.sensor("lidar_range000")
    spec.add_sensor(
        name="lidar_time",
        type=mujoco.mjtSensor.mjSENS_CLOCK,
        nsample=first_beam.nsample,
        interval=first_beam.interval,
    )
    return spec


def load_model(path: str | Path = SCENE, config: SceneConfig | None = None) -> mujoco.MjModel:
    """Compile the scene with the supplied configuration (or config/scene.toml)."""
    return load_spec(path, config).compile()


def print_lidar(reading: LidarReading) -> None:
    """Show a compact scan summary instead of printing all 360 distances."""
    ranges = np.asarray(reading.ranges_m)
    valid = ranges[np.isfinite(ranges)]
    nearest = f"{valid.min():.3f} m" if valid.size else "none"

    def distance(index: int) -> str:
        value = ranges[index]
        if np.isnan(value):
            return "too close"
        if np.isinf(value):
            return "no return"
        return f"{value:.3f} m"

    print(
        f"LiDAR t={reading.time:.3f} s: {valid.size}/{len(ranges)} valid, "
        f"{np.isnan(ranges).sum()} too close; nearest={nearest}; "
        f"front={distance(0)}, left={distance(90)}, "
        f"back={distance(180)}, right={distance(270)}",
        flush=True,
    )


def print_estimate(estimate: StateEstimate, data: mujoco.MjData) -> None:
    """Display the estimate and a separate evaluation-only truth comparison."""
    x, y, yaw = estimate.pose_world
    true_pose = read_true_pose(data)
    position_error, heading_error = pose_error(estimate.pose_world, true_pose)
    std = np.sqrt(np.maximum(np.diag(estimate.covariance), 0.0))
    print(
        f"Estimate t={estimate.time:.3f} s: x={x:.4f} m, y={y:.4f} m, "
        f"yaw={yaw:.4f} rad; std=({std[0]:.4f} m, {std[1]:.4f} m, "
        f"{std[2]:.4f} rad); evaluation truth="
        f"({true_pose[0]:.4f}, {true_pose[1]:.4f}, {true_pose[2]:.4f}); "
        f"position error={position_error:.4f} m, heading error={heading_error:.4f} rad",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the TurtleBot3 world with encoders, LiDAR and an RGB camera.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Scene starting poses in TOML (default: config/scene.toml in the repository).",
    )
    parser.add_argument("--headless", action="store_true", help="Check physics without opening a viewer.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Duration of a headless check (default: 2 s).")
    parser.add_argument("--left-wheel", type=float, default=0.0, help="Left wheel velocity target [rad/s].")
    parser.add_argument("--right-wheel", type=float, default=0.0, help="Right wheel velocity target [rad/s].")
    parser.add_argument("--seed", type=int, default=42, help="Encoder noise seed (default: 42).")
    parser.add_argument(
        "--ideal-encoders", action="store_true",
        help="Use clean encoder readings for prediction instead of the configured Gaussian noise.",
    )
    parser.add_argument(
        "--camera-output", type=Path,
        help=f"Sample RGB at {CAMERA_FPS} Hz during a headless run and save its latest frame as a PNG.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.camera_output is not None and not args.headless:
        parser.error("--camera-output requires --headless")

    rng = np.random.default_rng(args.seed)
    try:
        config = load_config(args.config)
    except ValueError as error:
        parser.error(str(error))
    model = load_model(SCENE, config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name, target in (("wheel_left", args.left_wheel), ("wheel_right", args.right_wheel)):
        actuator = model.actuator(name)
        lower, upper = actuator.ctrlrange
        if not math.isfinite(target) or not lower <= target <= upper:
            parser.error(f"{name} target must be between {lower:g} and {upper:g} rad/s")
        data.ctrl[actuator.id] = target

    print(f"Loaded: {SCENE}", flush=True)
    print(f"Scene config: {args.config.resolve()}", flush=True)
    for name, pose in (("Robot", config.robot), ("Charger + marker", config.charger)):
        print(f"{name} start: x={pose.x:g} m, y={pose.y:g} m, yaw={pose.yaw_deg:g} deg", flush=True)
    print(f"Wheel actuators: {model.nu}; sensor definitions: {model.nsensor}; "
          f"robot cameras: {model.ncam}", flush=True)
    print(f"Wheel targets [rad/s]: left={args.left_wheel:g}, right={args.right_wheel:g}", flush=True)
    print(
        f"EKF prediction: {ENCODER_HZ} Hz, {'ideal' if args.ideal_encoders else 'noisy'} encoders; "
        f"noise seed={args.seed}", flush=True,
    )
    print("Camera pose correction is available but perception is not connected yet.", flush=True)
    print(f"Initial ideal encoders: {read_absolute_wheel_encoders(data)}", flush=True)

    drive_model = DifferentialDriveModel()
    encoder_std = tuple(
        0.0 if args.ideal_encoders else float(model.sensor_noise[model.sensor(name).id])
        for name in ("wheel_left_encoder", "wheel_right_encoder")
    )

    def new_estimator() -> PoseEKF:
        """Use the same initial pose and uncertainty after a viewer reset."""
        return PoseEKF(
            drive_model, config.robot.pose_world, np.diag(np.square(INITIAL_POSE_STD)),
            encoder_noise_std_rad=encoder_std,
            process_noise_covariance_per_s=np.diag(np.square(PROCESS_NOISE_STD_PER_SQRT_S)),
        )

    estimator = new_estimator()
    last_encoders = None
    estimate = None
    last_encoder_slot = -1
    last_seen_time = -math.inf
    next_report_time = 0.0

    def sample_estimator(force: bool = False) -> None:
        """Sample once per encoder period and predict pose plus covariance."""
        nonlocal estimator, last_encoders, estimate, last_encoder_slot
        nonlocal last_seen_time, next_report_time
        now = float(data.time)
        if now < last_seen_time:
            # The viewer reset the scene: start a new encoder baseline.
            estimator = new_estimator()
            last_encoders = None
            last_encoder_slot = -1
            next_report_time = 0.0
        last_seen_time = now
        slot = math.floor(now * ENCODER_HZ + 1e-9)
        if slot == last_encoder_slot:
            if not force or last_encoders is None or now == last_encoders.time:
                return
        if args.ideal_encoders:
            last_encoders = read_absolute_wheel_encoders(data)
        else:
            last_encoders = read_noisy_wheel_encoders(model, data, rng)
        estimate = estimator.predict(last_encoders)
        last_encoder_slot = slot
        if now + 1e-9 >= next_report_time:
            print_estimate(estimate, data)
            next_report_time = math.floor(now) + 1.0

    sample_estimator()
    scan = read_lidar(data)
    print_lidar(scan)

    if args.headless:
        steps = math.ceil(args.seconds / model.opt.timestep)
        last_scan_time = scan.time
        # A physics-only check still needs no graphics context. Rendering is opt-in.
        with ExitStack() as stack:
            camera = stack.enter_context(RGBCamera(model)) if args.camera_output is not None else None
            frame = camera.read(data) if camera is not None else None
            frame_count = 1 if frame is not None else 0
            for _ in range(steps):
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                sample_estimator()
                # The XML clock advances only when a new 5 Hz scan is available.
                scan_time = float(data.sensor("lidar_time").data[0])
                if scan_time > last_scan_time:
                    scan = read_lidar(data)
                    print_lidar(scan)
                    last_scan_time = scan.time
                if camera is not None:
                    next_frame = camera.read(data)
                    if next_frame is not frame:
                        frame_count += 1
                    frame = next_frame
            if frame is not None:
                from PIL import Image

                args.camera_output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame.rgb).save(args.camera_output, format="PNG")
                print(
                    f"RGB camera: {CAMERA_WIDTH} x {CAMERA_HEIGHT}, {CAMERA_FPS} Hz, "
                    f"{frame_count} frames; saved t={frame.time:.3f} s to "
                    f"{args.camera_output.resolve()}",
                    flush=True,
                )
        if not all(math.isfinite(value) for value in (*data.qpos, *data.qvel)):
            raise RuntimeError("Simulation produced a non-finite state.")
        if any(data.warning.number):
            raise RuntimeError(f"MuJoCo warnings: {data.warning.number}")
        print(f"Physics check passed: {data.time:.3f} s, no MuJoCo warnings.")
    else:
        from mujoco import viewer

        if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("GUI encoder sampling requires a non-RK4 integrator.")
        # Keep the managed viewer. Its control callback runs after position
        # sensors refresh, allowing the same sampler to run during GUI stepping.
        # The time gate prevents new noise samples during pause/repeated forwards.
        previous_callback = mujoco.get_mjcb_control()

        def viewer_update(callback_model, callback_data):
            if previous_callback is not None:
                previous_callback(callback_model, callback_data)
            if callback_model is model and callback_data is data:
                sample_estimator()

        try:
            mujoco.set_mjcb_control(viewer_update)
            viewer.launch(model, data)
        finally:
            mujoco.set_mjcb_control(previous_callback)
        mujoco.mj_forward(model, data)
        print_lidar(read_lidar(data))

    # Include any final partial encoder period so truth and estimate agree in time.
    sample_estimator(force=True)
    print_estimate(estimate, data)
    print(f"Final ideal encoders: {read_absolute_wheel_encoders(data)}")
    print(f"Final encoders used by estimator: {last_encoders}")


if __name__ == "__main__":
    main()
