"""Open the default TurtleBot3 scene: python -m track_a_mobile_robot.main."""

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


SCENE = Path(__file__).resolve().parents[1] / "world/room_with_charger.xml"


def load_model(path: str | Path) -> mujoco.MjModel:
    """Expand the XML beam array, then add a clock with the same sampling rate.

    MuJoCo 3.15 recursively duplicates an unreferenced XML clock during
    replication. Adding it after parsing avoids that model-loading problem.
    The rangefinders and their sampling interval remain defined in robot XML.
    """
    spec = mujoco.MjSpec.from_file(str(path))
    first_beam = spec.sensor("lidar_range000")
    spec.add_sensor(
        name="lidar_time",
        type=mujoco.mjtSensor.mjSENS_CLOCK,
        nsample=first_beam.nsample,
        interval=first_beam.interval,
    )
    return spec.compile()


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the TurtleBot3 world with encoders, LiDAR and an RGB camera.")
    parser.add_argument("--headless", action="store_true", help="Check physics without opening a viewer.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Duration of a headless check (default: 2 s).")
    parser.add_argument("--left-wheel", type=float, default=0.0, help="Left wheel velocity target [rad/s].")
    parser.add_argument("--right-wheel", type=float, default=0.0, help="Right wheel velocity target [rad/s].")
    parser.add_argument("--seed", type=int, default=42, help="Encoder noise seed (default: 42).")
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
    model = load_model(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name, target in (("wheel_left", args.left_wheel), ("wheel_right", args.right_wheel)):
        actuator = model.actuator(name)
        lower, upper = actuator.ctrlrange
        if not math.isfinite(target) or not lower <= target <= upper:
            parser.error(f"{name} target must be between {lower:g} and {upper:g} rad/s")
        data.ctrl[actuator.id] = target

    print(f"Loaded: {SCENE}", flush=True)
    print(f"Wheel actuators: {model.nu}; sensor definitions: {model.nsensor}; "
          f"robot cameras: {model.ncam}", flush=True)
    print(f"Wheel targets [rad/s]: left={args.left_wheel:g}, right={args.right_wheel:g}", flush=True)
    print(f"Encoder noise seed: {args.seed}", flush=True)
    print(f"Initial ideal encoders: {read_absolute_wheel_encoders(data)}", flush=True)
    print(f"Initial noisy encoders: {read_noisy_wheel_encoders(model, data, rng)}", flush=True)
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

        # The built-in viewer manages physics and rendering until it is closed.
        viewer.launch(model, data)
        mujoco.mj_forward(model, data)
        print_lidar(read_lidar(data))

    print(f"Final ideal encoders: {read_absolute_wheel_encoders(data)}")
    print(f"Final noisy encoders: {read_noisy_wheel_encoders(model, data, rng)}")


if __name__ == "__main__":
    main()
