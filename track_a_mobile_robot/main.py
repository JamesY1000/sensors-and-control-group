"""Open the default TurtleBot3 scene: python -m track_a_mobile_robot.main."""

import argparse
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


SCENE = Path(__file__).resolve().parents[1] / "world/room_with_charger.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the default TurtleBot3 Waffle Pi world.")
    parser.add_argument("--headless", action="store_true", help="Check physics without opening a viewer.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Duration of a headless check (default: 2 s).")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"Loaded: {SCENE}", flush=True)
    print(f"Wheel actuators: {model.nu}; sensor definitions: {model.nsensor}; "
          f"robot cameras: {model.ncam}", flush=True)

    if args.headless:
        steps = math.ceil(args.seconds / model.opt.timestep)
        mujoco.mj_step(model, data, nstep=steps)
        if not all(math.isfinite(value) for value in (*data.qpos, *data.qvel)):
            raise RuntimeError("Simulation produced a non-finite state.")
        if any(data.warning.number):
            raise RuntimeError(f"MuJoCo warnings: {data.warning.number}")
        print(f"Physics check passed: {data.time:.3f} s, no MuJoCo warnings.")
    else:
        from mujoco import viewer

        # The built-in viewer manages physics and rendering until it is closed.
        # No navigation, sensor processing or controller is running yet.
        viewer.launch(model, data)


if __name__ == "__main__":
    main()
