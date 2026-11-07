"""Command-line entry point for the simulation and real TurtleBot demos."""

import argparse
import math
from pathlib import Path

import cv2


def main() -> None:
    # Small camera images do not need a worker for every CPU core. Leave room
    # for physics and the viewer while OpenCV processes each frame.
    cv2.setNumThreads(min(4, cv2.getNumThreads()))
    # Read the run options from the terminal command.
    parser = argparse.ArgumentParser(description="Run the TurtleBot estimation and docking demo.")
    parser.add_argument("--backend", choices=("mujoco", "rosbridge"), default="mujoco")
    parser.add_argument("--config", type=Path, help="Backend-specific YAML configuration.")
    parser.add_argument("--host", help="Override the rosbridge host in hardware configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Hardware: process sensors but publish zero velocity.")
    parser.add_argument("--show-camera", action="store_true", help="Hardware: show camera and ArUco pose overlays; Q stops demo.")
    parser.add_argument("--log-dir", type=Path, help="Parent folder for telemetry runs. Enables simulation logging; hardware defaults to outputs/hardware-runs.")
    parser.add_argument("--stop-on-dock", action="store_true", help="Simulation: finish immediately after recording the first DONE update.")
    parser.add_argument("--estimator", choices=('ekf', 'ukf'), default='ekf', help="Simulation estimator comparison.")
    parser.add_argument("--controller", choices=('lqi', 'pid'), default='lqi', help="Simulation tracker; PBVS remains shared.")
    parser.add_argument("--encoder-noise-scale", type=float, default=1., help="Simulation: multiply injected encoder noise; filter assumptions stay fixed.")
    parser.add_argument("--noise-seed", type=int, help="Simulation: override the configured measurement-noise seed.")
    parser.add_argument("--capture", action='store_true', help="Simulation: save replay state and GUI panels beside telemetry; requires --log-dir.")
    parser.add_argument("--speed-scale", type=float, default=1., help="Hardware: multiply configured navigation speeds (0 < scale <= 2).")
    parser.add_argument("--headless", action="store_true", help="Simulation: run without a viewer.")
    parser.add_argument("--verbose", action="store_true", help="Hardware: add marker visibility and sent velocity to concise terminal updates. Simulation: add pose diagnostics.")
    parser.add_argument(
        "--seconds", type=float,
        help="Run duration in seconds; simulation headless defaults to 2 s.",
    )
    args = parser.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite")
    if not math.isfinite(args.speed_scale) or not 0 < args.speed_scale <= 2:
        parser.error("--speed-scale must be positive and at most 2")
    if args.backend == "rosbridge":
        if args.estimator != 'ekf' or args.controller != 'lqi' or args.encoder_noise_scale != 1 or args.noise_seed is not None or args.capture:
            parser.error('Estimator/controller/noise/capture comparison options are simulation-only')
        if args.stop_on_dock:
            parser.error("--stop-on-dock is simulation-only")
        from .hardware.config import DEFAULT_HARDWARE_CONFIG, load_hardware_config, scale_hardware_speeds
        from .hardware.runner import run_hardware

        try:
            config = load_hardware_config(args.config or DEFAULT_HARDWARE_CONFIG, host=args.host)
            config = scale_hardware_speeds(config, args.speed_scale)
            run_hardware(config, seconds=args.seconds, dry_run=args.dry_run, verbose=args.verbose,
                         show_camera=args.show_camera, log_dir=args.log_dir)
        except ImportError as error:
            parser.error(f"Missing hardware dependency: {error}. See the README hardware setup.")
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            parser.error(str(error))
        return
    if args.host is not None or args.dry_run or args.show_camera or args.speed_scale != 1:
        parser.error("--host, --dry-run, --show-camera and --speed-scale require --backend rosbridge")

    from .simulation.runner import run_simulation

    try:
        run_simulation(config_path=args.config, seconds=args.seconds,
                       headless=args.headless, verbose=args.verbose,
                       log_dir=args.log_dir, stop_on_dock=args.stop_on_dock,
                       estimator_method=args.estimator, controller_method=args.controller,
                       encoder_noise_scale=args.encoder_noise_scale, noise_seed=args.noise_seed, capture=args.capture)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
