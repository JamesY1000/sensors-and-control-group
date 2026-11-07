"""Simulation loop with the same sensing, estimation, navigation and control modules."""

from __future__ import annotations

import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import replace

import numpy as np

from ..control import MotionController, navigation_command
from ..estimation import PoseEstimator, StateEstimate, create_estimator
from ..hal import RobotBackend
from ..navigation import Navigator
from ..perception import ArucoLandmarkDetector, Perception, PerceptionOutput
from ..scene_config import DEFAULT_CONFIG, load_config
from ..state_model import DifferentialDriveModel
from .scene import load_model

VIEWER_UPDATE_HZ = 30.0


def create_demo(model, data, settings, *, estimator_method='ekf', encoder_noise_scale=1.):
    """Build the pipeline from sensor calibration, without world-pose inputs."""
    from ..hal.mujoco import MujocoBackend

    encoder_seed, perception_seed = np.random.SeedSequence(settings.noise_seed).spawn(2)
    backend = MujocoBackend(model, data, rng=np.random.default_rng(encoder_seed), encoder_noise_scale=encoder_noise_scale)
    try:
        estimator = create_estimator(settings, DifferentialDriveModel(), backend.encoder_noise_std_rad, method=estimator_method)
        detector = ArucoLandmarkDetector(backend.camera_matrix, backend.dist_coeffs, backend.camera_to_base)
        perception = Perception(noise=True, rng=np.random.default_rng(perception_seed),
                                detector=detector, lidar_to_base=backend.lidar_to_base)
        backend.start()
    except BaseException:
        backend.close()
        raise
    return estimator, perception, backend


def update_demo(
    backend: RobotBackend,
    estimator: PoseEstimator,
    perception: Perception,
    *, observer=None,
) -> tuple[StateEstimate, PerceptionOutput]:
    """Run prediction and any camera correction at one simulation timestamp."""
    # Predict the current pose and covariance using this update's encoder readings.
    backend.read()
    encoders = backend.encoders
    if encoders is None or backend.lidar is None or backend.camera is None:
        raise RuntimeError("Simulation sensors are not ready")
    started = time.perf_counter()
    estimate = estimator.predict(encoders)
    estimator_duration = time.perf_counter() - started

    # Get LiDAR points, marker detections and any valid new robot-relative dock pose.
    output = perception.process(backend.now(), backend.lidar, backend.camera)
    observation = output.dock_observation
    # Start odometry at the first usable marker, with the robot pose defined as zero.
    if observation is not None:
        started = time.perf_counter()
        if not estimator.initialised:
            estimate = estimator.initialise_dock(encoders, observation.pose_base, observation.covariance)
        else:
            estimate = estimator.correct(observation.pose_base, observation.covariance, observation.time)
        estimator_duration += time.perf_counter() - started
        if not estimator.camera_correction_accepted:
            output = replace(output, dock_observation=None)

    if observer is not None:
        observer(observation, estimator_duration)

    # Without a camera correction, estimate is still the encoder prediction above.
    # Return both outputs for navigation and control to use in the next stage.
    return estimate, output


def run_simulation(*, config_path=None, seconds=None, headless=False, verbose=False,
                   log_dir=None, stop_on_dock=False, estimator_method='ekf', controller_method='lqi',
                   encoder_noise_scale=1., noise_seed=None, capture=False):
    """Run the interactive demo, or a bounded headless simulation."""
    # Import graphics and simulator dependencies only for the simulation path.
    if sys.platform == "linux" and os.environ.get("DISPLAY"):
        os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
    if sys.platform == "linux":
        # Match the desktop viewer's GL backend to avoid camera context conflicts.
        # Select before importing MuJoCo, while preserving explicit overrides.
        use_egl = headless or not os.environ.get("DISPLAY")
        os.environ.setdefault("MUJOCO_GL", "egl" if use_egl else "glfw")
    import mujoco

    from .display import SensorDisplay

    duration = seconds
    # Give headless runs a default duration; viewer runs can continue until closed.
    if headless and duration is None:
        duration = 2.0

    # Load our settings and build the configured simulation model.
    config = load_config(config_path or DEFAULT_CONFIG)
    if noise_seed is not None:
        config = replace(config, estimation=replace(config.estimation, noise_seed=noise_seed))
    if estimator_method not in ('ekf', 'ukf') or controller_method not in ('lqi', 'pid'):
        raise ValueError('Expected estimator ekf/ukf and controller lqi/pid')
    if capture and log_dir is None:
        raise ValueError('Capture requires --log-dir')
    model = load_model(config=config)
    if config.estimation.update_hz > 1.0 / model.opt.timestep:
        raise ValueError("estimation.update_hz cannot exceed the physics update rate")
    # MjData holds the changing simulation state. Refresh sensors before the first read.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Use the same loop in both modes; window will be None when running headless.
    viewer_context = nullcontext(None)
    if not headless:
        from mujoco import viewer

        viewer_context = viewer.launch_passive(model, data)

    print(f"Sensing and {estimator_method.upper()}: {config.estimation.update_hz:g} Hz; {controller_method.upper()} tracking; encoder noise ×{encoder_noise_scale:g}.", flush=True)
    print("Odometry starts at marker acquisition. Logs: camera age and simulation speed; --verbose adds pose diagnostics.", flush=True)
    with viewer_context as window:
        # Create these once so each update uses the previous estimate and sensor history.
        estimator, perception, backend = create_demo(model, data, config.estimation,
            estimator_method=estimator_method, encoder_noise_scale=encoder_noise_scale)
        navigator = Navigator(config.navigation)
        if controller_method == 'pid':
            from ..control_pid import PIDMotionController
            controller = PIDMotionController(config.navigation)
        else:
            controller = MotionController(config.navigation)
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
        run_log = None
        end_reason = 'viewer_closed'
        try:
            if log_dir is not None:
                from .evaluation import SimulationRunLog
                if capture:
                    from .capture import CapturedRunLog
                    log_class = CapturedRunLog
                else:
                    log_class = SimulationRunLog
                run_log = log_class(log_dir, config, config_path=config_path or DEFAULT_CONFIG,
                    options={'seconds': duration, 'headless': headless, 'stop_on_dock': stop_on_dock,
                             'estimator': estimator_method, 'controller': controller_method,
                             'encoder_noise_scale': encoder_noise_scale, 'capture': capture},
                    encoder_noise_std_rad=backend.encoder_noise_std_rad)
                print(f'Simulation telemetry: {run_log.path}', flush=True)
            while window is None or window.is_running():
                # Sensor scheduling uses simulation time; viewer pacing uses wall-clock time.
                now = float(data.time)
                if duration is not None and now >= duration:
                    end_reason = 'time_limit'
                    break

                # Viewer reset starts a new estimate and clears cached sensor samples.
                if now < last_physics_time:
                    if run_log is not None:
                        raise ValueError('Simulation reset during logging. Start a new run to keep one fixed evaluation frame.')
                    backend.close()
                    estimator, perception, backend = create_demo(model, data, config.estimation,
                        estimator_method=estimator_method, encoder_noise_scale=encoder_noise_scale)
                    navigator = Navigator(config.navigation)
                    controller.reset()
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
                    estimate, output = update_demo(backend, estimator, perception,
                                                   observer=None if run_log is None else run_log.observe)
                    last_update_slot = update_slot
                    camera_frames += int(output.new_camera_frame)
                    camera_corrections += int(output.dock_observation is not None)
                    if output.dock_observation is not None:
                        last_camera_time = output.dock_observation.time

                    # LQI tracks coarse/detour references. PBVS then commands
                    # velocities from the EKF's relative dock pose, without a reference.
                    request, command, allowed = navigation_command(navigator, controller, estimate, output)
                    backend.send_command(command)
                    if run_log is not None:
                        run_log.tick(data=data, backend=backend, estimator=estimator, estimate=estimate,
                                     output=output, navigator=navigator, request=request,
                                     command=command, allowed=allowed)

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
                        if verbose:
                            x, y, heading = estimate.pose_local
                            sigma = np.sqrt(np.diag(estimate.covariance))
                            line += (f' | pose=({x:+.3f}, {y:+.3f} m, {np.degrees(heading):+.1f} deg)'
                                     f' sigma=({sigma[0]:.3f}, {sigma[1]:.3f} m, {np.degrees(sigma[2]):.1f} deg)'
                                     f' camera={camera_corrections}/{camera_frames}')
                        print(line, flush=True)
                        last_report_event = event
                        next_report_time = now + 2.0

                    if stop_on_dock and navigator.phase == 'DONE':
                        end_reason = 'docked'
                        break

                # Hold the latest command between updates, then advance physics one timestep.
                backend.step()
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
            end_reason = 'interrupted'
            print("\nSimulation interrupted.", flush=True)
        except BaseException:
            end_reason = 'error'
            raise
        finally:
            # Stop wheel commands and release the renderer even if the run is interrupted.
            backend.close()
            if display is not None:
                display.close()
            if run_log is not None:
                run_log.close(reason=end_reason, time=float(data.time))

    print(f"Simulation stopped at t={data.time:.3f}s; camera corrections={camera_corrections}/{camera_frames} frames.")
    return None if run_log is None else run_log.folder
