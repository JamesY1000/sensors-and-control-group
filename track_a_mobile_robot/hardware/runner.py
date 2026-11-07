"""Real TurtleBot demo loop: sensor fusion, navigation, preview and telemetry."""

import math
import time
from dataclasses import replace

from ..control import MotionController, VelocityCommand, navigation_command, velocity_command_to_wheel_speeds
from ..estimation import create_estimator
from ..hal.rosbridge import RosbridgeBackend
from ..perception import ArucoLandmarkDetector, Perception
from ..state_model import DifferentialDriveModel
from .navigation import HardwareNavigator
from .sensor_sync import SensorFusion
from .run_log import DEFAULT_LOG_DIR, RunLog


def run_hardware(config, *, seconds=None, dry_run=False, verbose=False, backend=None, show_camera=False, log_dir=None):
    """Run hardware search/navigation/docking on real measurements; dry_run sends zero."""
    backend = RosbridgeBackend(config) if backend is None else backend
    detector = ArucoLandmarkDetector(
        config.camera_matrix, config.dist_coeffs, config.camera_to_base,
        config.marker_sizes_m, config.dictionary_id, accept_any_detection=config.accept_any_detection,
    )
    perception = Perception(detector=detector, lidar_to_base=config.lidar_to_base,
                            marker_to_dock=config.marker_to_dock, marker_as_dock=config.marker_as_dock,
                            accept_any_detection=config.accept_any_detection)
    model = DifferentialDriveModel(config.drive.wheel_radius_m, config.drive.wheel_separation_m)
    fusion = SensorFusion(create_estimator(config.estimation, model, config.encoder_noise_std_rad,
                                                  gate_camera=not config.accept_any_detection),
                          config.timing['history_s'], config.timing['max_encoder_gap_s'])
    navigator = HardwareNavigator(config.navigation, future_tolerance_s=config.timing['future_tolerance_s'],
                                  obstacle_checks_enabled=config.obstacle_checks_enabled)
    controller = MotionController(config.navigation, config.drive)
    period = 1/config.estimation.update_hz
    next_report, last_report, last_event = 0., -math.inf, None
    display_markers = []
    display_camera_time = None
    preview = None
    run_log = None
    estimator_epoch = 0
    end_reason, error = 'duration_complete', None
    final_stop_sent = False
    print(f'Rosbridge {config.host}:{config.port}; {"dry run (zero commands)" if dry_run else "autonomous docking"}.', flush=True)
    try:
        run_log = RunLog(config, directory=DEFAULT_LOG_DIR if log_dir is None else log_dir, dry_run=dry_run)
        print(f'Run log: {run_log.path}', flush=True)
        print(f'LiDAR obstacle checks: {"enabled" if config.obstacle_checks_enabled else "DISABLED (scans still logged)"}.', flush=True)
        if show_camera:
            from .preview import CameraPreview
            preview = CameraPreview(config)
            preview.start()
        backend.start()
        started = time.monotonic()
        while seconds is None or time.monotonic()-started < seconds:
            tick_started = time.monotonic()
            if preview is not None and not preview.running():
                print('Camera window closed; stopping hardware run.', flush=True)
                end_reason = 'camera_window_closed'
                break
            if not backend.robot.is_connected:
                raise ConnectionError('Rosbridge disconnected')
            encoders = backend.read()
            now = backend.now()
            reason = backend.health()
            if backend.encoder_gap_s:
                estimator_epoch += 1
                backend.stop()
                fusion = SensorFusion(create_estimator(config.estimation, model, config.encoder_noise_std_rad,
                                                  gate_camera=not config.accept_any_detection),
                                      config.timing['history_s'], config.timing['max_encoder_gap_s'])
                phase = navigator.phase
                navigator = HardwareNavigator(config.navigation,
                                              future_tolerance_s=config.timing['future_tolerance_s'],
                                  obstacle_checks_enabled=config.obstacle_checks_enabled)
                # Once acquired, never return to random search. Re-establish the
                # dock frame from a new image before resuming the same stage.
                if phase not in ('SCAN', 'EXPLORE'):
                    navigator.phase = phase
                controller.reset()
                reason = f'Encoder gap {backend.encoder_gap_s:.2f}s; resetting pose and waiting for fresh data'
            output = None
            measured = None
            if backend.camera is not None and backend.lidar is not None:
                output = perception.process(now, backend.lidar, backend.camera)
                measured = output.dock_observation
                navigator.observe_camera(output)
                if output.new_camera_frame:
                    display_markers = output.markers
                    display_camera_time = output.camera.time
            estimate, accepted = fusion.update(encoders, None if output is None else output.dock_observation)
            estimate_time = None if estimate is None else estimate.time
            request, allowed = None, False
            command = VelocityCommand(now, 0., 0.)
            if estimate is not None and output is not None:
                # The pose is the latest encoder-based estimate, held between
                # readings. Navigation timers use the current shared run clock.
                estimate = replace(estimate, time=now)
                output = replace(output, dock_observation=accepted)
                request, command, allowed = navigation_command(
                    navigator, controller, estimate, output, stop_requested=reason is not None,
                )
            requested = command
            # Recheck after image processing; a slow iteration must not send an
            # old nonzero command after the watchdog has stopped the robot.
            reason = backend.health() or reason
            if time.monotonic()-tick_started > config.timing['command_timeout_s']:
                reason = 'Control update exceeded command timeout'
            if reason is not None:
                # A late veto must pause search timing and discard the reference,
                # just like a stop detected before navigation computed a command.
                navigator.pause_for_sensors(now, reason)
                controller.reset()
                command = VelocityCommand(now, 0., 0.)
            sent = VelocityCommand(now, 0., 0.) if dry_run else command
            backend.send_command(sent)
            wheels = velocity_command_to_wheel_speeds(sent, config.drive)
            run_log.tick(backend=backend, now=now, navigator=navigator, output=output,
                         measured=measured, accepted=accepted, estimate=estimate, estimate_time=estimate_time,
                         request=request, allowed=allowed and reason is None, requested=requested, sent=sent,
                         wheels=wheels, stop_reason=reason, encoder_new=encoders is not None,
                         dropped_camera=fusion.dropped_observations, estimator_epoch=estimator_epoch,
                         control_duration_s=time.monotonic()-tick_started)
            if preview is not None:
                preview.publish(camera=backend.camera,
                                markers=display_markers if backend.camera is not None and
                                backend.camera.time == display_camera_time else [],
                                camera_age=None if backend.camera is None else backend.now()-backend.camera.time,
                                sent_command=(sent.v_mps, sent.omega_rad_s), wheels=wheels,
                                phase=navigator.phase, status=reason or navigator.status,
                                marker_status=navigator.marker_status(now), dry_run=dry_run)
            event = (navigator.phase, reason, None if request is None else request.mode)
            critical_change = last_event is None or event[:2] != last_event[:2]
            action_change = event != last_event and now-last_report >= 1.
            if now >= next_report or critical_change or action_change:
                status = reason or navigator.status
                message = f'{now:6.1f}s {navigator.phase}: {status}'
                if verbose:
                    # Detailed sensor/pose diagnostics live in telemetry.jsonl.
                    marker = navigator.marker_status(now).split(';', 1)[0]
                    message += f' | {marker} | v={sent.v_mps:+.2f} m/s, w={sent.omega_rad_s:+.2f} rad/s'
                print(message, flush=True)
                next_report, last_report, last_event = now+5., now, event
            if navigator.phase == 'DONE':
                end_reason = 'docked'
                break
            time.sleep(max(0., period-(time.monotonic()-tick_started)))
    except KeyboardInterrupt:
        end_reason = 'interrupted'
        print('Hardware run interrupted.', flush=True)
    except BaseException as exc:
        end_reason, error = 'error', {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        try:
            backend.close()
            final_stop_sent = True
        finally:
            try:
                if preview is not None:
                    preview.close()
            finally:
                if run_log is not None:
                    try:
                        run_log.record({'type': 'end', 'time': backend.now(), 'reason': end_reason,
                                        'error': error, 'phase': navigator.phase, 'final_stop_attempted': True,
                                        'backend_closed': final_stop_sent})
                    finally:
                        run_log.close()
