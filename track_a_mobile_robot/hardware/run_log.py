"""Append-only hardware telemetry for offline plots and animated replay."""

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import threading
import uuid

import numpy as np

from ..state_model import DifferentialDriveModel


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / 'outputs/hardware-runs'


def json_value(value):
    """Keep JSON strict while retaining distinct missing/invalid/no-return values."""
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class RunLog:
    """Bounded asynchronous file writer; never silently drop telemetry."""

    def __init__(self, config, *, directory=DEFAULT_LOG_DIR, dry_run=False):
        started = datetime.now(timezone.utc)
        self.run_id = started.strftime('%Y%m%dT%H%M%S.%fZ') + '-' + uuid.uuid4().hex[:8]
        folder = Path(directory).expanduser().resolve() / self.run_id
        folder.mkdir(parents=True, exist_ok=False)
        self.path = folder / 'telemetry.jsonl'
        self._file = self.path.open('x', encoding='utf-8', buffering=1)
        self._queue = queue.Queue(maxsize=256)
        self._error = None
        self._closed = False
        self._thread = threading.Thread(target=self._write, name='hardware-run-log', daemon=True)
        self._thread.start()
        self._model = DifferentialDriveModel(config.drive.wheel_radius_m, config.drive.wheel_separation_m)
        self._max_gap = config.timing['max_encoder_gap_s']
        self._odom_previous = None
        self._odom_pose = np.zeros(3)
        self._odom_epoch = 0
        self.record({
            'type': 'metadata', 'schema_version': 1, 'run_id': self.run_id,
            'started_utc': started.isoformat(), 'dry_run': dry_run, 'config': config,
            'units': {'distance': 'metres', 'angle': 'radians', 'time': 'seconds',
                      'source_stamps': 'ROS nanoseconds'},
            'frames': {'base': 'X forward, Y left, Z up',
                       'odometry': 'wheel-only origin at first encoder; reset after large gap',
                       'estimate': 'origin at first accepted marker; reset after large encoder gap',
                       'lidar_points_base': 'XY metres at scan capture time; no motion compensation'},
            'special_values': {'Infinity': 'no return / beyond maximum',
                               'NaN': 'invalid / below minimum', '-Infinity': 'negative infinity'},
            'sampling': 'one tick per control update; repeated sensor stamps are held samples, not new measurements',
            'images_saved': False,
        })

    def _write(self):
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._file.write(json.dumps(item, separators=(',', ':'), allow_nan=False) + '\n')
        except Exception as exc:  # Report file/disk failures to the control thread.
            self._error = exc
        finally:
            try:
                self._file.close()
            except Exception as exc:
                self._error = exc

    def record(self, item):
        if self._error is not None:
            raise RuntimeError(f'Run log failed: {self.path}: {self._error}') from self._error
        if self._closed:
            raise RuntimeError('Run log already closed')
        # Snapshot arrays before the worker sees them; no RGB image enters this queue.
        try:
            self._queue.put_nowait(json_value(item))
        except queue.Full as exc:
            raise RuntimeError('Run log writer cannot keep up; stopping to avoid missing telemetry') from exc

    def tick(self, *, backend, now, navigator, output, measured, accepted, estimate,
             estimate_time, request, allowed, requested, sent, wheels, stop_reason,
             encoder_new, dropped_camera, estimator_epoch, control_duration_s):
        encoder = backend.encoders
        if encoder is not None and (self._odom_previous is None or encoder.time != self._odom_previous.time):
            previous = self._odom_previous
            if previous is not None:
                if encoder.time-previous.time > self._max_gap:
                    self._odom_epoch += 1
                    self._odom_pose = np.zeros(3)
                else:
                    self._odom_pose = self._model.predict_pose(
                        self._odom_pose, encoder.left_encoder_rad-previous.left_encoder_rad,
                        encoder.right_encoder_rad-previous.right_encoder_rad)
            self._odom_previous = encoder
        camera = backend.camera
        sensor_ages = {name: None if getattr(backend, name) is None else now-getattr(backend, name).time
                       for name in ('encoders', 'camera', 'lidar')}
        self.record({
            'type': 'tick', 'time': now, 'phase': navigator.phase,
            'status': stop_reason or navigator.status, 'stop_reason': stop_reason,
            'control_duration_s': control_duration_s,
            'source_stamps_ns': dict(backend._stamps), 'source_origin_ns': backend._source_origin,
            'source_offset_s': backend._source_offset, 'sensor_ages_s': sensor_ages,
            'sensor_last_seen_s': dict(backend._seen_at),
            'encoder_new': encoder_new, 'encoders': encoder, 'encoder_gap_s': backend.encoder_gap_s,
            'odometry': None if encoder is None else {
                'time': encoder.time, 'pose_local': self._odom_pose, 'epoch': self._odom_epoch},
            'estimate': estimate, 'estimate_encoder_time': estimate_time, 'estimator_epoch': estimator_epoch,
            'dock_pose_base': None if estimate is None else estimate.dock_pose_base,
            'camera': None if camera is None else {'time': camera.time, 'image_shape': camera.rgb.shape,
                                                  'new_frame': output is not None and output.new_camera_frame},
            'markers': [] if output is None else output.markers,
            'measured_dock_observation': measured, 'accepted_dock_observation': accepted,
            'marker_status': navigator.marker_status(now), 'dropped_camera': dropped_camera,
            'lidar': backend.lidar, 'lidar_raw_ranges_m': backend.lidar_raw_ranges_m,
            'lidar_raw_zeros': backend.lidar_zero_count,
            'lidar_points_base': None if output is None else output.lidar_points_base,
            'obstacle_checks_enabled': navigator.obstacle_checks_enabled,
            'targets': {'offset_pose_local': navigator.coarse_pose, 'dock_pose_local': navigator.docking_pose,
                        'detour_pose_local': navigator.detour_pose},
            'trajectory_path_local': navigator.trajectory_path,
            'request': request, 'motion_allowed': allowed,
            'requested_command': requested, 'sent_command': sent, 'commanded_wheels_rad_s': wheels,
        })

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            try:
                self._queue.put(None, timeout=2.)
            except queue.Full as exc:
                raise RuntimeError(f'Run log could not finish: {self.path}') from exc
        self._thread.join(timeout=5.)
        if self._thread.is_alive():
            raise RuntimeError(f'Run log writer did not finish: {self.path}')
        if self._error is not None:
            raise RuntimeError(f'Run log failed: {self.path}: {self._error}') from self._error
