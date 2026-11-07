"""Simulation telemetry and offline reports; truth is used only for evaluation.

Run: python -m track_a_mobile_robot.simulation.evaluation RUN_DIRECTORY
"""

import argparse
import csv
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import zipfile

import numpy as np

from ..state_model import wrap_angle

REPO_ROOT = Path(__file__).resolve().parents[2]


def json_value(value):
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def true_pose(data, body_name):
    """Read a body's planar world pose; never called by autonomy."""
    body = data.body(body_name)
    rotation = np.asarray(body.xmat).reshape(3, 3)
    return np.array([*body.xpos[:2], np.arctan2(rotation[1, 0], rotation[0, 0])])


def pose_in_frame(pose, origin):
    c, s = np.cos(origin[2]), np.sin(origin[2])
    xy = np.array([[c, s], [-s, c]]) @ (np.asarray(pose[:2]) - origin[:2])
    return np.r_[xy, wrap_angle(pose[2] - origin[2])]


def source_snapshot(folder):
    """Retain the actual source, including uncommitted files, with each result."""
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    paths = set(REPO_ROOT.joinpath('track_a_mobile_robot').rglob('*.py'))
    paths.update(REPO_ROOT.joinpath('config').glob('*.yml'))
    paths.update(REPO_ROOT.joinpath('world').rglob('*.xml'))
    paths.update(REPO_ROOT.joinpath('robot_model').rglob('*.xml'))
    paths.update(REPO_ROOT.glob('requirements*.txt'))
    hashes = {}
    with zipfile.ZipFile(folder / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            name = path.relative_to(REPO_ROOT).as_posix()
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    return {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain'),
            'snapshot': 'source_snapshot.zip', 'sha256': hashes,
            'note': 'Snapshot includes code/config/XML; unchanged mesh and texture assets come from the commit.'}


class SimulationRunLog:
    """One JSONL record per update, with exact settings and a source snapshot."""

    def __init__(self, directory, config, *, config_path, options, encoder_noise_std_rad):
        started = datetime.now(timezone.utc)
        self.folder = Path(directory).expanduser().resolve() / started.strftime('%Y%m%dT%H%M%S.%fZ')
        self.folder.mkdir(parents=True, exist_ok=False)
        self.path = self.folder / 'telemetry.jsonl'
        self.origin = None
        self._measurement = None
        self._estimator_duration_s = None
        metadata = {
            'type': 'metadata', 'schema_version': 1, 'started_utc': started.isoformat(),
            'estimator': options.get('estimator', 'ekf').upper(),
            'controller': options.get('controller', 'lqi').upper() + ' + PBVS', 'config': config,
            'config_path': Path(config_path).resolve(), 'run_options': options,
            'encoder_noise_std_rad': encoder_noise_std_rad, 'source': source_snapshot(self.folder),
            'actual_encoder_noise_std_rad': [v*options.get('encoder_noise_scale', 1.) for v in encoder_noise_std_rad],
            'noise_experiment': 'Measurement noise is scaled; filter encoder/process/camera covariance assumptions stay nominal.',
            'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                            'packages': {name: version(name) for name in
                                         ('numpy', 'scipy', 'mujoco', 'opencv-contrib-python')}},
            'units': {'position': 'metres', 'heading': 'radians', 'time': 'simulation seconds'},
            'alignment': 'True robot pose at first accepted marker defines a fixed local frame. No subsequent alignment.',
            'measurements': 'Camera pose before EKF gating; acceptance recorded separately. Encoders are the actual noisy inputs.',
            'timing': 'Predict plus initialise/correct wall time, excluding perception, rendering and logging.',
            'images_saved': options.get('capture', False),
        }
        self._file = self.path.open('x', encoding='utf-8', buffering=1)
        try:
            self._write(metadata)
        except BaseException:
            self._file.close()
            raise

    def _write(self, value):
        self._file.write(json.dumps(json_value(value), allow_nan=False, separators=(',', ':')) + '\n')

    def observe(self, observation, estimator_duration_s):
        """Receive the same pre-gate measurement presented to the filter."""
        self._measurement = observation
        self._estimator_duration_s = estimator_duration_s

    def tick(self, *, data, backend, estimator, estimate, output, navigator, request, command, allowed):
        robot_world = true_pose(data, 'base')
        dock_world = true_pose(data, 'charging_station')
        if estimator.initialised and self.origin is None:
            self.origin = robot_world.copy()
            self._write({'type': 'alignment', 'time': estimate.time, 'origin_world': self.origin})
        self._write({
            'type': 'tick', 'time': estimate.time, 'initialised': estimator.initialised,
            'phase': navigator.phase, 'status': navigator.status,
            'encoders': backend.encoders, 'measured_dock_observation': self._measurement,
            'camera': {'time': output.camera.time, 'new_frame': output.new_camera_frame,
                       'correction_accepted': estimator.camera_correction_accepted},
            'estimate': estimate, 'estimated_dock_pose_base': estimate.dock_pose_base,
            'truth': {'robot_world': robot_world, 'dock_world': dock_world,
                      'robot_local': None if self.origin is None else pose_in_frame(robot_world, self.origin),
                      'dock_local': None if self.origin is None else pose_in_frame(dock_world, self.origin)},
            'request': request, 'command': command, 'motion_allowed': allowed,
            'estimator_duration_s': self._estimator_duration_s,
        })

    def close(self, *, reason, time):
        try:
            self._write({'type': 'end', 'reason': reason, 'time': time})
        finally:
            self._file.close()


def read_run(path):
    path = Path(path)
    path = path / 'telemetry.jsonl' if path.is_dir() else path
    with path.open(encoding='utf-8') as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records or records[0].get('type') != 'metadata' or records[0].get('schema_version') != 1:
        raise ValueError('Expected a simulation telemetry log with schema_version 1')
    ticks = [record for record in records if record['type'] == 'tick']
    if not ticks or np.any(np.diff([row['time'] for row in ticks]) <= 0):
        raise ValueError('Expected a nonempty, strictly increasing single-run log')
    aligned = [row for row in ticks if row['initialised']]
    if not aligned:
        raise ValueError('No marker was acquired; no local-frame estimation result exists yet')
    # Exclude the stationary ending so it cannot dilute the reported errors.
    first_done = next((i for i, row in enumerate(aligned) if row['phase'] == 'DONE'), None)
    scored = aligned if first_done is None else aligned[:first_done + 1]
    return path, records[0], ticks, scored


def pose_series(rows, *, dock=False):
    pose_key, cov_key = ('dock_pose_local', 'dock_covariance') if dock else ('pose_local', 'covariance')
    truth_key = 'dock_local' if dock else 'robot_local'
    estimate = np.array([row['estimate'][pose_key] for row in rows])
    truth = np.array([row['truth'][truth_key] for row in rows])
    covariance = np.array([row['estimate'][cov_key] for row in rows])
    variance = np.diagonal(covariance, axis1=1, axis2=2)
    if not np.isfinite(covariance).all() or np.any(variance < -1e-12):
        raise ValueError('Invalid covariance in the logged estimate')
    sigma = np.sqrt(np.maximum(variance, 0))
    error = estimate - truth
    error[:, 2] = wrap_angle(error[:, 2])
    return estimate, truth, error, sigma


def pose_metrics(error, sigma):
    # Zero variance at acquisition defines the origin; exclude it from coverage.
    coverage = []
    for i in range(3):
        valid = sigma[:, i] > 1e-12
        coverage.append(float(np.mean(np.abs(error[valid, i]) <= 2*sigma[valid, i])) if valid.any() else None)
    return {
        'position_rmse_m': float(np.sqrt(np.mean(np.sum(error[:, :2]**2, axis=1)))),
        'heading_rmse_deg': float(np.degrees(np.sqrt(np.mean(error[:, 2]**2)))),
        'peak_position_error_m': float(np.linalg.norm(error[:, :2], axis=1).max()),
        'two_sigma_coverage': dict(zip(('x', 'y', 'heading'), coverage)),
        'coverage_note': 'Empirical marginal coverage in one correlated trajectory; not a formal consistency test.',
    }


def generate_report(path):
    path, metadata, ticks, rows = read_run(path)
    folder = path.parent
    robot, dock = pose_series(rows), pose_series(rows, dock=True)
    last = rows[-1]
    dock_truth = np.asarray(last['truth']['dock_local'])
    nav = metadata['config']['navigation']
    offset = nav['dock_distance_m']
    target = dock_truth + [offset*np.cos(dock_truth[2]), offset*np.sin(dock_truth[2]), np.pi]
    target_error = np.asarray(last['truth']['robot_local']) - target
    target_error[2] = wrap_angle(target_error[2])
    timing_ms = 1000*np.array([row['estimator_duration_s'] for row in rows])
    summary = {
        'estimator': metadata['estimator'], 'controller': metadata['controller'],
        'encoder_noise_scale': metadata['run_options'].get('encoder_noise_scale', 1.),
        'noise_seed': metadata['config']['estimation']['noise_seed'],
        'source_commit': metadata['source']['commit'], 'source_status': metadata['source']['status'],
        'samples_logged': len(ticks), 'samples_scored': len(rows),
        'acquisition_time_s': rows[0]['time'], 'evaluation_end_time_s': last['time'],
        'reported_done_time_s': last['time'] if last['phase'] == 'DONE' else None,
        'robot': pose_metrics(robot[2], robot[3]), 'dock': pose_metrics(dock[2], dock[3]),
        'filter_update_median_ms': float(np.median(timing_ms)),
        'filter_update_p95_ms': float(np.percentile(timing_ms, 95)),
        'camera_measurements_present': sum(row['measured_dock_observation'] is not None for row in rows),
        'camera_corrections_accepted': sum(row['camera']['correction_accepted'] for row in rows),
        'true_final_target_position_error_m': float(np.linalg.norm(target_error[:2])),
        'true_final_target_heading_error_deg': float(np.degrees(abs(target_error[2]))),
        'true_final_pose_within_configured_tolerances': bool(
            np.linalg.norm(target_error[:2]) <= nav['position_tolerance_m'] and
            abs(target_error[2]) <= np.deg2rad(nav['heading_tolerance_deg'])),
        'scope': 'One simulation trial, acquisition through first DONE (or last sample). Truth-only scoring; no physical charging/contact claim.',
    }
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    with (folder / 'estimation.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time_s', 'phase', 'camera_correction_accepted'] +
                        [f'{name}_{axis}' for name in ('true', 'estimated', 'error', 'sigma')
                         for axis in ('x_m', 'y_m', 'heading_rad')])
        for i, row in enumerate(rows):
            writer.writerow([row['time'], row['phase'], int(row['camera']['correction_accepted']),
                             *robot[1][i], *robot[0][i], *robot[2][i], *robot[3][i]])
    plot_report(folder, metadata, rows, robot, dock)
    return summary


def plot_report(folder, metadata, rows, robot, dock):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    time = np.array([row['time'] for row in rows])
    title = f"{metadata['estimator']} baseline | seed {metadata['config']['estimation']['noise_seed']} | single trial"
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    blue, grey = '#2275bd', '#343a40'
    for name, series in (('robot', robot), ('dock', dock)):
        estimate, truth, error, sigma = (array.copy() for array in series)
        truth[:, 2] = np.unwrap(truth[:, 2])
        estimate[:, 2] = truth[:, 2] + error[:, 2]
        for array in (estimate, truth, error, sigma):
            array[:, 2] = np.degrees(array[:, 2])
        fig, axes = plt.subplots(3, 2, figsize=(12.8, 7.2), sharex=True, layout='constrained')
        for i, label in enumerate(('x (m)', 'y (m)', 'heading (deg)')):
            left, right = axes[i]
            left.fill_between(time, estimate[:, i]-2*sigma[:, i], estimate[:, i]+2*sigma[:, i],
                              color=blue, alpha=.18, label='Estimate +/- 2 sigma')
            left.plot(time, truth[:, i], color=grey, ls='--', lw=1.5, label='Simulator truth')
            left.plot(time, estimate[:, i], color=blue, lw=1.2, label='Estimate')
            right.fill_between(time, -2*sigma[:, i], 2*sigma[:, i], color=blue, alpha=.18, label='+/- 2 sigma')
            right.plot(time, error[:, i], color=blue, lw=1.2, label='Estimate minus truth')
            right.axhline(0, color=grey, lw=.7)
            left.set_ylabel(label)
            right.set_ylabel('error ' + label)
            for ax in (left, right):
                ax.grid(alpha=.2)
        axes[0, 0].set_title('Pose and uncertainty')
        axes[0, 1].set_title('Error and uncertainty bounds')
        axes[0, 0].legend(fontsize=8, loc='best')
        axes[0, 1].legend(fontsize=8, loc='best')
        for ax in axes[-1]:
            ax.set_xlabel('Simulation time (s)')
        fig.suptitle(f'{title}\n{name.capitalize()} pose: fixed frame at acquisition; heading errors wrapped')
        fig.savefig(folder / f'{name}_estimation.png', dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    ax.plot(robot[1][:, 0], robot[1][:, 1], '--', color=grey, label='Robot truth')
    ax.plot(robot[0][:, 0], robot[0][:, 1], color=blue, label='Robot estimate')
    ax.scatter(*robot[1][0, :2], marker='o', color='#239b56', label='Acquisition / local origin')
    ax.scatter(*dock[1][-1, :2], marker='x', s=80, color=grey, label='True dock origin')
    ax.scatter(*dock[0][-1, :2], marker='+', s=80, color=blue, label='Estimated dock origin')
    ax.set(xlabel='Local x (m)', ylabel='Local y (m)', title=title+'\nTrajectory after marker acquisition')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(folder / 'trajectory.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Run directory or telemetry.jsonl')
    args = parser.parse_args()
    try:
        summary = generate_report(args.run)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    print(f'Report saved beside {args.run}')


if __name__ == '__main__':
    main()
