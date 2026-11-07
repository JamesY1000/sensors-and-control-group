"""Compare EKF and UKF on identical logged inputs, without rerunning the robot.

Usage: python -m track_a_mobile_robot.simulation.estimator_comparison RUN_DIRECTORY
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from ..estimation import create_estimator
from ..sensing import WheelEncoderReading
from ..state_model import DifferentialDriveModel, wrap_angle
from .evaluation import read_run, pose_metrics, source_snapshot


def compare(path):
    path, metadata, ticks, scored = read_run(path)
    destination = path.parent/'estimator_comparison'
    destination.mkdir(exist_ok=True)
    summary = {'input_log':str(path), 'scope':'Identical-input replay, not a changed closed-loop trajectory.',
        'noise_seed':metadata['config']['estimation']['noise_seed'],
        'encoder_noise_scale':metadata['run_options'].get('encoder_noise_scale',1.),
        'source':source_snapshot(destination),
        'ukf_parameters':{'alpha':1.,'beta':2.,'kappa':0.,'prediction':'8D wheel-noise augmentation',
                          'noise':'Same nominal encoder, process and per-observation camera covariance as EKF'},
        'timing':'Per-update process CPU time, predict plus initialise/correct; excludes perception and rendering.'}
    for method in ('ekf','ukf'):
        estimator = create_estimator(SimpleNamespace(**metadata['config']['estimation']),
            DifferentialDriveModel(), metadata['encoder_noise_std_rad'], method=method)
        times=[];poses=[];truth=[];sigmas=[];accepted=[];duration=[]
        original_difference=[];original_cov_difference=[];original_acceptance=[]
        for row in ticks:
            start = time.process_time_ns()
            encoders = WheelEncoderReading(**row['encoders'])
            state = estimator.predict(encoders)
            observation = row['measured_dock_observation']
            if observation is not None:
                if not estimator.initialised:
                    state = estimator.initialise_dock(encoders, observation['pose_base'], observation['covariance'])
                else:
                    state = estimator.correct(observation['pose_base'], observation['covariance'], observation['time'])
            elapsed = (time.process_time_ns()-start)/1e6
            if not row['initialised'] or row['time']>scored[-1]['time']:
                continue
            times.append(row['time']);poses.append(state.pose_local);truth.append(row['truth']['robot_local'])
            sigmas.append(np.sqrt(np.maximum(0,np.diag(state.covariance))))
            accepted.append(estimator.camera_correction_accepted);duration.append(elapsed)
            if method=='ekf' and metadata['estimator']=='EKF':
                delta=np.array(state.pose_local)-row['estimate']['pose_local'];delta[2]=wrap_angle(delta[2])
                original_difference.append(float(np.max(np.abs(delta))))
                original_cov_difference.append(float(np.max(np.abs(state.covariance-row['estimate']['covariance']))))
                original_acceptance.append(estimator.camera_correction_accepted==row['camera']['correction_accepted'])
        times,poses,truth,sigmas=np.array(times),np.array(poses),np.array(truth),np.array(sigmas)
        error=poses-truth;error[:,2]=wrap_angle(error[:,2])
        metrics=pose_metrics(error,sigmas)
        metrics.update(update_cpu_median_ms=float(np.median(duration)),update_cpu_p95_ms=float(np.percentile(duration,95)),
                       accepted_corrections=int(sum(accepted)),samples=len(times))
        if original_difference:
            metrics['original_pose_max_difference']=max(original_difference)
            metrics['original_cov_max_difference']=max(original_cov_difference)
            metrics['original_acceptance_matches']=all(original_acceptance)
            if max(original_difference)>1e-9 or max(original_cov_difference)>1e-9 or not all(original_acceptance):
                raise ValueError('EKF replay does not reproduce recorded filter: comparison inputs must be checked')
        summary[method]=metrics
        np.savez_compressed(destination/f'{method}.npz',time=times,estimate=poses,truth=truth,sigma=sigmas,
            error=error,accepted=accepted,update_cpu_ms=duration)
    (destination/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ('ekf','ukf')},indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    compare(parser.parse_args().run)
