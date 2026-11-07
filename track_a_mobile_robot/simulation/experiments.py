"""Run the declared controller/noise matrix and retain every attempted trial.

python -m track_a_mobile_robot.simulation.experiments --seeds 42 43 44 --jobs 2
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
from .evaluation import generate_report, read_run
from ..control import trajectory_error

_REPORT_LOCK = threading.Lock()


def tracking_metrics(folder):
    _,_,_,rows=read_run(folder)
    drive=[r for r in rows if r['request']['reference'] and r['request']['reference']['kind']=='DRIVE']
    if not drive:return {'driving_samples':0}
    true=np.array([trajectory_error(r['truth']['robot_local'],r['request']['reference']['pose_local']) for r in drive])
    estimated=np.array([trajectory_error(r['estimate']['pose_local'],r['request']['reference']['pose_local']) for r in drive])
    return {'driving_samples':len(drive),
        'true_along_rmse_m':float(np.sqrt(np.mean(true[:,0]**2))),
        'true_cross_rmse_m':float(np.sqrt(np.mean(true[:,1]**2))),
        'true_heading_rmse_deg':float(np.degrees(np.sqrt(np.mean(true[:,2]**2)))),
        'estimated_cross_rmse_m':float(np.sqrt(np.mean(estimated[:,1]**2))),
        'drive_yaw_rate_rms_rad_s':float(np.sqrt(np.mean([r['command']['omega_rad_s']**2 for r in drive]))),
        'scope':'All DRIVE references in the closed-loop navigation; generated references may differ between runs.'}


def run_case(root, seed, controller, noise, capture):
    name=f'{controller}-noise{noise:g}-seed{seed}'
    parent=root/name;parent.mkdir(parents=True,exist_ok=True)
    log=root/f'{name}.stdout.txt'
    cmd=[sys.executable,'-m','track_a_mobile_robot.main','--headless','--seconds','240','--stop-on-dock',
        '--controller',controller,'--noise-seed',str(seed),'--encoder-noise-scale',str(noise),'--log-dir',str(parent)]
    if capture:cmd.append('--capture')
    started=time.monotonic()
    print('START',name,flush=True)
    with log.open('w') as output:
        completed=subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT)
    folders=sorted(p for p in parent.iterdir() if p.is_dir())
    result={'case':name,'seed':seed,'controller':controller,'encoder_noise_scale':noise,
            'capture':capture,'exit_code':completed.returncode,'wall_seconds':time.monotonic()-started,
            'stdout':str(log),'run_directory':str(folders[-1]) if folders else None}
    if completed.returncode==0 and folders:
        try:
            with _REPORT_LOCK:
                result['evaluation']=generate_report(folders[-1])
            result['tracking']=tracking_metrics(folders[-1])
            (folders[-1]/'tracking_summary.json').write_text(json.dumps(result['tracking'],indent=2))
        except ValueError as error:
            result['evaluation_error']=str(error)
    (root/f'{name}.result.json').write_text(json.dumps(result,indent=2))
    print('FINISH',name,'exit',completed.returncode,'done',result.get('evaluation',{}).get('reported_done_time_s'),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--output',type=Path,default=Path('outputs/simulation-runs/comparison-20261107'))
    args=parser.parse_args()
    if args.jobs<1:parser.error('jobs must be positive')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    cases=[(seed,controller,noise) for seed in args.seeds for controller in ('lqi','pid') for noise in (1.,5.)]
    protocol={'seeds':args.seeds,'controllers':['lqi','pid'],'estimator':'ekf','noise_scales':[1,5],
        'actual_encoder_std_rad':[.01,.05],'assumed_encoder_std_rad':.01,
        'fixed':'Initial scene, camera/LiDAR noise, process/camera covariance, controller gains, PBVS and stopping tolerances.',
        'timeout_sim_s':240,'representative_capture_seed':args.seeds[0],
        'comparison':'Closed-loop navigation: references can change with feedback. All attempted cases retained.',
        'planned_cases':[{'seed':s,'controller':c,'noise_scale':n} for s,c,n in cases]}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(run_case,root,s,c,n,s==args.seeds[0]) for s,c,n in cases]
        results=[f.result() for f in concurrent.futures.as_completed(futures)]
    (root/'results.json').write_text(json.dumps(sorted(results,key=lambda r:r['case']),indent=2))
    print('MATRIX SAVED',root,flush=True)


if __name__=='__main__':main()
