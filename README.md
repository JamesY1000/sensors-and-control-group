# Project Page

**Group 26 — Track A: Autonomous navigation and docking of a TurtleBot**

Autonomous TurtleBot3 docking in MuJoCo and on a real Waffle Pi. Both demos use
camera/LiDAR perception, an encoder/camera EKF, LQI navigation and PBVS docking.
The robot starts without a known world pose or dock location. The first accepted
marker establishes a local odometry frame; later observations correct that estimate.

## Individual contributions

| Area | Contributor |
| --- | --- |
| Perception module and sensor-noise extensions | James Baker |
| Estimation, navigation and state model, including EKF and UKF | James Yuan |
| Control: LQI and PID | Nikhil Kumar |
| Hardware implementation | James Yuan |
| Evaluation | Each member evaluated their respective components |

## Run the simulation

Use Python 3.12 or newer from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m track_a_mobile_robot.main
```

On Windows, use `python` and activate `.venv\Scripts\Activate.ps1`.
The demo starts automatically. Close the viewer or press Ctrl+C to finish.
For a bounded run without a viewer:

```bash
python -m track_a_mobile_robot.main --headless --seconds 5
```

Headless mode defaults to two simulated seconds. `--verbose` adds pose diagnostics.
`--config config/simulation_config.yml` selects spawn poses, sensor mounting,
noise, update rates and navigation settings. Restart after changing the file.
The default scene is an 8 × 6 m room with tables, chairs and a two-marker dock.

On Linux, desktop runs default to GLFW for both the viewer and camera renderer;
headless runs default to EGL. An explicit `MUJOCO_GL` setting overrides this
selection. If a desktop run reports `Failed to make the EGL context current`,
try `MUJOCO_GL=glfw python -m track_a_mobile_robot.main`.

The simulation progresses through SCAN/EXPLORE, LQI APPROACH (with obstacle
boundary following when needed), REACQUIRE, PBVS ALIGN, DOCK and DONE.
REACQUIRE requires recent accepted camera poses before PBVS begins. Once in
ALIGN/DOCK, the simulation can continue on encoder prediction during marker
occlusion; completion uses estimated pose tolerances. LiDAR stops remain active.

## Log and evaluate a baseline simulation

The existing demo uses EKF estimation and LQI/PBVS control. Log the whole run
without screen recording; `--stop-on-dock` stops after saving the first DONE
update, while `--seconds` provides a timeout:

```bash
python -m track_a_mobile_robot.main --headless --seconds 240 --stop-on-dock --log-dir outputs/simulation-runs/ekf-baseline
```

Remove `--headless` to watch the same run. Do not reset the viewer during logging.
The terminal prints a unique run directory containing `telemetry.jsonl` and a
source snapshot. Generate plots and metrics from that directory:

```bash
python -m track_a_mobile_robot.simulation.evaluation outputs/simulation-runs/ekf-baseline/RUN_TIMESTAMP
```

Outputs are `robot_estimation.png`, `dock_estimation.png`, `trajectory.png`,
`estimation.csv` and `summary.json`. The log includes configuration, noise seed,
source/version information, actual encoder and pre-gate camera-pose inputs,
estimates/covariances, simulator truth, references, commands and filter timing.
It does not save RGB images or LiDAR scans for a full sensor replay.

Evaluation aligns truth once at the first accepted marker and excludes the
pre-acquisition search and stationary ending from estimation metrics. Truth
never enters estimation or control. Heading errors are wrapped. Coverage is
empirical per-axis +/-2 sigma coverage in one correlated trajectory, not a
formal consistency test or a success-rate study. DONE means the controller's
estimated criteria were met; the summary separately checks the true final pose.

## Reproduce the estimator, controller and noise comparisons

Simulation defaults remain EKF + LQI/PBVS. Select the UKF or the PID tracker with
`--estimator ukf` and `--controller pid`. PID changes tracking and spot turns;
the search, obstacle checks, PBVS and command limits remain shared. It is ported
from Nikhil's `extension/LQI-vs-PID` branch, commit `2c0365c`.

```bash
python -m track_a_mobile_robot.main --headless --seconds 240 --stop-on-dock --estimator ukf --capture --log-dir outputs/simulation-runs/ukf
python -m track_a_mobile_robot.simulation.experiments --seeds 42 43 44 --jobs 2
python -m track_a_mobile_robot.simulation.estimator_comparison outputs/simulation-runs/ekf-baseline/RUN_TIMESTAMP
```

The experiment matrix uses both trackers, three matched seeds and nominal/5x
encoder noise. Actual absolute wheel-angle noise changes from 0.01 to 0.05 rad
standard deviation; the filter's assumed noise stays nominal. Camera/LiDAR noise,
scene, gains and PBVS settings stay fixed. Each attempted trial is retained,
including timeouts. Navigation references can change with feedback, so this is
a closed-loop system comparison, not a fixed-reference controller benchmark.
Use `--encoder-noise-scale` and `--noise-seed` for individual runs.

`--capture` saves simulation states, navigation overlays and the actual GUI sensor
panels alongside the telemetry. Recorded states can be rendered from different
camera positions without rerunning autonomy. Recording requires `--log-dir`.

The estimator comparison replays identical pre-gate camera and encoder inputs.
It verifies that EKF replay exactly reproduces the recorded baseline before
scoring the UKF. Replay changes no robot trajectory; claims about UKF docking
require a separate closed-loop run. Both filters use the same state, local-frame
initialisation, noise assumptions and innovation threshold. The UKF uses alpha=1,
beta=2, kappa=0; wheel uncertainty is augmented in prediction, with circular angle
means/residuals and semidefinite covariance handling. See
[Julier and Uhlmann (2004)](https://doi.org/10.1109/JPROC.2003.823141).
Reported replay timings measure process CPU time for filter updates, excluding
perception, rendering and logging. Per-axis 2-sigma coverage is descriptive for
correlated trajectories, not a formal consistency test.

### Recorded comparison results

The declared matrix retained all 12 trials: seeds 42, 43 and 44 for each row
below. Errors and command RMS are the mean of the three per-trial metrics.
Position estimation is scored from acquisition to first DONE; tracking and yaw
commands are scored only during DRIVE references.

| Tracker | Encoder std (rad) | Position RMSE (cm) | True cross-track RMSE (cm) | Yaw-command RMS (rad/s) | Reached DONE | True pose within tolerance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LQI | 0.01 | 6.81 | 2.10 | 0.0131 | 3/3 | 1/3 |
| LQI | 0.05 | 7.34 | 2.91 | 0.0210 | 3/3 | 0/3 |
| PID | 0.01 | 7.21 | 2.53 | 0.0157 | 3/3 | 1/3 |
| PID | 0.05 | 7.48 | 2.85 | 0.0286 | 3/3 | 0/3 |

The true-pose test uses the configured 3 cm / 3 degree tolerances at first DONE.
Every heading passed. Final position errors ranged from 2.96 to 3.17 cm;
the position misses were 0.3–1.7 mm beyond the threshold. Stopping uses the
estimated relative pose, so reporting DONE does not guarantee that truth meets
the same threshold. The higher-noise condition increased average estimation
and tracking errors for LQI, while final docking errors remained close to the
shared PBVS stopping boundary. Three seeds expose variability but do not
establish statistically significant superiority of either tracker.

The six LQI logs were also replayed through both filters, using exactly the same
inputs. EKF replay reproduced every recorded pose, covariance and camera-gating
decision. Mean position RMSE was 6.805 cm (EKF) versus 6.809 cm (UKF) for nominal
inputs, and 7.343 versus 7.318 cm for increased-noise inputs. Pooled median filter
CPU time was 0.0247 versus 0.0857 ms; the 95th percentiles were 0.1003 versus
0.2197 ms. The extra UKF computation brought little accuracy benefit in these
conditions. A separate closed-loop UKF integration check reached DONE with
3.07 cm true final position error; it is not part of the paired tracker matrix.

Local evidence is in `outputs/simulation-runs/`: the declared
`protocol.json`, all `*.result.json` records, `results.json`, `trial_metrics.csv`,
and timestamped telemetry/config/source snapshots. Each LQI run contains an
`estimator_comparison/` folder with replay arrays, metrics and source provenance.
These generated outputs are git-ignored; rerun the commands above to reproduce
them. The video shows representative error traces and the mean per-trial metrics;
the saved trial table retains every individual result.

## Run the hardware demo

Initialise the dependency submodule on a fresh checkout, then install into an
activated Python environment:

```bash
git submodule update --init --recursive
python -m pip install -r requirements-hardware.txt
python -m pip install --no-deps -e extern/turtlebot_rosbridge
```

The robot needs its drivers and rosbridge server running. Check
[hardware setup and calibration](track_a_mobile_robot/hardware/README.md) before
motion. The supplied hardware configuration includes a saved camera calibration
and an 80 mm ID 0 marker; pass the robot address with `--host`.

```bash
# Stationary check: compute navigation but send zero velocity.
python -m track_a_mobile_robot.main --backend rosbridge --host ROBOT_IP --dry-run --verbose --seconds 30

# Bounded hardware demo, with the camera preview.
python -m track_a_mobile_robot.main --backend rosbridge --host ROBOT_IP --show-camera --seconds 30
```

Hardware searches for a marker, uses LQI to reach a dock-facing offset, then PBVS
to dock. Its camera-loss recovery and timestamp alignment differ from simulation.
The current trial configuration disables LiDAR obstacle checks and accepts any
finite fitted pose for the configured marker. Review those settings in
`config/hardware_config.yml`; this cleanup does not change their values.

## Code layout

```text
track_a_mobile_robot/
  main.py                    CLI and backend selection
  sensing.py                 Timestamped camera, LiDAR and encoder readings
  perception.py              Marker poses and robot-frame obstacles
  estimation.py              Joint local robot/dock EKF
  estimation_ukf.py          Joint robot/dock UKF comparison
  navigation.py              Search, trajectories, avoidance and docking phases
  control.py                 LQI, PBVS and velocity/wheel limits
  control_pid.py             PID comparison with shared PBVS and limits
  state_model.py             Differential-drive model and Jacobians
  scene_config.py            Typed simulation/shared algorithm settings
  hal/
    __init__.py              RobotBackend I/O contract
    mujoco.py                Simulated sensing and wheel actuation
    rosbridge.py             ROS message conversion, commands and watchdog
  simulation/
    scene.py                 YAML placement and MuJoCo model construction
    runner.py                Simulation timing and autonomy loop
    display.py               Camera, LiDAR and viewer overlays
    evaluation.py            Opt-in telemetry, truth scoring and report plots
    capture.py               Actual GUI panels and simulation replay states
    experiments.py           Matched controller and encoder-noise trials
    estimator_comparison.py  Identical-input EKF/UKF replay and timing
    generate_markers.py      Regenerate the dock's texture assets
  hardware/
    runner.py                Real-robot autonomy loop
    config.py                Hardware calibration and config validation
    navigation.py            Hardware search and two-phase docking
    sensor_sync.py           Delayed-camera/encoder alignment
    preview.py               Camera preview process
    run_log.py               Automatic JSONL telemetry
    calibrate_camera.py      Checkerboard capture and calibration
    check_aruco.py            Camera-only marker/distance check
    check_timing.py           Sensor delivery timing check
    assets/                  Printable checkerboard and marker PDFs
config/                      Simulation and hardware YAML files
world/                       Room, furniture, dock and textures
robot_model/                 TurtleBot model and meshes
extern/turtlebot_rosbridge/   Separate rosbridge dependency submodule
```

The HAL owns acquisition, timestamps, cached readings, motor commands and cleanup.
Its adapters expose `start`, `now`, `read`, `health`, `send_command`, `stop` and
`close`. Each runner owns its clock/scheduling and passes sensor snapshots into
shared perception and estimation. Simulation advances physics through its adapter;
hardware retains the command watchdog and source-timestamp checks.

World placement stays in scene construction. The GUI reads simulator truth only
for drawing. Perception, estimation, navigation and control receive measurements
and static calibration, never world spawn poses. Hardware imports do not load
MuJoCo or the simulation viewer.

The repository contains the demo, setup utilities and reproducible evaluation
tools. Generated captures,
calibration results and telemetry remain under the git-ignored `outputs/` folder.
The dependency submodule retains its own upstream files.

## Assets and attribution

The two simulation markers are `DICT_4X4_50`: ID 0 has a 0.40 m black square and
ID 1 has a 0.04 m black square. Their measured mounting offsets are in
`perception.marker_mounts()`. To regenerate their checked-in textures:

```bash
python -m track_a_mobile_robot.simulation.generate_markers
```

The robot model is adapted from
[ROBOTIS's MuJoCo collection](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/d8344c0dbe7a00208d0301111523dde65efc174a/robotis_tb3).
Its licence is retained in `robot_model/turtlebot3_waffle/LICENSE`. The chassis
collision mesh is raised 2 mm and rear supports use frictionless normal contacts
to avoid floor drag; this is a simplified support model.

The dock geometry comes from the
[iRobot Create 3 simulator](https://github.com/iRobotEducation/create3_sim/blob/jazzy/irobot_create_common/irobot_create_description/urdf/dock/standard_dock.urdf.xacro).
Its licence and conversion notes remain under `world/assets/create3_dock/`.
It is a stationary prop; electrical charging/contact is not modelled.
