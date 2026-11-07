# Sensors and Control — Track A

Autonomous TurtleBot3 docking in MuJoCo and on a real Waffle Pi. Both demos use
camera/LiDAR perception, an encoder/camera EKF, LQI navigation and PBVS docking.
The robot starts without a known world pose or dock location. The first accepted
marker establishes a local odometry frame; later observations correct that estimate.

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
  navigation.py              Search, trajectories, avoidance and docking phases
  control.py                 LQI, PBVS and velocity/wheel limits
  state_model.py             Differential-drive model and Jacobians
  scene_config.py            Typed simulation/shared algorithm settings
  hal/
    __init__.py              RobotBackend I/O contract
    mujoco.py                Simulated sensing and wheel actuation
    rosbridge.py             ROS message conversion, commands and watchdog
  simulation/
    scene.py                 YAML placement and MuJoCo model construction
    runner.py                Simulation timing and autonomy loop
    display.py               Camera, map and viewer overlays
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

The repository contains the demo and its setup utilities. Development test files
and offline simulation diagnostics have been removed. Generated captures,
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
