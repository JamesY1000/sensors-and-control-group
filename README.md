# Sensors and Control — Track A

A MuJoCo simulation of the **TurtleBot3 Waffle Pi** in an **8 × 6 m room** with
two tables and four chairs as fixed obstacles, plus a **Create 3 charging dock
with an elevated ArUco panel**. By default the robot starts at (−0.5, −2.5),
in the southwest corner, facing northeast (45°) into the room. The dock is near
the east wall, facing -X. Both starting poses are configurable.

## Current capabilities

| Component | Current setup |
| --- | --- |
| Wheel encoders | Accumulated wheel angles in radians; ideal readings and Gaussian noise with σ = 0.01 rad |
| 2D LiDAR | 360 beams, 360° coverage, 0.16–8 m range, 5 Hz; based on the LDS-02 |
| RGB camera | Pi Camera v2-based model, 640 × 480 at 30 Hz, 48.8° vertical field of view |

LiDAR and camera measurements are ideal; their noise and other hardware effects
are not modelled yet. Camera resolution and frame rate are simulation choices.
The pose EKF samples the wheel angles at a chosen **50 Hz** and predicts
`[x, y, yaw]` and its 3×3 covariance. It runs in both viewer and headless modes,
printing the pose, standard deviations and an evaluation-only comparison with simulator truth once
per simulation second and at completion. It starts from the configured, known
robot pose; resetting simulation time restores that pose and starts a new encoder baseline.
The launcher also prints LiDAR summaries and can save a camera image.

Prediction uses noisy encoders by default. Camera-pose correction is implemented
in `PoseEKF.correct()`, but camera localisation is not connected yet, so current
simulation runs remain encoder-only. Navigation and feedback control can be
tested separately using the true-pose diagnostic below.
Even ideal encoders can show pose error because the kinematic model approximates
MuJoCo's physical motion.
The station is fixed to the world and contains the original Create 3 visual
mesh, a simple housing collision box, and a supported marker panel above it.
The navigation diagnostic targets an approach pose in front of the marker;
electrical charging is not simulated.

Initial pose standard deviations are **1 cm, 1 cm, 2°**. Additional motion-model
noise uses **1 cm/√s, 1 cm/√s, 5°/√s**; these are starter assumptions to tune in
`main.py`, not calibrated uncertainty. Encoder angle noise comes from the XML
(zero with `--ideal-encoders`); ideal encoders still retain motion-model uncertainty.
The filter maps wheel-increment variance `2σ²` into pose covariance. This is an
approximation that ignores correlation between consecutive angle differences.

## Run the simulation

Use **Python 3.12 or newer**. For a fresh checkout, run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows PowerShell, use `python` instead of `python3` and activate the
environment with `.venv\Scripts\Activate.ps1`.

With the environment active, launch the viewer:

```bash
python3 -m track_a_mobile_robot.main
```

The robot stays stationary until wheel commands are applied. Use the mouse to
inspect the room and the **Control** panel to adjust wheel velocity targets.

| Key | Action |
| --- | --- |
| Space | Pause/resume |
| Y | Toggle LiDAR rays |
| ] / Esc | Robot camera / free camera |
| F4 | Toggle native sensor plots (ideal values) |

Useful commands:

```bash
# Check loading, physics and sensors without a viewer.
python3 -m track_a_mobile_robot.main --headless --seconds 2

# Apply equal wheel velocity targets for a short forward-motion check.
python3 -m track_a_mobile_robot.main --headless --seconds 2 --left-wheel 4 --right-wheel 4

# Check encoder-only odometry with clean readings first.
python3 -m track_a_mobile_robot.main --headless --seconds 2 --left-wheel 4 --right-wheel 4 --ideal-encoders

# Sample the camera and save its latest frame as a PNG.
python3 -m track_a_mobile_robot.main --headless --seconds 2 --camera-output outputs/pi_camera_v2.png
```

Wheel targets are in **rad/s**, within ±7.88. Omit `--headless` from the motion
command to watch it in the viewer; `--seconds` only limits headless runs.
Encoder noise uses seed 42 by default; change it with `--seed 7`.

Camera capture requires graphics support, even with `--headless`. On Linux
without a display, prefix the capture command with `MUJOCO_GL=egl` if EGL is
available. Physics-only headless checks need no graphics context. The viewer's
refresh rate is independent of the Python camera's 30 Hz capture rate.
Generated images are saved under `outputs/`, which Git ignores.

### Navigation/control diagnostic

The separate diagnostic uses the simulator's true robot pose to test navigation
and LQI control without EKF drift. It reads the same scene configuration and
configured landmark map, searches for a marker, and approaches it:

```bash
python3 -m scripts.run_navigation_true_pose
MUJOCO_GL=egl python3 -m scripts.run_navigation_true_pose --headless --seconds 2
```

Use `--config config/my_layout.toml` to select another spawn configuration.
The navigator assumes an 8 × 6 m room with its long axis along world X; robot
spawn heading can vary. The short headless command checks startup and control,
not mission completion. The regular `track_a_mobile_robot.main` launcher keeps
its manual wheel controls and encoder-only estimator.

## Configure the starting positions

Edit **`config/scene.toml`**, then restart the simulation:

```toml
[robot]
x = -0.5
y = -2.5
yaw_deg = 45.0

[charger]
x = 6.9
y = 0.0
yaw_deg = 180.0
```

`x` and `y` are world coordinates in **metres**; `yaw_deg` is counterclockwise
heading in **degrees**. Zero points along +X (east), 90 along +Y (north), and
180 along -X (west). Robot heading points forward; charger heading points out
of the dock's front towards an approaching robot. Its position is the dock's
ground-level model origin, not the desired final robot pose.

The charger, post, panel and marker frame form one assembly, so they translate
and rotate together. Ground height and marker mounting geometry stay in the
MJCF model. Choose clear floor space inside the room, allowing for each object's
footprint; configuration validates field names and finite numbers, not collisions
or whether a position is reachable.

TOML supports comments and is read by Python's standard library. No YAML package
is needed. `scene_config.py` reads it; `main.py` applies the poses before MuJoCo
compiles the scene and uses the same robot pose to initialise the EKF. Viewer
reset restores the configured poses. To keep several layouts, copy the file and run:

```bash
python3 -m track_a_mobile_robot.main --config config/my_layout.toml
```

The default file is found relative to the repository, independent of the current
directory. An explicit `--config` path is relative to your current directory.
Loading the XML directly in a separate MuJoCo viewer uses its default poses;
use the Python launcher to apply the TOML configuration.

## Room and furniture

The room interior spans **x = −1 to 7 m** and **y = −3 to 3 m**. Two tables sit
at (2, 1.55) and (4.5, −1.55), each with two chairs facing the table. The robot
starts 0.5 m in from the west and south walls, at (−0.5, −2.5). A 0.94 m wide
strip along y = 0 is free of furniture and leads to the dock at (6.9, 0);
navigate from the corner into this central approach. The low walls keep the
scene easy to inspect from above.

Each table is 1.4 × 0.8 m and 0.75 m tall; each chair is 0.46 × 0.44 m and
0.85 m tall. Separate legs, seats, backs and tabletops have their own collision
shapes. The low LiDAR scan sees the legs, and the robot can move under furniture
where its full body fits between the legs and beneath the seat or tabletop.
Furniture is fixed in place; the robot cannot push it around.

Edit the placement frames in `world/room_with_charger.xml` to rearrange the
tables and chairs. Their reusable geometry and materials live in
`world/furniture/table.xml` and `world/furniture/chair.xml`. Robot and dock
starting poses remain in `config/scene.toml`.

## Marker panel and perception

Keep the **marker on a panel**, including its white margin: ID 0 from
`DICT_4X4_50`, a **20 cm black square** inside a **28 cm white panel**. The panel
centre is 26 cm above the floor, above the dock housing, on a narrow support.
The size of the black square matches `MARKER_SIDE_M` used for pose estimation.
With the robot's low, forward-facing camera, the raised marker will leave the
image at sufficiently close range; plan the final docking stage accordingly.
The central approach has a clear view of the panel. From the default corner
start, the robot needs to turn towards the dock and approach before the marker
is large enough in the image for reliable detection.

The marker's installed world pose is derived from the configured model's fixed
body/site transforms, so changing the dock pose also changes its known landmark
map. This uses model constants, not a live measurement of the robot's true pose.
Pass that map explicitly to perception helpers:

```python
with Perception(model) as perception:
    observations = perception.update(data)
    marker_map = perception.marker_map
    expected = expected_range_bearing(estimated_pose, marker_map=marker_map)
    # robot_pose_from_marker(observation, marker_map)
    # range_bearing_jacobian(estimated_pose, marker_map=marker_map)
```

The diagnostic tools use the same configuration. Put `--config` before the subcommand:

```bash
MUJOCO_GL=egl python3 -m track_a_mobile_robot.perception --config config/scene.toml test --repeats 3
```

Regression checks for configured poses, reset behaviour and rendered marker detection:

```bash
MUJOCO_GL=egl python3 -m unittest discover -s tests -v
```

## Key files

- `config/scene.toml`: robot and charger assembly starting positions/headings.
- `track_a_mobile_robot/scene_config.py`: typed configuration and validation.
- `world/room_with_charger.xml`: floor, walls, obstacle layout and lighting.
- `world/furniture/`: reusable table and chair models with individual collision legs.
- `world/charging_station.xml`: dock, post, marker panel and fixed marker frame.
- `world/assets/create3_dock/`: dock meshes, source details and retained licence.
- `robot_model/turtlebot3_waffle/turtlebot3_waffle_pi.xml`: robot geometry,
  joints, actuators and sensor definitions; `assets/` holds its meshes.
- `track_a_mobile_robot/main.py`: model loading, LiDAR timestamp setup,
  viewer/headless execution and command-line options.
- `track_a_mobile_robot/sensing.py`: encoder and LiDAR readings, RGB capture
  and camera calibration.
- `track_a_mobile_robot/perception.py`: ArUco detection, configured landmark map,
  LiDAR frame transforms and perception diagnostics.
- `track_a_mobile_robot/state_model.py`: wheel geometry, midpoint pose prediction
  and its Jacobians for covariance propagation.
- `track_a_mobile_robot/estimation.py`: `PoseEKF.predict()` / `correct()` and
  `StateEstimate` snapshots containing timestamp, world pose and covariance.
- `track_a_mobile_robot/evaluation.py`: simulator truth and pose-error calculations,
  used only for reporting results.
- `track_a_mobile_robot/navigation.py`: room search and marker approach references.
- `track_a_mobile_robot/control.py`: LQI feedback and conversion to wheel speeds.
- `scripts/run_navigation_true_pose.py`: navigation/control diagnostic using true pose.

The future perception connection is `correct(camera_pose_world, camera_covariance,
time)`: supply the robot's world-frame `(x, y, yaw)` and a positive-definite 3×3
measurement covariance in that order. Predict to the image capture time first;
correction rejects mismatched timestamps and repeated frames. The 30 Hz camera
and 50 Hz encoder schedules will need alignment when perception is connected.
If no marker pose is available, continue prediction without correction.

## Model source

The robot is adapted from
[ROBOTIS's MuJoCo collection](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/d8344c0dbe7a00208d0301111523dde65efc174a/robotis_tb3),
commit `d8344c0dbe7a00208d0301111523dde65efc174a`. Sensor definitions have been
added; robot geometry and physical parameters are unchanged. The upstream
licence is retained in `robot_model/turtlebot3_waffle/LICENSE`.

Sensor references: [LDS-02](https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_lds_02/)
and [Pi Camera v2](https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_raspi_cam/).

The dock geometry comes from the
[iRobot Create 3 simulator](https://github.com/iRobotEducation/create3_sim/blob/jazzy/irobot_create_common/irobot_create_description/urdf/dock/standard_dock.urdf.xacro).
Its BSD-3-Clause licence and conversion notes are retained under
`world/assets/create3_dock/`. The elevated marker mount is specific to this
simulation. The Create 3 dock is used as a stationary prop for the Waffle Pi;
this does not model a compatible real electrical charging interface.

## Project workflow (quick start)

1. Pull the latest changes from `main`.
2. Create a branch for your task, using one of the prefixes below.
3. Develop and test locally using the [setup above](#run-the-simulation), or in
   the team's devcontainer if one is provided.
4. Open a pull request (PR) with a clear description of what changed and why.
5. Get at least one teammate review.
6. Merge only when the relevant checks pass.

This repository does not currently include a devcontainer configuration. The
documented setup uses a local Python virtual environment. For simulation changes,
run the headless physics check above and any other checks relevant to your task.

## Branching and commit conventions

Use one branch per issue or feature, with a descriptive name such as
`docs/team-workflow`.

Suggested branch prefixes:

- `feature/` — new functionality.
- `fix/` — bug fixes.
- `test/` — tests and validation.
- `docs/` — documentation changes.

Commit messages should:

- Be concise and descriptive.
- Start with an action verb.
- Mention the subsystem when relevant.

Example: `Add headless simulation validation`.

## Pull request guidelines

Before opening a PR:

- Rebase your branch onto the latest `main`, or merge the latest `main` into it.
- Confirm your branch builds and runs successfully, and run the relevant checks.
- Update documentation if behavior, dependencies, or setup changed.

Each PR should have:

- A clearly defined, focused scope.
- A clear summary of what changed and why.
- A description of the checks performed and their results.
- At least one teammate review before merging.

## Team rules

- Never push directly to `main`.
- Pull the latest `main` before starting work.
- Use one branch per issue or feature.
- Use descriptive branch names and commit messages.
- Keep pull requests focused and reasonably small.
- Have at least one teammate review each pull request.
- Do not merge code that does not build or fails the relevant checks.
- Merge frequently instead of integrating everything near the deadline.
- Document dependency assumptions and setup procedures.
- Never commit secrets, generated build folders, or large datasets.

## Definition of done

A task is done when:

- [ ] The implementation works and has been reviewed by at least one teammate.
- [ ] Relevant checks pass, and the change does not break other parts of the system.
- [ ] Relevant documentation is updated, including dependency and setup changes.
