# Sensors and Control — Track A

A MuJoCo simulation of the **TurtleBot3 Waffle Pi** in a **6 × 4 m room** with
three fixed obstacles: two boxes and a cylinder. The robot starts at (0, 0),
facing +X, with space to drive around the obstacles.

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
per simulation second and at completion. It starts from the room's known
`(0, 0, 0)` pose; resetting simulation time starts a new encoder baseline.
The launcher also prints LiDAR summaries and can save a camera image.

Prediction uses noisy encoders by default. Camera-pose correction is implemented
in `PoseEKF.correct()`, but camera localisation is not connected yet, so current
simulation runs remain encoder-only. Navigation and feedback control remain to be added.
Even ideal encoders can show pose error because the kinematic model approximates
MuJoCo's physical motion.
There is no charging station or ArUco marker yet, despite the world filename.

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

## Key files

- `world/room_with_charger.xml`: floor, walls, obstacle layout and lighting.
- `robot_model/turtlebot3_waffle/turtlebot3_waffle_pi.xml`: robot geometry,
  joints, actuators and sensor definitions; `assets/` holds its meshes.
- `track_a_mobile_robot/main.py`: model loading, LiDAR timestamp setup,
  viewer/headless execution and command-line options.
- `track_a_mobile_robot/sensing.py`: encoder and LiDAR readings, RGB capture
  and camera calibration.
- `track_a_mobile_robot/state_model.py`: wheel geometry, midpoint pose prediction
  and its Jacobians for covariance propagation.
- `track_a_mobile_robot/estimation.py`: `PoseEKF.predict()` / `correct()` and
  `StateEstimate` snapshots containing timestamp, world pose and covariance.
- `track_a_mobile_robot/evaluation.py`: simulator truth and pose-error calculations,
  used only for reporting results.
- Navigation and control modules are placeholders for later work.

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
