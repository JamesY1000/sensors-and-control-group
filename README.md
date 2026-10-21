# Sensors and Control — Track A

A MuJoCo simulation of the **TurtleBot3 Waffle Pi** in an **8 × 6 m room** with
two tables and four chairs as fixed obstacles, plus a **Create 3 charging dock
with approach and close-range ArUco panels**. By default the robot starts at (−0.5, −2.5),
in the southwest corner, facing +X (0°). The dock is near
the east wall, facing -X. Both starting poses are configurable.

## Current capabilities

`main.py` runs sensing, perception and the pose EKF in both the viewer and
headless mode. Each scheduled update reads noisy wheel encoders, predicts
`[x, y, yaw]` and its 3×3 covariance, then corrects with a fresh, reliable
camera-derived robot world pose when one is available.

| Component | Current setup |
| --- | --- |
| Wheel encoders | Accumulated wheel angles in radians; Gaussian angle noise from the sensor XML (σ = 0.01 rad by default) |
| 2D LiDAR | 360 beams, 360° coverage, 0.16–8 m range, 5 Hz; based on the LDS-02 |
| RGB camera | Pi Camera v2-based model, 640 × 480 at 30 Hz; vertical field of view and mounting set in YAML |
| Shared update loop | 50 Hz by default, set by `estimation.update_hz` |

The shared loop keeps encoder prediction and camera acquisition/correction at
the same timestamp. Camera and LiDAR readings retain their own acquisition
rates; cached images are not repeatedly fused. Encoder noise and camera/LiDAR
noise use independent random streams derived from the configured seed.

The terminal prints estimated world pose, one-standard-deviation uncertainty
(`sigma`), and camera corrections/frames about once per simulation second.
A zero correction count is normal when the marker is out of view or its pose
is ambiguous. The estimator uses the configured robot pose as a declared
starting-pose prior and the installed dock as a known landmark map. The main
loop does not read live simulator robot pose or print a ground-truth comparison.
Rewinding simulation time recreates the estimator and sensor sampling state.

By default the wheels stay stopped for sensor checks. **Use `--navigate` to
enable autonomous approach and docking.** Navigation plans from the estimated
world pose and LiDAR points; the controller tracks its references using that
same estimate. LiDAR does not correct the EKF. Electrical charging is not simulated.

## Run the simulation

Use **Python 3.12 or newer**. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows PowerShell, use `python` instead of `python3` and activate the
environment with `.venv\Scripts\Activate.ps1`.

```bash
# Open the viewer and run sensing/estimation until the window closes.
python3 -m track_a_mobile_robot.main

# Enable autonomous approach, alignment and docking.
python3 -m track_a_mobile_robot.main --navigate

# Run the same sensor/estimator loop without the viewer for five seconds.
python3 -m track_a_mobile_robot.main --headless --seconds 5

# An explicit configuration path is optional; this is the default file.
python3 -m track_a_mobile_robot.main --config config/simulation_config.yml
```

The launcher accepts `--config`, `--headless`, `--navigate` and `--seconds`. Without
`--seconds`, headless runs last two seconds and viewer runs continue until the
window closes. Wheel commands come from the controller with `--navigate`, or
stay at zero otherwise. Viewer actuator sliders do not override these commands.

Viewer synchronisation is limited to 30 Hz of wall-clock time instead of every
physics step. Physics still runs at 500 Hz, estimation/control at the configured
50 Hz, and the camera at 30 Hz of simulation time. The loop accounts for sensor
processing time when pacing the simulation, sleeping only when it is ahead of
real time. Headless runs do not apply this pacing limit.

The viewer includes a live robot-camera inset at the bottom-left of the scene.
It shows the same image used by perception, with marker outlines, IDs and axes.
Green outlines indicate reliable marker orientation; orange indicates ambiguity.
The header shows the image timestamp and whether that frame supplied an EKF
correction, had a marker pose that was not accepted, or had no usable marker.
The preview stays visible between camera frames and scales to fit the window;
it is hidden when the scene area is too small. Headless runs do not create a preview.

The viewer also draws a floor overlay using the estimated world frame:

- Cyan lines and dots: the remaining waypoint route, or final docking segment.
- Orange dots: the latest LiDAR hit positions projected onto the floor; grey when stale.
- Yellow marker and heading line: the current trajectory reference.
- Blue line and circle: estimated robot heading and navigation stop clearance.
- Green and purple markers: the approach and final docking targets.

A legend and navigation status appear at the top-left. The route disappears
when no valid route is available; a stopped reference is not drawn as a motion
target. LiDAR hits are measurements, not the planner's inflated obstacle grid.
The overlay refreshes at 10 Hz, caches each scan's world points, and uses the
[MuJoCo viewer's separate `user_scn`](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).
It does not add physical obstacles or appear in the robot's camera images.
Without `--navigate`, the viewer still shows LiDAR hits and estimated heading.

Headless runs still render camera images. On Linux, the launcher selects EGL
for the offscreen camera in both modes unless `MUJOCO_GL` is already set. This
keeps camera rendering separate from the viewer's GLFW context. A working
offscreen graphics backend is required; the viewer uses the desktop graphics backend.

To inspect camera corrections near the dock, set the YAML robot pose to
`x: 6.0`, `y: 0.0`, `yaw_deg: 0.0`, then rerun. The default corner start may
remain prediction-only while stationary because the markers are too far away
or outside the camera view.

## Navigation and control

1. **APPROACH:** derive the approach pose from the configured dock pose. Take a
   direct path if clear; otherwise run A* through a grid inflated by robot
   clearance and simplify the result into waypoints. Check the next segment
   against incoming LiDAR scans and replan if it becomes blocked.
2. **ALIGN:** face into the dock and require consecutive reliable camera-pose
   observations. A raw or ambiguous marker detection does not confirm the dock.
3. **DOCK:** follow a slow straight reference along the dock centreline. Pause
   when reliable camera observations become stale or LiDAR reports an obstruction.
4. **DONE:** stop once estimated position and heading are both within tolerance.

Each straight segment begins with an in-place turn. Navigation generates a
time-varying pose and velocity reference with acceleration/braking and a limited
lead over the robot; the LQI controller tracks translating segments. Alignment
uses heading feedback with integral action and zero forward velocity. Wheel commands obey the
actuator speed limits. Navigation decides where to go; control decides how to
track the reference. Neither reads the live ground-truth robot pose.

The default approach and final robot-base targets are 0.50 m and 0.30 m out
from the dock origin. Both markers confirm the same dock target. The final
offset leaves room for the robot footprint and noisy LiDAR returns from the
dock; it represents a geometric docking demonstration, not electrical contact.

This is a small planner using the latest scan, not a persistent occupancy map.
It prefers an extra half-cell of planning clearance. If that search fails, it
retries with the configured robot clearance (27 cm by default); it never drops
below that safety distance. Exact start/goal positions connect to the grid via
collision-checked segments, so rounding an endpoint into an inflated cell does
not itself make a safe endpoint unreachable. A blocked destination is reported
as `Approach target obstructed`.
Unobserved space is provisionally traversable. It stops on blocked routes,
stale scans or obstacles inside its clearance and does not implement automatic
recovery manoeuvres. Long occlusions and encoder drift still need full-room
validation across different starts and noise seeds. Run the navigation and
camera-correction checks with:

```bash
python3 -m unittest discover -s tests -v
```

Before the chassis/support contact correction, rendered trials used seed 42
and configured sensor noise: the near-dock start
`(6.0, 0.0, 0°)` reached `DONE` at 6.12 simulated seconds. At that transition,
estimated position error was 2.98 cm; the separate simulator-truth score was
3.52 cm and 0.08° heading error. A second trial starting at 10° reached `DONE`
at 10.14 s, with a truth-scored position error of 3.51 cm and heading error of
0.36°. These two trials are not a success-rate estimate.
The earlier default corner-start trial advanced briefly, then stopped with a blocked
route before obtaining a camera correction. Full-room completion is therefore
not established; turning odometry and planning under that uncertainty need
further work. These results predate the contact correction described below;
they are not validation of the corrected model. The runtime still uses only
the estimated pose.

With the support-contact correction, a 15 s rendered trial using the default
configuration turned and drove away from the starting corner, reaching an
estimated `(1.19, -1.68)` m at 14.12 s without the previous blocked-goal stop.
It had not received a camera correction by 15 s. This verifies the startup
improvement, not completion of the full-room route.
A corrected-model near-dock trial from `(6.0, 0.0, 0°)` reached `DONE` at
5.96 s, with 2.89 cm estimated position error and a truth-scored error of
2.93 cm and 0.02°. Both rendered trials used seed 42 and the existing sensor
noise settings.

After adding physical camera-pose validation, EKF innovation gating and the
planning-margin fallback, the default corner start completed the rendered
route at 56.74 simulated seconds with seed 42. At `DONE`, estimated target
error was 2.83 cm; the separate truth score was 2.90 cm and 0.045 degrees.
Truth was used only for scoring, never for control or measurement acceptance.
All 20 unit tests passed. This is one successful route, not a success-rate claim.
A separate `(6.0, 0.0, 0°)` trial remained in `ALIGN` at 20 seconds: the existing
IPPE ambiguity filter rejected 115 of 120 frames sampled between 4 and 8 s,
while none failed the new height/tilt checks. Near-frontal marker ambiguity and
viewpoint recovery remain limitations of the current docking confirmation logic.

## Earlier experiment results

The results below were recorded with the earlier autonomous range/bearing
fusion pipeline, before the current sensing/estimation rebuild. They are
historical measurements, not validation of the current full-pose EKF loop.
The earlier autonomous launcher and its command-line flags are not supported
by the current `main.py`.

That implementation passed a close-range estimated-state docking trial with
injected noise. Full-room completion was not established: a 60 s default-route
trial accepted 314 visual corrections, then lost the marker during obstacle
avoidance and accumulated severe pose drift before timing out.

Recorded seed-42 short trials used the installed markers, a known initial pose,
identical encoder samples for fused and odometry estimates, and the earlier
noise tuning. Values are RMSE:

| Motion | Encoder position | Fused position | Encoder heading | Fused heading |
| --- | --- | --- | --- | --- |
| Straight, 8 s | 3.44 cm | 0.34 cm | 1.18° | 0.15° |
| Arc, 8 s | 2.28 cm | 0.94 cm | 7.78° | 2.28° |
| Spin, 2 s | 0.94 cm | 2.32 cm | 7.13° | 1.21° |

Fusion improved heading in these trials but did not improve every position
metric. These short simulation measurements do not establish hardware accuracy,
full-room docking success or robustness to long marker occlusions.

## Simulation configuration

Edit **`config/simulation_config.yml`**, then restart the simulation. This single file
controls the robot and dock starting poses, camera mounting and field of view,
LiDAR mounting, estimator timing/uncertainty, and navigation settings:

```yaml
robot:
  x: -0.5
  y: -2.5
  yaw_deg: 0.0
  # For a near-dock start, use x: 6.0, y: 0.0, yaw_deg: 0.0.

dock:
  x: 6.9
  y: 0.0
  yaw_deg: 180.0

camera:
  position_m: [0.076, 0.0, 0.103]
  rpy_deg: [-90.0, 0.0, -90.0]
  fov_y_deg: 48.8

lidar:
  position_m: [-0.049, 0.0, 0.1255]
  yaw_deg: 0.0

estimation:
  update_hz: 50.0
  noise_seed: 42
  initial_position_std_m: 0.05
  initial_heading_std_deg: 5.0
  process_position_std_m_per_sqrt_s: 0.01
  process_heading_std_deg_per_sqrt_s: 2.0

navigation:
  approach_distance_m: 0.5
  dock_distance_m: 0.30
  drive_speed_mps: 0.15
  dock_speed_mps: 0.06
  turn_speed_rad_s: 0.4
  acceleration_mps2: 0.25
  clearance_m: 0.27
  grid_resolution_m: 0.1
  position_tolerance_m: 0.03
  heading_tolerance_deg: 3.0
  camera_timeout_s: 0.5
  lidar_timeout_s: 0.5
  camera_confirmations: 3
  reference_lead_m: 0.12
```

Robot and dock `x`/`y` are world coordinates in **metres**. Their `yaw_deg` is
counterclockwise heading: 0 faces +X (east), 90 faces +Y, and 180 faces -X.
The dock origin is at ground level, and its local +X points out of its front.
The dock, post and markers move together. For the near-dock setup, edit only
the robot entry to `(6.0, 0.0, 0.0)`; no second configuration file is needed.

Sensor positions are in the robot base frame: +X forward, +Y left, +Z up.
Camera `rpy_deg` rotates the OpenCV optical frame into that base frame using
`Rz(yaw) @ Ry(pitch) @ Rx(roll)`. The default angles align optical +X right,
+Y down and +Z forward with the robot. `fov_y_deg` is the vertical field of view.
LiDAR stays horizontal; its `yaw_deg` rotates its forward direction around
the base's +Z axis.

All six `estimation` fields are required:

| Field | Meaning |
| --- | --- |
| `update_hz` | Shared sensing/perception/EKF updates per second; positive and no faster than physics |
| `noise_seed` | Nonnegative integer seed for repeatable independent encoder and perception noise streams |
| `initial_position_std_m` | Initial standard deviation for each of x and y, in metres; must be positive |
| `initial_heading_std_deg` | Initial heading standard deviation, in degrees; must be positive |
| `process_position_std_m_per_sqrt_s` | Model-noise standard deviation for each of x and y, in m/√s; may be zero |
| `process_heading_std_deg_per_sqrt_s` | Model-noise heading standard deviation, in degrees/√s; may be zero |

These standard deviations are **initial tuning assumptions, not calibrated
accuracy**. The launcher converts headings to radians and squares the standard
deviations to form the diagonal initial covariance and process covariance per
second. Prediction multiplies the latter by elapsed time. Encoder angle noise
comes separately from the sensor XML; the filter uses increment variance
`2σ²`, an approximation that ignores correlation between successive angle
differences. Camera-pose covariance is described below.

`scene_config.py` reads and validates the YAML. `main.load_spec()` applies it
before MuJoCo compiles the scene. `sensing.py` and `perception.py` then derive the
camera intrinsic matrix and fixed sensor transforms from that model, so the rendered
camera, LiDAR rays and perception calculations use the same settings. A
checkerboard calibration is not needed to run this simulated camera; the
offline calibration tools provide additional checks.

The same YAML robot pose also supplies the estimator's starting-pose prior;
the `estimation` section supplies its uncertainty. `main.update_demo()` connects
sensing, perception and estimation. With `--navigate`, the main loop then runs
navigation and control at the same rate. Navigation distances and speeds are
in metres and seconds; `turn_speed_rad_s` is in radians per second. Clearance
is measured from the robot base, including its footprint and a margin. Camera
and LiDAR timeouts determine when missing observations pause motion. The lead
limit keeps the trajectory reference from running far ahead of the estimate.
Marker mounting geometry and room furniture remain in the XML files. Allow
clearance for the robot and dock; valid configuration values do not guarantee
a collision-free placement or route.

```bash
python3 -m track_a_mobile_robot.main --config config/simulation_config.yml
```

The launcher uses this file by default, relative to the repository. An explicit
`--config` path is relative to the current directory. Loading XML directly in a
separate MuJoCo viewer bypasses these YAML settings.

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
starting poses and sensor settings remain in `config/simulation_config.yml`.

## Marker panel and perception

The dock uses two markers from `DICT_4X4_50`, each with a white margin:

| Marker | Black square | White panel | Margin per side | Centre above floor |
| --- | --- | --- | --- | --- |
| ID 0: approach | 20 × 20 cm | 28 × 28 cm | 4 cm | 26 cm |
| ID 1: final approach | 4 × 4 cm | 6 × 6 cm | 1 cm | 12 cm |

Pose estimation uses the **black square size**, selected by ID from
`MARKER_SIZES_M`; the surrounding white panel is not included in that size.
ID 1's face is at dock-local `(−0.025, 0, 0.120)` m. The camera mounting is
unchanged. Earlier installed
render checks detect ID 1 from 0.9 m through 0.2 m along the tested central
approach and at the final pose with ±2 cm lateral / ±3° heading offsets.
The original large panel leaves the image at close range; the low marker
provides the handover. These visibility checks do not establish docking success.
The central approach has a clear view of the panel. From the default corner
start, the robot needs to turn towards the dock and approach before the marker
is large enough in the image for reliable detection.

The marker's installed world pose is derived from the configured model's fixed
body/site transforms, so changing the dock pose also changes its known landmark
map. This uses model constants, not a live measurement of the robot's true pose.
`Perception` uses this map to turn a camera-relative marker pose into a robot
world-pose measurement. `main.update_demo()` reads noisy encoders, predicts the
pose, calls `perception.update(data)`, then passes a fresh `pose_observation` to
`estimator.correct()`. It returns the estimate and perception output for
navigation and control.

Prediction precedes correction at the camera's acquisition timestamp. Repeated polls, hidden
markers and ambiguous orientations produce no new pose measurement. If both
markers provide reliable poses, perception selects the larger image footprint for
one correction. Raw detections remain available in `observations.markers`.

Before extracting a planar pose, perception rejects solutions placing the base
origin more than 5 cm above/below the floor or tilting its upright axis by more
than 10 degrees. These tolerances express this demo's flat-floor model; they
do not read the robot's live simulator pose. Low reprojection error and a good
IPPE ambiguity ratio alone are insufficient for accepting a distant marker.

The EKF also gates the wrapped pose innovation using its combined prediction and
measurement covariance: `innovation.T @ solve(P + R, innovation) <= 11.345`
(the 99% chi-squared threshold for three components). A rejected frame leaves
the prediction and covariance unchanged, and is not counted as a correction or
a dock confirmation. The thresholds and camera covariance remain initial tuning
assumptions, not a guarantee that every outlier will be rejected.

Camera covariance is an initial tuning model in `[x, y, heading]` order:
translation and heading uncertainty grow for small/distant detections, and
heading uncertainty also contributes to world-position uncertainty. The
2 cm translation and 5 degree heading floors are assumptions to validate,
not measured sensor accuracy. Optional sensor noise is sampled once per
image/scan and cached between updates. `observations.lidar_too_close` preserves
below-minimum-range hits for navigation to handle.

The diagnostic tools use the same configuration. Put `--config` before the subcommand:

```bash
MUJOCO_GL=egl python3 -m scripts.perception_diagnostics --config config/simulation_config.yml test --repeats 3
```

Use `calibrate` for the checkerboard and mounting checks, or `sensitivity` for
mounting-error experiments. These offline checks use simulator truth only to
set up trials and score measurements; runtime perception does not read it.

## Key files

- `config/simulation_config.yml`: scene/sensor settings, estimator uncertainty and navigation limits.
- `track_a_mobile_robot/scene_config.py`: typed YAML configuration and validation.
- `world/room_with_charger.xml`: floor, walls, obstacle layout and lighting.
- `world/furniture/`: reusable table and chair models with individual collision legs.
- `world/charging_station.xml`: dock, post, marker panel and fixed marker frame.
- `world/assets/create3_dock/`: dock meshes, source details and retained licence.
- `robot_model/turtlebot3_waffle/turtlebot3_waffle_pi.xml`: robot geometry,
  joints, actuators and sensor definitions; `assets/` holds its meshes.
- `track_a_mobile_robot/main.py`: configures the model and runs shared sensing,
  perception and EKF updates in viewer/headless modes, with terminal reporting.
- `track_a_mobile_robot/display.py`: live camera inset with marker annotations
  and correction status, plus floor path/reference/LiDAR overlays.
- `track_a_mobile_robot/sensing.py`: encoder and LiDAR readings, RGB capture
  and camera calibration.
- `track_a_mobile_robot/perception.py`: ArUco detection, configured landmark map,
  camera-derived robot world poses/covariance and LiDAR frame transforms.
- `scripts/perception_diagnostics.py`: offline perception evaluation and camera calibration.
- `scripts/generate_marker.py`: regenerates both dock marker textures.
- `track_a_mobile_robot/state_model.py`: wheel geometry, midpoint pose prediction
  and its Jacobians for covariance propagation.
- `track_a_mobile_robot/estimation.py`: `PoseEstimator.predict()` / `correct()` and
  `StateEstimate` snapshots containing timestamp, world pose and covariance.
- `track_a_mobile_robot/evaluation.py`: simulator truth and pose-error calculations,
  used only for reporting results.
- `track_a_mobile_robot/navigation.py`: LiDAR waypoint planning, trajectory references
  and approach/alignment/docking stages.
- `track_a_mobile_robot/control.py`: LQI tracking, heading feedback for alignment
  and conversion to wheel speeds.
- `tests/test_navigation.py`: planning, observation-loss, docking and tracking checks.

The estimator accepts `correct(camera_pose_world, camera_covariance, time)`
with a positive-definite 3×3 covariance in metres/metres/radians order. It rejects
mismatched timestamps and repeated frames. Range and bearing remain available
for diagnostics, but the chosen EKF interface uses the full world-pose observation.

## Model source

The robot is adapted from
[ROBOTIS's MuJoCo collection](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/d8344c0dbe7a00208d0301111523dde65efc174a/robotis_tb3),
commit `d8344c0dbe7a00208d0301111523dde65efc174a`. Sensor definitions have been
added, and the support contacts have been corrected as described below. The
upstream licence is retained in `robot_model/turtlebot3_waffle/LICENSE`.

The chassis collision mesh is raised 2 mm relative to its original pose so it
does not drag on the floor. The visual mesh keeps its original position, and
the chassis still collides with walls and furniture. The two rear support
spheres approximate freely rolling supports using frictionless normal contacts
(`condim="1"`). Their `priority="1"` makes this setting override the floor's
frictional contact settings; without higher priority, MuJoCo uses the larger
contact dimension and friction coefficients. See the
[MuJoCo contact-parameter rules](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).
Drive-wheel geometry, wheel traction settings and joint friction are retained.
This is a simplified support model, not a detailed caster mechanism.

`tests/test_robot_contacts.py` checks that the rear supports carry load while
the chassis clears the floor. It also compares clean encoder odometry with
simulator motion during three-second straight, arc, clockwise and anticlockwise
trials, requiring less than 1 cm position error and 1° heading error. Truth is
used only to initialise and score these offline checks; wheel commands are
fixed in advance. These checks do not establish full-room docking reliability.

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
run the headless sensing/estimation check above and any other checks relevant to your task.

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
