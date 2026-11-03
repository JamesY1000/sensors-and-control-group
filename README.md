# Sensors and Control — Track A

A TurtleBot3 Waffle Pi simulation in an 8 × 6 m room with two tables, four
chairs, and a charging dock carrying two ArUco markers. The autonomy pipeline
has no known robot starting pose, dock world pose, room coordinates or room dimensions.
The room dimensions above describe the simulation, not an algorithm input.

## Run the simulation

Use Python 3.12 or newer from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m track_a_mobile_robot.main
```

The exploration and docking demo starts automatically on launch. Optional
`--config config/simulation_config.yml` selects the configuration. Use
`--headless --seconds 5` for a bounded run without a GUI. Headless defaults to
two simulated seconds; GUI runs continue until closed unless `--seconds` is set.
On Windows use `python` and activate `.venv\Scripts\Activate.ps1`.

On Linux, desktop runs default to GLFW for both the viewer and camera renderer;
headless runs default to EGL. An explicit `MUJOCO_GL` setting overrides this
selection. If a desktop run reports `Failed to make the EGL context current`,
try `MUJOCO_GL=glfw python3 -m track_a_mobile_robot.main`.

## Sensor-only acquisition and docking

1. **SCAN / EXPLORE:** send a timed turning command (`search_spin_s`, 20 s).
   When the forward LiDAR corridor is clear, drive briefly (`search_drive_s`, 3 s)
   and scan again. These commands do not use odometry, LQI or a visited-position
   map. LiDAR checks every command; blocked drives stop and return to scanning.
   Timings specify commands, not measured rotation angles or travel distances.
2. **Acquire:** the first usable marker interrupts exploration immediately. Define
   the robot pose as `(0, 0, 0)`, initialise the dock from the relative camera pose
   and save the current encoder readings as the baseline. Earlier wheel travel is
   excluded. Subsequent camera observations correct this estimate without resetting it.
3. **APPROACH:** LQI turns towards and tracks one full straight reference to a coarse target
   1.5 m in front of the estimated dock. Arrival heading is not constrained.
   Handover requires being within 0.25 m of that specific offset point, plus a
   clear corridor to the PBVS approach offset. Being close to the dock from the
   side is insufficient. When blocked, **AVOID** follows an obstacle boundary with
   LQI until the direct route clears.
4. **REACQUIRE:** hold position and use an LQI turn to face the estimated dock.
   Before inward travel, require three distinct EKF-accepted camera observations
   from the last 0.5 s and a dock bearing within 10 degrees of straight ahead.
   Recent observations from coarse travel can satisfy the gate immediately.
   Otherwise, wait 1 s when facing the dock, then sweep up to 30 degrees to either
   side, pausing at each end. Stop after 30 s without confirmation; fresh
   confirmations can release that stop when the heading and geometry also qualify.
   Recheck the coarse region and approach corridor after camera corrections or new
   obstacles; return to APPROACH/AVOID if either is invalidated.
5. **ALIGN:** PBVS directly commands forward/angular velocity to reach the configured
   approach offset (currently 0.70 m) and face into the dock. There is no timed trajectory in this phase.
   Reach both the 3 cm position and 3 degree heading tolerances before proceeding.
6. **DOCK:** PBVS moves to the 0.30 m offset at the lower docking speed.
7. **DONE:** stop when both estimated position and heading meet their tolerances.

Odometry starts at first acquisition. A detected ID alone is insufficient:
the marker pose must pass the size, reprojection, ambiguity and physical-consistency
checks.

After first acquisition, coarse travel continues using encoder prediction during
marker loss. REACQUIRE adds a camera confirmation gate at the handover to PBVS;
it does not reset odometry or the dock estimate. New accepted camera poses correct
the same EKF throughout. After that gate, ALIGN and DOCK can still use the predicted
relative dock pose during occlusion; completion is based on estimated pose tolerances.
There is no additional camera-loss timeout within ALIGN/DOCK. Longer gaps can
accumulate drift. LiDAR freshness/collision stops, obstacle-recovery limits and DONE
remain active, including during reacquisition turns and sweeps.

## Frames and estimation

The moving robot frame is used for sensor measurements. A fixed **local odometry
frame** starts at the first accepted marker observation and holds the joint robot/dock estimate.
Neither frame is aligned to the simulator world. Navigation derives the target
from the estimated dock, and control compares estimated and reference robot poses
in the same local frame. Translating/rotating that frame does not change commands.

The EKF's internal state is
`[robot_x, robot_y, robot_heading, dock_x, dock_y, dock_heading]` in local odometry.
The initial robot identity defines the frame, with zero robot covariance by
coordinate convention. The dock starts with camera measurement uncertainty.
Robot/dock cross-covariance develops during later corrections and is retained;
the first noisy dock observation is not treated as a perfectly known landmark.

Encoder increments predict the robot while the dock is stationary. A camera
observation measures `R(robot_heading).T @ (dock_xy - robot_xy)` and the wrapped
heading difference. Its Jacobian is with respect to both robot and dock states.
Correction uses `S = H @ P @ H.T + R`, normalized innovation gating at 11.345
(99% for three components), and Joseph-form covariance propagation. Rejected or
reused observations do not produce EKF corrections. Absolute world pose
remains unobservable; the robot does not need it for relative docking.

`StateEstimate` returns `pose_local`, robot covariance, `dock_pose_local` and dock
covariance. Before acquisition the dock fields are `None`. The differential-drive
motion model remains unchanged. Camera and encoder updates share acquisition time.

## Coarse LQI approach and PBVS docking

Navigation contains `MotionTrajectory`, used after acquisition during coarse
approach and detours to produce reference pose and forward/angular velocity.
The target comes from the camera/encoder EKF throughout approach and docking.
The dock's measured orientation defines its outward normal; the selected offset
along that normal gives the target position, with the robot facing into the dock.

Direct approach turns towards the coarse target if needed, then generates one
straight trajectory all the way to it. There are no one-metre intermediate stops.
The entire direct corridor is checked against observed LiDAR obstacles before
travel, and the remaining reference is rechecked while tracking. If blocked,
the robot first approaches along a clear prefix with a stand-off, then boundary
following supplies short detour targets; once the corridor clears,
navigation generates a new full direct reference from the current estimate.

Detours use the same spot-turn-then-straight-drive generator as direct approach,
with short targets selected by boundary following. Coarse approach hands over
within the tolerance around its stand-off point, without demanding dock-facing
arrival heading. PBVS then handles the finer position and heading alignment.
Between these stages, REACQUIRE uses an explicit turn-only reference to point the
camera towards the dock. Position corrections cannot turn that reference into a
drive. Its confirmation gate counts distinct accepted camera timestamps, not control
ticks or raw marker IDs. Configure its heading tolerance, observation age, frame
count, dwell, sweep and timeout with the `reacquire_*` navigation settings.
Each segment uses cubic progress `s=3u²-2u³`, where `u` is elapsed time
divided by duration. Reference velocity starts and ends at zero. Duration limits
peak speed and acceleration, reserving 20% of the speed limit for feedback.
Short detours can still slow or stop between segments. There is no A*, occupancy
grid or global route search; unseen space is not certified clear by a LiDAR check.

`MotionController` applies `u = u_ref - K z`. During straight-reference driving,
`z = [e_along, e_across, e_heading, integral(e_along)]`, with errors defined as
estimated minus reference pose in reference-heading axes. The linearised model is:

```text
e_along_dot   = delta_v
e_across_dot  = v_ref * e_heading
e_heading_dot = delta_omega
integral_dot  = e_along
```

The gain is computed from the continuous algebraic Riccati equation using SciPy,
with `Q=diag(4,16,2,4)` and `R=diag(4,1)`. Gains are cached for signed reference
speeds rounded to 0.01 m/s. At ramp endpoints a minimum design speed of 0.02 m/s
avoids the uncontrollable zero-speed Cartesian pose linearisation. This is a local
moving-reference design, not a claim of full-pose stability at rest. Navigation
starts a new segment if a finished straight drive leaves significant lateral error. Segments
start and end at zero reference speed; this is not a continuous-speed spline route.

In-place turns use a separate heading/integral model with `Q=diag(4,4)` and `R=1`;
forward velocity stays zero. Integral action is therefore part of an augmented
LQR design (LQI), rather than hand-added PI terms. Bounded integrals and conditional
integration limit windup; stops, phase changes and new segments clear the integral.
Search commands remain primitive timed commands until a usable dock estimate exists.

### PID comparison baseline

LQI is the default. `--controller pid` swaps in `PIDMotionController`, which
replaces only the coarse tracking laws. It keeps the same errors, feedforward,
saturation, anti-windup and PBVS, so differences come from the feedback alone:

```text
v     = v_ref     - (kp_x*e_along + ki_x*integral(e_along) + kd_x*d(e_along)/dt)
omega = omega_ref - (kp_h*e_heading + kd_h*d(e_heading)/dt + k_y*direction*e_across)
turn: omega = omega_ref - (kp*e_heading + ki*integral(e_heading) + kd*d(e_heading)/dt)
```

Cross-track error cannot be actuated directly, so it feeds the heading loop and
changes sign when reversing. Derivatives use a 0.1 s low-pass filter and reset
with each segment. References are straight lines and spot turns, where the LQI
gain has no cross-coupling, so the default P/I gains copy `driving_gain(0.22)`
and the turn gain. The PID run therefore measures the value of LQI's speed
scheduling of the heading gain, plus the effect of the PID derivative terms.

```bash
python3 -m track_a_mobile_robot.main --headless --seconds 160 --controller lqi --control-log lqi.csv
python3 -m track_a_mobile_robot.main --headless --seconds 160 --controller pid --control-log pid.csv
python3 -m scripts.compare_controllers lqi.csv pid.csv --plot comparison.png
```

The CSV has one row per control update: tracking errors, commands, limits,
LiDAR stops and simulator-truth pose. The truth pose is for evaluation only and
is never fed back into the pipeline. The script prints a metric table and, with
`--plot`, writes a PNG: driven paths, headline metrics, cross-track and heading
errors, and v/omega commands against time since each run's first reference.
Figures are redrawn from the CSVs without re-running the simulations.

Significant changes in the estimated target or excessive tracking error restart
the reference from the current estimate. A stop also resets the reference so it
cannot advance while the robot is held still. Small target corrections are included
when the next segment is generated. Large estimated final misalignment returns
to the approach offset; marker loss does not block these transitions.

For ALIGN and DOCK, `MotionController` switches to planar position-based visual
servoing (PBVS), using `StateEstimate.dock_pose_base`. This is the camera/encoder
EKF's relative pose, not raw camera data or simulator truth. Navigation supplies
an offset, with no trajectory reference. PBVS calculates target distance `rho`,
bearing `alpha`, and remaining orientation `beta`, then uses proportional distance
feedback and `omega = 1.5*alpha - 0.5*beta`. Large bearing errors cause a turn first;
small overshoots allow reversing while facing the dock. Once position is within
tolerance, only heading is corrected. Velocity and wheel limits still apply.
LQI integrals are cleared at handover and are not part of PBVS.
Small minimum driving/heading-correction commands counter the simulated
low-speed friction outside tolerance; they are still capped by the configured
speed limits. Inside both tolerances the command is zero.
The heading-only minimum is 0.30 rad/s: the wheel motor gain and joint friction
produce an approximate 0.23 rad/s in-place turning deadband. The former
0.15 rad/s minimum could leave the robot almost stationary in ALIGN despite
fresh camera measurements. ALIGN/DOCK logs show position and heading errors
relative to the estimated target, rather than requiring interpretation of odometry.

Before tracking, navigation checks the swept circular footprint continuously
between the endpoints of each straight segment against grouped LiDAR returns,
with 1 cm extra padding. It also checks the connection from the actual estimated
position to the remaining reference. If blocked, navigation stops and selects a
detour. This check repeats during tracking using the latest scan.

Before applying each command, LiDAR separately checks its swept circular footprint over the
next second, including curved, reverse and in-place motion. A blocked command
stops immediately; approach navigation can choose a detour on the next update. Sensor freshness, minimum-range hits,
robot clearance and wheel-speed limits remain active. Sensor blind spots and a
single scan height limit what this check can detect.

The 0.30 m target is geometric docking; electrical contact is not simulated.

## LiDAR boundary following

An observed obstacle anywhere along the direct goal corridor interrupts the full
reference. If it is distant, LQI first follows the checked clear prefix, leaving
a one-metre stand-off (increased for larger clearance/detour settings). Boundary
following starts only on reaching that vicinity; a far obstacle does not choose
a passing side immediately. Navigation connects neighbouring
LiDAR returns separated by at most `boundary_join_gap_m` (0.8 m), then follows
one passing side using short targets and LQI. Joining is separate from the robot's
0.27 m clearance: it closes small gaps between legs without inflating every wall
by half the joining distance. Hidden or widely separated furniture parts remain
separate; this is not object recognition.

The follower starts with the obstacle blocking the goal corridor, rather than a
nearer side wall. It retains its target until reached/obstructed and advances the
contact within a local section of the previously followed boundary. It returns towards the goal
only after making progress past the encounter distance and seeing a clear corridor
for 0.4 s. An entirely cleared scan also releases the detour. Checks cover the sensed
corridor only; unseen space is not proven free.

A return to the encounter position after travelling around a loop, 30 s without
position progress, or a 180 s overall detour budget stops motion with an explicit
status. Marker loss does not interrupt an active detour merely because the dock
is occluded. After leaving the boundary, travel resumes from the stored dock estimate.
Actual commands are separately checked against LiDAR, including reverse/curved
motion. This local behaviour does not guarantee finding a route through every
concave obstacle or furniture arrangement.

## Simulation versus robot inputs

`main.load_spec()` applies YAML robot/dock world poses only to place physical objects.
`create_demo()` receives estimation settings, not the spawn-pose configuration;
`Navigator` receives navigation settings and estimates, not a dock map.
`Perception` reads sensors and fixed sensor mounting calibration. It never reads
marker sites, installed marker world transforms, or the live robot pose.
`SensorDisplay` alone reads the live robot transform to place decorative 3D
graphics in the viewer. It sends no poses or other values back to robot algorithms.

Known physical calibration is still necessary: camera intrinsics/extrinsics,
wheel dimensions, marker sizes, and the rigid marker-to-dock mounting offsets.
`perception.marker_mounts()` specifies those offsets in the dock's own frame.
They let both markers describe the same unknown dock origin. Moving the dock in
the room requires no changes to these values. Altering the physical marker mount
requires updating this calibration, just as on hardware.

Offline `scripts/perception_diagnostics.py` and `evaluation.py` contain explicitly
separate truth-based scoring/calibration helpers. They are not imported by the
runtime pipeline. The current sensor-only rendered trials do not use them or
read true poses, including for scoring. MuJoCo itself necessarily knows object
poses to simulate sensors and collisions; these are not supplied to the robot.

## Configuration and sensors

`config/simulation_config.yml` contains simulation placement, sensor calibration,
estimation noise and navigation settings. Preserve safe physical spawn clearance.
Robot/dock `x`, `y`, `yaw_deg` are simulation placement only. Sensor mounting
positions are in the base frame (+X forward, +Y left, +Z up). Camera optical
axes are +X right, +Y down, +Z forward, with RPY applied as `Rz @ Ry @ Rx`.
Intrinsics derive from the configured vertical FOV and image size; distortion is
zero in this renderer. A checkerboard is not needed for the running simulation.

| Component | Setting |
| --- | --- |
| Physics | 500 Hz |
| Estimation/control | 50 Hz by default |
| Camera | 640 × 480, 30 Hz |
| LiDAR | 360 beams, 0.16–8 m, 5 Hz |
| GUI sync | Up to 30 Hz of wall time |
| Local map | 10 Hz |

`estimation` contains `update_hz`, `noise_seed`,
`process_position_std_m_per_sqrt_s` and `process_heading_std_deg_per_sqrt_s`.
There is no world-start covariance or known starting pose. Process standard
deviations are squared and multiplied by elapsed time. Encoder increment
variance uses `2*sigma**2`, an approximation ignoring inter-increment correlation.
The initial dock uncertainty comes from the camera. All noise values are tuning
assumptions, not calibrated accuracy claims.

## Markers and perception

| Marker | Black square | White panel | Margin per side | Centre above dock origin |
| --- | --- | --- | --- | --- |
| ID 0: approach | 40 × 40 cm | 56 × 56 cm | 8 cm | 40 cm |
| ID 1: final approach | 4 × 4 cm | 6 × 6 cm | 1 cm | 12 cm |

Both use `DICT_4X4_50`. Their black-square dimensions are in `MARKER_SIZES_M`;
white borders are excluded. Camera-to-marker pose is transformed through the
camera and marker mount calibrations to produce `DockObservation.pose_base`.
The largest reliable marker is selected once per image, avoiding double-counting
correlated measurements from two markers in the same frame.

The flat-floor check rejects relative dock heights beyond 5 cm and upright-axis
tilts beyond 10 degrees. The dock's world location is not needed for this check.
Small footprints, poor reprojection fits and ambiguous planar orientations are
also rejected. Near-frontal planar ambiguity can still delay usable measurements.
Camera covariance models relative translation and heading, with a small
correlated contribution from the marker-to-dock mounting offset.

## Display

The GUI contains the robot camera with detection annotations and a separate
local 2D sensor map. Orange points show LiDAR, blue shows the robot, and
green shows the coarse target, teal the PBVS approach, purple the final docking
target, and yellow a short boundary-following target.
Cyan shows the active trajectory segment and its final heading; a white ring marks
the current reference tracked by LQI. The grey-blue trail retains up to 600 recent
estimated robot positions. Turns show a heading arrow at the stationary position.
This displays the current LQI segment, not a complete planned route. PBVS has
no timed segment, so its trajectory/reference overlay clears at handover. The map
uses the current robot frame during exploration and fixed local odometry after
marker acquisition. Exploration has no pose trail or reference trajectory.

The 3D viewer adds a blue clearance circle and heading arrow around the robot,
a cyan active LQI segment (or heading arc for a turn), and a white reference ring.
During PBVS it shows a purple target and heading instead of a generated trajectory.
These decorative geoms are drawn at 10 Hz and cannot affect collisions or the
camera/LiDAR sensors. Local estimated coordinates are first converted into the
robot frame, then placed using a display-only simulator robot transform. The
preview follows the visible robot; it is not a world-frame localisation-error plot.
The circle shows `clearance_m`, which may differ from the physical chassis outline.

The top-left GUI overlay has two lines: phase/time and status. The terminal prints
a compact status every two simulated seconds, plus phase/stop changes, with the
age of the last accepted camera pose and simulation speed (`1.00x` is real time).
Use `python3 -m track_a_mobile_robot.main --verbose` to include the
estimated pose, one-standard-deviation uncertainty and accepted/total camera counts.
Initial zero robot covariance defines the local frame; uncertainty then grows with
encoder motion after marker acquisition. Before then, the displayed identity is
not an accumulated pose estimate.

## Simulation performance

Camera detection remains at 30 Hz and estimation/control at 50 Hz. LiDAR is
sampled at 5 Hz; perception now copies and transforms its points only when a new
scan arrives. Image-noise generation uses one single-precision working buffer
and performs its arithmetic in place; the Gaussian noise level is unchanged.
OpenCV uses at most four worker threads to leave CPU time for physics and the
viewer. The random sequence differs from the previous double-precision sampler,
but remains reproducible for a fixed seed with this implementation.

The camera/map overlay is prepared at 10 Hz and sent through a single background
publisher with a one-item queue. Slow/minimised viewers discard old overlays
instead of blocking physics and control. The worker only receives prepared images
and text; it does not read robot state or access the renderer used for perception.

On this laptop's NVIDIA RTX 3050, profiling five simulated seconds with the GUI
previously took 65.48 s including startup: 54.06 s was spent waiting in synchronous
`set_images`. With latest-only asynchronous uploads the same run took 8.52 s
including startup, with the active loop approximately real time. In the headless
baseline, image-noise generation took 1.58 s while camera rendering took 0.23 s.
These timings are machine/window-state dependent, not guaranteed frame rates.
The sandbox used Mesa llvmpipe software rendering; desktop GPU measurements were
used for the comparison above.

A later GUI profile reproduced a drop to `0.84x` at 44.4 simulated seconds.
With single-precision camera noise and the OpenCV thread limit, a 45-second run
stayed near `1.00x`; image-noise time fell from 18.46 s for 1,339 frames to
12.29 s for 1,350 frames. Sensor rates, noise standard deviation and physics
timestep were unchanged. These runs use the same scene settings, but the changed
noise sequence means their estimated trajectories are not identical.

## Validation and limitations

Run `python3 -m unittest discover -s tests -v`. All 78 automated checks pass.
Checks cover ignored pre-acquisition encoder travel, zero-origin initialisation,
post-acquisition correction without resetting, timed exploration, LiDAR stops,
LQI/PBVS tracking, turn/straight-drive kinematics, LQI stability, full-segment
collision checks, detour horizon limits and nonblocking display uploads. Additional checks
cover full direct references without intermediate stops, distant new obstacles,
resuming direct travel after clearance, ignoring obstacles beyond the goal, and
approaching distant blockers before starting boundary following. Handover checks
cover side arrivals, starts too close to the dock, rotated dock normals, free
arrival heading, and a blocked PBVS approach corridor. Reacquisition checks cover
distinct/recent accepted frames, immediate handover with existing visual evidence,
turning without translation despite wheel friction, bounded sweeps and timeout,
fresh-camera recovery, corrected target geometry, new obstacles, and LiDAR stops.
With the reacquisition gate enabled, a camera-rendered close angled start at robot
`(0.8, 0.15, 150 degrees)` and dock `(-0.8, 0, 0 degrees)` entered REACQUIRE at
0.74 s, ALIGN at 1.92 s, DOCK at 16.38 s and DONE at 35.88 s. The check asserted
zero commanded forward speed throughout REACQUIRE and finished within the
estimated 3 cm / 3 degree docking tolerances. World poses only placed the scene;
navigation and control used the encoder/camera EKF. This checks one placement,
not reliability across the whole room.
The rendered trials below predate the reacquisition gate unless stated otherwise.
With the stricter stand-off gate, a rendered side-start trial (robot 3, 1.2,
-90 degrees; dock 2.5, 0, 0 degrees) switched to PBVS at 0.247 m from the
coarse target and completed estimated docking at 101.64 simulated seconds.
The controller used only EKF poses; these coordinates placed the test scene.
A rendered replay of the configured corner start acquired its first marker at
21.70 s. It now follows the clear direct prefix first and starts boundary following
at 51.00 s, roughly one metre from the blocking return, rather than choosing a side
at acquisition while the obstruction is about 3.8 m away. This replay checks the
avoidance entry only; it does not establish docking success from that start.
A camera-rendered full-direct-approach trial from robot (6.5, 0, 180 degrees)
to dock (-0.8, 0, 0 degrees) reached estimated DONE at 99.98 simulated seconds,
including PBVS alignment and final docking. Simulation placements were supplied
only to the scene; navigation and control used the joint EKF estimate.
Before the full-direct-reference change, a rendered trial from robot (5, 0.6, 180 degrees) and dock (2.5, 0, 0 degrees)
tracked a curved LQI approach with simultaneous forward/angular commands, then
completed PBVS docking at 41.50 simulated seconds within the estimated tolerances.
These world coordinates only set the physical test scene; the pipeline starts
its local frame at first camera acquisition.
Marker-loss checks cover APPROACH, ALIGN and DOCK through 10,000 s without a
camera update, including phase transitions after successful reacquisition and
resumed EKF-derived target updates. Entering PBVS now requires confirmation first.
A camera-rendered close-start run withheld all dock observations after 5 s and
reached estimated DONE at 23.64 s using encoder prediction for the remaining travel.

Camera-rendered headless checks after the PBVS turn correction completed docking
in 26.42 s from a close start and 118.66 s from the configured centred start
(robot 6.5, 0, 180 degrees; dock -0.8, 0, 0 degrees). Both finished within the
estimated 3 cm / 3 degree tolerances. The close-start baseline remained in ALIGN
after 60 s with the previous minimum turn command.
Increasing the configured travel/PBVS approach/detour limits to 0.22/0.15/0.14 m/s
and the turn limit to 0.6 rad/s reduced the centred-start run to 87.40 s, within
the same estimated docking tolerances. Final docking remains limited to 0.06 m/s.
Earlier corner and delayed-marker cases encountered a boundary loop and an
alignment stall respectively; those placements have not been rerun with this fix.
These checks do not establish reliable completion from every room placement.

## Key files

- `main.py`: simulation construction, sensor loop and pipeline connections.
- `perception.py`: marker detection and relative dock measurements.
- `estimation.py`: encoder odometry and joint local robot/dock EKF.
- `navigation.py`: exploration, boundary following, trajectory generation and safety checks.
- `control.py`: LQI tracking, direct PBVS alignment/docking and wheel conversion.
- `display.py`: camera inset and local sensor map.
- `state_model.py`: differential-drive motion and Jacobians.

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
