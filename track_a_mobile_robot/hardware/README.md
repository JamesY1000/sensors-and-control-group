# Hardware setup and utilities

Run all commands from the repository root after the
[hardware dependency setup](../../README.md#run-the-hardware-demo).
The existing utility commands are unchanged.

## Configuration

`config/hardware_config.yml` is separate from simulation tuning. Supply the host
with `--host` for the demo or as the positional address for utilities. Use
`--config path/to/hardware.yml` to select another calibration/configuration.

The supplied configuration describes an unmodified TurtleBot3 Waffle Pi:

- Wheel radius 0.033 m, separation 0.287 m and wheel limit 7.88 rad/s.
- Camera optical centre `[0.076, 0, 0.103]` m and LiDAR origin
  `[-0.064, 0, 0.132]` m relative to floor-level `base_footprint`.
- Saved 640 × 480 camera calibration from the 51-view capture
  `outputs/camera_calibration/20261103-134652-132350/`.
- `DICT_4X4_50`, ID 0, **80 mm black-square width** excluding the white border.

Wheel geometry follows the
[ROBOTIS driver settings](https://github.com/ROBOTIS-GIT/turtlebot3/blob/main/turtlebot3_bringup/param/waffle_pi.yaml);
sensor mounting follows the
[Waffle Pi URDF](https://github.com/ROBOTIS-GIT/turtlebot3/blob/main/turtlebot3_description/urdf/turtlebot3_waffle_pi.urdf),
including the 10 mm `base_footprint` to `base_link` offset. Update transforms if
mounts have moved. Base axes are forward/left/up; camera optical axes are
right/down/forward. Transforms map source coordinates into the named destination.

Verify topic names, left/right joint names, signs and whether encoder positions
wrap. Encoders are measured wheel angles in radians, positive for forward motion.
Camera resolution, crop and focus must match calibration. The compressed topic
carries raw camera images and uses raw intrinsics K and distortion D; the runner
does not fetch `camera_info` automatically. Empty `frame_id` fields disable header
name checks; set them when the driver names are known.

With `markers.reference: marker`, the single upright marker's centre and face
normal define the planar docking target. Height is discarded; no measured mounting
translation is needed. Keep the marker stationary. For a separately mounted dock,
use `reference: dock` and measured 4 × 4 `to_dock` transforms. Dock +X points out of
the printed face and +Z points up. Multiple marker centres cannot define a single
target in direct-marker mode.

Current trial settings are retained:

| Setting | Value / effect |
| --- | --- |
| Control rate | 10 Hz |
| Navigation / detour speeds | 0.10 / 0.08 m/s |
| Docking / turning limits | 0.04 m/s / 0.3 rad/s |
| Navigation offset / final dock offset | 0.50 / 0.25 m from the marker plane |
| Position / heading tolerances | 0.10 m / 10 degrees |
| Sensor freshness limits | 1.2 s; encoder history 3 s |
| Command watchdog | 0.3 s |
| `lidar.obstacle_checks_enabled` | `false`: scans are logged, but obstacles do not stop or detour motion |
| `lidar.zero_is_no_return` | `true`: zero ranges are treated like positive infinity, a driver assumption |
| `markers.accept_any_detection` | `true`: use the best finite fitted pose for the configured ID; quality and innovation rejection gates are disabled |

Set `obstacle_checks_enabled: true` to enable footprint/corridor checks and boundary
following with the configured 0.20 m clearance. Set `accept_any_detection: false`
to restore pose quality and innovation gates. Freshness, marker-loss stops and
head-on alignment checks still apply in the trial mode. These settings and the
saved calibration require verification on the actual robot.

## Run and stop

Start with a stationary check:

```bash
python -m track_a_mobile_robot.main --backend rosbridge --host ROBOT_IP --dry-run --verbose --seconds 30
```

Then run a bounded trial:

```bash
python -m track_a_mobile_robot.main --backend rosbridge --host ROBOT_IP --show-camera --seconds 30
```

Omitting `--seconds` runs until DONE or interrupted. Ctrl+C, Q/Escape or closing
the camera preview ends the run and sends zero velocity. A preview failure also
ends the run. `--speed-scale 0.5` halves navigation/docking/turning limits; its
allowed range is greater than zero and at most two. The physical wheel-speed cap
is unchanged. `--verbose` adds marker visibility and sent velocity; full diagnostics
are always logged.

Verify the robot-side command timeout before allowing motion: a laptop watchdog
cannot deliver zero velocity over a lost connection or after its process dies.
The local watchdog, stale-stream stops and shutdown cleanup remain active.

## Navigation and timing

Before acquisition, SCAN alternates random spot turns with short EXPLORE drives.
The first accepted marker ends this search. NAVIGATE uses LQI to reach the offset
and face the dock. Off-axis starts join an entry point farther out on the marker
normal before approaching head-on. Handover requires the offset pose and heading
tolerances in both the estimate and a fresh accepted camera measurement.

DOCK uses PBVS directly to the final offset. A new frame without a usable marker
stops translation immediately; accepted visual evidence also expires after
`reacquire_observation_age_s` (0.5 s). After a one-second wait, recovery slowly
turns towards the estimated dock until a fresh pose is accepted. Recovery stays
inside NAVIGATE/DOCK. Final DONE requires estimated and visually confirmed pose
tolerances. Offsets refer to the drive-axle centre and do not model charging contact.

All sensor headers need advancing nonzero acquisition timestamps on the same robot
clock. The HAL anchors that clock to the first encoder sample, checks repeated or
backward timestamps, and stops for stale sensors. This does not measure initial
network latency. Small future offsets up to 0.1 s are tolerated. A stale stream
pauses the current phase; fresh data can resume it.

Camera observations are aligned with interpolated encoder positions in a bounded
history before predicting forward. Unbracketed/old frames are dropped. LiDAR uses
the latest scan without motion compensation. An absolute-encoder gap above 1.2 s
resets fusion and requires marker localisation again. After acquisition, recovery
uses slow turns rather than restarting random forward exploration. Wrapped-encoder
gaps, backward clocks and implausible wheel increments require correction/restart.

## Camera calibration

Print the [9 × 6 inner-corner checkerboard](assets/checkerboard_9x6_25mm.pdf)
in landscape at **100% / Actual size**, with Fit to page disabled. Mount it flat,
measure a printed square, and supply that side length:

```bash
python -m track_a_mobile_robot.hardware.calibrate_camera ROBOT_IP --square-size-mm 25
```

Only the camera stream is needed. Keep the robot still and move the board while
using the navigation resolution, crop and focus. The default board has 10 × 7
squares (9 × 6 internal corners). `--columns` and `--rows` specify internal corners.

- **Space:** capture a complete detected board; nearly identical views are skipped.
- **U:** remove the last capture.
- **C:** calibrate after at least 12 views; aim for 20–30 sharp views across the
  image, at different distances and tilts.
- **Q / Escape:** quit without calculating calibration.

A new folder under `outputs/camera_calibration/` contains captures, a
`calibration_report.yml` and a calibrated `hardware_config.yml`. Only camera
intrinsics/distortion/resolution and host are updated in that copy; the input
configuration and mounting/navigation settings are preserved. Use the generated
file with `--config`. `--output-dir` selects another destination.

The fit uses a pinhole model with five distortion coefficients. Reprojection error
above 1 px is flagged. A low error alone does not establish accuracy; compare
marker distances against physical measurements before moving the robot.

## Marker and timing checks

For camera-only detection and a measured-distance comparison:

```bash
python -m track_a_mobile_robot.hardware.check_aruco ROBOT_IP --distance 0.70
```

This subscribes only to images, capped at 5 Hz, and creates no command publisher.
The overlay shows IDs, axes, range, bearing and pose-fit quality using the active
configuration. Measure horizontal distance from the drive-axle midpoint to the
marker centre's projection on the floor. Keep the marker upright and visible.
**S** saves the raw image, overlay and measurements under `outputs/aruco_check/`;
**Q / Escape** quits. Stale images are flagged and cannot be saved as fresh samples.

Printable markers are kept in `assets/`: the
[80 mm ID 0 sheet](assets/aruco_4x4_50_id0_80mm_four.pdf) matches the current
configuration; [60 mm](assets/aruco_4x4_50_id0_60mm_four.pdf) and
[180 mm](assets/aruco_4x4_50_id0_180mm.pdf) alternatives require changing `size_m`.
Print at actual size and measure the black square.

To compare sensor delivery with and without camera traffic, close other clients
and run:

```bash
python -m track_a_mobile_robot.hardware.check_timing ROBOT_IP
```

This measures joint states alone, joint states plus camera/LiDAR, then joint states
alone again, for ten seconds each. It creates no command publisher, saves
`outputs/hardware_timing.json`, and does not decode/save images. `--seconds` changes
each phase duration; `--output` changes the report path.

## Run logs

Every hardware run writes
`outputs/hardware-runs/<UTC timestamp>-<ID>/telemetry.jsonl` and prints its path.
`--log-dir` changes the parent directory. The log contains configuration metadata,
one record per control tick and an end record. Tick data includes raw/processed
LiDAR, encoder angles, wheel-only odometry, fused poses/covariance, marker geometry,
sensor ages, navigation phase, requested/sent commands and stop reasons. RGB images
and video are not saved.

Use sensor timestamps to deduplicate held readings. `camera.new_frame: false`
is not a fresh negative detection. Wheel-only odometry starts at the first encoder
sample; fused odometry starts at the first accepted marker. Split trajectories at
`odometry.epoch` and `estimator_epoch` changes. LiDAR points use the base frame at
scan time, so interpolate the pose before placing them on a map.

JSON encodes special range values as `"Infinity"`, `"-Infinity"` and `"NaN"`;
`null` means absent data. Commands are velocity setpoints, not measured speeds.
Logging runs on a background thread; a full queue or write error stops the demo.
Ctrl+C/errors flush the partial log. After a hard process/power failure, only
complete JSON lines are usable. Generated data is git-ignored and retained locally.
