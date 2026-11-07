"""Preview ArUco detection and pose quality using ONLY the rosbridge camera.

    python -m track_a_mobile_robot.hardware.check_aruco ROBOT_IP --distance 0.70

Distance is optional: measure horizontally from the centre of the wheel axle to
the marker centre, not from the camera or bumper. No commands are published.
"""

import argparse
import base64
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import roslibpy

from ..perception import (
    MAX_REPROJECTION_ERROR_PX,
    MIN_MARKER_SIDE_PX,
    ArucoLandmarkDetector,
    dock_transform_from_marker,
    select_dock_observation,
)
from ..sensing import CameraReading
from .config import DEFAULT_HARDWARE_CONFIG, load_hardware_config
from ..hal.rosbridge import source_timestamp_ns

WINDOW = 'Camera-only ArUco check - S save | Q quit'


def decode_image(message, config):
    """Decode the raw compressed image without changing its calibrated geometry."""
    if config.camera_frame_id and message['header'].get('frame_id') != config.camera_frame_id:
        raise ValueError(f'Expected camera frame {config.camera_frame_id!r}')
    encoded = np.frombuffer(base64.b64decode(message['data'], validate=True), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not decode the compressed camera image')
    if (image.shape[1], image.shape[0]) != config.image_size:
        raise ValueError(f'Image is {image.shape[1]}x{image.shape[0]}, but calibration is '
                         f'{config.image_size[0]}x{config.image_size[1]}; use matching calibration')
    return image


def inspect_image(image, config, detector, raw_detector, distance_m=None):
    """Show raw IDs as well as poses passing the navigation perception checks."""
    frame = CameraReading(time=0., rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    observations = detector.detect(frame)
    marker_as_dock = getattr(config, 'marker_as_dock', False)
    accept_any = getattr(config, 'accept_any_detection', False)
    # This extra detection pass is diagnostic only: the shared detector returns
    # accepted poses, hiding IDs with unknown sizes or rejected pose fits.
    corners, ids, _ = raw_detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    preview = image.copy()
    rows = []
    lines = ['Range: axle-centre to marker, horizontal | S save | Q/Esc quit']
    if ids is None:
        lines.append('No marker ID decoded. Show the full black square and white margin.')
    else:
        for marker_corners, marker_id in zip(corners, ids.ravel(), strict=True):
            marker_id = int(marker_id)
            points = marker_corners.reshape(4, 2)
            side_px = float(np.mean(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)))
            row = {'id': marker_id, 'side_px': side_px, 'corners_px': points.tolist(),
                   'size_m': config.marker_sizes_m.get(marker_id), 'usable_for_docking': False}
            # Match corners too, so two printed copies of one ID stay distinct.
            observation = next((obs for obs in observations
                                if obs.marker_id == marker_id and np.allclose(obs.corners_px, points)), None)
            if marker_id not in config.marker_sizes_m:
                status = 'ID has no configured size/mount'
            elif not accept_any and side_px < MIN_MARKER_SIDE_PX:
                status = f'Too small (need {MIN_MARKER_SIDE_PX:g} px)'
            elif observation is None:
                status = f'Pose fit rejected (invalid or > {MAX_REPROJECTION_ERROR_PX:g} px error)'
            else:
                dock = select_dock_observation([observation], config.marker_to_dock,
                                               marker_as_dock=marker_as_dock, accept_any_detection=accept_any)
                transform = dock_transform_from_marker(observation, config.marker_to_dock,
                                                       marker_as_dock=marker_as_dock)
                height = float(transform[2, 3])
                tilt = float(np.rad2deg(np.arccos(np.clip(transform[2, 2], -1, 1))))
                row.update(
                    range_m=observation.range_m,
                    bearing_deg=float(np.rad2deg(observation.bearing_rad)),
                    camera_xyz_m=observation.T_camera_marker[:3, 3].tolist(),
                    reprojection_error_px=observation.reprojection_error_px,
                    ambiguity_ratio=observation.ambiguity_ratio,
                    dock_tilt_deg=tilt,
                    usable_for_docking=dock is not None,
                )
                row['marker_height_m' if marker_as_dock else 'dock_height_error_m'] = height
                if dock is not None:
                    status = 'Usable dock pose'
                    row['dock_pose_base'] = list(dock.pose_base)
                elif not observation.orientation_reliable:
                    status = 'Ambiguous orientation'
                else:
                    status = 'Marker tilt rejected' if marker_as_dock else 'Dock height/tilt rejected'
                rvec, _ = cv2.Rodrigues(observation.T_camera_marker[:3, :3])
                cv2.drawFrameAxes(preview, config.camera_matrix, config.dist_coeffs,
                                  rvec, observation.T_camera_marker[:3, 3], row['size_m'] / 2)
            row['status'] = status
            rows.append(row)
            colour = (80, 220, 80) if row['usable_for_docking'] else (0, 180, 255)
            cv2.polylines(preview, [points.astype(np.int32)], True, colour, 2)
            origin = (max(5, int(points[:, 0].min())), max(20, int(points[:, 1].min()) - 8))
            cv2.putText(preview, f'ID {marker_id}', origin,
                        cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2, cv2.LINE_AA)
            lines.append(f'ID {marker_id} | {side_px:.0f} px | {status}')
            if observation is not None:
                comparison = ''
                if distance_m is not None:
                    row['range_error_m'] = observation.range_m - distance_m
                    comparison = f" | error {row['range_error_m'] * 100:+.1f} cm"
                lines.extend([
                    f"  Range {row['range_m']:.3f} m | bearing {row['bearing_deg']:+.1f} deg{comparison}",
                    f"  Fit {row['reprojection_error_px']:.2f} px | ambiguity {row['ambiguity_ratio']:.2f}"
                    + (' (quality checks disabled)' if accept_any else ' (<0.50)'),
                    (f"  Marker height {height * 100:.1f} cm (ignored) | tilt {tilt:.1f} deg ({'not gated' if accept_any else 'limit 10 deg'})"
                     if marker_as_dock else
                     f"  Dock height error {height * 100:+.1f} cm | tilt {tilt:.1f} deg ({'not gated' if accept_any else 'limits 5 cm / 10 deg'})"),
                ])
    panel = np.full((max(4, len(lines)) * 23 + 16, max(800, image.shape[1]), 3), 28, dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (10, 25 + index * 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .50, (230, 230, 230), 1, cv2.LINE_AA)
    preview = cv2.copyMakeBorder(preview, 0, 0, 0, panel.shape[1] - image.shape[1], cv2.BORDER_CONSTANT)
    return np.vstack((preview, panel)), rows


def save_capture(output_dir, image, preview, report):
    """Keep the original pixels alongside the overlay and numerical results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    for suffix, pixels in (('.png', image), ('_overlay.png', preview)):
        if not cv2.imwrite(f'{stem}{suffix}', pixels):
            raise OSError(f'Could not save {stem}{suffix}')
    path = Path(f'{stem}.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved image, overlay and measurements: {path.resolve()}')


def run_check(config, output_dir, distance_m=None):
    """Subscribe only to images; decode outside the callback and keep the latest."""
    detector = ArucoLandmarkDetector(config.camera_matrix, config.dist_coeffs,
                                    config.camera_to_base, config.marker_sizes_m, config.dictionary_id,
                                    accept_any_detection=getattr(config, 'accept_any_detection', False))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    raw_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(config.dictionary_id), parameters)
    client = roslibpy.Ros(host=config.host, port=config.port)
    # Five frames/second and a one-image server queue reduce rosbridge traffic.
    topic = roslibpy.Topic(client, config.topics['image'], 'sensor_msgs/msg/CompressedImage',
                          throttle_rate=200, queue_length=1)
    lock = threading.Lock()
    pending = None

    def receive(message):
        nonlocal pending
        with lock:
            pending = (message, time.monotonic())

    try:
        client.run(timeout=config.timing['startup_timeout_s'])
        topic.subscribe(receive)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        print(f'Camera only: {config.topics["image"]} (up to 5 Hz); no robot commands.')
        print(f'Marker sizes: {config.marker_sizes_m}; calibration: {config.image_size}.')
        print('Focus the preview. S: save image and measurements; Q/Escape: quit.')
        preview = np.zeros((480, 800, 3), dtype=np.uint8)
        cv2.putText(preview, 'Waiting for camera...', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
        last_stamp = None
        last_fresh_at = time.monotonic()
        last_print_at = float('-inf')
        report = None
        while client.is_connected:
            with lock:
                received, pending = pending, None
            if received is not None:
                message, received_at = received
                stamp = source_timestamp_ns(message)
                if last_stamp is not None and stamp < last_stamp:
                    raise ValueError('Camera timestamp went backwards; restart the check')
                if stamp != last_stamp:
                    image = decode_image(message, config)
                    preview, rows = inspect_image(image, config, detector, raw_detector, distance_m)
                    report = {'source_stamp_ns': stamp, 'frame_id': message['header'].get('frame_id'),
                              'image_size': list(config.image_size), 'measured_range_m': distance_m,
                              'camera_matrix': config.camera_matrix.tolist(),
                              'dist_coeffs': config.dist_coeffs.tolist(), 'markers': rows}
                    last_stamp, last_fresh_at = stamp, received_at
                    if time.monotonic() - last_print_at >= 1.:
                        print(' | '.join(f"ID {row['id']}: {row['status']}"
                                         + (f"; range {row['range_m']:.3f} m" if 'range_m' in row else '')
                                         for row in rows) or 'No marker ID decoded.', flush=True)
                        last_print_at = time.monotonic()
            age = time.monotonic() - last_fresh_at
            if age > config.timing['startup_timeout_s']:
                raise RuntimeError('No new camera timestamp; check the image topic and camera publisher')
            stale = age > config.timing['camera_timeout_s']
            displayed = preview.copy()
            if stale:
                cv2.rectangle(displayed, (0, 0), (displayed.shape[1], 40), (0, 0, 150), -1)
                cv2.putText(displayed, f'NO FRESH IMAGE for {age:.1f}s - measurements held', (10, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            cv2.imshow(WINDOW, displayed)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return
            if key in (ord('s'), ord('S')):
                if report is None or stale:
                    print('No fresh image to save.')
                else:
                    save_capture(output_dir, image, preview, report)
        raise ConnectionError('Camera connection lost')
    finally:
        try:
            if client.is_connected:
                topic.unsubscribe()
        finally:
            client.terminate()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', help='Robot hostname or IP address')
    parser.add_argument('--config', type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument('--distance', type=float,
                        help='Measured horizontal axle-centre-to-marker distance in metres (optional)')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/aruco_check'))
    args = parser.parse_args()
    if args.distance is not None and (not np.isfinite(args.distance) or args.distance <= 0):
        parser.error('--distance must be positive and finite')
    try:
        config = load_hardware_config(args.config, host=args.host)
        run_check(config, args.output_dir, args.distance)
    except KeyboardInterrupt:
        print('\nCamera check stopped.')
    except (OSError, ValueError, RuntimeError, ConnectionError, KeyError, cv2.error) as error:
        parser.exit(1, f'Camera check failed: {error}\n')


if __name__ == '__main__':
    main()
