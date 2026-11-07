"""Calibrate camera intrinsics from checkerboard views streamed over rosbridge.

Run from the repository root:
    python -m track_a_mobile_robot.hardware.calibrate_camera ROBOT_IP --square-size-mm 25

The default board has 9 x 6 INNER corners (10 x 7 squares). Only camera images
are needed. Keep the robot still and move the board between captures.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import DEFAULT_HARDWARE_CONFIG

MIN_VIEWS = 12
WINDOW = 'Pi camera calibration - Space capture | U undo | C calibrate | Q quit'


def board_points(pattern_size, square_size_m):
    """Return planar checkerboard points in column/row order, in metres."""
    columns, rows = pattern_size
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_size_m
    return points


def find_corners(image, pattern_size):
    """Find subpixel internal corners in the original, unscaled BGR image."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        grey, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    # OpenCV 4/5 bindings differ in whether the singleton axis is retained.
    return corners.reshape(-1, 1, 2) if found else None


def repeated_view(corners, views, image_size):
    """Reject almost identical views, including reversed corner indexing."""
    threshold = 0.01 * np.hypot(*image_size)
    for previous in views:
        for ordered in (corners, corners[::-1]):
            rms = np.sqrt(np.mean(np.sum((ordered - previous) ** 2, axis=-1)))
            if rms < threshold:
                return True
    return False


def calibrate(views, image_size, pattern_size, square_size_m):
    """Fit the raw-image pinhole model and report reprojection error in pixels."""
    if len(views) < MIN_VIEWS:
        raise ValueError(f'Capture at least {MIN_VIEWS} different views before calibrating')
    if not np.isfinite(square_size_m) or square_size_m <= 0:
        raise ValueError('Square size must be positive and finite')
    views = [np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2) for corners in views]
    points = board_points(pattern_size, square_size_m)
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        [points] * len(views), views, image_size, None, None,
    )
    if (not np.isfinite(rms) or not np.isfinite(matrix).all()
            or not np.isfinite(distortion).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        raise ValueError('Calibration failed; collect sharper views with varied board tilts')
    errors = []
    for corners, rotation, translation in zip(views, rotations, translations, strict=True):
        projected, _ = cv2.projectPoints(points, rotation, translation, matrix, distortion)
        projected = projected.reshape(-1, 1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum((corners - projected) ** 2, axis=-1)))))
    camera = {
        'image_size': list(image_size),
        'matrix': matrix.tolist(),
        'dist_coeffs': distortion.ravel().tolist(),
    }
    report = {
        'inner_corners': list(pattern_size),
        'square_size_m': square_size_m,
        'views': len(views),
        'rms_error_px': float(rms),
        'view_errors_px': {f'view_{index:03d}.png': error
                           for index, error in enumerate(errors, start=1)},
    }
    return camera, report


def collect_views(robot, pattern_size, output_dir):
    """Preview and capture fresh images; never query encoders or LiDAR."""
    views = []
    image_size = None
    last_frame_at = time.monotonic()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print('Focus the camera window. Space: capture; U: undo; C: calibrate; Q/Escape: quit.')
    print(f'Capture 20-30 sharp views (minimum {MIN_VIEWS}); vary position, distance and tilt.')
    try:
        while robot.is_connected:
            image = robot.query_image(timeout=0.1, wait_for_new=True)
            corners = None
            if image is not None:
                last_frame_at = time.monotonic()
                size = (image.shape[1], image.shape[0])
                if image_size is not None and size != image_size:
                    raise ValueError('Image resolution changed; restart with a fixed camera mode')
                image_size = size
                corners = find_corners(image, pattern_size)
                preview = image.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(preview, pattern_size, corners, True)
                message = f'{len(views)} views | ' + (
                    'Board found - Space to capture' if corners is not None
                    else f'Show all {pattern_size[0]}x{pattern_size[1]} inner corners'
                )
                cv2.putText(preview, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (0, 255, 0) if corners is not None else (0, 0, 255), 2)
                cv2.imshow(WINDOW, preview)
            elif time.monotonic() - last_frame_at > 10:
                raise RuntimeError('No new camera images for 10 seconds; check the image topic')

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None, image_size
            if key in (ord('u'), ord('U')) and views:
                (output_dir / f'view_{len(views):03d}.png').unlink()
                views.pop()
                print(f'Removed last view; {len(views)} remaining.')
            elif key == ord(' '):
                if corners is None:
                    print('No fresh complete checkerboard detected; capture skipped.')
                elif repeated_view(corners, views, image_size):
                    print('View too similar to an existing capture; move or tilt the board.')
                else:
                    path = output_dir / f'view_{len(views) + 1:03d}.png'
                    if not cv2.imwrite(str(path), image):
                        raise OSError(f'Could not save {path}')
                    views.append(corners.copy())
                    print(f'Captured {path.name} ({len(views)} views).')
            elif key in (ord('c'), ord('C')):
                if len(views) < MIN_VIEWS:
                    print(f'Need at least {MIN_VIEWS} different views; currently {len(views)}.')
                else:
                    return views, image_size
        raise ConnectionError('Camera connection lost; captured images remain in the output directory')
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', help='Robot hostname or IP address')
    parser.add_argument('--square-size-mm', required=True, type=float,
                        help='Measured side length of ONE printed square, in millimetres')
    parser.add_argument('--columns', type=int, default=9, help='Internal corner columns (default: 9)')
    parser.add_argument('--rows', type=int, default=6, help='Internal corner rows (default: 6)')
    parser.add_argument('--config', type=Path, default=DEFAULT_HARDWARE_CONFIG,
                        help='Hardware config supplying connection settings and mounting transforms')
    parser.add_argument('--output-dir', type=Path,
                        help='New directory for captured images, report and calibrated hardware config')
    args = parser.parse_args()
    if not np.isfinite(args.square_size_mm) or args.square_size_mm <= 0:
        parser.error('--square-size-mm must be positive and finite')
    if args.columns < 3 or args.rows < 3:
        parser.error('--columns and --rows must each be at least 3 internal corners')
    if not args.host.strip():
        parser.error('host must not be empty')

    output_dir = args.output_dir or Path('outputs/camera_calibration') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    try:
        # The normal hardware loader intentionally rejects missing intrinsics/dock
        # calibration; read the template without requiring those values here.
        config = yaml.safe_load(args.config.read_text())
        connection = config['connection']
        image_topic = connection['topics']['image']
        port = connection['port']
        if not isinstance(config['camera'], dict) or not isinstance(image_topic, str) or not image_topic.strip():
            raise ValueError('Config must contain camera settings and a compressed image topic')
        if type(port) is not int or not 0 < port < 65536:
            raise ValueError('Config connection.port must be between 1 and 65535')
        output_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        parser.error(str(error))

    from rosbridge_api import RosbridgeRobot

    pattern_size = (args.columns, args.rows)
    print(f'Raw image topic: {image_topic}; output: {output_dir.resolve()}')
    try:
        with RosbridgeRobot(args.host, port=port, image_topic=image_topic,
                            cmd_vel_topic=connection['topics']['cmd_vel']) as robot:
            robot.stop()
            views, image_size = collect_views(robot, pattern_size, output_dir)
        if views is None:
            print('Cancelled; no calibration written. Captured images are retained.')
            return
        print('Calculating calibration...')
        camera, report = calibrate(views, image_size, pattern_size, args.square_size_mm / 1000)
        config['camera'].update(camera)
        config['connection']['host'] = args.host
        calibrated_path = output_dir / 'hardware_config.yml'
        calibrated_path.write_text(yaml.safe_dump(config, sort_keys=False))
        (output_dir / 'calibration_report.yml').write_text(yaml.safe_dump(report, sort_keys=False))
        print(f"RMS reprojection error: {report['rms_error_px']:.3f} px")
        if report['rms_error_px'] > 1.0:
            print('Error exceeds 1 px; review captures and repeat with sharper, more varied views.')
        print('A low error alone does not prove accuracy; check marker distances before docking.')
        print(f'Calibrated configuration: {calibrated_path.resolve()}')
        print('Camera mounting transform retained. Fill any remaining marker/dock settings in that file.')
        print(f'Use it with: python -m track_a_mobile_robot.main --backend rosbridge --config {calibrated_path} --dry-run --verbose --seconds 30')
    except KeyboardInterrupt:
        print('\nCancelled; captured images are retained.')
    except (OSError, ValueError, RuntimeError, ConnectionError, cv2.error) as error:
        parser.exit(1, f'Calibration failed: {error}\n')


if __name__ == '__main__':
    main()
