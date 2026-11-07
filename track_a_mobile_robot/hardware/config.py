"""Load measured hardware settings separately from simulation spawn poses."""

from dataclasses import dataclass, fields, replace
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..control import DriveConfig
from ..perception import MARKER_TO_DOCK_ROTATION, make_transform
from ..scene_config import EstimationConfig, NavigationConfig, _number, _section

DEFAULT_HARDWARE_CONFIG = Path(__file__).resolve().parents[2] / 'config/hardware_config.yml'


@dataclass(frozen=True)
class HardwareNavigationConfig(NavigationConfig):
    """Independent handover tolerances; shared position/heading tolerances finish docking."""

    offset_position_tolerance_m: float = .03
    offset_heading_tolerance_deg: float = 3.

    def __post_init__(self):
        super().__post_init__()
        if self.offset_heading_tolerance_deg >= 90:
            raise ValueError('Offset heading tolerance must be less than 90 degrees')


@dataclass(frozen=True)
class HardwareConfig:
    host: str
    port: int
    topics: dict[str, str]
    joint_names: tuple[str, str]
    joint_signs: tuple[float, float]
    encoder_noise_std_rad: tuple[float, float]
    encoders_wrap: bool
    drive: DriveConfig
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    camera_to_base: np.ndarray
    lidar_to_base: np.ndarray
    camera_frame_id: str
    lidar_frame_id: str
    zero_is_no_return: bool
    dictionary_id: int
    marker_sizes_m: dict[int, float]
    marker_to_dock: dict[int, np.ndarray]
    estimation: EstimationConfig
    navigation: NavigationConfig
    timing: dict[str, float]
    marker_as_dock: bool = False
    accept_any_detection: bool = False
    obstacle_checks_enabled: bool = True


def scale_hardware_speeds(config, scale):
    """Scale navigation velocity limits only; retain drive limits and sensor settings."""
    if not math.isfinite(scale) or not 0 < scale <= 2:
        raise ValueError('Speed scale must be positive and at most 2')
    names = ('drive_speed_mps', 'pbvs_speed_mps', 'dock_speed_mps',
             'detour_speed_mps', 'turn_speed_rad_s')
    navigation = replace(config.navigation, **{name: getattr(config.navigation, name)*scale for name in names})
    return replace(config, navigation=navigation)


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} requires measured numeric values') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain finite values with shape {shape}')
    return result


def _transform(value, name):
    result = _array(value, (4, 4), name)
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError(f'{name} must be a rigid transform with a proper rotation')
    return result


def _text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f'{name} must be a non-empty string')
    return value


def load_hardware_config(path=DEFAULT_HARDWARE_CONFIG, *, host=None):
    """Reject incomplete calibration before opening a robot connection."""
    path = Path(path)
    try:
        values = yaml.safe_load(path.read_text())
        if not isinstance(values, dict) or set(values) != {
            'connection', 'wheels', 'camera', 'lidar', 'markers',
            'estimation', 'navigation', 'timing',
        }:
            raise ValueError('Expected connection, wheels, camera, lidar, markers, estimation, navigation, timing')
        connection = _section(values, 'connection', {'host', 'port', 'topics'})
        topics = _section(connection, 'topics', {'cmd_vel', 'image', 'laser_scan', 'joint_states'})
        for name, topic in topics.items():
            _text(topic, f'topics.{name}')
        port = connection['port']
        if type(port) is not int or not 0 < port < 65536:
            raise ValueError('connection.port must be an integer between 1 and 65535')
        wheels = _section(values, 'wheels', {
            'joint_names', 'joint_signs', 'encoder_noise_std_rad', 'encoders_wrap', 'drive',
        })
        names = wheels['joint_names']
        if not isinstance(names, list) or len(names) != 2 or names[0] == names[1]:
            raise ValueError('wheels.joint_names requires distinct [left, right] names')
        for name in names:
            _text(name, 'wheels.joint_names')
        signs = _array(wheels['joint_signs'], (2,), 'wheels.joint_signs')
        noise = _array(wheels['encoder_noise_std_rad'], (2,), 'wheels.encoder_noise_std_rad')
        if not np.isin(signs, [-1, 1]).all() or np.any(noise < 0):
            raise ValueError('Wheel signs must be +/-1 and encoder noise non-negative')
        if type(wheels['encoders_wrap']) is not bool:
            raise ValueError('wheels.encoders_wrap must be true or false')
        drive_values = _section(wheels, 'drive', {field.name for field in fields(DriveConfig)})
        drive = DriveConfig(**{name: _number(value, f'wheels.drive.{name}')
                               for name, value in drive_values.items()})
        camera = _section(values, 'camera', {
            'matrix', 'dist_coeffs', 'image_size', 'to_base', 'frame_id',
        })
        matrix = _array(camera['matrix'], (3, 3), 'camera.matrix')
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
            raise ValueError('camera.matrix must be a calibrated pinhole intrinsic matrix')
        dist = np.asarray(camera['dist_coeffs'], dtype=float)
        if dist.ndim != 1 or dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
            raise ValueError('camera.dist_coeffs requires 4, 5, 8, 12 or 14 finite coefficients')
        size = camera['image_size']
        if not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
            raise ValueError('camera.image_size requires positive integer [width, height]')
        if isinstance(values.get('lidar'), dict):
            values['lidar'] = {'obstacle_checks_enabled': True, **values['lidar']}
        lidar = _section(values, 'lidar', {'to_base', 'frame_id', 'zero_is_no_return', 'obstacle_checks_enabled'})
        if type(lidar['obstacle_checks_enabled']) is not bool:
            raise ValueError('lidar.obstacle_checks_enabled must be true or false')
        if type(lidar['zero_is_no_return']) is not bool:
            raise ValueError('lidar.zero_is_no_return must be true or false')
        # Old calibrated configurations omit reference and keep the mounted-dock mode.
        marker_values = values.get('markers')
        if isinstance(marker_values, dict):
            values['markers'] = {'reference': 'dock', 'accept_any_detection': False, **marker_values}
        markers = _section(values, 'markers', {'reference', 'dictionary', 'definitions', 'accept_any_detection'})
        if type(markers['accept_any_detection']) is not bool:
            raise ValueError('markers.accept_any_detection must be true or false')
        if markers['reference'] not in ('dock', 'marker'):
            raise ValueError('markers.reference must be dock or marker')
        marker_as_dock = markers['reference'] == 'marker'
        dictionary_name = markers['dictionary']
        if not isinstance(dictionary_name, str) or not dictionary_name.startswith('DICT_'):
            raise ValueError('markers.dictionary must name an OpenCV ArUco dictionary')
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if not isinstance(dictionary_id, int):
            raise ValueError(f'Unknown ArUco dictionary: {dictionary_name}')  # noqa: TRY004 - invalid config value
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        definitions = markers['definitions']
        if not isinstance(definitions, list) or not definitions:
            raise ValueError('At least one marker definition is required')
        if marker_as_dock and len(definitions) != 1:
            raise ValueError('Marker reference requires exactly one marker; multiple centres are different targets')
        sizes, mounts = {}, {}
        for marker in definitions:
            expected = {'id', 'size_m'} if marker_as_dock else {'id', 'size_m', 'to_dock'}
            if not isinstance(marker, dict) or set(marker) != expected:
                raise ValueError(f'Each marker requires exactly {sorted(expected)} for {markers["reference"]} reference')
            marker_id = marker['id']
            if type(marker_id) is not int or not 0 <= marker_id < len(dictionary.bytesList) or marker_id in sizes:
                raise ValueError('Marker IDs must be unique and belong to the selected dictionary')
            size_m = _number(marker['size_m'], 'marker.size_m')
            if size_m <= 0:
                raise ValueError('marker.size_m must be positive')
            sizes[marker_id] = size_m
            mounts[marker_id] = (make_transform(MARKER_TO_DOCK_ROTATION, [0, 0, 0]) if marker_as_dock
                                 else _transform(marker['to_dock'], 'marker.to_dock'))
        timing = _section(values, 'timing', {
            'encoder_timeout_s', 'camera_timeout_s', 'startup_timeout_s',
            'history_s', 'max_encoder_gap_s', 'command_timeout_s', 'future_tolerance_s',
        })
        timing = {name: _number(value, f'timing.{name}') for name, value in timing.items()}
        if any(value <= 0 for value in timing.values()):
            raise ValueError('Timing limits must be positive')
        if timing['history_s'] < max(timing['max_encoder_gap_s'], timing['camera_timeout_s']):
            raise ValueError('History must cover the encoder gap and camera timeout')
        estimation = EstimationConfig(**values['estimation'])
        navigation_values = dict(values['navigation'])
        # Existing hardware files keep their previous shared tolerances.
        navigation_values.setdefault('offset_position_tolerance_m', navigation_values['position_tolerance_m'])
        navigation_values.setdefault('offset_heading_tolerance_deg', navigation_values['heading_tolerance_deg'])
        navigation = HardwareNavigationConfig(**navigation_values)
        if timing['command_timeout_s'] <= 1 / estimation.update_hz:
            raise ValueError('Command timeout must exceed one control period')
        return HardwareConfig(
            host=_text(host if host is not None else connection['host'], 'connection.host'),
            port=port, topics=dict(topics),
            joint_names=tuple(names), joint_signs=tuple(signs),
            encoder_noise_std_rad=tuple(noise), encoders_wrap=wheels['encoders_wrap'], drive=drive,
            camera_matrix=matrix, dist_coeffs=dist, image_size=tuple(size),
            camera_to_base=_transform(camera['to_base'], 'camera.to_base'),
            lidar_to_base=_transform(lidar['to_base'], 'lidar.to_base'),
            camera_frame_id=_text(camera['frame_id'], 'camera.frame_id', empty=True),
            lidar_frame_id=_text(lidar['frame_id'], 'lidar.frame_id', empty=True),
            zero_is_no_return=lidar['zero_is_no_return'],
            dictionary_id=dictionary_id, marker_sizes_m=sizes, marker_to_dock=mounts,
            estimation=estimation, navigation=navigation, timing=timing,
            marker_as_dock=marker_as_dock,
            accept_any_detection=markers['accept_any_detection'],
            obstacle_checks_enabled=lidar['obstacle_checks_enabled'],
        )
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Cannot read hardware config {path}: {exc}') from exc
