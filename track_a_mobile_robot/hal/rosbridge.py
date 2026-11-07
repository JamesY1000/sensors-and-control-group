"""Rosbridge sensor acquisition, wheel commands and command watchdog."""

import base64
import math
import threading
import time

import cv2
import numpy as np

from ..control import limited_command
from ..sensing import CameraReading, LidarReading, WheelEncoderReading


def source_timestamp_ns(message):
    """Require an integer, nonzero source stamp; never invent acquisition time."""
    header = message.get('header')
    stamp = header.get('stamp') if isinstance(header, dict) else None
    if not isinstance(stamp, dict):
        raise ValueError('Sensor message has no header.stamp')  # noqa: TRY004 - invalid wire data
    if 'sec' in stamp or 'nanosec' in stamp:
        sec, nano = stamp.get('sec'), stamp.get('nanosec')
    else:
        sec, nano = stamp.get('secs'), stamp.get('nsecs')
    if type(sec) is not int or type(nano) is not int or sec < 0 or not 0 <= nano < 1_000_000_000:
        raise ValueError('Invalid ROS source timestamp')
    value = sec*1_000_000_000+nano
    if value == 0:
        raise ValueError('Zero source timestamp: configure the sensor driver clock')
    return value


class RosbridgeBackend:
    """Translate API messages to shared readings; own connection and command expiry.

    Source durations are anchored to local elapsed time at the first observed
    encoder message. This removes the clock epoch difference, not transport delay.
    All sensor publishers must use the same source clock. Local freshness records
    when polling first observes a new stamp, not the API callback receipt time.
    """

    def __init__(self, config, *, robot=None, clock=time.monotonic):
        self.config = config
        if robot is None:
            from rosbridge_api import RosbridgeRobot

            robot = RosbridgeRobot(
                config.host, port=config.port,
                cmd_vel_topic=config.topics['cmd_vel'], image_topic=config.topics['image'],
                laser_scan_topic=config.topics['laser_scan'],
                joint_state_topic=config.topics['joint_states'],
                connection_timeout=config.timing['startup_timeout_s'],
            )
        self.robot = robot
        self.clock = clock
        self.started_at = clock()
        self._source_origin = None
        self._source_offset = 0.
        self._stamps = {}
        self._seen_at = {}
        self.encoders = self.camera = self.lidar = None
        self.lidar_zero_count = 0
        self.lidar_raw_ranges_m = None
        self.encoder_gap_s = 0.
        self._raw_angles = None
        self._command_lock = threading.Lock()
        self._last_command_at = clock()
        self._closed = threading.Event()
        self._watchdog = None
        self._watchdog_error = None

    def now(self):
        return self.clock()-self.started_at

    def start(self):
        self.robot.setup()
        self.stop()
        self._watchdog = threading.Thread(target=self._watch_commands, daemon=True)
        self._watchdog.start()

    def _watch_commands(self):
        while not self._closed.wait(min(.05, self.config.timing['command_timeout_s']/2)):
            try:
                with self._command_lock:
                    if self.clock()-self._last_command_at > self.config.timing['command_timeout_s']:
                        if self.robot.is_connected:
                            self.robot.stop()
                        self._last_command_at = self.clock()
            except Exception as exc:  # noqa: BLE001 - report worker failure to the control thread
                self._watchdog_error = exc
                return

    def _new_time(self, message, stream):
        stamp = source_timestamp_ns(message)
        previous = self._stamps.get(stream)
        if previous is not None:
            if stamp == previous:
                return None
            if stamp < previous:
                raise ValueError(f'{stream} source clock went backwards; restart the run')
        if self._source_origin is None:
            self._source_origin = stamp
            self._source_offset = self.now()
        capture_time = self._source_offset+(stamp-self._source_origin)/1e9
        if capture_time > self.now()+self.config.timing['future_tolerance_s']:
            raise ValueError(f'{stream} timestamp is in the future; check source clocks')
        self._stamps[stream] = stamp
        self._seen_at[stream] = self.now()
        return capture_time

    @staticmethod
    def _check_frame(message, expected):
        if expected and message['header'].get('frame_id') != expected:
            raise ValueError(f'Expected sensor frame {expected!r}, got {message["header"].get("frame_id")!r}')

    def read(self):
        """Poll without waiting; return only a new encoder sample and cache other streams."""
        new_encoder = None
        self.encoder_gap_s = 0.
        message = self.robot.query_joint_states()
        if message is not None:
            capture_time = self._new_time(message, 'encoders')
            if capture_time is not None:
                names, positions = message.get('name', []), message.get('position', [])
                if len(names) != len(positions) or any(names.count(n) != 1 for n in self.config.joint_names):
                    raise ValueError('Joint state positions must contain each configured wheel exactly once')
                angles = np.array([positions[names.index(n)] for n in self.config.joint_names], dtype=float)
                angles *= self.config.joint_signs
                if not np.isfinite(angles).all():
                    raise ValueError('Wheel positions must be finite')
                unwrapped = angles.copy()
                if self.encoders is not None:
                    dt = capture_time-self.encoders.time
                    if dt > self.config.timing['max_encoder_gap_s']:
                        if self.config.encoders_wrap:
                            raise ValueError('Encoder sampling gap exceeded with wrapped positions; restart required')
                        # Absolute wheel positions can resume, but the missing
                        # curved path cannot be reconstructed from their totals.
                        # The runner discards that pose and relocalises at a marker.
                        self.encoder_gap_s = dt
                    delta = angles-self._raw_angles
                    if self.config.encoders_wrap:
                        delta = (delta+np.pi) % (2*np.pi)-np.pi
                    if np.max(np.abs(delta)) > 1.5*self.config.drive.max_wheel_speed_rad_s*dt+.05:
                        raise ValueError('Implausible wheel increment; check signs, units or encoder reset')
                    unwrapped = np.array([self.encoders.left_encoder_rad, self.encoders.right_encoder_rad])+delta
                self._raw_angles = angles
                new_encoder = self.encoders = WheelEncoderReading(capture_time, *map(float, unwrapped))

        # The encoder clock anchors all streams. Do not consume camera/scan data
        # before it is available, or cached first frames could never be processed.
        if self.encoders is None:
            return new_encoder
        message = self.robot.query_image(timeout=0, wait_for_new=True, raw=True)
        if message is not None:
            capture_time = self._new_time(message, 'camera')
            if capture_time is not None:
                self._check_frame(message, self.config.camera_frame_id)
                encoded = np.frombuffer(base64.b64decode(message['data'], validate=True), dtype=np.uint8)
                bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if bgr is None or tuple(bgr.shape[1::-1]) != self.config.image_size:
                    raise ValueError('Camera image does not match the calibrated resolution')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb.setflags(write=False)
                self.camera = CameraReading(capture_time, rgb)
        message = self.robot.query_laser_scan(timeout=0, wait_for_new=True, raw=True)
        if message is not None:
            capture_time = self._new_time(message, 'lidar')
            if capture_time is not None:
                self._check_frame(message, self.config.lidar_frame_id)
                ranges = np.array(message['ranges'], dtype=float)
                low, high = float(message['range_min']), float(message['range_max'])
                start, step = float(message['angle_min']), float(message['angle_increment'])
                if (ranges.ndim != 1 or ranges.size < 2 or not all(map(math.isfinite, (low, high, start, step)))
                        or not 0 < low < high or step == 0):
                    raise ValueError('Invalid LiDAR scan geometry or range limits')
                self.lidar_zero_count = int(np.count_nonzero(ranges == 0))
                self.lidar_raw_ranges_m = tuple(map(float, ranges))
                if self.config.zero_is_no_return:
                    ranges[ranges == 0] = np.inf
                ranges[ranges < low] = np.nan
                ranges[ranges > high] = np.inf
                self.lidar = LidarReading(
                    capture_time, tuple(start+step*np.arange(ranges.size)), tuple(ranges), low, high,
                )
        return new_encoder

    def health(self):
        """Return a stop reason, or None when required streams are fresh."""
        if not self.robot.is_connected:
            return 'Rosbridge disconnected'
        if self._watchdog_error is not None:
            raise RuntimeError('Command watchdog failed') from self._watchdog_error
        now = self.now()
        limits = {
            'encoders': self.config.timing['encoder_timeout_s'],
            'camera': self.config.timing['camera_timeout_s'],
            'lidar': self.config.navigation.lidar_timeout_s,
        }
        for name, limit in limits.items():
            reading = getattr(self, name)
            if reading is None:
                return f'Waiting for {name}'
            if now-self._seen_at[name] > limit or now-reading.time > limit:
                return f'Stale {name}'
        return None

    def send_command(self, command):
        command = limited_command(command.time, command.v_mps, command.omega_rad_s, self.config.drive)
        with self._command_lock:
            self.robot.send_command(command.v_mps, command.omega_rad_s)
            self._last_command_at = self.clock()

    def stop(self):
        with self._command_lock:
            if self.robot.is_connected:
                self.robot.stop()
            self._last_command_at = self.clock()

    def close(self):
        try:
            self.stop()
        finally:
            self._closed.set()
            if self._watchdog is not None:
                self._watchdog.join(timeout=1.)
            self.robot.cleanup()


