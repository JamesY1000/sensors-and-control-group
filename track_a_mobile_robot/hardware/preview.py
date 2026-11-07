"""Optional camera overlay in a separate process; never subscribe or command robots."""

import multiprocessing
import os
import queue
import signal
import sys
import time

import cv2
import numpy as np

WINDOW = 'Robot demo camera - Q / Escape stops demo'


def render_preview(packet, config, *, elapsed=0.):
    """Draw existing navigation pose detections, without running detection again."""
    camera = packet['camera']
    if camera is None:
        frame = np.zeros((config.image_size[1], config.image_size[0], 3), np.uint8)
    else:
        frame = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2BGR)
    for marker in packet['markers']:
        colour = (80, 220, 80) if marker.orientation_reliable else (0, 180, 255)
        points = np.rint(marker.corners_px).astype(np.int32)
        cv2.polylines(frame, [points], True, colour, 2)
        origin = (max(5, int(points[:, 0].min())), max(20, int(points[:, 1].min())-8))
        cv2.putText(frame, f'ID {marker.marker_id}  {marker.range_m:.2f} m', origin,
                    cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2, cv2.LINE_AA)
        rvec, _ = cv2.Rodrigues(marker.T_camera_marker[:3, :3])
        cv2.drawFrameAxes(frame, config.camera_matrix, config.dist_coeffs, rvec,
                          marker.T_camera_marker[:3, 3], config.marker_sizes_m[marker.marker_id]/2)
    age = None if camera is None else packet['camera_age'] + elapsed
    stale = age is None or age > config.timing['camera_timeout_s']
    v, w = packet['sent_command']
    marker = packet.get('marker_status', 'Marker: waiting').split(';', 1)[0]
    mode = 'DRY RUN | ' if packet['dry_run'] else ''
    status = packet['status'].split(';', 1)[0]
    if elapsed > config.timing['command_timeout_s']:
        status = 'Display delayed - showing last update'
    lines = [
        f"{mode}{packet['phase']} | {marker}",
        status,
        f'v={v:+.2f} m/s   w={w:+.2f} rad/s  |  Q/Esc: stop',
    ]
    panel = np.full((len(lines)*24+12, max(640, frame.shape[1]), 3), 25, np.uint8)
    for i, line in enumerate(lines):
        if cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0] > panel.shape[1]-20:
            while cv2.getTextSize(line + '...', cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0] > panel.shape[1]-20:
                line = line[:-1]
            line += '...'
        cv2.putText(panel, line, (10, 25+i*24), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (230, 230, 230), 1, cv2.LINE_AA)
    frame = cv2.copyMakeBorder(frame, 0, 0, 0, panel.shape[1]-frame.shape[1], cv2.BORDER_CONSTANT)
    if stale:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 150), -1)
        cv2.putText(frame, 'WAITING FOR FRESH CAMERA', (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((frame, panel))


def _show_frames(frames, stopped, ready, config):
    # Ctrl+C is handled by the parent, which stops the robot before GUI cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)
    packet = dict(camera=None, markers=[], camera_age=None, sent_command=(0., 0.),
                  wheels=(0., 0.), phase='STARTUP', status='Waiting for demo', dry_run=True)
    updated_at = time.monotonic()
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW, render_preview(packet, config))
        cv2.waitKey(1)
        ready.set()
        while not stopped.is_set():
            try:
                packet = frames.get(timeout=.05)
                updated_at = packet['display_sent_at']
            except queue.Empty:
                pass
            cv2.imshow(WINDOW, render_preview(packet, config, elapsed=time.monotonic()-updated_at))
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                stopped.set()
    finally:
        cv2.destroyAllWindows()


class CameraPreview:
    """Bounded, nonblocking frame delivery keeps a slow GUI off the control loop."""

    def __init__(self, config):
        self.config = config
        context = multiprocessing.get_context('spawn')
        self.frames = context.Queue(maxsize=1)
        self.stopped = context.Event()
        self.ready = context.Event()
        self.process = context.Process(target=_show_frames,
                                       args=(self.frames, self.stopped, self.ready, config), daemon=True)
        self.next_frame_at = 0.

    def start(self):
        if sys.platform == 'linux' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('--show-camera requires a graphical desktop')
        self.process.start()
        if not self.ready.wait(timeout=5.) or not self.process.is_alive():
            raise RuntimeError('Camera window could not start; check the graphical desktop/OpenCV installation')

    def running(self):
        if self.stopped.is_set():
            return False
        if not self.process.is_alive():
            raise RuntimeError('Camera window exited unexpectedly')
        return True

    def publish(self, **packet):
        now = time.monotonic()
        if now < self.next_frame_at:
            return
        self.next_frame_at = now + .2  # Display at most 5 Hz; control stays independent.
        packet['display_sent_at'] = now
        try:
            self.frames.put_nowait(packet)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(packet)
            except queue.Full:
                pass

    def close(self):
        self.stopped.set()
        if self.process.pid is not None:
            self.process.join(timeout=1.)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.)
        self.frames.cancel_join_thread()
        self.frames.close()
