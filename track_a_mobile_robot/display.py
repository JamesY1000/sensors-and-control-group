"""Sensor panels and display-only robot-relative graphics in the MuJoCo viewer."""

import cv2
import mujoco
import numpy as np
import queue
import threading
from collections import deque

from .perception import lidar_points_to_local


class OverlayPublisher:
    """Keep only the latest overlay; viewer uploads must never block control."""

    def __init__(self, window):
        self.window = window
        self.pending = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='sensor-overlays')
        self.thread.start()

    def submit(self, images, texts):
        # MuJoCo waits for its render thread to consume the previous upload.
        # A minimised/slow viewer can take seconds. Drop old frames, not sim steps.
        try:
            self.pending.get_nowait()
        except queue.Empty:
            pass
        self.pending.put_nowait((images, texts))

    def _run(self):
        while not self.closed.is_set():
            try:
                images, texts = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            if not self.window.is_running():
                break
            self.window.set_images(images)
            if not self.closed.is_set():
                self.window.set_texts(texts)

    def close(self):
        self.closed.set()
        # Viewer closure also releases an outstanding native upload wait.
        self.thread.join(timeout=.2)


class SensorDisplay:
    """Cache the camera inset and refresh a local map at 10 Hz."""

    def __init__(self):
        self._publisher = None
        self.reset()

    def reset(self):
        self._camera_image = None
        self._draw_time = -np.inf
        self._scan_time = None
        self._points = np.empty((0, 2))
        self._acquired = False
        self._trail = deque(maxlen=600)

    def close(self):
        if self._publisher is not None:
            self._publisher.close()

    def _draw_scene(self, window, data, estimate, navigator):
        """Place relative graphics in the viewer, without changing robot inputs.

        The simulator pose is ONLY a drawing transform. This preview attaches
        the estimated relative plan to the visible robot; it is not a comparison
        between true and estimated world poses. Nothing here feeds back to control.
        """
        if navigator is None:
            return
        pose = np.asarray(estimate.pose_local)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        local_to_base = np.array([[c, -s], [s, c]])
        body = data.body('base')
        rotation = body.xmat.reshape(3, 3)[:2, :2]
        origin = body.xpos[:2].copy()

        def world(points):
            base = (np.asarray(points) - pose[:2]) @ local_to_base
            return np.column_stack((base @ rotation.T + origin, np.full(len(base), .025)))

        # user_scn contains decorative geoms only: no collisions or sensor returns.
        with window.lock():
            scene = window.user_scn
            scene.ngeom = 0
            def line(a, b, colour, width=.006, arrow=False):
                if scene.ngeom >= scene.maxgeom or np.linalg.norm(b-a) < 1e-8:
                    return
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                   np.zeros(3), np.zeros(3), np.eye(3).ravel(), colour)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW if arrow
                                    else mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
                scene.ngeom += 1

            def polyline(points, colour):
                vertices = world(points)
                for a, b in zip(vertices[:-1], vertices[1:]):
                    line(a, b, colour)

            blue, cyan, white = [0.2, .5, 1., 1.], [.1, .9, .9, 1.], [1., 1., 1., 1.]
            angles = np.linspace(0, 2*np.pi, 49) + pose[2]
            circle = np.column_stack((np.cos(angles), np.sin(angles)))
            # This is the configured collision-check radius, not a new physical size.
            polyline(pose[:2] + navigator.settings.clearance_m*circle, blue)
            heading = pose[:2] + .3*np.array([c, s])
            a, b = world([pose[:2], heading])
            line(a, b, blue, arrow=True)
            preview = navigator.trajectory_preview
            if preview is not None:
                start, end, reference = preview
                if np.linalg.norm(end[:2]-start[:2]) < 1e-8:
                    # A turn has no translational path; show its heading sweep.
                    angles = np.linspace(start[2], end[2], 25)
                    polyline(start[:2] + .32*np.column_stack((np.cos(angles), np.sin(angles))), cyan)
                else:
                    path = getattr(navigator, 'trajectory_path', None)
                    polyline(path[:, :2] if path is not None else [start[:2], end[:2]], cyan)
                polyline(reference[:2] + .035*circle, white)
            target = (navigator.approach_pose if navigator.phase == 'ALIGN' else
                      navigator.docking_pose if navigator.phase in ('DOCK', 'DONE') else None)
            if target is not None:
                polyline(target[:2] + .045*circle, [.9, .25, .9, 1.])
                a, b = world([target[:2], target[:2]+.18*np.array([np.cos(target[2]), np.sin(target[2])])])
                line(a, b, [.9, .25, .9, 1.], arrow=True)

    def update(self, window, perception, estimate, output, navigator=None, *, data=None):
        if output.new_camera_frame:
            image = perception.detector.draw(output.camera, output.markers)
            self._camera_image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
            self._camera_image = cv2.copyMakeBorder(self._camera_image, 42, 0, 0, 0,
                                                   cv2.BORDER_CONSTANT, value=(24, 28, 34))
            accepted = output.dock_observation is not None
            status = (f'Accepted marker {output.dock_observation.marker_id}' if accepted
                      else 'No accepted dock measurement')
            cv2.putText(self._camera_image, f'ROBOT CAMERA  t={output.camera.time:.2f}s',
                        (7, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
            cv2.putText(self._camera_image, status, (7, 34), cv2.FONT_HERSHEY_SIMPLEX, .4,
                        (100, 230, 130) if accepted else (255, 190, 80), 1)
        acquired = estimate.dock_pose_local is not None
        if acquired != self._acquired:
            self._acquired = acquired
            self._scan_time = None
            self._trail.clear()
        if output.lidar.time != self._scan_time:
            self._points = lidar_points_to_local(output.lidar_points_base, estimate.pose_local)
            self._scan_time = output.lidar.time
        if estimate.time - self._draw_time < .1 - 1e-9:
            return
        self._draw_time = estimate.time
        if data is not None:
            self._draw_scene(window, data, estimate, navigator)
        viewport = window.viewport
        if viewport is None:
            return
        pose = np.asarray(estimate.pose_local)
        # Only acquired poses share a fixed local frame; do not plot exploration as a trail.
        if acquired and (not self._trail or np.linalg.norm(pose[:2] - self._trail[-1]) >= .03):
            self._trail.append(pose[:2].copy())
        goals = []
        preview = None
        if navigator is not None:
            goals = [(p[:2], colour) for p, colour in zip(
                (navigator.coarse_pose, navigator.approach_pose, navigator.docking_pose, navigator.detour_pose),
                ((50, 240, 90), (80, 220, 180), (220, 70, 255), (255, 230, 50))) if p is not None]
            preview = navigator.trajectory_preview
        trajectory = navigator.trajectory_path if navigator is not None else None
        path = trajectory[:, :2] if trajectory is not None else np.empty((0, 2))
        bounds = np.vstack([pose[:2], self._points, path, *self._trail, *(p for p, _ in goals)])
        lower, upper = bounds.min(axis=0) - .5, bounds.max(axis=0) + .5
        centre = (lower + upper) / 2
        span = max(2.0, float((upper - lower).max()))
        size = 320
        def pixel(xy):
            xy = (np.asarray(xy) - centre) / span * (size - 36)
            return int(size / 2 + xy[0]), int(size / 2 - xy[1] + 15)
        local_map = np.full((size + 30, size, 3), (24, 28, 34), dtype=np.uint8)
        title = 'LOCAL ODOMETRY (metres)' if acquired else 'ROBOT FRAME - exploring'
        cv2.putText(local_map, title, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (235, 235, 235), 1)
        for point in self._points:
            cv2.circle(local_map, pixel(point), 2, (255, 130, 40), -1)
        if len(self._trail) > 1:
            trail = np.array([pixel(p) for p in self._trail], dtype=np.int32)
            cv2.polylines(local_map, [trail], False, (115, 145, 185), 1, cv2.LINE_AA)
        if preview is not None:
            start, end, reference = preview
            # Only show the segment LQI is tracking, not an unplanned route to the dock.
            vertices = np.array([pixel(p) for p in path], dtype=np.int32)
            cv2.polylines(local_map, [vertices], False, (40, 230, 230), 2, cv2.LINE_AA)
            heading = end[:2] + .25 * np.array([np.cos(end[2]), np.sin(end[2])])
            cv2.arrowedLine(local_map, pixel(end[:2]), pixel(heading), (40, 230, 230), 1, tipLength=.3)
            cv2.circle(local_map, pixel(reference[:2]), 6, (245, 245, 245), 1, cv2.LINE_AA)
        for point, colour in goals:
            cv2.circle(local_map, pixel(point), 5, colour, -1)
        cv2.circle(local_map, pixel(pose[:2]), 4, (70, 130, 255), -1)
        if navigator is not None:
            radius = max(1, round(navigator.settings.clearance_m / span * (size-36)))
            cv2.circle(local_map, pixel(pose[:2]), radius, (70, 130, 255), 1, cv2.LINE_AA)
        heading = pose[:2] + .35 * np.array([np.cos(pose[2]), np.sin(pose[2])])
        cv2.line(local_map, pixel(pose[:2]), pixel(heading), (70, 130, 255), 2)
        cv2.putText(local_map, 'Orange: LiDAR  Cyan: path  Blue: clearance', (7, size + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (220, 220, 220), 1)
        panels = [self._camera_image, local_map] if self._camera_image is not None else [local_map]
        scale = min(1.0, (viewport.width - 24) / (size * len(panels) + 12),
                    (viewport.height - 24) / max(p.shape[0] for p in panels))
        if scale < .45:
            return
        images = []
        left = viewport.left + 12
        for panel in panels:
            width, height = int(panel.shape[1] * scale), int(panel.shape[0] * scale)
            resized = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
            images.append((mujoco.MjrRect(left, viewport.bottom + 12, width, height), resized))
            left += width + 12
        title = f'{navigator.phase}  |  {estimate.time:.1f} s' if navigator else 'Sensor demo'
        status = navigator.status if navigator else 'Wheels stopped'
        texts = (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 title + '\n' + status, '')
        if self._publisher is None:
            self._publisher = OverlayPublisher(window)
        self._publisher.submit(images, texts)
