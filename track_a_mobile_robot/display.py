"""
Our display helpers show the camera feed and navigation overlays in the MuJoCo viewer.

It uses the latest image and marker observations already produced by perception:
1. Draw the marker outlines, IDs and axes onto a copy of the image
2. Add the image timestamp and whether this frame supplied an EKF correction
3. Show the preview in the bottom-left of the scene, keeping it between frames

The floor overlay shows the planned route, current reference and LiDAR hits
in the estimated world frame. Viewer-only geometry does not enter the camera,
LiDAR readings or collision model.
"""

from typing import TYPE_CHECKING

import cv2
import mujoco
import numpy as np

from .perception import Perception, PerceptionOutput, lidar_points_to_world

if TYPE_CHECKING:
    from mujoco.viewer import Handle
    from .estimation import StateEstimate
    from .navigation import Navigator, TrajectoryReference


def update_camera_preview(
    window: "Handle",
    perception: Perception,
    output: PerceptionOutput,
) -> None:
    """Display a new camera frame after estimation has processed it."""
    # Intermediate updates have no new marker list. Keep the previous preview.
    if not output.new_camera_frame:
        return

    viewport = window.viewport
    if viewport is None:
        return

    # Fit the preview inside the scene area, allowing for the viewer's side panels.
    margin = 12
    header_height = 50
    image_height, image_width = output.camera.rgb.shape[:2]
    available_height = viewport.height - 2 * margin - header_height
    width = min(400, viewport.width - 2 * margin, int(available_height * image_width / image_height))
    if width < 160:
        window.clear_images()
        return
    height = int(width * image_height / image_width)

    # draw() returns a separate RGB image, so annotations never alter sensing data.
    preview = perception.detector.draw(output.camera, output.markers)
    preview = cv2.resize(preview, (width, height), interpolation=cv2.INTER_AREA)
    preview = cv2.copyMakeBorder(
        preview, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 28, 34),
    )

    # Detection and a usable correction are different: show what this image supplied.
    observation = output.pose_observation
    if observation is not None:
        status = f"EKF corrected using marker {observation.marker_id}"
        colour = (100, 230, 130)
    elif output.markers:
        status = "Marker pose not accepted - encoders only"
        colour = (255, 190, 80)
    else:
        status = "No usable marker - encoders only"
        colour = (200, 205, 215)

    font_scale = 0.48 * width / 400
    cv2.putText(
        preview, f"ROBOT CAMERA   t = {output.camera.time:.2f} s", (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 245), 1, cv2.LINE_AA,
    )
    cv2.putText(
        preview, status, (8, 40), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, colour, 1, cv2.LINE_AA,
    )

    # MuJoCo keeps this overlay until replaced and handles the image orientation.
    rectangle = mujoco.MjrRect(
        viewport.left + margin, viewport.bottom + margin, width, height + header_height,
    )
    window.set_images((rectangle, preview))


def _marker(scene, xy, colour, radius=0.035):
    """Draw a small floor marker without adding a physical simulation geom."""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                       [radius, .008, 0], [*xy, .025], np.eye(3).ravel(), colour)
    scene.ngeom += 1


def _line(scene, start, end, colour, width=.012):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.asarray(end) - start) < 1e-8:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                       np.zeros(3), np.zeros(3), np.eye(3).ravel(), colour)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        width, [*start, .035], [*end, .035])
    scene.ngeom += 1


class NavigationOverlay:
    """Cache each scan's world points and refresh lightweight floor geometry at 10 Hz."""

    def __init__(self):
        self._scan_time = None
        self._points = np.empty((0, 2))
        self._draw_time = -np.inf

    def update(self, window: "Handle", estimate: "StateEstimate", output: PerceptionOutput,
               navigator: "Navigator | None" = None, reference: "TrajectoryReference | None" = None):
        # Transform once when a scan arrives. Moving the robot must not move cached hits.
        if estimate.time < self._draw_time:
            self.__init__()
        if output.lidar.time != self._scan_time:
            self._points = lidar_points_to_world(output.lidar_points_base, estimate.pose_world)
            self._scan_time = output.lidar.time
        if estimate.time - self._draw_time < .1 - 1e-9 or window.user_scn is None:
            return
        self._draw_time = estimate.time
        pose = np.asarray(estimate.pose_world)
        cyan, orange = (0, .9, 1, 1), (1, .35, .05, 1)
        blue, yellow = (.15, .4, 1, 1), (1, .9, 0, 1)
        green, purple = (.1, 1, .3, 1), (.85, .2, 1, 1)

        with window.lock():
            scene = window.user_scn
            scene.ngeom = 0
            # Draw the route and goals first so they survive any geometry-capacity limit.
            if navigator is not None:
                _marker(scene, navigator.approach_pose[:2], green, .08)
                _marker(scene, navigator.docking_pose[:2], purple, .08)
                path = [pose[:2], *navigator.waypoints] if navigator.phase == "APPROACH" else []
                if navigator.phase == "DOCK":
                    path = [navigator.approach_pose[:2], navigator.docking_pose[:2]]
                for start, end in zip(path, path[1:]):
                    _line(scene, start, end, cyan)
                    _marker(scene, end, cyan)
                # The circle is the navigator's stop clearance, centred on the estimate.
                angles = np.linspace(0, 2 * np.pi, 25)
                circle = pose[:2] + navigator.settings.clearance_m * np.column_stack((np.cos(angles), np.sin(angles)))
                for start, end in zip(circle, circle[1:]):
                    _line(scene, start, end, blue, .005)

            heading = pose[:2] + .35 * np.array([np.cos(pose[2]), np.sin(pose[2])])
            _line(scene, pose[:2], heading, blue)
            if reference is not None and not reference.stop_requested:
                target = np.asarray(reference.pose_world)
                _marker(scene, target[:2], yellow, .045)
                _line(scene, target[:2], target[:2] + .18 * np.array([np.cos(target[2]), np.sin(target[2])]), yellow)
            # Orange dots are measured hit positions, not an inflated occupancy map.
            timeout = navigator.settings.lidar_timeout_s if navigator is not None else .5
            fresh = 0 <= estimate.time - output.lidar.time <= timeout
            for point in self._points:
                _marker(scene, point, orange if fresh else (.5, .5, .5, 1), .025)

        status = f"{navigator.phase}: {navigator.status}" if navigator else "Stationary sensor demo"
        legend = ("Cyan: route | Orange: LiDAR hits (grey = stale)\n"
                  "Yellow: reference | Blue: estimated heading / clearance\n"
                  "Green: approach | Purple: dock")
        window.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          f"{status}\n{legend}", ""))
