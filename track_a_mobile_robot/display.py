"""
Our display helper should show the robot's camera feed inside the MuJoCo viewer.

It uses the latest image and marker observations already produced by perception:
1. Draw the marker outlines, IDs and axes onto a copy of the image
2. Add the image timestamp and whether this frame supplied an EKF correction
3. Show the preview in the bottom-left of the scene, keeping it between frames
"""

from typing import TYPE_CHECKING

import cv2
import mujoco

from .perception import Perception, PerceptionOutput

if TYPE_CHECKING:
    from mujoco.viewer import Handle


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
