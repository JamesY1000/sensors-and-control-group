"""Regenerate both dock marker textures: python -m scripts.generate_marker."""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    output_dir = Path(__file__).resolve().parents[1] / "world" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match the black-square / panel ratios in world/charging_station.xml:
    # ID 0: 0.40 m marker on a 0.56 m panel (300 + 2 * 60 pixels).
    # ID 1: 0.04 m marker on a 0.06 m panel (400 + 2 * 100 pixels).
    for marker_id, marker_pixels, margin_pixels in ((0, 300, 60), (1, 400, 100)):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_pixels)
        panel_pixels = marker_pixels + 2 * margin_pixels
        panel = np.full((panel_pixels, panel_pixels), 255, dtype=np.uint8)
        end = margin_pixels + marker_pixels
        panel[margin_pixels:end, margin_pixels:end] = marker

        # Save three-channel textures for MuJoCo.
        destination = output_dir / f"marker_{marker_id}.png"
        if not cv2.imwrite(str(destination), cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)):
            raise RuntimeError(f"Could not write {destination}")
        print(destination)


if __name__ == "__main__":
    main()
