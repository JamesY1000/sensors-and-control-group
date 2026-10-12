"""Regenerate the station's printable ArUco texture (requires OpenCV)."""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    # The outer black square is 300 px / 0.20 m. A 60 px white margin
    # corresponds to 0.04 m on each side of the 0.28 m station panel.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 300)
    panel = np.full((420, 420), 255, dtype=np.uint8)
    panel[60:360, 60:360] = marker
    destination = Path(__file__).resolve().parents[1] / "assets" / "marker_0.png"
    if not cv2.imwrite(str(destination), panel):
        raise RuntimeError(f"Could not write {destination}")
    print(destination)


if __name__ == "__main__":
    main()
