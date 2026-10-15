# Create 3 dock visual assets

The dock meshes come from iRobot's [`create3_sim`, `jazzy` branch](https://github.com/iRobotEducation/create3_sim/tree/jazzy), retrieved on 2026-10-15:

- [Original `visual.dae`](https://github.com/iRobotEducation/create3_sim/blob/jazzy/irobot_create_common/irobot_create_description/meshes/dock/visual.dae)
- [Original dock URDF/Xacro](https://github.com/iRobotEducation/create3_sim/blob/jazzy/irobot_create_common/irobot_create_description/urdf/dock/standard_dock.urdf.xacro)
- [Upstream BSD 3-Clause license](https://github.com/iRobotEducation/create3_sim/blob/jazzy/LICENSE), reproduced in [LICENSE](LICENSE)

The downloaded `visual.dae` has SHA-256:

```text
80829a8477dcc4e99ba94e17d67d602a7ab4bfa6553e6d8964b68a47dc321d4b
```

`dock_black.obj` and `dock_white.obj` are the source geometry split by material. The conversion preserves vertex coordinates, triangles, surface normals, metre units and Z-up orientation. Identical vertex/normal entries were deduplicated and redundant decimal zeroes removed; there is no geometric simplification or scale change. The two parts contain 55,820 and 988 triangles respectively. OBJ retains the original surface normals, which an STL conversion would discard.

The original visual bounds are approximately 0.14675 m along X, 0.13173 m along Y and 0.10206 m high. `world/charging_station.xml` retains the upstream 0.0046 m visual Z offset and simple housing collision box (centre `[-0.045, 0, 0.05]`, full dimensions `[0.04, 0.13, 0.1]`). Mesh geoms are visual only, so their convex hull does not obstruct the approach area. The broad front platform is visual geometry; wheel/ramp and charging-contact physics are not modelled.

The materials are adapted for MuJoCo lighting: source black becomes dark grey (`0.12 0.13 0.15`) and source white becomes light grey (`0.85 0.85 0.85`). This keeps the housing details visible without an emissive material.

The added marker panel, backing and post are project geometry, separate from the iRobot asset. The fixed `charging_station` body uses local +X out of the dock, +Z up, and a ground-level origin. A 0.28 m square panel sits above the dock at local `[-0.051, 0, 0.26]`, with a 0.20 m ArUco marker and 0.04 m white margin on each side. The visible marker face and `aruco_marker_0` site are exactly at `[-0.05, 0, 0.26]`. The site's axes are +Y, +Z and +X in the station frame: right in a front view, up and out of the panel respectively.

The station is a standalone MJCF component with its own mesh and texture directories. The room imports it through MuJoCo's `<asset><model .../></asset>` and `<attach .../>` mechanism, keeping its paths independent of the TurtleBot mesh directory and the shell's current directory. Moving the `charging_station` body moves the dock, panel and ground-truth marker frame together. This preserves the Create 3 dock's natural dimensions; it does not imply physical charging compatibility with a TurtleBot3 Waffle Pi.
