# Sensors and Control — Track A

Currently: load the unchanged ROBOTIS **TurtleBot3 Waffle Pi** model
on its default floor in MuJoCo. 

## Run the simulation

From this repository's root, activate the existing environment and launch:

```bash
source .venv/bin/activate
python3 -m track_a_mobile_robot.main
```

For a fresh checkout, first create the environment with `python3 -m venv .venv`,
activate it, and run `python3 -m pip install -r requirements.txt`.
The pinned dependencies require Python 3.12 or newer.

The built-in MuJoCo viewer opens the robot on a checkerboard floor. Use the
mouse to inspect the scene and close the window when finished. With no wheel
commands applied, the robot stays stationary apart from settling under gravity.

To check loading and physics without a window:

```bash
python3 -m track_a_mobile_robot.main --headless --seconds 2
```

The launcher prefers GLFW's X11/XWayland backend when Linux has `DISPLAY`
available, avoiding a native Wayland window-decoration issue on this machine.
It uses MuJoCo's managed viewer, which handles physics stepping and rendering.

## Which file does what?

- `world/scene_turtlebot3_waffle_pi.xml` is the world entry point: robot include,
  mesh directory, lighting, floor, textures and initial viewing settings.
- `robot_model/turtlebot3_waffle/turtlebot3_waffle_pi.xml` is the unchanged
  upstream robot: bodies, mass/inertia, wheel joints, actuators and geometry.
- `robot_model/turtlebot3_waffle/assets/` contains its four unchanged STL meshes.
- `track_a_mobile_robot/main.py` loads the scene and opens the viewer.
- The other algorithm modules are empty placeholders for later work.

The scene's include and mesh-directory paths are adapted to this repository's
folders. Robot geometry, actuator gains and other robot settings are unchanged.
The LiDAR and camera housings are visual geometry; they are not working sensors.

## Model source

The model and default scene are from
[ROBOTIS's MuJoCo collection](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/d8344c0dbe7a00208d0301111523dde65efc174a/robotis_tb3),
commit `d8344c0dbe7a00208d0301111523dde65efc174a`. The upstream licence is retained
in `robot_model/turtlebot3_waffle/LICENSE`.