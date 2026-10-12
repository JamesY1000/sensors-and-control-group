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

- `world/room_with_charger.xml` is the world entry point: robot include,
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

## Project workflow (quick start)

1. Pull the latest changes from `main`.
2. Create a branch for your task, using one of the prefixes below.
3. Develop and test locally using the [setup above](#run-the-simulation), or in
   the team's devcontainer if one is provided.
4. Open a pull request (PR) with a clear description of what changed and why.
5. Get at least one teammate review.
6. Merge only when the relevant checks pass.

This repository does not currently include a devcontainer configuration. The
documented setup uses a local Python virtual environment. For simulation changes,
run the headless physics check above and any other checks relevant to your task.

## Branching and commit conventions

Use one branch per issue or feature, with a descriptive name such as
`docs/team-workflow`.

Suggested branch prefixes:

- `feature/` — new functionality.
- `fix/` — bug fixes.
- `test/` — tests and validation.
- `docs/` — documentation changes.

Commit messages should:

- Be concise and descriptive.
- Start with an action verb.
- Mention the subsystem when relevant.

Example: `Add headless simulation validation`.

## Pull request guidelines

Before opening a PR:

- Rebase your branch onto the latest `main`, or merge the latest `main` into it.
- Confirm your branch builds and runs successfully, and run the relevant checks.
- Update documentation if behavior, dependencies, or setup changed.

Each PR should have:

- A clearly defined, focused scope.
- A clear summary of what changed and why.
- A description of the checks performed and their results.
- At least one teammate review before merging.

## Team rules

- Never push directly to `main`.
- Pull the latest `main` before starting work.
- Use one branch per issue or feature.
- Use descriptive branch names and commit messages.
- Keep pull requests focused and reasonably small.
- Have at least one teammate review each pull request.
- Do not merge code that does not build or fails the relevant checks.
- Merge frequently instead of integrating everything near the deadline.
- Document dependency assumptions and setup procedures.
- Never commit secrets, generated build folders, or large datasets.

## Definition of done

A task is done when:

- [ ] The implementation works and has been reviewed by at least one teammate.
- [ ] Relevant checks pass, and the change does not break other parts of the system.
- [ ] Relevant documentation is updated, including dependency and setup changes.
