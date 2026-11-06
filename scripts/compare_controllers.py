"""Summarise and plot control logs to compare LQI with the PID baseline.

Record one log per controller with the same config and seed, then compare:
    python -m track_a_mobile_robot.main --headless --seconds 160 --controller lqi --control-log lqi.csv
    python -m track_a_mobile_robot.main --headless --seconds 160 --controller pid --control-log pid.csv
    python -m scripts.compare_controllers lqi.csv pid.csv --plot comparison.png

The metric table is always printed; --plot also writes a PNG figure. CSV logs
stay the raw data, so figures can be redrawn without re-running simulations.
Tracking metrics use only rows where a coarse reference was tracked (APPROACH,
AVOID and REACQUIRE turns); PBVS rows are identical laws in both runs.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

_TEXT_COLUMNS = ("phase", "mode", "reference_kind")

# Categorical slots 1-2 of the reference data-viz palette, in fixed order, so
# a log keeps its colour in every panel. Chrome uses recessive ink and hairlines.
SERIES_COLOURS = ("#2a78d6", "#eb6834")
SURFACE, INK, INK_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def rows_to_columns(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Convert CSV rows into columns; blank numeric cells become NaN."""
    run = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        run[name] = (np.array(values) if name in _TEXT_COLUMNS
                     else np.array([float(value) if value else np.nan for value in values]))
    return run


def load_log(path: Path) -> dict[str, np.ndarray]:
    """Read one complete control log into columns."""
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{path} has no control rows")
    return rows_to_columns(rows)


def _rms(values):
    return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")


def _peak(values):
    return float(np.max(np.abs(values))) if len(values) else float("nan")


def docked_time(run) -> float:
    """First time the navigator reported DONE, or NaN if the run never docked."""
    done = np.flatnonzero(run["phase"] == "DONE")
    return float(run["time"][done[0]]) if len(done) else float("nan")


def tracked_window(run):
    """Return the rows to plot and their time since the first tracked reference.

    Runs acquire the marker at different times, so aligning there makes them
    comparable. Rows stop at docking, when the robot rests at the dock. Returns
    None until a reference has been tracked.
    """
    tracked = np.flatnonzero(run["reference_kind"] != "")
    if not len(tracked):
        return None
    start, docked = run["time"][tracked[0]], docked_time(run)
    end = docked if np.isfinite(docked) else run["time"][-1]
    shown = (run["time"] >= start) & (run["time"] <= end)
    return shown, run["time"][shown] - start


def time_series(run, shown) -> dict[str, np.ndarray]:
    """Return plotted signals in display units; NaN rows leave gaps in lines."""
    # Spot turns hold position, so cross-track error is only meaningful while driving.
    cross = np.where(run["reference_kind"] == "DRIVE", run["cross_error_m"], np.nan)
    return {
        "cross": 100 * cross[shown],
        "heading": np.degrees(run["heading_error_rad"][shown]),
        "v": run["v_mps"][shown],
        "omega": run["omega_rad_s"][shown],
    }


def summarise(run) -> dict[str, float]:
    time, kind = run["time"], run["reference_kind"]
    dt = np.diff(time, append=time[-1])
    drive, turn, tracked = kind == "DRIVE", kind == "TURN", kind != ""
    v, omega = run["v_mps"][tracked], run["omega_rad_s"][tracked]
    saturated = ((np.abs(v) >= run["speed_limit_mps"][tracked] - 1e-6)
                 | (np.abs(omega) >= run["turn_limit_rad_s"][tracked] - 1e-6))
    docked = docked_time(run)
    # Once docked the robot rests against the dock, so later rows are not stops.
    active = time < docked if np.isfinite(docked) else np.ones_like(time, dtype=bool)
    return {
        "docked at (s)": docked,
        "tracking time (s)": float(dt[tracked].sum()),
        "drive RMS along (m)": _rms(run["along_error_m"][drive]),
        "drive RMS cross (m)": _rms(run["cross_error_m"][drive]),
        "drive peak cross (m)": _peak(run["cross_error_m"][drive]),
        "drive RMS heading (deg)": np.degrees(_rms(run["heading_error_rad"][drive])),
        "turn RMS heading (deg)": np.degrees(_rms(run["heading_error_rad"][turn])),
        "turn peak heading (deg)": np.degrees(_peak(run["heading_error_rad"][turn])),
        "effort int(v^2+w^2)dt": float(np.sum((v**2 + omega**2) * dt[tracked])),
        "saturated (%)": 100 * float(np.mean(saturated)) if len(saturated) else float("nan"),
        "LiDAR stops": float(np.sum(run["allowed"][active] == 0)),
    }


# Headline RMS/total metrics drawn as bars (key, short title, display scale);
# the printed table keeps the full set in its original units.
_PLOTTED_METRICS = (
    ("docked at (s)", "Docked at (s)", 1),
    ("drive RMS cross (m)", "Cross-track RMS (cm)", 100),
    ("drive RMS heading (deg)", "Drive heading (°)", 1),
    ("turn RMS heading (deg)", "Turn heading (°)", 1),
    ("effort int(v^2+w^2)dt", "Effort ∫(v²+ω²)dt", 1),
    ("saturated (%)", "Saturated (%)", 1),
)


def style_axis(ax, title, xlabel=None, ylabel=None):
    """Recessive chrome: solid hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def plot_comparison(runs: dict[str, dict], summaries: dict[str, dict], path: Path) -> None:
    """Write a path, error, command and headline-metric figure for each log."""
    import matplotlib
    matplotlib.use("Agg")  # Render to files only; no display needed in WSL/headless.
    import matplotlib.pyplot as plt

    if len(runs) > len(SERIES_COLOURS):
        raise ValueError(f"Plotting supports at most {len(SERIES_COLOURS)} logs; compare them in pairs")
    fig = plt.figure(figsize=(12, 12), facecolor=SURFACE, layout="constrained")
    # Rows pair related panels: path + headline metrics, the two tracking
    # errors, then the two commands. Time-series panels share one time axis.
    grid = fig.add_gridspec(3, 2)
    path_ax, metric_grid = fig.add_subplot(grid[0, 0]), grid[0, 1].subgridspec(2, 3)
    cross_ax = fig.add_subplot(grid[1, 0])
    heading_ax = fig.add_subplot(grid[1, 1], sharex=cross_ax)
    v_ax = fig.add_subplot(grid[2, 0], sharex=cross_ax)
    omega_ax = fig.add_subplot(grid[2, 1], sharex=cross_ax)

    turn_limit = max(float(np.nanmax(run["turn_limit_rad_s"])) for run in runs.values())
    for (name, run), colour in zip(runs.items(), SERIES_COLOURS):
        window = tracked_window(run)
        if window is None:
            raise ValueError(f"{name} never tracked a reference; nothing to compare")
        shown, t = window
        line = dict(color=colour, linewidth=1.2, alpha=.9)

        path_ax.plot(run["true_x_m"], run["true_y_m"], label=name.upper(), **line)
        path_ax.plot(run["true_x_m"][0], run["true_y_m"][0], "o", color=colour, markersize=7,
                     markeredgecolor=SURFACE, markeredgewidth=1.5)
        last = np.flatnonzero(shown)[-1]
        path_ax.plot(run["true_x_m"][last], run["true_y_m"][last], "s", color=colour, markersize=7,
                     markeredgecolor=SURFACE, markeredgewidth=1.5)
        series = time_series(run, shown)
        cross_ax.plot(t, series["cross"], **line)
        heading_ax.plot(t, series["heading"], **line)
        v_ax.plot(t, series["v"], **line)
        omega_ax.plot(t, series["omega"], **line)

    style_axis(path_ax, "Driven path (simulator truth)", "x (m)", "y (m)")
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.plot([], [], "o", color=MUTED, label="start")
    path_ax.plot([], [], "s", color=MUTED, label="docked / end")
    path_ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    style_axis(cross_ax, "Cross-track error while driving", ylabel="error (cm)")
    style_axis(heading_ax, "Heading error (drive and spot turns)", ylabel="error (°)")
    style_axis(v_ax, "Forward velocity command", "time since first reference (s)", "v (m/s)")
    style_axis(omega_ax, "Yaw-rate command", "time since first reference (s)", "ω (rad/s)")
    for limit in (-turn_limit, turn_limit):
        omega_ax.axhline(limit, color=MUTED, linewidth=.8, linestyle=(0, (4, 3)))
    omega_ax.annotate("turn limit", (1, turn_limit), xycoords=("axes fraction", "data"),
                      xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                      color=MUTED, fontsize=8)
    for ax in (cross_ax, heading_ax):
        ax.axhline(0, color=AXIS, linewidth=.8)

    names = list(summaries)
    for index, (key, label, scale) in enumerate(_PLOTTED_METRICS):
        ax = fig.add_subplot(metric_grid[index // 3, index % 3])
        values = [scale * summaries[name][key] for name in names]
        bars = ax.bar(range(len(names)), values, color=SERIES_COLOURS[:len(names)], width=.6)
        ax.bar_label(bars, fmt="%.3g", padding=2, color=INK, fontsize=8)
        style_axis(ax, label)
        ax.title.set_fontsize(8)
        ax.set_xticks(range(len(names)), [name.upper() for name in names])
        ax.grid(False, axis="x")
        ax.set_ylim(0, 1.25 * np.nanmax(values) if np.isfinite(values).any() and np.nanmax(values) > 0 else 1)

    fig.suptitle("Controller comparison: " + " vs ".join(name.upper() for name in names),
                 x=.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def print_summaries(summaries: dict[str, dict]) -> None:
    width = max(len(name) for name in next(iter(summaries.values())))
    print(f"{'':<{width}}" + "".join(f"{name:>14}" for name in summaries))
    for metric in next(iter(summaries.values())):
        print(f"{metric:<{width}}" + "".join(f"{summary[metric]:>14.4g}" for summary in summaries.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare controller CSV logs from track_a_mobile_robot.main.")
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--plot", type=Path, help="Also write a comparison figure to this PNG file.")
    args = parser.parse_args()
    runs = {path.stem: load_log(path) for path in args.logs}
    summaries = {name: summarise(run) for name, run in runs.items()}
    print_summaries(summaries)
    if args.plot is not None:
        plot_comparison(runs, summaries, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
