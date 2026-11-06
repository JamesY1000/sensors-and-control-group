"""Plot control logs live while simulations write them.

Run the viewer in a second terminal; it can start before the log exists:
    python -m track_a_mobile_robot.main --controller lqi --control-log lqi.csv
    python -m scripts.live_plot lqi.csv

Pass two logs to overlay two runs. The viewer is a separate process so drawing
never stalls physics or control. It only reads the CSV, so it also works with
--headless runs and can be closed or restarted at any time. Metrics are printed
when a run docks, because RMS values are misleading part-way through a run.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from scripts.compare_controllers import (
    AXIS, INK, MUTED, SERIES_COLOURS, SURFACE, docked_time, print_summaries, rows_to_columns,
    style_axis, summarise, time_series, tracked_window,
)


class LogFollower:
    """Read rows appended to a control log that may still be being written."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.restarted = False
        self._offset = 0
        self._header = None

    def new_rows(self) -> list[dict[str, str]]:
        """Return complete rows added since the last call; [] if there are none."""
        self.restarted = False
        try:
            with open(self.path, "rb") as file:
                # A new run truncates the log, leaving it shorter than our position.
                if os.fstat(file.fileno()).st_size < self._offset:
                    self._offset, self._header, self.restarted = 0, None, True
                file.seek(self._offset)
                data = file.read()
        except FileNotFoundError:
            return []
        # Stop at the last newline: the writer may be part-way through a row,
        # which is read again once it is complete.
        complete = data[:data.rfind(b"\n") + 1]
        self._offset += len(complete)
        lines = complete.decode().splitlines()
        if self._header is None and lines:
            self._header, lines = next(csv.reader(lines[:1])), lines[1:]
        return [dict(zip(self._header, row)) for row in csv.reader(lines)]


class LiveRun:
    """Accumulate one followed log as plottable columns."""

    def __init__(self, path: Path) -> None:
        self.name = path.stem
        self.columns = None
        self.reported = False
        self._follower = LogFollower(path)

    def poll(self) -> None:
        rows = self._follower.new_rows()
        if self._follower.restarted:
            self.columns, self.reported = None, False
        if not rows:
            return
        new = rows_to_columns(rows)
        columns = new if self.columns is None else {
            name: np.concatenate((self.columns[name], new[name])) for name in new}
        # A viewer reset restarts simulation time in the same log; keep the latest attempt.
        restarts = np.flatnonzero(np.diff(columns["time"]) < 0)
        if len(restarts):
            columns = {name: values[restarts[-1] + 1:] for name, values in columns.items()}
            self.reported = False
        self.columns = columns


class LiveFigure:
    """Own the figure and update its existing lines as runs grow."""

    def __init__(self, runs: list[LiveRun], window_s: float) -> None:
        self.runs, self.window_s = runs, window_s
        self.fig = plt.figure(figsize=(14, 7.5), facecolor=SURFACE)
        # Fixed margins: a layout engine would re-solve on every redraw.
        grid = self.fig.add_gridspec(2, 3, left=.05, right=.985, top=.88, bottom=.08, wspace=.22, hspace=.28)
        self.path_ax = self.fig.add_subplot(grid[:, 0])
        cross_ax = self.fig.add_subplot(grid[0, 1])
        self.axes = {
            "cross": cross_ax,
            "heading": self.fig.add_subplot(grid[0, 2], sharex=cross_ax),
            "v": self.fig.add_subplot(grid[1, 1], sharex=cross_ax),
            "omega": self.fig.add_subplot(grid[1, 2], sharex=cross_ax),
        }
        style_axis(self.path_ax, "Driven path (simulator truth)", "x (m)", "y (m)")
        self.path_ax.set_aspect("equal", adjustable="datalim")
        style_axis(self.axes["cross"], "Cross-track error while driving", ylabel="error (cm)")
        style_axis(self.axes["heading"], "Heading error (drive and spot turns)", ylabel="error (°)")
        style_axis(self.axes["v"], "Forward velocity command", "time since first reference (s)", "v (m/s)")
        style_axis(self.axes["omega"], "Yaw-rate command", "time since first reference (s)", "ω (rad/s)")
        for name in ("cross", "heading"):
            self.axes[name].axhline(0, color=AXIS, linewidth=.8)

        self.lines = []
        for run, colour in zip(runs, SERIES_COLOURS):
            style = dict(color=colour, linewidth=1.2, alpha=.9)
            marker = dict(color=colour, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
            lines = {name: ax.plot([], [], **style)[0] for name, ax in self.axes.items()}
            lines["path"] = self.path_ax.plot([], [], label=run.name.upper(), **style)[0]
            lines["start"] = self.path_ax.plot([], [], "o", **marker)[0]
            lines["current"] = self.path_ax.plot([], [], "s", **marker)[0]
            self.lines.append(lines)
        self.path_ax.plot([], [], "o", color=MUTED, label="start")
        self.path_ax.plot([], [], "s", color=MUTED, label="current / docked")
        self.path_ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="best")
        self.status = self.fig.suptitle("", x=.01, ha="left", color=INK, fontsize=12, fontweight="bold")
        self._turn_limit_drawn = False

    def _draw_turn_limit(self, limit: float) -> None:
        ax = self.axes["omega"]
        for value in (-limit, limit):
            ax.axhline(value, color=MUTED, linewidth=.8, linestyle=(0, (4, 3)))
        ax.annotate("turn limit", (1, limit), xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="right", va="bottom", color=MUTED, fontsize=8)
        self._turn_limit_drawn = True

    def _update_run(self, run: LiveRun, lines: dict) -> tuple[str, float]:
        """Update one run's lines; return its status text and latest plotted time."""
        run.poll()
        name, columns = run.name.upper(), run.columns
        if columns is None:
            for line in lines.values():
                line.set_data([], [])
            return f"{name}: waiting for log", 0.
        x, y = columns["true_x_m"], columns["true_y_m"]
        lines["path"].set_data(x, y)
        lines["start"].set_data(x[:1], y[:1])
        lines["current"].set_data(x[-1:], y[-1:])
        status = f"{name}: {columns['time'][-1]:.1f} s {columns['phase'][-1]}"
        window = tracked_window(columns)
        if window is None:
            for key in self.axes:
                lines[key].set_data([], [])
            return status, 0.
        shown, t = window
        for key, values in time_series(columns, shown).items():
            lines[key].set_data(t, values)
        if not self._turn_limit_drawn:
            self._draw_turn_limit(float(np.nanmax(columns["turn_limit_rad_s"])))
        if np.isfinite(docked_time(columns)) and not run.reported:
            run.reported = True
            print_summaries({run.name: summarise(columns)})
        return status, float(t[-1])

    def update(self, _frame=None) -> None:
        results = [self._update_run(run, lines) for run, lines in zip(self.runs, self.lines)]
        self.status.set_text("   |   ".join(status for status, _ in results))
        latest = max(time for _, time in results)
        # The time axes are shared, so one call scrolls all four panels.
        if self.window_s > 0:
            self.axes["cross"].set_xlim(max(0., latest - self.window_s), max(latest, self.window_s))
        else:
            self.axes["cross"].set_xlim(0., max(latest, 1.))
        for ax in self.axes.values():
            ax.relim()
            ax.autoscale_view(scalex=False)
        self.path_ax.relim()
        self.path_ax.autoscale_view()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot control logs live as simulations write them.")
    parser.add_argument("logs", type=Path, nargs="+", help="One or two CSV logs from --control-log.")
    parser.add_argument("--window", type=float, default=60., help="Seconds shown on time plots; 0 shows the whole run.")
    # A redraw takes about 250 ms of CPU, which software-rendered simulations compete for.
    parser.add_argument("--refresh-ms", type=int, default=1000, help="Redraw period; raise it if the simulation slows down.")
    args = parser.parse_args()
    if len(args.logs) > len(SERIES_COLOURS):
        parser.error(f"at most {len(SERIES_COLOURS)} logs can be followed at once")
    if args.window < 0 or args.refresh_ms <= 0:
        parser.error("--window cannot be negative and --refresh-ms must be positive")
    # Matplotlib falls back to Agg, which cannot open a window, without a GUI toolkit or display.
    if plt.get_backend().lower() == "agg":
        parser.error("no interactive matplotlib backend; install requirements.txt and check a display is available")
    figure = LiveFigure([LiveRun(path) for path in args.logs], args.window)
    # Keep this reference: matplotlib stops an animation that is garbage collected.
    animation = FuncAnimation(figure.fig, figure.update, interval=args.refresh_ms, cache_frame_data=False)
    plt.show()
    del animation


if __name__ == "__main__":
    main()
