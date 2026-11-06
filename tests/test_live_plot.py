"""Live log following and figure updates, without opening a window."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.compare_controllers import load_log, time_series, tracked_window
from scripts.live_plot import LiveFigure, LiveRun, LogFollower
from track_a_mobile_robot.main import CONTROL_LOG_FIELDS

HEADER = ",".join(CONTROL_LOG_FIELDS) + "\r\n"


def row(time, phase="APPROACH", kind="DRIVE", error=.01):
    errors = f"{error},{error},{error}" if kind else ",,"
    return (f"{time:.3f},{phase},{phase},{kind},{errors},.1,.2,.3,.6,1,{time},{-time},0\r\n")


class LogFileTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "lqi.csv"

    def write(self, text, mode="a"):
        with open(self.path, mode, newline="") as file:
            file.write(text)


class LiveLogTests(LogFileTestCase):
    def test_missing_log_then_only_new_rows_are_returned(self):
        follower = LogFollower(self.path)
        self.assertEqual(follower.new_rows(), [])
        self.write(HEADER + row(0) + row(.02))
        self.assertEqual([r["time"] for r in follower.new_rows()], ["0.000", "0.020"])
        self.assertEqual(follower.new_rows(), [])
        self.write(row(.04))
        self.assertEqual([r["time"] for r in follower.new_rows()], ["0.040"])

    def test_half_written_row_waits_for_its_newline(self):
        follower = LogFollower(self.path)
        complete = row(.02)
        self.write(HEADER + row(0) + complete[:10])
        self.assertEqual(len(follower.new_rows()), 1)
        self.write(complete[10:])
        rows = follower.new_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["time"], "0.020")
        self.assertEqual(rows[0]["true_yaw_rad"], "0")

    def test_truncated_log_starts_a_new_run(self):
        run = LiveRun(self.path)
        self.write(HEADER + "".join(row(i * .02) for i in range(5)))
        run.poll()
        self.assertEqual(len(run.columns["time"]), 5)
        self.write(HEADER + row(0), mode="w")
        run.poll()
        self.assertEqual(len(run.columns["time"]), 1)

    def test_viewer_reset_keeps_only_the_latest_attempt(self):
        run = LiveRun(self.path)
        self.write(HEADER + row(0) + row(.02) + row(.04))
        run.poll()
        self.write(row(0) + row(.02))
        run.poll()
        np.testing.assert_allclose(run.columns["time"], [0, .02])

    def test_incremental_columns_match_a_complete_load(self):
        run = LiveRun(self.path)
        self.write(HEADER + row(0, kind="") + row(.02, kind=""))
        run.poll()
        self.write(row(.04, kind="TURN") + row(.06) + row(.08, phase="DONE", kind=""))
        run.poll()
        complete = load_log(self.path)
        for name, values in complete.items():
            np.testing.assert_array_equal(run.columns[name], values)


class LiveFigureTests(LogFileTestCase):
    def figure(self, window_s=60.):
        run = LiveRun(self.path)
        figure = LiveFigure([run], window_s)
        self.addCleanup(plt.close, figure.fig)
        return run, figure

    def test_waits_for_log_then_for_first_reference(self):
        run, figure = self.figure()
        figure.update()
        self.assertIn("waiting for log", figure.status.get_text())
        self.write(HEADER + row(0, phase="SCAN", kind=""))
        figure.update()
        self.assertIn("LQI: 0.0 s SCAN", figure.status.get_text())
        self.assertEqual(len(figure.lines[0]["path"].get_xdata()), 1)
        self.assertEqual(len(figure.lines[0]["cross"].get_xdata()), 0)

    def test_lines_match_the_png_series_and_time_axis_scrolls(self):
        run, figure = self.figure(window_s=1.)
        self.write(HEADER + row(1, phase="SCAN", kind="") + "".join(row(2 + i * .1) for i in range(30)))
        figure.update()
        shown, t = tracked_window(run.columns)
        np.testing.assert_array_equal(figure.lines[0]["heading"].get_ydata(),
                                      time_series(run.columns, shown)["heading"])
        self.assertAlmostEqual(t[0], 0)
        np.testing.assert_allclose(figure.axes["v"].get_xlim(), (t[-1] - 1, t[-1]))

    def test_summary_is_printed_once_when_docked(self):
        run, figure = self.figure()
        self.write(HEADER + row(0) + row(.02, phase="DONE", kind=""))
        with patch("scripts.live_plot.print_summaries") as report:
            figure.update()
            self.write(row(.04, phase="DONE", kind=""))
            figure.update()
        report.assert_called_once()
        self.assertEqual(len(run.columns["time"]), 3)


if __name__ == "__main__":
    unittest.main()
