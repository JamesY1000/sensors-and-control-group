"""Robot I/O contract shared by the MuJoCo and rosbridge adapters.

The HAL owns sensor acquisition, timestamps, actuation and resource cleanup.
Runners own scheduling; autonomy modules consume readings rather than device state.
Import each adapter explicitly so hardware runs never load MuJoCo or its viewer.
"""

from typing import TYPE_CHECKING, Protocol

from ..sensing import CameraReading, LidarReading, WheelEncoderReading

if TYPE_CHECKING:
    from ..control import VelocityCommand


class RobotBackend(Protocol):
    """Nonblocking acquisition with cached camera/scan readings on one run clock."""

    encoders: WheelEncoderReading | None
    camera: CameraReading | None
    lidar: LidarReading | None

    def start(self) -> None:
        """Initialise the device and leave motion stopped."""
        ...

    def now(self) -> float:
        """Current run time in seconds, on the sensor timestamp clock."""
        ...

    def read(self) -> WheelEncoderReading | None:
        """Refresh cached readings and return a new encoder sample, if available."""
        ...

    def health(self) -> str | None:
        """Return a stop reason while required sensor streams are unavailable."""
        ...

    def send_command(self, command: "VelocityCommand") -> None:
        """Apply a forward/angular velocity command within the drive limits."""
        ...

    def stop(self) -> None:
        """Command zero motion."""
        ...

    def close(self) -> None:
        """Stop motion and release device resources."""
        ...
