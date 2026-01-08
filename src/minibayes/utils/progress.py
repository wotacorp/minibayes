"""Progress bar utilities for MCMC sampling."""

import sys
import time
from typing import TextIO


class ProgressBar:
    """
    Minimal progress bar for terminal output.

    Parameters
    ----------
    total : int
        Total number of iterations.
    desc : str
        Description text (e.g., "Chain 1/4 [Warmup]").
    width : int
        Width of the progress bar in characters.
    file : TextIO
        Output stream (default: sys.stderr).
    disable : bool
        If True, do nothing (useful for conditional progress).

    Examples
    --------
    >>> with ProgressBar(100, desc="Sampling") as pbar:
    ...     for i in range(100):
    ...         # do work
    ...         pbar.update()
    """

    def __init__(
        self,
        total: int,
        desc: str = "",
        width: int = 20,
        file: TextIO | None = None,
        disable: bool = False,
    ) -> None:
        self._total: int = total
        self._desc: str = desc
        self._width: int = width
        self._file: TextIO = file if file is not None else sys.stderr
        self._disable: bool = disable
        self._current: int = 0
        self._start_time: float = 0.0
        self._last_update_time: float = 0.0

    def __enter__(self) -> "ProgressBar":
        """Start the progress bar."""
        self._start_time = time.perf_counter()
        self._last_update_time = self._start_time
        self._current = 0
        if not self._disable:
            self._render()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Finish the progress bar with newline."""
        if not self._disable:
            self._file.write("\n")
            self._file.flush()

    def update(self, n: int = 1) -> None:
        """
        Update progress by n steps.

        Parameters
        ----------
        n : int
            Number of steps completed (default: 1).
        """
        if self._disable:
            return
        self._current += n
        # Throttle updates to every 0.1s for performance
        now: float = time.perf_counter()
        if now - self._last_update_time >= 0.1 or self._current >= self._total:
            self._render()
            self._last_update_time = now

    def _render(self) -> None:
        """Render the progress bar to the output stream."""
        elapsed: float = time.perf_counter() - self._start_time

        # Calculate percentage and bar
        pct: float = self._current / self._total if self._total > 0 else 0.0
        filled: int = int(self._width * pct)
        if self._current >= self._total:
            bar: str = "=" * self._width  # Full bar at 100%
        else:
            bar = "=" * filled + ">" + " " * (self._width - filled - 1)

        # Calculate iterations per second
        rate: float = self._current / elapsed if elapsed > 0 else 0.0

        # Format: "Desc [=====>    ] 45% (450/1000) 12.3 it/s"
        line: str = f"\r{self._desc} [{bar}] {int(pct * 100):3d}% ({self._current}/{self._total}) {rate:.1f} it/s"

        self._file.write(line)
        self._file.flush()
