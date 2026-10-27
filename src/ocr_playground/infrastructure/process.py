"""Subprocesses, behind one small method.

Every external tool (poppler, ImageMagick, LiteParse) is called through here, so a test replaces one
object instead of patching `subprocess.run` in three modules. A failure keeps the tool's own stderr,
because "lit failed: Permission denied" is the message that actually helps.
"""
from __future__ import annotations

import subprocess
from collections.abc import Sequence


class CommandFailed(RuntimeError):
    """The tool ran and exited non-zero. A generic failure: the caller reports it as a 500."""


class CommandTimedOut(RuntimeError):
    """The tool did not finish in time. The HTTP layer turns this into a 504."""


class ProcessRunner:
    def __init__(self, timeout: float = 120) -> None:
        self.timeout = timeout

    def run(self, command: Sequence[str], timeout: float | None = None) -> str:
        try:
            result = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=self.timeout if timeout is None else timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise CommandTimedOut(f"{command[0]} timed out after {error.timeout}s") from error
        if result.returncode != 0:
            raise CommandFailed(f"{command[0]} failed: {result.stderr.strip()[:400]}")
        return result.stdout
