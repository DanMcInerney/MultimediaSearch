"""An exclusive lock on a file in the home: the daemon's (`daemon.lock`) and the index's creation (`index.lock`)."""

from __future__ import annotations

import os
import time
from pathlib import Path

LOCK_RETRY_S = 0.01  # a busy lock is tried again this often


class HomeLock:
    """An exclusive lock on a file in the home, dropped by the operating system when the process dies.

    The file stays in the home for good: unlinking it leaves it delete-pending on Windows, where the next open is
    refused, and on Linux lets two processes lock two different files of the same name.
    """

    def __init__(self, path: Path):
        self.path = path
        self.file = None

    def acquire(self, wait_s: float) -> bool:
        """Take the lock, trying again for `wait_s` while it is busy. A probe (`is_running`) or the index's creation holds
        it for microseconds and a daemon for its whole life, so a lock still busy after a short wait is a daemon's."""
        end = time.monotonic() + wait_s
        while True:
            error = None
            try:
                file = open(self.path, "a+b")
            except OSError as opening:  # the file is being opened or locked by another process, or the home is unusable
                error = opening
            else:
                try:
                    self._lock(file)
                except OSError:
                    file.close()
                else:
                    self.file = file
                    return True
            if time.monotonic() >= end:
                if error is not None:
                    raise error
                return False
            time.sleep(LOCK_RETRY_S)

    def release(self) -> None:
        if self.file is None:
            return
        try:
            self._unlock(self.file)
        except OSError:
            pass
        self.file.close()
        self.file = None

    @staticmethod
    def _lock(file) -> None:
        file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(file) -> None:
        file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(file, fcntl.LOCK_UN)
