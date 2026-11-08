"""Every child of the daemon dies with it (DESIGN.md §5). ffmpeg ignores a dead parent and keeps writing into its
launch folder, and a camera or microphone it holds blocks the next one.

On Windows the daemon puts itself in a Job Object that kills its members when the last handle to it closes, which is
when the daemon dies, however it dies; every child inherits the job. On Linux each child asks for SIGKILL when its
parent dies (PR_SET_PDEATHSIG), which reaches direct children only. The daemon starts every child with `popen`.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys

PR_SET_PDEATHSIG = 1
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

_job = None  # the Job Object's handle, held until the process ends


def contain() -> None:
    """Windows: put this process in a kill-on-close Job Object, so its children die when it does. Raises OSError
    if the system refuses. Elsewhere `popen` does the work, and this does nothing."""
    global _job
    if os.name != "nt" or _job is not None:
        return
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = Extended()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (not kernel.SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                           ctypes.sizeof(info))
            or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    _job = job


def popen(args, **kwargs) -> subprocess.Popen:
    """`subprocess.Popen` for a child of the daemon. On Windows: with no console window (the detached daemon runs under
    pythonw, which has no console, so a console program it starts would open a window of its own) and in its own
    process group (a console's Ctrl+C is the daemon's, not the child's). On Linux: asking to be killed when the daemon
    dies."""
    if os.name == "nt":
        kwargs.setdefault("creationflags", subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
    elif sys.platform.startswith("linux"):
        kwargs.setdefault("preexec_fn", _die_with_parent(os.getpid(), ctypes.CDLL(None, use_errno=True)))
    return subprocess.Popen(args, **kwargs)


def run(args, **kwargs) -> subprocess.CompletedProcess:
    """`subprocess.run` for a short command the daemon waits on: on Windows with no console window, as `popen`."""
    if os.name == "nt":
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
    return subprocess.run(args, **kwargs)


def _die_with_parent(parent: int, libc):
    def arrange() -> None:
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        if os.getppid() != parent:  # the parent died before the call took effect
            os._exit(1)
    return arrange
