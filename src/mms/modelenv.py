"""A model's env (DESIGN.md §13.3): where it is, whether it was prepared for the pins mms needs, and which Python runs the
model. d3 and Perplexity run in an env of their own that only `mms model prepare` builds, d1 runs in mms's own env, and
H2O's is the owner's. A backend says which by `served`, `in_mms_env`, `family`, `requirements` and `install`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from mms.worker.serve import canonical, installed

MARKER = "mms-env.json"  # in a prepared env: {"family", "requirements", "versions", "prepared"}
NO_CUDA = 3  # what the check below exits with when torch loads and finds no CUDA device
CUDA_CHECK = f"import sys, torch; sys.exit(0 if torch.cuda.is_available() else {NO_CUDA})"
CUDA_CHECK_S = 120.0  # importing torch takes a while the first time


def base(envs: str | None, models_dir: Path) -> Path:
    """Where the envs are: `envs` if the owner set it, else `envs/` beside the model folders."""
    return Path(envs).expanduser() if envs else Path(models_dir).expanduser() / "envs"


def folder(envs: str | None, model_folder: Path, family: str) -> Path:
    """Where a family's env is."""
    return base(envs, Path(model_folder).expanduser().parent) / family


def python(env: Path) -> Path:
    return Path(env) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def mms_python() -> Path:
    """The interpreter mms runs in. The detached daemon runs under pythonw, which has no console for a worker's pipes."""
    exe = Path(sys.executable)
    return exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" else exe


def mms_script() -> Path:
    """The `mms` script beside the interpreter (the project's venv), where it is: `mms` is on no PATH. Not checked to exist."""
    return mms_python().with_name("mms.exe" if os.name == "nt" else "mms")


def marker_problem(env: Path, requirements: tuple[str, ...]) -> str | None:
    """None when `env` was prepared for exactly these requirements (its `mms-env.json` says so); else why not."""
    marker = Path(env) / MARKER
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"{marker} is not there"
    except (OSError, ValueError) as error:
        return f"{marker} cannot be read: {error}"
    if not isinstance(recorded, dict) or list(recorded.get("requirements", ())) != list(requirements):
        return f"{marker} was made for other requirements than this mms needs"
    return None


def clear_marker(env: Path) -> None:
    """Make `env` unprepared: the first step of a prepare, so one that fails leaves it so."""
    (Path(env) / MARKER).unlink(missing_ok=True)


def write_marker(env: Path, family: str, requirements: tuple[str, ...], versions: dict) -> None:
    """Record that `env` holds `versions`, for `requirements`: the last step of a prepare, so an env without the marker
    is not prepared."""
    marker = Path(env) / MARKER
    record = {"family": family, "requirements": list(requirements), "versions": versions,
              "prepared": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    part = marker.with_name(marker.name + ".part")
    part.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(part, marker)


def pinned(requirement: str) -> tuple[str, str | None]:
    """(the package's canonical name, the version a `==` pins or None)."""
    found = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:==\s*([^\s;,]+))?", requirement)
    if found is None:
        raise ValueError(f"not a requirement: {requirement!r}")
    return canonical(found.group(1)), found.group(2)


def mismatch(requirement: str, held: dict) -> str | None:
    """Why the installed versions `held` do not meet one requirement, or None. Only `==` pins are compared (a `>=` just has
    to be installed); a pin without a local label meets torch's `2.11.0+cu128`."""
    name, pin = pinned(requirement)
    have = held.get(name)
    if have is None:
        return f"{name} is not installed"
    if pin and have != pin and ("+" in pin or have.split("+")[0] != pin):
        return f"{name} is {have}, not the {pin} that mms pins"
    return None


def compare(requirements: tuple[str, ...], held: dict) -> tuple[dict, list[str]]:
    """(the version `held` has of each requirement it meets, by name; why each other is not met)."""
    versions, wrong = {}, []
    for requirement in requirements:
        if text := mismatch(requirement, held):
            wrong.append(text)
        else:
            name = pinned(requirement)[0]
            versions[name] = held[name]
    return versions, wrong


def own_versions(backend) -> dict:
    """The versions mms's own env holds of the requirements of a backend that runs in it (`in_mms_env`)."""
    return compare(backend.requirements, installed())[0]


def cuda_problem(backend) -> str | None:
    """Why torch in mms's own env cannot run a model on CUDA, or None when it finds a device. A process of mms's interpreter
    asks (the daemon and the CLI never import torch), which takes seconds."""
    try:
        done = subprocess.run([str(mms_python()), "-c", CUDA_CHECK], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", stdin=subprocess.DEVNULL, timeout=CUDA_CHECK_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"torch could not be asked for a CUDA device: {type(error).__name__}: {error}"
    if done.returncode == 0:
        return None
    if done.returncode == NO_CUDA:
        return ("torch finds no CUDA device: the d1 models run on an NVIDIA GPU with a driver that PyTorch's CUDA 12.8 build "
                f"can use. Install the driver or use a PC that has one; `{backend.install}` installs the CUDA build")
    tail = "\n".join((done.stderr.strip() or done.stdout.strip()).splitlines()[-4:])
    return f"torch cannot be started in mms's own env (exit code {done.returncode}): {tail}"


def problem(backend, model_folder: Path, envs: str | None, *, cuda: bool = False) -> str | None:
    """Why a backend's env cannot be used yet, or None when it can (or it needs none). mms's own env is read from the
    installed metadata, so torch is not imported, unless `cuda` asks too whether torch finds a CUDA device (a process starts,
    so only what is about to download a model asks)."""
    if backend.served:
        return None
    if backend.in_mms_env:
        wrong = compare(backend.requirements, installed())[1]
        if wrong:
            return "; ".join(wrong) + f": run `{backend.install}` in mms's project"
        return cuda_problem(backend) if cuda else None
    return marker_problem(folder(envs, model_folder, backend.family), backend.requirements)


def ready_python(backend, model_folder: Path, envs: str | None, name: str) -> Path:
    """The Python of a prepared env of its own, or the `ValueError` that says to prepare it."""
    env = folder(envs, model_folder, backend.family)
    if (why := marker_problem(env, backend.requirements)) is not None:
        raise ValueError(f"{name}'s env ({env}) is not prepared ({why}): run mms model prepare {Path(model_folder).name}")
    return python(env)
