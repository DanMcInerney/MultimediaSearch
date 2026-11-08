"""`mms model` (DESIGN.md §13.3): list the models in a folder, switch the configured one, and build a model's env. A model mms
knows that is not in its folder is downloaded first (`fetch.py`).

An env is built only here, by the owner's command, never on first use: `prepare` runs uv (its cache beside the envs, so
the big downloads stay on the models' drive), asks the new env's own Python which versions it holds, and only when they
match the pins writes `mms-env.json`. d1 runs in mms's own env, so its prepare only checks that env; H2O is a server the
owner starts, so its prepare prints the commands and installs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from mms import client, decider, fetch, modelenv, transport
from mms import config as configuration

READ_ORDER = ("video", "image", "audio")
SOURCES_THAT_HEAR = ("microphone", "camera")
TAIL_LINES = 12
HEALTH_S = 1.0


class PrepareError(Exception):
    """A prepare step that failed, with why; `mms model prepare` exits 1 on it."""


def main(home: Path, argv: list[str]) -> int:
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["list", *argv]  # `mms model` and `mms model --dir D` list
    args = parser().parse_args(argv)
    try:
        if args.action == "list":
            return list_models(home, args.dir)
        if args.action == "use":
            return use(home, args.target, args.dir, args.url)
        return prepare(home, args.target, args.dir)
    except (PrepareError, fetch.FetchError) as error:
        print(f"mms: {error}", file=sys.stderr)
        return 1


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="mms model", description="List the decision models, switch the one mms uses, or "
                                  "build a model's env. A NAME is a model mms knows (mms model list shows them; one not in "
                                  "DIR is downloaded from Hugging Face) or a folder in DIR (default: the folder holding "
                                  "the configured model, else <home>/models); a FOLDER is a path.")
    commands = top.add_subparsers(dest="action", required=True, metavar="command")
    sub = commands.add_parser("list", help="each model folder, what reads it, and its env (the default)")
    sub.add_argument("--dir", help="the folder of model folders (default: the one holding the configured model, else "
                     "<home>/models)")
    sub = commands.add_parser("use", help="make a model the one mms uses: download it if it is not there, prepare its env, "
                              "and write it to mms.toml; the running daemon follows")
    sub.add_argument("target", metavar="NAME|FOLDER")
    sub.add_argument("--url", help="where the owner serves the model, for a served one (default: the model's own)")
    sub.add_argument("--dir", help="the folder a NAME is in")
    sub = commands.add_parser("prepare", help="download the model if it is not there, then build its env with uv, check "
                              "d1's, or print H2O's commands")
    sub.add_argument("target", metavar="NAME|FOLDER")
    sub.add_argument("--dir", help="the folder a NAME is in")
    return top


# what a folder is

def same(a: str | Path, b: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(Path(a).expanduser())) == os.path.normcase(os.path.abspath(Path(b).expanduser()))


def configured_folder(config) -> Path | None:
    """The folder of the configured model; None for none, or a `python:` file."""
    model = config.decider_model
    return Path(model).expanduser() if model and not model.startswith("python:") else None


def models_dir(config, given: str | None) -> Path:
    """DIR: `--dir`, else the folder holding the configured model, else `<home>/models`. It need not exist."""
    if given:
        return Path(given).expanduser()
    held = configured_folder(config)
    return held.parent if held else config.home / "models"


def locate(config, target: str, given: str | None) -> tuple:
    """(folder, backend, about, missing) for a NAME or a FOLDER. A NAME mms knows (in any case) is `DIR/<its name>`, missing
    when that is not the whole model; any other NAME must be a folder of DIR; a FOLDER is a path."""
    if Path(target).name != target or target in (".", ".."):
        folder = Path(target).expanduser()
        if not folder.is_dir():
            raise ValueError(f"there is no model folder at {folder}")
        backend, about = decider.claimed(folder)
        return folder, backend, about, False
    root = models_dir(config, given)
    if (known := decider.known(target)) is not None:
        backend, about = known
        folder = root / about.name
        return folder, backend, about, not fetch.ready(about, folder)
    folder = root / target
    if not folder.is_dir():
        here = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        holds = f" (it holds {', '.join(here)})" if here else ""
        raise ValueError(f"there is no model folder {target!r} in {root}{holds}; "
                         "`mms model list` shows the models mms can download")
    backend, about = decider.claimed(folder)
    return folder, backend, about, False


def keep_free(config, folder: Path) -> int:
    """The bytes a download into `folder` must leave free: mms's floor, `[memory] free_gb`, when the home is on that drive
    (capture pauses below it there); else none."""
    return int(config.free_gb * 1e9) if fetch.same_drive(folder, config.home) else 0


def reads_of(about) -> str:
    return ", ".join(kind for kind in READ_ORDER if kind in about.reads)


def hears_instead(about) -> str:
    """What the owner of a model that hears nothing can do about it; empty for a model that hears (or when none does)."""
    ears = [one.name for one in decider.known_abouts() if "audio" in one.reads]
    return f"mms model use {' or '.join(ears)} gives one that hears" if ears and "audio" not in about.reads else ""


# list

def list_models(home: Path, given: str | None) -> int:
    config = configuration.load(home)
    root = models_dir(config, given)
    if not root.is_dir() and (given or configured_folder(config)):
        raise ValueError(f"{root} is not a folder")
    envs = modelenv.base(config.decider_envs, root)  # not a model: not listed
    folders = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()) if root.is_dir() else []
    rows, claimed = [], set()
    for one in folders:
        if same(one, envs):
            continue
        star = "*" if config.decider_model and same(one, config.decider_model) else " "
        found = decider.claim(one)
        if found is None:
            rows.append((f"{star} {one.name}", "no backend reads it", "", ""))
            continue
        backend, about = found
        claimed.add(about.name)
        if (why := fetch.unfinished(one)) is not None:
            state = f"not all there ({why}): mms model use {about.name} finishes it"
        elif backend.served:
            url = config.decider_url if star == "*" and config.decider_url else backend.default_url
            state = f"served at {url}, {health(url)}"
        else:
            problem = modelenv.problem(backend, one, config.decider_envs)
            ready = "mms's own env" if backend.in_mms_env else "ready"
            state = ready if problem is None else f"not prepared: {problem}"
        rows.append((f"{star} {one.name}", about.name, reads_of(about), state))
    for about in decider.known_abouts():
        if about.repo and about.name not in claimed:
            rows.append((f"  {about.name}", about.name, reads_of(about),
                         f"not downloaded: {about.repo_gb:g} GB from huggingface.co/{about.repo}"))
    if not rows:
        print(f"no folders in {root}")
    table(rows)
    return 0


def health(url: str) -> str:
    """Whether the server at `url` answers `/health`, in words."""
    status = transport.Served(url, name="the server").probe(HEALTH_S)
    if status is None:
        return "does not answer"
    return {200: "answers", 503: "is warming"}.get(status, f"/health gives {status}")


def table(rows: list[tuple]) -> None:
    if not rows:
        return
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


# use

def switch(home: Path, target: str, given: str | None, url: str | None) -> tuple:
    """Make a model the configured one: download it if it is not there, build its env (d3, Perplexity) or print its server's
    commands (H2O), and write `[decider]`. Returns the folder, the model's About and whether anything changed."""
    config = configuration.load(home)
    folder, backend, about, missing = locate(config, target, given)
    if url is not None:
        if not backend.served:
            raise ValueError(f"{about.name} is not served, so --url has no use: leave it out")
        configuration.loopback(url)
    # torch is asked for a CUDA device only when a download is about to start: a model that is there is switched to as it is
    if backend.in_mms_env and (problem := modelenv.problem(backend, folder, config.decider_envs, cuda=missing)) is not None:
        raise ValueError(f"{about.name} runs in mms's own env, which is not ready: {problem}")  # before any download
    if missing:
        fetch.download(about, folder, keep_free(config, folder))
    built = (not backend.served and not backend.in_mms_env
             and modelenv.problem(backend, folder, config.decider_envs) is not None)
    if built:
        build(config, folder, backend)
    chosen = (url or backend.default_url) if backend.served else None
    if chosen and health(chosen) == "does not answer":
        server_commands(about, backend, folder)
    file = Path(home) / configuration.FILE
    before = file.read_bytes() if file.exists() else None
    Path(home).mkdir(parents=True, exist_ok=True)
    configuration.set_decider(home, model=Path(os.path.abspath(folder)).as_posix(), url=chosen)
    print(f"model: {about.name} ({folder})" + (f", served at {chosen}" if chosen else ""))
    if note := hears_instead(about):
        print(f"{about.name} hears nothing: {note}")
    if "audio" not in about.reads:
        ears = {kind: [n for n, s in config.sources.items() if s.kind == kind] for kind in SOURCES_THAT_HEAR}
        heard = [f"{kind} {', '.join(names)}" for kind, names in ears.items() if names]
        if heard:
            print(f"warning: {about.name} does not hear sound, so ask will not hear what {'; '.join(heard)} record",
                  file=sys.stderr)
    return folder, about, missing or built or file.read_bytes() != before


def use(home: Path, target: str, given: str | None, url: str | None) -> int:
    switch(home, target, given, url)
    if client.is_running(home):
        loop_s = configuration.load(home).loop_s
        print(f"the daemon is running and follows mms.toml: within about {loop_s:g} s it takes this model, "
              "unloading the old one between two windows; the next window loads the new one")
    else:
        print("no daemon is running; it takes this model when it starts")
    return 0


# prepare

def prepare(home: Path, target: str, given: str | None) -> int:
    config = configuration.load(home)
    folder, backend, about, missing = locate(config, target, given)
    if backend.in_mms_env:
        print(f"{about.name} runs in mms's own env, so nothing is installed; checking that env")
        if (problem := modelenv.problem(backend, folder, config.decider_envs, cuda=missing)) is not None:
            raise PrepareError(problem)  # before any download
    if missing:
        fetch.download(about, folder, keep_free(config, folder))
    if backend.served:
        server_commands(about, backend, folder)
        print(f"then: mms model use {folder.name}" + (f" --url URL (default {backend.default_url})" if backend.default_url else ""))
    elif backend.in_mms_env:
        print("ready: " + ", ".join(f"{name} {version}" for name, version in modelenv.own_versions(backend).items()))
    else:
        build(config, folder, backend)
        print(f"ready: mms model use {folder.name}")
    return 0


def server_commands(about, backend, folder: Path) -> None:
    """What the owner runs for a model that is a server: mms installs nothing and starts nothing."""
    print(f"{about.name} is a server you run, so mms installs nothing.")
    if backend.install:
        print("in WSL Ubuntu, once:")
        print(backend.install)
    print("then, to serve it:" if backend.install else "to serve it:")
    print(backend.start.format(folder=wsl(folder)))


def wsl(folder: Path) -> str:
    """The folder as WSL sees it, quoted for its shell: `C:\\models\\x` is `/mnt/c/models/x`."""
    path = Path(os.path.abspath(folder)).as_posix()
    if os.name == "nt":
        found = re.fullmatch(r"([A-Za-z]):(/.*)", path)
        if found is None:
            raise PrepareError(f"{folder} is not on a lettered drive, so WSL cannot see it under /mnt")
        path = f"/mnt/{found.group(1).lower()}{found.group(2)}"
    return shlex.quote(path)


def uv_command() -> list[str] | None:
    """How to run uv: the one on PATH, or None."""
    uv = shutil.which("uv")
    return [uv] if uv else None


def build(config, folder: Path, backend) -> None:
    uv = uv_command()
    if uv is None:
        raise PrepareError("uv is not on PATH: install it (https://docs.astral.sh/uv/) and run this again")
    env = modelenv.folder(config.decider_envs, folder, backend.family)
    cache = env.parent / ".uv-cache"
    running = {**os.environ, "UV_CACHE_DIR": str(cache)}
    python = modelenv.python(env)
    modelenv.clear_marker(env)  # until this ends well the env is not prepared
    print(f"{backend.family} env: {env} (uv cache {cache})")
    if python.exists():
        print(f"1/5 the env has a Python already: {python}")
    else:
        print("1/5 creating the env")
        run("uv venv", [*uv, "venv", "--python", "3.12", str(env)], running)
    print(f"2/5 installing {len(backend.requirements)} requirements (this downloads gigabytes the first time)")
    install = [*uv, "pip", "install", "--python", str(python)]
    if backend.torch_backend:
        install += ["--torch-backend", backend.torch_backend]
    run("uv pip install", [*install, *backend.requirements], running)
    print("3/5 reading the versions the env holds")
    out = run("the version check", [str(python), "-I", str(transport.SERVE), "--check-env"], running)
    try:
        held = json.loads(out)
    except ValueError:
        raise PrepareError(f"the env's Python did not print JSON for --check-env: {out[-300:]!r}") from None
    print("4/5 comparing them with the pins")
    versions, wrong = modelenv.compare(backend.requirements, held)
    if wrong:
        raise PrepareError("the env does not hold what mms pins: " + "; ".join(wrong))
    print(f"5/5 writing {env / modelenv.MARKER}")
    modelenv.write_marker(env, backend.family, backend.requirements, versions)


def run(step: str, argv: list[str], env: dict) -> str:
    """The stdout of a command that must succeed; if it does not, the step, its command and the tail of its stderr."""
    try:
        done = subprocess.run(argv, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              stdin=subprocess.DEVNULL)
    except OSError as error:
        raise PrepareError(f"{step} could not start {argv[0]}: {error}") from None
    if done.returncode != 0:
        tail = "\n".join((done.stderr.strip() or done.stdout.strip()).splitlines()[-TAIL_LINES:])
        raise PrepareError(f"{step} failed with exit code {done.returncode}: {' '.join(argv)}\n{tail}")
    return done.stdout
