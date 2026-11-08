"""`mms model` (DESIGN.md §13.3): list, use and prepare. No real uv, no weights, nothing outside tmp_path: the backends are
test classes that claim a folder by its name (so `model_folder`'s folders do), and `uv` is a script on PATH that records
what it was asked."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import tomllib
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from conftest import MODEL_FOLDERS, ROOT, model_folder, prepared_env, wait_for, write_config
from corpus import T0, scan
from fake_decider import ABOUT as FAKE_ABOUT
from fake_hub import FakeHub

from mms import cli, decider, decider_d1, fetch, modelenv, prepare
from mms.decider import About

FAKE_FILE = ROOT / "tests" / "fake_decider.py"
D3_PINS = ("torch==2.11.0", "transformers==5.17.0", "einops")
PPLX_PINS = ("torch==2.14.0", "pillow==12.3.0")
START = "cd {folder} && bash serve.sh"  # a template, as H2o.start is: `{folder}` is the model folder as WSL sees it
INSTALL = "python3 -m venv ~/h2o-venv\n~/h2o-venv/bin/pip install vllm==0.30.0"
BACKENDS = decider.BACKENDS  # the real ones, before the fixture below swaps them for the test classes
# what mms's own env holds in these tests: d1's pins, and a version for the rest
MMS_ENV = {name: pin or "1.0.0+cu128" for name, pin in map(modelenv.pinned, decider_d1.D1.requirements)}


def free_port() -> int:
    with socket.socket() as one:
        one.bind(("127.0.0.1", 0))
        return one.getsockname()[1]


DEAD_URL = f"http://127.0.0.1:{free_port()}"  # nothing listens there


class Fake:
    """A backend in the shape the real ones have; it claims a folder by its name and runs the fake decider."""

    abouts = ()
    reads_folder = "the test backend reads a folder named like its family"
    requirements = ()
    torch_backend = None
    served = False
    in_mms_env = False
    default_url = ""
    install = ""
    start = ""
    home = None
    made: list = []

    @staticmethod
    def check(about, questions):  # ask refuses a question the model cannot take: this one takes any
        pass

    @classmethod
    def claims(cls, folder):
        name = Path(folder).name
        return About(name=name, reads=("video", "image", "audio") if name == "d1-Omni-2B" else ("video", "image")) \
            if name.startswith(cls.prefix) else None

    def __init__(self, folder, *, url=None, envs=None):
        self.folder, self.url, self.envs = Path(folder), url, envs
        self.about = FAKE_ABOUT
        self.inner = decider.PythonFile(str(FAKE_FILE), Fake.home)
        Fake.made.append(self)

    def decide(self, state, questions, media=None) -> dict:
        return self.inner.decide(state, questions, media)


class FakeD1(Fake):
    prefix, family = "d1-", "d1"
    in_mms_env, requirements, install = True, decider_d1.D1.requirements, decider_d1.D1.install


class FakeD3(Fake):
    prefix, family = "d3-", "d3"
    requirements, torch_backend = D3_PINS, "cu128"


class FakePplx(Fake):
    prefix, family = "pplx", "pplx"
    requirements, torch_backend = PPLX_PINS, "cu128"


class FakeH2o(Fake):
    prefix, family = "h2o", "h2o"
    served, default_url, install, start = True, DEAD_URL, INSTALL, START


REAL_UV_COMMAND = prepare.uv_command  # taken at import, before conftest's `no_real_uv` replaces it for each test


@pytest.fixture(autouse=True)
def backends(monkeypatch, home):
    monkeypatch.setattr(decider, "BACKENDS", tuple(f"{__name__}:{c.__name__}" for c in (FakeD1, FakeD3, FakePplx, FakeH2o)))
    monkeypatch.setattr(Fake, "home", home)
    monkeypatch.setattr(Fake, "made", [])
    monkeypatch.setattr(modelenv, "installed", lambda: dict(MMS_ENV))  # d1 is ready unless a test says otherwise


@pytest.fixture
def hub(monkeypatch) -> FakeHub:
    """The Hub's two calls, faked: a download leaves the model's files in its folder."""
    return FakeHub(monkeypatch)


@pytest.fixture
def models(tmp_path) -> Path:
    """Every kind of model folder, and an envs folder in which only d3's env is prepared."""
    folder = tmp_path / "models"
    folder.mkdir()
    for name in MODEL_FOLDERS:
        model_folder(folder, name)
    prepared_env(folder / "envs", "d3", D3_PINS)
    return folder


def configure(home: Path, model: Path | None, extra: str = "", envs: str | None = None) -> str:
    lines = "# the owner's file\n[memory]\nloop_s = 0.1  # fast\nfree_gb = 0\n"
    if model is not None:
        lines += f'\n[decider]\nmodel = "{model.as_posix()}"  # the model\nidle_s = 600\n' + (f'envs = "{envs}"\n' if envs else "")
    write_config(home, lines + extra)
    return (home / "mms.toml").read_text(encoding="utf-8")


def model(home: Path, *argv: str) -> int:
    return prepare.main(home, list(argv))


def lines_of(capsys) -> dict[str, str]:
    """Each line `mms model list` printed, by the folder it is about."""
    return {line[2:].split()[0]: line for line in capsys.readouterr().out.splitlines()}


# list

def test_list_shows_each_folder_its_variant_what_it_reads_and_its_env(home, models, capsys):
    configure(home, models / "d3-edge")
    assert model(home) == 0  # no argument lists, in the folder that holds the configured model
    out = capsys.readouterr().out
    assert out == "\n".join([
        "  d1-2B                    d1-2B                    video, image         mms's own env",
        "  d1-3B                    d1-3B                    video, image         mms's own env",
        "  d1-Omni-2B               d1-Omni-2B               video, image, audio  mms's own env",
        "* d3-edge                  d3-edge                  video, image         ready",
        "  d3-flash                 d3-flash                 video, image         ready",
        "  d3-mini                  d3-mini                  video, image         ready",
        f"  h2o-lightning-4b         h2o-lightning-4b         video, image         served at {DEAD_URL}, does not answer",
        f"  pplx-decider-flash-v1.2  pplx-decider-flash-v1.2  video, image         not prepared: "
        f"{models / 'envs' / 'pplx' / 'mms-env.json'} is not there",
        "  unclaimed                no backend reads it",
    ]) + "\n", out
    # envs is not a model
    assert "envs" not in [line[2:].split()[0] for line in out.splitlines()]


def test_list_takes_a_dir_and_stars_only_the_configured_model(home, models, tmp_path, capsys):
    configure(home, None)
    assert model(home, "list", "--dir", str(models)) == 0
    assert not [line for line in capsys.readouterr().out.splitlines() if line.startswith("*")]
    configure(home, models / "d1-2B")
    assert model(home, "--dir", str(models)) == 0
    stars = [line for line in capsys.readouterr().out.splitlines() if line.startswith("*")]
    assert len(stars) == 1 and stars[0].split()[1] == "d1-2B"


def test_list_with_no_model_configured_looks_in_the_homes_models_folder_and_makes_nothing(home, capsys):
    configure(home, None)
    assert model(home) == 0
    assert capsys.readouterr().out == f"no folders in {home / 'models'}\n", "the test backends know no model to offer"
    assert not (home / "models").exists()


def test_list_offers_each_model_mms_knows_that_is_not_downloaded_and_makes_nothing(home, real_backends, capsys):
    before = configure(home, None)
    assert model(home) == 0
    rows = lines_of(capsys)
    assert list(rows) == [about.name for about in decider.known_abouts()]
    assert rows["d1-3B"].startswith("  d1-3B") and rows["d1-3B"].endswith(
        "video, image         not downloaded: 6.3 GB from huggingface.co/LiquidAI/d1-3B")
    assert rows["d1-Omni-2B"].startswith("  d1-Omni-2B") and rows["d1-Omni-2B"].endswith(
        "not downloaded: 4.4 GB from huggingface.co/LiquidAI/d1-Omni-2B")
    assert not (home / "models").exists() and (home / "mms.toml").read_text(encoding="utf-8") == before


def test_list_says_what_is_downloaded_what_is_not_all_there_and_what_is_still_to_download(home, real_backends, capsys):
    folder = home / "models"
    folder.mkdir()
    model_folder(folder, "d1-2B")
    (model_folder(folder, "d3-edge") / "model.safetensors").unlink()
    model_folder(folder, "unclaimed")
    configure(home, None)
    assert model(home) == 0
    rows = lines_of(capsys)
    still = [about.name for about in decider.known_abouts() if about.name not in ("d1-2B", "d3-edge")]
    assert list(rows) == ["d1-2B", "d3-edge", "unclaimed", *still], "the folders first, then the models not downloaded"
    assert rows["d1-2B"].endswith("mms's own env") and rows["unclaimed"].endswith("no backend reads it")
    assert rows["d3-edge"].endswith("not all there (no weights (*.safetensors) yet): mms model use d3-edge finishes it")
    assert all("not downloaded: " in rows[name] for name in still)


def test_list_refuses_a_dir_that_is_not_a_folder(home, tmp_path):
    configure(home, None)
    with pytest.raises(ValueError, match="nope is not a folder"):
        model(home, "list", "--dir", str(tmp_path / "nope"))
    configure(home, tmp_path / "gone" / "d1-2B")
    with pytest.raises(ValueError, match="gone is not a folder"):
        model(home)


def test_list_follows_the_configured_envs_folder(home, models, tmp_path, capsys):
    configure(home, models / "d3-edge", envs=(tmp_path / "elsewhere").as_posix())
    prepared_env(tmp_path / "elsewhere", "pplx", PPLX_PINS)
    model(home)
    rows = lines_of(capsys)
    assert "not prepared" in rows["d3-edge"] and "ready" in rows["pplx-decider-flash-v1.2"]


@contextmanager
def health_server(state: dict):
    """A server on a free loopback port whose `/health` answers `state["status"]`; yields its URL."""
    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(state["status"])
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def test_list_says_a_stale_env_is_not_prepared_and_a_served_model_that_answers(home, models, capsys):
    prepared_env(models / "envs", "pplx", ("torch==2.0.0",))  # made for other pins
    state = {"status": 200}
    with health_server(state) as url:
        write_config(home, f'[decider]\nmodel = "{(models / "h2o-lightning-4b").as_posix()}"\nurl = "{url}"\n')
        model(home)
        rows = lines_of(capsys)
        assert rows["h2o-lightning-4b"].startswith("*") and rows["h2o-lightning-4b"].endswith(f"served at {url}, answers")
        assert "other requirements" in rows["pplx-decider-flash-v1.2"]
        state["status"] = 503
        model(home)
        assert lines_of(capsys)["h2o-lightning-4b"].endswith("is warming")


# use

def test_use_by_name_writes_the_model_and_keeps_the_owners_comments(home, models, capsys):
    before = configure(home, models / "d1-Omni-2B")
    assert model(home, "use", "d3-mini") == 0
    after = (home / "mms.toml").read_text(encoding="utf-8")
    assert after == before.replace("d1-Omni-2B", "d3-mini"), "only the model changed: comments, order and values stay"
    assert configuration_model(home) == (models / "d3-mini").as_posix()
    out = capsys.readouterr().out
    assert "d3-mini" in out and "no daemon is running" in out


def configuration_model(home: Path) -> str:
    from mms import config

    return config.load(home).decider_model


def test_use_by_folder_needs_no_dir(home, models, capsys):
    configure(home, None)
    with pytest.raises(ValueError, match="no model folder 'd1-Omni-2B' in"):
        model(home, "use", "d1-Omni-2B")  # a bare name is in a models folder, and there is none there
    assert model(home, "use", str(models / "d1-2B")) == 0
    assert Path(configuration_model(home)) == models / "d1-2B"
    assert model(home, "use", "d1-Omni-2B") == 0, "now the configured model's folder is the one to go by"
    assert Path(configuration_model(home)) == models / "d1-Omni-2B"
    assert model(home, "use", "d3-edge", "--dir", str(models)) == 0


def test_use_sets_h2os_url_and_removes_it_when_switching_away(home, models, capsys):
    configure(home, models / "d1-Omni-2B")
    assert model(home, "use", "h2o-lightning-4b") == 0
    from mms import config

    assert config.load(home).decider_url == DEAD_URL, "a served model gets its default url"
    assert f"served at {DEAD_URL}" in capsys.readouterr().out
    other = "http://localhost:9999"
    assert model(home, "use", "h2o-lightning-4b", "--url", other) == 0
    assert config.load(home).decider_url == other
    text = (home / "mms.toml").read_text(encoding="utf-8")
    assert "# the owner's file" in text and "# fast" in text and "# the model" in text
    assert model(home, "use", "d3-edge") == 0
    assert config.load(home).decider_url is None and "\nurl" not in (home / "mms.toml").read_text(encoding="utf-8")


def test_use_refuses_what_cannot_work_and_changes_nothing(home, models):
    before = configure(home, models / "d1-Omni-2B")

    def refused(match: str, *argv: str) -> None:
        with pytest.raises(ValueError, match=match):
            model(home, "use", *argv)
        assert (home / "mms.toml").read_text(encoding="utf-8") == before, "a refusal writes nothing"

    refused("no decider reads the model", "unclaimed")
    refused("unclaimed", "unclaimed")  # it names what each backend reads, too
    refused("no model folder 'nope' in", "nope")
    refused("no model folder at", str(models / "nope" / "deeper"))
    refused("loopback|\\[decider\\] url", "h2o-lightning-4b", "--url", "http://192.168.1.5:8741")
    refused("loopback|\\[decider\\] url", "h2o-lightning-4b", "--url", "https://127.0.0.1:8741")
    refused("not served", "d3-edge", "--url", "http://127.0.0.1:8741")


def test_use_tries_to_build_an_env_that_is_not_prepared_and_a_build_that_fails_changes_nothing(home, models, capsys):
    """pplx's env is not there and d3's marker is gone: `use` runs the build, and the test's uv refuses."""
    before = configure(home, models / "d1-Omni-2B")
    (models / "envs" / "d3" / "mms-env.json").unlink()
    for name in ("pplx-decider-flash-v1.2", "d3-flash"):
        assert model(home, "use", name) == 1
        err = capsys.readouterr().err
        assert err.startswith("mms: uv venv failed with exit code 1") and "no real uv in tests" in err, err
        assert (home / "mms.toml").read_text(encoding="utf-8") == before, "a failed build writes nothing"


def test_use_refuses_d1_when_mms_own_env_lacks_torch_or_has_another_transformers(home, models, monkeypatch, capsys):
    configure(home, models / "d3-edge")
    held = dict(MMS_ENV)
    monkeypatch.setattr(modelenv, "installed", lambda: held)
    assert model(home, "use", "d1-2B") == 0, "torch and mms's transformers pin: ready"
    before = configure(home, models / "d3-edge")
    held["transformers"] = "5.22.1"
    with pytest.raises(ValueError, match=r"d1-2B runs in mms's own env, which is not ready: transformers is 5\.22\.1, not the "
                                         r"5\.21\.0 that mms pins: run `uv sync --group d1` in mms's project"):
        model(home, "use", "d1-2B")
    held["torch"] = None
    with pytest.raises(ValueError, match=r"d1-Omni-2B runs in mms's own env, which is not ready: torch is not installed.*"
                                         r"uv sync --group d1"):
        model(home, "use", "d1-Omni-2B")
    assert (home / "mms.toml").read_text(encoding="utf-8") == before, "a refusal writes nothing"
    # a refusal of bad input exits 2 like every mms command; prepare's own failures exit 1
    assert cli.main(["--home", str(home), "model", "use", "d1-2B"]) == 2
    assert "d1-2B runs in mms's own env, which is not ready" in capsys.readouterr().err
    assert cli.main(["--home", str(home), "model", "prepare", "d1-2B"]) == 1
    assert "torch is not installed" in capsys.readouterr().err
    assert cli.main(["--home", str(home), "model", "list"]) == 0
    assert "not prepared: torch is not installed" in lines_of(capsys)["d1-2B"]


def test_use_warns_when_the_model_cannot_hear_and_a_microphone_or_camera_records(home, models, capsys):
    mic = '\n[sources.mic]\nkind = "microphone"\ndevice = "Some Mic"\n[sources.cam]\nkind = "camera"\ndevice = "Some Cam"\n'
    configure(home, models / "d1-Omni-2B", extra=mic)
    assert model(home, "use", "d3-edge") == 0
    err = capsys.readouterr().err
    assert "d3-edge does not hear sound" in err and "microphone mic" in err and "camera cam" in err
    assert model(home, "use", "d1-Omni-2B") == 0
    assert "warning" not in capsys.readouterr().err, "a model that hears raises no warning"
    configure(home, models / "d1-Omni-2B")
    assert model(home, "use", "d3-edge") == 0
    assert "warning" not in capsys.readouterr().err, "no sound source, no warning"


def test_use_through_the_cli_and_its_refusals_exit_2(home, models, capsys):
    configure(home, models / "d1-Omni-2B")
    assert cli.main(["--home", str(home), "model", "use", "d1-2B"]) == 0
    assert cli.main(["--home", str(home), "model", "use", "unclaimed"]) == 2
    assert "mms: no decider reads the model" in capsys.readouterr().err
    assert cli.main(["--home", str(home), "model", "list"]) == 0
    assert "* d1-2B" in capsys.readouterr().out


def test_use_builds_an_env_that_is_not_prepared_and_then_writes_the_model(home, models, tmp_path, monkeypatch, capsys):
    configure(home, models / "d1-Omni-2B")
    log = fake_uv(tmp_path / "bin", monkeypatch)
    pins = installed_pins()
    monkeypatch.setattr(FakePplx, "requirements", pins)
    monkeypatch.setattr(modelenv, "python", lambda env: Path(sys.executable))
    env = models / "envs" / "pplx"
    env.mkdir()  # uv made it, with a python in it
    assert model(home, "use", "pplx-decider-flash-v1.2") == 0
    assert [c["argv"][:2] for c in calls(log)] == [["pip", "install"]]
    assert modelenv.marker_problem(env, pins) is None
    assert configuration_model(home) == (models / "pplx-decider-flash-v1.2").as_posix()
    out = capsys.readouterr().out
    assert out.index("5/5") < out.index("model: pplx-decider-flash-v1.2")
    assert model(home, "use", "pplx-decider-flash-v1.2") == 0
    assert len(calls(log)) == 1, "an env that is prepared is not built again"


def test_use_prints_h2os_commands_when_its_server_does_not_answer(home, models, capsys):
    configure(home, models / "d1-Omni-2B")
    start = START.format(folder=prepare.wsl(models / "h2o-lightning-4b"))
    assert model(home, "use", "h2o-lightning-4b") == 0  # nothing listens at DEAD_URL
    lines = capsys.readouterr().out.splitlines()
    chosen = f"model: h2o-lightning-4b ({models / 'h2o-lightning-4b'}), served at {DEAD_URL}"
    assert "h2o-lightning-4b is a server you run, so mms installs nothing." in lines and INSTALL.splitlines()[0] in lines
    assert lines.index("then, to serve it:") < lines.index(start) < lines.index(chosen), "the model is chosen all the same"


@pytest.mark.parametrize("status", [200, 503])
def test_use_prints_no_commands_when_h2os_server_answers_or_is_warming(home, models, capsys, status):
    configure(home, models / "d1-Omni-2B")
    with health_server({"status": status}) as url:  # it is running, so there is nothing to start
        assert model(home, "use", "h2o-lightning-4b", "--url", url) == 0
    out = capsys.readouterr().out
    assert "mms installs nothing" not in out and "serve it" not in out and f"served at {url}" in out


def test_a_running_daemon_follows_the_switch(home, models, served, red_clip, capsys):
    configure(home, models / "d1-Omni-2B")
    running = served()
    api, gpu = running.api, running.daemon.gpu
    assert api.call("status")["decider"]["model"] == (models / "d1-Omni-2B").as_posix()

    assert model(home, "use", "h2o-lightning-4b") == 0
    assert "the daemon is running" in capsys.readouterr().out
    wait_for(lambda: gpu.model == (models / "h2o-lightning-4b").as_posix() and gpu.url == DEAD_URL,
             what="the daemon to read the new model and its url")
    assert api.call("status")["decider"]["model"] == (models / "h2o-lightning-4b").as_posix()
    api.call("remember", path=str(red_clip), source="clips", t=T0)
    assert scan(api, T0 + 100, question="Is it red?")["sightings"], "the new model is the one that reads"
    assert Fake.made[-1].url == DEAD_URL and Fake.made[-1].folder == models / "h2o-lightning-4b"

    assert model(home, "use", "d3-edge") == 0
    wait_for(lambda: gpu.model == (models / "d3-edge").as_posix() and gpu.url is None, what="the url to go with the switch")
    assert scan(api, T0 + 100, question="Is it red?")["sightings"]
    assert Fake.made[-1].url is None and Fake.made[-1].folder == models / "d3-edge"


# prepare

def fake_uv(folder: Path, monkeypatch, fail_on: str | None = None) -> Path:
    """A `uv` that appends its argv and UV_CACHE_DIR to a log, and fails when its first argument is `fail_on`. It is run as
    a Python script, never through a .cmd, which would take a pin's `>=` for a redirect."""
    folder.mkdir(exist_ok=True)
    log = folder / "uv.jsonl"
    script = folder / "uv.py"
    script.write_text(
        "import json, os, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'cache': os.environ.get('UV_CACHE_DIR')}}) + '\\n')\n"
        f"if sys.argv[1:2] == [{fail_on!r}]:\n    print('uv: the wheel for torch was not found', file=sys.stderr)\n    sys.exit(3)\n",
        encoding="utf-8")
    monkeypatch.setattr(prepare, "uv_command", lambda: [sys.executable, str(script)])
    return log


def calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


def installed_pins() -> tuple[str, ...]:
    import PIL

    return (f"pytest=={pytest.__version__}", f"pillow=={PIL.__version__}", "numpy>=1")


def test_prepare_runs_uv_with_its_cache_beside_the_envs_and_writes_no_marker_until_the_versions_check(home, models, tmp_path,
                                                                                                    monkeypatch, capsys):
    configure(home, models / "d1-Omni-2B")
    log = fake_uv(tmp_path / "bin", monkeypatch)
    env = models / "envs" / "pplx"
    assert model(home, "prepare", "pplx-decider-flash-v1.2") == 1, "the fake uv made no python, so the check cannot run"
    venv, install = calls(log)
    cache = str(models / "envs" / ".uv-cache")
    assert venv == {"argv": ["venv", "--python", "3.12", str(env)], "cache": cache}
    assert install["cache"] == cache
    assert install["argv"] == ["pip", "install", "--python", str(modelenv.python(env)), "--torch-backend", "cu128", *PPLX_PINS]
    captured = capsys.readouterr()
    assert "version check could not start" in captured.err
    assert not (env / "mms-env.json").exists(), "no marker without the check"
    assert "1/5" in captured.out and "2/5" in captured.out and "3/5" in captured.out and "5/5" not in captured.out


def test_prepare_writes_the_marker_only_after_the_versions_match_and_use_then_works(home, models, tmp_path, monkeypatch, capsys):
    configure(home, models / "d1-Omni-2B")
    log = fake_uv(tmp_path / "bin", monkeypatch)
    pins = installed_pins()
    monkeypatch.setattr(FakePplx, "requirements", pins)
    monkeypatch.setattr(modelenv, "python", lambda env: Path(sys.executable))
    env = models / "envs" / "pplx"
    env.mkdir()  # uv made it, with a python in it
    assert model(home, "prepare", "pplx-decider-flash-v1.2") == 0
    assert [c["argv"][:2] for c in calls(log)] == [["pip", "install"]], "the env has a python already, so no venv is made"
    marker = json.loads((env / "mms-env.json").read_text(encoding="utf-8"))
    assert marker["family"] == "pplx" and marker["requirements"] == list(pins)
    assert marker["versions"] == {"pytest": pytest.__version__, "pillow": pins[1].split("==")[1], "numpy": marker["versions"]["numpy"]}
    assert marker["prepared"].startswith("20")
    assert modelenv.marker_problem(env, pins) is None
    out = capsys.readouterr().out
    assert "5/5" in out and "mms model use pplx-decider-flash-v1.2" in out
    assert model(home, "use", "pplx-decider-flash-v1.2") == 0, "a prepared env is a usable one"


def test_prepare_with_a_wrong_pin_fails_names_it_and_leaves_the_env_unprepared(home, models, tmp_path, monkeypatch, capsys):
    configure(home, models / "d1-Omni-2B")
    fake_uv(tmp_path / "bin", monkeypatch)
    monkeypatch.setattr(FakeD3, "requirements", ("pytest==0.0.1", "numpy>=1", "no-such-package-here"))
    monkeypatch.setattr(modelenv, "python", lambda env: Path(sys.executable))
    marker = models / "envs" / "d3" / "mms-env.json"
    assert marker.exists()
    assert model(home, "prepare", "d3-edge") == 1
    err = capsys.readouterr().err
    assert f"pytest is {pytest.__version__}, not the 0.0.1" in err and "no-such-package-here is not installed" in err
    assert "numpy" not in err
    assert not marker.exists(), "a prepare that fails leaves the env unprepared, not as it was"


def test_prepare_reports_the_failed_step_and_the_tail_of_its_stderr(home, models, tmp_path, monkeypatch, capsys):
    configure(home, models / "d1-Omni-2B")
    log = fake_uv(tmp_path / "bin", monkeypatch, fail_on="pip")
    assert model(home, "prepare", "pplx-decider-flash-v1.2") == 1
    err = capsys.readouterr().err
    assert "uv pip install failed with exit code 3" in err and "the wheel for torch was not found" in err
    assert len(calls(log)) == 2 and not (models / "envs" / "pplx" / "mms-env.json").exists()


def test_prepare_without_uv_says_so(home, models, monkeypatch, capsys):
    configure(home, models / "d1-Omni-2B")
    monkeypatch.setattr(prepare, "uv_command", REAL_UV_COMMAND)  # the one that looks on PATH
    monkeypatch.setattr(prepare.shutil, "which", lambda name: None)
    assert model(home, "prepare", "pplx-decider-flash-v1.2") == 1
    assert "uv is not on PATH" in capsys.readouterr().err


def test_prepare_keeps_the_uv_cache_under_a_moved_envs_folder(home, models, tmp_path, monkeypatch):
    configure(home, models / "d1-Omni-2B", envs=(tmp_path / "big" / "envs").as_posix())
    log = fake_uv(tmp_path / "bin", monkeypatch)
    model(home, "prepare", "d3-edge")
    assert {c["cache"] for c in calls(log)} == {str(tmp_path / "big" / "envs" / ".uv-cache")}
    assert calls(log)[0]["argv"][-1] == str(tmp_path / "big" / "envs" / "d3")


def test_prepare_d1_installs_nothing_and_checks_the_env_it_runs_in(home, models, tmp_path, monkeypatch, capsys):
    configure(home, models / "d1-Omni-2B")
    log = fake_uv(tmp_path / "bin", monkeypatch)
    versions = dict(MMS_ENV)
    monkeypatch.setattr(modelenv, "installed", lambda: versions)
    assert model(home, "prepare", "d1-Omni-2B") == 0
    assert not calls(log), "uv is not run"
    assert "ready: " + ", ".join(f"{name} {version}" for name, version in MMS_ENV.items()) in capsys.readouterr().out
    versions["transformers"] = "5.22.1"
    assert model(home, "prepare", "d1-2B") == 1
    assert "transformers is 5.22.1, not the 5.21.0 that mms pins: run `uv sync --group d1` in mms's project" in capsys.readouterr().err
    versions["torch"] = None
    assert model(home, "prepare", "d1-2B") == 1
    assert "torch is not installed" in capsys.readouterr().err
    assert not calls(log) and not (models / "envs" / "d1").exists()


def test_d1s_requirements_are_the_d1_group_of_pyproject():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert tuple(pyproject["dependency-groups"]["d1"]) == decider_d1.D1.requirements


def test_prepare_h2o_prints_the_install_then_the_start_command_and_installs_nothing(home, models, tmp_path, monkeypatch, capsys):
    configure(home, models / "d1-Omni-2B")
    log = fake_uv(tmp_path / "bin", monkeypatch)
    start = START.format(folder=prepare.wsl(models / "h2o-lightning-4b"))
    assert model(home, "prepare", "h2o-lightning-4b") == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines.index("in WSL Ubuntu, once:") < lines.index(INSTALL.splitlines()[0]) < lines.index("then, to serve it:") \
        < lines.index(start), out
    assert INSTALL.splitlines()[1] in lines and "mms model use h2o-lightning-4b" in out
    assert not calls(log) and not (models / "envs" / "h2o").exists()
    monkeypatch.setattr(FakeH2o, "install", "")
    assert model(home, "prepare", "h2o-lightning-4b") == 0
    out = capsys.readouterr().out
    assert "in WSL Ubuntu" not in out and "then," not in out and "to serve it:" in out.splitlines() and start in out.splitlines()


def test_wsl_sees_a_folder_as_wsl_does_and_quotes_what_the_shell_would_split():
    if os.name == "nt":
        assert prepare.wsl(Path("C:/models/h2o-lightning-4b")) == "/mnt/c/models/h2o-lightning-4b"
        assert prepare.wsl(Path("C:/Users/Some One/models")) == "'/mnt/c/Users/Some One/models'"
        with pytest.raises(prepare.PrepareError, match="not on a lettered drive"):
            prepare.wsl(Path("//server/share/models"))
    else:
        assert prepare.wsl(Path("/srv/models/x")) == "/srv/models/x"
        assert prepare.wsl(Path("/srv/my models")) == "'/srv/my models'"


def test_the_pins_are_compared_by_name_and_version():
    held = {"torch": "2.11.0+cu128", "einops": "0.8.1", "pillow": "12.3.0"}
    assert modelenv.mismatch("torch==2.11.0", held) is None, "torch's wheel ends in +cu128"
    assert modelenv.mismatch("torch==2.11.0+cu128", held) is None
    assert "torch is 2.11.0+cu128, not the 2.11.1" in modelenv.mismatch("torch==2.11.1", held)
    assert "torch is 2.11.0+cu128, not the 2.11.0+cu130" in modelenv.mismatch("torch==2.11.0+cu130", held)
    assert modelenv.mismatch("einops", held) is None and modelenv.mismatch("accelerate>=1.10", held) == "accelerate is not installed"
    versions, wrong = modelenv.compare(("torch==2.11.0", "einops", "accelerate>=1.10", "pillow==12.3.1"), held)
    assert versions == {"torch": "2.11.0+cu128", "einops": "0.8.1"}
    assert wrong == ["accelerate is not installed", "pillow is 12.3.0, not the 12.3.1 that mms pins"]
    assert modelenv.pinned("Pillow==12.3.0") == ("pillow", "12.3.0") and modelenv.pinned("huggingface_hub>=1.0") == ("huggingface-hub", None)


# the real backends, on the folders conftest makes (the fake uv, no weights)

@pytest.fixture
def real_backends(monkeypatch):
    monkeypatch.setattr(decider, "BACKENDS", BACKENDS)


@pytest.fixture
def real(real_backends, monkeypatch, models, home):
    """The real backend classes, d3's and pplx's envs prepared for their real requirements, h2o not answering."""
    from mms.decider_d3 import D3
    from mms.decider_pplx import Pplx

    monkeypatch.setattr(prepare, "health", lambda url: "does not answer")
    prepared_env(models / "envs", "d3", D3.requirements)
    prepared_env(models / "envs", "pplx", Pplx.requirements)
    configure(home, models / "d1-Omni-2B")
    return models


def test_list_with_the_real_backends_claims_every_conftest_folder_but_the_unclaimed_one(home, real, capsys):
    assert model(home) == 0
    rows = {line[2:].split()[0]: line for line in capsys.readouterr().out.splitlines()}
    assert list(rows) == ["d1-2B", "d1-3B", "d1-Omni-2B", "d3-edge", "d3-flash", "d3-mini", "h2o-lightning-4b",
                          "pplx-decider-flash-v1.2", "unclaimed"], "every model is in a folder: none is offered to download"
    assert rows["d1-Omni-2B"].startswith("*") and rows["d1-Omni-2B"].endswith("video, image, audio  mms's own env")
    assert rows["d1-2B"].endswith("video, image         mms's own env")
    assert rows["d1-3B"].endswith("video, image         mms's own env")
    for name in ("d3-edge", "d3-flash", "d3-mini", "pplx-decider-flash-v1.2"):
        assert rows[name].endswith("video, image         ready"), rows[name]
    assert rows["h2o-lightning-4b"].endswith("video, image         served at http://127.0.0.1:8741, does not answer")
    assert rows["unclaimed"].endswith("no backend reads it")
    assert "envs" not in rows


def test_list_with_the_real_backends_says_which_envs_are_not_prepared(home, real, capsys):
    (real / "envs" / "pplx" / "mms-env.json").unlink()
    prepared_env(real / "envs", "d3", ("torch==0.0.1",))
    model(home)
    rows = lines_of(capsys)
    assert "not prepared" in rows["pplx-decider-flash-v1.2"] and "mms-env.json is not there" in rows["pplx-decider-flash-v1.2"]
    assert "not prepared" in rows["d3-edge"] and "other requirements" in rows["d3-edge"]


@pytest.mark.parametrize("name", ["d1-Omni-2B", "d1-2B", "d1-3B", "d3-edge", "d3-mini", "d3-flash", "pplx-decider-flash-v1.2"])
def test_use_with_the_real_backends_switches_to_each_folder_that_is_ready(home, real, name, hub):
    configure(home, real / "d3-edge" if name == "d1-Omni-2B" else real / "d1-Omni-2B")
    assert model(home, "use", name) == 0
    from mms import config

    assert Path(config.load(home).decider_model) == real / name and config.load(home).decider_url is None
    assert hub.calls == [], "a model that is in its folder is not downloaded"


def test_use_with_the_real_backends_gives_h2o_its_default_url_and_refuses_what_is_not_ready(home, real, capsys):
    from mms import config
    from mms.decider_h2o import H2o

    assert model(home, "use", "h2o-lightning-4b") == 0
    assert config.load(home).decider_url == H2o.default_url
    before = configure(home, real / "d1-Omni-2B")
    (real / "envs" / "pplx" / "mms-env.json").unlink()
    capsys.readouterr()
    assert model(home, "use", "pplx-decider-flash-v1.2") == 1, "the env is built, and the test's uv refuses"
    assert "no real uv in tests" in capsys.readouterr().err
    assert (home / "mms.toml").read_text(encoding="utf-8") == before
    with pytest.raises(ValueError, match="no decider reads the model"):
        model(home, "use", "unclaimed")
    with pytest.raises(ValueError, match="not served"):
        model(home, "use", "d3-edge", "--url", "http://127.0.0.1:8741")
    assert (home / "mms.toml").read_text(encoding="utf-8") == before


def test_prepare_with_the_real_backends_runs_uv_for_the_real_pins_and_not_for_d1_or_h2o(home, real, tmp_path, monkeypatch, capsys):
    from mms.decider_d3 import D3
    from mms.decider_h2o import H2o
    from mms.decider_pplx import Pplx

    log = fake_uv(tmp_path / "bin", monkeypatch)
    assert model(home, "prepare", "pplx-decider-flash-v1.2") == 1, "the fake uv made no python, so the version check cannot run"
    venv, install = calls(log)
    env = real / "envs" / "pplx"
    assert venv["argv"] == ["venv", "--python", "3.12", str(env)]
    assert install["argv"] == ["pip", "install", "--python", str(modelenv.python(env)), "--torch-backend", Pplx.torch_backend,
                               *Pplx.requirements]
    assert model(home, "prepare", "d3-mini") == 1
    assert calls(log)[-1]["argv"][-len(D3.requirements):] == list(D3.requirements)
    seen = len(calls(log))
    capsys.readouterr()
    assert model(home, "prepare", "d1-Omni-2B") == 0, "mms's own env holds what the pins say"
    assert "ready: " + ", ".join(f"{name} {version}" for name, version in MMS_ENV.items()) in capsys.readouterr().out
    assert model(home, "prepare", "h2o-lightning-4b") == 0
    out = capsys.readouterr().out.splitlines()
    assert out.index("in WSL Ubuntu, once:") < out.index(H2o.install) < out.index("then, to serve it:")
    assert out[out.index("then, to serve it:") + 1] == H2o.start.format(folder=prepare.wsl(real / "h2o-lightning-4b"))
    assert len(calls(log)) == seen, "uv is not run for d1 or H2O"
    with pytest.raises(ValueError, match="no decider reads the model"):
        model(home, "prepare", "unclaimed")


# downloads, with the real backends and the Hub faked

def test_use_downloads_a_model_that_is_not_there_into_the_dir_under_its_own_name_and_writes_it(home, real_backends, tmp_path,
                                                                                               hub, capsys):
    fresh = tmp_path / "fresh"  # does not exist yet
    before = configure(home, None)
    assert model(home, "use", "d1-3b", "--dir", str(fresh)) == 0  # any case
    assert hub.calls == [("info", "LiquidAI/d1-3B"), ("download", "LiquidAI/d1-3B", fresh / "d1-3B", hub.commit)]
    assert configuration_model(home) == (fresh / "d1-3B").as_posix(), "the name as the model has it"
    written = (home / "mms.toml").read_text(encoding="utf-8")
    assert written.startswith(before) and tomllib.loads(written)["decider"] == {"model": (fresh / "d1-3B").as_posix()}
    out = capsys.readouterr().out
    assert out.startswith("downloading LiquidAI/d1-3B at 0000000 (6.3 GB, licence lfm1.0) into ")
    assert "model: d1-3B" in out and "d1-3B hears nothing: mms model use d1-Omni-2B gives one that hears" in out


def test_use_finishes_a_model_whose_weights_are_not_all_there(home, real, hub):
    (real / "d1-2B" / "model.safetensors").unlink()
    assert model(home, "use", "d1-2B") == 0
    assert hub.calls == [("info", "LiquidAI/d1-2B"), ("download", "LiquidAI/d1-2B", real / "d1-2B", hub.commit)]
    assert fetch.ready(decider.known("d1-2B")[1], real / "d1-2B") and configuration_model(home) == (real / "d1-2B").as_posix()


def test_a_failed_download_exits_1_and_changes_nothing(home, real, tmp_path, hub, capsys):
    before = configure(home, real / "d1-Omni-2B")
    hub.fail = fetch.FetchError(f"the download of LiquidAI/d1-3B failed: OSError: no route; {fetch.RETRY}")
    assert model(home, "use", "d1-3B", "--dir", str(tmp_path / "fresh")) == 1
    assert capsys.readouterr().err == f"mms: {hub.fail}\n"
    assert (home / "mms.toml").read_text(encoding="utf-8") == before
    hub.fail = None
    assert model(home, "use", "d1-3B", "--dir", str(tmp_path / "fresh")) == 0, "the same command downloads what is missing"


def test_a_folder_of_the_same_name_that_is_not_that_model_is_not_downloaded_into(home, real, hub, capsys):
    before = configure(home, real / "d1-Omni-2B")
    (real / "d1-3B").rename(real / "d1-3B-moved")
    (real / "d1-3B").mkdir()
    (real / "d1-3B" / "notes.txt").write_text("mine", encoding="utf-8")
    assert model(home, "use", "d1-3B") == 1
    assert "holds something that is not d1-3B, so nothing is downloaded into it" in capsys.readouterr().err
    assert hub.calls == [] and (real / "d1-3B" / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert (home / "mms.toml").read_text(encoding="utf-8") == before


@pytest.mark.parametrize("command", ["use", "prepare"])
def test_a_download_leaves_the_floor_of_mms_free_when_the_models_are_on_the_drive_of_the_home(
        home, real_backends, tmp_path, hub, monkeypatch, capsys, command):
    write_config(home, "[memory]\nfree_gb = 20\n")
    before = (home / "mms.toml").read_text(encoding="utf-8")
    fresh = tmp_path / "fresh"  # under tmp_path, as the home is
    hub.free = 25_000_000_000  # the repo is 6.3 GB: 18.7 GB would be left
    assert model(home, command, "d1-3B", "--dir", str(fresh)) == 1
    assert ("would leave 18.7 GB free, under [memory] free_gb (20 GB), which pauses capture: free some space or give "
            "another --dir") in capsys.readouterr().err
    assert hub.calls == [("info", "LiquidAI/d1-3B")] and not fresh.exists()
    assert (home / "mms.toml").read_text(encoding="utf-8") == before
    monkeypatch.setattr(fetch, "same_drive", lambda a, b: False)  # another drive: capture is not paused by what it takes
    assert model(home, command, "d1-3B", "--dir", str(fresh)) == 0
    assert hub.calls[-1] == ("download", "LiquidAI/d1-3B", fresh / "d1-3B", hub.commit)


def test_d1_is_refused_before_any_download_when_mms_own_env_is_not_ready(home, real_backends, tmp_path, hub, monkeypatch, capsys):
    monkeypatch.setattr(modelenv, "installed", dict)  # nothing is installed
    before = configure(home, None)
    fresh = tmp_path / "fresh"
    with pytest.raises(ValueError, match=r"d1-3B runs in mms's own env, which is not ready: .*uv sync --group d1"):
        model(home, "use", "d1-3B", "--dir", str(fresh))
    assert model(home, "prepare", "d1-3B", "--dir", str(fresh)) == 1
    assert "uv sync --group d1" in capsys.readouterr().err
    assert hub.calls == [] and not fresh.exists(), "no network call, nothing made"
    assert (home / "mms.toml").read_text(encoding="utf-8") == before


def test_prepare_downloads_a_model_and_checks_its_env_without_switching(home, real_backends, tmp_path, hub, capsys):
    before = configure(home, None)
    fresh = tmp_path / "fresh"
    assert model(home, "prepare", "d1-3B", "--dir", str(fresh)) == 0
    assert hub.calls == [("info", "LiquidAI/d1-3B"), ("download", "LiquidAI/d1-3B", fresh / "d1-3B", hub.commit)]
    assert "ready: " + ", ".join(f"{name} {version}" for name, version in MMS_ENV.items()) in capsys.readouterr().out
    assert (home / "mms.toml").read_text(encoding="utf-8") == before, "prepare switches nothing"


def test_prepare_downloads_h2o_and_prints_its_commands(home, real_backends, tmp_path, hub, capsys):
    from mms.decider_h2o import H2o

    configure(home, None)
    fresh = tmp_path / "fresh"
    assert model(home, "prepare", "h2o-lightning-4b", "--dir", str(fresh)) == 0
    assert hub.calls == [("info", "h2oai/h2o-lightning-4b"),
                         ("download", "h2oai/h2o-lightning-4b", fresh / "h2o-lightning-4b", hub.commit)]
    assert H2o.start.format(folder=prepare.wsl(fresh / "h2o-lightning-4b")) in capsys.readouterr().out.splitlines()


def test_use_downloads_then_builds_the_env_then_writes_the_model(home, real_backends, tmp_path, hub, monkeypatch, capsys):
    from mms.decider_d3 import D3

    pins = installed_pins()
    monkeypatch.setattr(D3, "requirements", pins)
    monkeypatch.setattr(modelenv, "python", lambda env: Path(sys.executable))
    log = fake_uv(tmp_path / "bin", monkeypatch)
    fresh = tmp_path / "fresh"
    (fresh / "envs" / "d3").mkdir(parents=True)  # uv made it, with a python in it
    configure(home, None)
    assert model(home, "use", "D3-EDGE", "--dir", str(fresh)) == 0
    assert hub.calls == [("info", "vllm-sr/d3-edge"), ("download", "vllm-sr/d3-edge", fresh / "d3-edge", hub.commit)]
    assert [c["argv"][:2] for c in calls(log)] == [["pip", "install"]]
    assert configuration_model(home) == (fresh / "d3-edge").as_posix()
    out = capsys.readouterr().out
    assert out.index("downloading vllm-sr/d3-edge") < out.index("5/5") < out.index("model: d3-edge")
