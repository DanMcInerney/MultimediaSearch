"""The config (TESTBED.md item 5): what the file may hold, the writer that keeps its comments, and the daemon's live
reload, with `source_set` and a hand edit."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest
from conftest import approve_hint, wait_for, write_config

from mms import config as configuration
from mms import consent
from mms.config import ConfigError, load_text, parse_ahead, parse_when
from mms.store import Store

PROJECT = "git:0123456789abcdef"
ENVS = (Path.cwd() / "elsewhere" / "envs").as_posix()  # absolute here, whatever the platform


def sources(status: dict) -> dict:
    return {s["name"]: s for s in status["sources"]}


# the file

def test_every_kind_loads_with_its_options_and_a_cap():
    config = load_text("h", """
[memory]
free_gb = 5
jobs = 2
remember_max_mb = 10
job_max_s = 60
[decider]
model = "python:x.py"
idle_s = 30
[sources.screen]
kind = "screen"
device = "0"
fps = 2
max_width = 1280
keep = { max_gb = 40, when_full = "thin", max_days = 30 }
[sources.mic]
kind = "microphone"
device = "Microphone (Realtek(R) Audio)"
[sources.cam]
kind = "camera"
device = "HD Webcam"
[sources.raw]
kind = "ffmpeg"
input = ["-f", "lavfi", "-i", "sine"]
preset = "audio"
[sources.notes]
kind = "folder"
path = "~/notes"
project = "git:0123456789abcdef"
[sources.logs]
kind = "log"
path = "/var/log"
time_field = "ts"
[sources.job]
kind = "command"
command = ["python", "x.py"]
[sources.prog]
kind = "program"
segment_s = 30
[sources.agent]
kind = "push"
""")
    assert (config.free_gb, config.jobs, config.remember_max_mb, config.job_max_s) == (5, 2, 10, 60)
    assert (config.decider_model, config.decider_idle_s) == ("python:x.py", 30)
    assert set(config.sources) == {"screen", "mic", "cam", "raw", "notes", "logs", "job", "prog", "agent"}
    assert config.sources["screen"].keep == configuration.Keep(40, "thin", 30)
    assert config.sources["notes"].project == PROJECT and config.sources["screen"].project is None
    assert config.sources["agent"].keep == configuration.Keep(None, "drop_oldest", None), "no cap: no max_gb"
    defaults = load_text("h", "")
    assert (defaults.free_gb, defaults.jobs, defaults.remember_max_mb, defaults.decider_idle_s, defaults.decider_model) == (
        20, 4, 2000, 600, None)


@pytest.mark.parametrize("text, message", [
    ('[sources.x]\nkind = "telepathy"', "telepathy"),
    ('[sources.x]\nkind = "folder"', "needs path"),
    ('[sources.x]\nkind = "folder"\npath = "a"\nfps = 2', "fps"),
    ('[sources.x]\nkind = "folder"\npath = "a"\nglob = "../x"', "glob"),
    ('[sources.x]\nkind = "log"\npath = "a"\nglob = "a/../../x"', "glob"),
    ('[sources.x]\nkind = "folder"\npath = "a"\nglob = "/etc/x"', "glob"),
    ('[sources.x]\nkind = "screen"\nmax_width = 1921', "even"),
    ('[sources.x]\nkind = "microphone"', "needs device"),
    ('[sources.x]\nkind = "command"\ncommand = "python x.py"', "list"),
    ('[sources.x]\nkind = "ffmpeg"\ninput = ["-i", "x"]', "preset"),
    ('[sources.x]\nkind = "ffmpeg"\ninput = ["-i", "x"]\npreset = "movie"', "preset"),
    ('[sources.x]\nkind = "program"\nsegment_s = 0', "segment_s"),
    ('[sources.x]\nkind = "push"\nkeep = { max_gb = -1 }', "max_gb"),
    ('[sources.x]\nkind = "push"\nkeep = { when_full = "never" }', "when_full"),
    ('[sources.x]\nkind = "push"\nproject = "my project"', "project"),
    ('[sources."bad name"]\nkind = "push"', "name"),
    ("[memory]\nfree_gigs = 1", "free_gigs"),
    ("[decider]\nmodel = 3", "model"),
    ("[decider]\nurl = 3", "url"),
    ("[decider]\nenvs = 3", "envs"),
    ('[decider]\nenvs = ""', "envs"),
    ("[extra]\na = 1", "extra"),
    ("not toml at all [", ""),
])
def test_a_file_the_loader_refuses_names_what_is_wrong(text, message):
    with pytest.raises(ConfigError) as refused:
        load_text("h", text)
    assert message in str(refused.value)


def test_times_are_parsed_and_written_as_the_design_says():
    now = 1_800_000_000.0
    assert parse_when("now", now) == now and parse_when("90s", now) == now - 90 and parse_when("2h", now) == now - 7200
    assert parse_when("1w", now) == now - 604800 and parse_when(12.5, now) == 12.5 and parse_when("12.5", now) == 12.5
    assert parse_when("2026-10-31T09:14:02Z") == pytest.approx(1793438042) and parse_when(None) is None
    assert parse_when("2026-10-31T09:14:02") == pytest.approx(
        time.mktime(time.strptime("2026-10-31T09:14:02", "%Y-%m-%dT%H:%M:%S")), abs=3700), "no zone is local"
    assert parse_ahead("30d", now) == now + 30 * 86400 and parse_ahead("2026-10-31T09:14:02Z") == pytest.approx(1793438042)
    with pytest.raises(ValueError):
        parse_when("tomorrow-ish")
    with pytest.raises(ValueError):
        parse_when(True)
    shown = configuration.iso(1793438042.35)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.350[+-]\d\d:\d\d", shown), "the host's zone, to the millisecond"
    assert parse_when(shown) == pytest.approx(1793438042.35, abs=0.001)


def test_the_writer_keeps_comments_and_order_and_refuses_what_would_not_load(tmp_path):
    original = """# my file
[memory]
free_gb = 20  # the floor

# notes
[sources.notes]  # the notes
kind = "folder"  # kind
path = "~/notes"
keep = { max_gb = 5 }  # cap

[sources.agent]
kind = "push"
"""
    (tmp_path / "mms.toml").write_text(original, encoding="utf-8")
    configuration.set_source(tmp_path, "notes", {"kind": "folder", "path": "~/notes", "glob": "**/*.md",
                                                 "keep": {"max_gb": 6, "when_full": "thin"}})
    configuration.set_source(tmp_path, "mic", {"kind": "microphone", "device": "Mic", "keep": {"max_gb": 2}})
    text = (tmp_path / "mms.toml").read_text(encoding="utf-8")
    for kept in ("# my file", "free_gb = 20  # the floor", "# notes", "[sources.notes]  # the notes", '# kind', "# cap",
                 "[sources.agent]"):
        assert kept in text, kept
    assert text.index("[sources.notes]") < text.index("[sources.agent]") < text.index("[sources.mic]")
    loaded = configuration.load(tmp_path)
    assert loaded.sources["notes"].keep.max_gb == 6 and loaded.sources["notes"].get("glob") == "**/*.md"
    assert configuration.remove_source(tmp_path, "mic") is True and configuration.remove_source(tmp_path, "mic") is False
    assert "[sources.mic]" not in (tmp_path / "mms.toml").read_text(encoding="utf-8")
    before = (tmp_path / "mms.toml").read_bytes()
    with pytest.raises(ValueError):
        configuration.set_source(tmp_path, "x", {"kind": "folder"})  # no path
    assert (tmp_path / "mms.toml").read_bytes() == before and not list(tmp_path.glob("*.part")), "nothing written"
    (tmp_path / "mms.toml").write_text("[sources.x\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        configuration.set_source(tmp_path, "y", {"kind": "push"})


def test_envs_is_an_absolute_path_or_starts_with_a_tilde_and_a_relative_one_is_refused():
    assert load_text("h", f'[decider]\nenvs = "{ENVS}"').decider_envs == ENVS
    assert load_text("h", '[decider]\nenvs = "~/models/envs"').decider_envs == "~/models/envs"
    for relative in ("envs", "../envs", "./models/envs"):
        with pytest.raises(ConfigError, match="envs is an absolute path"):
            load_text("h", f'[decider]\nenvs = "{relative}"')


def test_a_decider_url_is_a_loopback_address_with_a_port_and_envs_is_read():
    for url, host in (("http://127.0.0.1:8741", "127.0.0.1"), ("http://localhost:8741", "127.0.0.1"),
                      ("http://[::1]:8741", "::1"), ("http://LOCALHOST:1", "127.0.0.1"), ("http://127.0.0.1:65535", "127.0.0.1")):
        config = load_text("h", f'[decider]\nmodel = "C:/models/h2o"\nurl = "{url}"\nenvs = "{ENVS}"\n')
        assert (config.decider_url, config.decider_envs) == (url, ENVS)
        assert configuration.loopback(url) == (host, int(url.rpartition(":")[2]))
    defaults = load_text("h", '[decider]\nmodel = "x"')
    assert (defaults.decider_url, defaults.decider_envs) == (None, None)
    for url in ("http://192.168.1.5:8741", "http://example.com:80", "https://127.0.0.1:8741", "http://127.0.0.1",
                "http://localhost.evil.com:1", "http://u:p@127.0.0.1:1", "http://127.0.0.1:1/v1", "http://127.0.0.1:1/",
                "http://127.0.0.1:1?a=b", "http://127.0.0.1:1#x", "http://127.0.0.1:0", "http://127.0.0.1:99999",
                "http://127.0.0.1:x", "http://0.0.0.0:8741", "http://[::]:8741", "127.0.0.1:8741", "ftp://127.0.0.1:1",
                "http://@127.0.0.1:1", " http://127.0.0.1:1", ""):
        with pytest.raises(ValueError, match="loopback address with a port"):
            configuration.loopback(url)
        with pytest.raises(ConfigError, match=r"\[decider\] url"):
            load_text("h", f'[decider]\nurl = "{url}"')


def test_set_decider_keeps_the_file_and_sets_or_removes_the_url(tmp_path):
    original = """# my file
[memory]
free_gb = 20  # the floor

[decider]
model = "C:/models/d1-Omni-2B"  # the default
idle_s = 300  # keep it a while

[sources.agent]
kind = "push"
"""
    (tmp_path / "mms.toml").write_text(original, encoding="utf-8")
    configuration.set_decider(tmp_path, model="C:/models/h2o-lightning-4b", url="http://127.0.0.1:8741")
    text = (tmp_path / "mms.toml").read_text(encoding="utf-8")
    for kept in ("# my file", "free_gb = 20  # the floor", "idle_s = 300  # keep it a while", "[sources.agent]"):
        assert kept in text, kept
    config = configuration.load(tmp_path)
    assert (config.decider_model, config.decider_url, config.decider_idle_s) == (
        "C:/models/h2o-lightning-4b", "http://127.0.0.1:8741", 300)
    configuration.set_decider(tmp_path, model="C:/models/d1-Omni-2B", url=None)
    text = (tmp_path / "mms.toml").read_text(encoding="utf-8")
    assert "url" not in text and "idle_s = 300  # keep it a while" in text and "[sources.agent]" in text
    assert configuration.load(tmp_path).decider_model == "C:/models/d1-Omni-2B" and configuration.load(tmp_path).decider_url is None
    configuration.set_decider(tmp_path, model="C:/models/d1-Omni-2B", url=None)  # nothing to remove, nothing to change
    assert (tmp_path / "mms.toml").read_text(encoding="utf-8") == text

    before = (tmp_path / "mms.toml").read_bytes()
    with pytest.raises(ValueError, match="loopback"):
        configuration.set_decider(tmp_path, model="C:/models/h2o", url="http://192.168.1.5:8741")
    assert (tmp_path / "mms.toml").read_bytes() == before and not list(tmp_path.glob("*.part")), "nothing written"
    (tmp_path / "mms.toml").write_text("[decider\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        configuration.set_decider(tmp_path, model="C:/models/d1-Omni-2B", url=None)
    assert (tmp_path / "mms.toml").read_text(encoding="utf-8") == "[decider\n", "a file that does not parse is left alone"

    fresh = tmp_path / "fresh"
    fresh.mkdir()  # no file at all: the table is made
    configuration.set_decider(fresh, model="C:/models/d3-edge", url=None)
    assert configuration.load(fresh).decider_model == "C:/models/d3-edge"


# the daemon

def folder_toml(path: Path) -> str:
    return f'[sources.notes]\nkind = "folder"\npath = "{path.as_posix()}"\nglob = "*.md"\nrescan_s = 0.2\n'


def test_source_set_adds_a_folder_source_the_running_daemon_ingests_and_remove_keeps_its_history(home, served, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("alpha")
    write_config(home, "# the owner's\n[memory]\nloop_s = 0.1\nfree_gb = 0\n")
    api = served().api
    pid = api.call("status")["pid"]
    reply = api.call("source_set", name="notes", kind="folder", path=str(notes), glob="*.md", max_gb=1,
                     caller={"project": PROJECT, "project_name": "mine", "project_root": str(tmp_path),
                             "session": "s-0123456789ab", "client": "test"})
    added = reply["source"]
    assert reply["pending_reload"] is True and added["state"] == "pending_reload", "the daemon has not read it yet"
    assert added["name"] == "notes" and added["kind"] == "folder" and added["keep"]["max_gb"] == 1
    assert added["project"] == "mine" and "approve" not in added
    store = Store(home)
    item = wait_for(lambda: store.current_version("notes", "a.md"), what="the new source to be ingested")
    assert item.project == PROJECT, "a folder source gets the caller's project"
    assert api.call("status")["pid"] == pid, "no restart"
    assert "# the owner's" in (home / "mms.toml").read_text(encoding="utf-8")
    assert sources(api.call("status"))["notes"]["state"] == "running"

    (notes / "b.md").write_text("beta")
    wait_for(lambda: store.current_version("notes", "b.md"), what="a file of the new source")
    removed = api.call("source_set", name="notes", remove=True)
    assert removed["source"] == {"name": "notes", "removed": True} and removed["pending_reload"] is True
    time.sleep(0.5)
    (notes / "c.md").write_text("gamma")
    time.sleep(1)
    assert store.current_version("notes", "c.md") is None, "removing stops it"
    assert len(store.items_of("notes")) == 2, "and keeps its history"
    assert sources(api.call("status"))["notes"]["kind"] is None, "history outside every cap is still listed"
    assert api.failure("source_set", name="notes", remove=True).code == "not_found"


def test_a_hand_edit_is_picked_up_a_bad_one_is_refused_and_a_fix_clears_it(home, served, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("alpha")
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n")
    api = served().api
    store = Store(home)
    good = "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + folder_toml(notes)
    (home / "mms.toml").write_text(good, encoding="utf-8")
    wait_for(lambda: store.current_version("notes", "a.md"), what="the hand edit to be picked up")
    assert api.call("status")["config_error"] is None

    (home / "mms.toml").write_text(good + '\n[sources.broken]\nkind = "telepathy"\n', encoding="utf-8")
    status = wait_for(lambda: (s := api.call("status"))["config_error"] and s, what="config_error")
    assert "telepathy" in status["config_error"] and "notes" in sources(status), "the last good config is kept"
    (notes / "b.md").write_text("beta")
    wait_for(lambda: store.current_version("notes", "b.md"), what="the last good config to go on working")
    error = api.failure("source_set", name="x", kind="folder", path=str(notes), max_gb=1)
    assert (error.status, error.code) == (409, "config_error") and "telepathy" in error.error["message"]
    assert api.failure("source_set", name="notes", remove=True).code == "config_error"

    (home / "mms.toml").write_text(good, encoding="utf-8")
    wait_for(lambda: api.call("status")["config_error"] is None, what="the fixed file to be read")
    assert api.call("source_set", name="x", kind="folder", path=str(notes), max_gb=1)["source"]["name"] == "x"


def test_a_daemon_that_starts_with_a_bad_file_runs_with_no_sources_and_says_why(home, daemon):
    write_config(home, '[sources.x]\nkind = "telepathy"\n', approve=False)
    one = daemon()
    status = one.api.call("status")
    assert status["running"] is True and status["sources"] == [] and "telepathy" in status["config_error"]
    assert "running with no sources" in one.text()
    write_config(home, '[sources.x]\nkind = "push"\n')
    wait_for(lambda: one.api.call("status")["config_error"] is None, what="the file to be fixed")
    assert [s["name"] for s in one.api.call("status")["sources"]] == ["x"]


def test_source_set_refuses_what_would_make_the_janitor_delete_history_or_change_the_owners_tables(home, served, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + folder_toml(notes)
                 + '[sources.manual]\nkind = "ffmpeg"\npreset = "audio"\ninput = ["-i", "x"]\n'
                 '[sources.agent]\nkind = "push"\n[sources.job]\nkind = "program"\nkeep = { max_gb = 1 }\n', approve=False)
    store = Store(home)
    now = time.time()
    store.add_records("ghost", [(now - 10 * 86400 + i, json.dumps({"n": i, "pad": "p" * 200})) for i in range(100)])
    api = served().api
    for name in ("manual", "agent"):
        error = api.failure("source_set", name=name, kind="program", max_gb=1)
        assert (error.status, error.code, error.field) == (409, "wrong_kind", "name"), name
    assert api.failure("source_set", name="manual", remove=True).code == "wrong_kind"
    assert api.failure("source_set", name="notes", kind="log", path=str(notes), max_gb=1).code == "wrong_kind"
    held = store.source_bytes()["ghost"]  # the history of a source whose table was removed
    error = api.failure("source_set", name="ghost", kind="program", max_gb=held / 1e9 / 2)
    assert (error.status, error.code, error.field) == (409, "would_delete", "max_gb")
    error = api.failure("source_set", name="ghost", kind="program", max_gb=1, max_days=5)
    assert (error.code, error.field) == ("would_delete", "max_days")
    ok = api.call("source_set", name="ghost", kind="program", max_gb=held / 1e9 * 2, max_days=30)["source"]
    assert ok["keep"] == {"max_gb": held / 1e9 * 2, "when_full": "drop_oldest", "max_days": 30}
    loaded = configuration.load(home)
    assert loaded.sources["ghost"].keep.max_days == 30 and loaded.sources["manual"].kind == "ffmpeg"


def test_source_set_changes_the_fields_given_and_keeps_the_rest_and_never_lowers_a_cap(home, served, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    mine, theirs = "git:0123456789abcdef", "git:fedcba9876543210"
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + folder_toml(notes)
                 + 'keep = { max_gb = 2, when_full = "thin", max_days = 30 }\ncopy_max_mb = 7\n'
                 f'project = "{mine}"\n[sources.mic]\nkind = "microphone"\ndevice = "Some Mic"\nsegment_s = 5\n'
                 'keep = { max_gb = 1, when_full = "thin" }\n[sources.bare]\nkind = "program"\n', approve=False)
    api = served().api
    caller = {"project": theirs, "project_name": "theirs", "project_root": "/y", "session": "s-0123456789ab", "client": "t"}
    reply = api.call("source_set", name="notes", max_gb=3, caller=caller)  # only the cap is given
    table = configuration.load(home).sources["notes"]
    assert table.keep == configuration.Keep(3.0, "thin", 30.0), "when_full and max_days were not given: they stay"
    assert (table.get("glob"), table.get("copy_max_mb"), table.get("rescan_s")) == ("*.md", 7, 0.2), "every option stays"
    assert table.project == mine, "a later edit by another project does not move it"
    assert reply["source"]["kind"] == "folder" and reply["pending_reload"] is True
    api.call("source_set", name="notes", glob="*.txt", when_full="drop_oldest")
    table = configuration.load(home).sources["notes"]
    assert table.get("glob") == "*.txt" and table.keep == configuration.Keep(3.0, "drop_oldest", 30.0)
    api.call("source_set", name="notes", max_gb=3, max_days=30)  # the same: allowed
    api.call("source_set", name="notes", max_days=60)  # higher: allowed
    for field, value in (("max_gb", 2.9), ("max_days", 59)):
        error = api.failure("source_set", name="notes", **{field: value})
        assert (error.status, error.code, error.field) == (409, "would_delete", field)
        assert "mms.toml" in error.error["hint"], "the owner lowers a cap"
    assert api.failure("source_set", name="bare", max_gb=1).code == "would_delete", "a source with no cap keeps none"
    assert api.failure("source_set", name="notes", kind="log").code == "wrong_kind"
    assert api.failure("source_set", name="nothing", max_gb=1).field == "kind", "a new source needs its kind and cap"
    assert api.failure("source_set", name="nothing", kind="program").field == "max_gb"
    assert configuration.load(home).sources["notes"].keep.max_gb == 3, "refusals changed nothing"
    new = api.call("source_set", name="fresh", kind="folder", path=str(notes), max_gb=1, caller=caller)
    assert configuration.load(home).sources["fresh"].project == theirs, "set when the source is made"
    assert new["source"]["state"] == "pending_reload"


def test_a_cap_an_fps_or_a_name_edit_leaves_the_approval_standing_and_a_device_edit_does_not(home, served):
    screen = '[sources.screen]\nkind = "screen"\ndevice = "1"\nfps = 2\nkeep = { max_gb = 4 }\n'
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + screen, approve=False)
    consent.record(home, "screen", configuration.load(home).sources["screen"].what())
    api = served().api
    fingerprint = consent.load(home)["screen"]["fingerprint"]
    for change in ({"max_gb": 8}, {"fps": 5}, {"when_full": "thin"}):
        api.call("source_set", name="screen", **change)
        source = configuration.load(home).sources["screen"]
        assert consent.approved(home, "screen", source.what()), change
    assert configuration.load(home).sources["screen"].get("device") == "1" and \
        configuration.load(home).sources["screen"].get("fps") == 5, "the device stays when only a cap is given"
    api.call("source_set", name="screen", device="2")
    assert not consent.approved(home, "screen", configuration.load(home).sources["screen"].what())
    assert consent.load(home)["screen"]["fingerprint"] == fingerprint, \
        "the API wrote nothing: the old approval is simply no longer this source's"


def test_removing_a_source_takes_its_approval_with_it(home, served, tmp_path):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n", approve=False)
    api = served().api
    api.call("source_set", name="mic", kind="microphone", device="Some Mic", max_gb=1)
    api.call("source_set", name="job", kind="command", command=["python", "-c", "print(1)"], max_gb=1)
    wait_for(lambda: sources(api.call("status")).get("mic", {}).get("state") == "needs_approval", what="the source")
    consent.record(home, "mic", configuration.load(home).sources["mic"].what())  # what `mms approve` does
    consent.record(home, "job", configuration.load(home).sources["job"].what())
    consent.record(home, "gone", "microphone: A Mic That Is In No Table")
    wait_for(lambda: "gone" not in consent.load(home),
             what="an approval of a name that is in no table to be pruned")
    api.call("source_set", name="mic", remove=True)
    wait_for(lambda: "mic" not in consent.load(home),
             what="the removed source's approval to go")
    assert "job" in consent.load(home), "another source's stays"
    api.call("source_set", name="mic", kind="microphone", device="Some Mic", max_gb=1)  # the same name, the same device
    wait_for(lambda: sources(api.call("status")).get("mic", {}).get("state") == "needs_approval", what="the new source")
    assert not consent.approved(home, "mic", configuration.load(home).sources["mic"].what()), "it starts unapproved"
    (home / "mms.toml").write_text("[memory]\nloop_s = 0.1\nfree_gb = 0\n", encoding="utf-8")  # by hand
    wait_for(lambda: not consent.load(home), what="a hand removal to prune too")


def test_the_fingerprint_is_the_json_of_the_argument_list():
    one = load_text("h", '[sources.j]\nkind = "command"\ncommand = ["python", "a b"]').sources["j"]
    two = load_text("h", '[sources.j]\nkind = "command"\ncommand = ["python a", "b"]').sources["j"]
    assert one.what() == 'command: ["python", "a b"]' and one.what() != two.what(), "a joined string would read alike"
    ffmpeg = load_text("h", '[sources.f]\nkind = "ffmpeg"\npreset = "audio"\ninput = ["-i", "x y"]').sources["f"]
    assert ffmpeg.what() == 'ffmpeg: ["-i", "x y"]'


def test_a_folder_source_or_a_remember_never_reads_the_mms_home(home, served, tmp_path):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n"
                       f'[sources.everything]\nkind = "folder"\npath = "{home.parent.as_posix()}"\nrescan_s = 0.2\n'
                       f'[sources.inside]\nkind = "log"\npath = "{(home / "streams").as_posix()}"\n'
                       f'[sources.fine]\nkind = "folder"\npath = "{tmp_path.as_posix()}/fine"\nrescan_s = 0.2\n')
    (tmp_path / "fine").mkdir()
    (tmp_path / "fine" / "a.md").write_text("alpha")
    (home / "secret.txt").write_text("in the home")
    api = served().api
    state = sources(api.call("status"))
    for name in ("everything", "inside"):
        assert state[name]["state"] == "error" and "MultimediaSearch home" in state[name]["error"], name
    wait_for(lambda: Store(home).current_version("fine", "a.md"), what="a source outside the home to work")
    time.sleep(0.8)
    assert Store(home).items_of("everything") == [], "a folder at the home's parent scanned nothing"
    for path in (home, home.parent, home / "secret.txt", home / "streams"):
        error = api.failure("remember", path=str(path))
        assert (error.status, error.code, error.field) == (400, "bad_request", "path"), path
    for kind in ("folder", "log"):
        error = api.failure("source_set", name=f"x{kind}", kind=kind, path=str(home.parent), max_gb=1)
        assert (error.code, error.field) == ("bad_request", "path"), kind
    assert api.call("source_set", name="ok", kind="folder", path=str(tmp_path / "fine"), max_gb=1)["pending_reload"]
def test_source_set_checks_its_parameters_and_never_approves(home, served, tmp_path):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n")
    api = served().api
    first = api.failure
    assert first("source_set", name="x", kind="folder", path=str(tmp_path)).field == "max_gb", "a cap is required"
    assert first("source_set", name="x", max_gb=1).field == "kind", "and a kind, to add a source"
    assert first("source_set", name="x", kind="ffmpeg", max_gb=1).field == "kind", "owner's kinds are not written"
    assert first("source_set", name="x", kind="push", max_gb=1).field == "kind"
    assert first("source_set", name="x", kind="folder", max_gb=1).code == "bad_request", "a folder needs a path"
    assert first("source_set", name="x", kind="program", max_gb=1, path=str(tmp_path)).field == "path"
    assert first("source_set", name="x", kind="folder", path="relative/dir", max_gb=1).field == "path"
    assert first("source_set", name="bad name", kind="program", max_gb=1).field == "name"
    assert first("source_set", name="x", kind="program", max_gb=0).field == "max_gb"
    assert first("source_set", name="x", kind="program", max_gb=1, when_full="never").field == "when_full"
    assert first("source_set", name="x", kind="program", max_gb=1, project="git:aaaaaaaaaaaaaaaa").field == "project"
    for field in ("approved", "approve", "consent", "fingerprint"):
        assert first("source_set", name="x", kind="program", max_gb=1, **{field: True}).field == field
    assert first("source_set", name="x", remove=True, kind="program").field == "kind"
    before = (home / "mms.toml").read_text(encoding="utf-8")
    for name, kwargs in (("screen", {"kind": "screen"}), ("mic", {"kind": "microphone", "device": "Some Mic"}),
                         ("cam", {"kind": "camera", "device": "Some Cam"}),
                         ("job", {"kind": "command", "command": ["python", "-c", "print(1)"]})):
        entry = api.call("source_set", name=name, max_gb=1, **kwargs)["source"]
        assert entry["state"] == "pending_reload" and entry["approve"] == approve_hint(name), name
    assert not (home / "consent.json").exists(), "no call approved anything"
    assert before != (home / "mms.toml").read_text(encoding="utf-8")
    time.sleep(0.6)
    status = sources(api.call("status"))
    assert {n: status[n]["state"] for n in ("screen", "mic", "cam", "job")} == {n: "needs_approval" for n in ("screen", "mic", "cam", "job")}
    assert not list((home / "streams").glob("*/*")), "nothing opened a device"
    assert not (home / "consent.json").exists()


def globs_that_leave_their_path(home: Path) -> list[str]:
    """Patterns that reach `home` from a folder beside it, as a `glob` must never be able to."""
    out = ["../home/daemon.json", "..\\home\\daemon.json", "**/../home/daemon.json", "../*/daemon.json",
           (home / "daemon.json").as_posix()]
    return out + ["C:daemon.json"] if os.name == "nt" else out


def test_a_glob_that_leaves_its_path_is_refused_by_source_set_and_by_a_hand_edit(home, served, tmp_path):
    """`glob` runs under `path`, and the home guard looks at `path` alone: `../home/daemon.json` made the daemon's token a
    readable item."""
    notes = tmp_path / "notes"  # beside the home
    notes.mkdir()
    (notes / "a.md").write_text("alpha")
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + folder_toml(notes))
    api = served().api
    store = Store(home)
    wait_for(lambda: store.current_version("notes", "a.md"), what="the source to be read")
    good = (home / "mms.toml").read_text(encoding="utf-8")
    for glob in globs_that_leave_their_path(home):
        for body in ({"name": "notes"}, {"name": "more", "kind": "folder", "path": str(notes), "max_gb": 1},
                     {"name": "logs", "kind": "log", "path": str(notes), "max_gb": 1}):
            error = api.failure("source_set", glob=glob, **body)
            assert (error.status, error.code) == (400, "bad_request") and "glob" in error.error["message"], (glob, body)
        assert (home / "mms.toml").read_text(encoding="utf-8") == good, f"{glob}: a refusal writes nothing"
    assert api.call("source_set", name="notes", glob="**/*.md")["source"]["name"] == "notes", "a glob inside is fine"

    (home / "mms.toml").write_text(good, encoding="utf-8")
    for glob in globs_that_leave_their_path(home):
        (home / "mms.toml").write_text(good.replace('glob = "*.md"', f"glob = '{glob}'"), encoding="utf-8")
        status = wait_for(lambda: (s := api.call("status"))["config_error"] and s, what=f"{glob} to be refused")
        assert "glob" in status["config_error"] and "notes" in sources(status), "the last good config is kept"
        assert [item.name for item in store.items_of("notes")] == ["a.md"], "nothing outside the folder was read"
        (home / "mms.toml").write_text(good, encoding="utf-8")
        wait_for(lambda: api.call("status")["config_error"] is None, what="the good file to be read again")


BAD_FILES = [  # what an editor or a shell can leave in mms.toml: none of it is TOML text that a loader can trust
    ("utf-16", "[memory]\nfree_gb = 1\n".encode("utf-16"), "is not UTF-8 text"),  # PowerShell 5.1's `>`
    ("cp1252", b'[sources.x]\nkind = "folder"\npath = "caf\xe9"\n', "is not UTF-8 text"),
    ("sources-number", b"sources = 1\n", "sources is a table"),
    ("memory-number", b"memory = 1\n", r"\[memory\] is a table"),
    ("decider-text", b'decider = "x"\n', r"\[decider\] is a table"),
    ("inf", b"[memory]\njobs = inf\n", "finite number"),
    ("nan", b"[memory]\nfree_gb = nan\n", "finite number"),
    ("huge", b"[memory]\nfree_gb = 1" + b"0" * 400 + b"\n", "finite number"),
    ("cap-inf", b'[sources.x]\nkind = "program"\nkeep = { max_gb = inf }\n', "finite number"),
]
THE_FOUR = ["utf-16", "sources-number", "memory-number", "inf"]  # what an editor, a shell or a hand is likely to leave


@pytest.mark.parametrize("data, why", [(data, why) for _, data, why in BAD_FILES], ids=[name for name, _, _ in BAD_FILES])
def test_every_way_a_file_can_fail_is_a_config_error_for_the_loader_and_the_writer(home, data, why):
    (home / "mms.toml").write_bytes(data)
    with pytest.raises(ConfigError, match=why):
        configuration.load(home)
    with pytest.raises(ConfigError, match=why):
        configuration.set_source(home, "y", {"kind": "program", "keep": {"max_gb": 1}})
    assert (home / "mms.toml").read_bytes() == data, "a file that cannot be read is left as it is"


def the_four() -> list:
    return [pytest.param(data, id=name) for name, data, _ in BAD_FILES if name in THE_FOUR]


@pytest.mark.parametrize("data", the_four())
def test_a_running_daemon_keeps_its_last_good_config_when_a_file_cannot_be_read(home, served, tmp_path, data):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("alpha")
    good = "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + folder_toml(notes)
    write_config(home, good)
    api = served().api
    store = Store(home)
    wait_for(lambda: store.current_version("notes", "a.md"), what="the source to be read")
    (home / "mms.toml").write_bytes(data)
    status = wait_for(lambda: (s := api.call("status"))["config_error"] and s, what="config_error")
    assert status["running"] is True and "notes" in sources(status), "the last good config is kept"
    (notes / "b.md").write_text("beta")
    wait_for(lambda: store.current_version("notes", "b.md"), what="the last good config to go on working")
    (home / "mms.toml").write_text(good, encoding="utf-8")
    wait_for(lambda: api.call("status")["config_error"] is None, what="the fixed file to be read")


@pytest.mark.parametrize("data", the_four())
def test_a_daemon_that_starts_with_a_file_it_cannot_read_runs_with_no_sources_and_says_why(home, daemon, data):
    (home / "mms.toml").write_bytes(data)
    one = daemon()
    status = one.api.call("status")
    assert status["running"] is True and status["sources"] == [] and status["config_error"]
    assert "running with no sources" in one.text()
    (home / "mms.toml").write_text('[sources.x]\nkind = "push"\n', encoding="utf-8")
    wait_for(lambda: one.api.call("status")["config_error"] is None, what="the file to be fixed")
