# MultimediaSearch: testbed

How MultimediaSearch is tested. Goals are in [VISION.md](VISION.md); the design is in [DESIGN.md](DESIGN.md).

## The suite

`uv run pytest` runs everything except `gpu` and `codex` tests on four workers (pytest-xdist). It should take about two minutes or
less. Keep it at four, and run one suite at a time: more load than that, beside a GPU run, crashed the laptop once.
- **End-to-end tests matter most:**
  - the daemon as a process, reached through its API;
  - `mms mcp` as a process, driven over stdio by the MCP SDK's client, as Claude Code and Codex drive it;
  - the `mms` CLI;
  - real ffmpeg and real files.

  Unit tests are few, kept only where a rule is easier to pin alone (thinning order, time parsing, the project ID's
  URL normalizing).
- **Parallel-safe:**
  - every test gets its own home in `tmp_path` (the `home` fixture in `tests/conftest.py`);
  - every daemon binds a port the OS picks and writes it to its own `daemon.json`, so there are no fixed ports;
  - there are no shared folders: the corpus (`tests/corpus.py`) is made once per worker in that worker's own temporary
    folder and only read, and a test that watches or changes files copies them into its own `tmp_path`.
- **Two ways to run a daemon:** the `daemon` fixture starts `mms start` as a process (for what only a process shows: a
  kill, Ctrl+C, a restart), and the `served` fixture runs the same `Daemon` in a thread of the test process, with the
  same API on its own port and no interpreter to start. Tests that need many daemons or a quick one use `served`, and
  the adapter's process connects to it as to any daemon. A test that lets an adapter or the CLI start the daemon
  itself uses `detached`, which stops it at the end.
- **Never the real agent setup:** install tests set `USERPROFILE`, `HOME`, `APPDATA`, `CODEX_HOME` and
  `XDG_CONFIG_HOME` to folders in `tmp_path`, and pass `--claude-exe` a fake `claude` (a script that records its
  arguments). Nothing reads or writes the real `~/.claude`, `~/.codex` or Startup folder. The install test lists every
  file it finds changed and checks that each is under `tmp_path`.
- **No devices:** capture tests use kind `ffmpeg` with ffmpeg's synthetic inputs instead of a screen, camera or
  microphone:
  - `-f lavfi -i testsrc2=size=640x360:rate=2` for video;
  - `-re -f lavfi -i sine=frequency=440` for audio.

  Short segments (`segment_s = 1` or `2`) keep runs to seconds. The named kinds are tested through what they expand to
  (`capture.opens`), their consent, and `status`, never by opening a device. Device listings
  (`sources`, `status --devices`) run a fake `ffmpeg` that prints two devices (`fake_ffmpeg` in `tests/conftest.py`).
- **Real files:** `tests/corpus.py` makes, offline in a second or two, files as phones, cameras, browsers, Office and
  Windows write them: pictures EXIF-rotated, transparent, 16-bit and animated; sound in every common container; video
  with a display matrix, a variable frame rate and a soundtrack; text in UTF-8, UTF-16 and cp1252; broken files. Each
  entry says what mms must record and what d1 must get, and `tests/test_corpus.py` checks both; `tests/test_day.py` asks a
  synthetic day's questions. Its `documents/` part holds what d1 cannot read: PDFs (text, scanned, locked), OOXML, ODF,
  EPUB, OLE and archive files, a misnamed PDF, and RTF, ICS, VCF and EML as text. `legacy.doc`, `meeting.msg`,
  `photos.7z` and the HEIC are stubs that only sniff as their type (marked `stub` in the corpus): a decider that
  declares OLE, 7z or HEIC is tested for the plumbing alone, not for reading them.
- **Consent in tests:** a test has no terminal, so it records approvals with `consent.record`, the function
  `mms approve` calls once the owner has typed the name. `mms approve` itself is tested for refusing without a
  terminal. The `write_config(home, text)` helper approves every source in the text that needs it, unless told not to.
- **No GPU:** jobs use a test decider, set by `[decider] model = "python:<path>"`. The code has no default model (`mms install`
  sets one), so a test that forgets it fails with `decider` instead of loading d1. Two exist, with different `About`s (DESIGN.md §13.3), so
  a test shows that mms follows the decider's facts and not d1's:
  - `tests/fake_decider.py` has d1's facts and refuses what d1 would not be handed (a file, sound not at 16 kHz).
  - `tests/other_decider.py` differs in every field mms uses: it takes PDF, Office, ODF, EPUB and HEIC whole and
    reads their text with the stdlib, hears at 8 kHz, answers three question types, and caps pictures at 102,400
    pixels. It logs what it was handed to `other-decider.jsonl`.
  - `tests/test_narrow_deciders.py` writes two more into tmp files, an eyes-only (`reads={"video", "image"}`) and an
    ears-only (`reads={"audio"}`), each otherwise `fake_decider.py`.

  The five families need no weights either, by fakes that have the real call signatures:
  - `tests/fake_family.py` is a worker's family file. A real `Worker` runs it in a real child process (`sys.executable`), so
    the frames, the deadlines, crashes, hangs and stray prints are the real ones. It reads `fake.json` in the folder
    (`delay_s`, `crash_on`, `hang_on`, `print_on_load`), echoes the size and SHA-256 of the bytes of each picture, sound and
    file it is handed, and answers like the fake decider.
  - `tests/fake_shim.py` is H2O's server: a stdlib HTTP server on 127.0.0.1 and a free port. `/health` answers 503 and then
    200 with the model; `/v1/systemone` decodes each PNG data URI, records the SHA-256 of its pixels, refuses more than 4
    images with 422, and answers in the card's shapes.
  - `test_family_d1.py`, `_d1_3b.py`, `_d3.py` and `_pplx.py` run each real `worker/<family>.py` in the test's own process
    against a fake `torch` and model (d1; d1-3B's model takes `images=`, and fake `runner` and `hybrid` modules of its code give
    its picture cap and Windows' `_fused`), a fake `transformers` whose `AutoModel.from_pretrained` returns a model with d3's keyword-only
    `system_one` (d3), and a fake `pplx_decider` package in a temporary folder, so the real `sys.path` insert and import run
    (Perplexity). The translation is tested from canned native answers in the shapes of each model's code (`d3_runtime`,
    `pplx_decider.protocol`, H2O's README and `box.py`).
  - `model_folder(parent, name)` in `tests/conftest.py` writes the few files each backend's `claims` reads, with the real
    values, for `d1-Omni-2B`, `d1-2B`, `d1-3B`, `d3-edge`, `d3-mini`, `d3-flash`, `h2o-lightning-4b`,
    `pplx-decider-flash-v1.2` and `unclaimed`, and an empty `model.safetensors` for all but the last, so a folder reads as
    downloaded and whole (never real weights); `prepared_env` writes an env's `mms-env.json`.

  The fake decider decides from simple, known content:
  - a picture's colour;
  - a sound's loudness;
  - a sound's pitch (`pitch:<hz>`: at least 30% of its energy within 3% of that frequency);
  - a picture's top-left quarter (`corner:<colour>`: it is upright), its mean colour (`colour:<colour>`), and whether it
    has ink (`ink`);
  - a word in the text.

  So a test knows which windows must hit. A `slow:<s>` word in a question makes it sleep that long per window, for the
  tests of waiting, cancelling and turn-taking; a `load:<s>` word does so on the first call only, as d1 loading would.
  It appends what each call was given (picture sizes and corner colours, each sound's RMS, rate and loudest second, the
  questions, the state's length and its counts of NUL, U+FFFD, ESC and CR) to `fake-decider.jsonl` in the home, so a
  test checks what d1 would have read.
- **No download and no real uv:** nothing in a test reaches Hugging Face or builds a real env.
  - `tests/conftest.py`'s autouse `no_hugging_face` makes `import huggingface_hub` fail in the test's process, and `env()` gives
    every process a test starts `HF_HUB_OFFLINE=1`. A test of a download fakes the Hub with `tests/fake_hub.py`: its `FakeHub`
    replaces `fetch.hub_info` and `fetch.hub_download` (the one fake that `test_fetch.py`, `test_model_cli.py` and
    `test_install.py` share), records the calls, and leaves a `model_folder` in the folder as a download would. The tests assert
    those calls: a model that is already there makes none. `test_fetch.py` tries `_hub` against a fake `huggingface_hub` module.
  - Its autouse `no_real_uv` points `prepare.uv_command` at a script that exits `no real uv in tests`, so a `use` or `prepare`
    that reaches an env build fails instead of downloading gigabytes. `test_model_cli.py`'s `fake_uv` puts a script that records
    what it is asked in its place (`test_prepare_without_uv_says_so` restores the real lookup of `uv` on `PATH`).
- **Programs:** spool tests on the daemon side write spool files exactly as DESIGN.md §8.2 gives them. The client's own
  tests read back what it wrote against the same section, and `tests/test_seams.py` runs the client and the daemon together.
- **Wait on state, never on sleeps:** `wait_for` polls a condition, and a failure prints the daemon's log.

## GPU tests

`uv run pytest -m gpu -n 0 -s tests/test_gpu.py -k <folder>` runs once for every subfolder of `MMS_MODELS` (which these
tests require: with it unset none runs) that a backend claims, with one model loaded at a time (a module-scoped fixture). A
model that is not ready is skipped, saying what to do:
- weights still downloading (`fetch.unfinished`: a file the Hub's record in `.cache/huggingface/trees/` lists that is missing
  or short, or a shard missing from `model.safetensors.index.json`);
- its own env not prepared (`mms model prepare <name>`); d1 runs in mms's own env, so a pin missing there fails the run;
- H2O's `/health` not answering (start `serve.sh` in WSL, DESIGN.md §13.3).

Each model is asked about a generated picture (a red square, none, and a blue one), a short text, generated beeps and silence
(only if it hears), a box (only if it answers box), and one question of every type it takes, each answer parsed by
`ask.read`. It also prints:
- the load time (a fresh worker, torch and the weights);
- the median of 5 warm calls at 640×360, 1280×720 and 1080p with 1 and 4 questions, for 2,000 characters, and 5 s of sound
  where heard, with the `About` numbers they would set;
- the worker's peak VRAM (`stats`; a served model has none of ours).

After the last test it unloads the model and checks `nvidia-smi` is back within 300 MB of where it was before the load.

Only d1-Omni-2B's answers are held to the checks; for every other model they print (the p's, the timings, the VRAM), and are the
bake-off's data (DESIGN.md §16) rather than a gate. For d1-Omni-2B the same `worker/d1.py` also runs in a plain subprocess with
no wire, and its answers must equal the worker's exactly on the synthetic cases (the wire changes nothing), and the 640×360
time through the worker is printed against the 53 ms measured in process (the aim is +5% or less).

`tests/test_gpu_content.py`, parametrized the same way, sends real content through the store and the reader to the model, each
case against its absence: a terminal's error, a sideways whiteboard photo, a portrait video's arrow, a transparent logo,
speech, a doorbell, and a phone ringing in a meeting's soundtrack. For d1-Omni-2B it checks that the answers fall on the right
side of the threshold; for the others it prints them. Models that hear nothing skip the sound cases.

It is opt-in, because loading takes 20-24 s for d1 and needs the GPU. Run it alone, with no other suite or GPU job going, with
the daemon's model unloaded (`mms status` shows the decider's state), and one model at a time (H2O's server stopped when its run
is done, to free the VRAM), only with the maintainer's go-ahead. The order for a first pass is d1-Omni-2B (the default; its
identity and timings first), d1-3B, d1-2B, d3-edge, d3-mini, Perplexity, d3-flash (the most VRAM, 17.5 GiB), then H2O.

`tests/test_gpu.py` also checks, on the CPU, that each folder's `About` agrees with the weights' own numbers (for d1,
`config.json`; for d1-3B the loaded code's picture cap is checked when it loads). The GPU tests give the same counts again after any change to a decider's code: that shows d1 is unchanged.

## Codex recall

`uv run pytest -m codex -n 0 -s` asks Codex questions about a synthetic day (`tests/day.py`) through `mms mcp` against a
daemon in a throwaway home, one `codex exec` per question (`benchmarks/codex_run.py`: mms's tools only, its instructions, the
skill and the notes of the model it runs (d1's), `--ignore-user-config`, nothing written to `~/.codex`). `-m "codex and not gpu"` asks what `find` and `look`
answer, with no decider (about 10 min); `-m "codex and gpu"` the rest with d1 (`MMS_MODELS/d1-Omni-2B`, skipped when that is unset), after the GPU tests pass (about 30 min).
Q18 asks the total on a PDF that no tool in the bench reads (d1 does not, and Codex's shell is off); "unknown" passes,
so it checks that the agent does not invent one.
It uses the maintainer's Codex plan quota: only with their go-ahead, and it stops at 60% of the week's use, read from its own sessions'
files. Codex is given only `ask, find, look, label` (`mms mcp --tools`), so it cannot remember files or change sources.

## Benchmark: with mms and without

The published results, and how to read them, are in [benchmarks/README.md](benchmarks/README.md); this section is the method.

`benchmarks/bench.py` measures how well Claude Code and Codex answer an owner's questions with mms and without it, for each decision
model, over a synthetic week (`benchmarks/haystack.py`): phone photos and clips, screen recordings, the porch camera's clips, a
hallway microphone, notes, versions of `app.toml` and three services' logs, made with PIL, numpy and ffmpeg.
Each question has a needle (a code in a note, a setting at a time, a count and a time in the logs, a sign's number and a count of
photos, an error on screen and a parcel at the door, a doorbell and a smoke alarm) or asks, in the same words, about something
that is not there. The needles are hidden among distractors of the same colour or pitch that differ in meaning (a red banner
beside the red error dialog, a brown dog beside the parcel, other beeps beside the alarm), so a colour filter, frame
differencing, a band filter or a file's size gives a shortlist and not the answer. A model is asked only about what its `About`
reads: text, records, pictures and video for all, sound for d1-Omni-2B alone. None reads documents or understands speech, so
there are none.

**How vague the time hint is** is a factor of its own, the question's `level`:
- `part`: a day and a part of it ("on Tuesday 28 October, in the afternoon");
- `day`: the day alone;
- `week`: no hint, and the question asks for the last time ("when did the smoke alarm last beep?"): the answer is the template's
  newest needle with a day of haystack after it, the earlier needles are the wrong answers, and the answer names its day;
- `time`: `config-asof` alone, which keeps its exact time.

A part of a day or a day holds one needle of its template. The report gives each arm's accuracy, median seconds, tokens and
timeouts at each level, and names the crossover: the first of part, day and week with at least 10 paired questions where mms leads
by 20 points or more (accuracy), or costs half as much per correct answer or less (seconds, or tokens) without being less accurate
(a difference of 0 or more, or an interval whose low end is above -10 points, and 5 correct answers in each arm). It says which of
the two held, and which levels have too few questions to say.

Three scales (`haystack.SCALES`; times measured on the laptop, CPU only):

| Scale | Days | Questions (without sound) | Files | Screen states (frames) | Porch frames | Build | Probe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tiny` | 2 | 19 (15) | 92 | 80 (3,498) | 232 | not rebuilt | not rebuilt |
| `pilot` | 3 | 31 (24) | 1,572 | 2,358 (81,304) | 1,442 | 11 min 23 s | 94 s |
| `standard` | 7 | 31 (24) | 3,668 | 5,556 (208,434) | 3,324 | not rebuilt | not rebuilt |

`tiny` is the offline suite's week; `pilot` has the density of `standard` over 3 days, for a first run before the full one.
Counts are of seed 7 for `tiny` and seed 1 for the others, generator 6. The pilot's build and probe are `bench.py build` end to end (write,
freeze, load, probe; `--out` and `--temp` on a second drive), with the probe passing, on a laptop that another project's heavy jobs shared, so they
are upper bounds. Generator 5 built `tiny` in 24 s (probe 4 s) and
`pilot` in 7 min (probe 68 s). `tiny` is written in about 25 s on one process with another project's 16-worker suite on the machine, and the
suite's target is 30 s to write and 8 s to load (`WRITE_S`, `LOAD_S` in `test_haystack.py`).
The screen is written as the recorder writes it (DESIGN.md §4.1, §6): a block (09:00-11:30, 13:30-16:30, 20:00-21:00) is planned as
states of 10 to 60 s and cut into segments the way ffmpeg's segment muxer cuts them, at the first state change at or after each
multiple of 60 s from the block's start. A segment starts on a state change (no still frame repeated at its start), holds each
state once, and lasts 60 s on average (10 to about 120 s): 390 files a day for about 790 states. Its frames follow the recorder
(`ddagrab` at about 2 fps with the cursor drawn, then `mpdecimate=max=119` and `-fps_mode vfr`): a frame whenever the screen changes, at
most one every half second (a tick), and at least one every 60 s; each differs from the frame before it (the plan types only visible
characters, and no more than a decoration leaves showing). mms's reader takes the first frame of each second, so a segment's
windows are at most its seconds with a frame (`windows_in` counts those, an upper bound), and the files arm sees every frame. There is a
dark taskbar with the time and the date (ISO, `2026-11-01`) at the right and a white arrow for the cursor.

What the owner does is a state's **acts**, each at a tick: a cursor `move` to a place near the last, a `type` of 1 to 3 visible characters on the
app's typing line (a full line is followed by a `scroll` of 1 line, an enter), or a `scroll` of 1 to 3 lines of the app's content box
(a scroll empties the typing line). A state has a **level** of activity, drawn as a chain over its block so that it persists: `idle` (no
acts: reading, thinking, away; idle comes in stretches of several states, a few minutes), `light` or `busy` (runs of active seconds, each with 1 or 2
acts). The app sets the mix of acts: an editor, a chat, slides and a sheet mostly type; a terminal types and scrolls (output); a browser
scrolls and moves. The error state has the activity of any other: no special case. A frame is drawn from the state's layers
(`haystack_media.screen_layers`): the app card, scrolled by the lines (the content box is rolled up, wrapping, so a scroll brings no new
colour) and typed on (text of the state's seed, in the app's palette, cleared of the needles' shades, inside a line that holds the text's
whole height), then the decorations (the dialog, the accent and the taskbar) pasted over it through an exact mask found by drawing them on a
black and a white background, so that they never move or get covered, then the clock and the cursor. A frame costs about 2.3 ms to draw and
10 ms to write (with the encoding).

**Calibrated against two real hours** of a developer's screen (2026-11-04: reading docs, then coding; 2560x1440), put through mms's screen
preset (DESIGN.md §6: 2 fps, scale to 1920, `mpdecimate=max=119`, VFR, x264 crf 30, 60 s segments) and mms's reader. The constants
(`ACTIVITY_*`, `ACTIVE_*`, `TWO_ACTS`, `KIND_ODDS`, the typed text's size and weight in `haystack_media.TYPING`) were tuned on the plan's
counts of 150 hours (seeds 1 to 3, `pilot` and `standard`, every full screen hour) and on the encoded hours of pilot seed 1 (day 0 13:30,
day 0 09:00, day 1 14:30); two hours that were not tuned on (pilot seed 2 day 1 09:00, pilot seed 3 day 2 13:30) check them. Per recorded hour:

| | real: reading + coding | real: coding | haystack: 3 tuned hours | haystack: 2 other hours | haystack: 150 plan hours |
| --- | --- | --- | --- | --- | --- |
| seconds with a frame | 1,958 | 3,016 | 2,359 | 2,107 / 2,400 | 2,360 (hours under 2,200: 27%) |
| gated windows (`still_s` 30) | 1,386 | 2,590 | 2,329 | 2,065 / 2,346 | - |
| frames kept | 3,426 | 5,413 | 4,370 (1.85 a second with a frame) | 3,885 / 4,413 | 4,380 (1.86) |
| clock minutes with 2 or fewer gated windows | 11 of 60 | 0 | 18 of 179 (10%) | 10 of 60 / 0 of 60 | 3.3% with 2 or fewer seconds with a frame (10% when the clock's minute alone is not counted) |
| frame pairs under 0.05% of the frame changed | 50% | 28% | 34% | 35% / 33% | - |
| median changed area (pixels more than 24 levels apart at 640x360) | 0.05% | 0.32% | 0.084% | 0.084% / 0.083% | - |
| 90th percentile | 8.9% | 6.5% | 7.6% | 8.5% / 7.2% | - |
| A-B-A returns (a blink) | 2% | 5% | 0% | 0% / 0% | - |

The plan hours are counted from `screen_frames` alone, so they cannot count gated windows: a minute with 2 or fewer seconds that have a
frame is rarer (3.3%) than one with 2 or fewer gated windows (the clock's minute is a frame and no window), which is what the encoded hours
show and the real ones are counted by. The haystack's hours vary as the real two do (a plan hour's seconds with a frame run from 1,400 to
2,900).

Typing and scrolling are now changes of the kind and size a real screen's are: the median change is 0.08% of the frame (text-sized, as the
real hours' 0.05 and 0.32%), a scroll is 3 to 25%, and 34% of the frame pairs are under 0.05% (28% and 50% in the real hours). The
files arm has up to 2 frames a second to wade through, as on a real recording. (ffmpeg's scene filter was run on neither recording; a
change of 0.08% of the frame is below any usual scene threshold on both.) The typed text is larger and bolder than the card's (a typed
character at the card's size changes 0.025% of the frame, a real frame 0.05 to 0.3%), and the reader's change gate turns
98% of the haystack's seconds with a frame into windows against 71 to 86% of a real screen's, since a typed character at this size always
moves a cell by more than the gate's 6; so the gated windows (2,329 an hour) are above the two real hours' mean (about 1,990) while the
seconds with a frame (2,359) are below theirs (about 2,490), which makes mms's screen asks a little dearer than on a real screen. The cursor is
12x19 px at 1280x720, about 6x9.5 at the reader's 640x360; on the dev laptop (3840x2400
at 200%, pointer size 2) the owner's own pointer is 0.67 to 1 times that at 640 px. mms's change gate is sensitive to its size: at this size
87 to 99% of moves are a change, at 0.8 times it 19 to 89%, at 0.67 times 0 to 78%. A minute of the clock alone is not a change (a few
pixels, and `still_s` ends a run at 30 s). Key frames are at a file's start and at scene cuts (x264 puts one at each state change, 1 to 4
a file), and none is forced every 10 s. **Not modelled:** the recorder forces a key frame every 10 s and cuts a segment at the first key
frame at or after each 60 s (DESIGN.md §6), so its cuts land mid-state where the haystack cuts at a state's start; a scroll
wraps and brings no new lines; there are no caret blinks. A dialog's title is 40 px on the 720 px screen, 20 px at the 640x360 the reader hands a decider and about 8 px in a
contact sheet 5 frames across, so an agent opens or crops a frame to tell an error from the red "Confirm" or a red-headed warning.
The phone's photos and clips carry their own time as a phone's do (a photo's EXIF `DateTimeOriginal` and `OffsetTimeOriginal`, a clip's
`creation_time`; the names `IMG_YYYYMMDD_HHMMSS.jpg` and `VID_...mp4` say it too), and `haystack.load` remembers them with no `t`,
refusing one that is not stored at its time from its metadata, as an owner's agent remembers a camera roll; the other sources are given
their `t` (screen, porch and microphone stand in for captured segments, notes and config are timed text). A picture is a recording,
kept once per SHA-256, so the writer refuses photos of the same bytes as it does clips.

- **Offline, in the suite** (the harness is in `benchmarks/`: `bench.py`, `haystack.py`, `haystack_media.py`, `shortcuts.py`, `grade.py`,
  `egolife.py`, `claude_run.py` and `codex_run.py`; its tests, and the fixtures they use (`day.py`, `fake_decider.py`, `conftest.py`),
  stay in `tests/`, with `benchmarks` on pytest's `pythonpath`, and the scripts put `tests/` on theirs):
  - `test_haystack.py` loads a two-day week into a daemon with the fake decider, whose words find every needle at its time (the
    newest one for a `last` question), nothing for the absent questions and no distractor, and pins the levels, the graders, the cut of the screen's segments, its acts (kinds by app, the activity chain) and frames, and the phone's times (EXIF, `creation_time`) through `write` and `load`;
  - `test_haystack_media.py` pins the drawings and sounds (the shade of a needle survives the encoder, with the cursor over it too, no distractor has it), a screen segment's frames and the layers a frame is drawn from;
  - `test_bench_rules.py` pins the fake decider's `shade:` and the grader's date rule;
  - `test_shortcuts.py` pins the probe;
  - `test_agents.py` checks both agents' command lines and event checks in each arm (`mms`, `files` and `blind`), running none;
  - `test_bench.py` builds, copies and continues a run with a stub agent, checks the oracle's rules, the probe's wiring and the
    report's tables and statistics on stub results, and the choice of questions of `--subset` and `--controls`;
  - `test_egolife.py` is the same for EgoLifeQA, below.
- `python benchmarks/bench.py build --scale tiny|pilot|standard` writes the week, read-only, in the temp folder, loads it into a snapshot
  home under `~/mms-bench/<run>/`, and runs **the shortcut probe** (`benchmarks/shortcuts.py`) over the data: for each media question
  it counts what brute force has to read (screen states, clips, photos, sounds, and the 5x5 montages that show them all) and how
  many other frames a colour filter ("red", "green", "brown"), frame differencing or a band filter for the sound's class keeps
  beside the needle, and checks that the needle's file is not the largest or smallest of its scope and that the files arm's
  ffmpeg has no `ocr` filter. It writes `shortcuts.json` and `shortcuts.md` in the run folder (never in the data), and a run that
  fails a floor is kept, printed failure by failure, and makes `build` exit 1; `--no-probe` skips it, and the report and the
  commands say so for a run whose probe failed, was skipped or never ran.
  `run.json` also holds a hash of the questions and one of the files and log lines, and every command refuses a run whose haystack
  changed (a distractor too) without a `VERSION` bump.
  On Windows every data file must keep only the permissions its folder gives it: Codex's sandbox reads as the
  `CodexSandboxUsers` group, and a file moved out of a Python temporary folder keeps that folder's owner-only permissions, which
  the sandbox cannot open (the first pilot's files arm answered "unknown" to every video question so). The writers copy a
  finished file into place, and `freeze` resets the data's permissions (`icacls /reset`) and refuses data that still has some
  of its own.
- `python benchmarks/bench.py decider <run> --models ...` (GPU, the maintainer's go-ahead, alone on the GPU): each model in a copy of the
  snapshot asks each needle's yes/no over its source and range, with no agent: a `last` question for 5 sightings (`LAST_LIMIT`: as many as
  one `look` checks), newest first, the others for 20 (a count for 50) and 5 candidates, each allowed 1,800 s. The needle's `rank` is its
  place among the in-range sightings, newest first. It prints the minutes each model needs, from its `About` and the windows of each range. The report gives recall at the model's `hit` and at p of 0.5, the needle's p, false
  sightings and seconds per window, and for each template a verdict on a model that did not find a needle: "below its model's hit
  (calibration)" when it was in its top 5; "this model's miss" when any other model found it, or another family had it in its top 5;
  "haystack suspect" only when nobody found it and models of two or more families scanned it with it in none of their top 5;
  with one family the tier cannot tell a miss from a fault of the haystack. A scan that ended before the needle is told as it is, and is no
  miss and no family that ran: "stopped on a false sighting" (a `last` ask stops after its 5 sightings, newest first, all false or of
  earlier needles, before it reaches the needle), "cut at max_s".
- `python benchmarks/bench.py agents <run> --agents claude,codex --models ... --jobs 3` (the plans' quota and the GPU, the
  maintainer's go-ahead): without mms, the agent reads the same files with its own tools (Claude: Read, Glob, Grep and Bash for ffmpeg only,
  which its instructions say, since a denied `cd ... && find` once made it give up on Bash;
  Codex: its shell in Windows' unelevated sandbox, which writes to its working folder and the temp folder but not the home
  folder, and view_image, whose calls only the exec's rollout file tells), once per agent; with mms, it has `ask`,
  `find`, `look` and `label` only, the skill and the model's notes, once per agent and model, in its own copy of the snapshot.
  Each question is one exec in a working folder with a `.mms-project` of its own, so its labels stay out of every other
  question's default scope. Notes and versions of `app.toml` stay current in every later range, so an ask over a later
  slot reads them and labels an earlier needle with its answer: in an mms home these questions run one at a time, and every
  queue runs in ascending `until`. Codex stops at 95% of the week's use, and Claude at a usage limit or at 95% when a
  rate-limit event gives its use; either also stops on a `setup`, `limit` or `auth` problem, or on 3 `error` results in a row.
  Claude runs on the CLI's own login (`claude auth status` says `"loggedIn": true`), not the desktop app's.
  Claude Code saves a tool output over about 30 KB under `~/.claude/projects/<folder>/<session>/tool-results` and names the
  file in the result: the files arm may read back that folder of its own session, and nothing else under `~/.claude`.
  Codex runs the model and effort of `MMS_CODEX_MODEL` and `MMS_CODEX_EFFORT` (default high); each result records them, and
  the report's Agents line counts the results of each, so two settings pooled by mistake show.
  An mms exec that ends with the GPU lost (nvidia-smi fails, as after a driver crash) is a `setup` problem: the agent stops,
  and the next command asks it again. `--dry-run` writes its files under the run's `dry-run` folder, never in `execs`.
  The same command continues a run: an exec that escaped, timed out or was capped is a wrong answer and is not run again,
  the others are.
  - The text and records questions do not depend on the model, so they run in the files arm and in one mms arm, d1-3B's
    (`--controls NAME`; `--controls all` asks every model's arm); the other models are asked the media questions.
  - `--subset pilot` runs the media questions (those of kinds some model of the run reads) and 3 controls: a text needle, a
    records needle and an absent one. A first run on the `pilot` scale is `agents <run> --subset pilot --models d1-3B,d3-edge`.
    `--only QID,...` narrows the subset to those questions (a smoke).
  - Run mms execs with `--jobs 1` on this laptop: three Codex mms execs asking d1-3B at once ended in an NVIDIA driver fault
    and a bugcheck (0x133) during the first pilot; the decider tier, one ask at a time, ran clean.
  - A model that takes over 0.15 s a window (from its `About`: d3-mini, d3-flash, pplx-decider-flash) runs one exec at a
    time: parallel execs share one GPU queue, and its timeouts would be the queue's.
  - **The GPU is held by the long execs.** An mms exec is given an estimate of the GPU seconds one ask over its range may need (the
    windows of the range times the model's seconds a window, as the decider's minutes are). One over 300 s (a week's ask over the
    screen is tens of thousands of windows: 112,091 on the standard week, 6,838 s at d1-3B's 0.061 s a window) runs alone in its arm, when the others have finished, and nothing of
    the arm starts beside it: parallel ones would wait in the GPU queue past the exec's 900 s and count as wrong. Files-arm execs
    are not held. It starts when the daemon's jobs (`status`) are down to none, and records how many were left if they were not within
    a minute. Each result records the estimate, whether it is past 75% of the exec's timeout (`bench.ASK_SHARE`: the agent's own turns take the rest, so an ask estimated past 675 s of 900 cannot finish; d1-3B's screen day asks at pilot scale are 847 and 945 s, d3-edge's 708 and 790, and a week's ask over the screen is estimated
    at 5,717 s for d3-edge, 6,838 s for d1-3B, 19,392 s for d3-mini, 21,746 s for d3-flash and 32,058 s for pplx on a `standard` week, and at
    2,240 s for d3-edge on a `pilot` one, all past it: a timeout there is the model's speed, not a miss; `agents` names those questions before it runs, and the appendix flags them), the seconds the question was held back by
    a rule while first in line with a slot free (`held_s`), and whether the exec ran alone.
  - **An exec starts from the snapshot's state.** An mms home is shared by the execs of an agent and model, and what an exec writes
    there is what a later one could read: the labels of its asks (every sighting is labelled with its project, and `find` with scope
    all reads other projects), the labels of the `label` tool, the items that `label` kept, the project and session rows of its
    calls, and the jobs it left running. When an exec ends, the ask jobs its log shows open are cancelled and waited for (an ask
    labels what it found when it stops, a cancel included, so a label can land after the cancel; a minute at most), then the daemon's
    jobs are waited down to one for each exec still running (a job whose id was never logged, because the exec timed out inside the
    call that started it, is only in that count; with others running that wait is 10 s), and only then are its labels, keeps,
    projects and sessions taken out of the index (`scrub_home`) and its template released. The next exec's start and the arm's end
    scrub the execs that ended again, and an arm begins by sweeping the `id:bench-` projects of an interrupted run. The keeps are
    put back to the snapshot's, not to the live home's. Each result records the jobs cancelled, those that had not ended and the
    orphans left. What stays holds nothing of an exec: the `looks/`
    pictures (named by what they show), the daemon's job records (random ids) and its log. mms has no cache of its answers, and no
    agent has `remember`. Two questions about one template's needle (an absent question is of its template) never run at once.
  - **The files arm's folders say nothing and share nothing.** The data, the ffmpeg and the working slots are in the temp folder
    under an opaque tag (the run's name holds its seed and scale). A command may not open another slot of the run, by its path, by
    its name (`..\mmsb-<tag>-w1`) or by a glob (`mmsb-*-w1`, `mmsb-<tag>-w*`, `*-w1`; the deny list of each exec has them; a listing
    may name the slots), a slot is emptied when its exec ends, and a command that reaches the generator or its truth as code does
    (`import haystack`, `from haystack`, `haystack.py`, `haystack.plan`, `['truth']`, `.get('truth'`) or a path in the repo ends the
    exec as `escaped`. Prose is no escape: a Claude `description` is not checked, only its `command`, and what a tool printed is
    checked for the paths alone. Two questions about one template's needle also never run at once in the files arm. A tool result
    that starts like JSON is never hashed in `tools.json` (only a `data:` URL or base64 is), so the asks' jobs and windows are read.
  - `agents` refuses a run whose shortcut probe failed unless `--anyway` is given (a dry run is not refused); a run built with
    `--no-probe` runs with a warning.
  - Each result records the asks' windows and GPU seconds (from the last `progress` of each ask job in `tools.json`), the pictures
    viewed (`view_image`, or Claude's `Read` of a picture) and the shell commands.
- `python benchmarks/bench.py report <run> [<run> ...]`: per agent and model, accuracy with and without mms on that model's questions,
  the difference (the mean of the paired differences) with Newcombe's 95% hybrid score interval for paired data, and b, c and
  McNemar's exact p, on all its questions and on the media alone (pictures, video and sound: the part that depends on the model);
  from fewer than 10 pairs right in one arm only the interval would exclude 0 where the p does not, so the report gives b, c and
  the p alone, and a verdict rests on the p; wrong answers to absent questions, timed-out execs (a wrong answer, and counted on
  their own), seconds, tool calls, tokens (the input, the output and for Claude its cache read and written; Codex's input holds
  its cache) and problems by kind; by level (per model and pooled over models, with the crossover); how each arm worked (the share of media
  execs that called `ask`, asks, finds and looks, the windows and GPU seconds of the asks, asks stopped at `max_s`, pictures viewed,
  shell commands, the GPU estimate, the seconds held and the asks estimated past the timeout); the seconds, tokens and GPU seconds per correct answer, told from 5 correct
  answers in the arm (k is beside it); per data type; the decider tier and a summary per
  model; and one row per question. About 20 questions a model show only a large difference (6 or more right only with mms and none
  the other way for p < 0.05), and a level has a few questions, so pool seeds; the rows are not independent (on `pilot` the part
  and day questions of a template share a needle, and pooled seeds repeat the templates); with k models and agents about 0.05 k rows
  fall below p < 0.05 by chance; a run with another seed adds new needles, and `report` pools runs (two runs of one seed and scale are
  refused). The screen's totals and the size of its changes are calibrated against two real hours (typing, scrolling, a cursor, up to 2 frames a
  second), but the reader's gate makes a window of 98% of its seconds with a frame, where it does of 71 to 86% of a real screen's; `look` costs about 25 times more
  a frame than an ffmpeg montage: the report says all of it.
- The parts of a run that belong to its dataset are methods and attributes of `bench.Run`, which another dataset's run overrides, so
  that `run_agents`, `run_decider`, `show_agents`, `show_decider` and `dry_run` serve both: `plan`, `prompt`, `view`, `estimate`,
  `needle`, `extra`, `deny`, `check_paths`, `ensure_data`, `questions` (the subset), `arms` (the arms with no model, in order),
  `warm_params`, `mms_jobs`, `oracle` (the decider tier's record) and `release`. `run_arm` takes the arm: `blind` (no tools and no
  recordings), `files` or `mms` (`arm_dir(arm, model)`, `result_path(run, agent, arm, model, qid)`). The haystack's runs have the
  last two. `remove_tree` deletes a file with other links as a name (`unlink`), so a hardlink an agent made, or a home's media,
  cannot make the data or the download writable.

### EgoLifeQA: a public benchmark through the same loop

`benchmarks/egolife.py` asks Claude Code and Codex the questions of EgoLifeQA (multiple choice, A to D) about 3 days of EgoLife's A1_JAKE
(2,592 clips of 30 s from Jake's glasses, 21.6 h, and the transcript of what was said around him), three ways: `blind` (the floor,
and a measure of contamination: EgoLife is public), `files` (their own tools on the clips and the transcript) and `mms` (d1-3B over
mms's index). Third-party data, read and never run; the run folder and `--temp` go on the download's volume (the clips are
hardlinked, mms's snapshot holds a second copy, about 42 GB), not on the system drive.
- `build --source <volume>/egolife --seed 1 --out <volume>/mms-bench --temp <volume>/mmsb-temp` (CPU, 10 to 15 minutes), then
  `agents RUN --agents codex,claude --models d1-3B [--dry-run] [--subset pilot] [--only QID] [--jobs 3] [--timeout S]` (the plans'
  quota; the mms arm runs one exec at a time), `decider RUN --models d1-3B` (the GPU, alone) and `report RUN`.
- **Which questions** (`select`, by `--seed`): for each age of the question (the query less the latest target: under 2 h, 2-6 h,
  6-24 h, over 24 h) 10 that need no sound, from the categories in turn, the rarest first; and 10 that need sound, asked from the
  transcript alone (d1-3B does not hear); a question with a range or a list of moments in one string, a target in a gap in the
  recording or a speech question with no line near its target is dropped and counted in `run.json`. A bin too short is a
  `ValueError`. `--subset pilot` is the first 2 of each bin and of the speech ones.
- **The horizon.** Each question is asked as of H, the end of the last whole clip that ended by the query, and no arm can read a
  clip or a transcript line past it: for `mms` the index is cut to H before each exec (`set_horizon`: later clips are `gone`, later
  transcript records move to a side table, in one transaction on the live index); for `files` each slot gets a folder of hardlinks to
  the clips that ended by H and the transcript lines that ended by H (`files_view`), and a command naming the full data, the
  download or another slot's folder ends the exec as `escaped` (the dataset's name only in a command: a transcript line says it, and
  the build refuses a line that holds a word the files arm refuses in an output). A files result whose command outputs show a start
  or an end of a transcript line, or a clip's name (outside a line that reports a missing file), past H (the view holds none: it read
  them where `deny` could not see) is `escaped` too.
  What is lost is the seconds between H and the query (the report gives them). An mms result records `after_horizon`, the recording
  times past H in its tool results (a find's labels aside; a sheet counts the items its tiles were cut from, not the times asked for; any ends the whole
  command, a `HorizonLeak`), and `asks_past_horizon`, the asks mms
  took that ended past H (mms's clock reads days later than the question; harmless). The default days are the latest three before the
  build that cross no change of the clocks (elapsed time across one differs by an hour between mms and the files arm), and `build`
  refuses days that do.
- **The decider tier** asks each question's yes/no, written by hand once (`benchmarks/egolife_oracles.json`, by `ORACLE_RULES`: from the
  question and its keywords, never the answer, one present-tense thing, no person for the glasses, no time word, no word of an option
  that the question lacks; the build refuses a chosen ID with no oracle or a broken one), over +-5 minutes around each target, cut at H, and scores recall at the model's hit, at
  p of 0.5 and 0.3 and in the top 5; the sightings away from every target are told beside it, not as a fault.
- **The report** gives per agent and model, for the 40 video questions: each arm's accuracy with its Wilson interval (chance is
  25%), mms against files, files against blind and mms against blind with the difference, Newcombe's interval from 10 discordant
  pairs, b, c and McNemar's p; by age, with blind as the floor and the youngest age at which mms leads; by category; the 10 speech
  questions apart; how each arm worked and what a correct answer cost; the horizon's audit; the decider tier; one row per question.
  n = 10 per age: a sign, not a finding. There is no shortcut probe (a public dataset).
- `test_egolife.py` makes a two-day download of tiny clips, SRT files with the real file's oddities and about twenty questions, and
  checks the parsing, the horizon, the choice, the build, the cut of the mms index (a daemon with the fake decider) and of the files
  view, the decider tier, stub agents of the three arms (a recording past H in a result ends the command) and the report.

## What design C must show

Each item is an end-to-end test.

1. **ffmpeg sources**:
   - segments appear and are indexed with UTC times that match the wall clock to about 2 s;
   - a video segment records whether it has sound (`meta.sound`), the one found after a crash too;
   - a stopped daemon closes its open segment;
   - a killed ffmpeg is reconciled at the next start;
   - a killed daemon takes its ffmpeg and command children with it, and the next start indexes every segment.
2. **folder, log and command**:
   - a changed file is a new version and an unchanged one is not; a log is read from its saved offset, and a restart repeats no line;
   - a record's time is its `time_field` when one is set, and the read time otherwise: a `"t"` field that is not
     named is not read.
3. **Retention**:
   - caps, thinning, `max_days` and the floor;
   - a remembered note and file go after `expires`, unless kept;
   - a program source's clock rows count against its cap and go with `drop_oldest`;
   - a label with `keep`, or `mms keep`, that would take the source's kept bytes past its `max_gb` is `too_big` and marks
     nothing; with no `max_gb` it passes; two sessions that race for the last of the room get one `too_big`;
   - a label goes with its item and a label of no item stays; stray `.part`, `.ffconcat` and `.err` files go once an hour
     old, and a younger one stays;
   - `looks/` files go after a day;
   - every write passes `Store.reserve`: a folder remember of large files stops at the floor, file by file, and a write
     larger than its source's whole cap is `too_big`.
4. **Consent**:
   - a screen, microphone, camera, ffmpeg or command source with no approval is `needs_approval` and starts nothing;
     once approved it starts with no restart;
   - changing its device, input or command makes it `needs_approval` again; changing its cap or fps does not, also
     through `source_set`, which keeps every field it is not given;
   - removing a source takes its approval with it, and so does a hand edit that removes it;
   - `mms approve` refuses without a terminal; `--revoke` stops the source;
   - nothing reachable through MCP approves: the tools list holds no such tool, `POST /v1/approve` is `not_found`,
     `source_set` with an approval field is `bad_request`, and `consent.json` is unchanged after every tool has run.
5. **Config**:
   - `source_set` adds a folder source, and the running daemon ingests it with no restart; `remove` stops it and keeps
     its history;
   - an edit by hand is picked up the same way; a bad one keeps the last good config, and `status` gives
     `config_error`;
   - `source_set` keeps the file's comments, and refuses `wrong_kind`, a cap below what the source holds or below the
     cap it has (`would_delete`), and any change while the file is bad; it merges the fields given and keeps the
     source's project; its reply says `pending_reload` until the daemon has applied it;
   - a folder or log source at the home, above it or inside it, and a `remember` of such a path, are refused;
   - a daemon that starts with a bad file runs, with no sources, and says why.
6. **The API**:
   - `daemon.json` appears once the API answers and goes at a clean stop;
   - a call without the token is `unauthorized`;
   - each op of DESIGN.md §11 answers its documented shape, and each documented error comes back with its code and
     `field`;
   - `find` pages with `next` and `after` through rows that share a time, or differ by microseconds, giving each row
     once; a text with no word in it is `bad_request`, not a 500;
   - `mms start --detach` returns once the daemon answers, survives the shell that ran it, and `mms stop` stops it.
7. **Jobs and the GPU queue**:
   - `jobs/start` with a short `wait_s` returns a running job with the sightings so far, and `jobs/get` collects the
     rest;
   - two slow jobs take turns window by window;
   - cancel stops a job within one window, and while d1 loads it reports `cancelled` at once;
   - a model changed in the config (its `model`, `url` or `envs`) is the one the next window uses;
   - bad parameters come back as a failed job with `bad_request` within the call;
   - a job past `job_max_s` is cancelled;
   - a home copy of `ask.py` is used from the next job;
   - after a restart, `jobs/get` is `not_found`.
8. **ask**:
   - over video, audio, text versions, records and remembered recordings, the fake decider's hits come back as
     sightings with the right times and look handles;
   - `limit` 1 returns the newest, and a run longer than `max_sighting_s` is cut and marked `capped`;
   - a run of video frames that look the same is one window (`still_s`, 30 s): a still clip gives a window per 30 s that
     spans it, a small square appearing, a change of colour or of brightness starts a new one, mild noise does not,
     `still_s` 0 gives a window per frame, and each direction reads its own runs in its own order (a sighting's whole span
     is the same); a still clip that hits is one sighting that spans it, read in a few windows; `still_s` is cut to
     `max_sighting_s`; a bad `still_s` is `bad_request`;
   - a range's start keeps its frames, and a version saved before the range and unchanged in it is read;
   - nothing matching returns no sightings and up to five candidates, best first, and none is labelled; a yes/no says
     nothing in `said` (a sighting is a yes, a candidate is below the hit);
   - `candidates` 0 gives none, and 21 or 1.5 is `bad_request`;
   - candidates are distinct moments: windows of one source within `merge_s` are one (the best), a window within
     `merge_s` of a sighting's whole span is not one, two sources at the same time are two, and the only label is
     the sighting's; equal `p` comes in scan order;
   - the pool is bounded by moments, not windows: a sighting that flickers between hits and misses, and one file
     version of 250 windows, leave room for another source's candidate; equal `p` at the pool's edge keeps the
     ones scanned first;
   - an item that cannot be read is counted in the warnings (the first five by name) and the scan reads the rest; a job
     can override any key of the reader's profile, `max_px` included, which is cut to the model's cap, 0 being the cap (a
     model with no cap reads 0 as stored), for the pictures and for the estimate, which is the model's cost at the cap
     (tested on a model whose cost keeps rising past it, through `window_seconds` and through the job's lint);
   - the decider gets each frame and picture at most 640×360's area with its aspect kept, each records window at most
     3,000 characters, and every question in one call per window (DESIGN.md §16, from `fake-decider.jsonl`);
   - a choice from `answers` reaches the decider with a `none` option, and a window that picks it is no hit;
   - "yesterday" in a question, two things joined by "and", no `since`, more questions than a pass reads, a `text` question
     over video, and an estimate over `max_s` each give their warning; a model that reads one question a pass makes the
     estimate double at two questions, and its costs, when assumed, are said to be;
   - the default scope reads ambient rows and the caller's project, not another project's; `scope: "all"` reads both,
     and `searched` says which;
   - `max_s` stops a slow scan before the range's end (`stopped: max_s`), and a slow first answer (the fake's `load:<s>`,
     once per decider) is not counted: the scan then reads the whole range;
   - a yes/no with `hit` true or left out counts at the model's own `about.hit` (a fake with 0.6 counts a 0.7 window, one
     with 0.8 does not); the label keeps the whole text, `question: yes (p 0.70), <time>`;
   - before any window is read, with `bad_request`: a question key no question has, a key or a type the model does not take
     (from its `About`), and a rule of the model's own (`job.check`);
   - an item of `questions` with no `type` (a benchmark trial's `{"question": ...}`), an unknown `type`, no
     `instructions`, a choice without `criteria` or a score without 2 to 10 levels (what d1 itself refuses) is
     `bad_request` before any hit is looked at, and the first says d1's format and the `question`, `answers`, `hit` fix;
   - a yes/no's `criteria` with one side, an extra key or an empty side is `bad_request` before any scan, saying the fix and that criteria may be left out;
   - a box and a boxes answer say their edges (fractions of the picture, 2 decimals) in `said`, and with the question's
     name and p in the label, `at least` before a count that is a floor; an edge outside 0 to 1 or past its opposite makes
     the answer unreadable;
   - a sighting or candidate is lean: `t` for one instant and `t0`, `t1` for a span, `said` only for what a choice, score,
     value or box adds beyond `p` (with two questions that hit, `name: answer (p ...)` for each), `capped` only when true,
     `project` only when there is one;
   - a top-level `hit` or `answers` without `question` is `bad_request`, never ignored; a choice whose `hit` lists
     options hits when any one is picked.
9. **remember and look**:
   - a note at a given `t`; a file, again `unchanged`; a folder as versions; a recording as a segment, and a picture as a
     still, at its given time, else its metadata's (`creation_time`, EXIF with or without an offset), else its mtime, else
     now, a time outside [1970, now] believed in neither, with `time_from` on each entry (an `unchanged` one the stored
     item's own); a bad EXIF offset is no offset; a media file written under two rescans ago waits a pass; a name that
     becomes a picture ends its old version; an SVG stays a text version; a folder source copies recordings and pictures
     at their real times, and a file over `copy_max_mb` stays a version in place; pictures of one folder taken at
     different times are different sightings, and a range holds only the pictures taken in it;
   - above `remember_max_mb` is `too_big`, and below the floor is `floor`;
   - `look` at an item and `at`, at a source and time, and at a run and `tp`, gives the right frame (by colour), sound
     window, text page or records, with `labels`; nothing there is `not_found` with a `hint`;
   - `look` with `moments` gives the entries in order, each the same as a single look; a moment that fails is its own
     `error` and the rest still run; a malformed `moments` is `bad_request` with `field: moments`; text moments
     split the 20,000-character page between them, and `next_offset` pages on with a single look;
   - `look` with `source`, `since` and `until` gives a sheet: 9 numbered tiles row-wise over a changing clip, one tile
     spanning a still range, fewer across a gap (same-looking pictures either side of one are two tiles), pictures as well
     as video, an item that cannot be read left out like a gap (all of them: an error), `not_found` over sound alone, a bad
     combination of fields refused, a portrait video fitted inside its cells, a long video decoded only at the frames
     picked, and a sheet can be one of `moments`.
10. **Program logs:**
    - spool files written by the contract become records, clock rows and video segments whose frame times match
      the `wall` given to the millisecond; a crash between an encode and its commit encodes the frames again and
      repeats no record; a bad batch moves to `_bad/`; a stream named like a folder source gets `refused.json`; a
      backlog is cut into segments of `segment_s`; a run is removed when idle by its cursor and never by its files'
      times or with batches unread; a removed program table is made again at the next log;
    - the client's files match §8.2; 10,000 `log` calls take under 0.5 s in all (the time is printed); a full
      queue, a full spool and a `refused.json` each drop and count, never block or raise; a bad name or non-JSON data
      raises at the call;
    - a program logging records and frames, with the daemon running, stopped, and started again, ends with every
      record once and every frame in a segment, and `find` gives `run` and `tp`.
11. **The project**, with git repositories made in `tmp_path`:
    - the same ID from the top level, a subfolder, a worktree, a clone, a moved copy, and after a remote is added;
    - a shallow clone gets the `remote:` ID, and URLs that differ only in form match;
    - `.mms-project` wins;
    - a folder outside git has no project.
12. **The adapter**:
    - `initialize` answers within 5 s under the suite's load (the time is printed; about 1 s is the aim), with no
      torch, av, numpy or PIL imported;
    - the server instructions' first paragraph stands alone in 512 characters, and the whole is under 4,096;
    - `ask`'s description ends with a paragraph made from the configured model's `About` (each `model_folder`, `python:` and
      no model have theirs), the whole within 1,024 characters, and `tools/list` imports nothing heavy;
    - the tools and their schemas are as DESIGN.md §12.3;
    - each tool makes its one API call with `caller` and absolute paths, and gives the result as compact JSON;
    - `look` gives an image block before its text (a sheet's too); with `moments`, each picture comes before its own JSON, and the
      call is `isError` only when every moment failed;
    - `--tools` offers only the named tools, in the usual order; a call of another is `not_found`; a bad or empty name
      is refused at the command line, naming the valid tools;
    - errors are `isError` results carrying the API's code;
    - `ask` with a job maps to get or cancel, and refuses other arguments beside it, `all` without a job included;
    - `ask`'s first reply is the whole result without its None fields; a check-in holds `job`, `state` and `progress` and
      each other field whole when its value differs from the last sent (a sighting whose start moved, a rewritten warning
      line, a shorter list included), `all: true` and a job the adapter does not remember give the whole result, a job's
      memory goes once a final state was sent, and a check-in the client cancelled while it waited records nothing; a
      check-in that finishes after a newer one, or after the final state, gives only its progress;
    - the session ID is fixed for the process, and the project comes from `CLAUDE_PROJECT_DIR`, else the working
      directory;
    - `ask`'s `wait_s` is cut to what is left of the call's budget;
    - with no daemon running, the first call starts one and still returns within 50 s; a daemon that died since the
      session began is replaced, and the call asked again once.
13. **Install**:
    - the fake `claude` gets `mcp remove`, then `mcp add --scope user mms -- <mms> mcp` (with `-e MMS_HOME` when
      set), and a "not found" from remove passes;
    - the Codex file gets `[mcp_servers.mms]` with its other tables and comments kept;
    - the skill is in both skills folders, and the login entry in the Startup or autostart folder;
    - running it twice changes nothing more;
    - `--remove` undoes each step;
    - the `model` step runs in this process, with the home in `tmp_path` and the Hub faked: a fresh install downloads d1-Omni-2B;
      a model named that hears nothing says so; a model that is configured stays, with no call to the Hub, and a `--models-dir` without
      `--model` then fails the step; `--model` with `--models-dir` downloads there; mms's own env missing a pin fails the step, with why, before any download; `--remove` keeps the weights
      and the setting and says so; `--model` with `--remove`, or without the model step, is a usage error;
    - nothing outside `tmp_path` is written;
    - `mms install`'s registered command line starts the adapter, and it answers `initialize` and a tool call.
14. **Across the seams**:
    - an MCP session asks, collects with `job`, looks at a sighting and gets the frame, remembers a note, labels a range
      and finds both, all against one daemon with the fake decider;
    - two sessions from two projects see each other's notes only with `scope: "all"`;
    - `mms ask` and `mms run` start jobs through the daemon, starting it when none runs;
    - the CLI's writes carry its folder's project.
15. **Files as phones and programs write them** (`tests/test_corpus.py`):
    - each file is recorded as its content says, whatever its name (text that starts like a container is still text);
    - pictures, frames and sound reach the fake decider upright, flattened, within the budget and on time, from every
      listed format;
    - text is decoded from UTF-8, UTF-16 and cp1252, and a video's soundtrack is heard;
    - broken files are warned about while the scan goes on, and big files are read in bounded memory.

16. **A decider can be swapped** (`tests/test_deciders.py`, `tests/test_documents.py`, `tests/test_narrow_deciders.py`):
    - a model folder is read by the backend that claims it, and one no backend claims, a folder that is missing, or
      weights that differ from the backend's `About`, are refused with a message that says why; `About` refuses facts
      that switch a lint off, and its `profile` cannot be written by a script;
    - the decider's status carries no error of another model: a switch of model, or a successful `about`, clears it;
    - a `python:` file with no `ABOUT` or `decide` is refused, and its `load` and `unload` are called;
    - the same ask over one store gives each decider what its `About` declares: sound at its rate, pictures within its
      cap, windows by its profile, a question type it does not answer refused, its own lint;
    - a file no decider reads is recorded, found by name and named in one warning, and never reaches it; one a decider
      takes (PDF, Office, ODF, EPUB, HEIC) arrives whole, and a hit on it is a sighting;
    - a decider that reads only pictures, or only sound, is handed those and no others, over a mic, a screen and a camera
      segment, a picture, a text and a PDF: what it left is named in the warning by group (`audio segments`, a media
      type) with what `look` does for it, and `progress.of_about` equals the windows read;
    - `look` gives a copy of it, whatever its name (one Windows cannot hold is turned to `_`, as the store does), and a
      file over the limit is `too_big`; an item reported twice is warned of once, and the warning is in the final result;
    - a camera segment's soundtrack that an eyes-only decider cannot hear is one rolling warning (`the sound of N video items
      was not heard, ...`), and the frames of one an ears-only decider cannot see are another; a decider that reads both
      says nothing.

17. **Five families, each in its own process** (`tests/test_worker.py`, `test_family_d1.py`, `_d1_3b.py`, `_d3.py`, `_pplx.py`,
    `_h2o.py`, `test_deciders.py`, `test_config.py`, `test_fetch.py`, `test_model_cli.py`, `test_imports.py`):
    - **The wire** is bit-exact: RGB noise at 640×360 and 1.6 MP, float32 with NaN, inf and -0.0, and 5 MB of file bytes
      come back with equal SHA-256s from a real child. Bytes that are not a frame and an oversized header are named. The
      time to cross for a 640×360 frame and a 1.6 MP one is printed (asserted only to be sane under the suite's load; the
      aim is 5 ms or less at 640×360).
    - **The worker:** a `ValueError` in the model is a `ValueError` here and the worker serves on; any other error is an
      `ApiError` naming its type; a crash is an `ApiError` with the exit code and the stderr tail; a hang or a load past its
      deadline is an `ApiError` and the process is gone; a library printing during a load does not break a frame; `unload`
      ends the process; the worker's environment holds no token, key or proxy; every `worker/*.py` imports under
      `python -I -S`; a daemon killed outright takes its worker with it.
    - **The daemon over a worker:** a `served` daemon whose test backend runs a real `Worker` over `fake_family` asks over a
      video segment and gets its sightings; its worker is gone after `idle_s`; a model switch mid-scan ends the old worker;
      a worker that failed is made again by the next request.
    - **A served model:** `Served` against a stdlib server answers 503 and then 200; a refused connection gives the hint to
      start the server; 422 is a `ValueError`; a 302 is refused and not followed; a dead `HTTP_PROXY` is not used; the
      `localhost` and `[::1]` hosts connect; `probe` gives any status of `/health`, and None when nothing answers (refused, or
      not HTTP). The daemon over `fake_shim`, with an H2O `model_folder` and `[decider] url`,
      asks over a picture and a video segment and finds sightings, with no weights.
    - **Claims and facts:** each variant is claimed from its `model_folder`, d1-2B as itself and not as d1-Omni-2B, d1-3B
      only by its `auto_map` and its (2048, 30) `text_config` and by no other backend, a d3 folder with an unknown
      `model_name` not at all; weights that disagree with the `About` are refused naming both
      numbers; an env marker that is missing or stale gives `mms model prepare <name>`; a `url` is refused by a model that is
      not served; the `About` of a folder is read, and its questions checked, without constructing its backend.
    - **d1-3B's worker,** over a fake `torch` and model, and in a real `Worker` over its family file: pictures reach
      `system_one` as `images=` and a sound or file is refused; the picture cap of the loaded code (`VISION_MAX_PIXELS`) is
      reported, and a backend whose `max_px` differs is refused naming both numbers, with the worker gone; a `system_one`
      without `images` fails the load; on Windows every `hybrid` module's `_fused` is set to false, and finding none to set
      fails the load.
    - **Translation,** from canned native answers: a score with probabilities {0: .2, 1: .2, 2: .6} and expected level 1.6
      reads as level 2; a choice's `confidence` is dropped; an `error` entry is a `ValueError`; H2O's `found: false` and
      `present` become mms's box; every translated answer parses with `ask.read`, and a key or type the model does not take
      (or, for d1-3B and d3, a score of 11 levels) is refused before its worker or server is asked.
    - **Config and `mms model`:** `url` forms are accepted or refused by `config.loopback` (`http://192.168.1.5:8741`,
      `https://127.0.0.1:8741`, no port, credentials and a path are `config_error`); `envs` must be absolute; `set_decider`
      keeps comments and removes `url`. `list` shows each folder's variant, reads and env state, says why a folder's weights
      are not all there, and offers each model mms knows that is not downloaded (with its size and repo), making nothing. `use`
      by name and by folder writes the model (and H2O's `url`, removed when switching away), warns for a microphone a deaf
      model cannot hear and says it hears nothing, refuses what is unclaimed or served at a `--url` that is not loopback,
      builds an env that is not prepared (a build that fails changes nothing), refuses d1 and d1-3B before any download when
      mms's own env lacks a pin, and prints H2O's commands only when `/health` does not answer, not while it warms; a running
      daemon follows. `prepare` runs a fake `uv` with the cache under the envs folder, writes `mms-env.json` only after the
      version check passes, installs nothing for d1, d1-3B and H2O (it checks mms's own env against the `d1` group's pins, or
      prints H2O's install and start commands, with the folder as `wsl()` gives it: quoted, and a UNC path refused), and
      downloads first what is not there. A name is matched in any case and written as the model's own (`d1-3b` is
      `d1-3B`).
    - **Downloads** (`test_fetch.py`, `test_model_cli.py`, `test_install.py`, with `FakeHub`): a folder that is the model and
      whole makes no call to the Hub; `unfinished` finds a shard missing from the index, a file the Hub's record lists that
      is missing or short (the newest record counts), and no weights, and a stray `*.incomplete` file is nothing to it; a
      download says the size, licence and commit before it starts, asks for that repo at that commit only (a Hub that gives
      no commit is an error), and goes into an empty, stopped or
      unfinished folder after deleting the stray partial files a killed download left (which are not counted as bytes already
      there); a folder that holds anything else, another model's with `.cache/huggingface/` too, is refused with no call; a
      drive that is not there is refused before any call; too little room (the size less what is there, plus 1 GB, and mms's
      `free_gb` floor when the models are on the home's drive) is refused before the download; one that leaves nothing a
      backend claims, or a file of the Hub's record short, is an error that says so; the Hub's error is a `FetchError` that
      names the repo, and `mms model` exits 1 and changes nothing; the switches (no Xet, no token, no telemetry) are in force
      at the call, and a hub imported before them is refused; the Hub's advice to set `HF_TOKEN` is dropped, once however
      often the hub is reached, and its other warnings are not. `FakeHub` also says the drive has plenty of room.
    - **Heavy libraries stay out of mms:** no module outside `worker/` imports torch, transformers, vllm or `pplx_decider`, and
      importing the deciders, `gpu`, `modelenv`, the transport, `fetch`, `prepare` and the adapter loads none of them, nor
      huggingface_hub, which only `fetch.py` imports (when a download runs).
    - **The agent's text:** every model's `About` names an existing notes file and every notes file is named; SKILL.md names
      no model and none of a model's numbers; its examples and each notes file's run through `ask` against every `About`
      that names them, with no refusal and no unknown parameter.

## Manual checks (with the maintainer's go-ahead)

These touch the real `~/.claude`, `~/.codex`, login items or the GPU, so none is in the suite:
- `mms install`, then one Claude Code session and one Codex session that call every tool;
- whether a daemon started by an agent's adapter outlives that agent (Windows Job Objects and breakaway). Under the MCP
  Python SDK's stdio client, which the tests use, it does not: the client's job object kills it with the session;
- the working directory Codex gives an MCP server;
- the GPU tests above, model by model (DESIGN.md §13.3);
- one real download into a scratch folder, with the licence line and a run stopped and started again. Done on 2 November
  with d3-edge (1.2 GB, through `prepare.locate` and `fetch.download`, stopping before the env build): killed with the
  weights at 619 MB, `mms model list` said "not all there"; the same command deleted the partial file, fetched the rest in
  half a minute and found the folder ready; the weights' SHA-256 matched the Hub's, and a third run made no call. It showed
  that huggingface_hub writes `refs/main` under `~/.cache` for a branch and passes on the server's advice to set
  `HF_TOKEN`, which is why mms downloads a commit by its hash and hides that advice;
- preparing the envs (`mms model prepare d3-flash`, `pplx-decider-flash-v1.2`: gigabytes), which also checks the pins
  (torch 2.14 on `cu130` needs driver R580 or newer; d3 and Perplexity run without flash-linear-attention);
- starting H2O's server in WSL (`mms model prepare h2o-lightning-4b` prints the commands), whether `127.0.0.1:8741` reaches
  it from Windows, and stopping it afterwards to free the VRAM;
- the bake-off, which sets each model's costs and `hit` (DESIGN.md §16);
- DESIGN.md §16's deferred measurements, in one GPU session with one model alone on the GPU:
  - d1's time per window at 640×360, 1280×720 and full size, with 1 and 4 questions;
  - states of 1.5K and 3K tokens, and 5 s of sound alone;
  - batches of 4 and 8 windows, with peak VRAM;
  - hit agreement across the three sizes over a real hour of screen and camera.
