# MultimediaSearch (mms)

**A long memory for Claude Code and Codex, of what your machine sees and hears.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Platform: Windows](https://img.shields.io/badge/platform-Windows%2011-lightgrey.svg)

MultimediaSearch (`mms`) records what you choose on your own machine: the screen, a microphone, cameras, folders of files, notes and
program logs. When your coding agent has a question about the past, a small multimodal model running on your own GPU looks at,
or listens to, the raw recordings and returns the moments that match. The default model sees; `mms model use d1-Omni-2B`
switches to one that also hears.

Your agent can answer questions like:

- "When did I last have the CUDA error on screen?"
- "Did anyone ring the doorbell this afternoon?" (with a model that hears)
- "What did my notes say about the retention rule on Tuesday?"

It gets back the moments that match, each with a confidence and a handle it can open to check the frame, sound or file itself.

## Why MultimediaSearch

- **The raw record is the memory.** Nothing is summarized or extracted when it is written. Screen and sound are kept as
  one-minute video and audio segments, files as one copy per version, logs as rows. A better model later reads the same
  record better.
- **Local.** Recording, storage and the model all run on your machine. mms goes online only to download a model (and, for d3
  and Perplexity, its Python packages), when you run `mms install`, `mms model use` or `mms model prepare`: no account, no token,
  no telemetry. (What your agent reads through mms's tools, pictures included, goes to your agent's model provider like any
  other tool result.)
- **You approve every device.** A source that opens the screen, a microphone, a camera, or runs a program records nothing until
  you run `mms approve <name>` in a terminal. No agent tool can approve one.
- **Bounded disk.** You set a cap per source and a free-space floor. Over a cap, the oldest recordings go, or the most crowded
  for their age, as whole files. Nothing is rewritten.
- **Any source, by configuration.** A new source is a table in `mms.toml`, or a program that logs through
  `mms.log()`. No new capture code.

## Does it help?

On a benchmark of 24 questions about three days of synthetic recordings, both agents answered more questions with mms than with
the same files and their own tools, and used fewer tokens and less time per question:

| Agent | Without mms | With mms | Median time per question | Tokens |
| --- | --- | --- | --- | --- |
| Codex (gpt-6.1-terra, medium) | 22/24 | **23/24** | 28 s → **19 s** | 5.3M → **2.7M** |
| Claude Code (claude-sonnet-5-5) | 18/24 | **22/24** | 58 s → **32 s** | 29.9M → **~7.0M** |

This is one pilot run, and the gains are not yet statistically significant. On real first-person video with transcripts
(EgoLifeQA), mms did no better than the files: 34/40 with mms against 36/40 without, at about twice the tokens per correct
answer. [benchmarks/](benchmarks/README.md) has the method, the caveats, and four questions traced side by side.

## How it works

```
Claude Code, Codex ── MCP (stdio) ──> mms mcp ── HTTP, 127.0.0.1 ──> the mms daemon
you ───────────────── terminal ─────> mms (CLI) ───────────────────> │
your programs ─────── mms.log() ─> spool ──────────────────────────> │
                                                                     ├─ capture: ffmpeg for screen, microphone, cameras
                                                                     ├─ ingest: folders, logs, commands, the spool
                                                                     ├─ janitor: caps, ages and the free-space floor
                                                                     └─ GPU queue: the decision model, in its own process
```

One daemon per memory owns everything that writes. It starts at login, and the agent's MCP server starts it if it is not
running. The decision model loads at the first question (about 20 seconds) and unloads after ten idle minutes, so the GPU is
free between questions.

Your agent gets seven tools:

| Tool | What it does |
| --- | --- |
| `find` | Instant search by time, source and words: files, log records and labels, newest first. |
| `ask` | Has the local model look at or listen to the record in a time range, newest first, and returns sightings, each with a source, a time span and a confidence. Long scans run as jobs the agent collects. |
| `look` | Shows a moment: a frame or picture inline, a sound's file, text, or a file to open. Or a sheet of up to 9 numbered frames across a time range. |
| `remember` | Stores a note, file, folder or recording at its real time. |
| `label` | Marks a time range with text, and can keep what it covers from caps. |
| `sources` | Lists the sources, their state and size, the capture devices, the floor and the model. |
| `source_set` | Adds, changes or removes a source. It needs a cap, and a cap can only stay or rise. |

`mms install` also gives both agents the `mms-questions` skill, which teaches them how to write a question the model can
answer: one observable thing, with the time range in `since` and `until`, never in the question.

## Requirements

- Windows 11, where mms is developed and tested. Linux capture (x11grab, PulseAudio, v4l2) and login start are implemented but
  untested. macOS is not supported.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- An NVIDIA GPU that runs PyTorch's CUDA 12.8 builds, with room for the model: the default, d1-3B, is a 6.3 GB download.
- Claude Code, Codex, or both.
- ffmpeg is optional: mms uses the one on your PATH, else the binary that `imageio-ffmpeg` installs.

## Install

```bash
git clone https://github.com/DanMcInerney/MultimediaSearch.git
```

```bash
cd MultimediaSearch
```

```bash
uv sync
```

```bash
uv run mms install
```

`mms install` runs six steps and prints one line for each:

1. `claude`: registers mms's MCP server with Claude Code, at user scope;
2. `codex`: registers it with Codex, and lets Codex call mms's tools without asking each time (none of them can approve a
   device, lower a cap or delete history);
3. `skills`: copies the `mms-questions` skill to both agents;
4. `login`: starts the daemon at each login;
5. `model`: downloads the default model, d1-3B, naming its licence first, and configures it;
6. `daemon`: starts the daemon now.

It changes only what is mms's, and refuses to overwrite anything else of the same name. `mms install --only <step>` reruns one
step.

`mms` is installed in the clone's `.venv`, not on your PATH: run it as `uv run mms ...` from the clone, or activate `.venv`. The
agents find it without either.

### Hearing

d1-3B sees but does not hear. For questions about sound, switch to the model that hears:

```bash
uv run mms model use d1-Omni-2B
```

### Uninstall

```bash
uv run mms install --remove
```

This undoes each step of `mms install`. It keeps your recordings and the model's weights, and leaves the daemon running: stop it
with `uv run mms stop`, then delete `~/.mms` and the model folder by hand if you want them gone.

## Quick start

Ask your agent to record something:

> Keep a memory of my screen, capped at 40 GB.

The agent adds the source with `source_set` and tells you which command to run. A screen, microphone, camera or command source
waits for you:

```bash
uv run mms approve screen
```

`mms approve` shows what the source opens and asks you to type its name. Then ask your agent about what happened, with a time if
you know one:

> When did I last have a red error in the terminal today?

The agent searches with `find`, scans with `ask`, and checks what it found with `look` before answering.

## Configuration

The memory lives in `~/.mms` (set `MMS_HOME`, or pass `--home`, to put it elsewhere). Its `mms.toml` holds the
limits, the model and the sources. Agents edit sources through `source_set`; you can edit the file by hand, and the daemon
reloads it.

```toml
[memory]
free_gb = 20                 # the floor: below it, recording stops until 1 GB above it

[decider]
model = "~/.mms/models/d1-3B"        # set by mms install and mms model use
idle_s = 600                         # unload the model after this long with no question

[sources.screen]
kind = "screen"
device = "0"                 # the display
fps = 2
keep = { max_gb = 40, when_full = "thin" }

[sources.notes]
kind = "folder"
path = "~/notes"
glob = "**/*.md"
keep = { max_gb = 5 }
```

| Kind | Records | Needs `mms approve` |
| --- | --- | --- |
| `screen` | a display, as video | yes |
| `microphone` | an audio input | yes |
| `camera` | a video input | yes |
| `ffmpeg` | any ffmpeg input | yes |
| `command` | each line a program prints, as a record | yes |
| `folder` | each changed file under a path, as a version | no |
| `log` | each new line of the files under a path, as a record | no |
| `program` | what a program sends with `mms.log()` | no |

Every source takes `keep = { max_gb, when_full, max_days }`. `when_full` is `drop_oldest` (the default) or `thin`, which removes
the recordings most crowded for their age. `uv run mms status --devices` lists the screens, microphones and cameras mms can open.

### Logging from your own programs

A Python program running in mms's environment can log to the memory without the daemon running:

```python
import mms

mms.log("robot", {"event": "gripper closed", "force_n": 12.5})
```

Records wait in a spool that the daemon ingests as the `robot` source.

## Decision models

The model is what reads the record for `ask`. mms ships no weights: `mms model use <name>` downloads one and switches to it, and
`mms model` lists them. One model is loaded at a time, in a process of its own. The d1 models run in mms's own environment;
d3 and Perplexity get one of their own, which `mms model use` builds; H2O is a server you run.

| Model | Sees | Hears | Download | Licence |
| --- | --- | --- | --- | --- |
| **d1-3B** (default) | pictures, video | no | 6.3 GB | LFM 1.0 |
| d1-Omni-2B | pictures, video | yes | 4.4 GB | LFM 1.0 |
| d1-2B | pictures, video | no | 3.9 GB | LFM 1.0 |
| d3-edge, d3-mini, d3-flash | pictures, video | no | 1.2, 9.1, 16.8 GB | Apache-2.0 |
| pplx-decider-flash-v1.2 | pictures, video | no | 9.1 GB | Apache-2.0 |
| h2o-lightning-4b | pictures, video | no | 9.4 GB, served by you (vLLM in WSL) | Apache-2.0 |

All are public on Hugging Face and need no account; mms names each one's licence before it downloads. The default yes/no
threshold is measured only for d1-Omni-2B so far; the others' are assumed, and the `ask` tool's description says which.

## Command line

| Command | What it does |
| --- | --- |
| `mms status` | the daemon, the free-space floor, the model and each source |
| `mms approve <source>` | let a source that opens a device or runs a program record (`--revoke` to withdraw) |
| `mms find`, `mms look`, `mms ask` | what the agent's tools do, from a terminal |
| `mms remember <text>`, `mms remember --path <path>` | store a note, file, folder or recording at its real time |
| `mms label`, `mms keep` | mark a time range, or keep what it covers from caps |
| `mms model`, `mms model use <name>` | list the decision models, or switch to one, downloading it if needed |
| `mms start` / `mms stop` | run or stop the daemon (`mms start --detach` runs it in the background) |
| `mms install`, `mms install --remove` | set up or undo the agent registrations, skill, model and login start |

Times take ages (`90s`, `30m`, `2h`, `7d`), ISO 8601 or epoch seconds. Add `--json` for machine-readable output, and `--help` to
any command for its options.

## Development

```bash
uv run pytest
```

runs the offline suite on four workers: no GPU, no agent, no network, and every test uses a throwaway home. GPU and agent tests
are opt-in (`-m gpu`, `-m codex`) and described in [TESTBED.md](TESTBED.md).

| Path | What |
| --- | --- |
| [DESIGN.md](DESIGN.md) | the architecture: the store, the daemon, capture, ingest, `ask`, MCP, install |
| [TESTBED.md](TESTBED.md) | how mms is tested |
| [VISION.md](VISION.md) | goals, scope and open questions |
| [benchmarks/](benchmarks/README.md) | the with-mms-and-without benchmark harness and its published results |
| `src/mms/` | the package: the daemon, capture, ingest, the index, the CLI, the MCP adapter, the `ask` script, the skill |
| `tests/` | the test suite |

## Status

Version 0.1.0. mms works end to end, but it is young: expect rough edges, and changes to the store's format between
versions. [VISION.md](VISION.md) lists the open questions, starting with how a scan by the local
model compares with plain text search over transcripts and OCR on real history.

## License

[MIT](LICENSE). The decision models are downloaded from Hugging Face under their own licences.
