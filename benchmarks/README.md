# Benchmarks: Claude Code and Codex, with mms and without

Does an agent answer questions about its owner's recordings better with mms than with the same files and its own tools? This
folder holds the harness that measures it, and the published results.

Each question is asked twice, once per arm, in a fresh agent session each time:

- **Without mms:** the recordings are plain files in a folder (photos, screen recordings, camera clips, sound, notes, logs). The
  agent uses its own tools. Claude Code has Read, Glob, Grep and Bash for ffmpeg. Codex has its shell (in its Windows sandbox) and
  `view_image`.
- **With mms:** the same recordings are loaded into an mms store. The agent has only mms's tools (`ask`, `find`, `look`, `label`),
  plus mms's skill and the model's notes. `ask` has a small local model (d1-3B by default) read the windows of a time range on
  the GPU, newest first, until it has found what was asked for or its time is up, and return the moments that match.

Answers are graded by fixed rules (`grade.py`: numbers, codes, times within a tolerance), not by another model.

## Results: the haystack pilot (4 November 2026)

24 questions about 3 synthetic days, each asked of each agent in both arms. 16 of them are media questions (photos and video). The
other 8 are text and records, which both arms got right throughout.

| Agent | Right without mms | Right with mms | Media only | Median time per question | Tokens | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| **Codex** (gpt-6.1-terra, medium effort) | 22/24 (92%) | **23/24 (96%)** | 14/16 → 15/16 | 28 s → **19 s** | 5.3M → **2.7M** | (not reported) |
| **Claude Code** (claude-sonnet-5-5) | 18/24 (75%) | **22/24 (92%)** | 10/16 → 14/16 | 58 s → **32 s** | 29.9M → **~7.0M**¹ | $13.48 → **~$3.56**¹ |

¹ Two Claude mms execs timed out, and a timed-out exec records no usage. The report counts them as 0 (3.9M tokens, $2.70). Here
they are counted from their event streams (about 3.1M tokens), and priced at the rates the other results were billed at (about
$0.86). Their output tokens are slightly undercounted.

**What it says:**

- **mms never cost a right answer.** No question was right without mms and wrong with it. Codex gained 1 question and Claude 4.
- **Video is where the gap is.** Claude went from 3/9 to 7/9 on video, Codex from 7/9 to 8/9. Text, records and photos were 100%
  in both arms for both agents.
- **Fewer tokens.** With mms, Codex used half the tokens and Claude about a quarter. Claude's estimated cost fell from $13.48 to
  about $3.56 for the 24 questions.
- **Faster per question, not always in total.** The median question took a third less time with mms for Codex and nearly half
  less for Claude. In total, Claude's mms arm took 35% less time (2,942 s against 4,546 s), but Codex's took 13% more (1,513 s
  against 1,334 s): a few of its screen asks ran for minutes.
- **mms adds GPU time.** That cost is local, not tokens. Per correct media answer: Codex 77 GPU seconds, Claude 179.
- **Not yet significant.** With 24 paired questions, the gains are McNemar p = 1.0 (Codex, 1 to 0) and p = 0.125 (Claude, 4 to 0).
  p < 0.05 needs at least 6 discordant questions, all one way. This is one seed of the pilot scale: a sign to follow up, not a
  finding. `bench.py report` pools the runs of more seeds.

Cost per correct media answer:

| Agent | Arm | Correct | Seconds per correct | Tokens per correct | GPU seconds per correct |
| --- | --- | --- | --- | --- | --- |
| Claude | without mms | 10/16 | 420 | 2.84M | - |
| Claude | with mms | 14/16 | 201 | ~459k¹ (the report: -) | 179 |
| Codex | without mms | 14/16 | 85 | 353k | - |
| Codex | with mms | 15/16 | 95 | 163k | 77 |

Some results were set aside during the run and asked again; none of them is in these numbers. They were an earlier Codex arm
without mms at high effort, 20 Codex mms results after a GPU driver crash (mms's model could not load, so they were answered from
`find` and `look` alone), and 2 Claude results that a fault in the harness's guard had ended as escapes. The harness now catches
both faults.

### Where mms did not help

- **"What was the error code on the red error screen the last time one appeared, and on what day and at what time did it first
  appear?"** (`video-screen-error-3`): wrong for both agents in both arms. The range holds about 81,300 screen frames, with red
  warning dialogs and banners as decoys. The last error screen is on Monday, with about 47,600 frames after it. Without mms,
  Codex answered "unknown" and Claude hit its turn cap. With mms, one `ask` over three days of screen needs more GPU time than a
  question's 900 s limit allows, and the run flags this before it starts. Codex narrowed its search to Tuesday, found nothing
  there and answered "unknown". Claude asked over all three days, narrowed to Sunday and Monday late, and timed out just after
  a likely sighting came back.
- **`video-absent-error-2`** asks for the error code on Tuesday 3 November, when there was no error screen. The right answer
  says there was none. Codex got it right in both arms. Claude hit its turn cap without mms and timed out with mms.

### What was not tested

- **Sound.** d1-3B, mms's default model, does not hear. The 7 sound questions (doorbell, smoke alarm) ran without mms only: Codex
  5/7, Claude 5/7. `d1-Omni-2B` hears, and was not in this run.
- **Other decision models.** Only d1-3B. The harness runs any of mms's eight (`--models`).

## How each arm finds an answer: four questions side by side

The arm without mms does not read every frame. The agents are resourceful. They use file names as timestamps, tile frames into
contact sheets with ffmpeg (or System.Drawing in Codex's case), and write their own detectors: a colour counter for red pixels,
ffmpeg's `signalstats` to pick saturated frames, band filters and `silencedetect` to find a beep, spectrogram images to look at.
With mms, the pattern is `find` (what is recorded when), an `ask` (the model reads the range's windows, newest first, and returns
the likeliest moments), then `look` at the best ones to check. Below, "1 ask" is one scan of the model; collecting its results
can take several calls.

**1. "What number was on the last green sign that I took a photo of?"** (photos over three days; answer 585)

| | Codex, no mms | Codex, mms | Claude, no mms | Claude, mms |
| --- | --- | --- | --- | --- |
| Result | right | right | right | right |
| Time | 29 s | 19 s | 90 s | 50 s |
| Tokens | 111k | 53k | 671k | 274k |
| Cost | - | - | $0.38 | $0.19 |
| How | tiled the photos 60 to a sheet, newest first, and opened 2 sheets | 1 ask: 113 windows, 7 GPU s | 27 calls: 9 ffmpeg contact sheets and 2 photos opened | 4 asks: 292 windows, 19 GPU s |

**2. "On what day and at what time was a parcel last left at the front door?"** (90 porch-camera clips over three days, 59 of
them after the last parcel; answer Monday 07:30)

| | Codex, no mms | Codex, mms | Claude, no mms | Claude, mms |
| --- | --- | --- | --- | --- |
| Result | right | right | **wrong** (Tuesday 19:39) | right |
| Time | 77 s | 58 s | 114 s | 75 s |
| Tokens | 445k | 95k | 750k | 270k |
| Cost | - | - | $0.42 | $0.20 |
| How | one frame of each clip on a contact sheet, then strips of the likely clips: 16 calls, 6 pictures | 1 ask: 600 windows, 37 GPU s | 51 calls, 35 pictures opened | 2 asks: 626 windows, 42 GPU s |

**3. "What was the error code on the red error screen on Sunday 1 November, and at what time did it first appear?"** (one
day of screen recording: 390 one-minute files, about 25,600 frames, with red warning dialogs and banners as decoys; answer E-8755
at 14:01:22)

| | Codex, no mms | Codex, mms | Claude, no mms | Claude, mms |
| --- | --- | --- | --- | --- |
| Result | **wrong** ("unknown") | right | right | right |
| Time | 124 s | 253 s | 827 s | 170 s |
| Tokens | 576k | 416k | 5.5M | 400k |
| Cost | - | - | $2.36 | $0.23 |
| How | wrote a red-pixel scanner in C#, ran it with 4 settings, opened 4 frames and a montage, found nothing it trusted | 2 asks: 3,690 windows, 237 GPU s | 109 calls: ffmpeg's saturation measure (`signalstats`) to pick frames, 11 opened | 2 asks: 2,013 windows, 140 GPU s |

This is the clearest case. Without mms, Claude got there by brute force, taking 5 times as long, 14 times the tokens and 10 times
the cost of its mms run.

**4. "At what time did the smoke alarm beep on Sunday 1 November, in the morning?"** (microphone; answer 07:18:57; without mms
only, since d1-3B does not hear)

| | Codex, no mms | Claude, no mms |
| --- | --- | --- |
| Result | right | right |
| Time | 43 s | 82 s |
| Tokens | 134k | 332k |
| Cost | - | $0.24 |
| How | ffmpeg band-pass (2 to 4 kHz) and `silencedetect`, then `astats` on the hits | 35 calls, 15 pictures, mostly spectrograms |

## What was asked: the haystack

The data is a synthetic week (`haystack.py`, generator 6), written for this benchmark so that every answer is known. The pilot
scale is 3 days, seed 1: 1,572 files, among them about 81,000 screen frames, 1,442 porch-camera frames, phone photos and clips,
a hallway microphone, notes, versions of a config file and the logs of three services.

- **The screen** changes like a real one. It has typing, scrolling, a moving cursor, a taskbar clock, idle and busy stretches, and
  up to 2 frames a second, cut into one-minute files as mms's recorder cuts them. Its rate of change was calibrated against two real
  hours of a developer's screen recording ([TESTBED.md](../TESTBED.md#benchmark-with-mms-and-without)).
- **Each question has a needle** hidden among decoys that look or sound alike but mean something else: a red warning dialog or
  banner beside the red error dialog, a brown dog beside the parcel, other beeps beside the smoke alarm. Some questions ask, in the
  same words, about something that is not there; the right answer is that it is not there.
- **How vague the time hint is** varies: a part of a day ("on Sunday, in the morning"), a day, or no hint ("when did ... last
  ...?", over the whole week).
- **A shortcut probe** runs at build time. It checks that no cheap trick finds a needle alone: no colour filter, frame difference,
  band filter or file size puts it first. Its output for this run is in [`shortcuts.md`](results/2026-11-04-haystack-pilot/shortcuts.md).
  A filter tuned to a needle's exact shade or pitch does find the rest, once an agent has found one. That is by design and
  reported, not checked.
- **The agent cannot cheat.** It cannot reach the generator, the answer key, mms's store (in the arm without mms) or another
  question's working folder. A command that tries ends the question as `escaped`, which is graded wrong. So are timeouts (900 s
  per question) and Claude's turn cap (60).

### Read the results with these

- **Without mms, nothing would be recorded at all.** The arm without mms reads a screen recording, camera clips and microphone audio
  that exist only because the benchmark wrote them. In real use mms is also the recorder: the comparison is between two ways of
  reading the same recordings, not between having mms and having nothing.
- **Synthetic needles are easier to filter than real ones.** A drawn red dialog or a pure tone is easier to find with a colour
  or band filter than a real error or a real alarm in a noisy room. This favours the arm without mms.
- **mms's screen asks cost a little more here than on a real screen.** mms's change gate keeps 98% of this screen's seconds as
  windows to read, against 71 to 86% on the real recording.
- **`look` at chosen moments is dearer than a contact sheet.** It returns separate pictures, up to 8 per call, where a 5x5 ffmpeg
  montage is one picture of about 1.5k tokens. `look` over a time range returns one sheet of up to 9 frames.
- **Tokens are a measure of work, not a bill.** They count input, output and, for Claude, cache reads and writes, and a cached
  token costs much less. The cost column is Claude's list-price estimate. Codex reports no cost.
- **The questions are not independent.** On the pilot scale, a template's part-of-day and day questions share a needle.

## Results on real video: EgoLifeQA (3 November 2026)

[EgoLife](https://huggingface.co/datasets/lmms-lab/EgoLife) is a public dataset of first-person video from smart glasses with
transcripts of what was said, and EgoLifeQA asks multiple-choice questions (A to D) about it. We asked Codex (gpt-6.1-terra at
high effort) 40 video and 10 speech questions about 3 days of one wearer (2,592 clips of 30 s, 21.6 hours). Each question is
asked as of its own time, and no arm can see later clips or transcript lines. The blind arm gets no recordings at all: it is the
floor, and measures what the model already knows of a public dataset.

| Arm | Video questions right | Median time | Tokens per correct |
| --- | --- | --- | --- |
| Blind (no recordings) | 16/40 (40%) | 5 s | - |
| Without mms | 36/40 (90%) | 31 s | 123k |
| With mms (d1-3B) | 34/40 (85%) | 26 s | 244k |

**mms did no better than the files here** (34/40 against 36/40; 4 questions right only with mms, 6 only without; p = 0.75), and
it used twice the tokens per correct answer. Speech:
9/10 without mms, 8/10 with. On this data the transcripts carry most answers. The agent asked mms's model to look at video in only 14
of 40 questions, and its `find` calls returned long transcript records that cost tokens. That run used an earlier mms build and
Codex at high effort, so it is not directly comparable with the pilot. The full report is in
[`results/2026-11-03-egolife/`](results/2026-11-03-egolife/report.md). The clips and questions are not in this repository. Only our
scores and the hand-written yes/no questions for the decision-model tier (`egolife_oracles.json`) are.

## Run it

Needs: Python 3.12 and [uv](https://docs.astral.sh/uv/), the Claude Code and Codex CLIs signed in (the runs use your plans'
quota), and for the arm with mms an NVIDIA GPU with mms's model installed (`mms install` puts it in `~/.mms/models`). Tested on
Windows 11 only. A pilot run takes about 1 GB of disk. Seed 1's answers are now public in its `report.json`, so give a new run
you mean to publish another `--seed`.

```bash
uv run python benchmarks/bench.py build --scale pilot --seed 2 --out ~/mms-bench --temp ~/mmsb-temp
```

```bash
MMS_CODEX_EFFORT=medium uv run python benchmarks/bench.py agents ~/mms-bench/<run> --agents claude,codex --models none
```

```bash
MMS_CODEX_EFFORT=medium uv run python benchmarks/bench.py agents ~/mms-bench/<run> --agents claude,codex --models d1-3B --models-dir ~/.mms/models --jobs 1
```

```bash
uv run python benchmarks/bench.py report ~/mms-bench/<run>
```

In PowerShell, set the effort first with `$env:MMS_CODEX_EFFORT = "medium"`. Codex runs at high effort without it, and the model
is `MMS_CODEX_MODEL` (default gpt-6.1-terra).

`build` writes the week and runs the shortcut probe (CPU only, about 11 minutes for the pilot). `agents --models none` runs the arm
without mms. `agents --models d1-3B` runs the arm with mms; keep `--jobs 1` on a laptop GPU. `--models-dir` says where the model
folders are (default `MMS_MODELS`, else the home's `models` folder, `~/.mms/models`). `report` writes `report.md` and
`report.json` in the run folder and pools several runs (`report RUN RUN ...`). The same `agents` command continues an interrupted
run. `bench.py decider` measures the model alone, with no agent. `egolife.py` has the same commands for EgoLifeQA.
[TESTBED.md](../TESTBED.md#benchmark-with-mms-and-without) has the full method.

## What is here

| Path | What |
| --- | --- |
| `results/2026-11-04-haystack-pilot/` | The pilot above: `report.md` (every table, and one row per question), `report.json`, `shortcuts.md`. Recorded before the rename to `mms`, so it says `mm`. |
| `results/2026-11-03-egolife/` | The EgoLifeQA run: `report.md`, `report.json`. Also recorded before the rename. |
| `bench.py` | The command line: `build`, `decider`, `agents`, `report`; the exec loop, homes, statistics |
| `haystack.py`, `haystack_media.py` | The synthetic week: the plan and its questions; the drawings and sounds |
| `shortcuts.py` | The shortcut probe |
| `grade.py` | The graders |
| `claude_run.py`, `codex_run.py` | One Claude Code or Codex exec per question, with each arm's tools and guards |
| `egolife.py`, `egolife_oracles.json` | EgoLifeQA on the same loop |

The harness's own tests are in `tests/` (`test_bench.py`, `test_haystack.py`, `test_egolife.py` and others) and run offline with
the rest of the suite.

The pilot ran at commit 7895548 with the changes later committed as aca92f6 (generator 6) and b0f2b08 (harness fixes the run
needed).
