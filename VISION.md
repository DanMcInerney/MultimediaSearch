# MultimediaSearch: vision

Start here. This file says what we are trying to achieve and what is in and out of scope. How is in DESIGN.md, and
testing is in TESTBED.md.

| Document | Holds |
| --- | --- |
| **VISION.md** | the goals, scope, open questions, and how work gets decided |
| [DESIGN.md](DESIGN.md) | how: the architecture, the store, the build order |
| [TESTBED.md](TESTBED.md) | how it is tested |
| [benchmarks/](benchmarks/README.md) | the benchmark's published results |

## The goal

**MultimediaSearch gives Claude Code or Codex a long memory of whatever its owner chooses to record, and answers questions
about it by having a small local multimodal model look at the raw record.**

The owner names the sources: the screen, a microphone, cameras, folders of files, notes, logs, the agent's own
transcripts, sensors. MultimediaSearch keeps what they record for months on one laptop without filling the disk. The agent
asks in plain terms, with a question, the answers that count, some context and a time range, such as:
- "when did I last have the CUDA error on screen?";
- "did anyone ring the doorbell this afternoon?";
- "what did my notes say about the retention rule on Tuesday?".

It gets back pointers to the moments that match, and looks at them before trusting them.

## What the goal asks for

1. **Any source, by configuration.** A new kind of source needs a line of configuration or a small program that prints
   records, not new capture code.
2. **The raw record is the memory.** Nothing is summarised or extracted at write time. Text, labels and indexes are
   pointers back to it.
3. **Bounded disk.** The owner sets limits per source and a free-space floor, and they hold.
4. **Questions by time.** Answers come from a local decision model (d1-Omni-2B by default, which `mms install` downloads; seven
   others swap in with `mms model use`, which downloads them) reading the record in the requested range, and say "not found"
   when nothing matches.
5. **Future-proof.** A better model makes the memory better with no change to what is stored.
6. **Local and private.** Nothing leaves the laptop. mms goes online only to download a model (and, for d3 and
   Perplexity, its Python packages), when the owner runs `mms install`, `mms model use` or `mms model prepare`: no token, no
   telemetry. A model's own code from its Hugging Face repo runs on the owner's machine with the owner's rights, at a commit
   mms pins, never the newest.
7. **Either agent:** Claude Code or Codex.

## Scope

**Out of scope, unless the maintainer says otherwise:**
- cloud storage or hosted models;
- captioning, embedding or fact extraction built into the store (scripts may do them);
- a user interface beyond the CLI and MCP;
- training models.

## Open questions

- **Is the d1 scan good enough?** Against plain text search over transcripts and OCR, on real history, for recall,
  false hits and seconds per day.
- **Speech.** d1 is weak on long speech; how much should transcripts carry?
- **One model or two.** The default, d1-Omni-2B, is the only one that hears.
  Is one model enough, or should one that reads pictures better run beside it, kept for sound? Eight models swap in
  now, one loaded at a time. DESIGN.md §13.3 gives the path to two (`[decider] sound`); a bake-off of the eight on this laptop
  is what decides.
- **Screen as video or as pictures.** Video is the plan; whether it costs d1 accuracy is unmeasured.

## Status (8 November 2026)

- Research done, 30 and 31 October (the notes are kept outside the repository).
- Design C (DESIGN.md) is built: one daemon, MCP and CLI faces, program logs through a spool. It has run in real Claude Code and Codex sessions in the benchmark ([benchmarks/](benchmarks/README.md)).
- The decision model is swappable between eight models in five families, each in a process of its own: d1-Omni-2B (the default),
  d1-3B and d1-2B, d3 (edge, mini, flash), Perplexity Decider Flash, and H2O-Lightning-4B (a server the owner runs in
  WSL). `mms install` downloads the default and `mms model use` any other: there are no weights in the repo. Only d1-Omni-2B's
  default `hit` is measured. All eight have run on this laptop's GPU and their costs are measured; the other `hit`s stay
  assumed until the rest of the bake-off (DESIGN.md §13.3, §16).

## How work gets decided

- **The maintainer sets direction.** Agents propose with evidence: links, measurements, and what was not checked.
- **Ask first:** paid model runs, installs outside the project's own environment (`mms model use` and `prepare` build
  gigabytes of env), model downloads (`mms install`, `mms model use` and `prepare` start one), licence acceptance, and new
  accounts need the maintainer's go-ahead. Python packages installed with uv into
  the project's environment are pre-approved.
- **Prefer the smallest step** that answers the next open question.
- **Keep the documents few and short:** these three at the root, `README.md` as the front page that points to them, and
  `benchmarks/README.md` for published results. `CLAUDE.md` and `AGENTS.md` only point here.

## Changes

- **30 October 2026:** first version, from the initial research.
- **1 November 2026:** the decision model is swappable between five families, each in its own process.
- **2 November 2026:** d1-3B is the default model. `mms install` downloads it, and `mms model use` downloads any of the eight
  models mms knows.
- **4 November 2026:** the benchmark moved from `tests/` to `benchmarks/`, with its first published results, and `README.md`
  is the front page.
- **7 November 2026:** the project is renamed from Multimem to MultimediaSearch, with the short name `mms` for the package,
  the command, the MCP server, `~/.mms` and `MMS_*`.
- **8 November 2026:** d1-Omni-2B, the one model that hears, is the default in place of d1-3B. `mms install` downloads it.
- **8 November 2026:** each model is pinned to one commit of its Hugging Face repo, and mms downloads that commit, never the
  repo's newest: a model's own code runs on the owner's machine, so what runs is what was tested.
