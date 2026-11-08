# MultimediaSearch: design

How MultimediaSearch works. Goals and scope are in [VISION.md](VISION.md); testing is in [TESTBED.md](TESTBED.md). The
research behind it (a survey, a storage plan, and how to cut and store for d1) is kept outside the repository; §16 records
what it settled.

This is the spec of **design C**, as built: agents attach by MCP alone, one daemon per home owns the decider's queue and
everything that writes, the decision model runs in a process of its own (d1-3B by default, which `mms install` downloads;
§13.3 has the eight models that swap in), and programs log through a spool. §17 lists the files and what comes next.

## 1. Principles

1. **The raw record is the memory.** Labels, transcripts and indexes point back into it.
2. **Store for the disk; cut for the model when reading.** Streams are kept as segment files of about 60 s, files as
   one copy per version, small records as rows. What a model reads is cut from them at query time, so a new model
   changes a script, never the store, and what a model is handed is cut from the facts it declares (its `About`,
   §13.3).
3. **Fixed code only for time, safety, data and waking:** capture, the index, retention, consent, the daemon, its API
   and its job runner. Whatever runs a model or decides what to ask is a script the agent may copy and change: the
   `ask` scan, its hit rules and its lint.
4. **Retention deletes whole files and rows, and never rewrites.** No agent-facing call deletes. Data goes only by a
   cap, an age limit, or an expiry set when it was written, and an agent may not lower a cap below what a source holds.
5. **One daemon per home owns every writer and the decider.** Agents and programs reach it through faces that keep no memory of
   their own: the MCP adapter (`mms mcp`), the CLI and the program log client.
6. **One owner per job inside the daemon:**
   - capture owns ffmpeg;
   - ingest owns watched files, commands and the spool;
   - the janitor owns deletion;
   - the GPU queue owns the decider (a worker process mms starts, or the owner's server);
   - the job runner owns scripts.

   Each thread opens its own SQLite connection, and WAL lets readers run beside writers. Creating the index is the one
   step they must not race on: `Store` does it under `index.lock`.
7. **The owner approves, outside the agent.** Nothing opens a screen, microphone or camera, or starts a program, until
   the owner has run `mms approve` for it in a terminal (§4.3). No MCP tool or API call can approve.

## 2. The parts

```
Claude Code, Codex -- stdio MCP --> mms mcp (one per session) -- HTTP, 127.0.0.1 --> the daemon (one per home)
the owner ----------------------> mms, the CLI ---------------- jobs over HTTP ----> the daemon
                                    (store calls run in the CLI's own process)
programs -- mms.log() -------------> home/spool/ files -------- ingested by -------> the daemon

the daemon: capture (ffmpeg sources), ingest (folder, log, command, spool), janitor (floor, caps, ages, expiry),
            API (ops and jobs), GPU queue (drives the decider's worker, started on demand)
```

- The daemon starts at login (§14). An adapter or `mms ask` that finds none starts it.
- The decider (§13.3) starts at the first job that calls it (d1: 20-24 s to load) and is unloaded after
  `[decider] idle_s` with no call: its worker process is killed, which frees all its GPU memory, so the GPU is free for
  other work between questions. The daemon never imports torch.

## 3. The home

```
~/.mms/                           # MMS_HOME or --home
  mms.toml                        # the sources and limits (§4): the owner's, and the daemon writes it too (§4.4)
  consent.json                    # written by `mms approve`, and pruned by the daemon when a source goes (§4.3)
  index.sqlite                    # the index (§5), WAL
  daemon.lock, index.lock, consent.lock   # held while in use, never deleted
  daemon.json                     # the running daemon's address (§11); removed at a clean stop
  daemon.log                      # a detached daemon's output, cut at 10 MB (one older file kept, daemon.log.1)
  daemon.stop                     # `mms stop` writes it; the loop watches for it
  streams/<source>/<folder>/      # segments: a folder per ffmpeg launch (its UTC start) or per program run
      000000.mkv ... list.csv     #   ffmpeg numbers them; never -strftime (local time)
  files/<source>/<YYYYMMDDTHHMMSSZ>-<sha8>-<basename>[.zst]   # versions, and recordings and pictures (remembered, or copied from a folder)
  spool/<stream>/<run>/           # program logs not yet ingested (§8)
  looks/                          # files `look` wrote, deleted after a day
  scripts/                        # the agent's copies of shipped scripts, which win over the shipped ones
  models/<name>/                  # the decision models `mms install` and `mms model use` download here by default (§13.3)
```

**Time.** Stored times are UTC seconds since the epoch (floats) from the host's wall clock. The API, MCP and the CLI's
`--json` give times as ISO 8601 in the host's zone with its offset, to the millisecond
(`2026-10-31T09:14:02.350-04:00`). They take "now", an age (`90s`, `30m`, `2h`, `7d`, `1w`, meaning that long ago),
ISO 8601 (no zone is local) or epoch seconds (`config.parse_when`).

## 4. Configuration

### 4.1 Sources

`mms.toml`:

```toml
[memory]
free_gb = 20                 # the floor: below it capture, ingest and remember stop, until 1 GB above it
jobs = 4                     # jobs that run at once; more wait their turn
job_max_s = 3600             # a job that runs longer is cancelled
remember_max_mb = 2000       # the largest file `remember` copies
# ffmpeg = "C:/tools/ffmpeg.exe"   # default: ffmpeg on PATH, else the binary from imageio-ffmpeg

[decider]
model = "C:/models/d1-3B"   # a model folder a backend claims, or python:<file.py> (§13.3); `mms install` sets one; none: a job that decides fails
idle_s = 600                     # unload the model after this long with no call
# url = "http://127.0.0.1:8741"  # where the owner serves the model (H2O only); loopback only; default: the backend's
# envs = "C:/models/envs"        # where `mms model use` and `prepare` build envs (§13.3); default: envs/ beside the model folder

[sources.screen]
kind = "screen"
device = "0"                 # the display
fps = 2
keep = { max_gb = 40, when_full = "thin" }

[sources.mic]
kind = "microphone"
device = "Microphone (Realtek(R) Audio)"
keep = { max_gb = 20, when_full = "thin" }

[sources.notes]
kind = "folder"
path = "~/notes"
glob = "**/*.md"
keep = { max_gb = 5, when_full = "drop_oldest" }

[sources.robot]
kind = "program"             # added by the daemon at the stream's first log (§8)
keep = { max_gb = 5, when_full = "thin" }
```

| Kind | Records | Options (default) | Approval |
| --- | --- | --- | --- |
| `screen` | a display, as video | `device` ("0"), `fps` (2), `max_width` (1920), `segment_s` (60) | yes |
| `microphone` | an audio input | `device`, `segment_s` | yes |
| `camera` | a video input | `device`, `fps` (the device's), `segment_s` | yes |
| `ffmpeg` | any ffmpeg input (written by hand only) | `input`, `preset` or `encode`, `media`, `segment_s` | yes |
| `folder` | each changed file under `path` as a version | `path`, `glob` ("\*\*/\*"), `rescan_s` (5), `copy_max_mb` (50) | no |
| `log` | each new line of the files under `path` as a record | `path`, `glob`, `rescan_s`, `time_field` | no |
| `command` | each stdout line of a program as a record | `command` (a list), `time_field` | yes |
| `program` | what `mms.log` sends (§8) | `segment_s` (60) | no |
| `push` | what `remember` writes; a table is needed only to set a cap | none | no |

- **Every source** takes `keep = { max_gb, when_full, max_days }`: no `max_gb` means no cap; `when_full` is
  `drop_oldest` (the default) or `thin`. It also takes `project`: the project its rows belong to (§5.2); a source with
  none is ambient.
- **`time_field`** names the JSON field that holds a record's time: epoch seconds, or ISO 8601 (no zone is local).
  Without it, a record's time is when its line was read: a `"t"` field is not read unless named, since a program's `t`
  may be sim time.
- **Named devices** expand to an ffmpeg input in `capture.py` (`capture.opens(source)`):

  | Kind | Windows | Linux |
  | --- | --- | --- |
  | screen | `-f lavfi -i ddagrab=output_idx=<device>:framerate=<fps>` | `-f x11grab -framerate <fps> -i <device, default $DISPLAY>` |
  | microphone | `-f dshow -i audio=<device>` | `-f pulse -i <device>` |
  | camera | `-f dshow [-framerate <fps>] -i video=<device>` | `-f v4l2 [-framerate <fps>] -i <device>` |

  with the screen, audio and camera presets of §6.
- **`[decider] url`** is `http://` with the host `127.0.0.1`, `localhost` or `[::1]`, a port, no credentials, and no path,
  query or fragment (`config.loopback`); anything else is a `config_error`. A model that is not served, and `python:`,
  refuse a `url` (a `decider` error), so a stale one is not ignored after a switch. `envs` is an absolute path or starts
  with `~`.
- `config.py` parses the file into `Config` and `Source` once, at the boundary. A source's name is
  `[A-Za-z0-9_-]{1,64}` (`config.NAME`), since it is also a folder name.

### 4.2 Live reload

- The daemon checks the size and mtime of `mms.toml` and `consent.json` on every loop. When either changed, it
  loads them again and hands the new `Config` to each part's `apply(config)`:
  - removed and changed sources stop;
  - new ones start at the next step;
  - unchanged ones are left alone;
  - a new `[decider] model`, `url` or `envs` unloads the old one.
- A file that does not parse is refused: the daemon keeps the last good config, logs the error, and `status` gives it
  as `config_error` until the file is fixed. A daemon that starts with a bad file runs with no sources, so that its API
  can say why.

### 4.3 Consent

- **What needs it:** every source that opens a device or starts a program: `screen`, `microphone`, `camera`, `ffmpeg`
  and `command`.
- **What an approval binds:** the source's name to a fingerprint, the SHA-256 of what it opens or runs:
  - `kind` and `device` for screen, microphone and camera;
  - the `input` list for ffmpeg, as JSON;
  - the `command` list for command, as JSON (so `["python", "a b"]` and `["python a", "b"]` differ).

  A change to any of these needs a new approval. A change to caps, `fps` or other options does not.
- **Removing a source takes its approval with it:** whenever a config is applied, capture's `apply` drops the
  approvals of names that are not in the config as a source that needs one. A source added later under that name starts
  unapproved.
- **Where it lives:** `consent.json`, written by `mms approve` and the daemon's prune, every read and write under `consent.lock`:
  `{"<source>": {"fingerprint": "<sha256>", "what": "microphone: Microphone (Realtek(R) Audio)", "approved": "<ISO>"}}`.
- **`mms approve <source>`** prints what the source opens or runs, then asks the owner to type the source's name. It
  refuses unless stdin and stdout are terminals. `mms approve <source> --revoke` removes an approval, with no terminal
  needed.
- **Until approved,** the source's state is `needs_approval`, with the command the owner must run, and nothing starts.
- **No MCP tool and no API call approves:**
  - `source_set` writes only `[sources.<name>]` in `mms.toml`;
  - approval lives in a file the API never writes.
- **The limit:** an agent with a shell (owners often run agents with permissions bypassed) can run `mms approve` itself, through
  a pseudo-terminal, or edit `consent.json`. The terminal check and the typed name make that a deliberate act against
  the server instructions and the skill, which both tell the agent to ask the owner. They do not prevent it, and
  nothing in user space could.
- `consent.py` is a leaf: `fingerprint(what)`, `approved(home, name, what) -> bool`, `record(home, name, what)`,
  `revoke(home, name)` and `prune(home, names)`. Capture and ingest pass it what a source opens or runs.

### 4.4 The writer: `source_set` and new streams

- `config.set_source(home, name, table)` and `config.remove_source(home, name)` rewrite one `[sources.<name>]` table
  with tomlkit and keep the rest of the file as it was (comments, order). The new text is parsed by the same loader
  before it is written (to `.part`, then renamed), so a refused change leaves the file untouched. The daemon serializes
  its own writes; the owner's edits are not locked out, and the last save wins.
- **`source_set` refuses:**
  - a kind other than screen, microphone, camera, folder, log, command or program (`bad_request`);
  - a name whose table is of another kind, or of kind `ffmpeg` or `push`, which the owner changes by hand
    (`wrong_kind`);
  - a `max_gb` below the bytes the source holds now (§9's count), or a `max_days` shorter than the age of its oldest
    item or record (`would_delete`), so that no agent makes the janitor delete history (the owner may, by hand);
  - a `max_gb` or `max_days` of an existing source below what it is (a source with none counts as unlimited), so a cap
    can stay or rise through `source_set` and never fall (`would_delete`, with a hint that the owner edits
    `mms.toml`);
  - a folder or log `path` that is the MultimediaSearch home, holds it (a source at `~`) or is inside it (`bad_request`,
    `field: path`): nothing the home holds is scanned or remembered;
  - any change while `mms.toml` does not parse (`config_error`).
- **`source_set` merges the fields it is given.** A new source needs `kind` and `max_gb`. For one that exists, every
  option, `when_full`, `max_gb` and `max_days` not given keeps its value, so a change to a cap or `fps` leaves the
  consent fingerprint alone. `project` is the caller's when the source is made (folder, log and command) and is never
  moved by a later edit.
- **The reply does not guess the daemon's state.** It is `{"source": {...}, "pending_reload": bool}`, and while the
  running daemon has not yet applied the change `pending_reload` is true and `source.state` is `"pending_reload"`
  (a source that needs approval still carries `approve`). A removal replies
  `{"source": {"name", "removed": true}, "pending_reload": bool}`. The caller reads `status` for the applied state.
- **`remove`** deletes the table. The source's history stays, outside every cap, since the janitor caps configured
  sources only.
- `config.set_decider(home, model=, url=)` does the same for `[decider]` (it is what `mms model use` writes, §13.3).
- **A program stream's first log** adds `[sources.<stream>] kind = "program"` with `keep = { max_gb = 5, when_full =
  "thin" }` (§8), unless a table of that name exists.

## 5. The index

### 5.1 Tables

```sql
items    (id, source, kind, t0, t1, path, bytes, sha256, name, keep, in_place, state, meta, project, session, expires)
records  (id, source, t, data, project, session, expires)
labels   (id, source, item, t0, t1, by, text, meta, made, project, session)
clock    (id, source, run, wall, t)            -- a program run's own time against the wall clock (§5.4)
cursors  (source, name, offset, size, mtime, sha256)
launches (id, source, folder, started, ended, lines, first_start)
projects (id, name, root, seen)                -- for display; rows carry only the id
sessions (id, project, client, started)
-- records_fts over records.data, labels_fts over labels.text: FTS5 with external content
```

- **`items` holds the files.**
  - `kind = 'video' | 'audio' | 'image'`: a segment covering `[t0, t1)`, captured, encoded from a program's frames, or
    a recording placed at its real time; an `image` is a picture at one instant (`t0 = t1`), placed the same way.
    `path` is relative to the home. A video segment has `meta.sound` when it has an audio stream (§6, §7). A captured
    segment has no `meta.media`; a placed recording or picture has it and `meta.time_from` (§7).
  - `kind = 'file'`: a version of `name` (a watched path or a remembered name).
    - `t0` is when it was seen.
    - `t1` is when the next version replaced it, or when the file vanished; null while it is current.
    - `meta.media` is its MIME type.
    - `in_place = 1` when `path` is the owner's own absolute path. Such a file is never deleted or counted against a
      cap, and has no history: one row, with `t0` the last change seen and no `sha256`. When the file vanishes, or a
      copied version replaces it, the row ends and goes `gone`.
  - `keep = 1` exempts an item from caps, thinning and expiry. `state` is `ok` or `gone`; readers and `find` skip
    `gone`.
- **`records` holds small timed JSON text:** log and command lines, `remember` notes and program records, under 10 KB
  each. Freed pages are reused, so a capped table stops growing without `VACUUM`.
- **`labels` is text on a time range,** by anyone: `ask`'s sightings, transcripts, the agent's notes. `by` says who.
  A label with `keep` marks `keep = 1` on the overlapped items of the source it names (`keep` with no source is
  refused). `mms keep` sets or clears the mark on a range with no label.
- **What may be kept is limited by the source's cap:** a label with `keep`, or `mms keep`, is refused with `too_big`
  when the bytes of the source's kept items (`keep = 1`, not in place), with those it would mark, would exceed its
  `keep.max_gb`. Its message gives both numbers. A source with no `max_gb` has no limit, and clearing is never refused.
  The check and the mark are one transaction, so two sessions cannot together pass the cap.
- **A label that names an item goes when the item does** (the janitor deletes both in one transaction). A label with
  no item is a small note, and stays.
- **`expires`** is set by `remember`; the janitor deletes the row and its file after it (§9).
- **No migration:** the index records `PRAGMA user_version = 2`, and a home with an index of another version is
  refused with a message to move it aside. No home with data exists (checked 31 Oct).
- `store.py` owns the schema and every query.

### 5.2 Projects, sessions and scope

- **A row's `project` is its writer's:**
  - for a capture, folder, log or command source, its `project` key (none for ambient sources);
  - for anything written through the API (`remember`, `label`, a job's labels), the caller's project;
  - for a program's logs, the project of the program's run (§8).

  `session` is the API caller's session, and null otherwise.
- **The default read** is the ambient rows (`project IS NULL`) and the caller's project. `scope = "all"` reads every
  row. `sources` narrows either to the sources named. A caller with no project reads the ambient rows only.
- `store.Scope(project, all)` is the one filter every read takes (`find`, `look`, the reader, and so every job).
- **Every result says what was searched:**
  `searched = {now, since, until, scope, project: {id, name} | null, sources: [name, ...]}`.

### 5.3 Selecting by time

The one rule every reader and `find` uses:
- segments that overlap the range, and pictures whose instant falls in it (a picture is a segment of no length, so a
  range that does not hold it does not read it);
- versions that were current at some time in it (as of: a note saved Monday is still the note on Tuesday);
- records whose `t` falls in it;
- labels that overlap it.

**As of** `t` is the same view at one instant: the latest version of each name with `t0 <= t < COALESCE(t1, +inf)`.
`find` lists what changed in a range, so it keeps the versions whose `t0` falls in it and not those that only went on.

### 5.4 The clock

- A program has two clocks: the wall time of each `log` call, and its own `t` (sim time, say).
- Rows take the wall time.
- `clock` keeps, for each batch a run sends, the first line's `(wall, t)`.
- `store.program_time(source, wall) -> (run, tp) | None` interpolates `tp` between the clock rows around `wall`, or
  carries the last one on at the wall clock's rate.
- `store.wall_time(source, run, tp)` is the inverse, for `look`.
- Two runs of one stream at the same time are told apart only by the nearer clock row.

## 6. Capture (ffmpeg sources: screen, microphone, camera, ffmpeg)

One ffmpeg child per source, with named devices (§4.1), consent (§4.3) and `apply` (§4.2).
- **Each launch:**
  - The capture loop makes `streams/<source>/<launch>/` and a `launches` row, with `started` taken just before ffmpeg
    starts.
  - It runs ffmpeg there with `-f segment -segment_time <segment_s> -segment_format matroska -reset_timestamps 1
    -segment_list list.csv -segment_list_type csv %06d.<ext>`: `.mkv` for video and `.mka` for audio.
- **Presets** (checked on 30 Oct with the bundled ffmpeg 7.1 and synthetic input):
  - **screen:**
    - filters `hwdownload,format=bgra,` for `ddagrab`, then `scale='min(iw,<max_width>)':-2,format=yuv420p,
      mpdecimate=max=119`;
    - `-fps_mode vfr -c:v libx264 -crf 30 -tune zerolatency -force_key_frames expr:gte(t,n_forced*10)`.

    A real 3840x2400 grab at 2 fps made yuv444p at 310-410 KB per 2 s when the chain left bgra, hence the scale and
    yuv420p.
  - **camera:** `-c:v libx264 -crf 28 -tune zerolatency -force_key_frames expr:gte(t,n_forced*10)`.
  - **audio:** `-af aresample=async=1 -c:a libopus -b:a 24k`.
  - **`encode`** (ffmpeg kind) replaces everything a preset adds between input and segment options.
  - The key frame interval is `min(10, segment_s)` s, since a segment ends only on a key frame. A shorter one would
    speed only random access, for 7-10% more bytes (§16).
  - Frames are stored at these sizes and cut smaller for the model only when read (§16).
- **Indexing:** the loop tails `list.csv` and adds an item for each new complete line, with
  `t0 = started + (start - first_start)` and `t1 = started + (end - first_start)`. Time is good to about the startup
  delay (around a second).
  - A video segment gets `meta.sound` when PyAV finds an audio stream in the closed file (`media.has_sound`, as
    `of_file` does for a file): ffmpeg's arguments cannot say, since an `ffmpeg` source's input and `encode` are the
    owner's. It is one open of the file, 1 to 6 ms a 60 s segment, made before the row is added; a segment indexed by
    reconcile is probed the same way. An audio segment needs none.
- **Stopping** writes `q` to ffmpeg's stdin and waits, so the open segment closes and gets its line.
- **Every child dies with the daemon** (`children.py`): a kill-on-close Job Object on Windows, `PR_SET_PDEATHSIG` on
  Linux. Otherwise ffmpeg would go on writing segments nobody indexes, holding the device the next one needs.
- **No child opens a window:** on Windows every child of the daemon starts with `CREATE_NO_WINDOW`. The detached
  daemon runs under pythonw, which has no console, so a console program it starts would otherwise open its own.
- **Restarts:** at UTC midnight; when ffmpeg exits, with backoff; not while the floor is breached.
- **Reconcile** at startup, before any ffmpeg runs, for each launch with no `ended`: index list lines never indexed;
  index a media file no line names from its decoded duration if PyAV reads it, else delete it. The same settling runs
  when an ffmpeg ends.
- **`capture.devices(ffmpeg)`** lists what `source_set` may name: on Windows, dshow audio and video devices
  (`-list_devices true -f dshow -i dummy`) and the displays; on Linux, `/dev/video*`, pulse sources and `$DISPLAY`.
  A failed listing is given with its error, not left empty.

## 7. Ingest (folder, log, command, push)

- **What a file is** (`media.py`) comes from its bytes, never its name: known signatures, then a container's streams
  (video unless its only picture is cover art; `sound` when a video has an audio stream), then git's text rule (a BOM,
  or UTF-16's pattern of NULs, else no NUL in the first 8,000 bytes, runs of 32 or more being padding). It is written
  once, to `meta.media` (and `meta.sound`), when a version, a recording or an in-place file is stored. Text is
  `text/plain` (an SVG is `image/svg+xml`, read as text); a type the decider does not read (d1 reads no PDF, Office
  file or archive; another decider may take them whole, §13.3) is kept and found by name, and named in the job's
  warning.
  - A weak signature (a bare `moov`, `mdat`, `wide` or `free` box at byte 4, or MPEG-TS's sync byte at every 188th)
    counts only when PyAV finds a stream in the file, or text such as "I'm free on Thursday" would be video; a
    strong one (`ftyp`, EBML, OggS, RIFF) keeps its container's video type for a broken file, which the reader warns about.
  - An AVI, FLV or MPEG-PS with sound and no picture is audio. A file the reader expects a picture in and finds none
    (say an MKV of subtitles) is warned about, and the scan goes on.
  - Text is decoded as UTF-8 with U+FFFD for a bad byte, unless over half its non-ASCII characters are bad bytes (a
    legacy file): then cp1252.
- **folder:** every `rescan_s`, glob the path.
  - For each file whose size or mtime changed (kept in `cursors`), read it once, streaming it into `files/` (zstd when
    it is text) while it is hashed. A new hash adds a version and closes the last; the same hash discards the copy.
    The check and the write are one transaction, so concurrent writers leave one open version.
  - Above `copy_max_mb` the file is indexed in place, with no copy and no hash (§5.1): a version, whatever it holds.
  - A copied file whose content is video, sound or a picture is a **recording**, as `remember` makes one (below), at its
    real time (never a given `t`): a segment or a picture stored once by SHA-256 within the source, and never closed, so
    a recording whose file later vanishes stays. A video or sound PyAV finds nothing in is kept with no length, and the
    reader reports it. A copied file of any other type is a version.
  - A recording whose mtime is less than `2 * rescan_s` (default 5 s) ago may still be being written: it waits for a
    later pass, with no cursor set, so a slow recorder leaves one copy and not a partial one per pass. A future mtime is
    settled. A text version needs no wait: the next one supersedes it.
  - A pass over a big tree takes several steps of the loop: each looks at 500 files and copies 256 MB, then goes on
    where it left off, so the loop and the janitor are never held. What vanished is found when a pass ends.
  - A `path` that is the MultimediaSearch home, holds it or is inside it is never scanned (the source's state is `error`).
  - **Every write passes one gate,** `Store.reserve(n, source)`: `n` more bytes may not take free space below
    `free_gb` (`floor`), and one write may not be larger than its source's whole cap (`too_big`). Copying a file in
    (each file of a folder), a record, and a spool segment's encode all call it. A source whose write is refused shows
    the reason in `status` (`error`) and looks again in 5 s.
  - A file that vanished ends its version.
  - A watched path that is not there (an unmounted drive) is not scanned, so nothing ends and nothing is copied again
    when it returns. A single watched file that is deleted is not told apart from this.
- **log:** read new complete lines from the saved offset, 1 MB at a time, up to 16 chunks of a file in one pass. Each
  chunk's records and its cursor commit in one transaction, so a kill never repeats a line. A line is a record: its
  JSON if it parses as an object, else `{"line": text}`. A line longer than a chunk is cut there; a file that shrank
  starts again from 0.
- **command:** one process per source, restarted with backoff when it exits. Each stdout line is a record, as for
  log.
- **Lines** of a log or a command are decoded as UTF-8, else cp1252 (a log that starts with a UTF-16 BOM as UTF-16), and
  keep what a terminal shows: ANSI escape sequences go, and after a lone `\r` only what follows it stays.
- **The time of a log or command record** is its `time_field` (§4.1), else when the line was read.
- **push (`remember`, §11):**
  - a note is a record `{"text": ..., "name"?: ...}` at its `t`;
  - a file is a version;
  - a folder is a version of each file under it, named `<folder>/<relative path>`;
  - a recording (a file whose content is video, audio or a picture) is copied to `files/<source>/` and indexed at its
    real time: a video or sound as a segment item from its real start over its length, a picture (`kind = 'image'`) at
    one instant. An SVG is read as text (above), so it is a version. A picture is no version: a version holds from its
    `t0` until replaced, so a 2019 photo held as one would be read by every later range and `since` would never narrow
    it. An animation or a multi-page TIFF is one instant too; the reader plays its frames on their own clock from `t0`
    (§16).
  - **The time of a recording** is, in order (`time_from` says which):
    1. `t`, if the caller gave it (`given`);
    2. the file's own time (`metadata`): a video's or sound's `creation_time` tag, a picture's EXIF DateTimeOriginal, else
       DateTimeDigitized (with its OffsetTime tag when it has one, else read in this machine's zone). A blank, zeroed,
       malformed or unreadable one (a HEIC without a plugin, a truncated file, no EXIF) is none, and a picture whose EXIF
       cannot be read is still stored;
    3. the file's mtime (`mtime`), less its length for a video or sound, which then ends at the mtime;
    4. now (`now`), less its length, only when nothing better is believable.

    A time is believed in 2 and 3 only if the recording lies within [1970, now]: not ending after now (a camera clock set
    wrong, a bad copy), and not before the epoch. `time_from` is kept in the item's `meta`. Storing a recording under the
    name of a current version ends that version at now (a name that changes type). A video or sound PyAV finds nothing
    in is `bad_request`.

  A file whose content equals the source's current version of that name, or a recording already stored in that source
  (by SHA-256), is reported `unchanged`, not stored again.
- **Below the floor** ingest stops: commands end and start again on resume, watches are not scanned (their cursors
  catch up), the spool waits, and `remember` refuses.
- **Spool:** §8.

## 8. Program logs

### 8.1 The client (`mms.log`)

```python
import mms
mms.log("robot", {"event": "fell", "height_m": 0.12}, t=sim_t)        # a record
mms.log("front_cam", frame=rgb, t=sim_t)                                 # a frame: uint8, HxWx3 RGB or HxW grey
mms.flush(timeout_s=5.0) -> bool            # True once everything queued is in the spool
mms.stats() -> dict                         # {"queued", "written", "dropped": {"queue_full", "spool_full",
                                            #  "floor", "refused"}, "write_errors", "last_error"}
log = mms.Logger(home=None, project=None)        # the same calls as methods; the module functions share one Logger
```

- **`log` returns in microseconds and never blocks.** It:
  - checks the stream name (`config.NAME`);
  - serializes `data` to JSON at the call, so later changes to the dict are not seen;
  - copies the frame;
  - stamps `wall = time.time()`;
  - queues the result.

  It raises `ValueError` for a bad name, data that is not a JSON object, or a frame of the wrong shape or type. It
  never raises for the queue, the disk or the daemon.
- **A background thread** writes a batch every 0.2 s, or at 1,000 lines or 32 MB of frames, and encodes each frame as a
  JPEG at quality 90 (kept by §16).
- **Drops, counted in `stats()`, the first of each kind printed to stderr:**
  - the queue holds 64 MB, and `log` drops what does not fit;
  - the thread stops writing a stream while the spool holds over 2 GB (the daemon is down), the disk is below
    `[memory] free_gb` (read at start and every 60 s), or `spool/<stream>/refused.json` exists.
  - Any other write error (permissions, a bad path, an encoder) is not "the spool is full": its lines are counted in
    `write_errors`, with the error in `last_error`, and the first is printed. A full disk is `spool_full`.
- **`close()`** flushes and writes `end.json`; it runs at exit.
- **The run's project** is found from the program's working directory, as the adapter finds its own (§12.2), unless
  given.

### 8.2 The spool (the contract between the client and the daemon)

```
spool/<stream>/<run>/        # run = <UTC stamp>-<pid>-<4 hex>, one per Logger and stream
  run.json                   # written first: {"v": 1, "stream", "run", "pid", "project": "<id>" | null,
                             #                 "started": <wall>}
  <seq>.jsonl                # a batch; seq is 8 digits from 00000001; written as <seq>.jsonl.part, then renamed
  <seq>-<i>.jpg              # the frames of batch <seq>, i from 000, written before the batch is renamed
  end.json                   # written at close: {"ended": <wall>, "dropped": <n>}
spool/<stream>/refused.json  # written by the daemon: {"reason": "..."}; the client stops writing the stream
```

- Each line of a batch is one of:
  - `{"wall": <float>, "t": <float> | null, "data": {...}}`: a record;
  - `{"wall": <float>, "t": <float> | null, "frame": "<seq>-<i>.jpg", "w": <int>, "h": <int>}`: a frame.
- A batch is complete once renamed, and the daemon reads nothing else. The client makes the folder and `run.json`
  again if the daemon removed them.

### 8.3 Ingesting the spool (daemon)

- **A new stream** gets its table (§4.4). A stream whose table is of another kind is refused: the daemon writes
  `refused.json` and leaves the run alone.
- **Records:** batches after the run's records cursor (`cursors` row `(stream, <run>)`, offset = the last seq read)
  are read in order. Their records (`t` = wall, the run's project) and one clock row per batch commit with the cursor
  in one transaction, so a crash never repeats one.
- **Frames:**
  - They wait in the spool. A segment holds the frames of one size within `segment_s` of its first, so a backlog after
    an outage is cut into segments too. It is encoded when a later frame is past that span, or none came for 10 s, or
    the run ended: the daemon writes an ffconcat list (each frame with its `duration` to the next; the last takes the
    median gap), reserves the frames' bytes (§7), and runs one ffmpeg (a child, §6) to encode them with the camera
    preset into `streams/<stream>/<run>/<n>.mkv.part`.
  - On exit 0 it renames the file and adds the segment (`t0` = the first frame's wall, `t1` = the last frame's wall
    plus its duration, `meta.run`; no `meta.sound`, since the segment is frames alone), with the frames cursor
    (`(stream, <run>:frames)`), in one transaction.
  - A frame of another size starts a new segment. The segment's timestamps are the frames' wall times, to the
    millisecond, so the reader needs nothing new.
- **Cleanup:** batch files and frames are deleted once both cursors are past them. A run folder is removed, with its
  cursors, once nothing is left of it and it has an `end.json`, or its records cursor has not moved for a day with
  nothing unread (the cursor's time, kept in `cursors.mtime`, and never the files' times). A backlog read over many
  steps is never idle.
- **A stream's table** is settled once and remembered, until the table goes (`source_set remove`, or a hand edit) or
  becomes another kind: the next log of that stream then makes its table again, or is refused.
- **A crash** leaves at most a `.part` segment, deleted at start; its frames are still in the spool and are encoded
  again.
- **A batch that does not parse** moves to `spool/_bad/` with its error beside it, and is logged.

## 9. Retention (the janitor)

It runs every 30 s, and once at startup before capture.
1. **The floor:** below `free_gb`, the daemon logs it and `status` shows it, and everything that writes stops: capture
   pauses, ingest stops (§7), and `remember` refuses. It resumes 1 GB above the floor.
2. **Retainable** is the one rule for what a cap counts and may delete: an item with `keep = 0`, not in place, and not
   the newest of its source. Records are all retainable. Caps, thinning, dropping and `max_days` all use it.
   - Kept items are outside the count. What may be kept is limited instead, when it is marked: kept bytes may not pass
     the source's `max_gb` (§5.1), so a source holds at most about twice its `max_gb`.
   - An owner who lowers `max_gb` by hand below the kept bytes deletes nothing kept; further marks are refused.
3. **Per-source caps.** A source costs the bytes of its retainable items, plus its records' cost (below), plus 100
   bytes per clock row. Over `max_gb`:
   - **`drop_oldest`** takes retainable items, records and clock rows together by time, sums their cost from the
     oldest, and deletes everything before the one time at which the sum reaches the excess;
   - **`thin`** deletes the segment (or picture) most crowded for its age (the smallest
     gap to its neighbours against its age goes first). Pictures at one instant (edited copies) have no gap, so they go
     first. Unretainable segments stand in the gaps. Records are only dropped,
     never thinned; what thinning cannot reach is dropped as above.
4. **`max_days`:** retainable items, records and clock rows older than it are deleted. An item's age is its event time,
   so a recording or picture placed at an old time is old at once.
5. **Expiry:** items and records past `expires` are deleted, unless kept.
6. **Housekeeping:** `looks/` and `spool/_bad/` files older than a day go; so does any `.part`, `.ffconcat` or `.err`
   file under `streams/`, `files/` and `spool/` older than an hour (what a crash left); `daemon.log` over 10 MB becomes
   `daemon.log.1`. A label goes with the item it names (§5.1).
7. **Deleting** is unlink, then the row. An unlink that fails because a reader has the file open is retried on the
   next pass. A version's file is unlinked only when no other item points at it.

**What a record costs a cap** is `3 × the UTF-8 bytes of its text + 200` bytes, for the row, its two indexes and
FTS5, which keeps a delete marker for every row dropped until it merges. Measured on 30 Oct 2026 (SQLite 3.46, WAL, the
file after a checkpoint), the growth of `index.sqlite` per row, with its multiple of the row's text:

| Records | Text per row | Filled once | Replaced 6 times over, 20 rows a transaction |
| --- | --- | --- | --- |
| exec JSON, a temperature | 50 B | 150 B (3.0x) | 342 B (6.9x) |
| tailed log line | 112 B | 257 B (2.3x) | 533 B (4.8x) |
| `mms remember` note | 624 B | 958 B (1.5x) | 2001 B (3.2x) |

A fresh fill fits `1.4 × text + 100`; the settled table fits `3 × text + 200`, which the cap uses. FTS5 `optimize`
would narrow the gap but rewrites the whole index, so it does not run.

## 10. The daemon

- **`mms start`** runs the daemon in the terminal until `mms stop` or Ctrl+C, logging to stdout.
- **`mms start --detach`** starts it in the background, as nobody's descendant, so that it outlives the agent whose
  adapter started it. An agent's client may put the adapter in a kill-on-close job object (the MCP Python SDK's does),
  and `CREATE_BREAKAWAY_FROM_JOB` is accepted but does nothing there.
  - Windows: through WMI (`Win32_Process.Create`, by PowerShell's `Invoke-CimMethod`): the new process belongs to
    WmiPrvSE, outside every job. It has none of the caller's environment, so `--home` is on its command line, and it
    runs `pythonw` beside `sys.executable`, so that no console opens.
  - Linux: `start_new_session`.

  Its output goes to `daemon.log`. `--detach` waits until the daemon answers `status`, up to 30 s, and exits 0. A
  daemon already running or starting counts: it waits for that one the same way. It exits 1 with the log's tail if the
  daemon exits or none answers.
- **Whether a daemon lives is the lock's to say,** never HTTP's: `daemon.lock` is held for the daemon's whole life and
  dropped by the operating system when it dies. `client.connect` takes the lock as a probe; a free lock is no daemon,
  and a `daemon.json` left behind is deleted under that lock. HTTP only says whether a live daemon is ready. A probe
  holds the lock for microseconds, so a starting daemon, which waits 2 s for it, never loses it.
- **Startup:**
  1. the lock;
  2. `children.contain()`;
  3. the config (a bad one gives no sources and a `config_error`);
  4. `Store`;
  5. a janitor pass;
  6. capture's reconcile and the spool's `.part` cleanup;
  7. the API server, then `daemon.json`;
  8. the loop.
- **The loop** runs every `loop_s` (0.25 s):
  - the config check (§4.2);
  - `capture.step`;
  - unless the floor is breached, `ingest.step` (folder, log, command, spool);
  - the janitor every `janitor_s`;
  - expired jobs dropped.
- **Threads:**
  - the loop;
  - the HTTP server, one thread per request;
  - one thread per running job;
  - the GPU worker;
  - the readers of ffmpeg stderr and command stdout.
- **Stopping** (`daemon.stop`, a signal): refuse new jobs, cancel running ones, close the API and remove
  `daemon.json`, stop capture (`q`), ingest and any encoder (10 s, then killed), end the decider's worker, and release the
  lock. A decider's worker is a child like ffmpeg (§6): it dies with the daemon even when that is killed.
- **The GPU queue** (`gpu.Gpu`, in `gpu.py`): one worker thread owns the decider.
  - `gpu.decide(state, questions, media, cancelled)` queues a request and blocks until its answer.
  - Requests run in arrival order. Each job thread has at most one waiting, so concurrent jobs take turns window by
    window.
  - The worker loads the model at the first request and unloads it after `idle_s` with none. Loading starts the
    decider's worker process, or waits for the owner's server; unloading kills the worker, which frees all its GPU
    memory (the owner's server keeps its own). An unload runs outside the queue's lock, so a slow one holds no request.
  - A load that fails fails the waiting requests with `decider` (never as a window that does not fit); the next request
    tries again.
  - A decider that fails with an `ApiError` (its worker crashed, hung or wrote what is not a frame) is dropped and unloaded;
    the request fails, and the next one makes the decider again and loads it through the backend's `load()`.
  - `gpu.about()` gives the model's `About` (§13.3) without loading it: the decider is constructed to read it (cheap: it
    reads the model's files and checks its env, and starts nothing), and the result is kept per model, `url` and `envs`.
    With no model configured, or one that cannot be made, it raises `decider` at once. `gpu.check(...)` likewise gives the
    model's own rules for a set of questions (§13.3), with nothing loaded.
  - A cancelled job does not wait: its waiting request is dropped, and one the worker already has (a model may be
    loading) is finished for nobody while the job reports `cancelled` at once.
  - A new `[decider] model`, `url` or `envs` takes effect between windows: the worker unloads the old model before its next
    request, even with others waiting.
  - `status()` gives `{model, state: unloaded | loading | ready | failed, error, waiting}`.
  - It sends the decider one window at a time. Batching windows (d1's `system_one_batch`) is deferred (§16); when it
    comes, it goes here and in the backend and worker file of the model that batches, and nowhere else.
- **Jobs** (`jobs.py`): a job runs a script (§13) in its own thread with the decider handed in through the GPU queue.
  - At most `[memory] jobs` run at once; more wait as `queued`.
  - The script is loaded from `scripts/<name>.py` in the home, else the shipped one, fresh for each job, so an edit
    counts from the next job.
  - A job ends when its script returns (`done`), raises (`failed`), or is cancelled (`cancelled`). The daemon cancels
    one that runs past an hour.
  - A job is kept for an hour after it ends (at most 200 kept), in memory only. A daemon restart forgets jobs:
    `jobs.get` then answers `not_found`, and the agent asks again. Sightings it labelled stay.

## 11. The daemon API (the contract)

- **Discovery:** the daemon binds an HTTP server to `127.0.0.1` on a port the OS picks, then writes `daemon.json`
  (`.part`, then renamed): `{"api": 1, "pid": <int>, "port": <int>, "token": "<64 hex>", "started": "<ISO>"}`.
  Binding to 127.0.0.1 raises no firewall prompt. A stale file is known by a refused connection or a 401.
- **Calls:** `POST http://127.0.0.1:<port>/v1/<op>` with `Authorization: Bearer <token>` and a JSON object body: the
  op's parameters and `caller`.
  - A result is HTTP 200 and a JSON object.
  - An error is a 4xx or 5xx status and `{"error": {"code", "message", "field"?, "hint"?}}`.
  - An unknown field is refused (`bad_request`, naming it in `field`).
- **`caller`:** `{"session": "s-<12 hex>" | null, "project": "<id>" | null, "project_name": <str> | null,
  "project_root": <str> | null, "client": <str> | null}`. The daemon records a project's name and root, and a
  session's client, the first time it sees them. The CLI's caller has `session: null` and `client: "mms"`.
- **Paths** in parameters are absolute. The adapter makes them so against its own working directory.

**Errors:**

| Code | Status | Meaning | The caller |
| --- | --- | --- | --- |
| `bad_request` | 400 | a parameter is wrong or unknown; `field` names it | fixes it and calls again |
| `unauthorized` | 401 | no token, or not this daemon's | reads `daemon.json` again |
| `not_found` | 404 | no such op, item, source, job or path; `field` says which | |
| `wrong_kind` | 409 | the source exists with another kind | uses another name |
| `would_delete` | 409 | `source_set` would make the janitor delete history | asks the owner |
| `config_error` | 409 | `mms.toml` does not parse; `message` has the error | asks the owner |
| `too_big` | 413 | a file over `remember_max_mb` or a folder over 1,000 files; a `look` at a file over 64 MB (`field: item`); a `keep` past the source's `max_gb` (§5.1) | |
| `floor` | 507 | free space is below the floor | asks the owner |
| `decider` | 503 | no model is configured, or the decider could not load or failed (no GPU, no weights, no backend that reads the folder, an env not prepared, a server that does not answer, a worker that crashed or hung) | asks the owner |
| `internal` | 500 | a bug; the daemon log has the traceback | reports it |

**Ops.** Each op's parameters are the fields below; all are optional unless marked. "Repeats" says what calling it
again with the same body does.

| Op | Parameters | Result | Repeats |
| --- | --- | --- | --- |
| `find` | `text`, `since`, `until`, `sources`, `scope`, `types`, `as_of`, `limit`, `after` | `{results, next, searched}` | reads only |
| `look` | `item` + `at`, or `source` + `t`, or `source` + `run` + `tp`, or `source` + `since` + `until`; `offset`; `region`; or `moments`: up to 8 of these | one moment, a sheet, or `{moments}` (below) | writes a file in `looks/` |
| `remember` | `note` or `path` (one, required); `t`, `expires`, `name`, `source` | `{stored, unchanged, source, expires}` | a note is stored again; a file or recording is `unchanged` |
| `label` | `text`*, `since`*, `until`, `source`, `keep` | `{label, kept}` | adds another label |
| `keep` | `source`*, `since`*, `until`, `clear` | `{items, keep}` | same effect |
| `status` | `devices` (false) | below | reads only |
| `source_set` | below | `{source, pending_reload}` | same effect |
| `jobs/start` | `script`*, `params` ({}), `wait_s` (40) | a job | starts another job |
| `jobs/get` | `job`*, `wait_s` (40) | a job | reads only |
| `jobs/cancel` | `job`* | a job | same effect |

**`find`:**
- `text`: words that must all match: FTS5 over records and labels, and version names. A word is a run of letters and
  digits. Blank text is no text; text with no word in it (`"`, `*`) is `bad_request`, `field: text`.
- `since` and `until`: a range (default all, and now).
- `sources`: names. `scope`: `default` or `all` (§5.2).
- `types`: some of `item`, `record` and `label` (default all three).
- `as_of`: instead of a range, the version of each file current at that time.
- `limit`: 1-200, default 50.
- `after`: the `next` of an earlier call: the same call, one page on.
- **Results** come newest first by `(t0, type, id)`, the order a page ends in exactly: `next` is an opaque string, or
  null when the limit did not cut the results. The next page is the same call with `after` set to it. Each row is
  given once, however many share a time. Times are kept exact; the ISO strings are for display. Each result:

  ```
  {"type": "item" | "record" | "label", "id", "source", "t0", "t1", "project": <name> | null,
   "look": {"item", "at"} | {"source", "t"} | null,
   item:   "kind": "video" | "audio" | "image" | "file", "name", "media", "bytes", "keep",
   record: "text": <the first 300 characters of data>, "run"?, "tp"?,
   label:  "text", "by"}
  ```

**`look`:**
- `item` with `at` (seconds into it, default 0; a picture is an instant, so `at` matters only for an animation or a
  TIFF, where it picks the frame).
- Or `source` with `t`: the segment that covers `t`, else the newest (by `t0`) of the versions current at `t` and the
  source's latest picture at or before `t`, else the records within 30 s of `t`. A picture covers only its own instant
  and is shown at its own time; a time read back from a result is cut to the millisecond, so a picture within `t`'s
  millisecond counts. With none, `not_found`, whose `hint` names the nearest item and its time.
- Or `source`, `run` and `tp`: `t` from the run's clock (§5.4).
- Or `source`, `since` and `until` (after `since`): a **sheet** of the range (below). It needs all three and takes no
  `item`, `at`, `t`, `run`, `tp` or `offset` (`bad_request`, naming the field).
- `offset` (characters) pages through text. An `at` past the item's end is `bad_request`.
- `region`: `{top, left, bottom, right}`, fractions from 0 to 1 of the upright picture from its top left, as `ask` gives
  a box's edges. Only that part of a video frame or picture is shown, cut from the picture as `look` decodes it whole
  (at full size up to 64 MP; a larger JPEG at half or less): the PNG is the part at those pixels, `width` and `height`
  are its size, and the inline JPEG is the part enlarged or shrunk to 1568 px on its long side, so a small thing gets
  the whole inline picture. A video's `audio` is unchanged, and the result echoes `region`. Four
  edges from 0 to 1, top above bottom and left before right, or `bad_request` with `field: region`; so is a region on a
  moment that is not a frame or a picture (sound, text, a file, records). A sheet with a region shows that part of
  each frame (below).
- `moments`: a list of 1 to 8 moments, each the fields of one look (`item` + `at`, `source` + `t`, `source` + `run` +
  `tp`, `source` + `since` + `until`, `offset`, `region`), given with no other field. The result is `{"moments": [...]}`, one entry per moment in order: that
  moment's result, or `{"error": {...}}` (the error body) when that moment failed. A malformed `moments` (not a list,
  empty, more than 8, an entry that is not an object, another field beside it) is `bad_request` with `field: moments`.
  With `moments`, a text moment's page is 20,000 characters divided by the number of moments, so the call stays about
  as long; its `more` and `next_offset` say where to read on, with a single `look` and that `offset`.
- **No `scope`:** a handle names exactly one item or record, which `look` shows whatever its project. Scope filters
  what `find` and `ask` return (§5.2); it is not a security boundary.
- **Result:**

  ```
  {"kind": "image" | "audio" | "text" | "file", "source", "item": <id> | null, "t", "at" | null, "run"?, "tp"?, "region"?,
   "labels": [{"t0", "t1", "by", "text"}, ...],           # up to 10 of the source's labels over the moment
   image: "image": {"path": <full-size PNG>, "inline": <path of a JPEG, long side at most 1568 px>, "width", "height"}, "audio"?: {as audio's},
   audio: "audio": {"path": <5 s WAV, 16 kHz mono>, "seconds", "rms"},
   text:  "text": <at most 20,000 characters from offset>, "more", "next_offset"?,
   file:  "file": {"path": <a copy in looks/>, "media", "name", "bytes"}}
  ```

  A video frame and a picture are `image`; a video with sound adds `audio`, 5 s of its soundtrack from `at`. An
  animation's or a multi-page TIFF's `at` picks the frame. A text version, or records around `t`, are `text`.
  Any other file (a PDF, an archive) is `file`: a copy in `looks/` for the agent to open, never the store's own file.
  A range of a source is a `sheet`.
- **A sheet** is one picture of up to 9 frames of a range, so the agent sees what changed in it for the cost of one
  picture (about 1,790 Claude tokens, against that for each still).
  - **Times:** 9 spread evenly over the range, ends included (`since + k * (until - since) / 8`). Each is resolved as a
    `look` at the source and that `t` is: the segment that covers it, else the version current then or the latest picture
    at or before it, whichever is newer. A picture counts only if it is in the range (from `since`), and its tile says when
    it was taken; a version current at `since` still counts. One that is not a video or a picture (sound, text, records,
    nothing) is left out; with none left, `not_found` (`field: source`) whose `hint` names the nearest item and record.
  - **Tiles:** a time's picture is the frame at or before it, as a single look picks it, each distinct frame decoded
    from its own seek at most at the size one tile alone could take (`reader.SHEET_TILE_PX`), then cut to its cell. A
    picture that cannot be read (a HEIC, a gone or empty segment) is left out like a gap, and all of them failing is the
    error of the first. Neighbouring times (adjacent slots of the 9) whose pictures look the same (§16's test, the one the
    gate of `ask` uses, here on finer cells than the scan's, which errs toward more tiles) are one tile that spans their
    times, so a still range is one tile, and a gap or a sound leaves fewer than 9 and is never merged across. Tiles go
    row-wise in a grid as square as can be (`ceil(sqrt(n))` columns), each cell the shape of the first picture and
    each picture fitted inside its cell on dark grey, so mixed sizes and portrait video work. Each tile's number is burned
    into its bottom left corner (white on black, a seventh of the cell high, at least 16 px); times are in the JSON,
    never in the picture. The sheet is 1568 px on its long side, and its PNG is its own full size.
  - **With a `region`,** each frame is decoded as a single look decodes it and cut to that part at once, so a tile is the part, cells take
    its shape, and the look-the-same test compares the parts: motion elsewhere in the frame makes no new tile. For
    motion in one area (a door, a desk), it shows what changed there across the range.
  - **Result:**

    ```
    {"kind": "sheet", "source", "since", "until", "region"?, "labels": [up to 10 of the source's labels over the range],
     "image": {"path": <PNG>, "inline": <JPEG, long side at most 1568 px>, "width", "height"},
     "tiles": [{"n": 1, "t" | "t0" + "t1", "look": {"item", "at"}}, ...]}
    ```

    A tile's time is the requested times it stands for, not its frame's own: a still screen's frame from earlier is what
    the screen showed then. Its `look` handle is its first time's, to see that moment at full size.

**`remember`:**
- `note`: up to 100,000 characters.
- `path`: a file, a folder (every file under it; more than 1,000 is `too_big`), or a recording (§7).
- `t`: when it happened: a note's time (default now), a version's (default now), a recording's start (default: its own
  time, §7).
- `expires`: a time ahead, or a span from now (`30d`).
- `name`: a note's name, or a file's (default its file name).
- `source`: a push source (default `agent`).
- **Result:** `stored` is `[{"type": "record" | "item", "id", "kind"?, "name"?, "t0", "t1"?}]` and `unchanged` is
  `[{"name", "id"}]`. A recording's entry in either is `{"type": "item", "id", "kind", "name", "t0", "t1", "time_from"}`,
  and an `unchanged` one is the stored item's own, so its `t0` and `time_from` are what it was stored with, not this
  call's `t`. `time_from` is `given`, `metadata`, `mtime` or `now` (§7), per entry, since a folder of pictures mixes EXIF
  times and mtimes. A note's and a version's has none.

**`label`:** `until` defaults to `since`. `keep` needs `source`. The result is `{"label": <id>, "kept": <items
marked>}`.

**`keep`:** `until` defaults to now. The result is `{"items": <items changed>, "keep": <bool>}`.

**`status`:**

```
{"home", "running": bool, "pid", "started", "config_error": <str> | null,
 "floor": {"breached", "free_gb", "floor_gb"},
 "decider": {"model", "state", "error", "waiting"},
 "jobs": {"running", "queued"},
 "sources": [{"name", "kind", "state", "project", "keep": {"max_gb", "when_full", "max_days"},
              "bytes", "items", "records", "newest", "error"?, "approve"?: "mms approve <name>"}],
 "devices"?: {"displays": [...], "microphones": [...], "cameras": [...], "error"?}}
```

A source's `state` is `running`, `waiting` (between ffmpegs), `paused` (the floor), `needs_approval`, `missing` (a
watched path not there), `idle` (push and program sources) or `error`. Run in the CLI with no daemon, `status` gives
`running: false`, nulls for what only a daemon knows, and the sources as the store and config hold them.

**`source_set`:**
- `name`* and `kind`* (screen, microphone, camera, folder, log, command or program).
- `device` (screen, microphone, camera), `fps` (screen, camera).
- `path`, `glob` (folder, log). `command` (command). `time_field` (log, command).
- `max_gb`* (a number above 0), `when_full`, `max_days`.
- `remove` (true: delete the table; only `name` with it).
- There is no `project` parameter: a folder, log or command source gets the caller's project, and the other kinds are
  ambient.
- **Result:** `{source, pending_reload}` (§4.4): the source as `status` gives it, but with `state: "pending_reload"`
  until the running daemon has applied the change. A source that needs approval carries `approve` at once, and is
  `needs_approval` in `status` once applied.

**Jobs:**
- A job is `{"job": "j-<8 hex>", "script", "state": "queued" | "running" | "done" | "failed" | "cancelled",
  "started", "ended" | null, "error": {...} | null, "result": {...}}`.
- `result` is whatever the script has reported so far (§13), and it grows while the job runs.
- `wait_s` (0-50) returns early when the job ends.
- `jobs/cancel` waits up to 10 s, which is at most one window, and returns the job: `cancelled` once it stopped.
- `jobs/start` refuses an unknown script (`not_found`). The script's own checks of `params` come back as a
  `failed` job with `bad_request`, within the same call.

## 12. MCP

### 12.1 The adapter (`mms mcp`)

- **One process is one session,** serving one agent over stdio with the MCP Python SDK 2.x:
  `Server(..., instructions=..., on_list_tools=..., on_call_tool=...)`, results as
  `CallToolResult`.
- **It starts in about a second.** `mms mcp` imports the SDK, `client.py`, `project.py` and `config.py` only (the CLI
  keeps its other imports inside its commands): no torch, av, numpy, PIL, store or reader. On 31 Oct the SDK took about
  1.0 s to import, and `mms.reader` another 0.42 s. It answers `initialize` and `tools/list` before it touches git
  or the daemon, and writes only MCP messages to stdout (its log goes to stderr).
- **In a background thread** it finds the project (§12.2) and makes sure a daemon runs (`client.ensure`: connect; if
  none, run `mms start --detach`, up to 30 s). Tool calls wait for that thread. A call's whole time stays under 50 s
  (Codex stops a call at 60 s): each attempt of a call works out `ask`'s `wait_s` and its own timeout from the time
  left, so that the time spent waiting for the daemon, or in a first attempt, comes off the wait of the retry.
- **`--tools NAME,...` offers only those tools** (default: all seven), for an owner who wants a subset: for a host whose
  agent may not read the disk beyond the home, `remember`, `source_set` and `sources` are left out. `tools/list` shows
  the offered ones; a call of another is `not_found` like an unknown tool. The instructions stay as in §12.4.
- **Session:** `s-` and 12 random hex digits, made at start. `client` is the MCP client's name and version from
  `initialize`.
- **A tool call is one API call:** its arguments as they are, plus `caller`, with relative paths made absolute.
  - The result is one `TextContent` of compact JSON. `look`'s image (a frame, a picture or a sheet) comes first as an
    `ImageContent` (the file at `image.inline`, which the daemon wrote, base64). A `look` with `moments` gives, for each moment in order, its
    picture (when it has one) and then its JSON, and the call is an error only when every moment failed.
  - An API error is a result with `isError` and the error's JSON. The adapter's own errors take the same shape:
    `bad_request` and `not_found` (`ask` with `job` and other arguments, an unknown tool), `bad_project` (§12.2),
    `daemon_unavailable` (no daemon was reached) and `daemon_failed` (one was, and gave no usable answer).
  - A call that reaches no daemon (`daemon_unavailable`) runs `ensure` and asks once more, within the same 50 s. `daemon_failed` is not
    asked again, since the call may have been done.
  - Tools declare no output schema.
- **`ask`:**
  - With `job`, only `cancel`, `all` and `wait_s` may be given; it maps to `jobs/cancel` or `jobs/get`. `all` is a
    boolean, and needs `job`.
  - Without it, `question` or `questions` is needed, and it maps to `jobs/start` with `script: "ask"` and the other
    arguments as `params`.
  - The output is `{"job", "state", "error"?, ...result, "next"?}`, without the fields that are None (`error`, and
    `stopped` until the scan ends). `next` (while the job runs) says to call `ask` with this `job` to collect more, or
    with `cancel: true` to stop it.
  - **The first reply** (the call that started the job) is the whole result. **A later one** (`ask` with `job`, `cancel`
    too) holds `job`, `state` and `progress`, and each other field (`sightings`, `candidates`, `warnings`, `searched`,
    `stopped`, `error`...) only when its value differs from what was last sent for that job, and then whole, never in
    part. So a check-in does not repeat what the agent has, and a sighting that grew (newest first, its start moves
    earlier) or left the list comes back as the list is now. `all: true` with `job` gives the whole current result, as a
    first reply does, for an agent that lost track.
  - **The memory behind that** is the adapter's only one: for each job not yet ended, the field values last sent. It is
    recorded when the reply is made, on the event loop after the daemon's answer, not in the thread that waited for it, so
    a call the client cancelled records nothing. It is forgotten once a reply with a final state (`done`, `failed`,
    `cancelled`) has been sent, and a job the adapter has no memory of (it restarted, or the job had ended) gets the whole
    result. Overlapping check-ins on one job can finish out of order: one with fewer windows than the last sent, or of a
    job whose final state was sent (the last 200 ids are kept), gets `job`, `state` and `progress` only and records
    nothing. The daemon's `jobs/get` and `mms ask` always give the whole result.

### 12.2 The project

- **The folder:** `CLAUDE_PROJECT_DIR` if set, else the working directory. Roots are not asked for: the 2026-07-28
  spec deprecates them, Codex probably sends none, and Claude Code sets `CLAUDE_PROJECT_DIR`.
- **The ID**, first match wins, never the raw path:
  1. `id:<value>`: a `.mms-project` file in the folder or a parent up to the git top level, holding one ID
     (`[A-Za-z0-9_-]{4,64}`). The owner writes it to merge or split projects;
  2. `git:<the first 16 hex of the root commit>`: in a git repository with commits that is not shallow (the smallest
     root commit when there are several). It holds through renames, moves, clones and worktrees, and when a remote is
     added or renamed;
  3. `remote:<16 hex of the SHA-256 of the origin URL>`: in a shallow clone, whose root commit moves with its depth.
     The URL is normalized: lower case, no scheme, user or trailing `.git` or `/`, so `git@github.com:a/b.git` and
     `https://github.com/a/b` match;
  4. none: no project. Reads are ambient only, and writes are ambient.
- **The name** is the git top level's folder name, else the folder's. The root is kept for display only.
- `project.py` does this with at most three `git` calls and caches nothing.

### 12.3 Tools

| Tool | Op | Description (verbatim) |
| --- | --- | --- |
| `ask` | `jobs/*` | Have a local model look at or listen to the raw record in a time range, and return sightings. Returns within about 45 s with what it found so far and a job id; call again with job to collect more (it returns what changed since the last call; all: true returns everything), or cancel. |
| `find` | `find` | Instant search of the memory by time, source and words: items, records and labels, newest first. When the result's next is set, call again with after set to it for the next page. |
| `look` | `look` | Show one moment, or up to 8 at once with moments: a picture inline (and its file path), a sound's file, text, or a copy of a file that is none of these (a PDF, say). Or a source over since and until: one picture of up to 9 numbered frames across the range. With region it shows only that part of the picture, enlarged. Check a sighting with it before trusting it. |
| `remember` | `remember` | Store a note, file, folder or recording once, at its real time, with an optional expiry. |
| `label` | `label` | Mark a time range with text; with keep and a source, keep the media and files it covers from caps (records are not kept). |
| `sources` | `status` (`devices: true`) | List the sources, their state and size, the capture devices, the floor and the model. |
| `source_set` | `source_set` | Add, change or remove a continuous source: screen, microphone, camera, folder, log, command or program. A cap is required; only the fields given change, and a cap or max_days can only stay or rise (the owner lowers them in mms.toml). Screen, microphone, camera and command sources wait for the owner's approval. The reply says so, and says while the daemon has not yet applied the change. |

**The `ask` description** is the row above, then one paragraph made when the adapter lists its tools (`adapter.model_paragraph`)
from the configured model's facts. It reads `mms.toml` and the model's folder (`decider.about_of`), with no daemon, git,
loaded model or heavy import, so every host shows it, Codex's bench included. It gives the model's name; what it sees and
hears (a deaf model says `It hears nothing: sound is not read.`); its question types; the keys a question may carry when
that is fewer than all; its default yes/no threshold, marked `(assumed, not yet measured here)` unless `hit` is in
`About.measured`; how many questions it reads in one pass; and its notes file in the skill (§15). A `python:` decider gets
a generic line, and no model (or a folder no backend reads) says so. The whole description stays within 1,024 characters,
(Claude Code cuts a longer one). For the default:

> The model now is d1-3B. It sees pictures and video frames. It hears nothing: sound is not read. It answers noul, choice and
> score. A question carries only criteria besides type, instructions and hit. Its default yes/no threshold (hit) is 0.8
> (assumed, not yet measured here). It reads any number of questions in one pass. Its notes: models/d1_3b.md in the
> mms-questions skill.

A switch of model during a session is seen by the next session's agent; until then `ask`'s refusals and warnings name the
real model. `ask`'s `hit` schema says "the model's own if not given: this tool's description says it".

**Input schemas** are the op's parameters (§11; for `ask`, §13.2 plus `job`, `cancel`, `all` and `wait_s`), with types,
enums, bounds and defaults as given there, and `additionalProperties: false`. A time is a string or a number. `caller`
is not in any schema.

### 12.4 Server instructions (verbatim)

The first paragraph stands alone in 512 characters, the most Codex is sure to show. The whole is under Claude Code's
4,096.

> MultimediaSearch is the owner's long memory of this machine: screen, microphone, cameras, folders, logs, programs and agent
> notes, kept as the raw recording. Use find for instant search by time, source and words. Use ask to have a local
> model look at or listen to the raw record in a time range: put the range in since and until, never in the
> question, and ask about one thing that could be seen or heard. Check each sighting with look before trusting it.
>
> ask returns within about 45 s with what it has found so far and a job id: call ask with that job to collect more, or
> with cancel to stop it. A sighting gives a source, a time span, p and a look handle; candidates are the best windows
> that did not count, best first, to look at when the sightings do not hold what you need (look takes several at once
> in moments, and over a span, with source, since and until, shows up to 9 frames in one picture). Files the model does
> not read, such as PDFs, are named in ask's warnings, and look gives each as a file to open. Reads cover the ambient
> sources and this project unless scope is "all", and every result says what was searched. remember stores a note,
> file, folder or recording at its real time; label marks a time range. sources lists what is recorded and the capture
> devices; source_set adds or changes a source and needs a cap. Screen, microphone, camera and command sources wait
> until the owner runs `mms approve <name>` in a terminal: tell the owner, and never run it yourself. The
> mms-questions skill says how to write questions for ask.

## 13. Scripts and `ask`

### 13.1 Scripts

- A script is `scripts/<name>.py` in the home, else the shipped one. It defines:
  - `run(job) -> None`: required;
  - `cli(argv: list[str]) -> dict`: optional. It turns `mms <name> ...` or `mms run <name> ...` flags into `params`, so
    a script's flags live with the script.

  Its top-level imports stay light: it reads the store through `job`, and the CLI loads it to call `cli`.
- **The `job` it is handed** (`jobs.Job`):

  | Member | What |
  | --- | --- |
  | `id`, `params`, `caller`, `home`, `store`, `now` | the request, and where |
  | `about()` | the decider's `About` (§13.3), read without loading it and kept for the job; `decider` when no model is configured |
  | `check(questions)` | the model's own rules for these questions (§13.3), read without loading it: a `ValueError` if it refuses them |
  | `profile(overrides=None)` | the profile a scan uses: `reader.PROFILE`, the decider's `About.profile`, then `overrides` |
  | `windows(since, until, sources=None, scope="default", newest_first=True, profile=None)` | `reader.windows` with the caller's scope applied (§5.2), cut for the decider's `about`. `profile` overrides any key of `reader.PROFILE` (§16), which is the one place the defaults are. An item that cannot be read is counted in `result["warnings"]` (`item 12 unreadable: ...`, the first five by name, then `and N more items unreadable`) and the scan goes on. An item the decider does not read is named once in one rolling warning, which wakes no waiter and is in the final result, grouped by media type, or `audio segments` or `video segments` for a captured one (it has none), or `an unknown type` for a file with none, with up to five ids each. It ends by what `look` does: `look shows them` when every group is a picture, sound or video, `look gives a copy of each` when every group is a file, else `look shows a picture, sound or video and gives a copy of any other file`: `3 items were not read, since d1-Omni-2B does not read their types: application/pdf (items 14, 15), application/zip (item 16); find lists them and look gives a copy of each`. A video item whose frames are read but whose soundtrack the model cannot hear (or the reverse) is named in a second rolling warning, by source, which wakes no waiter either: `the sound of 2 video items was not heard, since d3-flash does not hear: cam (items 12, 14); find lists them` (`the frames of ... were not seen, since ... does not see` for an ears-only decider) |
  | `estimate(since, until, sources=None, scope="default", profile=None)` | how many windows of each kind `windows` would give, from the index |
  | `searched(since, until, sources, scope)` | what those windows cover, as §5.2 gives it |
  | `decide(state, questions, media) -> answers` | through the GPU queue; raises `job.Cancelled` once the job is cancelled |
  | `update(**fields)` | merges fields into the job's `result`, visible to waiters at once |
  | `warn(text)` | appends to `result["warnings"]` |
  | `label(t0, t1, text, *, source=None, item=None, meta=None)` | a label by the script's name, with the caller's project and session |
  | `cancelled` | true once cancelled |

- A `ValueError` from the script fails the job with `bad_request`; any other exception fails it with `internal`, and
  the daemon log has the traceback.
- **`mms ask ...`** and **`mms run <script> ...`** call `cli`, start the job through the API (starting the daemon if
  needed), wait for it to end, and print its `result` as JSON. The note on how the scan went goes to stderr.

### 13.2 The shipped `ask`

**Parameters:**

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `question` | string | | one thing that could be seen or heard; yes/no unless `answers` |
| `answers` | list of 2+ strings | | the options of a choice question |
| `hit` | number 0-1, string or list | the model's own (`about.hit`); the first answer | what counts: P(yes), or the options that count (the script's defaults: the schema gives none; the `ask` description says the model's) |
| `context` | string, at most 4,000 characters | | state text given with every window: what the source shows, not what is hoped for |
| `questions` | object | | instead of `question`, `answers` and `hit`: typed questions (§13.3) by name, each with `type`, `instructions` (and `criteria` for a choice or score) and an optional `hit`. To find any of several things, use `question`, `answers` (the things) and `hit` (all of them), not `questions` |
| `since`, `until` | time | 24 h ago (with a warning), now | the range |
| `sources`, `scope` | as `find` | | |
| `limit` | 1-50 | 10 | sightings to find; 1 is "last seen" |
| `candidates` | 0-20 | 5 | the best windows that did not count, to give as well: one per moment, best first; 0 for none |
| `oldest_first` | bool | false | |
| `best_first` | bool | false | read the whole range (or to `max_s`) and give the `limit` sightings with the highest `p`, best first, instead of the first `limit` found in scan order |
| `every_s` | number ≥ 0 | 1 | video: a frame per this many seconds of the clock; 0 for every frame |
| `still_s` | number ≥ 0 | 30 | video: a frame that looks the same as the one that began a window joins it, for at most this many seconds, and never more than `max_sighting_s` (§16); 0 reads every frame the step picks |
| `max_px` | integer ≥ 0 | 230,400 (640×360) | the most pixels of a frame or picture the model sees (unless its profile sets its own default), never above the model's own cap (`About.max_px`, d1: 1.6 million), which 0 gives; a model with no cap reads 0 as stored. Raise it to read small text (§16) |
| `merge_s` | number ≥ 0 | 10 | hits of a source this close are one sighting |
| `max_sighting_s` | number ≥ 0 | 600 | a sighting is cut here and marked `capped`; 0 for no cap |
| `max_s` | 1-3600 | 600 | stop scanning after this many seconds, counted from the model's first answer, so loading it is not counted |

**`hit`** per question type:

| Type | `hit` |
| --- | --- |
| noul | a P(yes) threshold (true, or `hit` left out, is the model's own `about.hit`) |
| choice | the option, or list of options, that count |
| score | the least level that counts |
| number | the least value that counts (a number with a `range`); true: counts when a value is found (no `range`) |
| box, boxes | true: counts when one is found |
| text, date | true: counts when a value is found |

A threshold may come as text that is a number (`"0.5"`), and `mms ask --hit` passes its text on for the question's type to read,
so `--hit 2` is a choice's option `2` or a score's level 2. A threshold that is not a finite number in its type's range is `bad_request`.

A window hits when every question with a `hit` hits. What a question says of a window is its answer's text: a yes/no `yes` or
`below <hit>`, a choice its pick (the picks that are in `hit` when there are any), a score `level 2`, a number
`3 (2 to 4) beeps` (its interval and unit), a text or date its value (`no value`), a box its edges, fractions of the picture from its top
left, to 2 decimals (`at top 0.15, left 0.06, bottom 0.24, right 0.27`, or `none`), and boxes the count then each box
(`3 found: top 0.10, left 0.20, bottom 0.30, right 0.40; top ...`; `at least 16 found: ...` when the answer's `at_least` is
true). A box with an edge outside 0 to 1, or past the opposite one, is `Unreadable`.

**`said`** is what a sighting or candidate says beyond its `p`:
- one question with a `hit`: that question's answer text alone, with no name and no p (`a terminal window`). A yes/no has no
  `said`: a sighting is a yes and a candidate is below `hit` by definition;
- two or more: each as `name: answer (p 0.95)`, joined by `; `. The names are the agent's own keys, and each question's p
  says which one held a window back (the item's `p` is the weakest).

The label `ask` writes keeps the full text whatever the number of questions (`name: answer (p 0.95), <time>`, then
`(capped)`), since a later `find` over labels needs the question's words.

**The answers** are parsed by one parser for each type, from the shapes of §13.3: a
ranged number is `{"number", "interval", "unit"}` with no `found`, and a text, date or number with `found` but no
value is not found. An answer a parser cannot read, or a window the questions do not fit, is not skipped in silence:
the result's `warnings` say how many windows were left out and why.

**The scan:**
- The windows come newest first (or oldest first), cut as the script's profile says (§16). A segment's windows sit at
  their own times. A run of video frames that look the same is one window (`still_s`): its picture and `at` are the
  run's first frame in scan order (the latest when the scan goes newest first), and its `t0` and `t1` the earliest and
  latest of its frames. A video or audio version (an in-place file too big to copy) is read as media too, every window at
  the version's `t0`, with `at` its offset into the file, as a text version's windows are. A picture (`image`) is a
  window at its instant, and an animation's windows share it (`at` is the frame's time).
- **Before it scans,** a question the decider cannot take is `bad_request` with 0 windows read, naming the decider and what
  it takes:
  - a key no question has (a question takes `type`, `instructions`, `hit` and `criteria`, `multiple`, `range`, `step`,
    `unit`; a key outside them was once passed on and ignored);
  - a type the decider does not answer (`about.types`), or a key it does not take (`about.question_keys`);
  - the model's own rules, which the backend declares (`job.check`, §13.3);
  - a yes/no's `criteria` that is not both `true` and `false`, each a non-empty text (d1 and d1-3B would show the missing side as `None`).
- Each window is one `decide` call carrying every question. A model that reads one question a pass (`about.questions_per_read`)
  reads the window again for each, and the estimate counts it: a window's cost is the model's, times
  `ceil(questions / questions_per_read)`.
- A choice made from `answers` gets one more option, `none` ("none of these"), unless an answer is already named
  `none`. It never counts.
- Hits of one source within `merge_s` are one sighting, labelled once (by `ask`) on its best window. A window may stand
  for a still run of frames, so a sighting spans the still frames its windows stand for, and its `look` handle is the
  window's picture.
- A sighting grows to at most `max_sighting_s`, and the next window starts another. `still_s` is cut to `max_sighting_s`, so no
  window outgrows the cap.
- The scan stops at `limit` sightings, the range's end, `max_s` (from the decider's first answer; `job_max_s` still bounds the
  whole job), or cancel.
- **`best_first`** reads on past `limit`: the sightings are made as before, and the `limit` with the highest `p` are
  given, best first, ties to the one scanned first (`oldest_first` still sets the scan order). It holds at most `limit`
  closed sightings, each with its picture: the worst goes when one more closes, and the non-hits kept near it go with
  it, so none of them becomes a candidate. Only the sightings given are labelled, and a running report shows the best so far. `stopped`
  is never `limit`. It is for a large range, or for sound, where many windows pass the default `hit` (below) and the
  first `limit` found are seldom the right ones.
- While it scans it keeps, for the candidates, the best non-hit of each source per `merge_s` of the clock (of each
  window's time when `merge_s` is 0), 200 moments at most: the worst goes first, and equal `p` keeps the one scanned
  first. A non-hit within `merge_s` of its source's open sighting is not kept: a sighting only grows, so it could
  never be a candidate. Only the fields a candidate shows are kept, never the window.

**`result`**, reported through `job.update` as the scan goes:

```
{"sightings": [{"source", "t" (or "t0" and "t1"), "p", "said"?, "kind", "capped"?, "project"?, "look", "run"?, "tp"?}],
 "candidates": [{"source", "t" (or "t0" and "t1"), "p", "said"?, "kind", "look"}],  # the best non-hits, best first, one per moment
 "progress":  {"windows", "of_about", "seconds", "decider"},
 "searched":  {...},                                               # §5.2
 "warnings":  ["..."],
 "stopped":   null | "limit" | "range_end" | "max_s" | "cancelled"}
```

- `sightings` come in scan order (newest first unless `oldest_first`), or, with `best_first`, highest `p` first.

- `look` is the handle for the `look` tool: `{"item", "at"}`, or `{"source", "t"}` for records.
- `t` is a moment (the start and end are the same instant, to the millisecond the times are written to: a frame, a picture, a
  version); `t0` and `t1` are a span (a sound, a run of records, a run of still frames, a sighting that grew). The fields that say nothing are left out:
  `said` as above, `capped` unless it is true (the sighting was cut at `max_sighting_s`), `project` unless the row has one.
- A candidate's `p` is its weakest question's progress toward its hit.
- Two windows of one source within `merge_s` are one moment (the better is kept), and a window within `merge_s` of a
  sighting belongs to it. Candidates come from the windows scanned, so a scan stopped early has seen only part of the
  range. They are not labelled.
- `hit` decides what is labelled, so it stays sure: on real frames d1-Omni-2B's P(yes) does not separate the right ones
  from the wrong ones (a pallet that was there scored 0.37-0.51, a look-alike 0.57-0.64), so no threshold returns
  the right frames alone. The candidates let the agent, a stronger model, judge the uncertain windows itself. The default
  is the model's `About.hit`, and only d1-Omni-2B's 0.8 is measured: every other model's is 0.8 until the bake-off (§16),
  and 0.8 does not carry across models (d1-2B's calibration temperature is 1.34 against d1-Omni-2B's 3.38, so the same
  logit gap reads higher on it). On real sound d1-Omni-2B's 0.8 passes far more clips than events: in
  MultiMediaSearchBench (6 November 2026, 6,123 FSD50K clips) asks for coughing gave 121-164 sightings across each
  agent's asks for 5 coughs, about 2% of the clips, and the coughs found ranked 1st, 9th and 96th of 121 by `p`, so
  ranking by `p` (`best_first`) finds what scan order misses.

**Lint** (warnings at the start; the rules are the script's, to change):
- a time word in the question or context ("yesterday", "this morning", "last week", "at 3pm", a date or weekday):
  put the time in since and until;
- two things joined by "and" or "or": ask about one thing;
- no `since`: the last 24 h were searched;
- an estimate (§16) over `max_s`: give it in minutes, and suggest narrowing the range or the sources, or raising
  `every_s`, and say "(its costs are assumed)" when `picture_s` is not in `About.measured`, and, with `best_first`, that
  it ranks only what is scanned before `max_s`;
- more than one question for a model that reads one a pass, or more than `about.questions_per_read` for one that reads
  several (d1: 16): the decider reads the window again for each;
- a `text` or `date` question, or a `number` with no `range`, over a kind the decider does not read those from
  (`about.extracts_from`; d1: text and records only);
- speech words ("said", "talk") over audio, when `about.speech` is false (d1 hears sounds better than speech).

### 13.3 The decider protocol

mms holds no fact about any one model. A decider declares what it takes in its `About`, and the reader, `ask` and the
queue read it. `decider.py` holds the protocol and the loader; `gpu.py` the daemon's queue in front of it; `modelenv.py` a
model's env; `transport.py` how a decider reaches its model; `native.py` the translation four families share; `fetch.py` the
download of a model's folder. Each family has a backend, `decider_<family>.py` (its facts, how it claims a folder, its
translation) and, when mms runs the model, a worker file `worker/<family>.py` (the half that holds it). Eight models in five
families swap in through `[decider] model`, one loaded at a time.

**A decider** is any object with:

| Member | What |
| --- | --- |
| `about` | an `About` (below), declared statically: reading it never loads the model |
| `decide(state, questions, media) -> {name: answer}` | the answers for one window. `state` is text or None, `questions` is `{name: question}`, `media` is a list holding the window's medium, or empty. A `ValueError` or `TypeError` says the questions do not fit this window: the scan counts it and goes on. A model that cannot answer at all raises `ApiError("decider")`, which fails the job |
| `load()`, `unload()` | optional: the queue calls them at the first request and after `idle_s`. For a backend, `load` starts the worker (or waits for the owner's server) and `unload` ends it |

**`About`** is a frozen dataclass, and is checked when it is made:
- a value that is no fact (an empty name, a kind that does not exist, a negative cost or `repo_gb`, a `hit` outside 0 to 1,
  a question key or a `measured` entry that does not exist, a `repo` that is not owner/name, picture points whose pixels
  do not increase) is a `ValueError`;
- so is one that would quietly switch a lint off (a cost that is not a finite number, a `speech` that is not true or
  false, no `extracts_from` while `types` has text, date or number);
- `profile` is kept read-only, since one `About` is shared by every job;
- a `profile` key that is not in `reader.PROFILE` is not checked here but by `reader.profile_of`, when a scan reads the
  profile: it is a `ValueError` that names the decider when the bad key is the decider's.

| Field | Meaning | d1-Omni-2B |
| --- | --- | --- |
| `name` | for messages | `d1-Omni-2B` |
| `reads` | the media window kinds it takes: `video`, `image`, `audio` (text and records are always read) | all three |
| `files` | media types of versions it takes whole, as a `file` window; an entry ending `/` or `.` is a prefix | none |
| `types` | the question types it answers | all eight |
| `question_keys` | the keys a question may carry besides `type`, `instructions` and `hit`: some of `criteria`, `multiple`, `range`, `step`, `unit` | all five |
| `extracts_from` | the window kinds it reads a text, date or unranged number from | text, records |
| `speech` | whether it understands what is said | no |
| `audio_rate` | Hz of the sound it gets | 16,000 |
| `max_px` | its own picture cap, 0 for none; `reader.profile_of` cuts a scan's `max_px` to it, and 0 is the cap | 1,638,400 |
| `questions_per_read` | questions it reads in one pass, 0 for any number | 16 |
| `picture_s`, `sound_s`, `char_s`, `file_s` | its costs, for the estimate: `picture_s` is (pixels, seconds) points, linear between and flat outside; the others are seconds per sound window, per character of state and per file | §16 |
| `hit` | its default yes/no threshold, for a question whose `hit` is true or left out | 0.8 |
| `notes` | the file in `skills/mms-questions/models/` that holds what an agent needs of the model (§15) | `d1.md` |
| `repo` | the Hugging Face repo (owner/name) that `mms model use` downloads the model's folder from; empty for none | `LiquidAI/d1-Omni-2B` |
| `repo_gb` | the repo's size in GB, for `mms model list` to show before a download | 4.4 |
| `measured` | which of `picture_s`, `sound_s`, `char_s`, `file_s` and `hit` were measured on this laptop; the rest are assumed | `picture_s`, `sound_s`, `char_s`, `hit` |
| `profile` | overrides of `reader.PROFILE` | none |

**The five families.** A model is named by its folder, and each family claims a folder by its files (`claims(folder)`, which
reads them on the CPU and loads nothing).

| Model (`About.name`) | Claimed by | Runs | Reads | Types | Keys | Per pass |
| --- | --- | --- | --- | --- | --- | --- |
| d1-Omni-2B | `config.json` with the five `d1_*` keys and `model_type` `lfm3_omni` | worker, mms's own env | pictures, video, sound | all 8 | all 5 | 16 |
| d1-2B | the same keys and `model_type` `lfm3_vl` | worker, mms's own env | pictures, video | all 8 | all 5 | 16 |
| **d1-3B** (the default) | `config.json` with `model_type` `lfm2_vl`, `auto_map` `AutoModel` `modeling_d1.D1Model` and a `text_config` of hidden size 2048 and 30 layers | worker, mms's own env | pictures, video | noul, choice, score | criteria | any number |
| d3-edge, d3-mini, d3-flash | `decision_config.json` with `format_id` `d3-code-readout-v1`, and a `MODEL_MANIFEST.json` whose `model_name` is one of the three | worker, `envs/d3` | pictures, video | noul, choice, score | criteria | 1 |
| pplx-decider-flash-v1.2 (Perplexity Decider Flash) | `joint_config.json` of version 3 with template `joint-question-inline-readout-v1`, and `source/src/pplx_decider/` | worker, `envs/pplx` | pictures, video | noul, choice, score | criteria | 1 |
| h2o-lightning-4b | `serve_config.json` for `h2oai/h2o-lightning-4b`, and `h2o_lightning_shim.py` | the owner's vLLM server in WSL, at a loopback URL | pictures, video | noul, choice, score, box | criteria | 1 |

- **What is measured** (§16): d1-Omni-2B's `hit`, `picture_s`, `sound_s` and `char_s`, and the `picture_s` and `char_s` of
  every other model. Every other model's `hit` is 0.8, assumed, until the rest of the bake-off. The `ask` description and the estimate say when a figure is assumed.
- **d1-Omni-2B and d1-2B have the same `d1_*` keys and values.** Only `model_type` tells them apart, so a d1-2B folder is
  never claimed as d1-Omni-2B and handed sound. Each refuses weights whose `d1_vision_max_pixels` or `d1_set_size` differs
  from its `About`, naming both numbers. d1-3B has none of the keys, and is not claimed by d1.
- **d1-3B** (`decider_d1_3b.py`, `worker/d1_3b.py`) is a family of its own that shares d1's env. It is claimed by a `config.json`
  with `model_type` `lfm2_vl`, `auto_map` `AutoModel` `modeling_d1.D1Model` and a `text_config` whose (`hidden_size`,
  `num_hidden_layers`) is (2048, 30), so another size of d1 on LFM2.5-VL is not handed its facts. Its code takes `images=` where
  d1-Omni-2B's takes `media=`, applies no calibration, and reads the state once with each question a branch of that pass
  (`questions_per_read` 0). A score has 2 to 10 levels (its README). Its worker refuses a model whose `system_one` takes no
  `images`, and reports the picture cap of the loaded code (`runner.VISION_MAX_PIXELS`, 1,048,576 pixels), which the
  backend's `load` refuses when it is not `About.max_px`, naming both numbers. On Windows the worker sets `_fused` to false in
  the model's `hybrid` module, since torch's Windows wheels have no flash attention, and fails if it finds none to set.
- **Only d1-Omni-2B hears.** The others do not read sound items, and name a video's soundtrack in a warning (below). Only
  d1-Omni-2B and d1-2B take `multiple`, `range`, `step` and `unit`, or answer text, date, number and boxes; only H2O answers
  box besides them.
- **d3 and Perplexity run without flash-linear-attention** (it needs triton, which Windows lacks), so their linear-attention
  layers run in PyTorch, at a speed not measured. Perplexity's fast path is off, which is the card's exact computation. d3's
  own server (`d3_server.py`) is not used.
- **`python:<file.py>`** is a decider in a file that defines `ABOUT` and `decide`, and may define `load`, `unload` and
  `check(questions)`. It runs in the daemon's process (the tests' path and the owner's own code). The loader sets `home` in
  it. Its top level stays cheap, since reading `about` runs it.

**What a decider is handed of an item** (`reader._read_as`):
- a version or a picture whose media type `about` takes: one `file` window of its bytes, at its `t0`, at most 64 MB;
- else text: always;
- else by what the item is: a picture if `reads` has `image`; a sound if `reads` has `audio`; a video's frames if
  `reads` has `video`, and its soundtrack if `reads` has `audio` and the item has `meta.sound` (a captured segment
  records it, §6). A video with neither gives nothing;
- else nothing: the item is not read. The estimate leaves it out and the job's warning names it (§13.1), all of them
  before the first window;
- a video whose frames are read and whose soundtrack is not (or the reverse) is read in part: `reader.windows` reports it to
  `on_partial(item, "sound" | "frames")`, and the job keeps one rolling warning for it (§13.1).

**What it receives** in `media`: a PIL image; `{"type": "audio", "audio": float32 mono, "sampling_rate":
about.audio_rate}`; or `{"type": "file", "media": the media type, "name", "data": the bytes}`.

**Questions and answers** are one typed format, whatever the model: 8 types (`decider.TYPES`). A backend for another
model translates to and from it, so the MCP schema, the skill and the parsers stay as they are. In this repo,
`ask.PARSERS` (with `ask.read`) is the reference for the answers, and `tests/fake_decider.py`'s `_answer` for answers
that parse:
- **A question** is `{name: question}`, and a question holds `type`, `instructions` (the text), and for a choice
  `criteria` (`{option: what it means}`) and `multiple`, for a score `criteria` (the levels, lowest first), for a number
  `range` (`[low, high]`) and `unit`. `ask` removes `hit` before the decider sees the questions: it is the script's rule.
- **An answer** is `{name: answer}`, each an object whose `type` is its question's. What the parsers read of it (the
  rest is ignored):

  | Type | The parsers read |
  | --- | --- |
  | `noul` | `noul`: P(yes), 0 to 1 |
  | `choice` | `choice` (or, with `multiple`, the list `choices`), `probabilities` `{option: p}`, and `confidence`, the picked option's p when `probabilities` lacks it |
  | `score` | `score`: a level from 0 |
  | `text`, `date` | the value under its own name, or None, and `found` (0 to 1) |
  | `number` | `number`, or None, and `found`, `unit`; for a question with a `range`, `interval` `[low, high]` in place of `found` |
  | `box` | `box` `{top, bottom, left, right}` (fractions of the picture from its top left, 0 to 1, top not below bottom nor left right of right; `said` gives them) or None, and `found` |
  | `boxes` | `boxes` (each with `box` and `found`), `count`, and `at_least` (true when `count` is a floor; default false) |

  An answer that is not in its type's shape is `Unreadable`: a counted warning, never a window skipped without a word.

**Translation.** d1-Omni-2B and d1-2B take and give mms's format as it is. d1-3B, d3, Perplexity and H2O share one answer
format and `native.py`:

| | Questions (`native.to_model`) | Answers (`native.from_model`) |
| --- | --- | --- |
| all | `type`, `instructions` and `criteria`; another key, or a type outside `about.types`, is a `ValueError` (ask refused it first, from the `About`); so is a noul `criteria` beyond true and false, a choice with no `criteria` or over 255 options, and a score with under 2 levels, which the models would pass over | an answer with an `error` (d3's per-question entry), or one missing, is a `ValueError`: the scan counts the window and goes on |
| noul | as is | as is |
| choice | as is | `confidence` dropped: theirs is not the picked option's p, and `probabilities` has it |
| score | as is | `score` := the **median** level of `probabilities` (the first level whose cumulative p reaches 0.5, d1's own readout); theirs is the expected level, and the mean is never used in silence. `probabilities` without the keys "0".."n-1" gives `score: None`, which `ask.read` rejects as `Unreadable` |
| box (H2O) | `instructions` alone, which name the thing ("the red mug") | `box` only when found; `found` := its `present` (P the thing is in view; else its `confidence`, or 0); `confidence` kept |

- **Pictures and state.** d1-3B and d3 get PIL pictures and `state` None as None. Perplexity gets PIL pictures and `state` None as `""`,
  and one request for each question (the card's bit-stable mode). H2O gets lossless PNG data URIs (at most 4) and `state`
  None as `""`.
- **The model's own rules.** `About` cannot state everything a model refuses, so a backend has a static
  `check(about, questions)` that raises `ValueError` for a question it will not take (the shared checks above, the 10 levels of a score d1-3B and d3 take, H2O's box). It
  is reached without constructing or loading the decider: `decider.check_of(spec, questions)`, `Gpu.check` and `job.check`.
  `ask` calls it before the scan, as it refuses a key or type from the `About`.

**Which decider runs.** `[decider] model` is a folder or `python:<file.py>`:
- a folder goes to the first of `decider.BACKENDS` (`"module:Class"`) whose `claims(folder)` is an `About`, and the class is
  made from the folder, `url` and `envs`. A path that is no folder (missing, or on a drive not mounted) is `there is no model
  folder at <folder>`, before any backend is tried. If none claims a folder, the error is `no decider reads the model in
  <folder>: ...`, with the `reads_folder` phrase of each backend;
- `decider.claim(folder)` is the one match of a folder to a backend, `(backend, About)` or None, and `decider.claimed(folder)`
  the same or those two errors. The loader and `mms model` both use them, so they never disagree;
- `decider.about_of(spec)` gives a folder's `About` from `claims` alone, constructing nothing (the adapter's paragraph uses
  it); `known_abouts()` gives every `About` of every backend (the skill's tests run each), and `known(name)` the backend and `About` of
  a model mms knows by name, in any case (`mms model` uses it).

**A backend** (the class `BACKENDS` names) has:

| Member | What |
| --- | --- |
| `claims(folder) -> About \| None`, `check(about, questions)` | static: they read the folder and the questions and construct nothing |
| `abouts`, `reads_folder` | every `About` it can claim, and one phrase for the error when none claims a folder |
| `requirements`, `torch_backend` | the pins of its env and uv's `--torch-backend` (none for H2O, whose env is the owner's). d1's and d1-3B's are the `d1` group of `pyproject.toml`, which mms's own env holds |
| `served`, `in_mms_env`, `default_url`, `start`, `install` | whether the owner runs the model; whether it runs in mms's own env (d1, d1-3B); where it listens; the command that starts it (for H2O a template with `{folder}`); and what installs it: H2O's commands in WSL, `uv sync --group d1` for d1 and d1-3B, run in mms's project, nothing for d3 and Perplexity (`mms model use` and `prepare` build their envs) |
| `__init__(folder, *, url=None, envs=None)` | cheap: it claims the folder, checks the weights' numbers against the `About` and (d3, Perplexity) the env's marker, makes a transport and starts nothing. A backend that is not served refuses a `url` |
| `load()`, `decide()`, `unload()` | through the transport, with the backend's translation |

**The transports** (`transport.py`) raise `ValueError` for input the model does not fit and `ApiError("decider")` for a model
that cannot answer.

- **`Worker`** is a process mms starts, in the env the backend names (`modelenv.mms_python()` for d1 and d1-3B: `sys.executable`, with
  `pythonw.exe` swapped for `python.exe`, since the detached daemon runs under pythonw).
  - It runs `python -I -X utf8 worker/serve.py --family-file worker/<family>.py` as a child of the daemon (`children.py`, §6),
    with stdin and stdout as pipes. Its environment is an allowlist (the system's variables, CUDA, locale and the caches; no
    tokens, keys or proxies) plus switches that keep it off the network (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
    `DO_NOT_TRACK` and the like).
  - **The wire** is mms's own frame format, not HTTP: a pipe is private to the parent, with no port or token, and the OS closes
    it when either side dies.

    ```
    frame  = b"MSW1" | header length (u32, little-endian) | header (UTF-8 JSON) | blob 0 | blob 1 | ...
    header = {"op": ..., ..., "blobs": [len0, len1, ...]}      # "blobs" is left out when there are none
    ```

    The header is at most 16 MiB and a frame's blobs 256 MiB (a file window is at most 64 MB). Each request has one reply,
    in order:

    | op | request | reply |
    | --- | --- | --- |
    | `load` | `{"folder", "options"}` | `{"ok": true, "facts", "seconds"}` |
    | `decide` | `{"state", "questions", "media": [descriptor]}` and the blobs | `{"ok": true, "answers", "seconds"}` |
    | `stats` | | `{"ok": true, "max_memory_allocated", "memory_allocated"}`: torch's, in bytes, or null |
    | `quit` | | none: the worker exits 0 |

    A medium is a descriptor and a blob, bit for bit: a picture is its raw RGB bytes (`{"kind": "image", "mode": "RGB", "size":
    [w, h]}`), sound is little-endian float32 (`{"kind": "audio", "rate", "samples"}`), a file is its bytes (`{"kind": "file",
    "media", "name"}`). There is no JPEG and no base64. An error reply is `{"ok": false, "error": "unfit" | "failed", "type",
    "message", "trace"}`: `unfit` (the worker saw a `ValueError` or `TypeError`) is a `ValueError` here, and `failed` an
    `ApiError("decider")` naming the type. The worker serves on after either.
  - **`worker/serve.py`** keeps private copies of the frame streams and points fd 1 at stderr, so whatever a library prints,
    from Python or C, cannot break a frame; stderr goes on to the daemon's log, and its last lines go into an error. It loads
    the family file by path. Nothing in `worker/` imports `mms`, and its top levels use the standard library only (a test
    imports each under `python -I -S`), so any env's Python runs it. The family file defines `load(folder, options) -> (model,
    facts)`, which also warms the model up, and `decide(model, state, questions, media)`, which gives the model's own answers.
  - **Failure.** At a deadline (`LOAD_S` 900, `DECIDE_S` 120) the worker is killed, and the call is an `ApiError` ("gave no
    answer in N s, so it was stopped"). An exit, or bytes that are not a frame, is an `ApiError` with the exit code and the
    stderr tail, or the first bytes. The worker is then gone, and the queue drops the decider (§10), so the next request makes
    and loads a new one. `ask` does not catch `ApiError`, so one failure fails its job rather than looping over windows.
  - **`unload()`** sends `quit` and kills the worker after 10 s. The OS then frees all its GPU memory, which an in-process
    unload does not (the CUDA context and the allocator's pools stay). The daemon never imports torch, and a CUDA fault takes
    down the worker, not capture and ingest. TESTBED prints what a 640×360 frame and a 1.6 MP one cost to cross the pipe
    against d1's 53 and 489 ms; shared memory is the fallback if that stops being small.
- **`Served`** is a model the owner runs (H2O), reached over HTTP at `[decider] url` (default `http://127.0.0.1:8741`).
  - It connects straight to the loopback host `config.loopback` names, with `http.client`: no proxy variable is honoured, and
    no redirect is followed (a 3xx is an `ApiError` naming its `Location`).
  - `load()` polls `GET /health` every second, up to `LOAD_S`: 200 gives its JSON, and a `model` other than the expected one
    is refused; 503 (warming) waits; a refused connection for 3 s is an `ApiError` with the `hint` the backend gave the
    transport (H2O: `start H2O's serve.sh in WSL (DESIGN.md §13.3)`).
  - `probe(timeout_s)` gives the status of `GET /health`, or None when nothing answers (`mms model list` uses it).
  - `post()` maps 400 and 422 to `ValueError`, and every other failure, a timeout and a reply that is not the documented JSON
    to `ApiError("decider")` at once.
  - `unload()` does nothing: the server and its GPU memory are the owner's, who stops it after use.

**Envs and `mms model`.** d3 and Perplexity run in envs that only `mms model use` and `prepare` build, never on first use. d1
and d1-3B run in mms's own env and H2O in the owner's. `modelenv.py` holds all of it: where an env is, its marker (written and
cleared), its Python, the one comparison of installed versions with pins (`compare`), and whether a backend's env is ready
(`problem`); the backends and `mms model` ask it and keep no copy of the rule.
- An env is `<envs>/<family>` (`d3`, `pplx`; the three d3 models share one): `[decider] envs`, else `envs/` beside the model
  folder (`C:/models/envs`). Building one runs uv with `UV_CACHE_DIR=<envs>/.uv-cache`, so the downloads stay on the models'
  drive, in five steps: `uv venv --python 3.12`; `uv pip install --torch-backend <backend> <requirements>`; the env's own
  Python running `worker/serve.py --check-env` (installed versions from `importlib.metadata`, no torch import); a check of
  those against the `==` pins; and only then `mms-env.json` (`{"family", "requirements", "versions", "prepared"}`). The pins
  are constants of each backend (d3: torch 2.11, transformers 5.17, on `cu128`; Perplexity: the card's six, torch 2.14 on
  `cu130`, which needs driver R580 or newer).
- d3's and Perplexity's `__init__` read the marker (`modelenv.ready_python`). A missing one, or one made for other
  requirements (mms was updated), is a `ValueError` that names `mms model prepare <name>`, and reaches the agent as a
  `decider` error before any scan.
- d1's env is mms's own: `in_mms_env`, and `requirements` are the pins of `pyproject.toml`'s `d1` group, which d1-3B shares. Its
  readiness is read from the installed metadata (torch is not imported), each pin met or `uv sync --group d1` named; its
  `__init__` does not check it.
- **Where a model's folder is.** DIR is `--dir`, else the folder holding the configured model, else `<home>/models` (the home
  is `MMS_HOME`, else `~/.mms`); it need not exist. A NAME that mms knows (`decider.known`, in any case) is
  `DIR/<its name>` in that name's own case (`use d1-3b` writes `DIR/d1-3B`), there or not; any other NAME must be a folder of
  DIR, and a FOLDER is a path, used as it is.
- **`fetch.py` downloads a model's folder** from `About.repo`. It is the one module that imports huggingface_hub, and only
  when a download runs. It turns off telemetry, the implicit token and Xet before the first import (Xet would cache under
  `~/.cache`; with it off everything lands in the model folder), passes `token=False`, and refuses a huggingface_hub that
  was imported before it set them. The Hub's advice to set `HF_TOKEN` is not shown, since mms sends none. mms pins no revision:
  it takes the commit `main` is at when it asks the size, and downloads that commit by its hash, so the size it checked is
  what it downloads, and huggingface_hub writes no ref under `~/.cache` (it does for a branch, even into a folder). The rules:
  - A folder that is that model and whole (`fetch.ready`: a backend claims it as that model, and `unfinished` finds every
    file that the Hub's record of the repo lists there at its size, no shard missing from the index, and some `*.safetensors`)
    is never downloaded: no network call. The record is `.cache/huggingface/trees/<commit>.json`, which huggingface_hub writes
    into the folder as a download begins (the newest one counts; one it cannot read is left out, as the hub leaves it).
  - A download goes into a folder that is missing, or empty, or holds a download that stopped (`.cache/huggingface/`) and no
    model of any kind, or that is this model unfinished. A folder a backend claims must be this model: another model's, with or
    without `.cache/huggingface/`, is refused, as is anything else that is somebody's: move it away, or give another `--dir`.
    A model with no `repo` (a folder placed by hand) is not downloaded.
  - It downloads what is still missing, and a file cut off part-way starts again: huggingface_hub writes each file under a
    name of its own (`*.incomplete`, for one process) and moves it into place whole, so it cannot continue one, and a killed
    download leaves its file behind. mms deletes those (`.cache/huggingface/download/**/*.incomplete`) before it starts, and
    does not count them as bytes already there.
  - It checks the drive is there (a path on a drive letter that is not mounted is a `FetchError` before any call to the Hub),
    asks the Hub the repo's size and licence, and refuses when the drive has less free than the size, less what is there,
    plus 1 GB; and, when the models are on the drive of the home, when it would leave less than `[memory] free_gb`, which
    pauses capture (§9). It then prints `downloading <repo> at <commit> (<size>, licence <name>) into <folder>; if it stops, the same
    command downloads what is still missing, and a file cut off part-way starts again`: the licence is named, and nobody is
    asked. When it ends it checks again that a backend claims the folder as that model and that it is whole. Any failure is a
    `FetchError` (the Hub's, with the repo named), and `mms model` exits 1.
- **`mms model [list] [--dir DIR]`:** each folder of DIR, `*` for the configured one: the model a backend claims, what it
  reads, and its env (ready, `mms's own env`, or why not), or for H2O its URL and whether it answers. A folder whose weights are
  not all there says why and that `mms model use <name>` finishes it; one no backend claims says so. Then each model mms knows
  that no folder is, as `not downloaded: 6.3 GB from huggingface.co/<repo>`. It needs no network and makes nothing. A `--dir`
  or the configured model's folder that is not a folder is an error; a missing `<home>/models` just lists no folders.
- **`mms model use NAME|FOLDER [--url URL] [--dir DIR]`:** makes the model the one mms uses, whatever is missing, in this order. It
  checks the claim and the `--url` (refused for a model that is not served, or not loopback). A d1 or d1-3B whose own env
  lacks a pin stops here, naming `uv sync --group d1`, before any download. It downloads the model if it is not there, builds
  the env of d3 or Perplexity if it is not prepared (as `prepare` does), and for H2O prints its install and start commands when
  `/health` does not answer (not while it is warming). It writes `[decider] model` (and `url`: H2O's, else removed) with
  `config.set_decider`, keeping the file's comments, and the daemon follows (§4.2). It prints `model: <name> (<folder>)`; for a
  model that hears nothing, `<name> hears nothing: mms model use d1-Omni-2B gives one that hears`; and it warns on stderr when the
  config has a microphone or camera such a model cannot hear. A download or build that fails changes nothing in the config.
- **`mms model prepare NAME|FOLDER [--dir DIR]`:** the same without switching: it downloads the model if it is not there and
  builds the env as above. For d1 and d1-3B it installs nothing and checks mms's own env against `D1.requirements` (`uv sync
  --group d1` when a pin fails) before any download. For H2O it installs nothing and prints the commands that install and
  start the server.
- **H2O's `start` is a template:** `{folder}` is the model folder as WSL sees it. `wsl()` turns `C:\models\x` into
  `/mnt/c/models/x`, quotes it for the shell, and refuses a folder that is not on a lettered drive (a UNC path), which WSL
  cannot see under `/mnt`.

**H2O's server** runs in WSL Ubuntu from the owner's venv (`~/h2o-venv`: vllm==0.30.0, torchcodec==0.17.0,
transformers==5.20.2), reading the weights from the model folder as WSL sees it (`/mnt/c/models/h2o-lightning-4b`).
`decider_h2o.H2o.install` also installs `python3-dev`: Triton compiles a helper against `Python.h` as vLLM starts, and without it the server exits at once (seen on
1 Nov). `decider_h2o.H2o.start` is the command:
one-way reading on stock vLLM (`TWO_WAY=0`: no patched files, which scores lower), 60 % of the GPU (`GPU_UTIL=0.6`) and a
16,384-token context (`MAX_LEN`), so the desktop keeps room; at most 16 reads at once (`VLLM_ARGS=--max-num-seqs=16`:
mms sends one window at a time, and vLLM's default of 256 needs more of the model's Mamba cache than 60 % holds, so the server
refused to start, 2 Nov); no usage statistics, no hub and no image links
(`VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK`, `HF_HUB_OFFLINE`, `LIGHTNING_IMAGE_URLS=off`). Its shim cuts a long record by BM25
rather than refusing it; mms's windows (at most 6,000 characters) are far below its context. It is ready about 4 minutes
after it starts and holds its GPU memory until the owner stops it. Both of its ports listen on 127.0.0.1 in WSL, and WSL's
default NAT networking forwards Windows' `127.0.0.1` to them (localhost forwarding), so mms reaches `127.0.0.1:8741` (seen
2 Nov) and nothing off the machine does.

**To add a model:** write a backend module whose class has the members above, and add it to `BACKENDS`; for a model mms runs, a
worker file with `load` and `decide`; and a notes file in `skills/mms-questions/models/` that its `About.notes` names
(§15); give its `About` a `repo` and `repo_gb` if `mms model use` should download it. Or write a `python:` file. Nothing else
in mms changes. A model with an env of its own makes its `Worker` from
`modelenv.ready_python(self, folder, envs, name)`, which refuses an env that is not prepared; a served one gives `Served` the
`hint` to show when nothing answers. Keep it cheap to make:
- `__init__(folder)` is cheap: `Gpu.about()` makes the class just to read `about` (a job needs the facts before the first
  window), and the queue makes it again to load. Anything heavy goes in `load()`, which the queue calls once, at the first
  request.
- `claims(folder)` is called on the class, not on an instance, and reads the folder without loading anything.
- `load_decider` imports every module of `BACKENDS` for any folder, so a backend's top level imports nothing heavy or optional
  (the heavy half is a worker file, run in the model's own env); a missing optional package must not break the folders of
  other models.

**One model at a time, and a second for sound later.** The default is deaf: a microphone is unread until
`mms model use d1-Omni-2B`. VRAM is 24 GB, so there is no reason yet to load two. When a deaf model should be joined by one
that hears, the path needs no rewrite: `[decider] sound = "<folder>"`; the queue
holding a worker for each role (they are processes, so either can be killed alone); `job.about()` composing one `About` (the
union of what they read, the sound model's rate and sound cost, the types both answer); and `job.decide` choosing the decider
by the window's medium. The reader, `ask` and the wire stay as they are. Not built: this; a `track` question type (no model
here answers one: d3 refuses all but choice, noul and score); batching; and d3 over its server.

**What stays the script's** (`ask.py`): the hit rules, the `none` option, the lints, `DECODE_S` (mms's own decode cost),
and the limits and defaults of its parameters. It reads the model's facts from `job.about()`.

## 14. Install

`mms install [--only claude|codex|skills|login|model|daemon ...] [--remove] [--claude-exe PATH] [--model NAME|FOLDER]
[--models-dir DIR]`. It changes only what is MultimediaSearch's, and fails, saying why, on anything else of the same name:
- **`<mms>`** is the absolute path of the `mms` script beside `sys.executable` (the project's venv), so `mms` need not be
  on PATH. When `MMS_HOME` is set, each registration passes it on.
- **Claude Code:** `claude mcp get mms`: a non-zero exit means none (its wording is not read; if `claude` is broken,
  `add` fails and says so). An entry whose command is `<mms>` with the args `mcp` is MultimediaSearch's and is removed
  (`claude mcp remove --scope user mms`); any other `mms` entry is refused. Then
  `claude mcp add --scope user mms [-e MMS_HOME=<home>] -- <mms> mcp`.
  `claude` is the one on PATH unless `--claude-exe` names another.
- **Codex:** in `$CODEX_HOME/config.toml` (default `~/.codex`), `[mcp_servers.mms]` gets `command = "<mms>"`,
  `args = ["mcp"]`, when set `env.MMS_HOME`, and `default_tools_approval_mode = "approve"` unless the owner set a
  mode. Without it Codex asks before every `mms` call, and `codex exec` refuses them all (seen on 31 Oct with Codex
  0.159.3); no `mms` tool can approve a device, lower a cap or delete history (§4.3, §4.4), so approving them is safe.
  The owner's other keys stay, and `--remove` takes out only these. It is written with tomlkit, and the rest of the
  file is kept. Codex's defaults (10 s to start, 60 s a call) fit: the adapter answers at once, and `ask` returns
  within 50 s.
- **Skills:** `src/mms/skills/mms-questions/` (with its `models/` notes) is copied to `~/.claude/skills/` and `$CODEX_HOME/skills/`,
  replacing an older copy of it. A folder of that name whose `SKILL.md` is not named `mms-questions` is refused.
- **Login:**
  - Windows: `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\mms.cmd`, running
    `start "" /min "<mms>" start --detach`. Task Scheduler's at-logon trigger needs an administrator; the Startup folder
    does not.
  - Linux: `$XDG_CONFIG_HOME/autostart/mms.desktop` (default `~/.config`) with `Exec=<mms> start --detach`.
- **Model:** the step is `mms model use` (`prepare.switch`, §13.3), and runs before the daemon, which then starts with a
  model. With `--model` it downloads and configures that model (a name mms knows, or a folder). With none, a model that is
  already configured stays (`unchanged`, saying `mms model use NAME` switches), so running `mms install` again never changes the
  owner's choice; with none configured it downloads and uses d1-3B (`DEFAULT_MODEL`), as mms ships no weights. `--models-dir`
  says where the folder goes (default: the configured model's folder, else `<home>/models`); with a model configured and no
  `--model`, nothing is downloaded, so it fails the step rather than being ignored. The licence is named before the
  download (§13.3), with no prompt. d1-3B hears nothing, so the line says so and names `mms model use d1-Omni-2B`. The work
  prints to stderr, so stdout keeps one line per step. A failure (mms's own env not ready, no room, a folder that is someone
  else's, a download that stopped) is this step's `failed (<why>)`, and the other steps go on. `--model` or `--models-dir`
  with `--remove`, or with an `--only` that leaves out `model`, is a usage error (exit 2). `--remove` keeps the weights and
  `[decider] model`, and says so: the owner deletes the folder by hand.
- **Daemon:** it is started once now, as `mms start --detach` starts it (§10); `--remove` leaves it running.
- **Every path** comes from the environment when it runs (`USERPROFILE` or `HOME`, `CODEX_HOME`, `APPDATA`,
  `XDG_CONFIG_HOME`), so tests point them all into `tmp_path` and pass a fake `--claude-exe`.
- **It prints** one line per step: done, unchanged, skipped (with why: "claude not on PATH") or failed. It exits 1 if
  any failed. `--remove` undoes each step.

## 15. The skill (`mms-questions`)

`src/mms/skills/mms-questions/SKILL.md` teaches the agent to write an `ask`. It is **model-neutral**: it names no
model and none of a model's numbers (no 0.8, no frame size, no cost, no logit gaps), and says that the end of the `ask` tool
description (§12.3) names the configured model, what it reads, its types and default `hit`, and its notes file, which the
agent reads before the first `ask`. Its examples use only noul, choice and score, with `hit` left out or `true`, so every
model takes them. It is under 400 lines, examples first, and each example is a call `ask` takes (`tests/test_skill.py` runs
each against every model's `About`). It covers:
- which tool: `find` first, `ask` when the answer is in pixels or sound, `look` to check (a sheet for motion, `region`
  to zoom in on a small thing or one area);
- the range in `since` and `until`, never in the question, `best_first` to rank a whole range, and one observable thing
  per question;
- the answer types every model answers, why the default `hit` stays sure, and when to raise `max_px` for small text;
- context, cost and jobs;
- checking sightings and candidates with `look`, and answering "not there" when only look-alikes hold up; writing back
  with `remember` and `label`, paging `find` with `after`;
- consent: `source_set` needs a cap that only rises, and the agent tells the owner to run `mms approve`, never runs it.

**The notes** are `skills/mms-questions/models/{d1,d1_3b,d3,h2o,pplx}.md`, one for each family, under 150 lines, installed with
the skill (§14) and named by `About.notes`. Each holds what that model needs the agent to know: the types and question keys it
takes, with examples; what it reads and hears; why its default `hit` is what it is and whether that was measured; its pass
size and cost (and which figures are assumed); and its weak spots. d1's carry the 640×360 and 1.6 MP figures, the compressed
probabilities and the pallet case that SKILL.md used to hold; d1_3b's say it hears nothing and name `mms model use d1-Omni-2B`
for sound, that it answers three types and takes `criteria` alone, and that its costs and `hit` are assumed; h2o's say a box
question names the thing in its instructions, and that its server holds the GPU until the owner stops it. A test checks that
every `About` names a file that exists and every
file is named. The Codex bench (`benchmarks/codex_run.py`) appends d1's notes to the skill, so what it is told is the same in
content.

## 16. d1 storage and recall

Settled on 31 Oct from two research notes (d1's own processors on the CPU and a GPU probe on the developer's laptop;
decoding, storage and the pipeline on the CPU) and `tests/test_gpu.py::test_timings`. Every choice is in the reader's profile
(`reader.PROFILE`, the one profile), the shipped `ask` or the presets, so a home copy of `ask.py` can change it. Only
`max_px` is new to §13.2's parameters; §8.2, §11 and §12 are unchanged.

**What each figure is.** *Measured* is a run on this laptop; *derived*, *extrapolated* and *assumed* are not. The figures
below are d1-Omni-2B's unless named. Every other model's `hit` is assumed until the rest of the bake-off (below).
- **Measured, the other models:** `tests/test_gpu.py`, 1-2 Nov, through the worker (H2O through its server), the median
  of 5 warm calls with one yes/no question. A 640×360 picture, a 1080p one and 2,000 characters: d1-3B 51, 177 (cut to its
  1 MP cap) and 50 ms; d1-2B 53, 487 and 47; d3-edge 41, 153 and 42; d3-mini 163, 573 and 135; d3-flash 184, 1,005 and
  225; pplx 276, 602 and 116; H2O 81, 113 and 87. Four questions cost about one on d1-3B (65 ms at 640×360), about twice
  one at 640×360 and four times at 1080p on d3 and pplx, and about twice on H2O. Loading took 14-46 s (d1-3B 20); peak
  GPU memory was 2.1 GiB (d3-edge) to 17.5 (d3-flash, which fits in 24 GB), d1-3B 6.1. H2O's server was ready in 233 s
  and holds 60 % of the GPU until it is stopped. d3 and pplx ran on transformers' reference kernels: their optional fast
  ones need Triton or a CUDA build, with no official Windows wheels. d1-3B ran with its `.hybrid` attention patched off
  the flash kernels (§13.3). d1-Omni-2B through the worker: 54 ms at 640×360 against 53 in process, with the same answers.
- **Measured, d1 alone:** `tests/test_gpu.py::test_timings`, 31 Oct, the median of 5 warm calls with one yes/no
  question, on synthetic pictures, a 5 s tone and 2,000 characters: a 640×360 picture 53 ms (220 tokens); 1920×1080
  489 ms (d1 caps it at 1.6 MP: 1,590 tokens); 5 s of sound 45 ms; 2,000 characters 44 ms.
- **Measured, the GPU probe,** with 4 questions: 640×360 95 ms and 1280×720 293 ms (880 tokens), so
  3× between them; 8 questions 103 ms in one call against 451 one by one; 4 windows batched 298 ms against 368.
- **Measured, the CPU**: decoding 2-35 ms a window; straight to size, 9 ms against 26 at 1080p.
  With frame threads (`reader.DECODE_THREADS`, 8) a 1408×1408 20 fps clip decodes 3-5× faster: 30-35 ms a window at `every_s` 1.
- **Not measured:** a window of text above 2,000 characters. The cost model (below) is an assumption, to change with
  the weights.

**The profile** (`reader.PROFILE`; a decider's `About.profile` overrides a key for it, and an ask's own override beats both):

| Key | Value | Why |
| --- | --- | --- |
| `max_px` | 230,400 (640×360's area); 0 for the model's own cap (as stored for a model with no cap) | Frames and pictures are cut to at most this many pixels, and to the model's cap (`About.max_px`) when that is lower, aspect kept, never up: 3× faster than 1280×720 (95 against 293 ms, probe) and 9× faster than 1080p (53 against 489 ms, `test_timings`). A 1920-wide screen reaches d1 at a third of its width, so small text is lost; an ask raises `max_px` for it (§13.2). |
| `every_s` | 1 | §13.2's default; a coarser step misses short events (deferred: coarse-to-fine). |
| `still_s` | 30 | A frame that looks the same as the one that began a window joins it, up to 30 s: an hour of a still camera is about 120 windows, not 3,600. The decider is deterministic, so a frame that looks the same would get the same verdict. "The same" is `reader.STILL_CELL_PX` and `STILL_LEVEL`: the mean colour of every 16 px cell within 6 of 255. A thing of contrast 100 moves a cell by more than 6 from about 9×9 px wherever it falls (about 5×5 inside one cell), roughly the smallest thing a model reading 640×360 resolves, about one of its vision patches; sensor and codec noise move it by under 1 and a change of light every cell, so it errs toward "changed": judging a frame again costs about 53 ms and skipping one can cost a sighting. Judged from the window's first frame, not the last, so slow drift adds up; the limit bounds a window's span. Unmeasured on real footage (deferred). 0 reads every frame the step picks. |
| `audio_s`, `audio_hop_s` | 5, 2.5 | 63 tokens; an event up to 2.5 s lies whole in one window. Not tested. |
| `quiet_rms` | 0.004 (-48 dBFS) | d1's own demo floor is 0.003; a home copy lowers it for faint sounds. |
| `text_chars` | 6,000 | A text version's window: about 1.7K tokens of prose or code (3.5 characters a token, d1's tokenizer). |
| `records_chars` | 3,000 | A run of records: about 1.5K tokens, at 2.0 characters a token. A short state answers better: 93% with one number in it, 60% with ten (probe 9). |

**640×360 for every scan departs from inputs' recommendation 1** (1280×720 for scene questions). The reasons: the
measured 3× in cost, which decides how much of a day a scan reads in `max_s`; presence of large things, which is what
the skill tells the agent to ask about, needs no more; and detail has its own knob, the per-ask `max_px`. What it
costs in accuracy is unmeasured (deferred: size against accuracy).

**Built:**
1. **Decode to size:** the reader decodes a frame straight to the profile's size in the decoder's scaler (PyAV
   `to_image(width=, height=)`, bicubic as d1 resizes): 9 against 26 ms a 1080p frame, 10.7 against 43.5 ms at
   2560×1600. A picture is resized with PIL to the same budget. `job.windows(profile=...)` takes an ask's
   overrides of the profile.
2. **`ask`** (§13.2):
   - one `decide` per window with every question;
   - the `none` option on a choice from `answers`: without one, d1 named a wrong option 96 times in 100; with one, it
     said none 100 times in 100 and never on an in-set case (probe 8);
   - the lint on more questions than a pass reads (d1: 16, each further 16 a new read of the window) and on types d1 reads
     only from text.
3. **The estimate** behind `progress.of_about` and the lint. Windows are counted from the index without decoding: a
   video's length over `every_s` (which over-counts still stretches), a sound's over the hop (and a video's
   sound when its meta has `sound`, which a captured segment records when it is indexed, §6), text and records by
   characters over the window size; an animation counts as one window. Frames count only for a decider that reads
   `video`, and sound only for one that reads `audio`. The cost of one window is the decider's, from
   its `About` (`ask.window_seconds` reads it, and multiplies it by the passes, `ceil(questions / questions_per_read)`);
   d1-Omni-2B's is `decider_d1.ABOUT`, and its `picture_s` points are the two measured sizes:
   - a frame or picture: 53 ms *measured* at 220 tokens, plus 0.32 ms for each token above that (*extrapolated* from the
     two measured sizes, 220 and 1,590 tokens; 1280×720 comes out at 0.27 s against 0.29 measured), plus 10 ms to
     decode (`ask.DECODE_S`, mms's own): 0.063 s at the default size, 0.49 s at d1's cap;
   - a sound window: 0.05 s (45 ms *measured*);
   - a window of text or records: 0.022 ms a character (*assumed* linear from 44 ms for 2,000 characters; nothing above
     2,000 is measured): 0.13 s for 6,000 characters, 0.07 s for 3,000. Inputs §3 *extrapolates* 0.6-0.9 s for a 2.9K-token
     state from the image slope, which the 44 ms contradicts at 2,000 characters;
   - `max_s` stays 600 s: about 9,500 frames at the default size.
4. **Storage stays** as §6 and §8 give it:
   - the screen at `max_width` 1920: d1's cap takes it only 11% smaller, `look`, OCR and a later model want the
     pixels, and storing at d1's size would tie the store to one model (principle 2);
   - key frames every 10 s, since a scan reads in order;
   - the camera and sound presets, the spool's JPEG at quality 90, and program `segment_s` 60;
   - no still-image sidecar or low-resolution track: it would save 2-30 ms of a 53-489 ms window, for a second copy of
     every frame.
5. **Unchanged:** the GPU queue sends one window at a time (§10), and `look`'s inline picture is a JPEG of at most 1568
   px on its long side (§11), since neither report covered the agent's model.
6. **Reading what phones and programs write** (`tests/test_corpus.py`): pictures upright by EXIF, transparency flattened
   onto white (onto dark grey when what is drawn is light), 16-bit and float scaled to 8 bits, a JPEG decoded at 1/2 to
   1/8 when that still covers the box, at most 64 MP decoded (`PICTURE_MAX_PX`); an animation a window per `every_s` of
   its own clock, a TIFF a window per page, 1 s apart (`every_s` does not thin them), an MPO's first picture only;
   video frames turned by their display matrix (not flipped); a video's soundtrack read as `audio` windows of that item,
   merged with its frames by start (oldest first) or end (newest first), its length that of the audio stream itself and
   only the last window set back to end where the sound does; sound decoded
   `AUDIO_BLOCK_S` (60 s) at a time from a seek, the resampler set up again when the rate or layout changes; text by
   `media.decode_text`, at most 64 MB of it (`TEXT_MAX_BYTES`); a version of a type the decider takes whole (`About.files`,
   d1 none) as one `file` window of its bytes, at most 64 MB (`FILE_MAX_BYTES`); any other file is not read, and `look`
   gives a copy (§11).
7. **The change gate** (`still_s`, §13.2): `reader._video` decodes each frame the step picks and joins it to the run it
   began when it looks the same as that run's first frame (`reader._same`) and is under `still_s` from it. A run is one
   window (its first frame in scan order, over its frames' span), never across items, so a captured segment gives at
   least one. `windows()` orders by `t1` going newest first and by `t0` oldest first, and a run's key frame is known
   when it starts, so the order holds. The estimate stays an upper bound. A camera that burns a running clock into its
   picture is never still (the seconds change a cell every frame), so it saves nothing: the bench's porch clips gave 26
   windows in 28 s against 29 without the gate. Turning the camera's overlay off gives the saving back; masking cells
   that change every frame would too, at some cost in small things, and waits on real footage.
8. **A sheet** (`look` over a range, §11): `reader.sheet` decodes up to 9 frames at the size of a tile alone, merges
   neighbours that look the same with the same test, and tiles them row-wise with burned-in numbers.

**Deferred**, each until what it waits on. The first four need one GPU session with one model alone on the GPU, with the
maintainer's go-ahead (TESTBED.md, manual checks):
- **The rest of the bake-off:** each model's `hit` on real hours. It replaces the assumed `hit`s and fills
  `About.measured`, as the costs did (above). TESTBED's GPU tier prints the figures.
- **Size against accuracy:** hit agreement at 640×360, 1280×720 and full size over a real hour of screen and camera.
  Then perhaps re-ask a sighting's first hit at full size or on a crop, or give screens a larger
  `max_px`. A full-size re-ask only removes false hits and never finds a missed one, and the agent's `look` already
  checks each sighting.
- **Batching** windows with `system_one_batch`: 19% faster at 4 (298 against 368 ms) for 0.4 GB more VRAM (probe g).
  It waits for a VRAM ramp at 4 and 8, since the machine crashed once under combined load.
- **Text window sizes** from d1's measured time at 1.5K and 3K tokens of state.
- **Prefetching** windows on a thread: at most the decode share, 4-8% at full size, and less at 640×360.
- **Signals at segment close** (a change score per second, RMS per hop) and frame times in `items.meta`: the gate now
  runs on decoded frames, so stored scores would save only the decode (and let the estimate count real windows).
- **The gate's threshold:** recall against the share of frames skipped, at a few thresholds, on real camera and screen
  hours with known small events. Until then it is a judgement that errs toward
  judging.
- **Coarse-to-fine in time** (a window every 5-10 s, then every second near hits): its evidence is video QA, where one
  frame answers, and it misses events shorter than its step. `every_s` is that knob for the agent today,
  and the lint suggests it.
- **Text first:** reading the windows around FTS matches in labels before the sweep, once `transcribe` and `ocr` write
  labels.
- **An answer cache** per window, model, profile and question set: d1 is deterministic, so a re-ask with another `hit`
  or a wider range would skip the windows already read. It would be a table of its own, made when missing, built once
  re-asks are seen to repeat windows. Activations cannot be cached: under set attention the state's keys depend on the
  questions after it.
- **Bursts** as one video item (2 frames for one frame's tokens) for "did it change" questions: no better than two
  pictures at motion (65-68%), "away" at 8-16%, and video frames skip d1's 1.6 MP cap.
- **Sound stored at 16 kHz mono:** untested.

**Not adopted:**
- storing at d1's size;
- key frames every 2 s;
- KV or vision-encoder caching;
- voice-activity detection at capture: it is a model in capture, Opus is already cheap in silence, and a transcript is
  the speech signal.

## 17. Code and what comes next

| File | Owns |
| --- | --- |
| `config.py` | `mms.toml` → `Config` and `Source`; kinds; the writer (§4.4, `set_decider`); `loopback` (`[decider] url`); `parse_when`, `iso`, `when` |
| `consent.py` | `consent.json`, fingerprints (§4.3) |
| `errors.py` | `ApiError`, the one error of the API (the daemon raises it, the client raises it again from the reply), and `Cancelled` |
| `store.py` | the schema and every read and write; `Scope`; the clock |
| `media.py` | what a file is from its content (§7): media type, a video's sound, text and line decoding, a media file's duration |
| `lock.py`, `children.py` | the home's lock; every child dies with the daemon |
| `capture.py` | ffmpeg sources: named devices, presets, launches, reconcile, `devices`, `apply` |
| `ingest.py` | folder, log and command sources; `remember` (note, file, folder, recording or picture at its real time); `apply` |
| `spool.py` | ingesting the spool: records, clock, frames into segments (§8.3) |
| `janitor.py` | floor, caps, thinning, ages, expiry, housekeeping |
| `reader.py` | windows over items and records, cut by the profile (§16; media versions read as media at the version's time; pictures upright and flattened, runs of frames that look the same as one window, soundtracks, sound a block at a time; `on_partial`); `look`'s files and a `sheet` of a range; `file` windows |
| `decider.py` | the decider protocol (`About`), loading one (`BACKENDS`, `claim`, `python:`), `about_of`, `check_of` and `known` (a model by name) |
| `gpu.py` | the GPU queue (`Gpu`, §10): the one way from the daemon's jobs to the decider |
| `modelenv.py` | a model's env: where it is, its `mms-env.json` marker, its Python, the pins compared with what is installed, and whether a backend's env is ready |
| `fetch.py` | downloading a model's folder from Hugging Face (§13.3): the one module that imports huggingface_hub, and what a finished folder is |
| `decider_d1.py`, `decider_d1_3b.py`, `decider_d3.py`, `decider_pplx.py`, `decider_h2o.py` | one family each (§13.3): its `ABOUT`s, how a folder is claimed, its pins, its translation, and the transport it uses |
| `native.py` | the translation d1-3B, d3, Perplexity and H2O share, and the score readout (`median_level`) |
| `transport.py` | `Worker` (a process, the frames of §13.3) and `Served` (the owner's loopback server) |
| `worker/` | what runs in a worker, with the standard library only and nothing from `mms`: `serve.py` (the loop, and `installed`, which `modelenv` reads too), `wire.py` (the frames), `d1.py`, `d1_3b.py`, `d3.py`, `pplx.py` (each family's `load` and `decide`) |
| `prepare.py` | `mms model`: list, use, prepare (§13.3), downloading what is missing (`fetch.py`) |
| `jobs.py` | jobs and the `Job` handed to scripts |
| `ops.py` | the ops of §11, from a JSON body to a JSON result, for the API and the CLI alike |
| `api.py` | the HTTP server, `daemon.json`, the token, dispatch to `ops` and `jobs` |
| `daemon.py` | `mms start` and `--detach`, startup, the loop, reload, stopping |
| `client.py` | the one way into the daemon: `is_running` and `connect` (by the lock), `call`, `ensure`, `wait_until_answering`; the adapter, the CLI and `--detach` all use it |
| `cli.py` | `mms start, stop, status, find, look, remember, label, keep, approve, ask, run, mcp, install, model`; it imports only what every command needs, so `mms mcp` starts light |
| `scripts/ask.py` | the shipped scan as `run(job)` and `cli(argv)` |
| `project.py` | the project ID (§12.2) |
| `adapter.py` | `mms mcp` (§12), and the model paragraph of `ask`'s description (§12.3) |
| `install.py` | `mms install` (§14), with its model step |
| `logger.py`, `__init__.py` | `mms.log`, `Logger`, `flush`, `stats` (§8.1) |
| `skills/mms-questions/SKILL.md`, `models/*.md` | the skill and the model notes (§15) |

**What comes next:**
- the rest of the bake-off and of §16's deferred measurements, in one GPU session (with the maintainer's go-ahead), and what they
  decide: each model's real `hit` (§13.3);
- sound by a second model, if a deaf model is to be used where sound matters (§13.3);
- the manual checks of TESTBED.md, which touch the real agent setup;
- reading documents for d1 by converting them (OOXML, ODF and EPUB text through `zipfile`; PDF pages through
  pypdfium2, with the maintainer's go-ahead), OLE sniffing (.doc, .xls, .msg), and HEIC through `pi-heif`;
- text read in blocks, for a text version over 64 MB.
- scripts for `transcribe` and `ocr` writing labels, and standing questions;
- the experiment of VISION.md's first open question.
