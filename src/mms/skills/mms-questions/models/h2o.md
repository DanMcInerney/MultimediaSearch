# H2O-Lightning-4B

Notes for the mms-questions skill, for H2O.ai's H2O-Lightning-4B (built on Qwen3.5-4B). The `ask` description
names the model now configured. "The card" is the model's own README, read from its folder.

## What it reads and answers

- **It sees** pictures and video frames (one picture a request here), up to 1,638,400 pixels (1080p is cut). It also
  reads text windows and records.
- **It hears nothing.** Sound is not read. A mic segment is named in the not-read warning, and a camera's soundtrack
  in the not-heard one. A question about a sound finds nothing: say so to the owner.
- **It answers four types:** `noul` (yes/no), `choice`, `score` and `box`. A `text`, `date`, `number` or `boxes`
  question is refused before the scan (`bad_request`), and so is any question key but `criteria`: no `multiple`,
  `range`, `step` or `unit`. It reads no value out of text.
- **`criteria`** is the one key. A choice needs it (`{option: what it means}`). A score needs it (2 to 10 levels,
  lowest first). A yes/no may carry `{"true": ..., "false": ...}`, both or neither. A box takes none.

## What the model is sent

Each question is a prompt of its own (mms's request and the card's server code). The model picks a letter, and each
option's probability is read from it:

```text
[the picture, when the window is one]
record: <context>, then the page of text when the window is text
question: <instructions>
options:
A) true: <criteria.true, else "the statement holds">
B) false: <criteria.false, else "it does not">
```

- A choice's options read `A) key: description`, and a score's `A) 0: first level`, `B) 1: second level`.
- With `question` and `answers`, `ask` gives no descriptions, so each answer is shown twice (`A) a dog: a dog`) and
  the added option reads `none: none of these`. In `questions`, give a short key and a full description.
- The prompt tells it to choose exactly one option, so make options that exclude each other. For any of several
  things, list them as options and put them all in `hit`.

## Writing the question

- **English, one line, one thing.** The card evaluated it mostly in English, and its questions are one line: "Is this
  a full outage?", "Which team should take it?", "How urgent is it?". It is not a chat model and explains nothing, so
  ask nothing open ("what does it say", "why").
- **A yes/no** may be a question or a statement (the card uses both: "The problem affects every customer."). `question`
  leaves the default descriptions; use `questions` with `criteria` to say what each side shows, as the card does.
  Inference, not measured here: describe what is likely to be there instead in `false` (a mat on the doorstep).
- **A choice is what it was mostly trained on:** the card's training questions were 72.1% choice, 18.5% yes/no, 5.9%
  score and 3.5% box.
  Inference, not measured here: for a scene with a few known states, or a look-alike, a choice may read better than a
  yes/no. Make each look-alike an option and put the target in `hit`.
- **A score** is "How X is Y?" with levels of a word or two, lowest first (the card's: "Low, Medium, High").
- **A count** has no type. The card asks one as a choice over a few numbers ("How many item lines are on the
  receipt?", options "3" to "6"), with no accuracy given: `look` at it.

## Context

- It is the `record:` line, read before the question. For a picture it is the only text (with none, `record:` is empty).
- Say what the picture is of, in one short sentence: "A photo from the porch camera.", "A screenshot of the laptop's
  screen." The card's records name each image by what it is (`receipt`, `photo`); that this helps here is an inference.
- It may hold a rule the question uses, as the card's record does ("Severity guide: P1 = a revenue path fully down;
  ..."). Not measured here for pictures. Never the answer you hope for.

## A box question

Its `instructions` name the thing, not a question: a noun phrase with a colour, place or relation ("the red mug", "the
leftmost dog", "the error dialog"). No backticks (the card uses them to name an image in `state`), and it reads no
`context`.

- The server asks for the box with the instructions alone, then reads "Is <the thing> visible in the image?" as a
  yes/no. A window hits when a box came back and that read is 0.5 or more, and `p` is that read.
- **`said` gives the box's edges,** fractions of the picture from its top left (`at top 0.15, left 0.06, bottom 0.24,
  right 0.27`; `p` says how sure it is, and `look` shows the picture to check them). A window with no box says `none`.
- A box question costs more than a yes/no: a short generation (about 20 tokens, from the card's code) and a second
  read, not timed here. A text window cannot hold a box: it is skipped, with a warning.

## Hit and score

- **The default `hit` is 0.8, and it is assumed:** it has not been measured here. `look` at the first sightings and
  at the `candidates`, and change `hit` on evidence.
- **On unfamiliar images the card's probabilities track how often it is right** (76.2% right, calibration error 0.059
  on its vision benchmark). **On images of the kinds it was trained on they run somewhat low** (91.6% right, error
  0.078). So a true sighting can land under 0.8: read the `candidates` first.

## One question a pass

H2O reads the window once for each question, and `ask` warns when a request has more than one. Its estimate counts a
full read for each, so it reads high (four questions took about twice one in the measured test), but ask one question
when you can. A choice over several things is one question. If one condition rarely hits, ask it first, then ask the
second over the moments it found.

## The server, and cost

- **The server is the owner's.** H2O runs as a server the owner starts in WSL (`serve.sh`); mms only talks to it. If
  `ask` fails with a `decider` error that it is not answering, tell the owner to start it. It holds the GPU until the
  owner stops it, so ask the owner to stop it when the asking is done.
- **Cost is measured** on the developer's machine with one yes/no question and a plain picture: 0.08 s for a 230,400-pixel
  picture, 0.11 s for a 1,638,400-pixel one and 0.09 s for 2,000 characters of text (a 6,000-character window, about
  0.26 s, assumes the time grows with the length). At one question and the default size, `max_s` of 600 covers at most about
  6,600 windows.
- **A bigger picture is cheap to read here.** For small text raise `max_px` to 921,600 (about 0.10 s, interpolated)
  or 1,638,400, its cap, which `0` gives; a larger value is cut to it. Each picture goes up as a lossless PNG, which
  the server refuses over 20 MiB, and a picture at the cap is far under that. A busy picture adds encoding time, not
  measured.

## Examples

**A yes/no** (a statement, with what each side shows):

```json
{"questions": {"parcel": {"type": "noul", "instructions": "A parcel is on the doorstep.",
                          "criteria": {"true": "a parcel is on the doorstep",
                                       "false": "the doorstep is bare, or holds only a mat or shoes"},
                          "hit": true}},
 "context": "A photo from the porch camera.", "sources": ["porch"], "since": "1d", "limit": 3}
```

**A choice** (a short key and a full description for each option, and a `none` that says what else it may be):

```json
{"questions": {"front": {"type": "choice", "instructions": "What is in the foreground of the screen?",
                         "criteria": {"editor": "a code editor with source code open",
                                      "terminal": "a terminal window with a command prompt",
                                      "browser": "a web browser showing a page",
                                      "call": "a video call with people's faces",
                                      "none": "something else, or an empty desktop"},
                         "hit": "call"}},
 "context": "A screenshot of the laptop's screen.", "sources": ["screen"], "since": "1d", "every_s": 10}
```

**Any of several things** (the options are the things, all in `hit`; `ask` adds the `none`):

```json
{"question": "Which animal is on the doorstep?", "answers": ["a dog", "a cat"], "hit": ["a dog", "a cat"],
 "context": "A photo from the porch camera.", "sources": ["porch"], "since": "6h"}
```

**Whether a named thing is in view, and where** (a box; no `context`):

```json
{"questions": {"mug": {"type": "box", "instructions": "the red mug on the left", "hit": true}},
 "sources": ["kitchen"], "since": "3h", "limit": 1}
```
