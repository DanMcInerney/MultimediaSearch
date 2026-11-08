# Perplexity Decider Flash (pplx-decider-flash-v1.2)

Notes for the mms-questions skill, for Perplexity's decision model (4.5B, built on Qwen3.5-4B, distilled from
Perplexity's 27B decision models). The `ask` description names the model now configured. What follows is from its card
and the code in its folder, or from mms's code. Its accuracy on mms's recordings and its default `hit` have not been
measured here.

## What it reads and answers

- **It sees** pictures and video frames, one frame a window, up to 1,638,400 pixels (1080p is cut). **It reads text:**
  a page of a file or a run of log records is the state it classifies.
- **It hears nothing.** Sound is not read. A mic segment is named in the not-read warning, and a camera's soundtrack
  in the not-heard one. A question about a sound finds nothing: say so to the owner.
- **It answers three types:** `noul` (yes/no), `choice` and `score`. A `text`, `date`, `number`, `box` or `boxes`
  question is refused before the scan (`bad_request`), and so is any question key but `criteria`: no `multiple`,
  `range`, `step` or `unit`. It reads no value out of text. For where something is, how many, or what a label says,
  find the moment with a yes/no and read it yourself with `look`.
- **`criteria`** is the one key. A choice needs it (`{option: what it means}`, up to 255 options). A score needs it
  (2 to 10 levels, lowest first). A yes/no may carry `{"true": ..., "false": ...}`.

## Questions are sent one at a time

mms sends each question to the model on its own, with the window. The card finds that more accurate than grouping for
most tasks (the questions of a group can influence each other), and it makes the answers repeatable. So the model
reads the window once for each question. Two questions cost up to twice a window, five up to five times. `ask` warns
when a request has more than one, and its estimate counts every pass. Ask one question when you can; put a second
condition in a second ask over the moments the first found.

The card finds grouping better only for facets of one decision, such as the options of a "select all that apply"
(+6.6 on its SATA-Bench), and worse for independent candidates (-4.2 to -20.1). mms never groups, so a "select all that
apply" cannot be asked the card's way:

- **Any of several things:** one choice, the things as `answers`, and `hit` listing them (one pass).
- **All of several things:** a yes/no for each in `questions` (a pass each), or the second in a second ask.
- **Which of several are there:** a choice names one option and cannot list them. Ask a yes/no about each, in its
  own ask over a narrow range, and `look`.

## Writing the question, answers and context

What the model reads for one question, from the card's code and mms's (a frame comes before all of it):

```text
State:
<context>

<the window's text, if it is a text or records window>

Question 0:
<instructions>
Options:
A: <first option>
B: <second option>
```

- **The state is `context`, then the window's text.** A frame has no text, so its state is `context` alone, and with
  no `context` it is empty: the model gets a frame and the question. The card says to put the context the questions
  need in the state. So give every picture ask a `context` that says what the picture is and where it is from (how
  much it helps has not been measured). For text, say what the lines are ("a time, then the message"). `context` is at
  most 4,000 characters and costs time like text (0.116 s for 2,000 characters): a sentence or two is enough.
- **`instructions` is the whole question.** Write one plain, full question about one thing, as the card's examples do
  ("Is the customer blocked right now?"), and name the thing as it looks.
- **Yes/no.** With no `criteria` the options are "No / false" and "Yes / true". With `criteria` the model reads your
  `false` text and then your `true` text as the two options, so give both (`ask` refuses one side). Say
  in `false` what the look-alike or the empty scene is, so the model has something to choose against.
- **Choice.** The model must pick one of the options it is shown (the card normalises over the listed options only),
  so cover the cases and add a "none" option. A choice names one option: use it for which one of exclusive
  possibilities, and a yes/no for a thing that may be there beside others.
  - In the `answers` form each answer is the whole option text, with no description: write each as a full phrase.
    `ask` adds `none: none of these`.
  - In the `questions` form an option is shown as `key: description` (the key alone when the description is null).
    Use a short plain key, add a description when the key could be read two ways, and keep the two from disagreeing.
    Add your own `none`. `hit` names keys.
  - The model's own `confidence` is not used for a choice's `p`.
- **Score.** The model sees only your level texts, lettered in your order, never numbers. Make each level something
  you could tell from its neighbours in a picture (what is on the desk, how much is covered), lowest first. Fewer
  levels are easier to tell apart (an inference, not measured).
- **Picture size.** A frame is cut to `max_px` (230,400 by default, 640x360). The card's cap is 1,638,400 pixels, the
  size the model was trained and evaluated at, so the default is well under it and small text is lost. Size costs
  little here: 7 times the pixels cost 2.2 times the time (see Cost). For a heading, a label or a small object, set
  `max_px` to 921,600 or 1,638,400 (the cap, which `0` gives) on a narrow range, after a default-size ask has found the
  moment.

## Hit and score

- **The default `hit` is 0.8, and it is assumed:** it has not been measured here. The card fitted a temperature of
  1.554 on held-out data and reports an expected calibration error of 0.021 on its development set (1,497 questions),
  so a probability is meant to be read at face value. That set is not mms's recordings, and the temperature was fitted
  on joint requests while mms sends one question each, so it may not carry over (an inference). `look` at the first
  sightings and at the `candidates`, and change `hit` on evidence.

## Cost

Measured on the developer's machine with one yes/no question (the median of 5 warm calls): 0.276 s for a 230,400-pixel picture,
0.602 s for a 1,638,400-pixel one, and 0.116 s for 2,000 characters of text. The estimate takes text as linear in
characters (assumed beyond 2,000): about 0.35 s for a 6,000-character page, 0.17 s for 3,000 characters of records. It
multiplies by the questions, and a large picture costs about that; at the default size four questions cost about
twice one. At one question and the default size, `max_s` of 600 covers at most about 2,100 windows.

## Examples

**A yes/no for a picture,** with the context and both meanings (the default `hit`):

```json
{"questions": {"parcel": {"type": "noul", "instructions": "Is a parcel on the doorstep?",
                          "criteria": {"true": "a parcel or box is on the doorstep",
                                       "false": "the doorstep is empty, or holds only a mat or shoes"},
                          "hit": true}},
 "context": "A frame of the front door camera, looking down at the doorstep.",
 "sources": ["porch"], "since": "1d", "limit": 3}
```

**A choice** (the `questions` form: short keys with descriptions, and a `none`; `hit` is a key):

```json
{"questions": {"build": {"type": "choice", "instructions": "What does the terminal show about the build?",
                         "criteria": {"failed": "red error text after a command",
                                      "passed": "green success text after a command",
                                      "running": "output still scrolling, with no final result",
                                      "none": "no terminal, or no build output"},
                         "hit": "failed"}},
 "context": "A frame of the laptop's screen.", "sources": ["screen"], "since": "6h", "limit": 1}
```

**A score,** each level something to tell by eye (`hit` is the least level that counts, counting from 0):

```json
{"questions": {"covered": {"type": "score", "instructions": "How much of the desk is covered?",
                           "criteria": ["the desk is bare", "a few items on it", "about half of it covered",
                                        "nearly all of it covered"],
                           "hit": 2}},
 "context": "A frame of the office camera, looking down at the desk.",
 "sources": ["office"], "since": "1d", "limit": 1}
```

**A yes/no over text,** where the words vary and `find` would miss some:

```json
{"question": "Do these lines show the program running out of memory?",
 "context": "Lines of a program's log: a time, then the message.",
 "sources": ["logs"], "since": "6h", "limit": 3}
```
