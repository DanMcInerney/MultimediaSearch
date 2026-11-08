# mm benchmark: with mm and without

Runs pooled: 20261104-0314-pilot-s1 (seed 1, pilot, 3 days, commit 7895548+dirty).
Agents: claude claude-sonnet-5-5 (55 results); codex gpt-6.1-terra at medium effort (55 results).

## claude: accuracy with and without mm, on each model's questions

| model | questions | paired | without mm | with mm | difference (95% CI) | b / c | McNemar p | wrong on absent (files / mm) | abstained on needles (files / mm) | median s (files / mm) | median calls (files / mm) | timed out (files / mm) | tokens k (files / mm) | cost $ (files / mm) | left out | scope all | problems (files / mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d1-3B | all | 24/24 | 18/24 (75%) | 22/24 (92%) | +17 pts (too few discordant pairs: b 4, c 0) | 4 / 0 | 0.125 | 1/7 / 0/7 | 1/17 / 0/17 | 57.9 / 31.9 | 28 / 6 | 0 / 2 | 29860 / 3944 | 13.48 / 2.70 | 0 | 1 | capped 3 / timeout 2 |
| d1-3B | media | 16 | 10/16 (62%) | 14/16 (88%) | +25 pts (too few discordant pairs: b 4, c 0) | 4 / 0 | 0.125 | 1/5 / 0/5 | 1/11 / 0/11 | 94.0 / 45.2 | 34 / 8 | 0 / 2 | 28426 / 3353 | 12.38 / 2.36 | - | - | - |

## codex: accuracy with and without mm, on each model's questions

| model | questions | paired | without mm | with mm | difference (95% CI) | b / c | McNemar p | wrong on absent (files / mm) | abstained on needles (files / mm) | median s (files / mm) | median calls (files / mm) | timed out (files / mm) | tokens k (files / mm) | cost $ (files / mm) | left out | scope all | problems (files / mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d1-3B | all | 24/24 | 22/24 (92%) | 23/24 (96%) | +4 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 0/7 / 0/7 | 2/17 / 1/17 | 28.1 / 18.6 | 5 / 3 | 0 / 0 | 5326 / 2703 | - / - | 0 | 10 | none / none |
| d1-3B | media | 16 | 14/16 (88%) | 15/16 (94%) | +6 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 0/5 / 0/5 | 2/11 / 1/11 | 42.7 / 27.2 | 8 / 4 | 0 / 0 | 4940 / 2444 | - / - | - | - | - |

## claude: by how vague the time hint is (the media questions, paired)

| model | level | paired | without mm | with mm | difference (95% CI) | b / c | McNemar p | median s (files / mm) | tokens k per exec (files / mm) | timed out (files / mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d1-3B | part | 5 | 3/5 (60%) | 5/5 (100%) | +40 pts (too few discordant pairs: b 2, c 0) | 2 / 0 | 0.500 | 28.5 / 33.4 | 1463 / 291 | 0 / 0 |
| d1-3B | day | 7 | 5/7 (71%) | 6/7 (86%) | +14 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 66.0 / 45.0 | 2114 / 200 | 0 / 1 |
| d1-3B | week | 4 | 2/4 (50%) | 3/4 (75%) | +25 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 105.7 / 62.5 | 1577 / 231 | 0 / 1 |

Crossover, the first of part, day and week with at least 10 paired questions where mm leads by 20 points or more, or costs half as much per correct answer or less (seconds, or tokens) without being less accurate:

- d1-3B: none: at no level with 10 or more paired questions does mm lead by 20 points, or cost half as much per correct answer without being less accurate; too few paired questions at part (5), day (7), week (4)

## codex: by how vague the time hint is (the media questions, paired)

| model | level | paired | without mm | with mm | difference (95% CI) | b / c | McNemar p | median s (files / mm) | tokens k per exec (files / mm) | timed out (files / mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d1-3B | part | 5 | 5/5 (100%) | 5/5 (100%) | +0 pts (too few discordant pairs: b 0, c 0) | 0 / 0 | 1.000 | 40.6 / 21.3 | 287 / 110 | 0 / 0 |
| d1-3B | day | 7 | 6/7 (86%) | 7/7 (100%) | +14 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 44.8 / 26.8 | 298 / 163 | 0 / 0 |
| d1-3B | week | 4 | 3/4 (75%) | 3/4 (75%) | +0 pts (too few discordant pairs: b 0, c 0) | 0 / 0 | 1.000 | 56.5 / 42.6 | 355 / 188 | 0 / 0 |

Crossover, the first of part, day and week with at least 10 paired questions where mm leads by 20 points or more, or costs half as much per correct answer or less (seconds, or tokens) without being less accurate:

- d1-3B: none: at no level with 10 or more paired questions does mm lead by 20 points, or cost half as much per correct answer without being less accurate; too few paired questions at part (5), day (7), week (4)

## How each arm worked (the media questions, per exec)

| agent | model | arm | execs | called ask | asks / finds / looks | ask windows | ask GPU s | asks stopped at max_s | GPU s estimated | held s (run alone) | estimated past the timeout | pictures viewed | shell commands |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude | d1-3B | files | 16 | - | - | - | - | - | - | - | - | 17.8 | 18.5 |
| claude | d1-3B | mm | 16 | 16/16 (100%) | 5.2 / 2.1 / 4.2 | 1835.8 | 156.2 | 0 | 340.3 | 0.0 (5 alone) | 3 | - | - |
| codex | d1-3B | files | 16 | - | - | - | - | - | - | - | - | 3.4 | 7.2 |
| codex | d1-3B | mm | 16 | 16/16 (100%) | 2.8 / 1.1 / 2.1 | 1071.0 | 72.2 | 0 | 340.3 | 0.0 (5 alone) | 3 | - | - |

## Cost per correct answer (the media questions)

| agent | model | arm | correct (k) | seconds | tokens k | GPU seconds |
| --- | --- | --- | --- | --- | --- | --- |
| claude | d1-3B | files | 10/16 | 419.9 | 2842.6 | - |
| claude | d1-3B | mm | 14/16 | 200.9 | - | 178.5 |
| codex | d1-3B | files | 14/16 | 84.9 | 352.9 | - |
| codex | d1-3B | mm | 15/16 | 95.1 | 162.9 | 77.1 |

A cost per correct answer is told from 5 correct answers (k) in the arm; below that it is `-`.

## By data type: without mm -> with mm, right/paired

| agent | model | text | records | image | video | audio |
| --- | --- | --- | --- | --- | --- | --- |
| claude | d1-3B | 4/4 -> 4/4 | 4/4 -> 4/4 | 7/7 -> 7/7 | 3/9 -> 7/9 | - |
| codex | d1-3B | 4/4 -> 4/4 | 4/4 -> 4/4 | 7/7 -> 7/7 | 7/9 -> 8/9 | - |

## The decider tier (each model's yes/no over each needle's source and range, no agent)

No decider results yet.

## Summary per model (the media questions)

| model | family | recall (at its hit) | recall at p >= 0.5 | top 5 | false sightings per ask | median s per window | claude: mm accuracy | claude: s / k tokens / GPU s per correct | codex: mm accuracy | codex: s / k tokens / GPU s per correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d1-3B | d1 | - | - | - | - | - | 14/16 (88%) | 200.9 / - / 178.5 (k=14) | 15/16 (94%) | 95.1 / 162.9 / 77.1 (k=15) |

## Read with these

- Without mm the screen, the microphone and the cameras would not be recorded at all: the files arm reads a week that exists only because the bench wrote it.
- The screen changes like a real one while the owner is active: typing (small changes), scrolling (large ones), a cursor that moves and a clock, up to 2 frames a second, with idle and busy stretches. Its totals and the size of its changes are calibrated against two real hours of a screen recording (TESTBED.md). A caret's blinks are not modelled, and a scroll wraps instead of bringing new lines; the reader's change gate turns 98% of the haystack's seconds with a frame into windows, where it turns 71 to 86% of a real screen's.
- `look` costs about 25 times more per frame than an ffmpeg montage: it returns separate pictures, 8 to a call, where a 5x5 montage is about 1.5k tokens. An agent that opens frames with `look` pays for it; one that tiles them does not.
- With about 20 paired questions, p < 0.05 needs at least 6 discordant pairs all one way (McNemar exact: mcnemar(6, 0) = 0.031, mcnemar(5, 0) = 0.0625). A run with another seed adds new needles, and `report` pools runs. By hint level there are a few questions each: pool seeds before reading it.
- The text and records questions do not depend on the model, so only one model's mm arm (d1-3B) is asked them: the other models' rows hold the media questions.
- The decider tier's recall at p >= 0.5 counts a needle whose best p within 10 s of it, as a sighting or a candidate, is 0.5 or more: where it is above the recall at the model's hit, that hit may be set too high.
- Tokens are the input, the output and for Claude the cache it read and wrote (Codex's input already holds its cached tokens): a size of work, not a bill, since a cached token costs much less. Cost is Claude's list-price estimate; Codex reports none.
- The difference is told in points with Newcombe's hybrid score interval for paired data, only from 10 pairs right in one arm only (b + c); below that the interval would exclude 0 where McNemar's exact p does not, so b and c and the p are given alone. A verdict rests on McNemar's p. A timed-out exec is a wrong answer and is counted on its own too (the `timed out` columns).
- An mm exec whose ask may need more GPU seconds than 75% of the exec's time (675 s of 900: the agent's own turns take the rest; the `held s` and `estimated past the timeout` columns, and the appendix) may time out because of the model's speed alone: that is not a miss of its eyes, and `agents` names those questions before it runs.
- The rows are not independent: the part and day questions of a template share a needle on the `pilot` scale, a template's levels share its week, and pooled seeds repeat templates. Read a level or a model with a few dozen questions as a sign to follow up, not as a finding.
- Multiple comparisons: this report has k = 2 agent-model rows, each tested on all its questions and on the media alone (4 p values). At p < 0.05 about 0.10 of k rows would fall below it by chance alone; 0 of the 4 p values here are.
- Ask labels with no project (an exec whose questions could leak into another's scope): 0. Problems by kind: capped 3, timeout 3. Results to run again: 0 of 110.
- Paired: a question counts when both arms have a done result. An escaped, timed-out or capped exec is a wrong answer; a `setup`, `limit`, `auth` or `error` result did not test the question, is run again by the next command, and is left out (the `left out` column).
- Runs pooled: 1; seeds 1.

## Appendix: claude, one row per question

| qid | level | type | files | mm d1-3B |
| --- | --- | --- | --- | --- |
| text-note-code-1 | part | text | pass 28 s v0 | pass 10 s a0 |
| text-note-code-2 | day | text | pass 14 s v0 | pass 18 s a0 |
| text-config-asof-1 | time | text | pass 10 s v0 | pass 12 s a0 |
| text-absent-note-1 | part | text | pass 221 s v79 | pass 23 s a0 |
| records-log-count-1 | part | records | pass 6 s v0 | pass 11 s a0 |
| records-log-count-2 | day | records | pass 18 s v0 | pass 15 s a0 |
| records-log-last-1 | part | records | pass 24 s v0 | pass 9 s a0 |
| records-absent-log-1 | day | records | pass 26 s v0 | pass 30 s a0 |
| image-photo-sign-1 | part | image | pass 22 s v20 | pass 11 s a1 |
| image-photo-sign-2 | day | image | pass 59 s v29 | pass 23 s a1 |
| image-photo-sign-3 | week | image | pass 90 s v11 | pass 50 s a4 |
| image-photo-count-1 | day | image | pass 43 s v8 | pass 14 s a1 |
| image-photo-count-2 | week | image | pass 98 s v7 | pass 38 s a1 |
| image-absent-sign-1 | part | image | pass 28 s v22 | pass 29 s a2 |
| image-absent-sign-2 | day | image | pass 66 s v6 | pass 45 s a3 |
| video-screen-error-1 | part | video | FAIL 412 s v7 | pass 46 s a1 |
| video-screen-error-2 | day | video | pass 827 s v11 | pass 170 s a4 (est. past timeout) |
| video-screen-error-3 | week | video | FAIL capped 665 s v4 | FAIL timeout 900 s a27 (est. past timeout) |
| video-porch-parcel-1 | part | video | pass 28 s v11 | pass 33 s a1 |
| video-porch-parcel-2 | day | video | pass 57 s v15 | pass 34 s a1 |
| video-porch-parcel-3 | week | video | FAIL 114 s v35 | pass 75 s a2 |
| video-absent-error-1 | part | video | FAIL capped 598 s v16 | pass 387 s a11 |
| video-absent-error-2 | day | video | FAIL capped 807 s v14 | FAIL timeout 900 s a21 (est. past timeout) |
| video-absent-parcel-1 | day | video | FAIL 284 s v68 | pass 57 s a2 |
| audio-mic-doorbell-1 | part | audio | pass 139 s v28 | - |
| audio-mic-doorbell-2 | day | audio | FAIL 156 s v9 | - |
| audio-mic-alarm-1 | part | audio | pass 82 s v15 | - |
| audio-mic-alarm-2 | day | audio | pass 158 s v33 | - |
| audio-mic-alarm-3 | week | audio | FAIL timeout 900 s v15 | - |
| audio-absent-doorbell-1 | part | audio | pass 376 s v16 | - |
| audio-absent-doorbell-2 | day | audio | pass 336 s v16 | - |

## Appendix: codex, one row per question

| qid | level | type | files | mm d1-3B |
| --- | --- | --- | --- | --- |
| text-note-code-1 | part | text | pass 11 s v0 | pass 14 s a0 |
| text-note-code-2 | day | text | pass 12 s v0 | pass 12 s a0 |
| text-config-asof-1 | time | text | pass 11 s v0 | pass 11 s a0 |
| text-absent-note-1 | part | text | pass 59 s v3 | pass 12 s a0 |
| records-log-count-1 | part | records | pass 11 s v0 | pass 7 s a0 |
| records-log-count-2 | day | records | pass 13 s v0 | pass 11 s a0 |
| records-log-last-1 | part | records | pass 12 s v0 | pass 8 s a0 |
| records-absent-log-1 | day | records | pass 18 s v0 | pass 12 s a0 |
| image-photo-sign-1 | part | image | pass 22 s v1 | pass 14 s a1 |
| image-photo-sign-2 | day | image | pass 25 s v2 | pass 18 s a1 |
| image-photo-sign-3 | week | image | pass 29 s v2 | pass 19 s a1 |
| image-photo-count-1 | day | image | pass 28 s v2 | pass 19 s a1 |
| image-photo-count-2 | week | image | pass 36 s v3 | pass 28 s a2 |
| image-absent-sign-1 | part | image | pass 22 s v1 | pass 15 s a1 |
| image-absent-sign-2 | day | image | pass 23 s v1 | pass 17 s a1 |
| video-screen-error-1 | part | video | pass 136 s v10 | pass 125 s a4 |
| video-screen-error-2 | day | video | FAIL 124 s v5 | pass 253 s a7 (est. past timeout) |
| video-screen-error-3 | week | video | FAIL 194 s v3 | FAIL 330 s a10 (est. past timeout) |
| video-porch-parcel-1 | part | video | pass 41 s v2 | pass 21 s a1 |
| video-porch-parcel-2 | day | video | pass 45 s v3 | pass 27 s a1 |
| video-porch-parcel-3 | week | video | pass 77 s v6 | pass 58 s a1 |
| video-absent-error-1 | part | video | pass 124 s v3 | pass 158 s a4 |
| video-absent-error-2 | day | video | pass 212 s v6 | pass 286 s a8 (est. past timeout) |
| video-absent-parcel-1 | day | video | pass 50 s v5 | pass 38 s a1 |
| audio-mic-doorbell-1 | part | audio | pass 69 s v7 | - |
| audio-mic-doorbell-2 | day | audio | FAIL 78 s v2 | - |
| audio-mic-alarm-1 | part | audio | pass 43 s v0 | - |
| audio-mic-alarm-2 | day | audio | pass 78 s v0 | - |
| audio-mic-alarm-3 | week | audio | FAIL 102 s v3 | - |
| audio-absent-doorbell-1 | part | audio | pass 107 s v12 | - |
| audio-absent-doorbell-2 | day | audio | pass 86 s v3 | - |
