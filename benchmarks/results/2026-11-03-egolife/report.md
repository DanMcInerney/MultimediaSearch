# EgoLifeQA through the mm bench

Run 20261103-0257-egolife-s1 (seed 1, 3 days, mm commit c7f62e9+dirty). 40 video questions (10 aged 0-2h, 10 aged 2-6h, 10 aged 6-24h, 10 aged 24h+; by category EventRecall 11, EntityLog 13, RelationMap 12, TaskMaster 4, HabitInsight 10) and 10 speech questions, asked as of each question's horizon. The pool before choosing: 143 video and 95 speech; dropped: 7 its time is a range or a list in one string, 1 a target is outside the clips (a gap) or after the horizon, 254 its query is after the last day. Each exec had 900 s. A glasses window decodes in 24 ms here (mm assumes 10).

Every arm saw the clips that ended by each question's horizon H (median 13 s before the question, at most 29 s; 0 questions more than 30 s, which fell after a gap) and the transcript lines that ended by H.

No shortcut probe: a public dataset, so the blind arm is the floor (what the agent knows without the recordings).

## codex with d1-3B: 40 video questions of 50 (paired: done in all three arms)

| arm | right (95% interval) | median s | median calls | tokens k | timed out | no letter |
| --- | --- | --- | --- | --- | --- | --- |
| blind | 16/40 (40%, 26-55) | 5.0 | 0 | 19 | 0 | 0 |
| files | 36/40 (90%, 77-96) | 30.5 | 6 | 4420 | 0 | 0 |
| mm | 34/40 (85%, 71-93) | 25.8 | 6 | 8302 | 0 | 0 |

| pair | n | difference (95% CI) | b / c | McNemar p |
| --- | --- | --- | --- | --- |
| mm vs files | 40 | -5 pts [-21, +11] | 4 / 6 | 0.754 |
| files vs blind | 40 | +50 pts [+29, +65] | 22 / 2 | <0.001 |
| mm vs blind | 40 | +45 pts [+24, +61] | 20 / 2 | <0.001 |

A difference is the second arm less the first, in points; b is the questions right only in the second, c only in the first.

### codex with d1-3B: by age (mm vs files; blind is the floor)

| age | n | blind (the floor) | files | mm | mm vs files | b / c | p | median s (files / mm) | tokens k (files / mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-2h | 10 | 4/10 (40%) | 9/10 (90%) | 8/10 (80%) | -10 pts | 1 / 2 | 1.000 | 28.1 / 27.5 | 64 / 151 |
| 2-6h | 10 | 3/10 (30%) | 9/10 (90%) | 10/10 (100%) | +10 pts | 1 / 0 | 1.000 | 45.0 / 46.6 | 156 / 192 |
| 6-24h | 10 | 4/10 (40%) | 9/10 (90%) | 9/10 (90%) | +0 pts | 1 / 1 | 1.000 | 25.4 / 21.4 | 103 / 220 |
| 24h+ | 10 | 5/10 (50%) | 9/10 (90%) | 7/10 (70%) | -20 pts | 1 / 3 | 0.625 | 44.6 / 41.0 | 120 / 267 |

The youngest age at which mm leads files: none: at no level with 10 or more paired questions does mm lead by 20 points, or cost half as much per correct answer without being less accurate (n = 10 per age: a sign, not a finding).

### codex with d1-3B: by category (right / n)

| category | n | blind | files | mm |
| --- | --- | --- | --- | --- |
| EventRecall | 9 | 4 | 8 | 5 |
| EntityLog | 9 | 3 | 9 | 8 |
| RelationMap | 9 | 3 | 9 | 8 |
| TaskMaster | 4 | 1 | 3 | 4 |
| HabitInsight | 9 | 5 | 7 | 9 |

TaskMaster has few questions in each age: read it as a count, not a rate.

### codex with d1-3B: the 10 speech questions, apart

| arm | right (95% interval) | median s | median calls | tokens k | timed out | no letter |
| --- | --- | --- | --- | --- | --- | --- |
| blind | 1/10 (10%, 2-40) | 5.5 | 0 | 5 | 0 | 0 |
| files | 9/10 (90%, 60-98) | 15.4 | 3 | 345 | 0 | 0 |
| mm | 8/10 (80%, 49-94) | 11.6 | 3 | 835 | 0 | 0 |

| pair | n | difference (95% CI) | b / c | McNemar p |
| --- | --- | --- | --- | --- |
| mm vs files | 10 | -10 pts | 0 / 1 | 1.000 |
| files vs blind | 10 | +80 pts | 8 / 0 | 0.008 |
| mm vs blind | 10 | +70 pts | 7 / 0 | 0.016 |

A difference is the second arm less the first, in points; b is the questions right only in the second, c only in the first.

Answered from the transcripts alone, since d1-3B does not hear.

### codex with d1-3B: how each arm worked, and what a correct answer cost (the video questions)

| arm | called ask | asks / finds / looks | ask windows | ask GPU s | asks stopped at max_s | pictures viewed | shell commands | correct | s per correct | tokens k per correct | GPU s per correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| files | - | - | - | - | - | 1.7 | 6.3 | 36/40 | 53.2 | 122.8 | - |
| mm | 14/40 (35%) | 1.4 / 4.1 / 1.1 | 271.4 | 38.7 | 1 | - | - | 34/40 | 76.6 | 244.2 | 45.6 |

### codex with d1-3B: the horizon, from the mm results

Recording times after H in mm tool results: 0 in 50 audited execs; asks started past H (the agent's confusion about mm's clock, harmless): 1/25.

## The decider tier (each model's yes/no over each question's evidence, no agent)

| model | set | asked | recall (at its hit) | p >= 0.5 | p >= 0.3 | top 5 | median needle p | sightings away from every target | median s per window | stopped at max_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d1-3B | all | 50 | 33/50 (66%) | 45/50 (90%) | 47/50 (94%) | 47/50 (94%) | 0.92 | 200 | 0.083 | 0 |
| d1-3B | video | 40 | 28/40 (70%) | 36/40 (90%) | 38/40 (95%) | 38/40 (95%) | 0.92 | 190 | 0.082 | 0 |
| d1-3B | speech | 10 | 5/10 (50%) | 9/10 (90%) | 9/10 (90%) | 9/10 (90%) | 0.92 | 10 | 0.162 | 0 |
| d1-3B | EventRecall | 11 | 8/11 (73%) | 10/11 (91%) | 10/11 (91%) | 10/11 (91%) | 0.92 | 68 | 0.081 | 0 |
| d1-3B | EntityLog | 13 | 8/13 (62%) | 12/13 (92%) | 12/13 (92%) | 12/13 (92%) | 0.93 | 29 | 0.086 | 0 |
| d1-3B | RelationMap | 12 | 9/12 (75%) | 11/12 (92%) | 11/12 (92%) | 11/12 (92%) | 0.88 | 48 | 0.083 | 0 |
| d1-3B | TaskMaster | 4 | 1/4 (25%) | 3/4 (75%) | 4/4 (100%) | 4/4 (100%) | 0.58 | 1 | 0.080 | 0 |
| d1-3B | HabitInsight | 10 | 7/10 (70%) | 9/10 (90%) | 10/10 (100%) | 10/10 (100%) | 0.94 | 54 | 0.082 | 0 |

Recall and the top 5 are read first. A sighting away from every target is told beside them and is no fault: the event may recur.

## Read with these

- The published scores (GPT-4o 36.2, Gemini-1.5-Pro 36.9 and EGAgent 57.5 in the paper) are on all 500 items and 7 days without tools: a scale, not a comparison.
- Chance is 25%. With 40 paired questions p < 0.05 needs about 6 or more discordant pairs all one way; by age, 10 questions each, only a sign.
- EgoLife has been public since 2025: the blind arm measures what the models know of it.
- The files and mm arms both get EgoLife's transcripts (a Chinese line and its English); mm reads them as records.
- A glasses window decodes in 24 ms here and mm assumes 10 ms, so its estimates and lint are low for these clips; asks stopped at `max_s` and timed-out execs are counted above.
- Names are neutral in the files and mm arms: the clips are `glasses/<date>/HH.MM.SS.mp4`, with no person or dataset name.
- A question is asked as of its horizon, so the seconds before it (above) are lost to every arm.
- Problems by kind: none. Results to run again: 0 of 150.

## Appendix: codex with d1-3B, one row per question

| qid | age | category | type | blind | files | mm |
| --- | --- | --- | --- | --- | --- | --- |
| ego-192 | 0-2h | HabitInsight | video | pass 13 s a0 | pass 28 s a0 | pass 10 s a0 |
| ego-129 | 0-2h | EventRecall | video | pass 5 s a0 | pass 66 s a0 | FAIL 101 s a1 |
| ego-184 | 0-2h | EntityLog | video | FAIL 4 s a0 | pass 10 s a0 | pass 10 s a0 |
| ego-004 | 0-2h | RelationMap | video | FAIL 4 s a0 | pass 19 s a0 | pass 15 s a0 |
| ego-092 | 0-2h | HabitInsight | video | FAIL 6 s a0 | pass 68 s a0 | pass 49 s a2 |
| ego-026 | 0-2h | EventRecall | video | FAIL 4 s a0 | pass 28 s a0 | FAIL 123 s a2 |
| ego-223 | 0-2h | EntityLog | video | pass 5 s a0 | pass 21 s a0 | pass 32 s a0 |
| ego-128 | 0-2h | RelationMap | video | pass 7 s a0 | pass 37 s a0 | pass 23 s a0 |
| ego-028 | 0-2h | HabitInsight | video | FAIL 10 s a0 | pass 21 s a0 | pass 13 s a0 |
| ego-216 | 0-2h | EventRecall | video | FAIL 4 s a0 | FAIL 57 s a0 | pass 38 s a1 |
| ego-196 | 2-6h | TaskMaster | video | FAIL 7 s a0 | pass 38 s a0 | pass 66 s a0 |
| ego-159 | 2-6h | EntityLog | video | pass 4 s a0 | pass 186 s a0 | pass 135 s a4 |
| ego-122 | 2-6h | EventRecall | video | FAIL 4 s a0 | pass 52 s a0 | pass 27 s a0 |
| ego-074 | 2-6h | HabitInsight | video | pass 4 s a0 | FAIL 60 s a0 | pass 73 s a2 |
| ego-093 | 2-6h | RelationMap | video | FAIL 5 s a0 | pass 20 s a0 | pass 20 s a0 |
| ego-036 | 2-6h | TaskMaster | video | FAIL 4 s a0 | pass 29 s a0 | pass 12 s a0 |
| ego-245 | 2-6h | EntityLog | video | FAIL 5 s a0 | pass 85 s a0 | pass 148 s a3 |
| ego-075 | 2-6h | EventRecall | video | FAIL 5 s a0 | pass 14 s a0 | pass 15 s a0 |
| ego-098 | 2-6h | HabitInsight | video | FAIL 4 s a0 | pass 18 s a0 | pass 25 s a0 |
| ego-155 | 2-6h | RelationMap | video | pass 4 s a0 | pass 91 s a0 | pass 139 s a3 |
| ego-109 | 6-24h | TaskMaster | video | FAIL 10 s a0 | FAIL 30 s a0 | pass 14 s a0 |
| ego-108 | 6-24h | HabitInsight | video | pass 4 s a0 | pass 76 s a0 | pass 19 s a0 |
| ego-142 | 6-24h | EventRecall | video | pass 4 s a0 | pass 20 s a0 | pass 11 s a0 |
| ego-163 | 6-24h | EntityLog | video | FAIL 7 s a0 | pass 131 s a0 | pass 196 s a6 |
| ego-062 | 6-24h | RelationMap | video | FAIL 5 s a0 | pass 45 s a0 | pass 35 s a0 |
| ego-119 | 6-24h | HabitInsight | video | FAIL 8 s a0 | pass 15 s a0 | pass 8 s a0 |
| ego-116 | 6-24h | EventRecall | video | FAIL 8 s a0 | pass 52 s a0 | FAIL 55 s a0 |
| ego-107 | 6-24h | EntityLog | video | pass 7 s a0 | pass 14 s a0 | pass 20 s a0 |
| ego-166 | 6-24h | RelationMap | video | FAIL 6 s a0 | pass 20 s a0 | pass 23 s a0 |
| ego-106 | 6-24h | HabitInsight | video | pass 5 s a0 | pass 13 s a0 | pass 24 s a1 |
| ego-157 | 24h+ | HabitInsight | video | pass 10 s a0 | FAIL 107 s a0 | pass 24 s a0 |
| ego-194 | 24h+ | TaskMaster | video | pass 8 s a0 | pass 58 s a0 | pass 53 s a0 |
| ego-203 | 24h+ | EventRecall | video | pass 4 s a0 | pass 18 s a0 | FAIL 29 s a0 |
| ego-233 | 24h+ | EntityLog | video | FAIL 6 s a0 | pass 26 s a0 | pass 94 s a2 |
| ego-214 | 24h+ | RelationMap | video | pass 4 s a0 | pass 77 s a0 | pass 16 s a0 |
| ego-195 | 24h+ | EventRecall | video | pass 5 s a0 | pass 29 s a0 | pass 120 s a4 |
| ego-200 | 24h+ | EntityLog | video | FAIL 7 s a0 | pass 31 s a0 | pass 74 s a2 |
| ego-219 | 24h+ | RelationMap | video | FAIL 7 s a0 | pass 61 s a0 | FAIL 24 s a0 |
| ego-193 | 24h+ | EntityLog | video | FAIL 7 s a0 | pass 137 s a0 | FAIL 683 s a21 |
| ego-168 | 24h+ | RelationMap | video | FAIL 4 s a0 | pass 9 s a0 | pass 11 s a0 |
| ego-204 | 24h+ | EventRecall | speech | FAIL 9 s a0 | pass 23 s a0 | FAIL 32 s a0 |
| ego-178 | 6-24h | EventRecall | speech | FAIL 6 s a0 | pass 20 s a0 | pass 12 s a0 |
| ego-053 | 0-2h | EntityLog | speech | FAIL 6 s a0 | pass 15 s a0 | pass 10 s a0 |
| ego-039 | 2-6h | EntityLog | speech | FAIL 5 s a0 | pass 13 s a0 | pass 12 s a0 |
| ego-201 | 24h+ | RelationMap | speech | FAIL 4 s a0 | pass 16 s a0 | pass 11 s a0 |
| ego-174 | 6-24h | EntityLog | speech | FAIL 5 s a0 | pass 13 s a0 | pass 9 s a0 |
| ego-077 | 0-2h | RelationMap | speech | FAIL 7 s a0 | FAIL 24 s a0 | FAIL 18 s a0 |
| ego-084 | 2-6h | RelationMap | speech | FAIL 6 s a0 | pass 10 s a0 | pass 10 s a0 |
| ego-133 | 24h+ | EntityLog | speech | pass 5 s a0 | pass 10 s a0 | pass 9 s a0 |
| ego-221 | 6-24h | HabitInsight | speech | FAIL 4 s a0 | pass 16 s a0 | pass 14 s a0 |
