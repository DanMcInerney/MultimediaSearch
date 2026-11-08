"""The graders of the agents' answers (TESTBED.md): pure rules, no model. An answer is casefolded, number words become digits,
times become minutes after midnight. `tests/test_codex.py` grades its day's questions with the rules below, and
`benchmarks/bench.py` grades the haystack's (`benchmarks/haystack.py`) with `grade(spec, answer)`, where each question carries its own
`spec` (kind, truth, time, tolerance in minutes, wrong answers to refuse, `ignore`: the numbers and clock times the question
itself holds, which an answer may repeat, and `date`: the day a time must be on, or None). An answer that hedges (a second
number, time or date the question did not ask) is wrong.
They are pinned by fast tests that run no agent (`test_codex.py`, `test_haystack.py`, `test_bench_rules.py`)."""

from __future__ import annotations

import re
from datetime import date

UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen " \
        "eighteen nineteen".split()
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ONCE = {"once": "1", "twice": "2", "thrice": "3"}
WORD = re.compile(r"\b(?:(" + "|".join(TENS) + r")(?:[ -](one|two|three|four|five|six|seven|eight|nine))?|("
                  + "|".join(UNITS) + r"))\b")
TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*([ap])\.?m\b\.?)?")
KINDS = ("number", "count", "time", "code_time", "absent", "choice")
MONTH = (r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul"
         r"|aug|sept|sep|oct|nov|dec)\b\.?")
DAY = r"\d{1,2}(?:st|nd|rd|th)?"
ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}[t ](\d{1,2}:\d{2}(?::\d{2})?)(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?(?!\d)")
DATES = (re.compile(rf"\b(?:{DAY}\s+(?:of\s+)?{MONTH}|{MONTH}\s+{DAY})(?:,?\s+\d{{4}})?(?!\d)"),
         re.compile(rf"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)"
                    rf"\.?,?\s+{DAY}(?!\d)"),
         re.compile(r"\b\d{4}-\d{2}-\d{2}\b"))
WEEKDAYS = "mon tue wed thu fri sat sun".split()
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
# a date an answer may name, as `date_mentions` reads it; a clock time ("21:04") is never the day of a month or a year
NOT_CLOCK = r"(?![\d:])"
ISO_DATE = re.compile(rf"(?<!\d)(\d{{4}})-(\d{{2}})-(\d{{2}}){NOT_CLOCK}")
DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH})(?:,?\s+(\d{{4}}))?{NOT_CLOCK}")
MONTH_DAY = re.compile(rf"\b({MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?{NOT_CLOCK}")
# a weekday, with the day of the month after it when it has one ("Thursday 29", "Thu the 29th"); mon, wed, sat and sun are
# words too, so they count only before a full stop, a comma or a number
WEEKDAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tues?|thu(?:rs?)?|fri"
                     r"|(?:mon|wed|sat|sun)(?=[.,]|\s+\d))\b\.?"
                     rf"(?:,?\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?{NOT_CLOCK}(?!\s*[ap]\.?m\b))?")
# "on the 29th", "on 29th", "the 29th at 21:04": a day of the month by its ordinal, when a date and not "the 2nd alarm" (the
# ordinal ends the sentence, or a comma, a bracket, "at" or a time follows it)
BARE_DAY = re.compile(r"\b(?:on\s+(?:the\s+)?|the\s+)(\d{1,2})(?:st|nd|rd|th)\b"
                      r"(?=\s*(?:$|[,.;:()!?]|(?:at|around|about|@)\b|\d{1,2}:\d{2}))")
# a year in brackets, "(2026)", which is no number of an answer (unless it is the answer: a code can look like a year)
YEAR = re.compile(r"\(\s*(?:in\s+)?((?:19|20)\d{2})\s*\)")
# an abstaining or negating word (and a lone digit 0, which is not the start of 0.5 or 0:30)
NEGATION = re.compile(r"\b(?:unknown|none|never|nothing|nobody|nowhere|zero|no|not|nor|cannot|unable)\b|n/a|\b\w+n't\b"
                      r"|(?<![\w.:])0(?![\w:]|[.,]\d)")
# the time range an answer to an absent question may say it searched: "No, nothing (I checked 9:00-17:00)", "None found from
# 9:00 to 17:00", "I looked between 9:00 and 17:00 and saw nothing". `searched` takes it out of the answer's times only when it is
# SEARCHED_MIN minutes long or more (a sighting is a moment: "No; I checked 10:05-10:06" is not an abstention), and the rest of
# the answer is negative or says what was searched (each clause holds a negation, or only NEUTRAL words), with no "yes", no word
# of CONTRAST and none of PRESENT (a thing that went on: "it never closed")
CLOCK = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\b\.?)?"
RANGE = re.compile(rf"\bbetween\s+({CLOCK})\s+and\s+({CLOCK})|\b({CLOCK})\s*(?:-|–|—|to|until|till|through|thru)\s*({CLOCK})")
NEUTRAL = re.compile(r"\b(?:i|we|my|the|a|an|from|between|in|over|during|within|for|at|on|of|to|through|all|whole|entire|full|"
                     r"footage|recording|recordings|camera|range|window|period|today|also|again|carefully|thoroughly|went|"
                     r"(?:check|search|look|scann?|cover|review|watch|listen|inspect|examin|scour)\w*)\b")
CONTRAST = re.compile(r"\b(?:but|however|although|though|except|yet|apart|besides|aside|other than)\b")
PRESENT = re.compile(r"\byes\b|\b(?:still|remain\w*|stay\w*|kept|continu\w*|stopp\w*|clos\w*|whole time|entire time|all along|"
                     r"all the time|throughout|nonstop)\b")
SEARCHED_MIN = 60
# an option letter at the start of an answer ("B", "(b)", "B) Alice", "B - Alice"): then its end, a space or one of `.):,-`
CHOICE = re.compile(r"\(?([A-D](?=$|[\s.):,-])|[a-d](?=$|[.):,-]))")  # a lower-case letter alone or before a mark: "a screwdriver" picks none
WRAPS = ("<>", "[]", '""', "''", "``")  # what an answer may be wrapped in whole: `<B>`, `[B]`, `"B"`
# a second letter named after "or", "and", "/", "&" or a comma: "B or C", "B/C", "B, C" are hedges
HEDGE = re.compile(r"(?i:\bor\b|\band\b)\s*\(?[A-D]\b(?!')|[/&,]\s*\(?[A-D]\b(?!')")


def norm(text: str) -> str:
    """Casefolded, number words as digits ("four eight two one" is "4 8 2 1", "thirty" is "30")."""
    def number(m: re.Match) -> str:
        if m.group(3):
            return str(UNITS.index(m.group(3)))
        return str(TENS[m.group(1)] + (UNITS.index(m.group(2)) if m.group(2) else 0))
    return WORD.sub(number, text.casefold())


def clocks(text: str) -> str:
    """`norm(text)` with each ISO date and time left as its time alone: `2026-10-28T14:03:10-04:00` is ` 14:03:10 `."""
    return ISO.sub(r" \1 ", norm(text))


def mentions(text: str) -> list[list[float]]:
    """Each time of day in `text` as the minutes after midnight it may be: "2:03" with no am or pm is both 2:03 and 14:03. An
    ISO time (`2026-10-28T14:03:10-04:00`) is read as written: 14:03."""
    out = []
    for m in TIME.finditer(clocks(text)):
        hour, minute, second, half = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), m.group(4)
        if minute > 59:
            continue
        if half:
            hours = [hour % 12 + (12 if half == "p" else 0)]
        else:
            hours = [hour, hour + 12] if hour < 12 else [hour]
        out.append([h * 60 + minute + second / 60 for h in hours])
    return out


def minutes(text: str) -> list[float]:
    """Every time of day in `text` as minutes after midnight (see `mentions`)."""
    return [t for one in mentions(text) for t in one]


def digits_in(plain: str, keep=()) -> list[str]:
    """The numbers written in digits in `plain` (already casefolded; times and dates left out: `28 October`, `October 28`,
    `Tuesday 28`, `on the 28th`, `2026-10-28`, and a year in brackets, `(2026)`, unless it is one of `keep`), a run of single
    digits being one number: "4 8 2 1" is 4821."""
    plain = ISO.sub(r" \1 ", plain)
    for dates in DATES:
        plain = dates.sub(" ", plain)
    plain = BARE_DAY.sub(" ", plain)
    plain = YEAR.sub(lambda m: m.group(0) if m.group(1) in keep else " ", plain)
    plain = TIME.sub(" ", plain)
    plain = re.sub(r"\b\d(?: \d)+\b", lambda m: m.group(0).replace(" ", ""), plain)
    return re.findall(r"\d+(?:\.\d+)?", plain)


def numbers(text: str, keep=()) -> list[str]:
    """The numbers in `text`, number words as digits (see `norm`, `digits_in`)."""
    return digits_in(norm(text), keep)


def at(clock: str) -> float:
    """Minutes after midnight of "12:07" or "10:00:35"."""
    h, m, *s = (int(p) for p in clock.split(":"))
    return h * 60 + m + (s[0] / 60 if s else 0)


def near(text: str, clock: str, minutes_: float) -> bool:
    """Whether `text` has a time within `minutes_` of `clock` ("12:07" or "10:00:35")."""
    truth = at(clock)
    return any(abs(t - truth) <= minutes_ for t in minutes(text))


def unknown(text: str) -> bool:
    return re.match(r"\W*unknown\b", norm(text)) is not None


def code(text: str, n: str) -> bool:
    return re.search(rf"\b(?:err[- ]?)?{n}\b", norm(text)) is not None


def asked(question: str) -> list[str]:
    """The numbers and the clock times ("HH:MM") a question holds itself, for a grade's `ignore`: an answer may repeat them
    ("ERR-731", the hours of "in the evening (18:00 to 24:00)"; dates are not numbers)."""
    out = list(dict.fromkeys(numbers(question)))
    for m in TIME.finditer(clocks(question)):
        if int(m.group(2)) < 60:
            out.append(f"{int(m.group(1)):02d}:{m.group(2)}")
    return list(dict.fromkeys(out))


def split(ignore) -> tuple[set[str], list[float]]:
    """An `ignore` list as its numbers and its clock times (minutes after midnight)."""
    return {x for x in ignore if ":" not in x}, [at(x) for x in ignore if ":" in x]


def timed(answer: str, ignore_times: list[float], truth: float | None = None, tol: float = 0.0) -> bool:
    """Whether every time of day in `answer` is within `tol` minutes of `truth`, or one of `ignore_times` (to 30 s)."""
    return all(any((truth is not None and abs(t - truth) <= tol) or any(abs(t - i) <= 0.5 for i in ignore_times) for t in one)
               for one in mentions(answer))


def span(start: list[float], end: list[float]) -> float | None:
    """The shortest time from a reading of `start` to a later reading of `end` (each as `mentions` gives it: "10:05" with no am
    or pm is 10:05 and 22:05), or None when no reading of `end` is after one of `start`. Two times with no am or pm are read
    in the same half of the day: "10:05-10:06" is a minute, never twelve hours; "8:00 to 20:00" is twelve, not none."""
    pairs = zip(start, end) if len(start) == len(end) == 2 else ((a, b) for a in start for b in end)
    gaps = [b - a for a, b in pairs if b > a]
    return min(gaps) if gaps else None


def searched(plain: str) -> str:
    """`plain` with the time ranges it says it searched taken out, so that they are no times of a sighting. A range counts
    when it is SEARCHED_MIN minutes long or more (`span`), and the whole answer is one that abstains: no "yes", no word of
    CONTRAST or PRESENT, and every clause holds a negation or says only what was searched (NEUTRAL words), as in "no, nothing (I
    checked 9:00-17:00)" and "none found from 9:00 to 17:00". Any other range, and any single time, stays."""
    plain = ISO.sub(r" \1 ", plain)  # an ISO time's offset ("14:03:10-04:00") is no range
    if CONTRAST.search(plain) or PRESENT.search(plain):
        return plain

    def blank(m: re.Match) -> str:
        bounds = [mentions(t) for t in (m[1] or m[3], m[2] or m[4])]  # a bound that is no time, such as 12:75, has none
        gap = span(bounds[0][0], bounds[1][0]) if all(bounds) else None
        return " " if gap is not None and gap >= SEARCHED_MIN else m.group(0)

    blanked = RANGE.sub(blank, plain)
    clauses = (c.strip() for c in re.split(r"[,;.:!?()\n]|\band\b", blanked))
    clean = all(NEGATION.search(c) or not NEUTRAL.sub("", c).strip() for c in clauses if c)
    return blanked if clean else plain


def abstained(text: str | None, ignore=()) -> bool:
    """Whether an answer says the thing is not there: it holds a word of abstaining or negation (`unknown`, `none`, `never`,
    `nothing`, `nobody`, `no`, `not` (so `did not`, `could not`), `cannot`, `unable`, `n/a`, `zero`, `didn't` and the like), and
    every number in digits and every time in it is in `ignore` (the question's own: "Not sure, maybe 47" is a guess; a number
    word such as "one" is not a number here: "No, not a single one" abstains), or is the bound of a range it says it searched
    in an answer that is negative throughout (`searched`)."""
    if text is None:
        return False
    plain = searched(text.casefold().replace("’", "'"))
    digits, times = split(ignore)
    return (NEGATION.search(plain) is not None and all(n == "0" or n in digits for n in digits_in(plain))
            and timed(plain, times))


def choice(answer: str | None) -> str | None:
    """The option an answer picks: the letter A to D it starts with (any case, in brackets or not, then its end, a space or
    one of `.):,-`), upper case; None for any other start ("Alice", "The answer is B") and for a hedge, an answer that
    names a second letter after the first ("B or C"). One pair of `<>`, `[]`, quotes or backticks around the whole answer
    (`<B>`, `"B) Alice"`) is taken off first."""
    if answer is None:
        return None
    text = answer.strip()
    if len(text) > 2 and text[0] + text[-1] in WRAPS:
        text = text[1:-1].strip()
    found = CHOICE.match(text)
    return None if found is None or HEDGE.search(text[found.end():]) else found[1].upper()


def date_mentions(text: str) -> list[tuple]:
    """Each date `text` names, as `(year, month, day, weekday)` with None for what it does not say (the weekday is 0 for
    Monday): `2026-10-29`, `29 October` and `October 29` with or without a year, `Thursday` (or `Thu`) alone or with its
    day of the month (`Thursday 29`), or the day of the month by its ordinal alone (`on the 29th`)."""
    plain, found = norm(text), []
    for pattern, read in ((ISO_DATE, lambda m: (int(m[1]), int(m[2]), int(m[3]), None)),
                          (DAY_MONTH, lambda m: (int(m[3]) if m[3] else None, MONTHS.index(m[2][:3]) + 1, int(m[1]), None)),
                          (MONTH_DAY, lambda m: (int(m[3]) if m[3] else None, MONTHS.index(m[1][:3]) + 1, int(m[2]), None)),
                          (WEEKDAY, lambda m: (None, None, int(m[2]) if m[2] else None, WEEKDAYS.index(m[1][:3]))),
                          (BARE_DAY, lambda m: (None, None, int(m[1]), None))):
        found += [read(m) for m in pattern.finditer(plain)]
        plain = pattern.sub(" ", plain)  # a date is read once: "Thursday 29 October" is a weekday and a day and month only
    return found


def on_date(text: str, day: str) -> bool:
    """Whether `text` names the date `day` (ISO) and no other: every date in it agrees with `day` on what it says."""
    truth = date.fromisoformat(day)
    wanted = (truth.year, truth.month, truth.day, truth.weekday())
    named = date_mentions(text)
    return bool(named) and all(all(m is None or m == w for m, w in zip(one, wanted)) for one in named)


def grade(spec: dict, answer: str | None) -> bool:
    """Whether `answer` is right by a question's `spec` (`haystack`'s `grade`): None is wrong, and so is a hedge, an answer with
    a number or a time that is neither the truth nor in `spec["ignore"]` (the question's own numbers and times).
    - `number`: the truth is among the answer's numbers, no other number is, and no wrong answer is;
    - `count`: the answer's numbers, with the `ignore` ones taken out (an error code's digits), are exactly the truth (`once`,
      `twice` and `thrice` are 1, 2 and 3);
    - `time`: a time of the answer is within `tol_min` of `time`, every time of it is, or is ignored, and none is a wrong time;
    - `code_time`: the truth's digits (`E-4417`, `e4417` and `4417` all pass), a time within `tol_min` of `time` (every time of
      the answer within it or ignored), and no other number of as many digits;
    - `absent`: the answer abstains (`abstained`);
    - `choice`: the answer's letter (`choice`) is the truth, one of A to D ("unknown", an option's words and a hedge are wrong).
    A `time` or `code_time` spec may hold a `date` (ISO; None or no key is none): the answer must also name that date, by its
    weekday, as "29 October", "October 29" or "29 Oct" (with a year or not), as "the 29th" or as `2026-10-29`, and name no other
    date: a time alone is wrong (`on_date`). A year in brackets, "(2026)", is no number of an answer (a `number` answer keeps
    one that is its truth or a wrong answer), and nor is the year of the `date` in a `code_time` answer."""
    if answer is None:
        return False
    kind = spec["kind"]
    digits, times = split(spec["ignore"])
    if kind in ("time", "code_time") and spec.get("date") and not on_date(answer, spec["date"]):
        return False
    if kind == "number":
        found = set(numbers(answer, keep=[spec["truth"], *spec["wrong"]]))
        return spec["truth"] in found and not set(spec["wrong"]) & found and found <= {spec["truth"]} | digits
    if kind == "count":
        text = re.sub(r"\b(?:once|twice|thrice)\b", lambda m: ONCE[m.group(0)], norm(answer))
        for ignored in digits:
            text = re.sub(rf"\b{re.escape(ignored)}\b", " ", text)
        return set(numbers(text)) == {spec["truth"]}
    if kind == "time":
        return (near(answer, spec["time"], spec["tol_min"]) and timed(answer, times, at(spec["time"]), spec["tol_min"])
                and not any(near(answer, w, spec["tol_min"]) for w in spec["wrong"]))
    if kind == "code_time":
        wanted = re.sub(r"\D", "", spec["truth"])
        year = (spec.get("date") or "")[:4]  # the year of the day it names is part of the date, not another code
        others = [n for n in numbers(answer, keep=[wanted])
                  if len(n) == len(wanted) and n not in (wanted, year) and n not in digits]
        return (re.search(rf"(?<!\d){wanted}(?!\d)", norm(answer)) is not None and not others
                and near(answer, spec["time"], spec["tol_min"]) and timed(answer, times, at(spec["time"]), spec["tol_min"]))
    if kind == "absent":
        return abstained(answer, spec["ignore"])
    if kind == "choice":
        return choice(answer) == spec["truth"]
    raise ValueError(f"a question's grade is one of {', '.join(KINDS)}, not {kind!r}")
