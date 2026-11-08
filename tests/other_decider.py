"""A second decider for tests (TESTBED.md, "No GPU"): a model that is nothing like d1, to prove a decider is swapped by
its `ABOUT` alone. Every fact mms uses differs from `fake_decider.ABOUT`, which is d1's, but the media window kinds it reads
(`tests/test_deciders.py` varies those on `About` objects):

- it reads documents whole: PDFs, Office and ODF files, EPUBs and HEIC pictures (`files`), and pulls the text out of them
  as a reader with a parser would (`doctext.py`, the standard library: a PDF's `Tj` strings, and a package followed from
  its root relationships, manifest or container to the parts that show words, so words in a part no path reaches are not
  found). A HEIC picture is handed to it whole and shows no words. It reads no document d1 cannot: a database, an archive,
  an OLE file, MIDI;
- it hears at 8 kHz in 2 s windows hopping 1 s, sees at most 102,400 px, reads 1,000 characters of text at a time, and
  understands speech;
- it answers only noul, choice and score, at most 4 questions in one pass, and reads text, dates and numbers from files
  and pictures as well as text;
- a question of it carries criteria and multiple and no other key, a yes/no `hit` of true counts at 0.6, only its sound
  cost was measured, and it has no notes file;
- it logs each call to `other-decider.jsonl` in the home (the loader sets `home`): per file its name, media type, sha256
  and size; the sizes of the pictures; each sound's length and rate; the state's length; the question types.

It answers as `fake_decider` does (the same words in a question pick what counts, and `contains:` also finds the words
inside a file), because it loads `fake_decider.py` and `doctext.py` by their paths: a daemon in another process has no
`tests/` on `sys.path`. It raises a `ValueError` for a question type it does not answer, and for sound not at its rate;
`doctext.Unreadable` (a `ValueError`) for a document whose package it cannot follow."""

import hashlib
import importlib.util
import json
import time
from pathlib import Path

from mms.decider import About

ABOUT = About(
    name="other", reads={"video", "image", "audio"},
    files=("application/pdf", "application/vnd.openxmlformats-officedocument.", "application/vnd.oasis.opendocument.",
           "application/epub+zip", "image/heic"),
    types=("noul", "choice", "score"), extracts_from={"text", "records", "image", "video", "file"},
    speech=True, audio_rate=8000, max_px=102_400, questions_per_read=4,
    picture_s=((102_400, 0.02),), sound_s=0.01, char_s=1e-5, file_s=0.2, hit=0.6,
    question_keys={"criteria", "multiple"}, notes="", measured={"sound_s"},
    profile={"max_px": 102_400, "audio_s": 2.0, "audio_hop_s": 1.0, "text_chars": 1000})

home = None  # set by the loader
_siblings: dict = {}


def sibling(name: str):
    """`tests/<name>.py` beside this file, loaded by its path (once)."""
    if name not in _siblings:
        found = importlib.util.spec_from_file_location(f"{name}_of_other", Path(__file__).with_name(f"{name}.py"))
        _siblings[name] = importlib.util.module_from_spec(found)
        found.loader.exec_module(_siblings[name])
    return _siblings[name]


def file_text(medium: dict) -> str:
    media, data = medium["media"], medium["data"]
    if media == "application/pdf" or media == "application/epub+zip" or media.startswith(
            ("application/vnd.openxmlformats-officedocument.", "application/vnd.oasis.opendocument.")):
        return sibling("doctext").text_of(data)
    return ""  # a picture: no words to pull out


def decide(state, questions: dict, media: list) -> dict:
    for name, question in questions.items():
        if question["type"] not in ABOUT.types:
            raise ValueError(f"other answers {', '.join(ABOUT.types)}, not {question['type']!r} (question {name!r})")
    files, pictures, sounds, words = [], [], [], []
    for medium in media:
        if not isinstance(medium, dict):
            pictures.append(list(medium.size))
        elif medium.get("type") == "audio":
            if medium["sampling_rate"] != ABOUT.audio_rate:
                raise ValueError(f"other is given sound at {ABOUT.audio_rate} Hz, not {medium['sampling_rate']}")
            sounds.append({"len": len(medium["audio"]), "rate": medium["sampling_rate"]})
        elif medium.get("type") == "file":
            files.append({"name": medium["name"], "media": medium["media"], "bytes": len(medium["data"]),
                          "sha256": hashlib.sha256(medium["data"]).hexdigest()})
            words.append(file_text(medium))
    seen = "\n".join(part for part in [str(state or ""), *words] if part)
    if home is not None:
        with open(home / "other-decider.jsonl", "a", encoding="utf-8") as out:
            out.write(json.dumps({"files": files, "pictures": pictures, "sounds": sounds, "state_len": len(state or ""),
                                  "types": [q["type"] for q in questions.values()], "at": time.time()}) + "\n")
    model = sibling("fake_decider")
    return {name: model._answer(question, model._yes(question["instructions"], seen, media), seen)
            for name, question in questions.items()}
