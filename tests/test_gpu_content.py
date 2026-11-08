"""Real content to the real models through mms's store and reader (TESTBED.md, GPU tests): each case stores a file that a
phone, a screen or a speaker would make, reads it back with `reader.windows` (what the model gets from `ask`), and asks it
one question of it and of its absence. For d1-Omni-2B a case passes when the file is on the right side of 0.5 and its
absence on the other; for every other model folder in `MMS_MODELS` it only prints the p's, which are bake-off data.
Opt-in, alone on the GPU: `uv run pytest -m gpu -n 0 -s tests/test_gpu_content.py -k d1-Omni-2B`. It prints each case's
P(yes) and saves every window the model was given (a PNG or a WAV) under the test's folder, which it prints: a miss is a
finding about the model or about mms, so look at the saved window before changing either. Models that hear nothing skip
the sound cases; ones that are not ready are skipped as in test_gpu.py."""

from __future__ import annotations

from pathlib import Path

import day
import pytest
from PIL import Image, ImageDraw
from test_gpu import folder, made, model  # noqa: F401  fixtures: one model of MMS_MODELS at a time

from mms import reader
from mms.store import Scope, Store

T0 = 1_700_000_000.0
pytestmark = [pytest.mark.gpu, pytest.mark.xdist_group("gpu")]  # one worker, so one copy of the model


@pytest.fixture
def store(home):
    one = Store(home)
    yield one
    one.close()


def stage(store: Store, case: str, yes: Path, no: Path | None) -> None:
    """Store the files as versions, the yes file in the source `<case>-yes`, the other in `<case>-no`."""
    for which, file in (("yes", yes), ("no", no)):
        if file is not None:
            store.add_version(f"{case}-{which}", file.name, T0, src=file)


def windows(model, store: Store, source: str, kinds: tuple[str, ...], **profile) -> list:
    """What the reader hands the model for what `source` holds (`quiet_rms` 0, so a quiet room is read too)."""
    found = [w for w in reader.windows(store, sources=[source], scope=Scope(all=True), about=model.about,
                                       newest_first=False, profile={"quiet_rms": 0, **profile}) if w.kind in kinds]
    assert found, f"the reader gave {model.name} no {'/'.join(kinds)} window for {source}"
    return found


def save(found: list, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(found):
        if w.kind == "audio":
            day.write_wav(folder / f"{i:02d}-at{w.at:g}s.wav", w.media["audio"], w.media["sampling_rate"])
        else:
            w.media.save(folder / f"{i:02d}-at{w.at or 0:g}s.png")


def yes_no(model, question: str, media) -> float:
    return model.decide(None, {"q": {"type": "noul", "instructions": question}}, [media])["q"]["noul"]


def measure(model, store, tmp_path: Path, case: str, question: str, kinds=("image", "video"), **profile) -> dict[str, float]:
    """P(yes) for the yes file and for the no file (the highest of their windows), printed with where the windows are."""
    out, tag = {}, "".join(f"-{k}{v}" for k, v in profile.items())
    for which in ("yes", "no"):
        found = windows(model, store, f"{case}-{which}", kinds, **profile)
        save(found, tmp_path / f"{case}-{which}{tag}")
        out[which] = max(yes_no(model, question, w.media) for w in found)
    print(f"\n{model.name} {case} {profile or ''}: {question} P(yes) with it {out['yes']:.2f}, without {out['no']:.2f}; "
          f"windows saved in {tmp_path} ({case}-yes{tag}, {case}-no{tag})")
    return out


def sides(model, p: dict[str, float]) -> None:
    """d1-Omni-2B must be on the right side of 0.5 for both; for another model the p's printed are the result."""
    if not model.asserts:
        return
    assert p["yes"] > 0.5, f"d1 does not see what is there (P(yes) {p['yes']:.2f})"
    assert p["no"] < 0.5, f"d1 sees what is not there (P(yes) {p['no']:.2f})"


def hears(model) -> None:
    if "audio" not in model.about.reads:
        pytest.skip(f"{model.about.name} does not hear")


def test_a_terminals_error_is_seen_and_a_success_is_not(model, store, tmp_path):
    def terminal(text: str, colour: tuple) -> Image.Image:
        image = Image.new("RGB", (1920, 1080), (24, 26, 30))
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(("$ python train.py --epochs 3", "epoch 1/3  loss 0.912", "epoch 2/3  loss 0.640")):
            draw.text((40, 40 + i * 60), line, fill=(200, 200, 200), font=day.font(40))
        draw.text((40, 40 + 3 * 60), text, fill=colour, font=day.font(40))
        return image

    terminal("RuntimeError: CUDA error: out of memory", (230, 60, 60)).save(tmp_path / "error.png")
    terminal("Build succeeded", (60, 200, 90)).save(tmp_path / "ok.png")
    stage(store, "terminal", tmp_path / "error.png", tmp_path / "ok.png")
    question = "Is there an error message on the screen?"
    full = measure(model, store, tmp_path, "terminal", question, max_px=0)  # printed, not asserted
    sides(model, measure(model, store, tmp_path, "terminal", question))
    print(f"at max_px 0 (the model's own cap): yes {full['yes']:.2f}, no {full['no']:.2f}")


def test_a_sideways_whiteboard_photo_is_read_upright(model, store, tmp_path):
    def board(text: str | None) -> Image.Image:
        image = Image.new("RGB", (1600, 900), (244, 244, 240))
        if text:
            ImageDraw.Draw(image).text((300, 330), text, fill=(15, 15, 40), font=day.font(170))
        return image

    day.sideways(tmp_path / "written.jpg", board("SHIP 2.1 FRI"), 6)
    day.sideways(tmp_path / "blank.jpg", board(None), 6)
    stage(store, "whiteboard", tmp_path / "written.jpg", tmp_path / "blank.jpg")
    sides(model, measure(model, store, tmp_path, "whiteboard", "Is there writing on the whiteboard?"))
    if "text" not in model.about.types:
        return
    for w in windows(model, store, "whiteboard-yes", ("image",)):
        try:
            said = model.decide(None, {"q": {"type": "text", "instructions": "What does the whiteboard say?"}}, [w.media])["q"]
        except Exception as error:  # printed, not asserted: d1 reads text from pictures poorly
            said = f"{type(error).__name__}: {error}"
        print(f"{model.name} whiteboard text question: {said}")


def test_a_portrait_videos_arrow_points_up(model, store, tmp_path):
    day.portrait_video(tmp_path / "arrow.mp4", day.arrow_card(), tmp_path, rotation=-90)
    stage(store, "arrow", tmp_path / "arrow.mp4", None)
    options = {"up": "the arrow points up", "down": "the arrow points down", "left": "the arrow points left",
               "right": "the arrow points right", "none": "none of these"}
    found = windows(model, store, "arrow-yes", ("video",))
    save(found, tmp_path / "arrow-yes")
    for w in found:
        answer = model.decide(None, {"q": {"type": "choice", "instructions": "Which way does the arrow point?",
                                           "criteria": options}}, [w.media])["q"]
        print(f"\n{model.name} arrow at {w.at:g} s, frame {w.media.size}: {answer['choice']} {answer['probabilities']}; "
              f"windows saved in {tmp_path / 'arrow-yes'}")
        if model.asserts:
            assert answer["probabilities"]["up"] > 0.5, f"the frame is not read as an arrow pointing up: {answer}"


def test_a_transparent_logos_words_are_seen_and_a_blank_one_is_not(model, store, tmp_path):
    logo = Image.new("RGBA", (640, 360), (0, 0, 0, 0))
    ImageDraw.Draw(logo).text((60, 130), "ACME CLOUD", fill=(0, 0, 0, 255), font=day.font(90))
    logo.save(tmp_path / "logo.png")
    Image.new("RGBA", (640, 360), (0, 0, 0, 0)).save(tmp_path / "blank.png")
    stage(store, "logo", tmp_path / "logo.png", tmp_path / "blank.png")
    sides(model, measure(model, store, tmp_path, "logo", "Is there text in the picture?"))


def test_speech_is_heard_and_a_quiet_room_is_not(model, store, tmp_path):
    hears(model)
    try:
        day.speak("Please bring the blue folder to the meeting", tmp_path / "speech.wav")
    except day.SpeechUnavailable as error:
        pytest.skip(str(error))
    day.write_wav(tmp_path / "quiet.wav", day.quiet(5))
    stage(store, "speech", tmp_path / "speech.wav", tmp_path / "quiet.wav")
    sides(model, measure(model, store, tmp_path, "speech", "Is someone speaking?", kinds=("audio",)))


def test_a_doorbell_is_heard_inside_a_window_and_a_quiet_room_is_not(model, store, tmp_path):
    hears(model)
    day.write_wav(tmp_path / "doorbell.wav", day.place(5, (1.0, day.ding_dong())))
    day.write_wav(tmp_path / "quiet.wav", day.quiet(5))
    stage(store, "doorbell", tmp_path / "doorbell.wav", tmp_path / "quiet.wav")
    sides(model, measure(model, store, tmp_path, "doorbell", "Is a doorbell ringing?", kinds=("audio",)))


def test_a_phone_ringing_in_a_meetings_soundtrack_is_heard(model, store, tmp_path):
    hears(model)
    card = Image.new("RGB", (320, 180), (110, 110, 115))
    for name, sound in (("ring", day.quiet(12) + day.place(12, (6.0, day.us_ring()))), ("none", day.quiet(12))):
        work = tmp_path / f"work-{name}"
        work.mkdir()
        day.still_with_sound(tmp_path / f"{name}.mp4", card, sound, work)
    stage(store, "meeting", tmp_path / "ring.mp4", tmp_path / "none.mp4")
    sides(model, measure(model, store, tmp_path, "meeting", "Is a phone ringing?", kinds=("audio",)))
