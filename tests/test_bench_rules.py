"""The rules the benchmark's harder haystack leans on (TESTBED.md, Benchmark), pinned with no agent and no daemon: the fake
decider's `shade:R,G,B` (a part of a frame in one shade, found after JPEG and video coding, and not found in a shade 60
away), and the grader's `date` (a time must be on the right day, by weekday, "29 October" or ISO, and on no other)."""

from __future__ import annotations

import io
from datetime import date

import av
import day
import fake_decider
import grade
import numpy as np
import pytest
from PIL import Image

from mms.decider_d1 import ABOUT

SIZE = (640, 360)  # what the reader hands a decider for a screen or a porch camera
BACKDROP = (70, 90, 130)  # 100 or more from every shade below
RED, BROWN, GREEN = (200, 40, 40), (125, 80, 45), (30, 160, 60)


def framed(rgb, share: float = 0.2) -> Image.Image:
    """A frame of faint noise with one box of exactly `rgb` over about `share` of it."""
    w, h = SIZE
    box_w, box_h = int(w * share ** 0.5), int(h * share ** 0.5)
    pixels = np.random.default_rng(1).integers(-6, 7, (h, w, 3)) + BACKDROP
    x, y = (w - box_w) // 3, (h - box_h) // 3
    pixels[y:y + box_h, x:x + box_w] = rgb
    return Image.fromarray(pixels.astype(np.uint8))


def jpeg(image: Image.Image) -> Image.Image:
    """The picture after a JPEG round trip at quality 90 (what a photo of the haystack is saved at)."""
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def x264(image: Image.Image, folder) -> Image.Image:
    """The picture after an x264 round trip, as the haystack's screen and porch files are written, decoded by PyAV as the
    reader does."""
    path = folder / "frame.mp4"
    day.vfr_video(path, folder, [(image, 2.0)])
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_image()


def says(instructions: str, *media) -> bool:
    """Whether the fake decider answers yes to a yes or no question over `media`."""
    answer = fake_decider.decide(None, {"q": {"type": "noul", "instructions": instructions}}, list(media))
    return answer["q"]["noul"] > 0.5


def shade_of(rgb) -> str:
    return "shade:" + ",".join(str(v) for v in rgb)


def test_a_shade_is_the_pixels_within_30_by_euclidean_distance():
    def flat(rgb) -> Image.Image:
        return Image.new("RGB", (64, 36), rgb)

    assert (fake_decider.SHADE_TOL, fake_decider.SHADE_SHARE) == (30, 0.02)
    assert fake_decider.shade_share(flat(RED), RED) == 1.0
    assert fake_decider.shade_share(flat((200, 69, 40)), RED) == 1.0  # 29 away
    assert fake_decider.shade_share(flat((200, 71, 40)), RED) == 0.0  # 31 away
    # 25 off in two channels is 35 away: within 30 in each channel is not within 30
    assert fake_decider.shade_share(flat((200, 65, 65)), RED) == 0.0
    assert fake_decider.shade_share(flat((200, 52, 52)), RED) == 1.0
    assert fake_decider.shade_share(Image.new("RGBA", (8, 8), RED + (255,)), RED) == 1.0, "any mode is read as RGB"
    assert fake_decider.shade_share(Image.new("RGB", (0, 0)), RED) == 0.0


def test_a_shaded_box_is_found_at_the_share_threshold_and_not_below_it():
    def with_pixels(n: int) -> Image.Image:
        """A frame with exactly `n` pixels of RED, the first ones in reading order."""
        pixels = np.full((SIZE[1] * SIZE[0], 3), BACKDROP, dtype=np.uint8)
        pixels[:n] = RED
        return Image.fromarray(pixels.reshape(SIZE[1], SIZE[0], 3))

    at = round(SIZE[0] * SIZE[1] * fake_decider.SHADE_SHARE)
    assert at == 4608
    assert not says(shade_of(RED), with_pixels(at - 100)) and says(shade_of(RED), with_pixels(at + 100))
    assert says(shade_of(RED), with_pixels(0), with_pixels(at + 100)), "one picture of several is enough"
    assert not says(shade_of(RED), with_pixels(0))
    assert not says(shade_of(RED))


@pytest.mark.parametrize("rgb", [RED, BROWN, GREEN])
def test_a_shaded_box_of_a_fifth_of_the_frame_survives_jpeg_and_x264(rgb, tmp_path):
    image = framed(rgb)
    assert fake_decider.shade_share(image, rgb) == pytest.approx(0.2, abs=0.01)
    for how, coded in (("jpeg", jpeg(image)), ("x264", x264(image, tmp_path))):
        assert coded.size == SIZE
        share = fake_decider.shade_share(coded, rgb)
        assert 0.17 <= share <= 0.23, f"{how}: the box is {share:.3f} of the frame after coding, from 0.2"
        assert says(shade_of(rgb), coded), how


def test_a_shaded_box_as_small_as_a_needle_may_be_still_clears_the_threshold_after_coding(tmp_path):
    image = framed(RED, 0.05)
    for how, coded in (("jpeg", jpeg(image)), ("x264", x264(image, tmp_path))):
        assert fake_decider.shade_share(coded, RED) >= 0.03, f"{how}: a needle's 3% floor must not fall under the 2% threshold"
        assert says(shade_of(RED), coded), how


def test_a_shade_60_away_does_not_match_before_or_after_coding(tmp_path):
    near = (140, 40, 40)  # 60 from RED
    assert float(np.linalg.norm(np.subtract(near, RED))) == 60.0
    image = framed(near)
    for how, picture in (("raw", image), ("jpeg", jpeg(image)), ("x264", x264(image, tmp_path))):
        assert fake_decider.shade_share(picture, RED) < 0.005, how
        assert not says(shade_of(RED), picture), how
        assert says(shade_of(near), picture), how


def test_shade_finds_a_box_anywhere_and_leaves_the_other_predicates_as_they_were():
    corner = Image.new("RGB", SIZE, (245, 245, 240))
    corner.paste(Image.new("RGB", (SIZE[0] // 2, SIZE[1] // 2), (255, 0, 0)), (0, 0))  # the top-left quarter in red
    lower = framed(RED)  # a box that is not in the top-left quarter and is not the mean colour
    assert says("corner:red", corner) and not says("corner:red", lower)
    assert not says("colour:red", lower) and says("colour:red", Image.new("RGB", SIZE, (255, 0, 0)))
    assert not says(shade_of(RED), corner), "255,0,0 is not within 30 of 200,40,40"
    assert says(shade_of(RED), lower) and says(shade_of((255, 0, 0)), corner)
    tone = np.sin(2 * np.pi * 659 * np.arange(ABOUT.audio_rate) / ABOUT.audio_rate).astype(np.float32) * 0.3
    sound = {"type": "audio", "audio": tone, "sampling_rate": ABOUT.audio_rate}
    assert says("pitch:659", sound) and not says("pitch:900", sound) and not says(shade_of(RED), sound)


# -- the grader's date --------------------------------------------------------------------------------------------------

DAY = "2026-10-29"
assert date.fromisoformat(DAY).strftime("%A") == "Thursday"


def spec(kind: str = "time", when=DAY, truth=None, time="21:04:10", ignore=()) -> dict:
    return {"kind": kind, "truth": truth, "time": time, "tol_min": 1, "wrong": [], "ignore": list(ignore), "date": when}


def test_a_time_with_a_date_set_is_right_on_that_day_in_each_form():
    one = spec()
    right = ["Thursday 29 October 2026 at 21:04", "Thursday 29 October at 21:04", "21:04 on 29 October", "On 29 October, 21:04",
             "29 Oct 2026, 21:04", "October 29 at 21:04", "Oct. 29th, 2026, 9:04 pm", "The 29th of October at 21:04",
             "2026-10-29T21:04:10-04:00", "2026-10-29 21:04", "21:04, 2026-10-29", "Thursday at 21:04", "21:04 on Thursday",
             "Thursday, October 29 at 21:04", "Thu 29 Oct 21:04", "Thu. 21:04", "Thursday the 29th at 21:04",
             "Thursday 29 at 21:04", "twenty-nine October at 21:04", "October 29 21:04", "Thursday 9:04 pm"]
    assert [a for a in right if not grade.grade(one, a)] == []


def test_a_time_with_a_date_set_is_wrong_with_no_date_or_with_another_one():
    one = spec()
    wrong = ["21:04", "at 21:04 yesterday", "the sun was low at 21:04", "unknown",
             "Thursday 29 October at 22:30", "Thursday 29 October",  # the right day and a wrong time, and no time
             "Tuesday 27 October at 21:04", "Tuesday at 21:04",  # another date in the week, by weekday too
             "Thursday 22 October at 21:04", "Thursday 22 at 21:04",  # the right weekday and another date
             "Wednesday 29 October at 21:04", "Wednesday 2026-10-29 at 21:04",  # a weekday that is not that date's
             "29 October 2025 at 21:04", "2026-10-28T21:04:10-04:00", "2026-10-30 21:04",
             "29 October or 30 October at 21:04", "Thursday 29 October at 21:04, or Friday",
             "29 October at 21:04, 30 October at 21:04",
             "Thursday 29 October and Tuesday 27 October at 21:04", "2026-10-29 21:04 and 2026-10-30 21:04"]
    assert [a for a in wrong if grade.grade(one, a)] == []
    assert grade.grade(one, None) is False


def test_code_time_with_a_date_set_wants_the_code_the_time_and_the_day():
    one = spec("code_time", truth="E-4417", time="21:04:20", ignore=["18:00", "24:00"])
    right = ["E-4417 at 21:04 on Thursday 29 October", "e4417, Oct 29, 21:04", "ERR 4417 at 21:04, 2026-10-29",
             "Thursday 29 October 2026: E-4417 at 21:04 (18:00 to 24:00)", "2026-10-29T21:04:20-04:00 E-4417"]
    wrong = ["E-4417 at 21:04", "E-4417 on 29 October", "E-4417 at 21:04 on 28 October", "E-4417 at 21:04 on Tuesday",
             "E-4471 at 21:04 on 29 October", "E-4417 at 21:04 on 29 October, E-5521 at 21:10"]
    assert [a for a in right if not grade.grade(one, a)] == []
    assert [a for a in wrong if grade.grade(one, a)] == []


def test_a_time_with_no_date_in_its_spec_grades_as_it_did():
    answers = {"21:04": True, "9:04 pm": True, "Tuesday 27 October at 21:04": True, "2026-10-28T21:04:10-04:00": True,
               "21:04 and 20:00": False, "22:30": False, "unknown": False}
    for kind, truth in (("time", None), ("code_time", "E-4417")):
        for dated in ({"date": None}, {"date": ""}, {}):  # a date of None, an empty one, and no key at all
            one = {**spec(kind, truth=truth), **dated}
            if not dated:
                del one["date"]
            for answer, expected in answers.items():
                text = f"E-4417, {answer}" if truth else answer
                assert grade.grade(one, text) is expected, (kind, dated, answer)
    for kind in ("number", "count", "absent"):  # a date means nothing to the other kinds
        one = {"kind": kind, "truth": "3", "time": None, "tol_min": None, "wrong": [], "ignore": [], "date": DAY}
        assert grade.grade(one, "unknown" if kind == "absent" else "3") is True


def graded(one: dict, answers: list[str], want: bool) -> list[str]:
    """The answers whose grade by `one` is not `want`."""
    return [a for a in answers if grade.grade(one, a) is not want]


def test_a_day_of_the_month_by_its_ordinal_is_a_date_when_it_is_one():
    one = spec()
    right = ["On the 29th at 21:04", "21:04 on the 29th", "On 29th at 21:04", "At 21:04, on the 29th.",
             "Thursday the 29th, 21:04", "On 29th October at 21:04", "On the 29th (21:04)",
             "The 2nd alarm was at 21:04 on Thursday 29 October"]  # "the 2nd alarm" is no date, so no other date
    wrong = ["On the 28th at 21:04", "On the 30th, 21:04", "21:04 on the 29th and the 30th", "On the 22nd at 21:04",
             "Thursday the 22nd, 21:04", "The 2nd alarm was at 21:04", "21:04 on the 29th ring"]  # no date, or not that one
    assert graded(one, right, True) == [] and graded(one, wrong, False) == []
    assert graded(spec("code_time", truth="E-4417", time="21:04:20"), ["E-4417 at 21:04 on the 29th"], True) == []
    # a date is no number: "the 29th" is not a second count
    assert grade.numbers("3 times on the 29th") == ["3"] and grade.numbers("the 2nd alarm") == ["2"]
    count = spec("count", truth="3", time=None, ignore=["731"])
    assert graded(count, ["ERR-731 three times on the 29th", "3 times, on the 29th"], True) == []
    assert graded(count, ["3 times on the 29th and 5 more", "3 times on the 29th, 5 times on the 30th"], False) == []


def test_a_year_in_brackets_is_no_number_unless_it_is_the_answer():
    code = spec("number", truth="4417", time=None)
    assert graded(code, ["4417 (2026)", "The code is 4417 (in 2026)", "4417, noted Thursday 29 October (2026)",
                         "4417 on 29 October 2026", "4417 ( 2026 )"], True) == []
    assert graded(code, ["4417 or 2026", "4417, 2026", "4417 (1234)", "4417 (2026 or 4471)", "2026 (4417)", "4417 (2026) 4471"],
                  False) == []
    # a code can look like a year: it is the answer, in brackets or not, and then a second number is a hedge as ever
    year = spec("number", truth="2026", time=None)
    assert graded(year, ["2026", "The code is (2026)", "2026 (29 October 2026)", "2026 (in 2026)"], True) == []
    assert graded(year, ["2026 or 4417", "4417 (2026)", "(4417)"], False) == []
    assert graded({**code, "wrong": ["2026"]}, ["4417 (2026)"], False) == [], "a wrong answer in brackets is still wrong"
    assert graded({**code, "wrong": ["2026"]}, ["4417 (2027)"], True) == []
    # the other kinds read years the same way
    count = spec("count", truth="3", time=None, ignore=["731"])
    assert graded(count, ["ERR-731 three times (2026)", "3 times (in 2026) on 29 October 2026"], True) == []
    assert graded(count, ["3 times (2026 or 2027)", "3 times (2026), 2026"], False) == []
    absent = spec("absent", truth=None, time=None, ignore=["845"])
    assert graded(absent, ["No doorbell rang (2026)", "No ring (in 2026)"], True) == []
    assert graded(absent, ["No, but the code is 2026", "No ring, code (4417)"], False) == []
    # a bare year in a code_time answer is the year of its date, and a second code is still a second code
    coded = spec("code_time", truth="E-4417", time="21:04:20")
    assert graded(coded, ["E-4417 at 21:04 on Thursday 29 October (2026)", "E-4417 at 21:04, 2026-10-29, 2026"], True) == []
    assert graded(coded, ["E-4417 at 21:04 on 29 October, E-2027", "E-4417 or E-4471 at 21:04 on 29 October"], False) == []
    yearly = spec("code_time", truth="E-2026", time="21:04:20")
    assert graded(yearly, ["E-2026 at 21:04 on 29 October", "E-2026 at 21:04, 29 October 2026"], True) == []
    assert graded(yearly, ["E-2026 or E-4417 at 21:04 on 29 October"], False) == []
    assert grade.numbers("Tuesday 28 October 2026 at 14:03, ERR-731 three times, 2026-10-28 (2026)") == ["731", "3"]


def test_an_absent_answer_may_give_the_range_it_searched_and_no_other_time():
    absent = spec("absent", truth=None, time=None, ignore=["12:00", "18:00"])
    clean = ["no, nothing (I checked 9:00-17:00)", "I searched 09:00 to 17:00 and found nothing",
             "Nothing found (searched 9:00–17:00)", "None. I looked between 8:00 and 20:00 and saw nothing",
             "I checked from 2:00 pm to 6:00 pm: nothing", "Nothing (I scanned 18:00 until 24:00)",
             "No doorbell rang between 12:00 and 18:00 (I checked 9:00-17:00)", "I checked 9:00-17:00 and found nothing, no ring",
             "None found from 9:00 to 17:00", "No parcel was left at the door (I checked 14:00-15:00)",
             "It did not ring at 13:00-14:00", "Nothing; I checked 9:00 to 5:00 pm", "I checked 9:30-10:30 and it never appeared"]
    assert graded(absent, clean, True) == []
    sighting = ["nothing, but there was one at 14:05",  # a time that is no range
                "No. I checked 9:00-17:00 but it rang at 14:05", "no, nothing (I checked 9:00-17:00) except 14:05",
                "I checked 9:00-17:00 and heard it once at 14:05, no ring after",  # the range is searched, the time is not
                "No, I checked 9:00-17:00, 14:05", "no ring; heard 14:05",
                "nothing, but I heard one between 14:05 and 14:10",  # a range that is short, in an answer of contrast
                "no, nothing: I checked 14:00-14:30", "No. I checked 14:10-14:05",  # under an hour, or backwards
                "No, I checked 12:75-17:00", "nothing found, I checked at 2026-10-29T21:04:10-04:00"]  # no range at all
    assert graded(absent, sighting, False) == []


def test_a_short_range_is_not_read_as_twelve_hours_more():
    # with no am or pm a time is two times of day, and two of them are read in the same half: a minute is a minute
    short = ["No; I checked 10:05-10:06.", "Could not confirm it; I listened 7:27-7:28 and it sounded like the alarm.",
             "Not certain; looked at 9:58-10:01, the dialog with the code showed then.", "No, I checked 10:05 to 10:06",
             "Nothing; I looked between 3:10 and 3:40", "No; I checked 11:30-12:20", "Nothing, I checked 10:00-10:00"]
    absent = spec("absent", truth=None, time=None, ignore=["12:00", "18:00"])
    assert graded(absent, short, False) == []
    assert [grade.span(*(grade.mentions(t)[0] for t in pair)) for pair in
            (("10:05", "10:06"), ("7:27", "7:28"), ("9:58", "10:01"), ("9:00", "17:00"), ("8:00", "20:00"), ("2:00", "6:00"),
             ("9:00", "5:00 pm"), ("11:30", "12:30"), ("10:05", "10:05"), ("9:00", "5:00"), ("14:10", "14:05"))] \
        == [1, 1, 3, 480, 720, 240, 480, 60, None, None, None]
    # an hour or more is a range, whatever the hour
    long = ["Nothing; I checked 7:00-9:30", "No; I checked 10:00-11:00", "No, I looked 22:00 to 23:30", "None found 1:00 to 3:00"]
    assert graded(absent, long, True) == []


def test_an_absent_answer_that_says_the_thing_was_there_keeps_its_times_whatever_it_negates():
    absent = spec("absent", truth=None, time=None, ignore=["12:00", "18:00"])
    claims = ["Yes, a parcel was left; I couldn't read the label. Checked 14:00-15:00.",
              "I could not pin the exact minute; scanning 14:00-16:00 shows the parcel was delivered.",
              "The error dialog never closed; I checked 13:00-15:00 and it was on screen the whole time.",
              "The alarm never stopped, I checked 14:00-16:00", "No, the alarm sounded; I checked 14:00-16:00",
              "No. It rang. I checked 13:00-15:00.", "A parcel was left at the door. Not sure when (I checked 14:00-15:00)",
              "No parcel? It rang between 14:00 and 16:00", "Not sure, but I checked 14:00-16:00 and it was there",
              "I did not see the label; the parcel was still there at 14:00-16:00",
              "Nothing in the log but yes, a ring between 14:00 and 16:00"]
    assert graded(absent, claims, False) == []
    # the same ranges in answers that negate throughout
    assert graded(absent, ["No parcel was left (I checked 14:00-15:00)", "No alarm sounded between 14:00 and 16:00",
                           "Not found; I checked 13:00-15:00 and nothing was on screen"], True) == []
    # `searched` leaves an answer it does not trust as it was, and blanks a range only where it trusts it
    assert grade.searched("no, nothing (i checked 9:00-17:00)").strip() == "no, nothing (i checked  )"
    assert grade.searched("no, nothing (i checked 9:00-17:00) but 14:05") == "no, nothing (i checked 9:00-17:00) but 14:05"
    assert grade.searched("it rang; i checked 9:00-17:00") == "it rang; i checked 9:00-17:00"


def test_a_spec_date_that_is_not_iso_is_an_error():
    with pytest.raises(ValueError):
        grade.grade(spec(when="29 October"), "Thursday 29 October at 21:04")


# -- the grader's choice (EgoLife's multiple-choice questions) -----------------------------------------------------------

def choice_spec(truth: str = "B") -> dict:
    return {"kind": "choice", "truth": truth, "time": None, "tol_min": None, "wrong": [], "ignore": [], "date": None}


def test_a_choice_is_right_when_the_letter_the_answer_starts_with_is_the_truth():
    one = choice_spec()
    right = ["B", "b", "(B)", "B) Alice", "B.", "B - Alice", "  B  ", "(b) Alice", "B: Alice", "B, Alice", "b)", "B\nbecause"]
    wrong = ["Alice", "B or C", "The answer is B", "unknown", "A", "C) Tasha", "Bob", "E", "", "ANSWER: B"]
    assert graded(one, right, True) == [] and graded(one, wrong, False) == []
    assert grade.grade(one, None) is False and "choice" in grade.KINDS
    assert [grade.choice(a) for a in ("a", "(D)", "d.", "C - Lucia", "(c) the cup")] == ["A", "D", "D", "C", "C"]
    assert grade.choice("a screwdriver") is None and grade.choice("b the cup") is None, "a lower-case word is no option"
    # one pair of brackets, quotes or backticks around the whole answer is taken off; a lower-case word inside is still no option
    wrapped = ["<B>", "[B]", '"B"', "'b'", "`B`", '"B) Alice"', "<B - Alice>", "[ b ]"]
    assert [grade.choice(a) for a in wrapped] == ["B"] * len(wrapped) and graded(choice_spec(), wrapped, True) == []
    assert [grade.choice(a) for a in ("'a screwdriver'", "<B or C>", '"Alice"', "<B", "B>", "<>", '""', "<<B>>")] == [None] * 8


def test_an_answer_that_names_a_second_letter_is_a_hedge_and_is_wrong_whichever_one_is_right():
    hedges = ["B or C", "B OR C", "B/C", "B, C", "B and C", "B & C", "B) Alice or C) Lucia", "(B) or (C)"]
    for truth in "BC":
        assert graded(choice_spec(truth), hedges, False) == [], truth
    assert [grade.choice(a) for a in hedges] == [None] * len(hedges)
    # a word that starts with a letter, or an option's own words, are no second letter
    for answer in ("B) Alice or Dan's cup", "B - Charlie, Dana", "B) a cup or a plate", "B, because Alice used it"):
        assert grade.choice(answer) == "B", answer


def test_an_unknown_choice_is_wrong_and_also_abstains_and_a_choice_has_no_other_rule():
    one = choice_spec("D")
    assert grade.grade(one, "unknown") is False and grade.abstained("unknown", one["ignore"]) is True
    assert grade.abstained("B", one["ignore"]) is False and grade.abstained(None, one["ignore"]) is False
    assert grade.grade(one, "D") is True and grade.grade({**one, "truth": "A"}, "D") is False
    with pytest.raises(ValueError, match="one of number, count.*choice"):
        grade.grade({**one, "kind": "pick"}, "D")
