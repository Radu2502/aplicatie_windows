"""Urmărirea călărețului țintă, doar din geometrie și mișcare, pe piste sintetice cu două casete de cal.
Rezultatele se scriu în foldere temporare."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ecvestru import masurare as M
from ecvestru import raport as R
from ecvestru import stocare as ST
from ecvestru import suprapunere as S
from ecvestru import tinta as T
from ecvestru.rezultat import AnalysisResult, FrameMeasure, TargetSeed

FPS = 30.0
GAP = T.max_gap_frames(FPS)


def track(*args):
    return T.track_target(*args, fps=FPS)


def box(cx, cy=200.0, w=100.0, h=80.0):
    return (cx - w / 2, cy - h / 2, w, h)


def two_riders(n=60, first=lambda f: 100.0 + 4 * f, second=lambda f: 500.0 + 4 * f):
    return {f: [box(first(f)), box(second(f))] for f in range(n)}


def seed(frame, x, y=200.0):
    return TargetSeed(frame=frame, x=x, y=y)


def centres(states):
    return {f: s.box[0] + s.box[2] / 2 for f, s in states.items() if s.box is not None}


def test_seed_selects_the_box_nearest_to_the_click():
    assert T.seed_box([box(100), box(500)], seed(0, 480)) == box(500)


def test_seed_far_from_every_horse_is_refused():
    with pytest.raises(ValueError, match="cal"):
        T.seed_box([box(100)], seed(0, 800))


def test_target_is_followed_forward_and_backward_from_the_click():
    states = track(two_riders(), 60, seed(30, 620))
    assert all(s.status is None for s in states.values()) and len(states) == 60
    assert centres(states)[0] == pytest.approx(500) and centres(states)[59] == pytest.approx(736)


def test_two_targets_never_share_a_box_when_they_stay_apart():
    a, b = (centres(track(two_riders(), 60, seed(0, x))) for x in (100, 500))
    assert all(abs(a[f] - b[f]) > 100 for f in a)


def test_missing_detections_keep_the_target_and_mark_no_problem():
    boxes = two_riders()
    for f in range(20, 26):
        boxes[f] = [boxes[f][1]]
    states = track(boxes, 60, seed(0, 100))
    assert all(s.status is None for s in states.values())
    assert centres(states)[59] == pytest.approx(336)


def test_crossing_riders_become_unsure_then_continue_on_the_same_horse():
    boxes = two_riders(100, lambda f: 100.0 + 6 * f, lambda f: 694.0 - 6 * f)
    states = track(boxes, 100, seed(0, 100))
    assert any(s.status == T.REASON_UNSURE for s in states.values())
    assert centres(states)[99] == pytest.approx(694, abs=1)


def test_a_box_disappearing_for_too_long_loses_the_target_until_the_end():
    n = 2 * GAP + 20
    boxes = {f: ([box(100 + f)] if f < 10 else []) for f in range(n)}
    states = track(boxes, n, seed(0, 100))
    assert states[9].status is None and states[n - 1].status == T.REASON_LOST


def test_a_box_of_very_different_size_is_not_the_same_horse():
    n = 2 * GAP + 20
    boxes = {f: [box(100, w=100, h=80)] for f in range(5)} | {f: [box(105, w=300, h=240)] for f in range(5, n)}
    assert track(boxes, n, seed(0, 100))[n - 1].status == T.REASON_LOST


def raw_on(x, frames=60):
    return {f: {"points": {"shoulder": (x, 100.0), "hip": (x, 165.0), "heel": (x, 260.0), "facing": 1.0},
                "motiv": None} for f in range(frames)}


def test_filter_keeps_frames_on_the_target_and_names_the_others():
    states = track(two_riders(10, lambda f: 100.0, lambda f: 500.0), 10, seed(0, 100))
    assert T.filter_raw(raw_on(100.0, 10), states, 10)[3]["motiv"] is None
    other = T.filter_raw(raw_on(500.0, 10), states, 10)[3]
    assert other == {"points": None, "motiv": T.REASON_OTHER}


def test_filter_never_replaces_an_existing_reason():
    states = track(two_riders(10, lambda f: 100.0, lambda f: 500.0), 10, seed(0, 100))
    raw = {f: {"points": None, "motiv": "calaret_negasit"} for f in range(10)}
    assert T.filter_raw(raw, states, 10)[3]["motiv"] == "calaret_negasit"


def test_filter_marks_unsure_and_lost_frames_even_with_points():
    states = {0: T.TargetState(None, T.REASON_UNSURE), 1: T.TargetState(None, T.REASON_LOST)}
    out = T.filter_raw(raw_on(100.0, 2), states, 2)
    assert out[0] == {"points": None, "motiv": T.REASON_UNSURE} and out[1]["motiv"] == T.REASON_LOST


def test_every_new_reason_has_a_text_for_the_user():
    assert {T.REASON_UNSURE, T.REASON_LOST, T.REASON_OTHER} <= set(M.REASON_TEXTS)


def write_clip(path: Path, frames: int) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    rng = np.random.default_rng(int(frames))
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def clip(tmp_path, monkeypatch):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    monkeypatch.setattr(ST, "detect_horses", lambda path: two_riders(20, lambda f: 100.0, lambda f: 500.0))
    monkeypatch.setattr(ST, "run_ratios", lambda path: {})
    write_clip(tmp_path / "doi.mp4", 20)
    return tmp_path / "doi.mp4"


def alternating(path):
    return {f: raw_on(100.0 if f % 2 else 500.0, 20)[f] for f in range(20)}


def test_each_target_gets_its_own_measures_and_cache_file(clip, tmp_path):
    first = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(0, 100))
    second = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(0, 500))
    assert first.target != second.target and first.trunk.coverage_pct == 50 and second.trunk.coverage_pct == 50
    assert {f.frame for f in first.frames if f.trunk_deg is not None}.isdisjoint(
        {f.frame for f in second.frames if f.trunk_deg is not None})
    assert len(list((tmp_path / "rezultate" / "analiza").glob("*.json"))) == 2
    assert len(list((tmp_path / "rezultate" / "analiza" / "brut").glob("*.json"))) == 1


def test_analysis_without_target_keeps_the_same_file_and_json(clip):
    plain = M.analyze(clip, "pas", (0, 19), predictor=alternating)
    assert plain.target is None and "target" not in json.loads(plain.to_json())
    assert M.result_cache_path("a.mp4", "h", "c@1", (0, 9), "pas").name == "a-h__c_1__0-9__pas__q2__v1.4.json"


def test_target_survives_the_json_round_trip(clip):
    result = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(0, 100))
    assert AnalysisResult.from_json(result.to_json()).target == seed(0, 100)


def test_report_names_the_click_and_passes_the_number_check(clip):
    result = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(4, 100))
    text = R.report_text(result)
    assert "alegere manuală, cadrul 4" in text and R.report_passes_checks(text, result)
    assert M.REASON_TEXTS[T.REASON_OTHER] in text


def test_same_content_under_another_name_gives_the_same_measures(clip, tmp_path):
    other = tmp_path / "alt_nume.mp4"
    other.write_bytes(clip.read_bytes())
    a = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(0, 100))
    b = M.analyze(other, "pas", (0, 19), predictor=alternating, target=seed(0, 100))
    assert a.frames == b.frames and a.trunk == b.trunk


def test_film_of_a_target_has_a_frame_for_each_interval_frame(clip):
    result = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(0, 100))
    assert S.clip_film(clip, result)


def test_upload_with_a_target_keeps_the_video_off_the_disk(clip, tmp_path):
    result = M.analyze_upload(clip.read_bytes(), "incarcat.mp4", "pas", (0, 19), predictor=alternating,
                              target=seed(0, 100))
    assert result.target == seed(0, 100)
    assert not list((tmp_path / "rezultate").rglob("*.mp4"))


def test_horse_boxes_are_cached_by_content(clip, monkeypatch):
    M.load_boxes(clip, "doi.mp4", "abc")
    monkeypatch.setattr(ST, "detect_horses", lambda path: pytest.fail("trebuia să vină din cache"))
    assert M.load_boxes(clip, "doi.mp4", "abc")[3] == [list(b) for b in two_riders(20, lambda f: 100.0,
                                                                                    lambda f: 500.0)[3]]


def test_frame_missing_from_the_candidate_keeps_its_usual_reason():
    states = track(two_riders(10, lambda f: 100.0, lambda f: 500.0), 10, seed(0, 100))
    measures = M.frame_measures(T.filter_raw({}, states, 10), 10)
    assert measures[3].reason == "calaret_negasit"


def test_read_frame_refuses_a_frame_outside_the_clip(clip):
    assert M.read_frame(clip, 19).shape[:2] == (360, 640)
    with pytest.raises(ValueError, match="Cadrul"):
        M.read_frame(clip, 20)


def shift(dx):
    return SimpleNamespace(dx=dx, dy=0.0)


def panning(n=40, step=-120.0):
    """Calul stă pe loc în scenă, dar camera se rotește: în imagine, caseta sare cu `step` pe cadru."""
    boxes = {f: [box(3000.0 + step * f)] for f in range(n)}
    return boxes, {f: shift(step) for f in range(1, n)}


def test_fast_camera_pan_loses_the_target_without_compensation():
    n = GAP + 20
    boxes, _ = panning(n)
    assert track(boxes, n, seed(0, 3000))[n - 1].status == T.REASON_LOST


def test_camera_compensation_keeps_the_target_during_a_fast_pan():
    boxes, motion = panning()
    states = track(boxes, 40, seed(0, 3000), motion)
    assert all(s.status is None for s in states.values())
    assert centres(states)[39] == pytest.approx(3000.0 - 120.0 * 39)


def test_camera_compensation_also_works_backward():
    boxes, motion = panning()
    states = track(boxes, 40, seed(39, 3000.0 - 120.0 * 39), motion)
    assert centres(states)[0] == pytest.approx(3000.0)


def test_missing_motion_on_a_frame_is_not_a_failure():
    boxes, motion = panning(10, -20.0)
    del motion[5]
    assert all(s.status is None for s in track(boxes, 10, seed(0, 3000), motion).values())


GAP_END = 10 + 3 * GAP
GAP_FRAMES = GAP_END + 20


def gap_clip():
    """Caseta dispare mult timp, apoi reapare în alt loc."""
    return {f: ([box(100 + f)] if f < 10 else [] if f < GAP_END else [box(600 + f)]) for f in range(GAP_FRAMES)}


LATE_CLICK = seed(GAP_END + 5, 605 + GAP_END)


def test_a_second_click_recovers_the_target_after_it_was_lost():
    states = track(gap_clip(), GAP_FRAMES, [seed(0, 100), LATE_CLICK])
    assert states[5].status is None and states[GAP_FRAMES - 1].status is None
    assert states[10 + GAP + GAP // 2].status == T.REASON_LOST
    assert centres(states)[GAP_END] == pytest.approx(600 + GAP_END)


def test_clicks_can_be_given_in_any_order():
    assert (track(gap_clip(), GAP_FRAMES, [LATE_CLICK, seed(0, 100)])
            == track(gap_clip(), GAP_FRAMES, [seed(0, 100), LATE_CLICK]))


def test_lost_frames_are_listed_as_intervals():
    states = track(gap_clip(), GAP_FRAMES, seed(0, 100))
    assert T.lost_intervals(states, (0, GAP_FRAMES - 1)) == [(states_first_lost(states), GAP_FRAMES - 1)]
    assert T.lost_intervals(states, (0, 5)) == []


def states_first_lost(states):
    return min(f for f, s in states.items() if s.status == T.REASON_LOST)


def test_lost_frame_is_marked_even_without_an_answer():
    states = {0: T.TargetState(None, T.REASON_LOST)}
    gone = {0: {"points": None, "motiv": "calaret_negasit"}}
    assert T.filter_raw(gone, states, 1)[0]["motiv"] == T.REASON_LOST


def test_second_click_changes_the_cache_and_is_kept_in_the_result(clip):
    one = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(0, 100))
    two = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(0, 100), anchors=[seed(10, 100)])
    assert two.anchors == [seed(10, 100)] and "anchors" not in json.loads(one.to_json())
    paths = {M.result_cache_path("a", "h", "c", (0, 1), "pas", seed(0, 1), True, a) for a in ((), (seed(5, 1),))}
    assert len(paths) == 2


def test_frame_summary_counts_measured_frames_and_each_reason(clip):
    result = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(0, 100))
    measured, reasons = M.frame_summary(result)
    assert measured == 10 and dict(reasons) == {M.REASON_TEXTS[T.REASON_OTHER]: 10}


def test_frame_summary_counts_a_measured_trunk_without_leg_as_measured():
    frames = [FrameMeasure(frame=0, trunk_deg=1.0, leg_deg=None, reason=M.REASON_HEEL),
              FrameMeasure(frame=1, reason=T.REASON_LOST), FrameMeasure(frame=2, reason=T.REASON_UNSURE)]
    measured, reasons = M.frame_summary(SimpleNamespace(frames=frames))
    assert measured == 1
    assert dict(reasons) == {M.REASON_TEXTS[T.REASON_LOST]: 1, M.REASON_TEXTS[T.REASON_UNSURE]: 1}


def gap_of(seconds, fps):
    """O casetă care dispare `seconds` secunde, la `fps` dat; la capăt, caseta e în același loc."""
    hidden = round(seconds * fps)
    n = 10 + hidden + 10
    return {f: ([box(100.0)] if f < 10 or f >= 10 + hidden else []) for f in range(n)}, n


@pytest.mark.parametrize("fps", [24.0, 30.0, 60.0])
def test_a_gap_under_the_limit_in_seconds_keeps_the_target_at_any_fps(fps):
    boxes, n = gap_of(T.MAX_GAP_SECONDS * 0.9, fps)
    assert T.track_target(boxes, n, seed(0, 100), fps=fps)[n - 1].status is None


@pytest.mark.parametrize("fps", [24.0, 30.0, 60.0])
def test_a_gap_over_the_limit_in_seconds_loses_the_target_at_any_fps(fps):
    boxes, n = gap_of(T.MAX_GAP_SECONDS * 1.2, fps)
    assert T.track_target(boxes, n, seed(0, 100), fps=fps)[n - 1].status == T.REASON_LOST


def test_the_same_gap_in_frames_means_different_things_at_different_fps():
    hidden = round(T.MAX_GAP_SECONDS * 30.0 * 1.2)
    boxes = {f: ([box(100.0)] if f < 10 or f >= 10 + hidden else []) for f in range(hidden + 20)}
    assert T.track_target(boxes, hidden + 20, seed(0, 100), fps=30.0)[hidden + 19].status == T.REASON_LOST
    assert T.track_target(boxes, hidden + 20, seed(0, 100), fps=60.0)[hidden + 19].status is None
