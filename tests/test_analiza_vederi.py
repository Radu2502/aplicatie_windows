"""Modulul față/spate lipit de profil, cu vederea din router sau din intervale, într-un singur JSON (1.5) și un
singur raport. Testul de identitate: cu intervalele vechi (tot intervalul din profil), partea de profil e exact
analiza de acum. Predictorul e fals (fără MediaPipe); scheletul e sintetic; nicio etichetă reală."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ecvestru import analiza_vederi as AV
from ecvestru import intervale as IV
from ecvestru import masurare as M
from ecvestru import raport as RP
from ecvestru.rezultat import SCHEMA_VERSION, CombinedResult

SHORT = "clip_152429.mp4"
have_short = (M.V.VIDEOS / SHORT).exists()
needs_clip = pytest.mark.skipif(not have_short, reason="clipul dev lipsește")
N = 500


@pytest.fixture
def tmp_results(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    return tmp_path / "analiza"


def pts(sh=(100.0, 100.0), hip=(100.0, 200.0), heel=(90.0, 300.0), facing=1.0):
    return {"shoulder": sh, "hip": hip, "heel": heel, "facing": facing}


def fake(path):
    return {i: ({"points": pts(sh=(100.0 + i % 7, 100.0)), "motiv": None} if i % 4 else
                {"points": None, "motiv": "nu_e_pe_cal"}) for i in range(N)}


def skeleton(n=N, shoulder_dy=10.0, hip_dy=0.0, visible=1.0):
    """Umerii la 100 px distanță pe x, dreapta imaginii mai jos cu `shoulder_dy` px; la fel șoldurile."""
    xy = np.zeros((n, 33, 2))
    xy[:, AV.LEFT_SHOULDER] = (150, 100 + shoulder_dy)    # din față: stânga călărețului în dreapta imaginii
    xy[:, AV.RIGHT_SHOULDER] = (50, 100)
    xy[:, AV.LEFT_HIP] = (140, 200 + hip_dy)
    xy[:, AV.RIGHT_HIP] = (60, 200)
    return SimpleNamespace(frame=np.arange(n), xy_px=xy, visibility=np.full((n, 33), visible))


# ---------- vederea pe cadre ----------

def test_the_views_from_the_old_profile_interval_are_profile_on_it_and_unlabeled_elsewhere():
    views = AV.views_from_segment((2, 4), n_frames=7)
    assert list(views) == ["", "", IV.PROFILE, IV.PROFILE, IV.PROFILE, "", ""]


def test_the_views_from_intervals_follow_the_annotator_intervals():
    intervals = [IV.Interval(video="v.mp4", start=0, stop=1, view=IV.FRONT, id="a", note="", annotator="A")]
    assert list(AV.views_from_intervals(intervals, 3)) == [IV.FRONT, IV.FRONT, ""]


def test_view_spans_group_consecutive_frames_of_the_same_view_inside_the_segment():
    views = np.array([IV.PROFILE] * 3 + [IV.FRONT] * 2 + [""] + [IV.BACK], dtype=object)
    spans = AV.view_spans(views, (1, 6))
    assert [(s.start, s.end, s.view) for s in spans] == [(1, 2, IV.PROFILE), (3, 4, IV.FRONT), (5, 5, AV.NO_VIEW),
                                                          (6, 6, IV.BACK)]


# ---------- înclinările din față / spate ----------

def test_the_tilt_is_the_line_angle_modulo_180_so_a_left_right_swap_does_not_change_it():
    assert AV.tilt_deg((0, 0), (100, 10)) == pytest.approx(AV.tilt_deg((100, 10), (0, 0)))
    assert AV.tilt_deg((0, 0), (100, 10)) == pytest.approx(np.degrees(np.arctan(0.1)))
    assert -90 < AV.tilt_deg((0, 0), (0, 10)) <= 90


def test_front_back_frames_measure_shoulders_and_pelvis_only_on_front_or_back_frames():
    views = np.array([IV.FRONT, IV.PROFILE, IV.BACK, IV.OBLIQUE], dtype=object)
    frames = AV.front_back_frames(skeleton(4, shoulder_dy=10, hip_dy=-5), views, (0, 3))
    assert [f.frame for f in frames] == [0, 2] and [f.view for f in frames] == [IV.FRONT, IV.BACK]
    assert frames[0].shoulder_tilt_deg == pytest.approx(np.degrees(np.arctan(10 / 100)))   # dreapta imaginii mai jos
    assert frames[0].pelvis_tilt_deg == pytest.approx(-np.degrees(np.arctan(5 / 80)))


def test_points_below_the_visibility_threshold_are_not_measured():
    views = np.array([IV.FRONT], dtype=object)
    frames = AV.front_back_frames(skeleton(1, visible=AV.MIN_VISIBILITY / 2), views, (0, 0))
    assert frames[0].shoulder_tilt_deg is None and frames[0].pelvis_tilt_deg is None


def test_without_a_skeleton_the_front_back_frames_are_kept_unmeasured():
    views = np.array([IV.BACK, IV.BACK], dtype=object)
    frames = AV.front_back_frames(None, views, (0, 1))
    assert len(frames) == 2 and all(f.shoulder_tilt_deg is None for f in frames)


# ---------- profilul restrâns la cadrele din profil ----------

@needs_clip
def test_identity_with_the_old_interval_the_profile_part_is_exactly_the_current_analysis(tmp_results, monkeypatch):
    monkeypatch.setattr(AV, "profile_skeleton", lambda *a: skeleton())
    current = M.analyze(M.V.VIDEOS / SHORT, "trap", (100, 139), predictor=fake)
    views = AV.views_from_segment((100, 139), N)
    combined = AV.analyze_views(M.V.VIDEOS / SHORT, "trap", views, AV.SOURCE_INTERVALS, (100, 139), predictor=fake)
    assert combined.profile == current
    assert combined.profile.to_json() == current.to_json()
    assert combined.front_back is None


@needs_clip
def test_the_default_path_is_unchanged_by_the_module(tmp_results):
    before = M.analyze(M.V.VIDEOS / SHORT, "trap", (100, 139), predictor=fake)
    assert before.schema_version == SCHEMA_VERSION == "1.4"
    assert AV.VIEWS_LAYER_DEFAULT is False


@needs_clip
def test_frames_of_another_view_leave_the_profile_measurement_with_their_reason(tmp_results, monkeypatch):
    monkeypatch.setattr(AV, "profile_skeleton", lambda *a: skeleton())
    views = AV.views_from_segment((100, 139), N)
    views[120:130] = IV.FRONT
    combined = AV.analyze_views(M.V.VIDEOS / SHORT, "trap", views, AV.SOURCE_ROUTER, (100, 139), predictor=fake)
    p = combined.profile
    moved = [f for f in p.frames if 120 <= f.frame < 130]
    assert all(f.trunk_deg is None and f.leg_deg is None and f.reason == AV.REASON_VIEW for f in moved)
    assert p.trunk.n_frames == 30 - sum(1 for i in range(120, 130) if i % 4)
    assert any(u.reason == AV.REASON_VIEW for u in p.unmeasured)
    assert combined.front_back.frames_front == 10 and combined.front_back.frames_back == 0
    assert combined.front_back.shoulders.n_frames == 10
    assert combined.view_source == AV.SOURCE_ROUTER


@needs_clip
def test_the_combined_json_is_version_1_5_and_round_trips(tmp_results, monkeypatch):
    monkeypatch.setattr(AV, "profile_skeleton", lambda *a: skeleton())
    views = AV.views_from_segment((100, 139), N)
    views[130:140] = IV.BACK
    combined = AV.analyze_views(M.V.VIDEOS / SHORT, "trap", views, AV.SOURCE_INTERVALS, (100, 139), predictor=fake)
    data = json.loads(combined.to_json())
    assert data["schema_version"] == "1.5" and data["profile"]["schema_version"] == "1.4"
    assert CombinedResult.from_json(combined.to_json()) == combined
    assert data["frames_by_view"][IV.BACK] == 10


def test_an_unknown_view_source_is_refused():
    with pytest.raises(ValueError, match="Sursă"):
        AV.analyze_views(M.V.VIDEOS / SHORT, "trap", np.array([], dtype=object), "ghicit")


# ---------- raportul unic ----------

@needs_clip
def test_the_single_report_has_the_profile_text_then_the_views_and_front_back_lines(tmp_results, monkeypatch):
    monkeypatch.setattr(AV, "profile_skeleton", lambda *a: skeleton(shoulder_dy=10))
    views = AV.views_from_segment((100, 139), N)
    views[120:140] = IV.FRONT
    combined = AV.analyze_views(M.V.VIDEOS / SHORT, "trap", views, AV.SOURCE_ROUTER, (100, 139), predictor=fake)
    text = AV.report_text(combined)
    assert text.startswith(RP.report_text(combined.profile).splitlines()[0])
    assert "router" in text and "Umerii" in text and "Bazinul" in text
    assert AV.FRONT_BACK_LIMITATION in text
    assert AV.report_passes_checks(text, combined), (RP.numbers_in(text) - AV.allowed_numbers(combined), text)


@needs_clip
def test_the_report_number_check_refuses_a_number_not_in_the_json(tmp_results, monkeypatch):
    monkeypatch.setattr(AV, "profile_skeleton", lambda *a: skeleton())
    views = AV.views_from_segment((100, 139), N)
    combined = AV.analyze_views(M.V.VIDEOS / SHORT, "trap", views, AV.SOURCE_ROUTER, (100, 139), predictor=fake)
    assert not AV.report_passes_checks(AV.report_text(combined) + " Umerii la 37°.", combined)


# ---------- calitatea recalculată în profilul restrâns ----------

def _result_with_quality(frames):
    from ecvestru.rezultat import AnalysisResult, FrameMeasure
    measured = [FrameMeasure(**f) for f in frames]
    return AnalysisResult.model_construct(frames=measured, quality=M._quality_summary(measured))


def test_the_quality_summary_is_recalculated_on_the_frames_left_in_profile():
    from ecvestru import calitate
    warn = calitate.WARNING_SIGNALS[0]
    result = _result_with_quality([
        {"frame": 0, "trunk_deg": 1.0, "leg_deg": 2.0, "warnings": [warn]},
        {"frame": 1, "trunk_deg": 1.0, "leg_deg": None, "reason": M.REASON_HEEL_ON_HORSE},
        {"frame": 2, "trunk_deg": 1.0, "leg_deg": 2.0, "warnings": [warn]},
    ])
    assert result.quality.excluded_frames == 1 and result.quality.warnings[0].frames == 2
    views = np.array([IV.PROFILE, IV.FRONT, IV.FRONT], dtype=object)
    restricted = AV.restrict_profile(result, views)
    assert restricted.quality.excluded_frames == 0
    assert [w.frames for w in restricted.quality.warnings] == [1]


def test_without_a_quality_layer_the_restricted_profile_keeps_none():
    from ecvestru.rezultat import AnalysisResult, FrameMeasure
    result = AnalysisResult.model_construct(frames=[FrameMeasure(frame=0, trunk_deg=1.0)], quality=None)
    restricted = AV.restrict_profile(result, np.array([IV.FRONT], dtype=object))
    assert restricted.quality is None


def test_with_every_frame_in_profile_the_quality_is_the_same_object():
    result = _result_with_quality([{"frame": 0, "trunk_deg": 1.0, "leg_deg": 2.0}])
    assert AV.restrict_profile(result, np.array([IV.PROFILE], dtype=object)) is result
