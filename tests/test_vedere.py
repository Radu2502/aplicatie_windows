"""Motivul „nu e din profil”: raportul lățime umeri / trunchi, pe puncte sintetice, și legătura cu analiza.
Fără model; rezultatele stau în foldere temporare."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ecvestru import masurare as M
from ecvestru import stocare as ST
from ecvestru import vedere as W


def person(shoulder_half=0.05, torso=0.2):
    """Puncte normalizate; umerii sunt la `shoulder_half` de o parte și de alta a axei."""
    p = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]
    p[W.LEFT_SHOULDER] = SimpleNamespace(x=0.5 - shoulder_half, y=0.3, visibility=0.9)
    p[W.RIGHT_SHOULDER] = SimpleNamespace(x=0.5 + shoulder_half, y=0.3, visibility=0.9)
    p[W.LEFT_HIP] = SimpleNamespace(x=0.5, y=0.3 + torso, visibility=0.9)
    p[W.RIGHT_HIP] = SimpleNamespace(x=0.5, y=0.3 + torso, visibility=0.9)
    return p


def test_ratio_uses_pixels_not_normalised_values():
    assert W.shoulder_ratio(person(0.1, 0.25), 1000, 1000) == pytest.approx(0.8)
    assert W.shoulder_ratio(person(0.1, 0.25), 500, 1000) == pytest.approx(0.4)


def test_series_keeps_only_frames_with_a_rider():
    def source(path):
        yield 0, person(), 100, 100, None
        yield 1, None, 100, 100, "calaret_negasit"

    series = W.shoulder_ratios("a.mp4", source)
    assert sorted(series) == [0] and series[0] == pytest.approx(0.5)


def answer():
    return {"points": {"shoulder": (1.0, 1.0), "hip": (1.0, 2.0), "heel": (1.0, 3.0), "facing": 1.0}, "motiv": None}


@pytest.mark.parametrize("ratio,kept", [(W.PROFILE_MAX_RATIO - 0.01, True), (W.PROFILE_MAX_RATIO, False), (0.9, False)])
def test_threshold_excludes_oblique_frames(ratio, kept):
    out = W.apply_profile({0: answer()}, {0: ratio})
    assert (out[0]["motiv"] is None) is kept
    if not kept:
        assert out[0] == {"points": None, "motiv": W.REASON_NOT_PROFILE}


def test_frame_without_ratio_or_with_a_reason_is_left_alone():
    gone = {"points": None, "motiv": "calaret_negasit"}
    assert W.apply_profile({0: answer(), 1: gone}, {1: 0.9}) == {0: answer(), 1: gone}


def test_reason_has_a_text_for_the_user():
    assert W.REASON_NOT_PROFILE in M.REASON_TEXTS


def write_clip(path: Path, frames: int) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(frames):
        writer.write(np.full((240, 320, 3), 90, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def clip(tmp_path, monkeypatch):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    monkeypatch.setattr(ST, "run_ratios", lambda path: {f: 0.9 for f in range(0, 10, 2)})
    return write_clip(tmp_path / "c.mp4", 10)


def predictor(path):
    return {f: answer() for f in range(10)}


def test_extended_analysis_excludes_oblique_frames_with_the_reason(clip):
    result = M.analyze(clip, "pas", (0, 9), predictor=predictor, extended=True)
    assert result.trunk.coverage_pct == 50 and {u.reason for u in result.unmeasured} == {W.REASON_NOT_PROFILE}


def test_plain_analysis_ignores_the_view_and_keeps_its_file(clip, tmp_path):
    result = M.analyze(clip, "pas", (0, 9), predictor=predictor)
    assert result.trunk.coverage_pct == 100
    assert M.result_cache_path("a.mp4", "h", "c@1", (0, 9), "pas").name == "a-h__c_1__0-9__pas__q2__v1.4.json"
    assert not (tmp_path / "rezultate" / "analiza" / "vedere").exists()


def test_extended_and_plain_results_never_share_a_file(clip):
    plain = M.result_cache_path("a.mp4", "h", "c@1", (0, 9), "pas")
    assert M.result_cache_path("a.mp4", "h", "c@1", (0, 9), "pas", extended=True) != plain


def test_ratios_are_cached_and_only_as_numbers(clip, monkeypatch, tmp_path):
    M.analyze(clip, "pas", (0, 9), predictor=predictor, extended=True)
    saved = list((tmp_path / "rezultate" / "analiza" / "vedere").glob("*.json"))
    assert len(saved) == 1 and "shoulder" not in saved[0].read_text(encoding="utf-8")
    monkeypatch.setattr(ST, "run_ratios", lambda path: pytest.fail("trebuia să vină din cache"))
    M.load_ratios(clip, "c.mp4", M.file_hash(clip), M.candidate_label())
