"""Pagina „Analiză”: comutatorul pentru stratul „călcâi stabil” (oprit implicit), cu modelul RTMPose m implicit și
l opțional; filmul desenează călcâiul din modelul folosit la măsurare."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ecvestru import masurare as M
from ecvestru import stocare as ST
from ecvestru import suprapunere as S
from ecvestru.rezultat import HEEL_MODEL_L, HEEL_MODEL_M, LayerVersions

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "pages" / "analiza.py")
needs_clips = pytest.mark.skipif(not M.dev_videos(), reason="clipurile nu sunt locale")
STABLE, MODEL = "an-heel-stable", "an-heel-model"


def fake(path):
    return {i: {"points": {"shoulder": (110.0, 100.0), "hip": (100.0, 200.0), "heel": (110.0, 300.0),
                           "facing": 1.0}, "motiv": None} for i in range(2000)}


@pytest.fixture
def calls(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    monkeypatch.setattr(ST, "run_candidate", fake)
    seen = []
    original = M.analyze

    def spy(*args, **kwargs):
        seen.append((kwargs.get("heel_stable", False), kwargs.get("heel_model", HEEL_MODEL_M)))
        kwargs.pop("heel_stable", None)
        kwargs.pop("heel_model", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(M, "analyze", spy)
    return seen


def start() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def analyse(at: AppTest) -> AppTest:
    at.button(key="an-run").click().run()
    assert not at.exception, at.exception
    return at


@needs_clips
def test_the_stable_heel_switch_is_off_by_default_and_the_analysis_does_not_ask_for_it(calls):
    at = start()
    assert at.checkbox(key=STABLE).value is False
    analyse(at)
    assert calls == [(False, HEEL_MODEL_M)]


@needs_clips
def test_with_the_switch_on_the_layer_runs_with_model_m_by_default(calls):
    at = start()
    at.checkbox(key=STABLE).check().run()
    assert at.radio(key=MODEL).value == M.HEEL_MODEL_LABELS[HEEL_MODEL_M]
    analyse(at)
    assert calls == [(True, HEEL_MODEL_M)]


@needs_clips
def test_model_l_can_be_chosen_only_with_the_layer_on(calls):
    at = start()
    at.checkbox(key=STABLE).check().run()
    at.radio(key=MODEL).set_value(M.HEEL_MODEL_LABELS[HEEL_MODEL_L]).run()
    analyse(at)
    assert calls == [(True, HEEL_MODEL_L)]


# ---------- filmul: călcâiul din modelul folosit la măsurare ----------

def test_the_model_of_a_result_comes_from_its_layers():
    from ecvestru import rtmpose_l
    assert M.heel_model_of(LayerVersions(rtmpose_model=rtmpose_l.MODEL_ZIP_SHA256[:8], heel_stable=1)) == HEEL_MODEL_L
    assert M.heel_model_of(LayerVersions(rtmpose_model="abcdef12")) == HEEL_MODEL_M
    assert M.heel_model_of(None) == HEEL_MODEL_M


def test_the_film_draws_the_heel_of_the_model_used_in_the_measurement(monkeypatch):
    from ecvestru import rtmpose, rtmpose_l
    asked = []
    monkeypatch.setattr(M, "rtmpose_points", lambda *a, heel_model=HEEL_MODEL_M, **k: asked.append(heel_model)
                        or np.zeros((3, 26, 3)))
    monkeypatch.setattr(ST, "load_boxes", lambda *a: {})
    monkeypatch.setattr(rtmpose, "horse_boxes", lambda skeleton, boxes: {})
    monkeypatch.setattr(rtmpose, "side_heels", lambda points, skeleton: {})
    monkeypatch.setattr(rtmpose, "film_raw", lambda raw, frames, heels, boxes: raw)
    monkeypatch.setattr(rtmpose, "heel_sources", lambda raw, frames, heels, boxes: {})

    class Result:
        video, video_hash, frames = "v.mp4", "h", []
        layers = LayerVersions(rtmpose_model=rtmpose_l.MODEL_ZIP_SHA256[:8], heel_stable=1)

    S._rtmpose_raw(Path("v.mp4"), Result(), {}, object())
    Result.layers = LayerVersions(rtmpose_model=rtmpose.MODEL_ZIP_SHA256[:8])
    S._rtmpose_raw(Path("v.mp4"), Result(), {}, object())
    assert asked == [HEEL_MODEL_L, HEEL_MODEL_M]
