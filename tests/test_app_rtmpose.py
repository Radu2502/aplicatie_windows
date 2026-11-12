"""Pagina „Analiză”: RTMPose e calea implicită; comutatorul „Doar MediaPipe (linia de bază)” e oprit implicit; orice
cădere pe linia de bază se vede în pagină; fraza despre licență."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ecvestru import masurare as M
from ecvestru import stocare as ST
from ecvestru.rezultat import HEEL_MEDIAPIPE, HEEL_RTMPOSE

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "pages" / "analiza.py")
pytestmark = pytest.mark.skipif(not M.dev_videos(), reason="clipurile nu sunt locale")
KEY = "an-baseline"


def fake(path):
    return {i: {"points": {"shoulder": (110.0, 100.0), "hip": (100.0, 200.0), "heel": (110.0, 300.0),
                           "facing": 1.0}, "motiv": None} for i in range(2000)}


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    monkeypatch.setattr(ST, "run_candidate", fake)
    calls = []
    original = M.analyze

    def spy(*args, **kwargs):
        calls.append(kwargs.get("heel_source", HEEL_MEDIAPIPE))
        return original(*args, **kwargs)

    monkeypatch.setattr(M, "analyze", spy)
    return calls


def start(monkeypatch) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def analyse(at: AppTest) -> AppTest:
    at.button(key="an-run").click().run()
    assert not at.exception, at.exception
    return at


def test_the_baseline_switch_is_off_by_default_and_named_plainly(setup, monkeypatch):
    box = start(monkeypatch).checkbox(key=KEY)
    assert box.value is False and box.label == "Doar MediaPipe (linia de bază)"


def test_by_default_the_analysis_asks_for_rtmpose(setup, monkeypatch):
    analyse(start(monkeypatch))
    assert setup == [HEEL_RTMPOSE]


def test_with_the_switch_on_the_analysis_asks_for_the_baseline(setup, monkeypatch):
    at = start(monkeypatch)
    at.checkbox(key=KEY).check().run()
    analyse(at)
    assert setup == [HEEL_MEDIAPIPE]
    assert not any("linia de bază" in w.value and "nu a putut" in w.value for w in at.warning)


def test_a_fallback_to_the_baseline_is_visible_with_its_reason(setup, monkeypatch):
    original = M.analyze
    monkeypatch.setattr(M, "analyze", lambda *a, **k: original(*a, **{**k, "heel_source": HEEL_MEDIAPIPE}).model_copy(
        update={"fallback_reason": "onnxruntime lipsește"}))
    at = analyse(start(monkeypatch))
    assert any("linia de bază" in w.value and "onnxruntime lipsește" in w.value for w in at.warning)


def test_when_rtmpose_ran_the_page_names_the_path(setup, monkeypatch):
    original = M.analyze
    monkeypatch.setattr(M, "analyze", lambda *a, **k: original(*a, **{**k, "heel_source": HEEL_MEDIAPIPE}).model_copy(
        update={"heel_source": HEEL_RTMPOSE, "pipeline": "rtmpose"}))
    at = analyse(start(monkeypatch))
    assert any("RTMPose" in c.value and "calea implicită" in c.value for c in at.caption)
    assert not any("nu a putut" in w.value for w in at.warning)


def test_the_page_says_the_model_is_used_for_non_commercial_research(setup, monkeypatch):
    at = start(monkeypatch)
    assert any("necomercial" in c.value and "Licențe" in c.value for c in at.caption)
