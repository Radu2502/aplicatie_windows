"""RTMPose devine calea implicită a aplicației (fuziune + interval + B4 + „profil neclar”, neschimbate). Schema 1.4
spune ce cale a rulat și versiunile straturilor. Fără onnxruntime, fără model sau fără modul (copia portabilă),
analiza cade pe linia de bază (MediaPipe) și spune de ce. Identitatea cu main (d32d036) pe cele 15 clipuri dev."""
import hashlib
import json
from pathlib import Path

import pytest
from test_rtmpose import FakeEstimator, setup

from ecvestru import masurare as M
from ecvestru import rtmpose as R
from ecvestru.rezultat import QUALITY_VERSION, SCHEMA_VERSION, AnalysisResult

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "date" / "identitate_rtmpose_implicit.json"
NEW_FIELDS = ("schema_version", "pipeline", "fallback_reason", "layers")


@pytest.fixture
def tmp_results(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    return tmp_path / "analiza"


def run(tmp_path, monkeypatch, source, estimator=None):
    predictor = setup(monkeypatch, tmp_path)
    return M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source=source,
                     heel_estimator=estimator)


# ---------- schema 1.4 ----------

def test_the_schema_is_1_4():
    assert SCHEMA_VERSION == "1.4"


def test_the_rtmpose_path_says_so_with_the_versions_of_its_layers(tmp_results, tmp_path, monkeypatch):
    res = run(tmp_path, monkeypatch, "rtmpose", FakeEstimator())
    data = json.loads(res.to_json())
    assert data["pipeline"] == "rtmpose" and data["fallback_reason"] is None and data["heel_source"] == "rtmpose"
    assert data["layers"] == {"quality": QUALITY_VERSION, "rtmpose": R.RTMPOSE_VERSION,
                              "orientation": R.ORIENTATION_VERSION, "rtmpose_model": R.MODEL_ZIP_SHA256[:8]}
    assert AnalysisResult.from_json(res.to_json()) == res


def test_the_baseline_says_so_and_has_no_rtmpose_layers(tmp_results, tmp_path, monkeypatch):
    data = json.loads(run(tmp_path, monkeypatch, "mediapipe").to_json())
    assert data["pipeline"] == "baseline" and data["fallback_reason"] is None
    assert data["layers"] == {"quality": QUALITY_VERSION, "rtmpose": None, "orientation": None, "rtmpose_model": None}


# ---------- căderea pe linia de bază ----------

def test_without_onnxruntime_the_analysis_runs_the_baseline_and_says_why(tmp_results, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "runtime_available", lambda: False)
    monkeypatch.setattr(R, "load_points", lambda *a, **k: pytest.fail("RTMPose a rulat fără onnxruntime"))
    res = run(tmp_path, monkeypatch, "rtmpose")
    assert res.pipeline == "baseline" and res.heel_source == "mediapipe" and res.orientation is None
    assert "onnxruntime" in res.fallback_reason


def test_the_model_is_downloaded_on_first_use(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(R, "runtime_available", lambda: True)
    monkeypatch.setattr(R, "model_path", lambda folder=None: None)
    monkeypatch.setattr(R, "download_model", lambda *a, **k: calls.append(1) or tmp_path / "end2end.onnx")
    assert M.rtmpose_problem() is None and calls == [1]


def test_a_failed_download_falls_back_to_the_baseline(tmp_results, tmp_path, monkeypatch):
    def fail(*a, **k):
        raise OSError("fără rețea")

    monkeypatch.setattr(R, "runtime_available", lambda: True)
    monkeypatch.setattr(R, "model_path", lambda folder=None: None)
    monkeypatch.setattr(R, "download_model", fail)
    res = run(tmp_path, monkeypatch, "rtmpose")
    assert res.pipeline == "baseline" and "descărca" in res.fallback_reason and "fără rețea" in res.fallback_reason


def test_a_fallback_is_not_remembered_as_the_rtmpose_result(tmp_results, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "runtime_available", lambda: False)
    assert run(tmp_path, monkeypatch, "rtmpose").pipeline == "baseline"
    monkeypatch.setattr(R, "runtime_available", lambda: True)
    again = run(tmp_path, monkeypatch, "rtmpose", FakeEstimator())
    assert again.pipeline == "rtmpose" and again.fallback_reason is None


def test_an_injected_estimator_needs_no_runtime_or_model(tmp_results, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "runtime_available", lambda: False)
    assert run(tmp_path, monkeypatch, "rtmpose", FakeEstimator()).pipeline == "rtmpose"


def test_the_app_default_is_rtmpose():
    assert M.APP_DEFAULT_HEEL_SOURCE == "rtmpose"


# ---------- identitatea cu main pe cele 15 clipuri dev ----------

def fingerprint(result: AnalysisResult) -> str:
    data = json.loads(result.to_json())
    for name in NEW_FIELDS:
        data.pop(name, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _dev_ready() -> bool:
    return (R.available() and len(M.dev_videos()) == 15 and all(
        R.points_cache_path(v, M.file_hash(M.V.VIDEOS / v)).exists() for v in M.dev_videos()))


@pytest.mark.skipif(not _dev_ready(), reason="lipsesc onnxruntime, modelul, clipurile dev sau punctele RTMPose")
def test_on_the_dev_clips_the_new_default_and_the_baseline_equal_the_old_switch_on_and_off(monkeypatch, tmp_path):
    original = M.result_cache_path
    monkeypatch.setattr(M, "result_cache_path", lambda *a, **k: tmp_path / original(*a, **k).name)
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))["clipuri"]
    for video in M.dev_videos():
        n_frames, _ = M.video_info(M.V.VIDEOS / video)
        segment = M.default_segment(video, n_frames)
        new = M.analyze(M.V.VIDEOS / video, "pas", segment, heel_source=M.APP_DEFAULT_HEEL_SOURCE)
        base = M.analyze(M.V.VIDEOS / video, "pas", segment, heel_source="mediapipe")
        assert new.pipeline == "rtmpose" and base.pipeline == "baseline"
        assert fingerprint(new) == expected[video]["rtmpose"], video
        assert fingerprint(base) == expected[video]["mediapipe"], video
