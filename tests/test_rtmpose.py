"""Stratul RTMPose: partea numpy, model, cache, fuziune, intervalul și calea din analiză.
Testele care cer onnxruntime, modelul sau clipurile se sar dacă lipsesc."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_masurare import _gate_setup, _synthetic_clip

from ecvestru import masurare as M
from ecvestru import rtmpose as R
from ecvestru import stocare as ST
from ecvestru.rezultat import SCHEMA_VERSION, AnalysisResult

ROOT = Path(__file__).resolve().parents[1]
HORSE = [0.0, 20.0, 100.0, 60.0]       # x, y, lățime, înălțime: treimea de jos începe la y = 60


@pytest.fixture
def tmp_results(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    return tmp_path / "analiza"


# ---------- izolare ----------

def test_importing_the_app_modules_does_not_import_onnxruntime():
    code = ("import sys; import ecvestru.rtmpose, ecvestru.masurare, ecvestru.calitate, ecvestru.rezultat; "
            "assert 'onnxruntime' not in sys.modules, 'onnxruntime a fost importat'")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_cv2_has_what_the_layer_needs():
    assert callable(cv2.warpAffine) and callable(cv2.resize)


# ---------- partea numpy ----------

def test_halpe26_names_have_the_heels_where_the_layer_reads_them():
    assert len(R.HALPE26) == 26
    assert R.HALPE26[R.HEEL_INDEX["left"]] == "left_heel" and R.HALPE26[R.HEEL_INDEX["right"]] == "right_heel"


def test_center_scale_pads_and_fixes_aspect():
    center, scale = R.center_scale(np.array([10.0, 20.0, 110.0, 220.0]))
    assert np.allclose(center, (60, 120)) and np.allclose(scale, (187.5, 250))


def test_point_goes_to_model_input_and_back():
    center, scale = np.array([300.0, 200.0]), np.array([150.0, 200.0])
    point = np.array([[320.0, 150.0]])
    assert np.allclose(R.to_image(R.to_input(point, center, scale), center, scale), point)


def test_prepare_shapes_and_channel_order():
    image = np.zeros((400, 600, 3), np.uint8)
    image[..., 0] = 255   # albastru în BGR
    tensor, _, _ = R.prepare(image, np.array([100.0, 50.0, 300.0, 350.0]))
    assert tensor.shape == (1, 3, R.INPUT_H, R.INPUT_W) and tensor.dtype == np.float32
    middle = (0, slice(None), R.INPUT_H // 2, R.INPUT_W // 2)
    assert tensor[middle][2] > tensor[middle][0]


def test_decode_simcc_finds_peaks_and_marks_missing_points():
    simcc_x = np.zeros((1, 26, R.INPUT_W * 2), np.float32)
    simcc_y = np.zeros((1, 26, R.INPUT_H * 2), np.float32)
    simcc_x[0, 3, 100], simcc_y[0, 3, 41] = 0.8, 0.4
    locs, scores = R.decode_simcc(simcc_x, simcc_y)
    assert np.allclose(locs[3], (50.0, 20.5)) and scores[3] == pytest.approx(0.6) and np.isnan(locs[0]).all()


def test_rider_box_needs_enough_points():
    points = np.full((33, 2), np.nan)
    points[:4] = [[1, 2], [3, 4], [5, 6], [7, 8]]
    assert R.rider_box(points) is None
    points[4] = [0, 9]
    assert R.rider_box(points).tolist() == [0.0, 2.0, 7.0, 9.0]


# ---------- fuziunea și intervalul, un cadru ----------

def test_fusion_keeps_rtmpose_outside_the_lower_third():
    assert R.fuse_heel((50.0, 30.0), (10.0, 10.0), HORSE) == ((50.0, 30.0), R.SOURCE_RTMPOSE)


def test_fusion_takes_the_original_when_rtmpose_is_in_the_lower_third():
    assert R.fuse_heel((50.0, 70.0), (10.0, 10.0), HORSE) == ((10.0, 10.0), R.SOURCE_ORIGINAL)


def test_fusion_excludes_when_both_are_in_the_lower_third():
    assert R.fuse_heel((50.0, 70.0), (10.0, 80.0), HORSE) == (None, R.SOURCE_EXCLUDED)


def test_fusion_excludes_when_rtmpose_is_missing_and_the_original_is_in_the_lower_third():
    assert R.fuse_heel(None, (10.0, 80.0), HORSE) == (None, R.SOURCE_EXCLUDED)
    assert R.fuse_heel(None, (10.0, 10.0), HORSE) == ((10.0, 10.0), R.SOURCE_ORIGINAL)


def test_fusion_without_a_horse_box_never_sees_the_lower_third():
    assert R.fuse_heel((50.0, 99.0), (10.0, 99.0), None) == ((50.0, 99.0), R.SOURCE_RTMPOSE)


# ---------- piciorul pe calea RTMPose ----------

def frame(i, leg=-5.0, trunk=1.0, reason=None):
    from ecvestru.rezultat import FrameMeasure
    return FrameMeasure(frame=i, trunk_deg=trunk, leg_deg=leg, reason=reason)


def raw_point(heel=(50.0, 95.0)):
    return {"points": {"shoulder": (50.0, 10.0), "hip": (50.0, 30.0), "heel": heel, "facing": 1.0}, "motiv": None}


def test_the_leg_comes_from_the_fused_heel_and_the_trunk_stays():
    frames = [frame(0, leg=-5.0)]
    out = R.measure_legs(frames, {0: raw_point()}, np.array([[50.0, 50.0]]), {0: HORSE})
    assert out[0].leg_deg == pytest.approx(0.0) and out[0].trunk_deg == 1.0 and out[0].reason is None


def test_the_angle_follows_the_app_convention():
    out = R.measure_legs([frame(0)], {0: raw_point()}, np.array([[40.0, 50.0]]), {0: HORSE})
    assert out[0].leg_deg == pytest.approx(M.angle_from_vertical((50.0, 30.0), (40.0, 50.0), 1.0))
    assert out[0].leg_deg != 0


def test_a_heel_in_the_lower_third_on_both_models_is_excluded_with_the_gate_a_reason():
    out = R.measure_legs([frame(0)], {0: raw_point((50.0, 95.0))}, np.array([[50.0, 70.0]]), {0: HORSE})
    assert out[0].leg_deg is None and out[0].reason == M.REASON_HEEL_ON_HORSE and out[0].trunk_deg == 1.0


def test_a_missing_rtmpose_heel_falls_back_to_the_original_when_that_is_fine():
    out = R.measure_legs([frame(0)], {0: raw_point((50.0, 50.0))}, np.array([[np.nan, np.nan]]), {0: HORSE})
    assert out[0].leg_deg == pytest.approx(0.0)


def test_an_implausible_angle_is_excluded_after_the_fusion():
    out = R.measure_legs([frame(0)], {0: raw_point()}, np.array([[250.0, 40.0]]), {0: HORSE})
    assert out[0].leg_deg is None and out[0].reason == M.REASON_IMPLAUSIBLE_ANGLE


def test_unmeasured_legs_stay_untouched():
    unmeasured = frame(0, leg=None, reason=M.REASON_HEEL)
    out = R.measure_legs([unmeasured], {0: raw_point()}, np.array([[50.0, 50.0]]), {0: HORSE})
    assert out[0] == unmeasured


# ---------- modelul ----------

def make_zip(path: Path) -> str:
    inner = "20230831/rtmpose_onnx/rtmpose-x/end2end.onnx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(inner, b"onnx")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_model_url_and_checksum_are_pinned():
    assert R.MODEL_URL.startswith("https://download.openmmlab.com/") and R.MODEL_URL.endswith(".zip")
    assert R.MODEL_ZIP_SHA256 == "55b81170e236040b59fc792ad0a8315301ac4c079a3bdb1095d838aad3088d18"


def test_download_checks_the_checksum_and_unpacks_the_model(tmp_path):
    archive = tmp_path / "m.zip"
    digest = make_zip(archive)
    found = R.download_model(archive.as_uri(), digest, tmp_path / "models")
    assert found.name == "end2end.onnx" and found.read_bytes() == b"onnx"
    assert R.model_path(tmp_path / "models") == found


def test_a_wrong_checksum_leaves_no_model(tmp_path):
    archive = tmp_path / "m.zip"
    make_zip(archive)
    with pytest.raises(ValueError, match="sha256"):
        R.download_model(archive.as_uri(), "0" * 64, tmp_path / "models")
    assert R.model_path(tmp_path / "models") is None and not list((tmp_path / "models").glob("*"))


def test_the_layer_is_unavailable_without_the_runtime_or_without_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(R, "runtime_available", lambda: True)
    assert not R.available()                              # fără model
    make_zip(tmp_path / "z.zip")
    R.download_model((tmp_path / "z.zip").as_uri(), hashlib.sha256((tmp_path / "z.zip").read_bytes()).hexdigest(),
                     tmp_path)
    assert R.available()
    monkeypatch.setattr(R, "runtime_available", lambda: False)
    assert not R.available()                              # fără onnxruntime


def test_the_download_command_is_a_romanian_verb(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(R, "download_model", lambda *a, **k: called.append(1) or Path("/x/end2end.onnx"))
    assert R.main(["descarca"]) == 0 and called and "end2end.onnx" in capsys.readouterr().out


# ---------- rularea pe clip și cache ----------

class FakeEstimator:
    def __init__(self, heel=(50.0, 50.0)):
        self.heel, self.calls = heel, []

    def __call__(self, image, bbox):
        self.calls.append(np.asarray(bbox))
        points = np.zeros((26, 2))
        points[R.HEEL_INDEX["left"]] = points[R.HEEL_INDEX["right"]] = self.heel
        return points, np.ones(26)


def test_run_clip_runs_the_estimator_on_rider_frames_only_and_caches(tmp_results, tmp_path):
    from test_calitate import skeleton
    sk = skeleton(6)
    from ecvestru import schelet as Z
    sk.source[[2, 4]] = Z.SOURCE_NONE
    _synthetic_clip(tmp_path / "a.mp4", 6)
    fake = FakeEstimator()
    points = R.load_points(tmp_path / "a.mp4", "a.mp4", "hash", sk, estimator=fake)
    assert points.shape == (6, 26, 2) and len(fake.calls) == 4
    assert np.isnan(points[2]).all() and np.isnan(points[4]).all() and not np.isnan(points[0]).any()
    again = R.load_points(tmp_path / "a.mp4", "a.mp4", "hash", sk, estimator=FakeEstimator())
    assert np.array_equal(again, points, equal_nan=True)
    path = R.points_cache_path("a.mp4", "hash")
    assert path.exists() and path.parent.name == "rtmpose" and f"r{R.RTMPOSE_VERSION}" in path.name
    assert R.MODEL_ZIP_SHA256[:8] in path.name


def test_the_estimator_gets_the_box_of_the_skeleton_points(tmp_results, tmp_path):
    from test_calitate import skeleton
    sk = skeleton(2)
    _synthetic_clip(tmp_path / "a.mp4", 2)
    fake = FakeEstimator()
    R.load_points(tmp_path / "a.mp4", "a.mp4", "h2", sk, estimator=fake)
    assert np.allclose(fake.calls[0], R.rider_box(sk.xy_px[0]))


def test_only_selected_frames_can_be_run(tmp_results, tmp_path):
    from test_calitate import skeleton
    sk = skeleton(6)
    _synthetic_clip(tmp_path / "a.mp4", 6)
    fake = FakeEstimator()
    points = R.run_clip(tmp_path / "a.mp4", sk, fake, only={1, 4})
    assert len(fake.calls) == 2 and np.isnan(points[0]).all() and not np.isnan(points[4]).any()


# ---------- calea RTMPose în analiză ----------

def setup(monkeypatch, tmp_path):
    """Ca _gate_setup, dar scheletul rămâne în memorie: analizele repetate nu-l citesc din cache."""
    predictor = _gate_setup(monkeypatch, tmp_path)
    skeleton = ST.extract_skeleton(None)
    monkeypatch.setattr(ST, "load_rider_skeleton", lambda *a: skeleton)
    return predictor


def test_with_rtmpose_the_gate_a_is_not_applied_and_the_source_is_in_the_json(tmp_results, tmp_path, monkeypatch):
    predictor = setup(monkeypatch, tmp_path)       # cadrul 11 e A pe calea implicită
    default = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor)
    assert default.frames[11].leg_deg is None and default.heel_source == "mediapipe"
    fake = FakeEstimator(heel=(50.0, 50.0))
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source="rtmpose",
                    heel_estimator=fake)
    assert res.heel_source == "rtmpose" and json.loads(res.to_json())["heel_source"] == "rtmpose"
    assert res.frames[11].leg_deg == pytest.approx(0.0) and res.leg.n_frames == 12
    assert res.quality.excluded_frames == 0


def test_with_rtmpose_the_interval_and_the_fusion_exclude_with_reasons(tmp_results, tmp_path, monkeypatch):
    predictor = setup(monkeypatch, tmp_path)
    fake = FakeEstimator(heel=(50.0, 70.0))               # în treimea de jos, ca și originalul (50, 95)
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source="rtmpose",
                    heel_estimator=fake)
    assert res.leg.n_frames == 0 and res.quality.excluded_frames == 12 and res.trunk.n_frames == 12
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "alt"))     # alt cache: altă estimare
    far = FakeEstimator(heel=(250.0, 40.0))
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source="rtmpose",
                    heel_estimator=far)
    assert res.quality.implausible_frames == 12 and res.unmeasured[0].reason == M.REASON_IMPLAUSIBLE_ANGLE


def test_the_two_paths_have_separate_result_files(tmp_results, tmp_path, monkeypatch):
    predictor = setup(monkeypatch, tmp_path)
    M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor)
    M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source="rtmpose",
              heel_estimator=FakeEstimator())
    names = sorted(p.name for p in (tmp_results).glob("*.json"))
    assert len(names) == 2 and sum("rtm" in n for n in names) == 1


def test_with_the_switch_off_the_rtmpose_code_is_never_called(tmp_results, tmp_path, monkeypatch):
    predictor = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "load_points", lambda *a, **k: pytest.fail("RTMPose a rulat cu comutatorul oprit"))
    assert M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor).heel_source == "mediapipe"


def test_default_json_equals_the_one_built_from_the_gate_alone(tmp_results, tmp_path, monkeypatch):
    """Cu comutatorul oprit, JSON-ul e cel de după poarta A + interval; diferă doar câmpurile de versiune/sursă."""
    from ecvestru import calitate as C
    predictor = setup(monkeypatch, tmp_path)
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor)
    raw = predictor(tmp_path / "a.mp4")
    sk = ST.extract_skeleton(None)
    clip = C.ClipData("a.mp4", sk, raw, M.frame_measures(raw, 12), ST.load_boxes())
    frames = C.gate_frames(clip)
    assert res.frames == frames
    assert res.quality == C.summarize_quality(frames)
    assert SCHEMA_VERSION == "1.4" and AnalysisResult.from_json(res.to_json()) == res



# ---------- dependențele ----------

def test_onnxruntime_is_a_dependency_of_the_app_and_rtmlib_is_not():
    import tomllib
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert any(d.startswith("onnxruntime") for d in project["dependencies"])
    assert not any("rtmlib" in d for d in project["dependencies"])
    assert "rtmpose" not in project.get("optional-dependencies", {})


# ---------- filmul: călcâiul folosit în măsurare ----------

def test_film_raw_carries_the_heel_used_in_the_measurement_without_touching_the_original():
    raw = {0: raw_point((50.0, 95.0)), 1: raw_point((50.0, 95.0))}
    frames = [frame(0), frame(1, leg=None, reason=M.REASON_HEEL_ON_HORSE)]
    heels = np.array([[50.0, 50.0], [50.0, 50.0]])
    out = R.film_raw(raw, frames, heels, {0: HORSE, 1: HORSE})
    assert out[0]["points"]["heel"] == (50.0, 50.0) and out[1]["points"]["heel"] == (50.0, 95.0)
    assert raw[0]["points"]["heel"] == (50.0, 95.0)
    assert out[0]["points"]["hip"] == raw[0]["points"]["hip"] and out[0]["motiv"] is None


def test_film_raw_uses_the_original_where_the_fusion_took_it():
    out = R.film_raw({0: raw_point((50.0, 50.0))}, [frame(0)], np.array([[50.0, 70.0]]), {0: HORSE})
    assert out[0]["points"]["heel"] == (50.0, 50.0)       # RTMPose era în treimea de jos: s-a luat originalul


def film_inputs(monkeypatch, tmp_path, heel_source, estimator_heel=(50.0, 50.0)):
    predictor = setup(monkeypatch, tmp_path)
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source=heel_source,
                    heel_estimator=FakeEstimator(heel=estimator_heel))
    seen = {}
    from ecvestru import suprapunere as S
    monkeypatch.setattr(S, "film_bytes", lambda video, frames, raw, fps, blur=False, skeleton=None:
                        seen.update(raw=raw, frames=frames) or b"film")
    assert S.clip_film(tmp_path / "a.mp4", res) == b"film"
    return seen


def test_the_film_of_the_rtmpose_path_draws_the_rtmpose_heel(tmp_results, tmp_path, monkeypatch):
    seen = film_inputs(monkeypatch, tmp_path, "rtmpose")
    assert all(seen["raw"][i]["points"]["heel"] == (50.0, 50.0) for i in range(12))


def test_the_film_of_the_default_path_keeps_the_mediapipe_heel(tmp_results, tmp_path, monkeypatch):
    seen = film_inputs(monkeypatch, tmp_path, "mediapipe")
    assert all(tuple(seen["raw"][i]["points"]["heel"]) == (50.0, 95.0) for i in range(12))


def test_the_film_of_the_rtmpose_path_fails_cleanly_when_the_points_cannot_be_read(tmp_results, tmp_path, monkeypatch):
    from ecvestru import suprapunere as S
    predictor = setup(monkeypatch, tmp_path)
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source="rtmpose",
                    heel_estimator=FakeEstimator())
    for cached in (tmp_results / "rtmpose").glob("*.npz"):
        cached.unlink()
    monkeypatch.setattr(R, "RtmPose", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("fără model")))
    with pytest.raises(S.FilmError, match="RTMPose"):
        S.clip_film(tmp_path / "a.mp4", res)


def test_heel_sources_name_where_the_heel_of_each_measured_leg_comes_from():
    raw = {0: raw_point((50.0, 50.0)), 1: raw_point((50.0, 50.0)), 2: raw_point((50.0, 95.0))}
    frames = [frame(0), frame(1), frame(2, leg=None, reason=M.REASON_HEEL_ON_HORSE)]
    heels = np.array([[50.0, 40.0], [50.0, 70.0], [50.0, 70.0]])       # 1: RTMPose în treimea de jos → originalul
    assert R.heel_sources(raw, frames, heels, {i: HORSE for i in range(3)}) == {0: R.SOURCE_RTMPOSE,
                                                                               1: R.SOURCE_ORIGINAL}
