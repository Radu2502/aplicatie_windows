"""Vederea călărețului cu routerul V5 (opțiunea „Vederea călărețului (router V5, experimental)”, oprită implicit):
oprită, rezultatul și filmul sunt cele de dinainte; pornită, câmpul `rider_views` (schema 1.7) are vederea pe fiecare
cadru analizat, banda de jos a filmului o arată, iar „Scheletul calului” merge în continuare, singur și împreună cu
V5. Fără trecerea grea: vederea pe clip e simulată unde e nevoie."""
import json

import numpy as np
import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest
from test_schelet_cal import BOX_XYWH, HEIGHT, PAGE, POINT, WIDTH, Counting, N, synthetic_clip

from ecvestru import analiza_urmarita as AU
from ecvestru import masurare as M
from ecvestru import router_v5 as RV5
from ecvestru import stocare as ST
from ecvestru import suprapunere as S
from ecvestru.rezultat import (
    COMBINED_SCHEMA_VERSION as VIEWS_RESULT_SCHEMA,
)
from ecvestru.rezultat import (
    HORSE_SCHEMA_VERSION,
    RIDER_VIEW_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AnalysisResult,
    FrameMeasure,
)

VIEWS = ("profil", "fata", "spate", "oblic", "nu_se_vede")


@pytest.fixture
def clip(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    monkeypatch.setattr(ST, "run_candidate", lambda path: {i: POINT for i in range(N)})
    monkeypatch.setattr(ST, "run_ratios", lambda path: {})
    monkeypatch.setattr(ST, "detect_horses", lambda path: {i: [BOX_XYWH] for i in range(N)})
    return synthetic_clip(tmp_path / "c.mp4")


def fake_model(tmp_path, stay=0.9, evaluation=0.42) -> RV5.Model:
    """Un model mic: vederea e dată doar de semnalul 0 (pozitiv → profil, negativ → spate)."""
    k, d = len(VIEWS), len(RV5.ALL_SIGNALS)
    weights = np.zeros((2 * d + 1, k))
    weights[0, VIEWS.index("profil")], weights[0, VIEWS.index("spate")] = 5.0, -5.0
    data = {"versiune": 1, "tip": "logistica_echilibrata", "semnale": list(RV5.ALL_SIGNALS), "clase": list(VIEWS),
            "medie": [0.0] * d, "abatere": [1.0] * d, "ponderi": weights.tolist(), "hmm_ramane": stay,
            "antrenare": {}, "evaluare": {"echilibrata": evaluation}}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return RV5.load_model(path)


def views_for(n: int) -> np.ndarray:
    out = np.array(["profil"] * n, dtype=object)
    out[2] = "spate"
    out[-1] = None
    return out


# ---------- implicit oprită ----------

def test_the_option_is_off_by_default_and_the_analysis_does_not_know_about_it():
    import inspect
    assert RV5.APP_DEFAULT_RIDER_VIEW is False
    for fn in (M.analyze, M.analyze_upload, AU.analyze):
        assert "rider_view" not in inspect.signature(fn).parameters


def test_without_views_the_json_has_no_field_and_the_old_schema(clip):
    data = json.loads(M.analyze(clip, "pas", (1, 5)).to_json())
    assert "rider_views" not in data and data["schema_version"] == SCHEMA_VERSION == "1.4"


# ---------- modelul ----------

def test_the_shipped_model_loads_with_the_signals_of_this_code():
    model = RV5.load_model()
    assert set(model.classes) <= set(VIEWS) and len(model.classes) >= 2
    assert model.weights.shape == (2 * len(RV5.ALL_SIGNALS) + 1, len(model.classes))
    assert 0 < model.stay < 1 and len(model.sha) == 8
    score = RV5.balanced_accuracy(model)
    assert score is None or 0 <= score <= 1
    proba = model.predict_proba(np.full((3, len(RV5.ALL_SIGNALS)), np.nan))
    assert np.allclose(proba.sum(axis=1), 1)


def test_a_model_with_other_signals_is_refused(tmp_path):
    fake_model(tmp_path)
    data = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    data["semnale"] = data["semnale"][::-1]
    (tmp_path / "model.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        RV5.load_model(tmp_path / "model.json")


def test_predict_follows_the_signal_smooths_and_leaves_frames_without_target_empty(tmp_path):
    model = fake_model(tmp_path, stay=0.6)
    X = np.zeros((8, len(RV5.ALL_SIGNALS)))
    X[:, 0] = [1, 1, 1, -1, -1, -1, 1, 1]
    X[7] = np.nan
    views = RV5.predict(model, X)
    assert list(views[:6]) == ["profil"] * 3 + ["spate"] * 3 and views[7] is None
    strong = fake_model(tmp_path, stay=0.999999)
    X[3:6, 0] = -0.05                            # un salt scurt și slab: HMM-ul îl netezește
    assert list(RV5.predict(strong, X)[:7]) == ["profil"] * 7


def test_the_hmm_keeps_a_constant_signal():
    proba = np.tile([0.1, 0.6, 0.1, 0.1, 0.1], (5, 1))
    assert list(RV5.hmm(proba, VIEWS, 0.9)) == ["fata"] * 5


# ---------- semnalele (aceleași formule ca la antrenare) ----------

def test_the_motion_is_the_box_move_minus_the_camera_over_the_width():
    n, fps = 31, 10.0
    boxes = np.array([[x, 0.0, x + 10.0, 20.0] for x in np.arange(n, dtype=float)])
    flow = np.zeros((n, 2))
    lateral, scale = RV5.motion(boxes, flow, fps)
    assert lateral[15] == pytest.approx(10 / 10) and scale[15] == pytest.approx(0.0)
    flow[:, 0] = 1.0                                       # camera se mișcă la fel: calul stă față de fundal
    assert RV5.motion(boxes, flow, fps)[0][15] == pytest.approx(0.0)


def test_new_signals_are_nan_without_target():
    n = 4
    out = RV5.new_signals(np.ones((n, 26)), np.zeros((n, 26, 2)), None, None, np.ones((n, 17)),
                          np.tile([0.0, 0.0, 10.0, 10.0], (n, 1)), np.zeros((n, 2)), 10.0,
                          np.array([True, False, True, True]))
    assert out.shape == (n, len(RV5.NEW_SIGNALS)) and np.isnan(out[1]).all()
    assert out[0, RV5.NEW_SIGNALS.index("fata_rtm")] == 1.0 and out[0, RV5.NEW_SIGNALS.index("cal_cap")] == 1.0


def test_the_background_flow_follows_a_shifted_texture():
    rng = np.random.default_rng(0)
    a = (rng.random((120, 160)) * 255).astype(np.uint8)
    import cv2
    a = cv2.GaussianBlur(a, (0, 0), 2)
    b = np.roll(a, 3, axis=1)
    shift = RV5.background_flow(a, b, [])
    assert shift[0] == pytest.approx(3, abs=0.5) and abs(shift[1]) < 0.5


# ---------- în rezultat ----------

def test_attach_gives_schema_1_7_with_one_view_per_analysed_frame(clip, tmp_path):
    model = fake_model(tmp_path)
    result = M.analyze(clip, "pas", (1, 5))
    on = RV5.attach(result, views_for(N), model)
    data = json.loads(on.to_json())
    assert data["schema_version"] == RIDER_VIEW_SCHEMA_VERSION == "1.7"
    assert RIDER_VIEW_SCHEMA_VERSION not in (SCHEMA_VERSION, HORSE_SCHEMA_VERSION, VIEWS_RESULT_SCHEMA)
    rv = data["rider_views"]
    assert rv["model"] == model.sha and rv["balanced_accuracy"] == 0.42 and rv["version"] == RV5.VERSION
    assert [f["frame"] for f in rv["frames"]] == [1, 2, 3, 4, 5]
    assert [f["view"] for f in rv["frames"]] == ["profil", "spate", "profil", "profil", None]
    rest = {k: v for k, v in data.items() if k not in ("schema_version", "rider_views")}
    assert rest == {k: v for k, v in json.loads(result.to_json()).items() if k != "schema_version"}
    assert AnalysisResult.model_validate_json(on.to_json()) == on


def test_the_schema_follows_the_fields(clip, tmp_path):
    model = fake_model(tmp_path)
    plain = M.analyze(clip, "pas", (1, 5))
    on = RV5.attach(plain, views_for(N), model)
    for wrong in ("1.4", "1.6"):
        with pytest.raises(ValidationError):
            AnalysisResult.model_validate({**on.model_dump(), "schema_version": wrong})
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({**plain.model_dump(), "schema_version": "1.7"})


def test_the_horse_skeleton_works_alone_and_with_v5(clip, tmp_path):
    model = fake_model(tmp_path)
    horse = M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=Counting())
    assert json.loads(horse.to_json())["schema_version"] == HORSE_SCHEMA_VERSION
    both = RV5.attach(horse, views_for(N), model)
    data = json.loads(both.to_json())
    assert data["schema_version"] == "1.7" and data["horse_keypoints"] and data["rider_views"]
    assert AnalysisResult.model_validate_json(both.to_json()) == both


def test_with_views_uses_the_tracking_target_of_a_tracked_result(clip, tmp_path, monkeypatch):
    model, seen = fake_model(tmp_path), []

    def fake(video_path, name, video_hash, click=None, stages=None, model=None, pose=None, horse_estimator=None):
        seen.append(click)
        return views_for(N), click
    monkeypatch.setattr(RV5, "clip_views", fake)
    old = M.analyze(clip, "pas", (1, 5))
    assert RV5.with_views(clip, old, model=model).rider_views is not None and seen == [None]


def test_without_target_the_views_are_unavailable(clip, monkeypatch, tmp_path):
    monkeypatch.setattr(AU, "heavy_pass", lambda *a, **k: (object(), []))

    def no_target(*a, **k):
        raise AU.TargetNeeded("doi călăreți")
    monkeypatch.setattr(AU, "resolve_target", no_target)
    with pytest.raises(RV5.ViewsUnavailable):
        RV5.clip_views(clip, "c.mp4", "h", model=fake_model(tmp_path))


# ---------- filmul ----------

def test_the_film_shows_the_view_in_a_bottom_band_only_when_asked():
    frame = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
    measure = FrameMeasure(frame=0, trunk_deg=None, leg_deg=None, reason="x")
    plain = S.overlay_frame(frame, None, measure)
    assert np.array_equal(S.overlay_frame(frame, None, measure, view=None), plain)
    shown = S.overlay_frame(frame, None, measure, view="spate")
    band = S.band_height(shown)
    assert not np.array_equal(shown[-band:], plain[-band:]) and np.array_equal(shown[:-band], plain[:-band])


def test_the_clip_film_passes_the_views_only_when_the_result_has_them(clip, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(S, "film_bytes", lambda *a, **k: calls.append(k) or b"film")
    model = fake_model(tmp_path)
    plain = M.analyze(clip, "pas", (1, 5))
    horse = M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=Counting())
    S.clip_film(clip, plain)
    S.clip_film(clip, RV5.attach(plain, views_for(N), model))
    S.clip_film(clip, RV5.attach(horse, views_for(N), model))
    assert calls[0] == {}
    assert calls[1] == {"views": {1: "profil", 2: "spate", 3: "profil", 4: "profil"}}
    assert set(calls[2]) == {"horse", "views"}


# ---------- pagina ----------

def page(monkeypatch, tmp_path, seen: list) -> AppTest:
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    monkeypatch.setattr(ST, "run_candidate", lambda path: {i: POINT for i in range(N)})
    monkeypatch.setattr(ST, "run_ratios", lambda path: {})
    monkeypatch.setattr(ST, "detect_horses", lambda path: {i: [BOX_XYWH] for i in range(N)})
    model = fake_model(tmp_path)

    def fake(video_path, result, click=None, stages=None, **kwargs):
        seen.append(result.video)
        return RV5.attach(result, views_for(result.n_frames), model)
    monkeypatch.setattr(RV5, "with_views", fake)
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    at.radio(key="an-source").set_value("upload").run()
    at.file_uploader[0].upload("clip.mp4", synthetic_clip(tmp_path / "u.mp4").read_bytes(), "video/mp4").run()
    return at


def test_the_page_option_is_off_by_default_and_explains_itself(monkeypatch, tmp_path):
    seen = []
    at = page(monkeypatch, tmp_path, seen)
    box = at.checkbox(key="an-rider-view")
    assert box.value is False and box.label == "Vederea călărețului (router V5, experimental)"
    assert any(c.value == "experimental, doar afișare" for c in at.caption)
    at.button(key="an-run").click().run()
    assert not at.exception, at.exception
    assert seen == [] and not any("router V5" in m.value for m in at.markdown)


def test_the_page_option_on_shows_the_views(monkeypatch, tmp_path):
    seen = []
    at = page(monkeypatch, tmp_path, seen)
    at.checkbox(key="an-rider-view").check().run()
    at.button(key="an-run").click().run()
    assert not at.exception, at.exception
    assert len(seen) == 1 and any("router V5" in m.value for m in at.markdown)
    assert len(at.get("video")) == 1


def test_the_page_runs_both_options_together(monkeypatch, tmp_path):
    seen = []
    at = page(monkeypatch, tmp_path, seen)
    original = M.analyze

    def with_horse(*args, **kwargs):
        horse = {"horse_estimator": Counting()} if kwargs.get("horse_skeleton") else {}
        return original(*args, **{**kwargs, **horse})
    monkeypatch.setattr(M, "analyze", with_horse)
    at.checkbox(key="an-rider-view").check().run()
    at.checkbox(key="an-horse-skeleton").check().run()
    at.button(key="an-run").click().run()
    assert not at.exception, at.exception
    assert len(seen) == 1 and any(c.value == S.HORSE_LEGEND for c in at.caption)


def test_the_page_has_only_the_user_options_all_off(monkeypatch, tmp_path):
    """Pagina are doar bifele pentru utilizator: alegerea călăreților, călcâiul stabil, scheletul calului și vederea."""
    at = page(monkeypatch, tmp_path, [])
    boxes = {c.key: c.value for c in at.checkbox}
    assert boxes == {"an-pick": False, "an-heel-stable": False, "an-horse-skeleton": False, "an-rider-view": False}
