"""Stratul „călcâi stabil”: pe calea RTMPose, după fuziune și interval, Hampel pe poziția
călcâiului față de genunchi și călcâiul pe piciorul calului (AP-10K) scot cadre, cu motiv. Oprit implicit: atunci
rezultatul și JSON-ul sunt cele de acum. Clip sintetic (ca în test_rtmpose), estimatori falși."""
import inspect
import json

import numpy as np
import pytest
from test_rtmpose import FakeEstimator, setup

from ecvestru import cal_puncte as CP
from ecvestru import calcai_stabil as CS
from ecvestru import masurare as M
from ecvestru.rezultat import FrameMeasure


@pytest.fixture
def tmp_results(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    return tmp_path / "analiza"


class SpikeEstimator(FakeEstimator):
    """Călcâiul RTMPose la (50, 50), cu un salt la (60, 50) pe al `spike`-lea cadru cu călăreț."""

    def __init__(self, spike=6):
        super().__init__()
        self.spike = spike

    def __call__(self, image, bbox):
        points, scores = super().__call__(image, bbox)
        if len(self.calls) - 1 == self.spike:
            points[:] = (60.0, 50.0)
        return points, scores


class HorseOnHeel:
    """Punctele calului: pe al `on`-lea cadru cu cal, copita și cotul din stânga-față trec prin călcâi (50, 50)."""

    def __init__(self, on=3):
        self.on, self.calls = on, 0

    def __call__(self, image, bbox):
        points, scores = np.full((17, 2), 500.0), np.full(17, 0.9)
        if self.calls == self.on:
            points[CP.AP10K.index("left_elbow")] = (50.0, 40.0)
            points[CP.AP10K.index("left_front_paw")] = (50.0, 60.0)
        self.calls += 1
        return points, scores


def analyze(tmp_path, monkeypatch, estimator, **kwargs):
    predictor = setup(monkeypatch, tmp_path)
    return M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source="rtmpose",
                     heel_estimator=estimator, **kwargs)


# ---------- regulile, pe cadre ----------

def frames(n=20):
    return [FrameMeasure(frame=i, trunk_deg=1.0, leg_deg=0.0) for i in range(n)]


def test_hampel_and_the_horse_rule_unmeasure_frames_with_their_reasons_and_keep_the_trunk():
    n = 20
    heels = {i: (50.0, 50.0) for i in range(n)}
    heels[8] = (60.0, 50.0)
    points, scores = np.full((n, 17, 2), np.nan), np.full((n, 17), np.nan)
    points[15, CP.AP10K.index("left_knee")], points[15, CP.AP10K.index("left_back_paw")] = (50, 40), (50, 60)
    scores[15, [CP.AP10K.index("left_knee"), CP.AP10K.index("left_back_paw")]] = 0.9
    knee, torso = np.tile([50.0, 40.0], (n, 1)), np.full(n, 20.0)
    out = CS.stabilize(frames(n), heels, knee, torso, 30.0, points, scores)
    unmeasured = {f.frame: f.reason for f in out if f.leg_deg is None}
    assert unmeasured == {8: M.REASON_HEEL_JUMP, 15: M.REASON_HEEL_ON_HORSE_LEG}
    assert all(f.trunk_deg == 1.0 for f in out)


def test_already_unmeasured_frames_stay_as_they_were():
    given = frames(10)
    given[3] = given[3].model_copy(update={"leg_deg": None, "reason": M.REASON_IMPLAUSIBLE_ANGLE})
    heels = {i: (50.0, 50.0) for i in range(10) if i != 3}
    out = CS.stabilize(given, heels, np.tile([50.0, 40.0], (10, 1)), np.full(10, 20.0), 30.0,
                       np.full((10, 17, 2), np.nan), np.full((10, 17), np.nan))
    assert out[3] == given[3] and all(f.leg_deg == 0.0 for f in out if f.frame != 3)


def test_the_new_reasons_have_texts():
    assert M.REASON_TEXTS[M.REASON_HEEL_JUMP] and M.REASON_TEXTS[M.REASON_HEEL_ON_HORSE_LEG]
    assert M.REASON_HEEL_ON_HORSE_LEG != M.REASON_HEEL_ON_HORSE


# ---------- în analiză ----------

def test_the_layer_is_off_by_default():
    assert inspect.signature(M.analyze).parameters["heel_stable"].default is False


def test_off_the_result_and_its_json_are_the_ones_of_the_default_path(tmp_results, tmp_path, monkeypatch):
    default = analyze(tmp_path, monkeypatch, SpikeEstimator())
    off = analyze(tmp_path, monkeypatch, SpikeEstimator(), heel_stable=False)
    assert off.to_json() == default.to_json()
    assert set(json.loads(off.to_json())["layers"]) == {"quality", "rtmpose", "orientation", "rtmpose_model"}
    assert off.frames[6].leg_deg is not None                      # saltul rămâne măsurat pe calea de acum


def test_on_the_jump_and_the_heel_on_the_horse_leg_are_unmeasured(tmp_results, tmp_path, monkeypatch):
    res = analyze(tmp_path, monkeypatch, SpikeEstimator(spike=6), heel_stable=True, horse_estimator=HorseOnHeel(3))
    unmeasured = {f.frame: f.reason for f in res.frames if f.leg_deg is None}
    assert unmeasured == {6: M.REASON_HEEL_JUMP, 3: M.REASON_HEEL_ON_HORSE_LEG}
    assert res.layers.heel_stable == CS.VERSION and json.loads(res.to_json())["layers"]["heel_stable"] == CS.VERSION
    assert res.leg.n_frames == 10 and res.trunk.n_frames == 12
    assert {u.reason for u in res.unmeasured} == {M.REASON_HEEL_JUMP, M.REASON_HEEL_ON_HORSE_LEG}


def test_on_and_off_are_cached_separately(tmp_results, tmp_path, monkeypatch):
    assert analyze(tmp_path, monkeypatch, SpikeEstimator()).frames[6].leg_deg is not None
    on = analyze(tmp_path, monkeypatch, SpikeEstimator(), heel_stable=True, horse_estimator=HorseOnHeel(99))
    assert on.frames[6].leg_deg is None
    assert analyze(tmp_path, monkeypatch, SpikeEstimator()).frames[6].leg_deg is not None


def test_on_the_baseline_path_the_layer_does_not_run(tmp_results, tmp_path, monkeypatch):
    predictor = setup(monkeypatch, tmp_path)
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source="mediapipe",
                    heel_stable=True, horse_estimator=HorseOnHeel(3))
    assert res.layers.heel_stable is None and "heel_stable" not in json.loads(res.to_json())["layers"]


# ---------- alegerea modelului (pasul c a trecut regula: m implicit, l opțional) ----------

def test_the_model_is_m_by_default_and_l_only_with_the_layer(tmp_results, tmp_path, monkeypatch):
    from ecvestru.rezultat import HEEL_MODEL_L, HEEL_MODEL_M
    assert inspect.signature(M.analyze).parameters["heel_model"].default == HEEL_MODEL_M
    with pytest.raises(ValueError):
        analyze(tmp_path, monkeypatch, SpikeEstimator(), heel_model=HEEL_MODEL_L)               # fără strat
    with pytest.raises(ValueError):
        analyze(tmp_path, monkeypatch, SpikeEstimator(), heel_stable=True, heel_model="xl")


def test_with_l_the_whole_rtmpose_path_uses_the_l_points(tmp_results, tmp_path, monkeypatch):
    from ecvestru import rtmpose_l as RL
    from ecvestru.rezultat import HEEL_MODEL_L
    res = analyze(tmp_path, monkeypatch, SpikeEstimator(spike=6), heel_stable=True, heel_model=HEEL_MODEL_L,
                  horse_estimator=HorseOnHeel(99))
    assert res.frames[6].leg_deg is None and res.frames[6].reason == M.REASON_HEEL_JUMP
    assert res.layers.rtmpose_model == RL.MODEL_ZIP_SHA256[:8] and res.layers.heel_stable == CS.VERSION
    assert list((tmp_results / "rtmpose_l").glob("*.npz")) and not list((tmp_results / "rtmpose").glob("*.npz"))


def test_m_and_l_results_are_cached_separately(tmp_results, tmp_path, monkeypatch):
    from ecvestru.rezultat import HEEL_MODEL_L
    m = analyze(tmp_path, monkeypatch, SpikeEstimator(spike=99), heel_stable=True, horse_estimator=HorseOnHeel(99))
    l_estimator = SpikeEstimator(spike=6)
    l_res = analyze(tmp_path, monkeypatch, l_estimator, heel_stable=True, heel_model=HEEL_MODEL_L,
                    horse_estimator=HorseOnHeel(99))
    assert l_estimator.calls                                     # l rulează, nu vine din rezultatul m
    assert l_res.frames[6].leg_deg is None and m.frames[6].leg_deg is not None
    assert l_res.layers.rtmpose_model != m.layers.rtmpose_model
    again = analyze(tmp_path, monkeypatch, SpikeEstimator(spike=99), heel_stable=True, horse_estimator=HorseOnHeel(99))
    assert again.to_json() == m.to_json()


def test_a_missing_l_model_falls_back_with_a_reason(monkeypatch):
    from ecvestru import rtmpose as R
    from ecvestru import rtmpose_l as RL
    from ecvestru.rezultat import HEEL_MODEL_L, HEEL_MODEL_M
    monkeypatch.setattr(R, "runtime_available", lambda: True)
    monkeypatch.setattr(R, "model_path", lambda folder=None: tmp_model)
    tmp_model = object()

    def fail():
        raise OSError("fără rețea")
    monkeypatch.setattr(RL, "ensure_model", fail)
    assert M.rtmpose_problem(heel_model=HEEL_MODEL_M) is None
    problem = M.rtmpose_problem(heel_model=HEEL_MODEL_L)
    assert problem is not None and "fără rețea" in problem and "RTMPose-l" in problem
