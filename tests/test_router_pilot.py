"""Routerul pilot pe ținta urmărită (ecvestru/router_pilot.py) și opțiunea lui din „Analiză”: semnalele din RTMPose și
AP-10K pe ținta aleasă, „necunoscut” fără țintă, validarea încrucișată, confuzia, intervalele afișate. Date inventate,
fără video real; vechiul mod (router_vedere pe candidat) rămâne neatins."""
from pathlib import Path

import numpy as np
import pytest

from ecvestru import cal_puncte as CP
from ecvestru import evaluare_router as ER
from ecvestru import intervale as IV
from ecvestru import masurare as M
from ecvestru import router_pilot as RP
from ecvestru import router_vedere as RV
from ecvestru import urmarire_multi as U

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "app" / "pages" / "analiza.py")


def rider(shoulder_dx: float) -> np.ndarray:
    points = np.zeros((26, 2), np.float32)
    points[5], points[6] = (100 + shoulder_dx, 100), (100 - shoulder_dx, 100)       # umerii stâng, drept
    points[7], points[8] = (100 + shoulder_dx, 150), (100 - shoulder_dx, 150)       # coatele
    points[11], points[12] = (100, 200), (100, 200)                                 # șoldurile
    return points


def test_the_pilot_is_off_by_default_and_uses_the_tracked_target():
    assert RP.ROUTER_PILOT_DEFAULT is False and "k = 1 s" in RP.TRACKING


def test_signals_come_from_the_target_rider_and_horse_and_are_unknown_without_a_target():
    tracks = U.ClipTracks(fps=30.0, n_frames=3, horses=[{1: (0.0, 150.0, 200.0, 250.0)}] * 3,
                          people=[{4: (50.0, 50.0, 150.0, 250.0)}] * 3, keypoints=[{4: rider(20.0)}] * 3)
    choice = U.Choice([U.ON_C1, U.ON_C1, U.NO_SKELETON], [1, 1, 1], [4, 4, None], 0, None, 0, 0)
    xy = RP.rider_xy(tracks, choice)
    assert xy.shape == (3, 33, 2) and tuple(xy[0, RV.LEFT_SHOULDER]) == (120.0, 100.0) and np.isnan(xy[2]).all()
    horse = CP.ClipRun(np.full((3, 17, 2), np.nan), np.full((3, 17), np.nan), RP.target_boxes(tracks, [1, 1, 1]), 0.0)
    signals = RP.target_signals(tracks, choice, horse)
    row = dict(zip(signals.names, signals.values[0], strict=True))
    assert row["raport_umeri"] == pytest.approx(0.4) and row["umeri_stanga_dreapta"] == pytest.approx(0.4)
    assert row["cal_caseta"] == pytest.approx(2.0) and np.isnan(row["orientare_cos"])       # fără world pe RTMPose
    assert np.isnan(signals.values[2]).all()                                                 # fără țintă


def test_the_target_horse_points_run_only_on_frames_with_a_target_box(tmp_path):
    import cv2
    path = tmp_path / "s.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    boxes = np.array([[0, 0, 10, 10], [np.nan] * 4, [5, 5, 20, 20]], float)
    calls = []
    fake = lambda image, box: (calls.append(tuple(box)) or np.ones((17, 2)), np.ones(17))  # noqa: E731
    run = RP.run_target_horse(path, boxes, estimator=fake)
    assert calls == [(0, 0, 10, 10), (5, 5, 20, 20)] and np.isnan(run.points[1]).all() and run.points[2, 0, 0] == 1
    RP.save_horse(tmp_path / "h.npz", run)
    back = RP.load_horse(tmp_path / "h.npz")
    assert np.array_equal(back.boxes, run.boxes, equal_nan=True) and back.seconds == run.seconds


def synthetic_clip(seed: int, n: int = 120) -> ER.ClipData:
    """Profil când raportul umerilor e mic, față când e mare; ultimele 10 cadre fără țintă."""
    rng = np.random.default_rng(seed)
    labels = np.array([IV.PROFILE] * (n // 2) + [IV.FRONT] * (n - n // 2), dtype=object)
    values = np.full((n, len(RV.SIGNALS)), np.nan)
    values[:, 0] = np.where(labels == IV.PROFILE, 0.1, 0.9) + rng.normal(0, 0.02, n)
    values[-10:] = np.nan
    return ER.ClipData(values=values, labels=labels, fps=30.0)


def test_leave_one_out_predicts_each_clip_with_a_model_trained_on_the_others():
    clips = {f"c{k}": synthetic_clip(k) for k in range(3)}
    preds = RP.leave_one_out(clips)
    assert set(preds) == set(clips)
    for c, p in preds.items():
        assert list(p[-10:]) == [RP.UNKNOWN] * 10
        result = RP.evaluate(clips[c].labels, p)
        assert result["cu_tinta"]["r1"]["value"] == 0.0 and result["cadre_etichetate_cu_tinta"] == 110
        assert result["necunoscut"][IV.FRONT] == 10 and result["toate"]["r2"]["n"] == 60


def test_the_confusion_counts_human_view_against_router_view():
    truth = np.array([IV.PROFILE, IV.PROFILE, IV.FRONT, IV.BACK], dtype=object)
    voted = np.array([IV.PROFILE, RV.UNDECIDED, IV.BACK, IV.BACK], dtype=object)
    table = RP.confusion(truth, voted)
    assert table[IV.PROFILE][IV.PROFILE] == 1 and table[IV.PROFILE][RV.UNDECIDED] == 1
    assert table[IV.FRONT][IV.BACK] == 1 and table[IV.BACK][IV.BACK] == 1 and table[IV.OBLIQUE][IV.PROFILE] == 0


def test_views_are_saved_as_intervals_and_read_back_by_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    views = np.array([IV.PROFILE] * 3 + [RP.UNKNOWN] * 2 + [IV.FRONT], dtype=object)
    assert RP.intervals(views) == [(0, 2, IV.PROFILE), (3, 4, RP.UNKNOWN), (5, 5, IV.FRONT)]
    RP.save_views("abc", "c.mp4", 10.0, views, "notă")
    data = RP.load_views("abc")
    assert RP.load_views("altul") is None and data["clip"] == "c.mp4"
    rows = RP.view_rows(data)
    assert rows[0] == {"de la (s)": 0.0, "până la (s)": 0.3, "cadre": "0–2", "vedere": "Profil",
                       "merge la": "măsurarea de profil"}
    assert rows[1]["merge la"] == "nicăieri" and rows[2]["merge la"] == "modulul față/spate"
    assert RP.view_at(data, 4) == RP.VIEW_NAMES[RP.UNKNOWN] and RP.view_at(data, 99) is None


@pytest.mark.skipif(not M.dev_videos(), reason="clipurile dev nu sunt locale")
def test_the_analysis_page_shows_the_pilot_router_only_when_ticked(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    box = at.checkbox(key="an-router-pilot")
    assert box.value is False and not at.exception
    assert box.label == "Router vedere (pilot — nu atinge pragurile stabilite)"
    name = at.selectbox(key="an-clip").value
    box.check().run()
    assert any("doar pentru cele 5 clipuri" in i.value for i in at.info)                   # fără vederi pentru el
    views = np.array([IV.PROFILE] * 5 + [IV.BACK] * 5, dtype=object)
    RP.save_views(M.file_hash(M.V.VIDEOS / name), name, 30.0, views, "test")
    at.run()
    assert any("Vederea la cadrul 0: Profil" in m.value for m in at.markdown) and len(at.dataframe) == 1
