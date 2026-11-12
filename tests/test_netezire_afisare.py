"""A2 doar pentru afișare (filmul și graficul): Butterworth de ordin 4, fără întârziere, 6 Hz; segmente tăiate la
goluri, la schimbarea sursei și a orientării; segmentele scurte rămân nenetezite. Valorile, medianele și JSON-ul nu
se schimbă."""
import copy

import numpy as np
import pytest

from ecvestru import masurare as M
from ecvestru import suprapunere as S
from ecvestru.rezultat import FrameMeasure

# Referința: scipy 1.18.1, scipy.signal.butter(4, 6 / (30 / 2)) și filtfilt (padlen implicit 15), calculate o dată
# într-un mediu temporar; scipy nu e dependență a proiectului.
REF_B = [0.046582906636443676, 0.1863316265457747, 0.27949743981866204, 0.1863316265457747, 0.046582906636443676]
REF_A = [1.0, -0.7820951980233375, 0.6799785269162995, -0.18267569775303227, 0.030118875043169235]
REF_X = [0.0, 6.942, 10.854, 10.132, 5.35, -1.008, -5.716, -6.325, -2.313, 4.668, 11.57, 15.382, 14.546, 9.691, 3.335,
         -1.297, -1.792, 2.319, 9.336, 16.196]
REF_Y = [-0.001535, 6.915456, 10.814171, 10.09559, 5.335737, -0.989976, -5.675567, -6.288588, -2.302315, 4.6602,
         11.568434, 15.381622, 14.496482, 9.57839, 3.285501, -1.111876, -1.478288, 2.263788, 8.689328, 16.20118]
# fps 25, primele 15 eșantioane, padlen 14 (cel mai mare permis de scipy pe 15 eșantioane)
REF_Y15 = [0.00354, 6.940433, 10.847172, 10.123457, 5.349009, -0.997463, -5.710602, -6.338566, -2.316012, 4.708308,
           11.588615, 15.290749, 14.483475, 9.892073, 3.332058]


def test_the_constants_are_the_ones_asked_for():
    assert (S.SMOOTH_ORDER, S.SMOOTH_CUTOFF_HZ, S.SMOOTH_MIN_FRAMES) == (4, 6.0, 15)


def test_the_filter_has_the_butterworth_coefficients():
    b, a = S.butter_lowpass(S.SMOOTH_CUTOFF_HZ, 30.0)
    assert np.allclose(b, REF_B, atol=1e-12) and np.allclose(a, REF_A, atol=1e-12)


def test_the_zero_phase_filter_gives_what_scipy_gives():
    b, a = S.butter_lowpass(S.SMOOTH_CUTOFF_HZ, 30.0)
    assert np.allclose(S.zero_phase(b, a, np.array(REF_X)), REF_Y, atol=1e-5)
    b, a = S.butter_lowpass(S.SMOOTH_CUTOFF_HZ, 25.0)
    assert np.allclose(S.zero_phase(b, a, np.array(REF_X[:15])), REF_Y15, atol=1e-5)


def test_a_slow_movement_stays_and_fast_jitter_goes_without_delay():
    t = np.arange(90) / 30.0
    slow, fast = np.sin(2 * np.pi * 1.0 * t), 0.5 * np.sin(2 * np.pi * 12.0 * t)
    b, a = S.butter_lowpass(S.SMOOTH_CUTOFF_HZ, 30.0)
    out = S.zero_phase(b, a, slow + fast)
    middle = slice(20, 70)
    assert np.max(np.abs(out[middle] - slow[middle])) < 0.05
    assert np.argmax(out[20:45]) == np.argmax(slow[20:45])        # fără întârziere: vârful rămâne pe același cadru


def test_without_room_under_nyquist_there_is_no_filter():
    assert S.butter_lowpass(S.SMOOTH_CUTOFF_HZ, 12.0) is None


# ---------- segmentele ----------

def test_segments_are_cut_at_gaps_and_at_any_change_of_the_key():
    keys = [("a", 1), ("a", 1), None, ("a", 1), ("b", 1), ("b", 1), ("b", -1)]
    assert S.display_segments(keys) == [(0, 1), (3, 3), (4, 5), (6, 6)]


def wavy(n, seed=1):
    rng = np.random.default_rng(seed)
    return np.stack([np.linspace(0, 50, n) + rng.normal(0, 3, n), 100 + rng.normal(0, 3, n)], axis=1)


def test_long_segments_are_smoothed_and_short_ones_stay():
    n = 40
    values = wavy(n)
    keys = [("x",)] * 30 + [None] + [("x",)] * 9
    out = S.smooth_track(values, keys, 30.0)
    assert not np.allclose(out[:30], values[:30]) and np.allclose(out[31:], values[31:])
    assert np.isnan(out[30]).all() or np.allclose(out[30], values[30])
    assert np.std(np.diff(out[:30, 1])) < np.std(np.diff(values[:30, 1]))


def test_a_change_of_source_cuts_the_segment():
    n = 30
    values = wavy(n)
    keys = [("x",)] * 15 + [("y",)] * 15
    joined = S.smooth_track(values, [("x",)] * n, 30.0)
    cut = S.smooth_track(values, keys, 30.0)
    assert np.allclose(cut[:15], S.smooth_track(values[:15], keys[:15], 30.0))
    assert not np.allclose(cut, joined)


# ---------- brutul și unghiurile pentru afișare ----------

def raw_and_frames(n=30, facing=1.0, seed=2):
    rng = np.random.default_rng(seed)
    raw, frames = {}, []
    for i in range(n):
        shoulder = (60.0 + rng.normal(0, 2), 10.0 + rng.normal(0, 2))
        hip = (50.0 + rng.normal(0, 2), 30.0 + rng.normal(0, 2))
        heel = (50.0 + rng.normal(0, 2), 70.0 + rng.normal(0, 2))
        raw[i] = {"points": {"shoulder": shoulder, "hip": hip, "heel": heel, "facing": facing}, "motiv": None}
        frames.append(FrameMeasure(frame=i, trunk_deg=M.angle_from_vertical(shoulder, hip, facing),
                                   leg_deg=M.angle_from_vertical(hip, heel, facing)))
    return raw, frames


def test_display_keeps_the_raw_points_and_the_frames_untouched():
    raw, frames = raw_and_frames()
    before_raw, before_frames = copy.deepcopy(raw), [f.model_copy() for f in frames]
    shown = S.smooth_display(raw, frames, 30.0)
    assert raw == before_raw and frames == before_frames
    assert shown.raw[5]["points"]["hip"] != raw[5]["points"]["hip"]
    assert shown.raw[5]["points"]["facing"] == raw[5]["points"]["facing"] and shown.raw[5]["motiv"] is None


def test_the_display_angles_come_from_the_smoothed_points():
    raw, frames = raw_and_frames()
    shown = S.smooth_display(raw, frames, 30.0)
    p = shown.raw[10]["points"]
    assert shown.trunk[10] == pytest.approx(M.angle_from_vertical(p["shoulder"], p["hip"], 1.0))
    assert shown.leg[10] == pytest.approx(M.angle_from_vertical(p["hip"], p["heel"], 1.0))
    assert np.std(np.diff(shown.trunk)) < np.std(np.diff([f.trunk_deg for f in frames]))


def test_unmeasured_frames_stay_gaps_and_short_pieces_keep_their_values():
    raw, frames = raw_and_frames(40)
    frames[30] = frames[30].model_copy(update={"trunk_deg": None, "leg_deg": None, "reason": "calaret_negasit"})
    frames[35] = frames[35].model_copy(update={"leg_deg": None, "reason": M.REASON_HEEL_ON_HORSE})
    shown = S.smooth_display(raw, frames, 30.0)
    assert shown.trunk[30] is None and shown.leg[30] is None and shown.leg[35] is None
    assert shown.trunk[31:] == [f.trunk_deg for f in frames[31:]]               # 9 cadre: prea puține
    assert shown.raw[33]["points"] == raw[33]["points"]


def test_a_change_of_facing_or_of_the_heel_source_cuts_the_segment():
    raw, frames = raw_and_frames(30)
    for i in range(15, 30):
        frames[i] = frames[i].model_copy(update={"facing": -1.0, "facing_source": "laba",
                                                 "trunk_deg": -frames[i].trunk_deg, "leg_deg": -frames[i].leg_deg})
    shown = S.smooth_display(raw, frames, 30.0)
    alone = S.smooth_display({i: raw[i] for i in range(15)}, frames[:15], 30.0)
    assert shown.trunk[:15] == pytest.approx(alone.trunk)
    assert shown.trunk[20] < 0                                                  # orientarea cadrului, nu cea din brut
    sources = {i: "rtmpose" if i < 15 else "original" for i in range(30)}
    by_source = S.smooth_display(raw, raw_and_frames(30)[1], 30.0, heel_sources=sources)
    first = S.smooth_display({i: raw[i] for i in range(15)}, raw_and_frames(15)[1], 30.0)
    heels = lambda shown: [shown.raw[i]["points"]["heel"] for i in range(15)]  # noqa: E731
    assert np.allclose(heels(by_source), heels(first))   # călcâiul: doar segmentul lui; șoldul: tot trunchiul


def test_a_change_of_the_mediapipe_pass_cuts_the_segment():
    raw, frames = raw_and_frames(30)
    source = np.array([1] * 15 + [2] * 15)
    shown = S.smooth_display(raw, frames, 30.0, skeleton_source=source)
    alone = S.smooth_display({i: raw[i] for i in range(15)}, frames[:15], 30.0)
    assert shown.trunk[:15] == pytest.approx(alone.trunk)


# ---------- filmul ----------

def _film_setup(monkeypatch, tmp_path):
    from test_rtmpose import setup
    predictor = setup(monkeypatch, tmp_path)
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor)
    seen = []
    monkeypatch.setattr(S, "film_bytes", lambda video, frames, raw, fps, blur=False, skeleton=None:
                        seen.append(raw) or b"film")
    return res, seen


def test_the_film_draws_the_smoothed_chain_only_when_asked(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    res, seen = _film_setup(monkeypatch, tmp_path)
    before = res.to_json()
    S.clip_film(tmp_path / "a.mp4", res)
    S.clip_film(tmp_path / "a.mp4", res, smooth=True)
    plain, smooth = seen
    assert plain == M.load_raw(tmp_path / "a.mp4", res.video, res.video_hash, res.candidate)
    assert set(smooth) == set(plain) and res.to_json() == before


def test_display_series_returns_the_measured_values_when_smoothing_is_off(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    res, _ = _film_setup(monkeypatch, tmp_path)
    trunk, leg = S.display_series(tmp_path / "a.mp4", res, smooth=False)
    assert trunk == [f.trunk_deg for f in res.frames] and leg == [f.leg_deg for f in res.frames]
    trunk, leg = S.display_series(tmp_path / "a.mp4", res, smooth=True)
    assert len(trunk) == len(res.frames) and [v is None for v in leg] == [f.leg_deg is None for f in res.frames]


def test_smoothing_works_in_the_portable_copy_without_the_rtmpose_module(monkeypatch, tmp_path):
    import sys
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    res, seen = _film_setup(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "ecvestru.rtmpose", None)       # copia portabilă nu are ecvestru/rtmpose.py
    trunk, _ = S.display_series(tmp_path / "a.mp4", res, smooth=True)
    S.clip_film(tmp_path / "a.mp4", res, smooth=True)
    assert len(trunk) == len(res.frames) and len(seen) == 1
