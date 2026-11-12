"""Regula B4 în stratul RTMPose: orientarea după direcția labei, votul pe
cadrele vecine, doar în intervalul analizat; semnalul „profil neclar”; JSON 1.3; calea implicită neschimbată."""
import json

import numpy as np
import pytest
from test_rtmpose import FakeEstimator, setup

from ecvestru import masurare as M
from ecvestru import rtmpose as R
from ecvestru import schelet as Z
from ecvestru.rezultat import FACING_FOOT, FACING_NOSE_EAR, AnalysisResult

FPS = 30.0
NAN = np.nan


@pytest.fixture
def tmp_results(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    return tmp_path / "analiza"


# ---------- constantele regulii ----------

def test_the_constants_of_the_rule():
    assert (R.VOTES, R.MAX_SECONDS, R.MIN_VOTES, R.SHARE) == (9, 0.5, 5, 2 / 3)
    assert R.PROFILE_UNCLEAR_SHARE == 0.05


# ---------- votul ----------

def test_the_vote_flips_the_facing_when_enough_neighbours_agree():
    current = np.ones(20)
    foot = -np.ones(20)
    facing, decided = R.vote_facing(current, foot, FPS)
    assert (facing == -1).all() and decided.all()


def test_the_vote_keeps_the_current_facing_with_fewer_than_five_votes():
    current = np.ones(20)
    foot = np.full(20, NAN)
    foot[[0, 1, 2, 3]] = -1
    facing, decided = R.vote_facing(current, foot, FPS)
    assert (facing == 1).all() and not decided.any()


def test_the_vote_needs_two_thirds():
    current = np.ones(9)
    foot = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1], float)      # 5 din 9: sub 2/3
    facing, decided = R.vote_facing(current, foot, FPS)
    assert (facing == 1).all() and not decided.any()
    foot[5] = -1                                                    # 6 din 9 = 2/3
    facing, decided = R.vote_facing(current, foot, FPS)
    assert (facing == -1).all() and decided.all()


def test_the_vote_uses_the_nine_closest_frames_within_half_a_second():
    current = np.ones(60)
    foot = np.full(60, NAN)
    foot[16:30] = -1             # cadrul 0 ajunge doar până la 15 (0,5 s la 30 cadre/s): niciun vot
    facing, decided = R.vote_facing(current, foot, FPS)
    assert facing[0] == 1 and not decided[0]
    assert facing[23] == -1 and decided[23]
    foot[:] = 1
    foot[[30, 31, 32, 33]] = -1  # pe cadrul 30, cele 9 mai apropiate: 4 „−1” și 5 „+1”, sub 2/3 pentru oricare
    facing, decided = R.vote_facing(np.full(60, -1.0), foot, FPS)
    assert facing[30] == -1 and not decided[30]


def test_frames_without_a_facing_stay_without_one():
    current = np.array([NAN] + [1.0] * 9)
    facing, decided = R.vote_facing(current, -np.ones(10), FPS)
    assert np.isnan(facing[0]) and not decided[0] and (facing[1:] == -1).all()


def test_the_minority_share_counts_only_frames_with_a_facing():
    assert R.minority_share(np.array([1, 1, 1, -1, NAN])) == 0.25
    assert R.minority_share(np.array([-1.0, -1.0])) == 0.0
    assert np.isnan(R.minority_share(np.array([NAN])))


# ---------- direcția labei ----------

def sk_with_feet(n, mp_dx=0.0, side="left"):
    """Scheletul de test, cu vârful labei MediaPipe deplasat cu `mp_dx` pixeli față de călcâi."""
    from test_calitate import W, skeleton
    sk = skeleton(n, side=side)
    for s in ("left", "right"):
        sk.xy_norm[:, Z.LANDMARK_NAMES.index(f"{s}_foot_index"), 0] = 0.5 + mp_dx / W
        sk.xy_norm[:, Z.LANDMARK_NAMES.index(f"{s}_foot_index"), 1] = 0.72
    return sk


def raw_for(n, facing=1.0, hip=(50.0, 30.0)):
    return {i: {"points": {"shoulder": (60.0, 10.0), "hip": hip, "heel": (50.0, 72.0), "facing": facing},
                "motiv": None} for i in range(n)}


def rtm_points(n, toe_dx):
    points = np.full((n, 26, 2), NAN)
    for s in ("left", "right"):
        points[:, R.HEEL_INDEX[s]] = (50.0, 72.0)
        for toe in R.TOE_INDEX[s]:
            points[:, toe] = (50.0 + toe_dx, 72.0)
    return points


def test_the_foot_direction_comes_from_rtmpose_first():
    sk = sk_with_feet(4, mp_dx=+10)
    foot = R.foot_direction(raw_for(4), sk, rtm_points(4, -10))
    assert (foot == -1).all()


def test_mediapipe_fills_in_where_rtmpose_is_missing():
    sk = sk_with_feet(4, mp_dx=+10)
    points = rtm_points(4, -10)
    points[2] = NAN
    foot = R.foot_direction(raw_for(4), sk, points)
    assert foot.tolist() == [-1, -1, 1, -1]


def test_a_flat_foot_or_a_frame_without_a_measure_has_no_direction():
    sk = sk_with_feet(3, mp_dx=0.0)
    raw = raw_for(3)
    raw[1] = {"points": None, "motiv": "calaret_negasit"}
    points = rtm_points(3, -10)
    points[2] = NAN                     # RTMPose lipsă, iar laba MediaPipe e orizontală pe x: nicio direcție
    foot = R.foot_direction(raw, sk, points)
    assert foot[0] == -1 and np.isnan(foot[1]) and np.isnan(foot[2])


def test_the_mediapipe_foot_is_read_on_the_side_of_the_measured_hip():
    sk = sk_with_feet(2, mp_dx=+10, side="left")
    sk.xy_norm[:, Z.LANDMARK_NAMES.index("right_foot_index"), 0] = 0.4           # dreapta: vârful în stânga
    sk.xy_norm[:, Z.LANDMARK_NAMES.index("right_hip"), 0] = 0.7
    raw = raw_for(2, hip=(70.0, 30.0))                                            # șoldul măsurat e cel drept
    foot = R.foot_direction(raw, sk, rtm_points(2, NAN))
    assert (foot == -1).all()


# ---------- orientarea în brut, doar în interval ----------

def test_orient_changes_the_facing_only_inside_the_segment_and_keeps_the_original_raw():
    n = 30
    sk = sk_with_feet(n)
    raw = raw_for(n, facing=1.0)
    oriented, summary, sources = R.orient(raw, sk, rtm_points(n, -10), FPS, (10, 19))
    assert all(oriented[i]["points"]["facing"] == -1.0 for i in range(10, 20))
    assert all(oriented[i]["points"]["facing"] == 1.0 for i in [*range(10), *range(20, n)])
    assert all(raw[i]["points"]["facing"] == 1.0 for i in range(n))
    assert summary.changed_frames == 10 and summary.foot_frames == 10 and summary.frames_with_facing == 10
    assert sources[10] == FACING_FOOT and 9 not in sources


def test_a_clear_profile_keeps_the_report():
    n = 30
    oriented, summary, _ = R.orient(raw_for(n), sk_with_feet(n), rtm_points(n, -10), FPS, (0, n - 1))
    assert summary.minority_share == 0 and summary.minority_pct == 0
    assert not summary.profile_unclear and summary.reason is None


def test_more_than_five_percent_minority_marks_the_profile_unclear():
    n = 40
    points = rtm_points(n, +10)
    points[:6] = rtm_points(6, -10)     # cadrele 0–5 votează −1, restul +1: primele cadre se schimbă
    oriented, summary, sources = R.orient(raw_for(n), sk_with_feet(n), points, FPS, (0, n - 1))
    minority = R.minority_share(np.array([oriented[i]["points"]["facing"] for i in range(n)]))
    assert summary.minority_share == pytest.approx(minority) and minority > R.PROFILE_UNCLEAR_SHARE
    assert summary.profile_unclear and summary.minority_pct == round(100 * minority)
    assert summary.reason == f"profil neclar: orientarea se schimbă pe {summary.minority_pct}% din cadre"


def test_exactly_five_percent_is_not_unclear():
    facing = np.array([-1.0] + [1.0] * 19)
    assert R.minority_share(facing) == 0.05 and not R.profile_unclear(0.05)
    assert R.profile_unclear(0.0501)


# ---------- în analiză: JSON 1.3 ----------

class TurnedEstimator(FakeEstimator):
    """Călcâiul ca în FakeEstimator, iar vârfurile labei la stânga lui: laba arată spre −1."""

    def __call__(self, image, bbox):
        points, scores = super().__call__(image, bbox)
        for s in ("left", "right"):
            for toe in R.TOE_INDEX[s]:
                points[toe] = (self.heel[0] - 20.0, self.heel[1])
        return points, scores


def leaning(monkeypatch, tmp_path):
    """Ca setup din test_rtmpose, dar cu umărul în fața șoldului: trunchiul are semn, deci orientarea se vede."""
    predictor = setup(monkeypatch, tmp_path)
    point = {"shoulder": (60.0, 10.0), "hip": (50.0, 30.0), "heel": (50.0, 95.0), "facing": 1.0}
    return lambda path: {i: {"points": point, "motiv": None} for i in range(12)}, predictor


def test_with_rtmpose_the_rule_flips_the_angles_and_the_json_says_why(tmp_results, tmp_path, monkeypatch):
    predictor, _ = leaning(monkeypatch, tmp_path)
    default = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor)
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source="rtmpose",
                    heel_estimator=TurnedEstimator(heel=(50.0, 50.0)))
    assert default.trunk.median_deg > 0 and res.trunk.median_deg == pytest.approx(-default.trunk.median_deg)
    data = json.loads(res.to_json())
    assert data["schema_version"] == "1.4"
    assert data["orientation"]["changed_frames"] == 12 and data["orientation"]["profile_unclear"] is False
    assert {(f["facing"], f["facing_source"]) for f in data["frames"]} == {(-1.0, FACING_FOOT)}
    assert AnalysisResult.from_json(res.to_json()) == res


def test_with_rtmpose_frames_the_rule_did_not_decide_keep_the_nose_ear_source(tmp_results, tmp_path, monkeypatch):
    predictor, _ = leaning(monkeypatch, tmp_path)
    flat = FakeEstimator(heel=(50.0, 50.0))       # vârfurile la (0, 0), în stânga călcâiului: laba spre −1
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 3), predictor=predictor, heel_source="rtmpose",
                    heel_estimator=flat)
    assert res.orientation is not None and all(f.facing_source == FACING_FOOT for f in res.frames)
    monkeypatch.setattr(R, "MIN_VOTES", 50)
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "alt"))
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 3), predictor=predictor, heel_source="rtmpose",
                    heel_estimator=flat)
    assert all(f.facing == 1.0 and f.facing_source == FACING_NOSE_EAR for f in res.frames)
    assert res.orientation.changed_frames == 0


DEFAULT_KEYS_1_2 = {"schema_version", "video", "video_hash", "gait", "fps", "n_frames", "segment", "candidate",
                    "trunk", "leg", "unmeasured", "frames", "limitation", "quality", "heel_source"}
FRAME_KEYS_1_2 = {"frame", "trunk_deg", "leg_deg", "reason", "warnings"}


def test_the_baseline_json_has_the_fields_of_1_2_plus_the_path_and_the_layer_versions(tmp_results, tmp_path,
                                                                                       monkeypatch):
    predictor, _ = leaning(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "load_points", lambda *a, **k: pytest.fail("RTMPose a rulat pe linia de bază"))
    data = json.loads(M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor).to_json())
    assert set(data) == DEFAULT_KEYS_1_2 | {"pipeline", "fallback_reason", "layers"}
    assert all(set(f) == FRAME_KEYS_1_2 for f in data["frames"])
    assert data["schema_version"] == "1.4"


def test_the_rtmpose_result_file_names_the_orientation_rule(tmp_results, tmp_path, monkeypatch):
    path = M.result_cache_path("a.mp4", "h", "c@1", (0, 9), "pas", heel_source="rtmpose")
    assert f"o{R.ORIENTATION_VERSION}" in path.name and "__v1.4" in path.name
    assert f"o{R.ORIENTATION_VERSION}" not in M.result_cache_path("a.mp4", "h", "c@1", (0, 9), "pas").name


def test_the_film_shows_the_angles_after_the_rule(tmp_results, tmp_path, monkeypatch):
    from ecvestru import suprapunere as S
    predictor, _ = leaning(monkeypatch, tmp_path)
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor, heel_source="rtmpose",
                    heel_estimator=TurnedEstimator(heel=(50.0, 50.0)))
    seen = {}
    monkeypatch.setattr(S, "film_bytes", lambda video, frames, raw, fps, blur=False, skeleton=None:
                        seen.update(frames=frames) or b"film")
    S.clip_film(tmp_path / "a.mp4", res)
    assert seen["frames"] == res.frames and res.frames[0].trunk_deg < 0
    assert S.band_text(res.frames[0]).startswith("#0 · trunchi -")


# ---------- pe clipurile dev ----------

USABLE_CHANGED = {"clip_150957.mp4": 3, "clip_151155.mp4": 3, "clip_152429.mp4": 4}
UNUSABLE = ("clip_150110.mp4", "clip_150202.mp4", "clip_150307.mp4", "clip_150415.mp4", "clip_151902.mp4",
            "clip_152334.mp4")


def _dev_ready() -> bool:
    if not R.runtime_available() or not M.dev_videos():
        return False
    for video in M.dev_videos():
        path = M.V.VIDEOS / video
        if not R.points_cache_path(video, M.file_hash(path)).exists():
            return False
    return True


@pytest.mark.skipif(not _dev_ready(), reason="lipsesc clipurile dev sau punctele RTMPose din cache")
def test_on_the_dev_clips_the_rule_gives_the_numbers_of_the_b4_results():
    from ecvestru import calitate as C
    changed, unclear = {}, {}
    for video in M.dev_videos():
        clip = C.load_clip(video)
        path = M.V.VIDEOS / video
        points = R.load_points(path, video, M.file_hash(path), clip.skeleton)
        n_frames, fps = M.video_info(path)
        _, summary, _ = R.orient(clip.raw, clip.skeleton, points, fps, M.default_segment(video, n_frames))
        changed[video], unclear[video] = summary.changed_frames, summary.profile_unclear
    usable = [v for v in M.dev_videos() if v not in UNUSABLE]
    assert {v: changed[v] for v in usable if changed[v]} == USABLE_CHANGED
    assert not any(unclear[v] for v in usable) and all(unclear[v] for v in UNUSABLE)
    assert sum(changed[v] for v in UNUSABLE) == 323
