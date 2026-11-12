"""Routerul de vedere: semnalele pe cadru, clasificatorul mic și votul pe ~1 s. Doar date sintetice:
nicio etichetă reală și niciun clip nou nu intră în teste."""
from types import SimpleNamespace

import numpy as np
import pytest

from ecvestru import cal_puncte as CP
from ecvestru import intervale as IV
from ecvestru import router_vedere as RV

N_POINTS = 33
FPS = 30.0


def rider(n: int, shoulder_dx: float, elbow_dx: float, yaw_deg: float, torso: float = 100.0):
    """Un schelet sintetic: umerii și coatele la distanța dată pe x (stânga − dreapta), trunchiul vertical."""
    xy = np.zeros((n, N_POINTS, 2))
    xy[:, RV.LEFT_SHOULDER] = (200 + shoulder_dx / 2, 100)
    xy[:, RV.RIGHT_SHOULDER] = (200 - shoulder_dx / 2, 100)
    xy[:, RV.LEFT_ELBOW] = (200 + elbow_dx / 2, 150)
    xy[:, RV.RIGHT_ELBOW] = (200 - elbow_dx / 2, 150)
    xy[:, RV.LEFT_HIP] = (200, 100 + torso)
    xy[:, RV.RIGHT_HIP] = (200, 100 + torso)
    world = np.zeros((n, N_POINTS, 3))
    yaw = np.radians(yaw_deg)
    world[:, RV.RIGHT_SHOULDER] = (0.2 * np.cos(yaw), 0, 0.2 * np.sin(yaw))
    return xy, world


# ---------- semnalele călărețului ----------

def test_the_shoulder_ratio_is_the_shoulder_width_over_the_torso():
    xy, world = rider(4, shoulder_dx=50, elbow_dx=80, yaw_deg=0)
    s = RV.rider_signals(xy, world)
    assert np.allclose(s["raport_umeri"], 0.5)


def test_the_left_right_signs_are_signed_and_relative_to_the_torso():
    xy, world = rider(3, shoulder_dx=-30, elbow_dx=60, yaw_deg=0)
    s = RV.rider_signals(xy, world)
    assert np.allclose(s["umeri_stanga_dreapta"], -0.3) and np.allclose(s["coate_stanga_dreapta"], 0.6)


def test_the_3d_orientation_is_the_yaw_of_the_shoulder_line_as_cosine_and_sine():
    xy, world = rider(2, shoulder_dx=10, elbow_dx=10, yaw_deg=90)
    s = RV.rider_signals(xy, world)
    assert np.allclose(s["orientare_cos"], 0, atol=1e-9) and np.allclose(s["orientare_sin"], 1)


def test_missing_rider_points_give_nan_not_zero():
    xy, world = rider(2, shoulder_dx=10, elbow_dx=10, yaw_deg=0)
    xy[1] = np.nan
    world[1] = np.nan
    s = RV.rider_signals(xy, world)
    assert all(np.isnan(v[1]) for v in s.values()) and not any(np.isnan(v[0]) for v in s.values())


# ---------- semnalele calului ----------

def horse_run(n: int, head=(300.0, 100.0), tail=(100.0, 100.0), box=(50.0, 50.0, 350.0, 250.0), with_box=None):
    points = np.full((n, len(CP.AP10K), 2), np.nan)
    scores = np.zeros((n, len(CP.AP10K)))
    nose, tail_i = CP.AP10K.index("nose"), CP.AP10K.index("root_of_tail")
    points[:, nose], points[:, tail_i] = head, tail
    scores[:, nose] = scores[:, tail_i] = 0.9
    boxes = np.tile(np.array(box, float), (n, 1))
    if with_box is not None:
        points[~with_box], scores[~with_box], boxes[~with_box] = np.nan, np.nan, np.nan
    return CP.ClipRun(points, scores, boxes, 0.0)


def test_the_horse_signals_are_the_head_tail_length_its_horizontality_and_the_box_shape():
    s = RV.horse_signals(horse_run(3))
    assert np.allclose(s["cal_lungime"], 1.0)          # 200 px / caseta de 200 px înălțime
    assert np.allclose(s["cal_orizontal"], 1.0)        # coadă → cap pe orizontală
    assert np.allclose(s["cal_caseta"], 1.5)           # 300 × 200


def test_a_horse_seen_head_on_is_short_and_not_horizontal():
    s = RV.horse_signals(horse_run(1, head=(200.0, 60.0), tail=(200.0, 100.0)))
    assert s["cal_lungime"][0] == pytest.approx(0.2) and s["cal_orizontal"][0] == pytest.approx(0.0, abs=1e-9)


# ---------- semnalele pe cadrele clipului ----------

def test_the_frame_signals_are_aligned_on_the_video_frames_with_nan_where_missing():
    xy, world = rider(2, shoulder_dx=50, elbow_dx=50, yaw_deg=0)
    skeleton = SimpleNamespace(frame=np.array([1, 3]), xy_px=xy, world_xyz=world)
    signals = RV.frame_signals(skeleton, horse_run(5), n_frames=5)
    assert signals.values.shape == (5, len(RV.SIGNALS)) and tuple(signals.names) == RV.SIGNALS
    ratio = signals.values[:, RV.SIGNALS.index("raport_umeri")]
    assert np.isnan(ratio[[0, 2, 4]]).all() and np.allclose(ratio[[1, 3]], 0.5)
    assert np.allclose(signals.values[:, RV.SIGNALS.index("cal_caseta")], 1.5)


def test_without_skeleton_or_horse_the_signals_are_all_nan():
    signals = RV.frame_signals(None, None, n_frames=4)
    assert np.isnan(signals.values).all()


def test_the_signals_survive_a_round_trip_through_the_cache(tmp_path):
    xy, world = rider(3, shoulder_dx=50, elbow_dx=50, yaw_deg=0)
    signals = RV.frame_signals(SimpleNamespace(frame=np.arange(3), xy_px=xy, world_xyz=world), horse_run(3), 3)
    RV.save_signals(tmp_path / "s.npz", signals)
    back = RV.load_signals(tmp_path / "s.npz")
    assert back.names == signals.names and np.array_equal(back.values, signals.values, equal_nan=True)


# ---------- clasificatorul ----------

def separable(n_per_class: int = 60, seed: int = 0):
    """Trei vederi bine separate pe două semnale, plus un semnal lipsă pe o parte din cadre."""
    rng = np.random.default_rng(seed)
    centers = {IV.PROFILE: (0.1, 0.0), IV.FRONT: (0.9, 1.0), IV.BACK: (0.9, -1.0)}
    X, y = [], []
    for view, (a, b) in centers.items():
        X.append(np.column_stack([a + 0.05 * rng.standard_normal(n_per_class),
                                  b + 0.05 * rng.standard_normal(n_per_class),
                                  np.where(rng.random(n_per_class) < 0.3, np.nan, rng.random(n_per_class))]))
        y += [view] * n_per_class
    return np.vstack(X), np.array(y)


def test_the_logistic_classifier_learns_separable_views():
    X, y = separable()
    model = RV.make_classifier(RV.LOGISTIC).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.98


def test_the_probabilities_sum_to_one_and_follow_the_classes():
    X, y = separable()
    model = RV.make_classifier(RV.LOGISTIC).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(X), 3) and np.allclose(p.sum(axis=1), 1)
    assert set(model.classes) == {IV.PROFILE, IV.FRONT, IV.BACK}


def test_nan_inputs_are_imputed_and_never_break_the_prediction():
    X, y = separable()
    model = RV.make_classifier(RV.LOGISTIC).fit(X, y)
    assert len(model.predict(np.full((2, 3), np.nan))) == 2


def test_the_classifier_type_is_a_parameter_and_an_unknown_type_is_refused():
    with pytest.raises(ValueError, match="necunoscut"):
        RV.make_classifier("padure")


def test_the_classifier_is_deterministic_and_survives_a_round_trip_through_json():
    X, y = separable()
    a = RV.make_classifier(RV.LOGISTIC).fit(X, y)
    b = RV.make_classifier(RV.LOGISTIC).fit(X, y)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
    back = RV.classifier_from_dict(a.to_dict())
    assert np.allclose(back.predict_proba(X), a.predict_proba(X))


def test_fitting_needs_at_least_two_classes():
    with pytest.raises(ValueError, match="două vederi"):
        RV.make_classifier(RV.LOGISTIC).fit(np.zeros((4, 2)), np.array([IV.PROFILE] * 4))


# ---------- votul pe ~1 s ----------

def test_a_short_blip_is_voted_away():
    views = np.array([IV.PROFILE] * 30 + [IV.FRONT] * 3 + [IV.PROFILE] * 30, dtype=object)
    voted = RV.vote(views, FPS)
    assert (voted == IV.PROFILE).all()


def test_without_a_clear_majority_the_frame_stays_undecided():
    views = np.array([IV.PROFILE, IV.FRONT] * 30, dtype=object)
    voted = RV.vote(views, FPS)
    assert (voted[10:-10] == RV.UNDECIDED).all()


def test_frames_without_a_prediction_do_not_vote_and_too_few_votes_stay_undecided():
    views = np.array([""] * 40 + [IV.BACK] * 2 + [""] * 40, dtype=object)
    voted = RV.vote(views, FPS)
    assert (voted == RV.UNDECIDED).all()


def test_the_vote_window_is_about_one_second():
    assert RV.VOTE_SECONDS == pytest.approx(1.0)
    views = np.array([IV.PROFILE] * 60 + [IV.BACK] * 60, dtype=object)
    voted = RV.vote(views, FPS)
    assert (voted[:50] == IV.PROFILE).all() and (voted[70:] == IV.BACK).all()


# ---------- traseul: profil, față/spate sau nicăieri ----------

def test_only_profile_goes_to_the_profile_and_only_front_or_back_to_the_front_back_module():
    views = np.array([IV.PROFILE, IV.FRONT, IV.BACK, IV.OBLIQUE, IV.HIDDEN, RV.UNDECIDED], dtype=object)
    assert list(RV.routes(views)) == [RV.ROUTE_PROFILE, RV.ROUTE_FRONT_BACK, RV.ROUTE_FRONT_BACK,
                                      RV.ROUTE_NONE, RV.ROUTE_NONE, RV.ROUTE_NONE]


def test_the_router_layer_is_off_by_default():
    assert RV.ROUTER_LAYER_DEFAULT is False


def test_predicting_a_clip_classifies_votes_and_skips_frames_without_any_signal():
    X, y = separable()
    model = RV.make_classifier(RV.LOGISTIC).fit(X, y)
    clip = np.vstack([np.tile([0.1, 0.0, 0.5], (40, 1)), np.full((5, 3), np.nan)])
    voted = RV.predict_clip(model, clip, FPS)
    assert (voted[:30] == IV.PROFILE).all() and len(voted) == 45
