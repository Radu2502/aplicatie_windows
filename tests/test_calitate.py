"""Indicatorii de calitate, pe schelete și cadre construite de mână."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ecvestru import calitate as C
from ecvestru import schelet as Z

W, H = 100, 100


def skeleton(frames=6, sources=None, shin=None, side="left"):
    """Un om vertical: umăr y=10, șold y=30, genunchi y=50, gleznă y=70, călcâi y=72 (pixeli)."""
    n = frames
    xy = np.full((n, 33, 2), 0.5)
    ys = {"shoulder": 10, "hip": 30, "knee": 50, "ankle": 70, "heel": 72}
    for part, y in ys.items():
        for s in ("left", "right"):
            xy[:, Z.LANDMARK_NAMES.index(f"{s}_{part}"), 1] = y / H
    if shin is not None:
        for s in ("left", "right"):
            xy[:, Z.LANDMARK_NAMES.index(f"{s}_ankle"), 1] = (50 + np.asarray(shin)) / H
    rows = Z._empty_rows(n)
    rows["xy_norm"] = xy
    rows["visibility"][:] = 0.9
    rows["source"][:] = Z.SOURCE_FULL if sources is None else sources
    rows["side"][:] = side
    rows["size"] = (W, H)
    Z._add_rejected(rows, [[] for _ in range(n)])
    Z._mark_doubles(rows)
    return Z.make_skeleton(rows, np.arange(n) / 30, {"width": W, "height": H})


def clip_with(sk, frames, raw=None, boxes=None):
    return C.ClipData("a.mp4", sk, raw or {}, frames, boxes or {})


def measure(leg=-5.0, reason=None):
    return NS(leg_deg=leg, trunk_deg=1.0, reason=reason)


def points(heel=(50.0, 72.0)):
    return {"shoulder": (50.0, 10.0), "hip": (50.0, 30.0), "heel": heel, "facing": 1.0}


def test_side_part_follows_the_side_of_each_frame():
    sk = skeleton(2)
    sk.side[1] = "right"
    sk.xy_norm[1, Z.LANDMARK_NAMES.index("right_knee"), 0] = 0.9
    assert C.side_part(sk, "knee")[0, 0] == 50 and C.side_part(sk, "knee")[1, 0] == 90


def test_side_visibility_reads_the_chosen_side():
    sk = skeleton(1, side="right")
    sk.visibility[0, Z.LANDMARK_NAMES.index("right_heel")] = 0.2
    assert C.side_visibility(sk, "heel")[0] == pytest.approx(0.2)


def test_midpoint_and_segment():
    sk = skeleton(1)
    assert C.midpoint(sk, "hip")[0, 1] == 30 and C.segment(sk, "knee", "ankle")[0] == 20


@pytest.mark.parametrize("hip,found", [((50, 20), True), ((50, -10), False), ((50, 60), False), ((5, 20), False)])
def test_rider_box_uses_the_rule_of_the_candidate(hip, found):
    assert (C.rider_box(hip, [[20, 10, 60, 80]]) is not None) is found


def test_rider_box_prefers_the_largest():
    assert C.rider_box((50, 20), [[40, 10, 20, 20], [20, 10, 60, 80]]) == [20, 10, 60, 80]


@pytest.mark.parametrize("heel,low", [((50, 70), True), ((50, 69), False), ((5, 80), False)])
def test_heel_in_lower_zone(heel, low):
    assert C.heel_in_lower_zone(heel, [20, 10, 60, 90]) is low


def test_no_box_means_not_low():
    assert C.heel_in_lower_zone((1, 1), None) is False


def test_long_shin_is_relative_to_the_clip():
    assert list(C.long_shin(skeleton(5, shin=[20, 20, 20, 20, 30]))) == [False] * 4 + [True]


def test_indicator_a_needs_both_a_low_heel_and_a_long_shin():
    sk = skeleton(4, shin=[20, 20, 20, 40])
    raw = {i: {"points": points((50.0, 95.0))} for i in range(4)}
    clip = clip_with(sk, [measure()] * 4, raw, {i: [[0, 20, 100, 60]] for i in range(4)})
    assert list(C.indicator_a(clip)) == [False, False, False, True]


def test_unmeasured_legs_are_not_flagged():
    sk = skeleton(2, shin=[20, 40])
    raw = {i: {"points": points((50.0, 95.0))} for i in range(2)}
    clip = clip_with(sk, [measure(None), measure(None)], raw, {i: [[0, 20, 100, 60]] for i in range(2)})
    assert not C.indicator_a(clip).any()


def test_jitter_is_zero_on_a_still_rider():
    values = C.jitter(skeleton(6))
    assert values["hip"] == 0 and values["acc"] == 0


def test_jitter_is_in_torso_lengths():
    sk = skeleton(3)
    sk.xy_norm[1:, :, 0] += 0.02                     # 2 px, trunchi 20 px
    assert C.jitter(sk)["hip"] == pytest.approx(0.05)


def test_variation_of_equal_values_is_zero():
    assert C.variation(np.array([2.0, 2.0, np.nan])) == 0


def test_source_jumps_are_split_by_transition():
    sk = skeleton(3, sources=[1, 2, 2])
    sk.xy_norm[1:, :, 0] += 0.2                      # 20 px = un trunchi
    jumps = C.source_jumps(sk)
    assert jumps["1>2"][0] == pytest.approx(1.0) and jumps["2>2"][0] == 0 and len(jumps["1>1"]) == 0


def test_reason_counts():
    assert C.reason_counts(clip_with(skeleton(2), [measure(reason="x"), measure()]))["x"] == 1


def _row(x=50.0, verdict="corect"):
    return {"shoulder_x": "50", "shoulder_y": "10", "hip_x": "50", "hip_y": "30", "heel_x": str(x), "heel_y": "72",
            "verdict": verdict}


# ---------- semnale ----------

def test_bad_proportions_flags_a_segment_far_from_its_median():
    assert list(C.bad_proportions(skeleton(5, shin=[20, 20, 20, 20, 30]))) == [False] * 4 + [True]


def test_low_chain_visibility_looks_at_the_used_side_only():
    sk = skeleton(2, side="left")
    sk.visibility[0, Z.LANDMARK_NAMES.index("left_knee")] = 0.2
    sk.visibility[1, Z.LANDMARK_NAMES.index("right_knee")] = 0.2
    assert list(C.low_chain_visibility(sk)) == [True, False]


def test_a_sudden_jump_is_flagged_on_the_later_frame():
    sk = skeleton(3)
    sk.xy_norm[2, :, 0] += 0.2                       # un trunchi
    assert list(C.sudden_jump(sk)) == [False, False, True]


def test_a_small_jump_counts_only_at_a_change_of_source():
    plain, changed = skeleton(2), skeleton(2, sources=[1, 2])
    for sk in (plain, changed):
        sk.xy_norm[1, :, 0] += 0.06                  # 0,3 trunchiuri
    assert not C.sudden_jump(plain).any() and C.sudden_jump(changed)[1]


def test_an_isolated_side_differs_from_both_neighbours():
    sk = skeleton(5)
    sk.side[2] = "right"
    assert list(C.isolated_side(sk)) == [False, False, True, False, False]


def test_a_lasting_change_of_side_is_not_isolated():
    sk = skeleton(5)
    sk.side[2:] = "right"
    assert not C.isolated_side(sk).any()


def test_clean_frames_have_a_rider_and_no_signal():
    sk = skeleton(3)
    sk.source[2] = Z.SOURCE_NONE
    clip = clip_with(sk, [measure()] * 3)
    signals = {"x": np.array([True, False, False])}
    assert list(C.clean_frames(clip, signals)) == [False, True, False]


def test_the_frame_signals_have_a_text_each():
    clip = clip_with(skeleton(2), [measure(None)] * 2)
    assert set(C.frame_signals(clip)) == set(C.SIGNAL_TEXTS)


def test_frame_reasons_name_the_active_signals():
    assert C.frame_reasons({C.SIGNAL_JUMP: np.array([True])}, 0) == [C.SIGNAL_TEXTS[C.SIGNAL_JUMP]]


# ---------- poarta A ----------

def _real_clip(shin, leg=-5.0):
    from ecvestru.rezultat import FrameMeasure
    n = len(shin)
    sk = skeleton(n, shin=shin)
    raw = {i: {"points": points((50.0, 95.0)), "motiv": None} for i in range(n)}
    frames = [FrameMeasure(frame=i, trunk_deg=1.0, leg_deg=leg) for i in range(n)]
    return clip_with(sk, frames, raw, {i: [[0, 20, 100, 60]] for i in range(n)})


def test_the_gate_removes_only_the_leg_of_a_flagged_frame():
    gated = C.gate_frames(_real_clip([20, 20, 20, 40]))
    flagged, normal = gated[3], gated[0]
    assert flagged.leg_deg is None and flagged.trunk_deg == 1.0 and flagged.reason == C.RZ.REASON_HEEL_ON_HORSE
    assert normal.leg_deg == -5.0 and normal.reason is None


def test_the_reason_has_the_text_of_the_decision():
    assert C.RZ.REASON_TEXTS[C.RZ.REASON_HEEL_ON_HORSE] == "călcâi pe piciorul calului"


def test_other_signals_only_warn():
    clip = _real_clip([20, 20, 20, 20])
    sk = clip.skeleton
    sk.visibility[1, Z.LANDMARK_NAMES.index("left_knee")] = 0.2
    gated = C.gate_frames(clip)
    assert gated[1].leg_deg == -5.0 and gated[1].warnings == [C.SIGNAL_VISIBILITY]


def test_the_excluding_signal_is_not_also_a_warning():
    assert C.SIGNAL_LEG not in C.gate_frames(_real_clip([20, 20, 20, 40]))[3].warnings


def test_the_summary_counts_excluded_frames_and_warnings_with_their_version():
    from ecvestru.rezultat import QUALITY_VERSION
    clip = _real_clip([20, 20, 20, 40])
    clip.skeleton.visibility[1, Z.LANDMARK_NAMES.index("left_knee")] = 0.2
    summary = C.summarize_quality(C.gate_frames(clip))
    assert summary.version == QUALITY_VERSION and summary.excluded_frames == 1
    assert summary.excluded_reason == "călcâi pe piciorul calului"
    found = {w.code: w.frames for w in summary.warnings}
    assert found[C.SIGNAL_VISIBILITY] == 1 and all(w.text for w in summary.warnings)


# ---------- poarta B, partea 1: intervalul plauzibil al unghiului piciorului ----------

def _clip_with_legs(legs):
    from ecvestru.rezultat import FrameMeasure
    n = len(legs)
    sk = skeleton(n, shin=[20] * n)
    raw = {i: {"points": points((50.0, 95.0)), "motiv": None} for i in range(n)}
    frames = [FrameMeasure(frame=i, trunk_deg=1.0, leg_deg=leg) for i, leg in enumerate(legs)]
    return clip_with(sk, frames, raw, {i: [[0, 20, 100, 60]] for i in range(n)})


def test_the_plausible_interval_is_a_named_constant():
    assert (C.LEG_ANGLE_MIN, C.LEG_ANGLE_MAX) == (-50.0, 40.0)


def test_a_leg_angle_outside_the_interval_becomes_unmeasured_with_a_reason():
    gated = C.gate_frames(_clip_with_legs([-60.0, -50.0, -5.0, 40.0, 45.0, 90.0]))
    assert [f.leg_deg for f in gated] == [None, -50.0, -5.0, 40.0, None, None]
    assert [f.reason for f in gated] == [C.RZ.REASON_IMPLAUSIBLE_ANGLE, None, None, None,
                                         C.RZ.REASON_IMPLAUSIBLE_ANGLE, C.RZ.REASON_IMPLAUSIBLE_ANGLE]
    assert all(f.trunk_deg == 1.0 for f in gated)


def test_the_implausible_angle_reason_has_a_text():
    assert C.RZ.REASON_TEXTS[C.RZ.REASON_IMPLAUSIBLE_ANGLE] == "unghi neplauzibil"


def test_a_missing_leg_stays_as_it_was():
    gated = C.gate_frames(_clip_with_legs([None, -5.0]))
    assert gated[0].leg_deg is None and gated[0].reason is None


def test_gate_a_keeps_priority_over_the_interval():
    gated = C.gate_frames(_real_clip([20, 20, 20, 40], leg=-80.0))
    assert gated[3].reason == C.RZ.REASON_HEEL_ON_HORSE and gated[0].reason == C.RZ.REASON_IMPLAUSIBLE_ANGLE


def test_the_summary_counts_the_implausible_frames_apart_from_gate_a():
    summary = C.summarize_quality(C.gate_frames(_clip_with_legs([-60.0, -5.0, 45.0])))
    assert summary.implausible_frames == 2 and summary.excluded_frames == 0
