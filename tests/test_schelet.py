"""Scheletul complet: identificarea pe decupaj, cadrele fără călăreț, citire-scriere, unghiul de exemplu și
lanțul față de brut/. Testele rapide folosesc schelete construite de mână; cel real, un clip dev scurt."""
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ecvestru import masurare as M
from ecvestru import schelet as Z

WIDTH, HEIGHT = 640, 360
BOX = (100, 50, 300, 250)
SHORT = "clip_152429.mp4"
points_for_frame = M.candidate_module().points_for_frame


def pose(seed=0, visibility=0.9):
    rng = np.random.default_rng(seed)
    return [NS(x=float(a), y=float(b), z=float(c), visibility=visibility + float(d), presence=0.95)
            for a, b, c, d in zip(rng.uniform(.1, .9, 33), rng.uniform(.1, .9, 33), rng.uniform(-.2, .2, 33),
                                  rng.uniform(0, .09, 33), strict=True)]


def world(seed=0):
    rng = np.random.default_rng(seed + 100)
    return [NS(x=float(a), y=float(b), z=float(c)) for a, b, c in rng.uniform(-.8, .8, (33, 3))]


def on_full_frame(raw, box=BOX):
    x0, y0, x1, y1 = box
    return [NS(x=(x0 + p.x * (x1 - x0)) / WIDTH, y=(y0 + p.y * (y1 - y0)) / HEIGHT, visibility=p.visibility)
            for p in raw]


def run(*frames):
    """frames: (ales, rezultatul detectorului simplu, rezultatele pe decupaje, motiv)."""
    log = Z.Recording()

    def source():
        for index, (chosen, single, crops, reason) in enumerate(frames):
            log.single, log.crops = single, crops
            log.backs = [[on_full_frame(p) for p in r.pose_landmarks] for r in crops]
            yield index, chosen, WIDTH, HEIGHT, reason

    return Z.assemble(source(), points_for_frame, log)


def result(poses=(), worlds=()):
    return NS(pose_landmarks=list(poses), pose_world_landmarks=list(worlds))


def full_frame():
    p = pose()
    return p, result([p], [world()]), [], None


def crop_frame(seed=1, others=()):
    raw = pose(seed)
    return on_full_frame(raw), result(), [result([raw, *others], [world(seed)] * (1 + len(others)))], None


def test_the_33_names_follow_the_order_of_mediapipe():
    assert len(Z.LANDMARK_NAMES) == 33
    assert Z.LANDMARK_NAMES[0] == "nose" and Z.LANDMARK_NAMES[11] == "left_shoulder"
    assert Z.LANDMARK_NAMES[30] == "right_heel" and Z.LANDMARK_NAMES[32] == "right_foot_index"


def test_a_full_frame_keeps_every_value_of_mediapipe():
    p, single, crops, reason = full_frame()
    rows, _, stats = run((p, single, crops, reason))
    assert rows["source"][0] == Z.SOURCE_FULL and rows["n_poses"][0] == 1
    assert rows["xy_norm"][0, 11, 0] == p[11].x and rows["z"][0, 11] == p[11].z
    assert rows["presence"][0, 11] == pytest.approx(.95)
    assert rows["world_xyz"][0, 11, 2] == pytest.approx(world()[11].z)
    assert np.isnan(rows["crop_box"][0]).all()
    assert stats == {"riders": 1, "identified": 1, "unmatched": 0, "ambiguous": 0}


def test_a_crop_frame_recovers_z_presence_world_and_the_box():
    frame = crop_frame()
    rows, _, stats = run(frame)
    raw = frame[2][0].pose_landmarks[0]
    assert rows["source"][0] == Z.SOURCE_CROP and tuple(rows["crop_box"][0]) == BOX
    assert rows["z"][0, 5] == pytest.approx(raw[5].z * (BOX[2] - BOX[0]) / WIDTH)
    assert rows["presence"][0, 5] == pytest.approx(.95)
    assert rows["world_xyz"][0, 5, 0] == pytest.approx(world(1)[5].x)
    assert stats["identified"] == 1


def test_n_poses_counts_the_largest_detection_of_the_frame():
    frame = crop_frame(others=[pose(7), pose(8)])
    assert run(frame)[0]["n_poses"][0] == 3


def test_n_poses_is_zero_without_any_detection():
    assert run((None, result(), [], "calaret_negasit"))[0]["n_poses"][0] == 0


def test_a_frame_with_no_match_on_the_crops_is_not_guessed():
    chosen = crop_frame()[0]
    rows, raw, stats = run((chosen, result(), [result([pose(9)], [world(9)])], None))
    assert rows["reason"][0] == Z.REASON_UNIDENTIFIED and rows["source"][0] == Z.SOURCE_NONE
    assert np.isnan(rows["xy_norm"][0]).all() and np.isnan(rows["crop_box"][0]).all()
    assert raw[0]["points"] is not None and raw[0]["motiv"] is None          # brut/ nu se schimbă
    assert stats == {"riders": 1, "identified": 0, "unmatched": 1, "ambiguous": 0}


def test_two_poses_that_fit_the_same_frame_are_ambiguous_and_not_guessed():
    frame = crop_frame()
    raw = frame[2][0].pose_landmarks[0]
    double = result([raw, raw], [world(1), world(2)])
    rows, _, stats = run((frame[0], result(), [double], None))
    assert rows["reason"][0] == Z.REASON_UNIDENTIFIED and np.isnan(rows["world_xyz"][0]).all()
    assert stats["ambiguous"] == 1 and stats["unmatched"] == 0


def test_equal_visibility_at_another_place_is_not_a_match():
    frame = crop_frame()
    raw = frame[2][0].pose_landmarks[0]
    moved = [NS(x=p.x * .5 + b, y=p.y * .5, z=p.z, visibility=p.visibility, presence=p.presence)
             for p, b in zip(raw, np.linspace(0, .3, 33), strict=True)]
    assert run((frame[0], result(), [result([moved], [world(1)])], None))[2]["unmatched"] == 1


def test_frames_without_a_rider_stay_with_their_reason():
    rows, raw, _ = run(full_frame(), (None, result(), [], "nu_e_pe_cal"))
    assert list(rows["frame"]) == [0, 1] and rows["reason"][1] == "nu_e_pe_cal"
    assert np.isnan(rows["xy_norm"][1]).all() and rows["source"][1] == Z.SOURCE_NONE
    assert raw[1] == {"points": None, "motiv": "nu_e_pe_cal"}


def test_a_rider_with_hidden_shoulder_has_the_reason_of_brut():
    p = pose()
    for i in (11, 12, 23, 24):
        p[i].visibility = 0.1
    rows, raw, _ = run((p, result([p], [world()]), [], None))
    assert raw[0]["motiv"] == rows["reason"][0] == M.candidate_module().REASON_POINTS
    assert rows["source"][0] == Z.SOURCE_FULL


def skeleton():
    rows, _, stats = run(full_frame(), crop_frame(), (None, result(), [], "nu_e_pe_cal"))
    return Z.make_skeleton(rows, np.array([0.0, 0.033, 0.066]), {"schema_version": Z.SCHEMA_VERSION, "fps": 30.0,
                                                                 "width": WIDTH, "height": HEIGHT, "stats": stats})


def test_write_and_read_lose_nothing(tmp_path):
    original = skeleton()
    path = tmp_path / "a" / "s.npz"
    Z.save_skeleton(path, original)
    back = Z.load_skeleton(path)
    assert back.meta == original.meta
    for name in Z.ARRAYS:
        left, right = getattr(original, name), getattr(back, name)
        assert left.dtype == right.dtype and left.shape == right.shape, name
        assert np.array_equal(left, right, equal_nan=left.dtype.kind == "f"), name


def test_only_the_final_file_stays_on_disk(tmp_path):
    Z.save_skeleton(tmp_path / "s.npz", skeleton())
    assert [p.name for p in tmp_path.iterdir()] == ["s.npz"]


def test_a_file_with_another_format_version_is_refused(tmp_path):
    old = skeleton()
    Z.save_skeleton(tmp_path / "s.npz", replace(old, meta={**old.meta, "schema_version": 99}))
    with pytest.raises(ValueError, match="99"):
        Z.load_skeleton(tmp_path / "s.npz")


def test_pixels_come_from_the_normalized_values():
    s = skeleton()
    assert s.xy_px[0, 3, 0] == s.xy_norm[0, 3, 0] * WIDTH and s.xy_px[0, 3, 1] == s.xy_norm[0, 3, 1] * HEIGHT


def test_a_point_can_be_read_by_name_in_each_space():
    s = skeleton()
    assert s.point("left_knee").shape == (3, 2) and s.point("left_knee", "norm")[0, 0] == s.xy_norm[0, 25, 0]
    assert s.point("left_knee", "world").shape == (3, 3)
    with pytest.raises(ValueError):
        s.point("coada")
    with pytest.raises(ValueError):
        s.point("nose", "alt_spatiu")


def test_has_rider_marks_the_frames_with_points():
    assert list(skeleton().has_rider) == [True, True, False]


def test_angle_at_a_right_angle():
    assert Z.angle_at(np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]]), np.array([[0.0, 2.0]]))[0] == pytest.approx(90)


def test_angle_at_works_in_3d_and_on_a_straight_line():
    a, b, c = np.array([[1.0, 1.0, 1.0]]), np.array([[0.0, 0.0, 0.0]]), np.array([[-2.0, -2.0, -2.0]])
    assert Z.angle_at(a, b, c)[0] == pytest.approx(180)


def test_angle_at_on_a_hand_made_skeleton_gives_nan_without_a_rider():
    s = skeleton()
    angle = Z.angle_at(s.point("left_hip"), s.point("left_knee"), s.point("left_ankle"))
    assert angle.shape == (3,) and 0 <= angle[0] <= 180 and np.isnan(angle[2])


def test_angle_at_a_zero_length_segment_is_nan():
    same = np.array([[1.0, 1.0]])
    assert np.isnan(Z.angle_at(same, same, np.array([[2.0, 2.0]]))[0])


def _assert_chain_matches(sk, raw):
    chain = sk.chain()
    identified = 0
    for frame, row in raw.items():
        i, p = int(frame), row["points"]
        if sk.reason[i] == Z.REASON_UNIDENTIFIED:
            continue
        assert (p is None) == (sk.source[i] == Z.SOURCE_NONE), frame
        if p is None:
            assert sk.reason[i] == (row["motiv"] or "")
            continue
        identified += 1
        assert sk.facing[i] == p["facing"], frame
        for k in ("shoulder", "hip", "heel"):
            if p[k] is None:
                assert np.isnan(chain[k][i]).all(), (frame, k)
            else:
                assert tuple(chain[k][i]) == tuple(p[k]), (frame, k)      # exact, nu aproximativ
    return identified


def test_the_chain_of_a_hand_made_pass_equals_brut():
    rows, raw, _ = run(full_frame(), crop_frame(), (None, result(), [], "nu_e_pe_cal"))
    sk = Z.make_skeleton(rows, np.zeros(3), {"width": WIDTH, "height": HEIGHT})
    assert _assert_chain_matches(sk, raw) == 2


def _real_pass(video):
    done = Z.extract_pass(M.V.VIDEOS / video, M.candidate_label())
    saved = next((M.results_dir() / "brut").glob(f"{Path(video).stem}-*.json"), None)
    return done, json.loads(saved.read_text(encoding="utf-8")) if saved else None


@pytest.mark.skipif(not (M.V.VIDEOS / SHORT).exists(), reason="clipul nu e local")
def test_the_chain_of_a_real_clip_equals_brut_and_the_pass_gives_the_same_raw():
    done, saved = _real_pass(SHORT)
    plain = M.candidate_module().predict_points_with_reasons(str(M.V.VIDEOS / SHORT))
    assert json.loads(json.dumps(dict(done))) == json.loads(json.dumps({str(k): v for k, v in plain.items()}))
    assert _assert_chain_matches(done.skeleton, plain) > 0
    meta = done.skeleton.meta
    assert meta["schema_version"] == Z.SCHEMA_VERSION and meta["candidate"] == M.candidate_label()
    assert meta["pose_model"] == "full" and len(meta["pose_model_sha256"]) == 64
    assert done.skeleton.time_s[0] == 0.0 and (np.diff(done.skeleton.time_s) > 0).all()


@pytest.mark.skipif(os.environ.get("ECVESTRU_TESTE_LENTE") != "1", reason="test lent: ECVESTRU_TESTE_LENTE=1")
def test_the_chain_of_every_cached_skeleton_equals_brut():
    root = M.ROOT / "data" / "results" / "analiza"
    files = sorted((root / "schelet").glob(f"*__s{Z.SCHEMA_VERSION}.npz"))
    assert len(files) == len(M.dev_videos())
    assert all(Z.load_skeleton(path).meta["schema_version"] == Z.SCHEMA_VERSION for path in files)
    for path in files:
        sk = Z.load_skeleton(path)
        raw = json.loads(next((root / "brut").glob(f"{path.name.split('__')[0]}__*.json")).read_text(encoding="utf-8"))
        assert _assert_chain_matches(sk, raw) >= 0, path.name


@pytest.mark.parametrize("visibility,tier", [(0.49, Z.TIER_LOW), (0.5, Z.TIER_MID), (0.8, Z.TIER_MID),
                                             (0.81, Z.TIER_HIGH), (float("nan"), None)])
def test_visibility_tiers_have_fixed_limits(visibility, tier):
    assert Z.visibility_tier(visibility) == tier


def test_the_tiers_start_at_the_threshold_of_the_candidate():
    assert Z.VISIBILITY_MID == M.V.MIN_VISIBILITY and Z.VISIBILITY_HIGH == 0.8


# ---------- persoanele respinse ----------

def test_the_format_is_version_three():
    assert Z.SCHEMA_VERSION == 3


def test_an_old_file_is_refused_with_a_clear_message(tmp_path):
    old = skeleton()
    Z.save_skeleton(tmp_path / "s.npz", replace(old, meta={**old.meta, "schema_version": 1}))
    with pytest.raises(ValueError, match="(?s)1.*regener"):
        Z.load_skeleton(tmp_path / "s.npz")


def test_a_rejected_person_of_the_full_frame_keeps_the_reason_of_the_candidate():
    p = pose(3)
    rows, raw, _ = run((None, result([p], [world(3)]), [], "nu_e_pe_cal"))
    assert rows["rejected_count"][0] == 1 and rows["rejected_source"][0, 0] == Z.SOURCE_FULL
    assert rows["rejected_reason"][0, 0] == "nu_e_pe_cal" and rows["rejected_xy_norm"][0, 0, 5, 0] == p[5].x
    assert rows["rejected_z"][0, 0, 5] == p[5].z and rows["rejected_presence"][0, 0, 5] == pytest.approx(.95)
    assert rows["rejected_world_xyz"][0, 0, 5, 1] == pytest.approx(world(3)[5].y)
    assert np.isnan(rows["rejected_crop_box"][0, 0]).all()
    assert rows["source"][0] == Z.SOURCE_NONE and np.isnan(rows["xy_norm"][0]).all()      # nu devine călărețul ales
    assert raw[0] == {"points": None, "motiv": "nu_e_pe_cal"}


def test_every_rejected_person_is_kept_without_choosing_one():
    crop_a, crop_b = pose(4), pose(5)
    frame = (None, result([pose(3)], [world(3)]),
             [result([crop_a, crop_b], [world(4), world(5)]), result([pose(6)], [world(6)])], "trunchi_peste_45")
    rows, _, _ = run(frame)
    assert rows["rejected_count"][0] == 4 > rows["n_poses"][0] == 2
    assert list(rows["rejected_source"][0]) == [Z.SOURCE_FULL, Z.SOURCE_CROP, Z.SOURCE_CROP, Z.SOURCE_CROP]


def test_a_rejected_person_of_a_crop_comes_back_on_the_full_frame():
    raw = pose(4)
    rows, _, _ = run((None, result(), [result([raw], [world(4)])], "calaret_negasit"))
    assert rows["rejected_reason"][0, 0] == Z.REASON_NOT_CHOSEN
    assert tuple(rows["rejected_crop_box"][0, 0]) == BOX
    assert rows["rejected_xy_norm"][0, 0, 7, 0] == (BOX[0] + raw[7].x * (BOX[2] - BOX[0])) / WIDTH
    assert rows["rejected_z"][0, 0, 7] == pytest.approx(raw[7].z * (BOX[2] - BOX[0]) / WIDTH)


def test_next_to_a_rider_from_a_crop_the_others_stay_rejected():
    chosen, _, crops, _ = crop_frame(others=[pose(7)])
    single = result([pose(3)], [world(3)])
    rows, _, stats = run((chosen, single, crops, None))
    assert rows["source"][0] == Z.SOURCE_CROP and stats["identified"] == 1
    assert rows["rejected_count"][0] == 2
    assert list(rows["rejected_reason"][0]) == [Z.REASON_NOT_CHOSEN, Z.REASON_NOT_CHOSEN]
    assert rows["rejected_xy_norm"][0, 1, 0, 0] != rows["xy_norm"][0, 0, 0]


def test_a_rider_of_the_full_frame_has_nobody_rejected():
    rows = run(full_frame())[0]
    assert rows["rejected_count"][0] == 0 and rows["rejected_source"].shape == (1, 0)


def test_an_unidentified_frame_does_not_guess_who_was_rejected_on_the_crops():
    frame = crop_frame()
    wrong = result([pose(9)], [world(9)])
    assert run((frame[0], result(), [wrong], None))[0]["rejected_count"][0] == 0


def test_empty_places_stay_empty():
    rows = run((None, result([pose(3)], [world(3)]), [], "nu_e_pe_cal"), (None, result(), [], "calaret_negasit"))[0]
    assert rows["rejected_count"][1] == 0 and rows["rejected_source"][1, 0] == 0
    assert rows["rejected_reason"][1, 0] == "" and np.isnan(rows["rejected_xy_norm"][1]).all()


def test_rejected_people_are_read_with_pixels_and_reasons(tmp_path):
    rows, _, _ = run((None, result([pose(3)], [world(3)]), [], "nu_e_pe_cal"))
    sk = Z.make_skeleton(rows, np.zeros(1), {"schema_version": Z.SCHEMA_VERSION, "width": WIDTH, "height": HEIGHT})
    Z.save_skeleton(tmp_path / "s.npz", sk)
    (xy, visibility, reason), = Z.load_skeleton(tmp_path / "s.npz").rejected_at(0)
    assert xy[5, 0] == pose(3)[5].x * WIDTH and visibility[5] == pytest.approx(pose(3)[5].visibility)
    assert reason == "nu_e_pe_cal"


def test_the_chain_ignores_rejected_people():
    rows, raw, _ = run((None, result([pose(3)], [world(3)]), [], "nu_e_pe_cal"), full_frame())
    sk = Z.make_skeleton(rows, np.zeros(2), {"width": WIDTH, "height": HEIGHT})
    assert _assert_chain_matches(sk, raw) == 1


# ---------- dublurile călărețului printre persoanele respinse ----------

def body(shift=0.0, torso=100.0, legs=0.0):
    """33 de puncte în pixeli: umerii la y=100, șoldurile la y=100+torso, restul depărtate cu `legs`."""
    xy = np.tile([200.0 + shift, 300.0], (33, 1)) + np.stack([np.linspace(-90, 90, 33), np.zeros(33)], axis=1)
    xy[25:33, 0] += legs
    for name, x, y in (("left_shoulder", 190, 100), ("right_shoulder", 210, 100),
                       ("left_hip", 190, 100 + torso), ("right_hip", 210, 100 + torso)):
        xy[Z.LANDMARK_NAMES.index(name)] = (x + shift, y)
    return xy


def test_the_same_trunk_is_the_same_person_even_with_other_legs():
    assert Z.same_as_rider(body(), body(shift=10, legs=400)) is True


def test_the_limit_of_the_same_person_is_included():
    assert Z.same_as_rider(body(), body(shift=30)) is True
    assert Z.same_as_rider(body(), body(shift=31)) is None


def test_a_trunk_close_but_not_on_the_rider_is_unsure():
    assert Z.same_as_rider(body(), body(shift=80)) is None


def test_a_person_far_from_the_rider_is_another_person():
    assert Z.same_as_rider(body(), body(shift=500)) is False


def test_overlapping_boxes_are_unsure_even_with_distant_trunks():
    wide, other = body(), body(shift=150)
    wide[25], wide[26] = (0.0, 0.0), (500.0, 500.0)
    other[25], other[26] = (0.0, 0.0), (500.0, 500.0)
    assert Z.same_as_rider(wide, other) is None


def _double_of_the_rider():
    """Călărețul vine din decupaj, iar detectorul pe cadrul întreg a găsit aceeași persoană."""
    chosen, _, crops, _ = crop_frame()
    twin = [NS(x=p.x, y=p.y, z=0.0, visibility=p.visibility, presence=.9) for p in chosen]
    return chosen, result([twin], [world(3)]), crops, None


def test_a_double_of_the_rider_is_marked_and_stays_in_the_data():
    rows, _, _ = run(_double_of_the_rider())
    assert rows["rejected_count"][0] == 1 and rows["rejected_is_rider"].dtype == bool
    assert rows["rejected_is_rider"][0, 0] and not np.isnan(rows["rejected_xy_norm"][0, 0]).any()


def test_a_person_on_a_frame_without_a_rider_is_never_the_rider():
    rows = run((None, result([pose(3)], [world(3)]), [], "nu_e_pe_cal"))[0]
    assert not rows["rejected_is_rider"].any()


def test_unsure_cases_are_false_and_marked(monkeypatch):
    monkeypatch.setattr(Z, "same_as_rider", lambda rider, other: None)
    rows, _, _ = run(_double_of_the_rider())
    assert not rows["rejected_is_rider"][0, 0] and rows["rejected_unsure"][0, 0]


def test_a_double_of_the_rider_is_not_among_the_people_to_draw():
    rows, _, _ = run(_double_of_the_rider(), (None, result([pose(3)], [world(3)]), [], "nu_e_pe_cal"))
    sk = Z.make_skeleton(rows, np.zeros(2), {"width": WIDTH, "height": HEIGHT})
    assert len(sk.rejected_at(0)) == 1 and sk.shown_rejected_at(0) == []
    assert len(sk.shown_rejected_at(1)) == 1


def test_two_rejected_doubles_of_each_other_are_both_drawn():
    p = pose(3)
    frame = (None, result([p], [world(3)]), [result([pose(4)], [world(4)])], "nu_e_pe_cal")
    rows, _, _ = run(frame)
    sk = Z.make_skeleton(rows, np.zeros(1), {"width": WIDTH, "height": HEIGHT})
    assert len(sk.shown_rejected_at(0)) == 2


# ---------- ce se desenează ----------

def test_the_drawn_points_are_the_head_arms_trunk_and_legs_only():
    assert Z.DRAWN_POINTS == (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30)


def test_the_drawn_lines_join_only_drawn_points():
    assert {i for line in Z.DRAWN_LINES for i in line} == set(Z.DRAWN_POINTS)
    assert (0, 7) in Z.DRAWN_LINES and (27, 29) in Z.DRAWN_LINES and (29, 31) not in Z.DRAWN_LINES


def test_every_point_stays_in_the_data_even_if_not_drawn():
    assert skeleton().xy_norm.shape[1] == len(Z.LANDMARK_NAMES) == 33


@pytest.mark.parametrize("line,side,far", [((11, 13), "left", False), ((12, 14), "left", True),
                                           ((11, 13), "right", True), ((11, 12), "left", False),
                                           ((23, 24), "right", False), ((11, 23), "right", True),
                                           ((0, 7), "right", True), ((0, 8), "right", False), ((11, 13), "", False)])
def test_a_line_belongs_to_the_far_side_only_if_it_is_on_the_other_side(line, side, far):
    assert Z.is_far_line(line, side) is far


@pytest.mark.parametrize("point,side,far", [(11, "left", False), (12, "left", True), (0, "left", False),
                                            (7, "right", True), (12, "", False)])
def test_a_point_belongs_to_the_far_side_by_its_number(point, side, far):
    assert Z.is_far_point(point, side) is far
