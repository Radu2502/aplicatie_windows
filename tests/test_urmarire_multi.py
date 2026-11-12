"""Urmărirea multi-obiect (ecvestru/urmarire_multi.py), pe casete inventate, fără video real: detectorul
(decodarea, NMS), asignarea, ByteTrack, alegerea ID-ului după click, omul de pe cal, reasocierea și pierderea."""
import numpy as np
import pytest

from ecvestru import urmarire_multi as U
from ecvestru.rezultat import TargetSeed

FPS = 30.0
GAP = 60                     # tinta.MAX_GAP_SECONDS · 30 fps


# ---------- detectorul ----------

def test_nms_keeps_the_best_of_overlapping_boxes_and_the_separate_ones():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert U.nms(boxes, np.array([0.9, 0.8, 0.7])) == [0, 2]


def test_detections_keep_only_people_and_horses_above_the_default_score():
    def row(cx, cy, cls, score):
        out = np.zeros(85)
        out[:4], out[4], out[5 + cls] = (cx, cy, 20, 40), 1.0, score
        return out
    predictions = np.array([row(100, 100, U.PERSON, 0.9), row(300, 100, U.HORSE, 0.6), row(500, 100, 2, 0.9),
                            row(700, 100, U.PERSON, U.SCORE_MIN / 2)])
    found = U.detections(predictions, ratio=0.5)
    assert found[U.PERSON][:, :4].tolist() == [[180.0, 160.0, 220.0, 240.0]]       # înapoi în pixelii cadrului
    assert found[U.HORSE][:, 4].tolist() == [0.6] and len(found) == 2


def test_decode_puts_the_first_cell_at_the_stride():
    out = U.decode(np.zeros((sum((640 // s) ** 2 for s in U.STRIDES), 85)))
    assert out[0, :4].tolist() == [0.0, 0.0, 8.0, 8.0] and out[1, 0] == 8.0


def test_the_defaults_are_the_published_ones():
    assert (U.NMS_IOU, U.SCORE_MIN, U.DETECTOR_INPUT) == (0.45, 0.1, (640, 640))
    assert (U.TRACK_THRESH, U.TRACK_BUFFER, U.MATCH_THRESH, U.SECOND_MATCH, U.UNCONFIRMED_MATCH) == (
        0.5, 30, 0.8, 0.5, 0.7)


# ---------- asignarea și ByteTrack ----------

def test_iou_uses_the_plus_one_pixel_convention():
    assert U.iou_matrix([np.array([0, 0, 9, 9.0])], [np.array([0, 0, 9, 9.0])])[0, 0] == 1.0
    assert U.iou_matrix([np.array([0, 0, 9, 9.0])], [np.array([5, 0, 14, 9.0])])[0, 0] == pytest.approx(50 / 150)


def test_linear_assignment_is_optimal_and_leaves_costs_above_the_limit_unmatched():
    cost = np.array([[0.1, 0.2], [0.15, 0.9]])
    assert U.linear_assignment(cost, 0.8) == ([(0, 1), (1, 0)], [], [])
    assert U.linear_assignment(np.array([[0.9]]), 0.8) == ([], [0], [0])
    assert U.linear_assignment(np.zeros((0, 2)), 0.8) == ([], [], [0, 1])


def det(x: float, y: float = 100.0, score: float = 0.9, w: float = 40.0, h: float = 80.0) -> list[float]:
    return [x, y, x + w, y + h, score]


def run_tracker(frames: list[list[list[float]]]) -> list[dict[int, tuple]]:
    tracker = U.ByteTracker(FPS)
    return [{t[0]: t[1:5] for t in tracker.update(np.array(f, float).reshape(-1, 5))} for f in frames]


def test_bytetrack_keeps_the_ids_of_two_moving_boxes():
    out = run_tracker([[det(10 + 2 * k), det(300 - 2 * k)] for k in range(20)])
    assert all(set(o) == {1, 2} for o in out)
    assert out[-1][1][0] == pytest.approx(48, abs=2) and out[-1][2][0] == pytest.approx(262, abs=2)


def test_bytetrack_finds_the_same_id_after_a_short_gap_and_a_new_one_after_the_buffer():
    short = run_tracker([[det(10)]] * 5 + [[]] * 10 + [[det(10)]] * 3)
    assert {i for o in short for i in o} == {1}
    long = run_tracker([[det(10)]] * 5 + [[]] * (U.TRACK_BUFFER + 2) + [[det(10)]] * 3)
    assert {i for o in long[-2:] for i in o} == {2}


def test_a_low_score_box_only_continues_a_track_and_never_starts_one():
    out = run_tracker([[det(10)]] * 3 + [[det(10, score=0.3)]] * 3)
    assert all(set(o) == {1} for o in out)
    assert run_tracker([[det(10, score=0.3)]] * 5) == [{}] * 5


# ---------- alegerea după click ----------

HORSE_A, HORSE_B = (100.0, 200.0, 300.0, 300.0), (500.0, 200.0, 700.0, 300.0)


def test_the_click_picks_the_id_of_the_horse_under_it():
    horses = [{7: HORSE_A, 9: HORSE_B}] * 3
    assert U.seed_horse(horses, TargetSeed(frame=1, x=600, y=250), FPS) == (1, 9)
    with pytest.raises(ValueError):
        U.seed_horse(horses, TargetSeed(frame=1, x=2000, y=250), FPS)


def test_without_a_horse_on_the_clicked_frame_the_nearest_frame_is_used():
    horses = [{}, {}, {4: HORSE_A}, {}]
    assert U.seed_horse(horses, TargetSeed(frame=0, x=150, y=250), FPS) == (2, 4)


def pose_at(hip: tuple[float, float]) -> np.ndarray:
    points = np.zeros((26, 2), np.float32)
    points[U.LEFT_HIP] = points[U.RIGHT_HIP] = hip
    return points


def test_the_rider_is_the_person_with_the_hip_on_the_horse_by_the_candidate_rule():
    assert U.on_horse((200.0, 210.0), HORSE_A) and U.on_horse((200.0, 190.0), HORSE_A)    # 0,2·h deasupra
    assert not U.on_horse((200.0, 260.0), HORSE_A) and not U.on_horse((50.0, 210.0), HORSE_A)
    people = {1: (150, 100, 250, 260), 2: (160, 100, 260, 260), 3: (10, 100, 60, 300)}
    keypoints = {1: pose_at((150.0, 215.0)), 2: pose_at((200.0, 205.0)), 3: pose_at((30.0, 210.0))}
    assert U.rider_on(HORSE_A, people, keypoints) == 2         # cel mai aproape de mijlocul marginii de sus
    assert U.rider_on(HORSE_A, {3: people[3]}, {3: keypoints[3]}) is None


def boxes(*ids_x: tuple[int, float]) -> dict[int, tuple]:
    return {tid: (x, 0.0, x + 10.0, 10.0) for tid, x in ids_x}


def test_follow_keeps_the_id_and_waits_for_it_during_a_short_gap():
    track = [boxes((1, 0), (2, 100))] * 3 + [boxes((2, 100))] * 5 + [boxes((1, 5), (2, 100))] * 3
    result = U.follow(track, 0, 1, GAP)
    assert result.ids == [1] * 3 + [None] * 5 + [1] * 3 and result.switches == 0 and result.lost_from is None


def test_follow_reassociates_with_the_nearest_new_id_of_the_same_class():
    track = [boxes((1, 0), (2, 100))] * 3 + [boxes((2, 100))] * 2 + [boxes((2, 100), (5, 90), (6, 20))] * 3
    result = U.follow(track, 0, 1, GAP)
    assert result.ids[5:] == [6] * 3 and result.switches == 1          # 6 e cel mai aproape de ultima poziție a lui 1
    assert U.follow([boxes((1, 0), (2, 100))] * 2 + [boxes((2, 100))] * 3, 0, 1, GAP).ids[2:] == [None] * 3


def test_follow_loses_the_target_after_the_gap_for_good_in_that_direction():
    track = [boxes((1, 0))] * 2 + [{}] * (GAP + 1) + [boxes((1, 0))] * 2
    result = U.follow(track, 0, 1, GAP)
    assert result.lost_from == 2 + GAP and result.ids[-2:] == [None, None]


def test_follow_also_goes_backwards_from_the_start():
    track = [boxes((3, 0))] * 2 + [boxes((1, 0))] * 3
    result = U.follow(track, 4, 1, GAP)
    assert result.ids == [3, 3, 1, 1, 1] and result.switches == 1 and result.lost_from is None


def clip(horses, people, keypoints) -> U.ClipTracks:
    return U.ClipTracks(fps=FPS, n_frames=len(horses), horses=horses, people=people, keypoints=keypoints)


def test_choose_gives_the_frame_categories_with_the_old_criterion():
    on, off = pose_at((200.0, 210.0)), pose_at((200.0, 290.0))
    person = (150.0, 100.0, 250.0, 260.0)
    horses = [{1: HORSE_A, 2: HORSE_B}] * 4 + [{2: HORSE_B}] * (GAP + 2)
    people = [{10: person}, {10: person}, {}, {10: person}] + [{10: person}] * (GAP + 2)
    keypoints = [{10: on}, {10: off}, {}, {10: on}] + [{10: on}] * (GAP + 2)
    choice = U.choose(clip(horses, people, keypoints), TargetSeed(frame=0, x=200, y=250))
    assert choice.categories[:4] == [U.ON_C1, U.OTHER_RIDER, U.NO_SKELETON, U.ON_C1]
    assert set(choice.categories[4:]) == {U.LOST} and choice.lost_from == 3 + GAP + 1
    assert choice.horse_ids[:4] == [1] * 4 and choice.rider_ids[:4] == [10, 10, None, 10]
    assert U.shares(choice.categories)[U.ON_C1] == pytest.approx(100 * 2 / len(horses))


def test_the_rider_is_chosen_again_when_the_person_id_is_lost():
    person = (150.0, 100.0, 250.0, 260.0)
    n = GAP + 6
    horses = [{1: HORSE_A}] * n
    people = [{10: person}] * 2 + [{}] * (GAP + 1) + [{11: person}] * 3
    keypoints = [{10: pose_at((200.0, 210.0))}] * 2 + [{}] * (GAP + 1) + [{11: pose_at((200.0, 210.0))}] * 3
    choice = U.choose(clip(horses, people, keypoints), TargetSeed(frame=0, x=200, y=250))
    assert choice.rider_ids[-1] == 11 and choice.categories[-1] == U.ON_C1
    assert choice.categories[2] == U.NO_SKELETON


def test_the_tracks_are_saved_and_read_back(tmp_path):
    tracks = clip([{1: HORSE_A}, {}], [{4: (1.0, 2.0, 3.0, 4.0)}, {}], [{4: pose_at((5.0, 6.0))}, {}])
    tracks.seconds, tracks.cpu_seconds = 1.5, 2.5
    U.save_tracks(tmp_path / "t.npz", tracks)
    back = U.load_tracks(tmp_path / "t.npz")
    assert (back.n_frames, back.fps, back.seconds, back.cpu_seconds) == (2, FPS, 1.5, 2.5)
    assert back.horses == [{1: HORSE_A}, {}] and back.people == [{4: (1.0, 2.0, 3.0, 4.0)}, {}]
    assert U.hip_mid(back.keypoints[0][4]) == (5.0, 6.0)


def test_the_cache_name_says_when_only_the_first_frames_were_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    assert U.cache_path("a b.mp4", "abc", 1800).name == f"a b-abc__v{U.VERSION}__primele1800.npz"
    assert U.cache_path("a b.mp4", "abc").parent == tmp_path / U.CACHE_DIR


# ---------- ancora = calul, recuperare prin eliminare, RTMPose doar lângă calul țintă ----------

def test_follow_anchor_recovers_with_the_first_never_seen_id_and_not_with_the_other_horse():
    track = ([boxes((1, 0), (2, 100))] * 2 + [boxes((2, 100))] * 3            # ținta (1) dispare, 2 e alt cal
             + [boxes((2, 100), (7, 10))] * 2 + [boxes((2, 100), (7, 12))] * (GAP + 5))
    result = U.follow_anchor(track, 0, 1)
    assert result.ids[:2] == [1, 1] and result.ids[2:5] == [None] * 3          # golul: fără țintă, nu alt cal
    assert set(result.ids[5:]) == {7} and result.switches == 1 and result.lost_from is None


def test_follow_anchor_never_gives_up_and_picks_the_nearest_of_several_new_ids():
    track = [boxes((1, 0), (2, 300))] + [boxes((2, 300))] * (3 * GAP) + [boxes((2, 300), (8, 250), (9, 5))] * 2
    result = U.follow_anchor(track, 0, 1)
    assert result.ids[-2:] == [9, 9] and result.ids[1] is None and result.switches == 1


def test_follow_anchor_also_works_backwards_from_the_click():
    track = [boxes((5, 0), (2, 100))] * 2 + [boxes((1, 0), (2, 100))] * 3
    assert U.follow_anchor(track, 4, 1).ids == [5, 5, 1, 1, 1]


def test_the_rider_is_whoever_sits_on_the_target_horse_on_each_frame():
    on = pose_at((200.0, 210.0))
    tracks = clip([{1: HORSE_A}, {1: HORSE_A}, {}], [{10: (0, 0, 1, 1)}, {11: (0, 0, 1, 1)}, {}],
                  [{10: on}, {11: on}, {}])
    assert U.riders_on_target(tracks, [1, 1, None]) == [10, 11, None]          # ID-ul omului nu contează


def test_the_anchored_choice_has_no_other_rider_and_counts_recoveries():
    person, on, off = (150.0, 100.0, 250.0, 260.0), pose_at((200.0, 210.0)), pose_at((200.0, 290.0))
    horses = [{1: HORSE_A, 2: HORSE_B}] * 2 + [{2: HORSE_B}] * 2 + [{2: HORSE_B, 3: HORSE_A}] * 2
    people = [{10: person}] * 6
    keypoints = [{10: on}, {10: off}, {10: on}, {10: on}, {10: on}, {10: on}]
    tracks = clip(horses, people, keypoints)
    anchored = U.choose(tracks, TargetSeed(frame=0, x=200, y=250), recover=True, rider_by_horse=True)
    assert anchored.categories == [U.ON_C1, U.NO_SKELETON, U.LOST, U.LOST, U.ON_C1, U.ON_C1]
    assert anchored.horse_ids == [1, 1, None, None, 3, 3] and anchored.horse_switches == 1
    assert U.choose(tracks, TargetSeed(frame=0, x=200, y=250)).categories[1] == U.OTHER_RIDER   # fără ancoră


def test_boxes_overlap_with_any_common_area():
    assert U.overlaps((0, 0, 10, 10), (9, 9, 20, 20)) and not U.overlaps((0, 0, 10, 10), (10, 0, 20, 10))


def test_the_second_pass_runs_pose_only_on_people_over_the_target_horse(tmp_path):
    import cv2
    path = tmp_path / "s.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    first = clip([{1: (0.0, 20.0, 30.0, 40.0)}] * 2 + [{}],
                 [{5: (5.0, 0.0, 20.0, 30.0), 6: (40.0, 0.0, 60.0, 10.0)}] * 3, [{}] * 3)
    calls = []
    fake = lambda image, box: (calls.append(tuple(box)) or np.ones((26, 2)), np.ones(26))  # noqa: E731
    out = U.run_target_poses(path, first, [1, 1, None], pose=fake)
    assert calls == [(5.0, 0.0, 20.0, 30.0)] * 2 and [set(k) for k in out.keypoints] == [{5}, {5}, set()]
    assert out.pose_seconds > 0 and out.seconds >= out.pose_seconds
    U.save_tracks(tmp_path / "t.npz", out)
    back = U.load_tracks(tmp_path / "t.npz")
    assert [set(k) for k in back.keypoints] == [{5}, {5}, set()] and back.pose_seconds == out.pose_seconds
