"""Urmărirea după aspect (ecvestru/aspect_urmarire.py): histograma HSV, scorul relativ, recuperarea după aspect,
varianta „schimb de ID” și „același om”. Pe imagini și casete inventate, fără video real."""
import math

import numpy as np
import pytest

from ecvestru import aspect_urmarire as AS
from ecvestru import urmarire_multi as U

RED, BLUE, GREEN = (0, 0, 255), (255, 0, 0), (0, 255, 0)


def image_with(*patches):
    image = np.zeros((100, 200, 3), np.uint8)
    for (x0, y0, x1, y1), color in patches:
        image[y0:y1, x0:x1] = color
    return image


def test_the_histogram_tells_colours_apart_and_ignores_boxes_outside_the_frame():
    image = image_with(((0, 0, 50, 50), RED), ((100, 0, 150, 50), BLUE))
    red, blue = AS.histogram(image, (0, 0, 50, 50)), AS.histogram(image, (100, 0, 150, 50))
    assert red.shape == AS.HIST_BINS and AS.distance(red, red) == pytest.approx(0.0, abs=1e-6)
    assert AS.distance(red, blue) > 0.9
    assert AS.histogram(image, (300, 0, 400, 50)) is None and AS.histogram(image, (-20, -5, 10, 10)) is not None


def test_the_relative_score_prefers_what_looks_like_the_target():
    image = image_with(((0, 0, 50, 50), RED), ((100, 0, 150, 50), BLUE), ((60, 60, 90, 90), GREEN))
    red, blue, green = (AS.histogram(image, b) for b in ((0, 0, 50, 50), (100, 0, 150, 50), (60, 60, 90, 90)))
    target, other = AS.Look(red, green), AS.Look(blue, green)
    assert AS.relative(AS.Look(red, green), target, other) > 0 > AS.relative(AS.Look(blue, green), target, other)
    assert AS.look_distance(AS.Look(red), AS.Look(red, green)) == pytest.approx(0.0, abs=1e-6)   # fără călăreț


def boxes(*ids_x):
    return {tid: (x, 0.0, x + 10.0, 10.0) for tid, x in ids_x}


def test_recovery_picks_the_most_target_like_horse_not_the_first_new_id():
    track = [boxes((1, 0), (2, 100))] * 2 + [boxes((2, 100))] * 2 + [boxes((2, 100), (7, 5))] * 3
    scores = [{1: 1.0, 2: -1.0}] * 2 + [{2: 1.0}] * 2 + [{2: 0.8, 7: -0.5}] * 3   # 2 arată acum ca ținta
    assert U.follow_anchor(track, 0, 1).ids[4:] == [7] * 3                        # ancora pe cal: ID-ul nou
    result = AS.follow_aspect(track, 0, 1, scores)
    assert result.ids[4:] == [2] * 3 and result.switches == 1
    assert AS.follow_aspect(track, 0, 1, scores, all_visible=False).ids[4:] == [7] * 3   # doar ID-uri noi


def test_the_swap_variant_moves_the_target_after_k_bad_frames_only_when_asked():
    track = [boxes((1, 0), (2, 100))] * 8
    scores = [{1: 1.0, 2: -1.0}] * 2 + [{1: -0.4, 2: 0.6}] * 6                    # 1 începe să arate ca celălalt
    assert AS.follow_aspect(track, 0, 1, scores).ids == [1] * 8                   # implicit: niciun schimb
    swapped = AS.follow_aspect(track, 0, 1, scores, swap_frames=3)
    assert swapped.ids == [1] * 4 + [2] * 4 and swapped.switches == 1
    assert AS.follow_aspect(track, 0, 1, scores, swap_frames=7).ids == [1] * 8


def pose(hip, shoulder):
    points = np.zeros((26, 2), np.float32)
    points[list(AS.HIPS)] = hip
    points[list(AS.SHOULDERS)] = shoulder
    return points


def test_same_person_uses_the_hip_distance_in_torso_lengths():
    a = pose((100.0, 200.0), (100.0, 100.0))                                      # trunchi 100 px
    assert AS.torso(a) == pytest.approx(100.0)
    assert AS.same_person(a, pose((125.0, 200.0), (0.0, 0.0)), 0.3)
    assert not AS.same_person(a, pose((140.0, 200.0), (0.0, 0.0)), 0.3)
    assert not AS.same_person(pose((0.0, 0.0), (0.0, 0.0)), a, 0.3)


def test_the_other_horse_is_another_horse_with_a_rider_on_it():
    horse_a, horse_b, horse_c = (0.0, 100.0, 100.0, 200.0), (200.0, 100.0, 300.0, 200.0), (400.0, 100.0, 450.0, 150.0)
    on_b = pose((250.0, 110.0), (250.0, 30.0))
    tracks = U.ClipTracks(fps=30.0, n_frames=2, horses=[{1: horse_a}, {1: horse_a, 2: horse_b, 3: horse_c}],
                          people=[{}, {9: (220.0, 0.0, 280.0, 150.0)}], keypoints=[{}, {9: on_b}])
    assert AS.other_horse(tracks, 1, 1) == 2                                      # 3 n-are călăreț
    assert AS.other_frame(tracks, 0, [1, 1]) == (1, 2) and AS.other_frame(tracks, 0, [1, None]) is None


def test_scores_are_saved_and_read_back(tmp_path):
    scores = AS.Scores([{1: 0.5, 2: -0.25}, {}], 0, 1, 1, 2, 3.0, 4.0)
    AS.save_scores(tmp_path / "s.npz", scores)
    assert AS.load_scores(tmp_path / "s.npz") == scores


def test_compute_scores_reads_only_box_pixels_of_a_synthetic_clip(tmp_path, monkeypatch):
    import cv2
    path = tmp_path / "s.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (200, 100))
    for _ in range(3):
        writer.write(image_with(((0, 40, 60, 100), RED), ((120, 40, 180, 100), BLUE), ((10, 0, 50, 45), GREEN),
                                ((130, 0, 170, 45), GREEN)))
    writer.release()
    horses = {1: (0.0, 40.0, 60.0, 100.0), 2: (120.0, 40.0, 180.0, 100.0)}
    people = {5: (10.0, 0.0, 50.0, 45.0), 6: (130.0, 0.0, 170.0, 45.0)}
    keypoints = {5: pose((30.0, 45.0), (30.0, 10.0)), 6: pose((150.0, 45.0), (150.0, 10.0))}
    tracks = U.ClipTracks(fps=30.0, n_frames=3, horses=[horses] * 3, people=[people] * 3, keypoints=[keypoints] * 3)
    scores = AS.compute_scores(path, tracks, 0, 1, [1, 1, 1])
    assert (scores.other_frame, scores.other_id) == (0, 2)
    assert all(row[1] > 0 > row[2] for row in scores.relative) and scores.seconds > 0
    choice = AS.choose(tracks, 0, 1, scores)
    assert choice.categories == [U.ON_C1] * 3 and choice.rider_ids == [5, 5, 5] and math.isfinite(scores.cpu_seconds)
