"""Mărimile pentru direcția calului, pe puncte sintetice: unghiul cap–coadă, lungimea aparentă cap–coadă raportată la
înălțimea casetei, raportul lățime/înălțime al casetei. Nu se compară cu nicio etichetă de vedere."""
import numpy as np
import pytest

from ecvestru import cal_puncte as CP
from ecvestru import directie_cal as D

NAN2 = [np.nan, np.nan]


def horse(head=(200.0, 100.0), tail=(100.0, 100.0), neck=(180.0, 110.0), score=0.9, nose_score=None):
    """Punctele unui cal (17, 2) și scorurile, cu nasul, gâtul și rădăcina cozii puse; restul la mijloc."""
    points = np.full((17, 2), 150.0)
    points[CP.AP10K.index("nose")] = head
    points[CP.AP10K.index("neck")] = neck
    points[CP.AP10K.index("root_of_tail")] = tail
    scores = np.full(17, score)
    if nose_score is not None:
        scores[CP.AP10K.index("nose")] = nose_score
    return points, scores


@pytest.mark.parametrize("head,tail,angle", [
    ((200, 100), (100, 100), 0.0),        # capul spre dreapta imaginii
    ((0, 100), (100, 100), 180.0),        # spre stânga
    ((100, 0), (100, 100), 90.0),         # spre sus în imagine (y crește în jos)
    ((100, 200), (100, 100), -90.0),      # spre jos
    ((200, 0), (100, 100), 45.0)])
def test_the_head_tail_angle_is_measured_from_the_right_counterclockwise(head, tail, angle):
    got = D.head_tail_angle(np.array([head], float), np.array([tail], float))
    assert got[0] == pytest.approx(angle)


def test_the_angle_is_missing_when_a_point_is_missing():
    got = D.head_tail_angle(np.array([NAN2, [1.0, 1.0]]), np.array([[0.0, 0.0], NAN2]))
    assert np.isnan(got).all()


def test_the_apparent_length_is_divided_by_the_box_height():
    head, tail = np.array([[200.0, 100.0], [110.0, 100.0]]), np.array([[100.0, 100.0], [100.0, 100.0]])
    boxes = np.array([[0.0, 0.0, 300.0, 50.0], [0.0, 0.0, 30.0, 100.0]])
    assert D.head_tail_length(head, tail, boxes) == pytest.approx([2.0, 0.1])   # profil lung, față/spate scurt


def test_the_length_is_missing_without_a_box_or_with_an_empty_box():
    head, tail = np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.0]])
    boxes = np.array([[np.nan] * 4, [0.0, 5.0, 10.0, 5.0]])
    assert np.isnan(D.head_tail_length(head, tail, boxes)).all()


def test_the_box_aspect_is_width_over_height():
    boxes = np.array([[0.0, 0.0, 300.0, 100.0], [10.0, 10.0, 40.0, 110.0], [np.nan] * 4])
    got = D.box_aspect(boxes)
    assert got[:2] == pytest.approx([3.0, 0.3]) and np.isnan(got[2])


def test_the_head_is_the_nose_or_the_neck_when_the_nose_is_below_the_threshold():
    points, scores = horse()
    weak_points, weak_scores = horse(nose_score=0.1)
    heads, sources = D.head_points(np.stack([points, weak_points]), np.stack([scores, weak_scores]))
    assert heads == pytest.approx(np.array([[200.0, 100.0], [180.0, 110.0]]))
    assert list(sources) == [D.HEAD_NOSE, D.HEAD_NECK]


def test_no_head_and_no_tail_below_the_threshold():
    points, scores = horse(score=0.1)
    heads, sources = D.head_points(points[None], scores[None])
    assert np.isnan(heads).all() and list(sources) == [D.HEAD_NONE]
    assert np.isnan(D.tail_points(points[None], scores[None])).all()


def test_the_features_of_a_clip_run_have_one_value_per_frame():
    points, scores = horse(head=(300.0, 100.0), tail=(100.0, 100.0))
    run = CP.ClipRun(points=np.stack([points, np.full((17, 2), np.nan)]),
                     scores=np.stack([scores, np.full(17, np.nan)]),
                     boxes=np.array([[50.0, 50.0, 350.0, 150.0], [np.nan] * 4]), seconds=0.0)
    got = D.direction_features(run)
    assert got.angle_deg[0] == pytest.approx(0.0) and got.length_ratio[0] == pytest.approx(2.0)
    assert got.aspect[0] == pytest.approx(3.0) and got.head_source[0] == D.HEAD_NOSE
    assert np.isnan([got.angle_deg[1], got.length_ratio[1], got.aspect[1]]).all()
    assert got.head_source[1] == D.HEAD_NONE


def test_a_horse_seen_from_behind_has_a_short_length_and_a_narrow_box():
    """Sintetic: din spate, capul e chiar deasupra cozii, aproape de ea; caseta e îngustă."""
    head, tail = np.array([[150.0, 60.0]]), np.array([[150.0, 80.0]])
    box = np.array([[110.0, 40.0, 190.0, 200.0]])
    assert D.head_tail_angle(head, tail)[0] == pytest.approx(90.0)
    assert D.head_tail_length(head, tail, box)[0] == pytest.approx(20 / 160)
    assert D.box_aspect(box)[0] == pytest.approx(0.5)
