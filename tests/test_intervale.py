"""Intervalele de vedere pe clipurile noi: salvarea doar adaugă rânduri; corecturile și ștergerile sunt rânduri noi;
contează ultimul rând al fiecărui (adnotator, id); al doilea adnotator sau a doua completare nu suprascrie nimic."""
import csv

import pytest

from ecvestru import intervale as IV


@pytest.fixture
def path(tmp_path):
    return tmp_path / IV.ANSWER_FILE


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_file_is_created_only_at_the_first_save(path):
    assert not path.exists()
    assert IV.current_intervals(path, "adnotator1") == []
    IV.save_interval(path, "a.mp4", 10, 40, "profil", "adnotator1", n_frames=100)
    assert [r["actiune"] for r in rows(path)] == [IV.ADDED]
    assert list(rows(path)[0]) == IV.FIELDS


def test_a_saved_interval_is_read_back_with_its_fields(path):
    saved = IV.save_interval(path, "a.mp4", 10, 40, "fata", "adnotator1", note="doi cai", n_frames=100)
    (got,) = IV.current_intervals(path, "adnotator1", "a.mp4")
    assert got == saved
    assert (got.video, got.start, got.stop, got.view, got.note, got.annotator) == ("a.mp4", 10, 40, "fata",
                                                                                     "doi cai", "adnotator1")


def test_a_correction_adds_a_row_and_keeps_the_first_one(path):
    first = IV.save_interval(path, "a.mp4", 10, 40, "profil", "adnotator1", n_frames=100)
    IV.save_interval(path, "a.mp4", 12, 45, "oblic", "adnotator1", interval_id=first.id, n_frames=100)
    saved = rows(path)
    assert [(r["start"], r["vedere"], r["actiune"]) for r in saved] == [("10", "profil", IV.ADDED),
                                                                        ("12", "oblic", IV.CORRECTED)]
    (current,) = IV.current_intervals(path, "adnotator1")
    assert (current.id, current.start, current.stop, current.view) == (first.id, 12, 45, "oblic")


def test_a_deletion_adds_a_row_and_the_interval_is_no_longer_current(path):
    first = IV.save_interval(path, "a.mp4", 10, 40, "profil", "adnotator1", n_frames=100)
    IV.delete_interval(path, first.id, "adnotator1")
    assert [r["actiune"] for r in rows(path)] == [IV.ADDED, IV.DELETED]
    assert IV.current_intervals(path, "adnotator1") == []


def test_a_second_annotator_does_not_overwrite_the_first(path):
    IV.save_interval(path, "a.mp4", 10, 40, "profil", "adnotator1", n_frames=100)
    IV.save_interval(path, "a.mp4", 10, 40, "spate", "Ana", n_frames=100)   # aceleași cadre: alt adnotator
    assert [i.view for i in IV.current_intervals(path, "adnotator1")] == ["profil"]
    assert [i.view for i in IV.current_intervals(path, "ana")] == ["spate"]
    assert len(rows(path)) == 2


def test_a_second_completion_by_the_same_annotator_keeps_the_first_rows(path):
    first = IV.save_interval(path, "a.mp4", 0, 49, "profil", "adnotator1", n_frames=100)
    IV.delete_interval(path, first.id, "adnotator1")
    second = IV.save_interval(path, "a.mp4", 0, 49, "oblic", "adnotator1", n_frames=100)
    assert second.id != first.id
    assert [r["vedere"] for r in rows(path)] == ["profil", "profil", "oblic"]
    assert [i.view for i in IV.current_intervals(path, "adnotator1")] == ["oblic"]


def test_another_annotators_id_cannot_be_corrected_or_deleted(path):
    first = IV.save_interval(path, "a.mp4", 10, 40, "profil", "adnotator1", n_frames=100)
    with pytest.raises(ValueError):
        IV.save_interval(path, "a.mp4", 10, 40, "oblic", "Ana", interval_id=first.id, n_frames=100)
    with pytest.raises(ValueError):
        IV.delete_interval(path, first.id, "Ana")
    assert len(rows(path)) == 1


def test_overlapping_intervals_of_the_same_annotator_are_refused(path):
    IV.save_interval(path, "a.mp4", 10, 40, "profil", "adnotator1", n_frames=100)
    with pytest.raises(ValueError, match="se suprapune"):
        IV.save_interval(path, "a.mp4", 40, 60, "oblic", "adnotator1", n_frames=100)
    IV.save_interval(path, "a.mp4", 41, 60, "oblic", "adnotator1", n_frames=100)
    IV.save_interval(path, "b.mp4", 10, 40, "oblic", "adnotator1", n_frames=100)   # alt clip
    assert len(rows(path)) == 3


def test_a_correction_may_overlap_its_own_old_version(path):
    first = IV.save_interval(path, "a.mp4", 10, 40, "profil", "adnotator1", n_frames=100)
    IV.save_interval(path, "a.mp4", 5, 45, "profil", "adnotator1", interval_id=first.id, n_frames=100)
    assert IV.current_intervals(path, "adnotator1")[0].start == 5


@pytest.mark.parametrize("start,stop,view,annotator", [
    (40, 10, "profil", "adnotator1"), (-1, 10, "profil", "adnotator1"), (10, 100, "profil", "adnotator1"),
    (10, 40, "lateral", "adnotator1"), (10, 40, "profil", "  ")])
def test_bad_intervals_are_refused_and_nothing_is_written(path, start, stop, view, annotator):
    with pytest.raises(ValueError):
        IV.save_interval(path, "a.mp4", start, stop, view, annotator, n_frames=100)
    assert not path.exists()


def test_current_intervals_are_sorted_by_clip_and_start(path):
    IV.save_interval(path, "b.mp4", 0, 5, "profil", "adnotator1", n_frames=100)
    IV.save_interval(path, "a.mp4", 50, 60, "fata", "adnotator1", n_frames=100)
    IV.save_interval(path, "a.mp4", 0, 9, "spate", "adnotator1", n_frames=100)
    assert [(i.video, i.start) for i in IV.current_intervals(path, "adnotator1")] == [("a.mp4", 0), ("a.mp4", 50),
                                                                                ("b.mp4", 0)]


def test_covered_frames_count_each_frame_once(path):
    IV.save_interval(path, "a.mp4", 0, 9, "spate", "adnotator1", n_frames=100)
    IV.save_interval(path, "a.mp4", 20, 29, "profil", "adnotator1", n_frames=100)
    assert IV.covered_frames(IV.current_intervals(path, "adnotator1", "a.mp4")) == 20


def test_annotators_lists_who_has_current_intervals(path):
    IV.save_interval(path, "a.mp4", 0, 9, "spate", "adnotator1", n_frames=100)
    gone = IV.save_interval(path, "a.mp4", 0, 9, "fata", "Ana", n_frames=100)
    IV.delete_interval(path, gone.id, "Ana")
    assert IV.annotators(path) == ["adnotator1"]


def test_the_view_at_a_frame_comes_from_the_interval_that_covers_it(path):
    IV.save_interval(path, "a.mp4", 0, 9, "spate", "adnotator1", n_frames=100)
    current = IV.current_intervals(path, "adnotator1", "a.mp4")
    assert IV.view_at(current, 9) == "spate" and IV.view_at(current, 10) is None


def test_the_timeline_colours_each_interval_by_its_view_and_marks_the_current_frame(path):
    IV.save_interval(path, "a.mp4", 0, 49, "profil", "adnotator1", n_frames=100)
    image = IV.timeline_image(IV.current_intervals(path, "adnotator1"), 100, current=75, width=200)
    assert image.shape == (IV.TIMELINE_HEIGHT, 200, 3)
    assert tuple(image[1, 10]) == IV.VIEW_RGB["profil"]
    assert tuple(image[1, 120]) == IV.UNLABELED_RGB
    assert tuple(image[1, 150]) == IV.CURRENT_RGB
    assert all(IV.VIEWS[v] in IV.TIMELINE_LEGEND for v in IV.VIEWS)
