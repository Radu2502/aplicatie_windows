"""Evaluarea routerului: etichetele pe cadre din intervale, criteriile R1–R3 cu pragurile lor,
intervalul Wilson și validarea încrucișată pe clipuri. Doar date sintetice, în foldere temporare: nicio etichetă
reală."""
import csv
import json

import numpy as np
import pytest

from ecvestru import evaluare_router as ER
from ecvestru import intervale as IV
from ecvestru import router_vedere as RV

P, F, B, OB, H, U = IV.PROFILE, IV.FRONT, IV.BACK, IV.OBLIQUE, IV.HIDDEN, RV.UNDECIDED


def arr(*views):
    return np.array(views, dtype=object)


# ---------- pragurile ----------

def test_the_thresholds():
    assert (ER.R1_MAX, ER.R2_MAX, ER.R3_MIN) == (0.05, 0.15, 0.90)


# ---------- etichetele pe cadre ----------

def test_frame_labels_come_from_the_intervals_and_uncovered_frames_are_unlabeled():
    intervals = [IV.Interval(id="a", video="v.mp4", start=1, stop=2, view=P, note="", annotator="A"),
                 IV.Interval(id="b", video="v.mp4", start=4, stop=4, view=F, note="", annotator="A")]
    labels = ER.frame_labels(intervals, n_frames=6)
    assert list(labels) == [ER.UNLABELED, P, P, ER.UNLABELED, F, ER.UNLABELED]


# ---------- criteriile ----------

def test_r1_is_the_share_of_non_profile_frames_among_those_sent_to_the_profile():
    truth = arr(P, P, P, F, OB, H, P)
    pred = arr(P, P, P, P, P, U, B)
    c = ER.criteria(truth, pred)
    assert c.r1.k == 2 and c.r1.n == 5 and c.r1.value == pytest.approx(0.4) and not c.r1.passes


def test_r2_is_the_share_of_profile_frames_not_sent_to_the_profile():
    truth = arr(P, P, P, P)
    pred = arr(P, P, P, OB)
    c = ER.criteria(truth, pred)
    assert c.r2.k == 1 and c.r2.n == 4 and c.r2.value == pytest.approx(0.25) and not c.r2.passes


def test_r3_is_the_share_of_correct_views_among_front_back_frames_sent_to_the_front_back_module():
    truth = arr(F, F, B, B, F)
    pred = arr(F, B, B, B, P)            # ultimul nu ajunge la modulul față/spate
    c = ER.criteria(truth, pred)
    assert c.r3.k == 3 and c.r3.n == 4 and c.r3.value == pytest.approx(0.75) and not c.r3.passes
    assert c.front_back_routed.k == 4 and c.front_back_routed.n == 5


def test_unlabeled_frames_never_count():
    truth = arr(ER.UNLABELED, ER.UNLABELED, P)
    pred = arr(P, F, P)
    c = ER.criteria(truth, pred)
    assert c.r1.n == 1 and c.r2.n == 1 and c.r3.n == 0


def test_a_perfect_router_passes_all_three_and_the_extra_share_is_reported():
    truth = arr(P, P, F, B, OB, H)
    c = ER.criteria(truth, truth.copy())
    assert c.r1.passes and c.r2.passes and c.r3.passes and c.passes
    assert c.non_profile_into_profile.k == 0 and c.non_profile_into_profile.n == 4


def test_an_empty_criterion_has_no_value_and_does_not_pass():
    c = ER.criteria(arr(P), arr(P))
    assert c.r3.n == 0 and c.r3.value is None and not c.r3.passes and not c.passes


def test_the_wilson_interval_matches_the_known_value():
    lo, hi = ER.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert ER.wilson(0, 0) == (None, None)


# ---------- validarea încrucișată pe clipuri ----------

def synthetic_clip(views: list[tuple[str, int]], seed: int):
    """Un clip sintetic: semnalele depind de vedere (separabile), cu zgomot mic; fiecare vedere ține `n` cadre."""
    rng = np.random.default_rng(seed)
    centers = {P: (0.1, 0.0), F: (0.9, 1.0), B: (0.9, -1.0), OB: (0.5, 0.0)}
    X, y = [], []
    for view, n in views:
        a, b = centers[view]
        X.append(np.column_stack([a + 0.03 * rng.standard_normal(n), b + 0.03 * rng.standard_normal(n)]))
        y += [view] * n
    return ER.ClipData(values=np.vstack(X), labels=np.array(y, dtype=object), fps=30.0)


def synthetic_clips():
    return {f"c{i}.mp4": synthetic_clip([(P, 90), (F, 60), (B, 60), (OB, 30)], seed=i) for i in range(4)}


def test_cross_validation_holds_out_each_clip_and_a_separable_problem_passes():
    result = ER.cross_validate(synthetic_clips(), kind=RV.LOGISTIC)
    assert set(result.per_clip) == {f"c{i}.mp4" for i in range(4)}
    assert result.pooled.passes and result.set_name == ER.DEV_SET


def test_clips_of_the_same_group_are_held_out_together(monkeypatch):
    seen = []
    real_fit = RV.LogisticClassifier.fit

    def fit(self, X, y):
        seen.append(len(X))
        return real_fit(self, X, y)
    monkeypatch.setattr(RV.LogisticClassifier, "fit", fit)
    clips = synthetic_clips()
    ER.cross_validate(clips, kind=RV.LOGISTIC, groups={"c0.mp4": "g", "c1.mp4": "g", "c2.mp4": "c2", "c3.mp4": "c3"})
    assert len(seen) == 3 and sorted(seen) == [480, 720, 720]


def test_unlabeled_frames_are_left_out_of_the_training(monkeypatch):
    clips = synthetic_clips()
    clips["c0.mp4"].labels[:90] = ER.UNLABELED
    seen = []
    real_fit = RV.LogisticClassifier.fit
    monkeypatch.setattr(RV.LogisticClassifier, "fit", lambda self, X, y: seen.append(set(y)) or real_fit(self, X, y))
    ER.cross_validate(clips, kind=RV.LOGISTIC)
    assert all(ER.UNLABELED not in s for s in seen)


def test_results_per_rider_pool_the_clips_of_each_rider():
    result = ER.cross_validate(synthetic_clips(), kind=RV.LOGISTIC,
                               riders={"c0.mp4": ("C1",), "c1.mp4": ("C1",), "c2.mp4": ("C2",), "c3.mp4": ("C1", "C2")})
    assert set(result.per_rider) == {"C1", "C2"}
    assert result.per_rider["C2"].r2.n == 2 * 90


def test_the_report_says_development_and_shows_the_three_criteria_with_their_thresholds():
    result = ER.cross_validate(synthetic_clips(), kind=RV.LOGISTIC)
    text = ER.report_markdown(result)
    assert "dezvoltare" in text and "nu decid" in text
    assert "R1" in text and "≤ 5%" in text and "R2" in text and "≤ 15%" in text and "R3" in text and "≥ 90%" in text
    assert "Wilson" in text and "c0.mp4" in text


# ---------- datele după împărțire: doar din fișiere date, niciodată înainte de împărțire ----------

def write_split(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "hash", "calareti", "set"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_without_the_split_file_nothing_is_read_and_the_reason_is_given(tmp_path):
    with pytest.raises(FileNotFoundError, match="împărțire"):
        ER.split_clips(tmp_path / "impartire.csv", ER.DEV_SET)


def test_the_split_gives_only_the_clips_of_the_asked_set_with_hash_and_riders(tmp_path):
    split = write_split(tmp_path / "impartire.csv", [
        {"video": "a.mp4", "hash": "h1", "calareti": "C1", "set": "dezvoltare"},
        {"video": "b.mp4", "hash": "h2", "calareti": "C1;C2", "set": "verificare"},
        {"video": "c.mp4", "hash": "h1", "calareti": "C2", "set": "dezvoltare"}])
    clips = ER.split_clips(split, ER.DEV_SET)
    assert [c.video for c in clips] == ["a.mp4", "c.mp4"]
    assert clips[0].riders == ("C1",) and clips[1].group == "h1" == clips[0].group


def test_an_unknown_set_is_refused(tmp_path):
    split = write_split(tmp_path / "impartire.csv", [])
    with pytest.raises(ValueError, match="Set necunoscut"):
        ER.split_clips(split, "test")


# ---------- comanda pentru verificare, cu protecții ----------

def holdout_split(tmp_path):
    return write_split(tmp_path / "impartire.csv", [
        {"video": f"c{i}.mp4", "hash": f"h{i}", "calareti": "C1", "set": "dezvoltare" if i < 2 else "verificare"}
        for i in range(4)])


def holdout_args(tmp_path, split, **over):
    args = dict(split_csv=split, split_hash=ER.file_sha256(split) if split.exists() else "x", method_commit="abc",
                journal=tmp_path / "jurnal.json", report=tmp_path / "rezultat.md",
                loader=lambda clip: synthetic_clips()[clip.video], is_ancestor=lambda commit: True)
    args.update(over)
    return args


def test_the_holdout_is_refused_without_the_split_and_reads_nothing(tmp_path):
    read = []
    args = holdout_args(tmp_path, tmp_path / "impartire.csv", loader=lambda clip: read.append(clip))
    with pytest.raises(ER.HoldoutRefused, match="Lipsește împărțirea"):
        ER.run_holdout(**args)
    assert not read and not (tmp_path / "jurnal.json").exists()


def test_the_holdout_is_refused_when_the_split_hash_differs(tmp_path):
    args = holdout_args(tmp_path, holdout_split(tmp_path), split_hash="0" * 64)
    with pytest.raises(ER.HoldoutRefused, match="SHA-256"):
        ER.run_holdout(**args)
    assert not (tmp_path / "jurnal.json").exists()


def test_the_holdout_is_refused_for_a_method_commit_that_is_not_an_ancestor(tmp_path):
    args = holdout_args(tmp_path, holdout_split(tmp_path), is_ancestor=lambda commit: False)
    with pytest.raises(ER.HoldoutRefused, match="strămoș"):
        ER.run_holdout(**args)


def test_the_holdout_runs_once_writes_the_journal_and_the_report_and_refuses_a_second_run(tmp_path):
    args = holdout_args(tmp_path, holdout_split(tmp_path))
    result = ER.run_holdout(**args)
    assert result.set_name == ER.CHECK_SET and set(result.per_clip) == {"c2.mp4", "c3.mp4"}
    journal = json.loads((tmp_path / "jurnal.json").read_text(encoding="utf-8"))
    assert journal["commit_metoda"] == "abc" and journal["impartire_sha256"] == args["split_hash"]
    assert "verificare" in (tmp_path / "rezultat.md").read_text(encoding="utf-8")
    with pytest.raises(ER.HoldoutRefused, match="a rulat deja"):
        ER.run_holdout(**args)


def test_an_interrupted_holdout_still_counts_as_run(tmp_path):
    def broken(clip):
        raise RuntimeError("întrerupt")
    args = holdout_args(tmp_path, holdout_split(tmp_path), loader=broken)
    with pytest.raises(RuntimeError):
        ER.run_holdout(**args)
    with pytest.raises(ER.HoldoutRefused, match="a rulat deja"):
        ER.run_holdout(**holdout_args(tmp_path, holdout_split(tmp_path)))


def test_the_command_refuses_with_exit_2_and_a_reason(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ER, "SPLIT_CSV", tmp_path / "impartire.csv")
    monkeypatch.setattr(ER, "HOLDOUT_JOURNAL", tmp_path / "jurnal.json")
    monkeypatch.setattr(ER, "HOLDOUT_REPORT", tmp_path / "rezultat.md")
    assert ER.main(["verificare", "--hash-impartire", "x", "--commit-metoda", "HEAD"]) == 2
    assert "Refuzat" in capsys.readouterr().err


def test_the_command_needs_the_hash_and_the_commit():
    with pytest.raises(SystemExit):
        ER.main(["verificare"])
