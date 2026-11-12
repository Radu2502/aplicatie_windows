"""Măsurarea: unghiuri, motive, rezumat, intervale, cache. Candidatul e înlocuit cu un predictor
fals (fără MediaPipe)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ecvestru import masurare as M
from ecvestru import schelet as Z
from ecvestru import stocare as ST
from ecvestru.rezultat import LIMITATION, AnalysisResult, TargetSeed

ROOT = Path(__file__).resolve().parents[1]
SHORT = "clip_152429.mp4"
have_short = (M.V.VIDEOS / SHORT).exists()


@pytest.fixture
def tmp_results(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    return tmp_path / "analiza"


def pts(sh=(100.0, 100.0), hip=(100.0, 200.0), heel=(90.0, 300.0), facing=1.0):
    return {"shoulder": sh, "hip": hip, "heel": heel, "facing": facing}


def test_round_to_is_symmetric_and_steps_of_five():
    assert [M.round_to(v) for v in (7.4, 7.5, -7.5, -2.4, 0.0, 12.6)] == [5, 10, -10, 0, 0, 15]


def test_frame_measures_angles_signs_and_reasons():
    raw = {0: {"points": pts(sh=(110.0, 100.0), heel=(110.0, 300.0)), "motiv": None},  # înainte; călcâi în față
           1: {"points": pts(heel=None), "motiv": None},                   # fără călcâi: doar trunchiul
           2: {"points": None, "motiv": "nu_e_pe_cal"},
           3: {"points": pts(sh=None), "motiv": "puncte_invizibile"}}
    f = M.frame_measures(raw, 5)
    assert f[0].trunk_deg == pytest.approx(5.71, abs=0.01) and f[0].leg_deg < 0 and f[0].reason is None
    assert f[1].trunk_deg == pytest.approx(0.0) and f[1].leg_deg is None and f[1].reason == M.REASON_HEEL
    assert (f[2].trunk_deg, f[2].reason) == (None, "nu_e_pe_cal")
    assert (f[3].trunk_deg, f[3].reason) == (None, "puncte_invizibile")
    assert f[4].reason == "calaret_negasit"                                # cadru lipsă din ieșire
    assert set(M.REASON_TEXTS) >= {r.reason for r in f if r.reason}
    assert M.REASON_TEXTS["trunchi_peste_45"] == "aplecare peste 45° sau persoană greșită"


def test_summarize_median_quantiles_coverage_and_rounded_fields():
    s = M.summarize([1.0, 2.0, 3.0, 4.0, 13.0, None, None, None, None, None])
    assert (s.n_frames, s.coverage, s.coverage_pct) == (5, 0.5, 50)
    assert s.median_deg == 3.0 and s.q25_deg == 2.0 and s.q75_deg == 4.0
    assert (s.median_rounded, s.median_abs_rounded, s.q25_rounded, s.q75_rounded) == (5, 5, 0, 5)
    assert M.summarize([-16.0, -14.0, -15.0]).median_abs_rounded == 15
    empty = M.summarize([None, None])
    assert empty.n_frames == 0 and empty.median_deg is None and empty.coverage_pct == 0


def test_unmeasured_intervals_group_consecutive_same_reason():
    raw = {i: {"points": pts(), "motiv": None} for i in range(10)}
    for i in (2, 3, 4):
        raw[i] = {"points": None, "motiv": "nu_e_pe_cal"}
    raw[5] = {"points": pts(heel=None), "motiv": None}
    raw[8] = {"points": None, "motiv": "nu_e_pe_cal"}
    got = [(u.start, u.end, u.frames, u.reason) for u in M.unmeasured_intervals(M.frame_measures(raw, 10))]
    assert got == [(2, 4, 3, "nu_e_pe_cal"), (5, 5, 1, M.REASON_HEEL), (8, 8, 1, "nu_e_pe_cal")]


@pytest.mark.skipif(not (ROOT / "data" / "split.csv").exists(), reason="lipsesc datele de dezvoltare (data/)")
def test_default_segment_and_dev_only_list():
    assert M.default_segment("clip_150826.mp4", 420) == (60, 150)     # cel mai lung segment din profil
    assert M.default_segment("necunoscut.mp4", 100) == (0, 99)
    import csv
    test_clips = {r["video"] for r in csv.DictReader((ROOT / "data" / "split.csv").open()) if r["set"] == "test"}
    assert not set(M.dev_videos()) & test_clips


def test_cache_paths_include_clip_candidate_interval_schema_and_gait(tmp_results):
    p = M.result_cache_path("clip_x.mp4", "abc123", "candidate@ff", (10, 20), "trap")
    assert p.name == "clip_x-abc123__candidate_ff__10-20__trap__q2__v1.4.json"
    assert M.raw_cache_path("clip_x.mp4", "abc123", "candidate@ff").parent.name == "brut"


@pytest.mark.skipif(not have_short, reason="clipul nu e local")
def test_analyze_with_fake_predictor_uses_segment_and_caches(tmp_results):
    calls = []

    def fake(path):
        calls.append(path)
        return {i: ({"points": pts(sh=(100.0 + i % 7, 100.0)), "motiv": None} if i % 4 else
                    {"points": None, "motiv": "nu_e_pe_cal"}) for i in range(500)}

    res = M.analyze(M.V.VIDEOS / SHORT, "trap", (100, 139), predictor=fake)
    assert res.segment == (100, 139) and len(res.frames) == 40 and res.frames[0].frame == 100
    assert res.trunk.n_frames == 30 and res.trunk.coverage_pct == 75
    assert res.candidate == M.candidate_label() and res.gait == "trap"
    again = M.analyze(M.V.VIDEOS / SHORT, "trap", (100, 139), predictor=fake)
    assert again == res and len(calls) == 1                           # rezultatul vine din cache
    M.analyze(M.V.VIDEOS / SHORT, "trap", (100, 120), predictor=fake)
    assert len(calls) == 1                                              # alt interval: ieșirea brută din cache
    assert AnalysisResult.from_json(M.result_cache_path(SHORT, res.video_hash, res.candidate, (100, 139),
                                                        "trap").read_text()) == res
    with pytest.raises(ValueError, match="Interval gol"):
        M.analyze(M.V.VIDEOS / SHORT, "trap", (150, 100), predictor=fake)


@pytest.mark.skipif(not have_short, reason="clipul nu e local")
def test_uploaded_clip_lives_only_in_a_temporary_folder(tmp_results):
    seen = []

    def fake(path):
        seen.append(Path(path))
        assert Path(path).exists() and "incarcat" not in Path(path).name
        return {i: {"points": pts(), "motiv": None} for i in range(300)}

    data = (M.V.VIDEOS / SHORT).read_bytes()
    n, fps = M.upload_info(data, "incarcat.mp4")
    assert (n, fps) == M.video_info(M.V.VIDEOS / SHORT)
    res = M.analyze_upload(data, "incarcat.mp4", "pas", (0, 49), predictor=fake)
    assert res.video == M.upload_name(res.video_hash) and res.segment == (0, 49)
    assert seen and not seen[0].exists() and not seen[0].parent.exists()      # folderul temporar s-a șters
    assert [p.name for p in tmp_results.glob("*.json")] == [M.result_cache_path(
        res.video, res.video_hash, res.candidate, (0, 49), "pas").name]        # plus cache-ul brut/


def _synthetic_clip(path: Path, frames: int) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def test_bytes_hash_matches_file_hash(tmp_path):
    path = tmp_path / "clip.bin"
    path.write_bytes(b"continut")
    assert M.bytes_hash(b"continut") == M.file_hash(path)


@pytest.mark.skipif(not M.dev_videos(), reason="lipsesc clipurile dev din data/videos/ (numele unui clip dev)")
def test_upload_without_segment_uses_the_whole_clip_even_with_a_dev_name(tmp_results, tmp_path):
    data = _synthetic_clip(tmp_path / "c.mp4", 40)
    res = M.analyze_upload(data, M.dev_videos()[0], "pas", predictor=lambda p: {})
    assert res.segment == (0, 39)


@pytest.mark.parametrize("frames,fragment", [(0, "nu poate fi citit"), (1, "un singur cadru")])
def test_upload_problem_names_the_cause(frames, fragment):
    assert fragment in M.upload_problem(frames)


def test_upload_problem_accepts_a_usable_clip():
    assert M.upload_problem(M.MIN_UPLOAD_FRAMES) is None


@pytest.mark.skipif(not (ROOT / "data" / "profil.csv").exists(), reason="lipsește data/profil.csv")
def test_has_profile_interval():
    segments = M.V.profile_segments()
    assert M.has_profile_interval(next(iter(segments)))
    assert not M.has_profile_interval("clip_inexistent.mp4")


# ---------- textul limitării ----------

def test_an_unevaluated_clip_gets_the_neutral_text(tmp_results, tmp_path):
    _synthetic_clip(tmp_path / "b.mp4", 12)
    res = M.analyze(tmp_path / "b.mp4", "pas", predictor=lambda p: {})
    assert res.limitation == M.LIMITATION_UNVERIFIED and "Verificarea a arătat" not in res.limitation
    assert "Precizia unghiurilor" in res.limitation


def test_an_upload_gets_the_neutral_text(tmp_results, tmp_path):
    data = _synthetic_clip(tmp_path / "x.mp4", 12)
    assert M.analyze_upload(data, "a.mp4", "pas", predictor=lambda p: {}).limitation == M.LIMITATION_UNVERIFIED


def test_a_chosen_target_gets_the_neutral_text():
    assert M.limitation_for("a.mp4", target=TargetSeed(frame=0, x=1.0, y=1.0)) == M.LIMITATION_UNVERIFIED


def test_a_cached_result_gets_the_text_of_today(tmp_results, tmp_path):
    _synthetic_clip(tmp_path / "b.mp4", 12)
    first = M.analyze(tmp_path / "b.mp4", "pas", predictor=lambda p: {})
    path = next(Path(tmp_results).rglob("b-*__pas__q*.json"))
    path.write_text(first.model_copy(update={"limitation": LIMITATION}).to_json(), encoding="utf-8")
    assert M.analyze(tmp_path / "b.mp4", "pas").limitation == M.LIMITATION_UNVERIFIED


# ---------- ce rămâne pe disc după o încărcare ----------

def test_the_text_names_everything_left_on_disk_after_an_upload(tmp_results, tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "run_ratios", lambda p: {})
    monkeypatch.setattr(ST, "detect_horses", lambda p: {})
    data = _synthetic_clip(tmp_path / "x.mp4", 12)
    M.analyze_upload(data, "clip.mp4", "pas", predictor=lambda p: {}, extended=True)
    root = M.results_dir()
    folders = {p.name for p in root.iterdir() if p.is_dir()}
    assert folders == {"brut", "vedere", "cai", "camera"}
    assert all(f"{name}/" in M.UPLOAD_KEPT for name in folders) and "JSON" in M.UPLOAD_KEPT
    assert list(root.glob("*.json")) and not list(root.rglob("*.mp4"))


# ---------- numele original al unui clip încărcat nu ajunge pe disc ----------

def test_the_original_name_stays_off_the_disk(tmp_results, tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "run_ratios", lambda p: {})
    monkeypatch.setattr(ST, "detect_horses", lambda p: {})
    data = _synthetic_clip(tmp_path / "x.mp4", 12)
    res = M.analyze_upload(data, "Prenume_Nume.mp4", "pas", predictor=lambda p: {}, extended=True)
    assert res.video == f"incarcat-{res.video_hash}"
    files = [p for p in tmp_results.rglob("*") if p.is_file()]
    assert len(files) == 5
    assert all(p.name.startswith(f"upload-{res.video_hash}") for p in files)
    assert not [p for p in files if "Prenume" in p.read_text(encoding="utf-8")]


def test_the_same_content_reuses_the_cache_under_another_name(tmp_results, tmp_path):
    data = _synthetic_clip(tmp_path / "x.mp4", 12)
    M.analyze_upload(data, "unu.mp4", "pas", predictor=lambda p: {})
    again = M.analyze_upload(data, "doi.mp4", "pas", predictor=lambda p: pytest.fail("trebuia să vină din cache"))
    assert again.video == M.upload_name(again.video_hash)


def test_the_temporary_copy_has_no_original_name_but_keeps_the_extension():
    seen = M.with_temp_file(b"x", "Prenume_Nume.MOV", lambda path: path.name)
    assert "Prenume" not in seen and seen.endswith(".MOV")


def test_a_set_clip_keeps_its_name_in_the_cache(tmp_results):
    assert M.clip_key("clip_x.mp4", "abc") == "clip_x-abc"
    assert M.clip_key(M.upload_name("abc"), "abc") == "upload-abc"


# ---------- scheletul complet în cache ----------

def _blank_pass(path, frames=12):
    from ecvestru import schelet as Z
    log = Z.Recording()
    source = ((i, None, 64, 48, "calaret_negasit") for i in range(frames))
    rows, raw, stats = Z.assemble(source, M.candidate_module().points_for_frame, log)
    return Z.PassResult(raw, Z.make_skeleton(rows, np.arange(frames) / 30, {"schema_version": Z.SCHEMA_VERSION,
                                                                                    "width": 64, "height": 48}))


def test_one_pass_fills_brut_and_schelet_with_the_hash_in_the_name(tmp_results, tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "run_candidate", _blank_pass)
    data = _synthetic_clip(tmp_path / "x.mp4", 12)
    res = M.analyze_upload(data, "Prenume.mp4", "pas")
    path = M.skeleton_cache_path(res.video, res.video_hash, res.candidate)
    assert path.exists() and path.parent.name == "schelet" and path.name.startswith(f"upload-{res.video_hash}")
    assert Z.load_skeleton(path).meta["video"] == res.video
    assert "Prenume" not in path.read_bytes().decode("latin-1")


def test_a_clip_with_brut_gets_its_skeleton_once(tmp_results, tmp_path, monkeypatch):
    _synthetic_clip(tmp_path / "a.mp4", 12)
    res = M.analyze(tmp_path / "a.mp4", "pas", predictor=lambda p: {})
    calls = []
    monkeypatch.setattr(ST, "extract_skeleton", lambda p: calls.append(p) or _blank_pass(p).skeleton)
    first = M.load_rider_skeleton(tmp_path / "a.mp4", res.video, res.video_hash, res.candidate)
    again = M.load_rider_skeleton(tmp_path / "a.mp4", res.video, res.video_hash, res.candidate)
    assert len(calls) == 1 and first.meta["video"] == again.meta["video"] == "a.mp4"


def test_without_an_extractor_the_film_stays_with_the_chain(tmp_results, tmp_path):
    assert M.load_rider_skeleton(tmp_path / "a.mp4", "a.mp4", "abc", "candidate@ff") is None


def test_the_text_of_an_upload_names_the_skeleton():
    assert "schelet/" in M.UPLOAD_KEPT


def test_the_text_of_an_upload_says_other_people_are_kept():
    assert "alte persoane" in M.UPLOAD_KEPT and "instructor" in M.UPLOAD_KEPT


def test_an_old_skeleton_on_disk_is_not_read_as_the_new_format(tmp_results):
    assert M.skeleton_cache_path("a.mp4", "abc", "candidate@ff").name.endswith(f"__s{Z.SCHEMA_VERSION}.npz")
    assert Z.SCHEMA_VERSION == 3


# ---------- poarta A în analiză ----------

def _gate_setup(monkeypatch, tmp_path):
    from test_calitate import skeleton
    shin = [20] * 11 + [40]
    sk = skeleton(12, shin=shin)
    sk.facing[:] = 1.0
    monkeypatch.setattr(ST, "extract_skeleton", lambda path: sk)
    monkeypatch.setattr(ST, "load_boxes", lambda *a: {i: [[0, 20, 100, 60]] for i in range(12)})
    _synthetic_clip(tmp_path / "a.mp4", 12)
    point = {"shoulder": (50.0, 10.0), "hip": (50.0, 30.0), "heel": (50.0, 95.0), "facing": 1.0}
    return lambda path: {i: {"points": point, "motiv": None} for i in range(12)}


def test_the_analysis_applies_the_gate_and_keeps_the_trunk(tmp_results, tmp_path, monkeypatch):
    predictor = _gate_setup(monkeypatch, tmp_path)
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=predictor)
    assert res.frames[11].leg_deg is None and res.frames[11].trunk_deg is not None
    assert res.leg.n_frames == 11 and res.trunk.n_frames == 12
    assert res.quality.excluded_frames == 1
    assert res.unmeasured[0].reason == M.REASON_HEEL_ON_HORSE


def test_the_json_has_the_version_of_the_quality_layer(tmp_results, tmp_path, monkeypatch):
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 11), predictor=_gate_setup(monkeypatch, tmp_path))
    data = json.loads(res.to_json())
    assert data["quality"]["version"] == 2 and data["schema_version"] == "1.4"


def test_the_gate_counts_only_the_chosen_interval(tmp_results, tmp_path, monkeypatch):
    res = M.analyze(tmp_path / "a.mp4", "pas", (0, 5), predictor=_gate_setup(monkeypatch, tmp_path))
    assert res.quality.excluded_frames == 0


def test_without_a_skeleton_the_gate_does_not_run_and_says_so(tmp_results, tmp_path):
    _synthetic_clip(tmp_path / "a.mp4", 12)
    res = M.analyze(tmp_path / "a.mp4", "pas", predictor=lambda p: {})
    assert res.quality is None and json.loads(res.to_json())["quality"] is None


def test_a_change_of_the_quality_layer_invalidates_the_cache():
    assert f"__q{M.QUALITY_VERSION}__" in M.result_cache_path("a.mp4", "h", "c@1", (0, 9), "pas").name


def test_the_rtmpose_path_has_its_own_cache_name():
    default = M.result_cache_path("a.mp4", "h", "c@1", (0, 9), "pas")
    rtmpose = M.result_cache_path("a.mp4", "h", "c@1", (0, 9), "pas", heel_source="rtmpose")
    assert default.name == "a-h__c_1__0-9__pas__q2__v1.4.json"
    from ecvestru import rtmpose as R
    assert rtmpose.name.startswith(f"a-h__c_1__0-9__pas__rtm{R.RTMPOSE_VERSION}") and rtmpose != default
