"""Scheletul calului (opțiunea „Scheletul calului”, oprită implicit): oprită, rezultatul și filmul sunt cele
obișnuite; pornită, câmpul `horse_keypoints` are 17 puncte AP-10K pe fiecare cadru analizat (schema 1.6), pe metoda
veche și pe urmărirea țintei; cache-ul etapei nu recalculează și se invalidează la schimbarea versiunii. Fără model:
estimatorul AP-10K e simulat (HorsePose.from_run), fără detecție și fără MediaPipe."""
import inspect
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from ecvestru import analiza_urmarita as AU
from ecvestru import cal_puncte as CP
from ecvestru import masurare as M
from ecvestru import progres as P
from ecvestru import schelet_cal as SC
from ecvestru import stocare as ST
from ecvestru import suprapunere as S
from ecvestru import urmarire_multi as U
from ecvestru.rezultat import (
    COMBINED_SCHEMA_VERSION,
    HORSE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AnalysisResult,
    FrameMeasure,
    HorseFrame,
    HorsePoint,
    TargetSeed,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "app" / "pages" / "analiza.py")
WIDTH, HEIGHT, N = 160, 120, 6
BOX_XYWH = [20.0, 30.0, 100.0, 60.0]                 # caseta calului din detector (x, y, w, h)
BOX_XYXY = (20.0, 30.0, 120.0, 90.0)
NO_HORSE = N - 1                                      # ultimul cadru n-are casetă de cal
LOW = (0,)                                            # punctul 0 (left_eye) are scorul sub prag
POINT = {"points": {"shoulder": (60.0, 10.0), "hip": (60.0, 25.0), "heel": (60.0, 40.0), "facing": 1.0},
         "motiv": None}


def synthetic_clip(path: Path, frames: int = N) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (WIDTH, HEIGHT))
    for i in range(frames):
        writer.write(np.full((HEIGHT, WIDTH, 3), 20 * i % 255, np.uint8))
    writer.release()
    return path


def fake_run(tensor):
    """SimCC simulat: punctul i la (60 + 8i, 100 + 4i) pe intrarea 256×256, scor 0,9 (0,1 pentru cele din LOW)."""
    x = np.zeros((1, 17, CP.INPUT_W * CP.SPLIT_RATIO), np.float32)
    y = np.zeros((1, 17, CP.INPUT_H * CP.SPLIT_RATIO), np.float32)
    for i in range(17):
        peak = 0.1 if i in LOW else 0.9
        x[0, i, (60 + 8 * i) * CP.SPLIT_RATIO] = peak
        y[0, i, (100 + 4 * i) * CP.SPLIT_RATIO] = peak
    return x, y


class Counting:
    """Estimatorul simulat, cu numărul de apeluri (câte cadre a calculat)."""

    def __init__(self):
        self.calls = 0
        self.pose = CP.HorsePose.from_run(fake_run)

    def __call__(self, image, box):
        self.calls += 1
        return self.pose(image, box)


@pytest.fixture
def clip(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    monkeypatch.setattr(ST, "run_candidate", lambda path: {i: POINT for i in range(N)})
    monkeypatch.setattr(ST, "run_ratios", lambda path: {})
    monkeypatch.setattr(ST, "detect_horses", lambda path: {i: [BOX_XYWH] for i in range(N) if i != NO_HORSE})
    return synthetic_clip(tmp_path / "c.mp4")


def fresh(tmp_path, monkeypatch, folder: str) -> None:
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / folder))


def check_horse(data: dict, frames: list[int]) -> None:
    assert data["schema_version"] == HORSE_SCHEMA_VERSION == "1.6"
    horse = data["horse_keypoints"]
    assert horse["version"] == SC.VERSION and horse["model"] == CP.MODEL_ZIP_SHA256[:8]
    assert horse["score_min"] == CP.SCORE_MIN
    assert [f["frame"] for f in horse["frames"]] == frames
    for f in horse["frames"]:
        if f["frame"] == NO_HORSE:
            assert f["points"] is None
            continue
        assert [p["name"] for p in f["points"]] == list(CP.AP10K) and len(f["points"]) == 17
        for i, p in enumerate(f["points"]):
            if i in LOW:
                assert p["x"] is None and p["y"] is None and p["score"] == pytest.approx(0.1)
            else:
                assert 0 <= p["x"] <= WIDTH and 0 <= p["y"] <= HEIGHT and p["score"] == pytest.approx(0.9)


# ---------- implicit oprită; oprită = rezultatul obișnuit ----------

def test_the_option_is_off_by_default_everywhere():
    assert SC.APP_DEFAULT_HORSE_SKELETON is False
    for fn in (M.analyze, M.analyze_upload, AU.analyze):
        assert inspect.signature(fn).parameters["horse_skeleton"].default is False


def test_off_the_json_is_the_one_from_before_without_the_field(clip, tmp_path, monkeypatch):
    before = M.analyze(clip, "pas", (1, 5)).to_json()
    fresh(tmp_path, monkeypatch, "oprit")
    estimator = Counting()
    off = M.analyze(clip, "pas", (1, 5), horse_skeleton=False, horse_estimator=estimator)
    assert off.to_json() == before and estimator.calls == 0
    data = json.loads(off.to_json())
    assert "horse_keypoints" not in data and data["schema_version"] == SCHEMA_VERSION == "1.4"
    assert not (tmp_path / "oprit" / "analiza" / "cal").exists()


# ---------- pornită: metoda veche ----------

def test_on_the_old_method_adds_seventeen_named_points_per_analysed_frame(clip):
    off = M.analyze(clip, "pas", (1, 5))
    on = M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=Counting())
    data = json.loads(on.to_json())
    check_horse(data, [1, 2, 3, 4, 5])
    rest = {k: v for k, v in data.items() if k not in ("schema_version", "horse_keypoints")}
    assert rest == {k: v for k, v in json.loads(off.to_json()).items() if k != "schema_version"}
    assert AnalysisResult.from_json(on.to_json()) == on


def test_the_old_method_uses_the_box_of_the_horse_under_the_rider(clip):
    on = M.analyze(clip, "pas", (0, 1), horse_skeleton=True, horse_estimator=Counting())
    run = SC.old_method_points(clip, on.video, on.video_hash, on.candidate)
    assert tuple(run.boxes[0]) == BOX_XYXY and np.isnan(run.boxes[NO_HORSE]).all()


def test_an_uploaded_clip_gets_the_horse_too(clip):
    data = clip.read_bytes()
    on = M.analyze_upload(data, "x.mp4", "pas", (0, 2), horse_skeleton=True, horse_estimator=Counting())
    check_horse(json.loads(on.to_json()), [0, 1, 2])


def test_a_cached_result_still_gets_the_horse_and_the_result_cache_is_the_old_one(clip, tmp_path):
    M.analyze(clip, "pas", (1, 5))
    files = sorted((tmp_path / "rezultate" / "analiza").glob("*.json"))
    on = M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=Counting())
    assert on.horse_keypoints is not None
    assert sorted((tmp_path / "rezultate" / "analiza").glob("*.json")) == files
    assert json.loads(files[0].read_text(encoding="utf-8"))["schema_version"] == "1.4"


def test_the_stage_is_announced_for_the_progress_bar(clip):
    stages = P.Stages()
    M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=Counting(), stages=stages)
    assert P.HORSE_SKELETON in [t.name for t in stages.durations()]


# ---------- cache-ul etapei ----------

def test_the_old_method_stage_is_cached_and_a_new_version_recomputes(clip, monkeypatch):
    first = Counting()
    M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=first)
    assert first.calls == N - 1                       # fiecare cadru cu casetă, o dată, pe tot clipul
    again = Counting()
    M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=again)
    M.analyze(clip, "pas", (2, 3), horse_skeleton=True, horse_estimator=again)
    assert again.calls == 0
    monkeypatch.setattr(CP, "CAL_VERSION", CP.CAL_VERSION + 1)
    newer = Counting()
    M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=newer)
    assert newer.calls == N - 1


def test_the_tracked_stage_is_cached_by_clip_target_and_version(clip, monkeypatch):
    seed = TargetSeed(frame=0, x=70.0, y=60.0)
    boxes = np.full((N, 4), np.nan)
    boxes[:NO_HORSE] = BOX_XYXY
    first = Counting()
    run = SC.tracked_points(clip, "c.mp4", "h1", AU.CANDIDATE_LABEL, seed, lambda: boxes, first)
    assert first.calls == N - 1 and run.inferred == N - 1
    path = SC.tracked_cache_path("c.mp4", "h1", AU.CANDIDATE_LABEL, seed)
    assert path.exists() and "@" not in path.name and path.name.isascii()
    assert f"__h{SC.VERSION}_" in path.name and f"__t{seed.frame}-70-60__" in path.name

    def no_boxes():
        raise AssertionError("din cache, fără alegerea țintei")
    again = Counting()
    cached = SC.tracked_points(clip, "c.mp4", "h1", AU.CANDIDATE_LABEL, seed, no_boxes, again)
    assert again.calls == 0 and np.array_equal(cached.points, run.points, equal_nan=True)
    other = SC.tracked_cache_path("c.mp4", "h1", AU.CANDIDATE_LABEL, TargetSeed(frame=0, x=10.0, y=10.0))
    assert other != path
    monkeypatch.setattr(SC, "VERSION", SC.VERSION + 1)
    newer = Counting()
    SC.tracked_points(clip, "c.mp4", "h1", AU.CANDIDATE_LABEL, seed, lambda: boxes, newer)
    assert newer.calls == N - 1


# ---------- pornită: urmărirea țintei ----------

def fake_tracks() -> tuple[U.ClipTracks, list]:
    tracks = U.ClipTracks(fps=30.0, n_frames=N, horses=[{7: BOX_XYXY, 8: (0.0, 0.0, 30.0, 30.0)} for _ in range(N)],
                          people=[{} for _ in range(N)], keypoints=[{} for _ in range(N)])
    return tracks, [{} for _ in range(N)]


def fake_choice() -> U.Choice:
    return U.Choice(categories=[""] * N, horse_ids=[7] * NO_HORSE + [None], rider_ids=[None] * N, seed_frame=0,
                    lost_from=None, horse_switches=0, rider_switches=0)


@pytest.fixture
def tracked(clip, monkeypatch):
    monkeypatch.setattr(AU, "heavy_pass", lambda *a, **k: fake_tracks())
    chosen = []
    monkeypatch.setattr(AU, "choose", lambda *a: chosen.append(1) or fake_choice())
    return clip, chosen


def test_the_target_horse_boxes_follow_the_chosen_horse():
    tracks, _ = fake_tracks()
    boxes = AU.target_horse_boxes(tracks, fake_choice())
    assert [tuple(b) for b in boxes[:NO_HORSE]] == [BOX_XYXY] * NO_HORSE and np.isnan(boxes[NO_HORSE]).all()


def test_on_the_tracked_method_adds_the_points_of_the_tracked_horse(tracked, tmp_path, monkeypatch):
    clip, chosen = tracked
    seed = TargetSeed(frame=0, x=70.0, y=60.0)
    off = AU.analyze(clip, "pas", (0, 5), click=seed)
    assert "horse_keypoints" not in json.loads(off.to_json()) and off.schema_version == "1.4"
    fresh(tmp_path, monkeypatch, "pornit")
    estimator = Counting()
    on = AU.analyze(clip, "pas", (0, 5), click=seed, horse_skeleton=True, horse_estimator=estimator)
    data = json.loads(on.to_json())
    check_horse(data, [0, 1, 2, 3, 4, 5])
    assert {k: v for k, v in data.items() if k not in ("schema_version", "horse_keypoints")} == {
        k: v for k, v in json.loads(off.to_json()).items() if k != "schema_version"}
    assert estimator.calls == N - 1 and len(chosen) == 2       # o alegere pe rulare; etapa refolosește alegerea

    again = Counting()
    monkeypatch.setattr(AU, "choose", lambda *a: pytest.fail("rezultat și etapă din cache: fără alegerea țintei"))
    cached = AU.analyze(clip, "pas", (0, 5), click=seed, horse_skeleton=True, horse_estimator=again)
    assert again.calls == 0 and cached.to_json() == on.to_json()


# ---------- modelul de date ----------

def horse_frame(n: int = 17) -> HorseFrame:
    return HorseFrame(frame=0, points=[HorsePoint(name=name, x=1.0, y=2.0, score=0.9) for name in CP.AP10K[:n]])


def test_the_schema_version_follows_the_horse_field(clip):
    plain = M.analyze(clip, "pas", (1, 2))
    on = M.analyze(clip, "pas", (1, 2), horse_skeleton=True, horse_estimator=Counting())
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({**plain.model_dump(), "schema_version": "1.6"})
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({**on.model_dump(), "schema_version": "1.4"})
    assert AnalysisResult.model_validate(on.model_dump()) == on
    with pytest.raises(ValidationError):    # 1.5 e rezultatul cu vederi, nu acesta
        AnalysisResult.model_validate({**on.model_dump(), "schema_version": COMBINED_SCHEMA_VERSION})
    assert COMBINED_SCHEMA_VERSION == "1.5" and HORSE_SCHEMA_VERSION != COMBINED_SCHEMA_VERSION


def test_a_frame_has_exactly_seventeen_points_or_none():
    assert len(horse_frame().points) == 17 and HorseFrame(frame=3).points is None
    with pytest.raises(ValidationError):
        horse_frame(16)


@pytest.mark.parametrize(("xy", "score", "shown"), [((10.0, 20.0), 0.9, True), ((10.0, 20.0), 0.29, False),
                                                    ((-1.0, 20.0), 0.9, False), ((10.0, HEIGHT + 1), 0.9, False),
                                                    ((np.nan, np.nan), 0.0, False), ((WIDTH, HEIGHT), 0.3, True)])
def test_a_point_has_coordinates_only_above_the_threshold_and_inside_the_image(xy, score, shown):
    p = SC.point("nose", np.array(xy), score, WIDTH, HEIGHT)
    assert (p.x is not None and p.y is not None) == shown and p.name == "nose"
    assert p.score == pytest.approx(score)


def test_the_bones_are_the_ap10k_links_between_known_points():
    assert len(CP.AP10K_BONES) == 17 and len(set(CP.AP10K_BONES)) == 17
    assert {n for bone in CP.AP10K_BONES for n in bone} == set(CP.AP10K)


# ---------- filmul ----------

def test_the_film_draws_the_horse_in_its_own_colour_only_when_asked():
    frame = np.full((HEIGHT, WIDTH, 3), 30, np.uint8)
    measure = FrameMeasure(frame=0, trunk_deg=None, leg_deg=None, reason="x")
    plain = S.overlay_frame(frame, None, measure)
    assert np.array_equal(S.overlay_frame(frame, None, measure, horse=None), plain)
    points = [None] * 17
    points[CP.AP10K.index("neck")], points[CP.AP10K.index("root_of_tail")] = (40.0, 60.0), (120.0, 60.0)
    drawn = S.overlay_frame(frame, None, measure, horse=points)
    assert (drawn == S.HORSE_BGR).all(axis=2).sum() > 0 and not (plain == S.HORSE_BGR).all(axis=2).any()
    assert S.HORSE_BGR not in (S.SKELETON_BGR, S.TRUNK_BGR, S.LEG_BGR, S.VERTICAL_BGR, S.HIGH_BGR, S.MID_BGR,
                               S.LOW_BGR, S.REJECTED_BGR)


def test_the_clip_film_passes_the_horse_only_when_the_result_has_it(clip, monkeypatch):
    calls = []
    monkeypatch.setattr(S, "film_bytes", lambda *a, **k: calls.append(k) or b"film")
    S.clip_film(clip, M.analyze(clip, "pas", (1, 5)))
    on = M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=Counting())
    S.clip_film(clip, on)
    assert calls[0] == {}
    assert sorted(calls[1]["horse"]) == [1, 2, 3, 4]               # cadrul fără cal nu se desenează
    assert calls[1]["horse"][1][0] is None and len(calls[1]["horse"][1]) == 17


def probe(data: bytes, tmp_path: Path) -> str:
    path = tmp_path / "film.mp4"
    path.write_bytes(data)
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                           "stream=codec_name,nb_read_frames", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_the_film_with_the_horse_works_without_memfd(clip, tmp_path, monkeypatch):
    """Ca tests/test_portabil.py: pe Windows și Mac nu există memfd."""
    monkeypatch.delattr(os, "memfd_create", raising=False)
    on = M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=Counting())
    assert probe(S.clip_film(clip, on), tmp_path) == "h264,5"


def test_the_film_with_the_horse_works_with_ffmpeg_only_from_the_path(clip, tmp_path, monkeypatch):
    import imageio_ffmpeg
    real = imageio_ffmpeg.get_ffmpeg_exe()

    def missing():
        raise RuntimeError("fără binar")
    monkeypatch.setattr(S, "FFMPEG", None)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", missing)
    monkeypatch.setattr("shutil.which", lambda name: real if name == "ffmpeg" else None)
    on = M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=Counting())
    assert probe(S.clip_film(clip, on), tmp_path) == "h264,5"


def horse_pixels(data: bytes, tmp_path: Path) -> list[int]:
    """Pe fiecare cadru al filmului decodat: câți pixeli sunt aproape de turcoazul calului (H.264 nu e exact)."""
    path = tmp_path / "film_cal.mp4"
    path.write_bytes(data)
    capture, counts = cv2.VideoCapture(str(path)), []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        counts.append(int((np.abs(image.astype(int) - np.array(S.HORSE_BGR)).max(axis=2) < 40).sum()))
    capture.release()
    return counts


def test_the_decoded_film_shows_the_horse_only_with_the_option_on(clip, tmp_path):
    """Cap-coadă: filmul real (ffmpeg), decodat; cu opțiunea pornită punctele calului se văd pe cadrele cu cal, oprită
    nu apare niciun pixel turcoaz."""
    off = horse_pixels(S.clip_film(clip, M.analyze(clip, "pas", (1, 5))), tmp_path)
    on_result = M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=Counting())
    assert AnalysisResult.model_validate_json(on_result.to_json()).schema_version == "1.6"
    on = horse_pixels(S.clip_film(clip, on_result), tmp_path)
    assert len(off) == len(on) == 5 and max(off) == 0
    assert all(count > 20 for count in on[:4]) and on[4] == 0   # cadrul 5 (NO_HORSE) n-are cal


def test_without_ffmpeg_the_horse_film_is_a_film_error(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "FFMPEG", str(tmp_path / "nu-exista" / "ffmpeg"))
    on = M.analyze(clip, "pas", (1, 5), horse_skeleton=True, horse_estimator=Counting())
    with pytest.raises(S.FilmError):
        S.clip_film(clip, on)


# ---------- pagina ----------

def page_with_upload(monkeypatch, tmp_path, seen: list) -> AppTest:
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    monkeypatch.setattr(ST, "run_candidate", lambda path: {i: POINT for i in range(N)})
    monkeypatch.setattr(ST, "run_ratios", lambda path: {})
    monkeypatch.setattr(ST, "detect_horses", lambda path: {i: [BOX_XYWH] for i in range(N)})
    original = M.analyze

    def spy(*args, **kwargs):
        seen.append(kwargs.get("horse_skeleton", False))
        horse = {"horse_estimator": Counting()} if kwargs.get("horse_skeleton") else {}
        return original(*args, **{**kwargs, **horse})
    monkeypatch.setattr(M, "analyze", spy)
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    at.radio(key="an-source").set_value("upload").run()
    at.file_uploader[0].upload("clip.mp4", synthetic_clip(tmp_path / "u.mp4").read_bytes(), "video/mp4").run()
    return at


def test_the_page_option_is_off_by_default_and_the_analysis_is_the_one_from_before(monkeypatch, tmp_path):
    seen = []
    at = page_with_upload(monkeypatch, tmp_path, seen)
    box = at.checkbox(key="an-horse-skeleton")
    assert box.value is False and box.label == "Scheletul calului"
    at.button(key="an-run").click().run()
    assert not at.exception, at.exception
    assert seen == [False] and not any(c.value == S.HORSE_LEGEND for c in at.caption)


def test_the_page_option_on_asks_for_the_horse_and_explains_the_film(monkeypatch, tmp_path):
    seen = []
    at = page_with_upload(monkeypatch, tmp_path, seen)
    at.checkbox(key="an-horse-skeleton").check().run()
    at.button(key="an-run").click().run()
    assert not at.exception, at.exception
    assert seen == [True] and any(c.value == S.HORSE_LEGEND for c in at.caption)
    assert len(at.get("video")) == 1
