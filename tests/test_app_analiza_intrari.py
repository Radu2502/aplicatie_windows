"""Regresii pe pagina „Analiză”: rezultatul și intervalul țin de conținutul clipului, nu de nume;
intrările ciudate dau un mesaj clar; clipurile fără interval verificat manual sunt spuse ca atare.
Clipurile încărcate sunt sintetice (fără persoane), iar rezultatele se scriu doar în foldere temporare."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ecvestru import masurare as M
from ecvestru import stocare as ST
from ecvestru import tinta as T
from ecvestru.rezultat import TargetSeed

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "pages" / "analiza.py")
CLIPS = M.dev_videos()
PROFILE = M.V.profile_segments()
EVALUATED = [v for v in CLIPS if PROFILE.get(v)]
NOT_EVALUATED = [v for v in CLIPS if not PROFILE.get(v)]
pytestmark = pytest.mark.skipif(not EVALUATED or not NOT_EVALUATED, reason="clipurile nu sunt locale")


def synthetic_clip(path: Path, frames: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def points(x: float) -> dict:
    return {"points": {"shoulder": (x, 100.0), "hip": (100.0, 200.0), "heel": (100.0, 300.0), "facing": 1.0},
            "motiv": None}


@pytest.fixture
def page(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    monkeypatch.setattr(ST, "run_candidate", lambda path: {i: points(110.0) for i in range(120)})
    monkeypatch.setattr(ST, "run_ratios", lambda path: {})
    monkeypatch.setattr(ST, "detect_horses", lambda path: {})
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def upload(at, name: str, data: bytes):
    at.radio(key="an-source").set_value("upload").run()
    at.file_uploader[0].upload(name, data, "video/mp4").run()
    return at


def interval_captions(at) -> list[str]:
    return [c.value for c in at.caption if c.value.startswith("Cadrele")]


def test_same_name_and_frame_count_never_shows_the_previous_file(page, tmp_path, monkeypatch):
    first = synthetic_clip(tmp_path / "a.mp4", 60, seed=1)
    second = synthetic_clip(tmp_path / "b.mp4", 60, seed=2)
    first_hash = M.file_hash(tmp_path / "a.mp4")
    monkeypatch.setattr(ST, "run_candidate", lambda path: {
        i: points(110.0 if M.file_hash(path) == first_hash else 90.0) for i in range(60)})

    upload(page, "acelasi_nume.mp4", first)
    page.button(key="an-run").click().run()
    assert page.metric[0].value == "+5°"

    page.file_uploader[0].upload("acelasi_nume.mp4", second, "video/mp4").run()
    assert not page.metric, "rezultatul fișierului vechi e afișat pentru fișierul nou"
    page.button(key="an-run").click().run()
    assert page.metric[0].value == "-5°"


def test_upload_named_like_a_dev_clip_gets_the_whole_clip_not_its_profile_interval(page, tmp_path):
    data = synthetic_clip(tmp_path / "a.mp4", 60, seed=1)
    upload(page, EVALUATED[0], data)
    captions = interval_captions(page)
    assert captions and captions[0].startswith("Cadrele 0–59 din 60"), captions
    assert "profil.csv" not in captions[0]


def test_single_frame_clip_gives_a_clear_message_without_exception(page, tmp_path):
    upload(page, "un_cadru.mp4", synthetic_clip(tmp_path / "a.mp4", 1, seed=1))
    assert not page.exception, page.exception
    assert len(page.error) == 1 and "un singur cadru" in page.error[0].value
    assert not interval_captions(page) and not page.button


def test_unreadable_file_gives_one_clear_message(page):
    upload(page, "stricat.mp4", b"nu e video" * 100)
    assert not page.exception, page.exception
    assert len(page.error) == 1 and "nu poate fi citit" in page.error[0].value
    assert not interval_captions(page) and not page.button


def test_clip_without_profile_interval_says_so(page):
    page.selectbox(key="an-clip").set_value(NOT_EVALUATED[0]).run()
    text = interval_captions(page)[0]
    assert "tot clipul" in text and "nu există interval verificat manual" in text
    assert "data/profil.csv" not in text

    page.selectbox(key="an-clip").set_value(EVALUATED[0]).run()
    assert "Intervalul implicit vine din data/profil.csv." in interval_captions(page)[0]


def test_upload_message_says_what_stays_on_disk(page):
    page.radio(key="an-source").set_value("upload").run()
    text = page.info[0].value
    assert all(name in text for name in ("JSON", "brut/", "vedere/", "cai/", "camera/"))
    assert "doar rezultatul" not in text


def test_uploaded_clip_gets_a_film_and_no_new_file(page, tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "run_candidate", lambda path: {i: points(110.0) for i in range(20)})
    upload(page, "incarcat.mp4", synthetic_clip(tmp_path / "a.mp4", 20, seed=1))
    page.button(key="an-run").click().run()
    assert not page.exception, page.exception
    assert len(page.get("video")) == 1
    files = [p for p in (tmp_path / "rezultate").rglob("*") if p.is_file()]
    assert sorted(p.parent.name for p in files) == ["analiza", "brut", "cai", "camera", "vedere"]
    assert all(p.suffix == ".json" for p in files)
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".mp4" and p.name != "a.mp4"]


def test_film_of_the_previous_file_disappears_with_its_result(page, tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "run_candidate", lambda path: {i: points(110.0) for i in range(20)})
    upload(page, "acelasi_nume.mp4", synthetic_clip(tmp_path / "a.mp4", 20, seed=1))
    page.button(key="an-run").click().run()
    page.file_uploader[0].upload("acelasi_nume.mp4", synthetic_clip(tmp_path / "b.mp4", 20, seed=2), "video/mp4").run()
    assert not page.get("video")


def boxes(path):
    return {f: [(70.0, 60.0, 100.0, 80.0), (470.0, 60.0, 100.0, 80.0)] for f in range(40)}


def alternating_points(path):
    out = {}
    for f in range(20):
        x = 120.0 if f % 2 else 520.0
        out[f] = {"points": {"shoulder": (x, 40.0), "hip": (x, 70.0), "heel": (x, 120.0), "facing": 1.0}, "motiv": None}
    return out


@pytest.fixture
def two_targets(page, tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "run_candidate", alternating_points)
    monkeypatch.setattr(ST, "detect_horses", boxes)
    data = synthetic_clip(tmp_path / "a.mp4", 20, seed=1)
    upload(page, "doi.mp4", data)
    page.checkbox(key="an-pick").check().run()
    assert not page.exception, page.exception
    page.session_state["an-targets"] = {M.bytes_hash(data): {
        "Călărețul 1": [TargetSeed(frame=0, x=120.0, y=100.0)], "Călărețul 2": [TargetSeed(frame=0, x=520.0, y=100.0)]}}
    page.run()
    return page


def test_picking_shows_a_frame_to_click(two_targets):
    assert two_targets.radio(key="an-slot").options == ["Călărețul 1", "Călărețul 2"]


def test_two_targets_give_two_tabs_with_two_films(two_targets):
    two_targets.button(key="an-run").click().run()
    assert not two_targets.exception, two_targets.exception
    assert [t.label for t in two_targets.tabs] == ["Călărețul 1", "Călărețul 2"]
    assert len(two_targets.get("video")) == 2
    assert [m.value for m in two_targets.metric if m.label == "Măsurat: trunchi"] == ["50%", "50%"]


def test_each_tab_says_how_many_frames_are_measured_and_why_not_the_rest(two_targets):
    two_targets.button(key="an-run").click().run()
    assert not two_targets.exception, two_targets.exception
    text = " ".join(m.value for m in two_targets.markdown)
    assert text.count("Măsurate: 10 din 20 de cadre") == 2
    assert text.count("metoda a ales alt călăreț: 10") == 2


def test_clearing_the_choice_removes_the_targets(two_targets):
    two_targets.button(key="an-clear").click().run()
    assert not two_targets.exception, two_targets.exception
    assert not list(two_targets.session_state["an-targets"].values())[0]


def test_click_far_from_any_horse_is_a_clear_error(two_targets):
    data = next(iter(two_targets.session_state["an-targets"]))
    two_targets.session_state["an-targets"] = {data: {"Călărețul 1": [TargetSeed(frame=0, x=300.0, y=330.0)]}}
    two_targets.run()
    two_targets.button(key="an-run").click().run()
    assert not two_targets.exception and "cal" in two_targets.error[0].value


def test_page_shows_where_the_target_was_lost_and_a_second_click_is_used(page, tmp_path, monkeypatch):
    n = 2 * T.max_gap_frames(30.0) + 20
    horse = (70.0, 60.0, 100.0, 80.0)
    monkeypatch.setattr(ST, "run_candidate", alternating_points)
    visible = {f: [horse] if f < 2 or f == n - 1 else [] for f in range(n)}
    monkeypatch.setattr(ST, "detect_horses", lambda path: visible)
    data = synthetic_clip(tmp_path / "a.mp4", n, seed=1)
    upload(page, "doi.mp4", data)
    page.checkbox(key="an-pick").check().run()
    seeds = [TargetSeed(frame=0, x=120.0, y=100.0)]
    page.session_state["an-targets"] = {M.bytes_hash(data): {"Călărețul 1": seeds}}
    page.run()
    page.button(key="an-run").click().run()
    assert not page.exception, page.exception
    assert any("s-a pierdut în cadrele" in w.value for w in page.warning)
    more = [*seeds, TargetSeed(frame=n - 1, x=120.0, y=100.0)]
    page.session_state["an-targets"] = {M.bytes_hash(data): {"Călărețul 1": more}}
    page.run()
    page.button(key="an-run").click().run()
    assert not page.exception, page.exception
