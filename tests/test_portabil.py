"""Aplicația pe Windows și Mac: filmul fără memfd și /proc, ffmpeg din imageio-ffmpeg sau din PATH, analiza merge și
fără film, sursa „set” doar cu data/split.csv, clipul încărcat cu nume ASCII, fără căi Linux în cod."""
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ecvestru import masurare as M
from ecvestru import stocare as ST
from ecvestru import suprapunere as S
from ecvestru.rezultat import FrameMeasure

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "app" / "pages" / "analiza.py")


def synthetic_clip(path: Path, frames: int) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), 20 * i % 255, np.uint8))
    writer.release()
    return path


def measure(i: int) -> FrameMeasure:
    return FrameMeasure(frame=i, trunk_deg=0.0, leg_deg=0.0, reason=None)


def probe(data: bytes, tmp_path: Path) -> str:
    path = tmp_path / "film.mp4"
    path.write_bytes(data)
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                           "stream=codec_name,nb_read_frames", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True, check=True).stdout.strip()


# ---------- a) filmul fără memfd, /proc sau pass_fds; fișierul temporar se șterge ----------

def test_the_film_works_without_memfd(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "memfd_create", raising=False)
    clip = synthetic_clip(tmp_path / "c.mp4", 4)
    assert probe(S.film_bytes(clip, [measure(i) for i in range(4)], {}, 30.0), tmp_path) == "h264,4"


def test_the_temporary_film_file_is_removed(tmp_path, monkeypatch):
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(temp))
    clip = synthetic_clip(tmp_path / "c.mp4", 3)
    S.film_bytes(clip, [measure(i) for i in range(3)], {}, 30.0)
    assert not list(temp.iterdir())


# ---------- b) ffmpeg din imageio-ffmpeg, apoi din PATH ----------

def test_ffmpeg_comes_from_imageio_ffmpeg_first(monkeypatch):
    import imageio_ffmpeg
    monkeypatch.setattr(S, "FFMPEG", None)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/din/pachet/ffmpeg")
    assert S.ffmpeg_path() == "/din/pachet/ffmpeg"


def test_ffmpeg_falls_back_to_the_path_and_then_to_a_clear_error(monkeypatch):
    import imageio_ffmpeg

    def missing():
        raise RuntimeError("fără binar")
    monkeypatch.setattr(S, "FFMPEG", None)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", missing)
    monkeypatch.setattr("shutil.which", lambda name: "/din/path/ffmpeg" if name == "ffmpeg" else None)
    assert S.ffmpeg_path() == "/din/path/ffmpeg"
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(S.FilmError, match="ffmpeg"):
        S.ffmpeg_path()


# ---------- c) orice eroare la film devine FilmError ----------

def test_a_missing_ffmpeg_binary_is_a_film_error(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "FFMPEG", str(tmp_path / "nu-exista" / "ffmpeg"))
    clip = synthetic_clip(tmp_path / "c.mp4", 3)
    with pytest.raises(S.FilmError):
        S.film_bytes(clip, [measure(i) for i in range(3)], {}, 30.0)


def test_a_film_error_is_a_runtime_error_so_the_page_keeps_the_analysis():
    assert issubclass(S.FilmError, RuntimeError)


def page_with_upload(monkeypatch, tmp_path) -> AppTest:
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    point = {"points": {"shoulder": (30.0, 10.0), "hip": (30.0, 25.0), "heel": (30.0, 40.0), "facing": 1.0},
             "motiv": None}
    monkeypatch.setattr(ST, "run_candidate", lambda path: {i: point for i in range(6)})
    monkeypatch.setattr(ST, "run_ratios", lambda path: {})
    monkeypatch.setattr(ST, "detect_horses", lambda path: {})
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    at.radio(key="an-source").set_value("upload").run()
    at.file_uploader[0].upload("clip.mp4", synthetic_clip(tmp_path / "u.mp4", 6).read_bytes(), "video/mp4").run()
    at.button(key="an-run").click().run()
    return at


@pytest.mark.parametrize("break_film", ["ffmpeg", "memfd"])
def test_the_analysis_shows_without_a_film(monkeypatch, tmp_path, break_film):
    if break_film == "ffmpeg":
        monkeypatch.setattr(S, "FFMPEG", None)
        monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: str(tmp_path / "lipsa"))
    else:
        monkeypatch.delattr(os, "memfd_create", raising=False)
    at = page_with_upload(monkeypatch, tmp_path)
    assert not at.exception, at.exception
    assert len(at.metric) == 4 and at.subheader
    if break_film == "ffmpeg":
        assert any("Filmul" in w.value for w in at.warning) and not at.get("video")
    else:
        assert len(at.get("video")) == 1


# ---------- d) „Din setul de dezvoltare” doar cu data/split.csv ----------

def test_the_dev_set_exists_only_with_its_split_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ST, "SPLIT_CSV", tmp_path / "split.csv")
    assert not M.has_dev_set()
    (tmp_path / "split.csv").write_text("video,calaret,set\n", encoding="utf-8")
    assert M.has_dev_set()


def test_without_the_dev_set_the_page_starts_on_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(ST, "SPLIT_CSV", tmp_path / "nu-exista.csv")
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.radio(key="an-source").options == ["Încarc un clip"]
    assert at.file_uploader


# ---------- e) clipul încărcat: nume ASCII, folder temporar cu cale ASCII ----------

@pytest.mark.parametrize(("name", "suffix"), [("Călăreț în față.MOV", ".MOV"), ("clip", ".mp4"),
                                              ("a.mp4 ", ".mp4"), ("x.ăvi", ".mp4"), ("ciudat.m4v", ".m4v")])
def test_the_uploaded_clip_gets_a_simple_ascii_name(name, suffix):
    seen = M.with_temp_file(b"x", name, lambda path: path)
    assert seen.name == f"{M.TEMP_STEM}{suffix}" and seen.name.isascii()
    assert not seen.exists()


def test_the_temporary_folder_avoids_a_non_ascii_temp_dir(monkeypatch, tmp_path):
    accented = tmp_path / "Ștefan"
    accented.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(accented))
    assert M.temp_root() == M.ascii_fallback_temp()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    assert M.temp_root() == tmp_path


# ---------- f) fără căi Linux scrise de mână în cod ----------

LINUX_ONLY = re.compile(r"memfd_create|/proc/|pass_fds|[\"']/tmp\b|/dev/shm")


@pytest.mark.parametrize("path", sorted((ROOT / "ecvestru").glob("*.py")) + sorted((ROOT / "app").rglob("*.py")),
                         ids=lambda p: p.name)
def test_no_linux_only_paths_in_the_app(path):
    assert not LINUX_ONLY.findall(path.read_text(encoding="utf-8"))


# ---------- e) folderul aplicației: modelele MediaPipe se deschid după cale ----------

@pytest.mark.parametrize(("root", "problem"), [(Path("C:/postura"), False), (Path("C:/Program Files/postura"), False),
                                                (Path("C:/Users/Ștefan/postura"), True)])
def test_a_root_with_diacritics_is_reported_on_windows(root, problem, monkeypatch):
    monkeypatch.setattr(M, "is_windows", lambda: True)
    message = M.root_path_problem(root)
    assert bool(message) is problem
    if problem:
        assert "C:\\postura" in message


def test_elsewhere_a_root_with_diacritics_is_fine(monkeypatch):
    monkeypatch.setattr(M, "is_windows", lambda: False)
    assert M.root_path_problem(Path("/srv/ana/Călăreț")) is None


def test_the_page_stops_with_a_clear_message_in_a_folder_with_diacritics(monkeypatch):
    monkeypatch.setattr(ST, "ROOT", Path("C:/Users/Călăreț/postura"))
    monkeypatch.setattr(M, "is_windows", lambda: True)
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.error and "C:\\postura" in at.error[0].value and not at.file_uploader
