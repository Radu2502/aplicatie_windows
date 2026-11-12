"""Clipurile noi: inventarul (fără imagini), cititorul exact de cadre și umplerea cache-ului (fără măsurători).
Clipurile din teste sunt sintetice; fiecare cadru are culoarea după indicele lui."""
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from ecvestru import clipuri_noi as CN
from ecvestru import stocare as ST

SIZE = (64, 48)
FPS = 25


def write_clip(path: Path, frames: int, size=SIZE, fps=FPS) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), (i * 5) % 250, np.uint8))
    writer.release()
    return path


def write_inventory(path: Path, rows: list[dict]) -> Path:
    fields = ["nr", "clip", "durata_s", "calaret_1", "calaret_2", "cal_1", "cal_2", "observatii"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" for k in fields} | row)
    return path


@pytest.fixture
def folder(tmp_path):
    write_clip(tmp_path / "b.mp4", 30)
    write_clip(tmp_path / "a (1).mp4", 12, size=(48, 64))
    (tmp_path / "imagine.jpg").write_bytes(b"nu e clip")
    write_inventory(tmp_path / CN.INVENTORY_FILE, [
        {"nr": "0", "clip": "a (1).mp4", "calaret_1": "C1"},
        {"nr": "1", "clip": "b.mp4", "calaret_1": "C2", "calaret_2": "C1"}])
    return tmp_path


def test_only_the_videos_are_listed_in_name_order(folder):
    assert CN.list_clips(folder) == ["a (1).mp4", "b.mp4"]


def test_the_folder_can_be_moved_with_an_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_CLIPURI_NOI", str(tmp_path))
    assert CN.clips_dir() == tmp_path


def test_probe_counts_frames_by_reading_and_gives_the_decoded_size(folder):
    info = CN.probe(folder / "a (1).mp4")
    assert (info.frames, info.width, info.height) == (12, 48, 64)
    assert info.fps == pytest.approx(FPS)
    assert info.seconds == pytest.approx(12 / FPS)


def test_riders_come_from_the_inventory_csv_and_empty_cells_are_skipped(folder):
    riders = CN.read_riders(folder / CN.INVENTORY_FILE)
    assert riders == {"a (1).mp4": ("C1",), "b.mp4": ("C2", "C1")}


def test_the_inventory_has_one_row_per_clip_with_the_riders(folder):
    rows = CN.inventory(folder)
    assert [r.clip for r in rows] == ["a (1).mp4", "b.mp4"]
    assert rows[1].nr == 1 and rows[1].frames == 30 and rows[1].riders == ("C2", "C1")


def test_a_clip_missing_from_the_csv_has_no_rider_and_no_number(folder):
    write_clip(folder / "c.mp4", 5)
    row = CN.inventory(folder)[-1]
    assert row.clip == "c.mp4" and row.riders == () and row.nr is None


def test_the_markdown_says_when_the_rider_is_not_filled_in(folder):
    write_clip(folder / "c.mp4", 5)
    text = CN.inventory_markdown(CN.inventory(folder))
    assert "| c.mp4 |" in text and CN.NOT_FILLED in text
    assert "C2, C1" in text
    assert "47 cadre" in text  # totalul: 12 + 30 + 5


def test_the_frame_reader_gives_exactly_the_frame_with_that_index_forward_and_backward(folder):
    reader = CN.FrameReader(folder / "b.mp4")
    for index in (3, 4, 20, 2, 29, 0):
        assert int(reader.read(index)[0, 0, 0]) == pytest.approx((index * 5) % 250, abs=3)
    reader.close()


def test_the_frame_reader_refuses_a_frame_past_the_end(folder):
    reader = CN.FrameReader(folder / "b.mp4")
    with pytest.raises(ValueError):
        reader.read(30)
    with pytest.raises(ValueError):
        reader.read(-1)


# ---------- cache-ul: doar timpul și erorile, nicio măsurătoare ----------

def test_filling_the_cache_runs_mediapipe_then_rtmpose_and_reports_only_time_and_errors(folder, monkeypatch):
    calls = []
    monkeypatch.setattr(ST, "load_raw", lambda path, name, h, c: calls.append(("raw", name)) or {})
    monkeypatch.setattr(ST, "load_rider_skeleton", lambda path, name, h, c: calls.append(("schelet", name)) or "S")
    monkeypatch.setattr(CN, "prepare_rtmpose_points", lambda path, name, h, s: calls.append(("rtmpose", s)))
    run = CN.fill_cache(folder / "b.mp4")
    assert calls == [("raw", "b.mp4"), ("schelet", "b.mp4"), ("rtmpose", "S")]
    assert run.clip == "b.mp4" and run.error is None and run.seconds >= 0 and run.rtmpose
    assert set(CN.CacheRun.model_fields) == {"clip", "seconds", "rtmpose", "error"}


def test_without_a_skeleton_rtmpose_is_skipped_and_said(folder, monkeypatch):
    monkeypatch.setattr(ST, "load_raw", lambda *a: {})
    monkeypatch.setattr(ST, "load_rider_skeleton", lambda *a: None)
    monkeypatch.setattr(CN, "prepare_rtmpose_points", lambda *a: pytest.fail("fără schelet nu rulează RTMPose"))
    run = CN.fill_cache(folder / "b.mp4")
    assert not run.rtmpose and run.error == CN.NO_SKELETON


def test_an_error_on_one_clip_is_caught_and_reported(folder, monkeypatch):
    def broken(*a):
        raise RuntimeError("decodare eșuată")
    monkeypatch.setattr(ST, "load_raw", broken)
    run = CN.fill_cache(folder / "b.mp4")
    assert run.error == "RuntimeError: decodare eșuată" and not run.rtmpose


def test_the_cache_report_has_only_times_and_errors(folder):
    runs = [CN.CacheRun(clip="a.mp4", seconds=61.0, rtmpose=True, error=None),
            CN.CacheRun(clip="b.mp4", seconds=2.5, rtmpose=False, error="RuntimeError: x")]
    text = CN.cache_markdown(runs, workers=2)
    assert "| a.mp4 | 61 | da | – |" in text and "| b.mp4 | 2 | nu | RuntimeError: x |" in text
    assert "2 procese" in text


def test_the_inventory_document_has_a_title_the_table_and_no_measurement(folder):
    text = CN.inventory_doc(CN.inventory(folder))
    assert text.startswith("# Inventarul clipurilor noi")
    assert "| nr | clip |" in text and "unghi" not in text.lower()


def test_the_cache_document_has_a_title_the_commit_and_only_times_and_errors():
    runs = [CN.CacheRun(clip="a.mp4", seconds=61.0, rtmpose=True, error=None)]
    text = CN.cache_doc(runs, workers=6, commit="abc1234")
    assert text.startswith("# Cache-ul scheletelor pe clipurile noi") and "abc1234" in text
    assert "| a.mp4 | 61 | da | – |" in text and "Total: 61 s" in text


# ---------- cache-ul calului: casetele, apoi punctele AP-10K; doar cadrele, timpul și erorile ----------

def _fake_horse(calls, inferred=3):
    from ecvestru import cal_puncte as CP

    def points(path, name, h, boxes, hips):
        calls.append(("puncte", name, boxes, hips))
        n = 30
        used = np.full((n, 4), np.nan)
        used[:inferred] = 1.0
        return CP.ClipRun(np.zeros((n, 17, 2)), np.zeros((n, 17)), used, 0.5)
    return points


def test_filling_the_horse_cache_runs_the_boxes_then_the_points_under_the_rider_hips(folder, monkeypatch):
    calls = []
    skeleton = {"left_hip": np.zeros((30, 2)), "right_hip": np.ones((30, 2))}
    monkeypatch.setattr(ST, "load_boxes", lambda path, name, h: calls.append(("casete", name)) or {0: [[1, 2, 3, 4]]})
    monkeypatch.setattr(ST, "load_rider_skeleton", lambda path, name, h, c: skeleton)
    monkeypatch.setattr(CN, "rider_hips", lambda s: "H")
    monkeypatch.setattr(CN, "prepare_horse_points", _fake_horse(calls))
    run = CN.fill_horse_cache(folder / "b.mp4")
    assert calls == [("casete", "b.mp4"), ("puncte", "b.mp4", {0: [[1, 2, 3, 4]]}, "H")]
    assert run.clip == "b.mp4" and run.frames == 30 and run.with_box == 3 and run.error is None
    assert set(CN.HorseCacheRun.model_fields) == {"clip", "frames", "with_box", "seconds", "error"}


def test_without_a_rider_skeleton_the_largest_box_is_used(folder, monkeypatch):
    calls = []
    monkeypatch.setattr(ST, "load_boxes", lambda *a: {})
    monkeypatch.setattr(ST, "load_rider_skeleton", lambda *a: None)
    monkeypatch.setattr(CN, "prepare_horse_points", _fake_horse(calls, inferred=0))
    run = CN.fill_horse_cache(folder / "b.mp4")
    assert calls[0][3] is None and run.with_box == 0 and run.error is None


def test_an_error_in_the_horse_cache_is_caught_and_reported(folder, monkeypatch):
    def broken(*a):
        raise RuntimeError("detector lipsă")
    monkeypatch.setattr(ST, "load_boxes", broken)
    run = CN.fill_horse_cache(folder / "b.mp4")
    assert run.error == "RuntimeError: detector lipsă" and run.frames == 0 and run.with_box == 0


def test_the_horse_cache_report_has_only_frames_times_and_errors():
    runs = [CN.HorseCacheRun(clip="a.mp4", frames=300, with_box=280, seconds=61.0, error=None),
            CN.HorseCacheRun(clip="b.mp4", frames=0, with_box=0, seconds=0.4, error="RuntimeError: x")]
    text = CN.horse_cache_doc(runs, workers=6, commit="abc1234")
    assert text.startswith("# Cache-ul calului pe clipurile noi") and "abc1234" in text and "6 procese" in text
    assert "| a.mp4 | 300 | 280 | 61 | – |" in text and "| b.mp4 | 0 | 0 | 0 | RuntimeError: x |" in text
    assert "erori: 1" in text


# ---------- călăreții, salvați din pagina „Vedere rapidă” ----------

def test_saving_the_riders_changes_only_their_cells_and_keeps_the_rest(folder):
    path = folder / CN.INVENTORY_FILE
    before = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    CN.save_riders("a (1).mp4", " C2 ", "", path)
    after = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert [r["clip"] for r in after] == [r["clip"] for r in before]
    assert after[0] == before[0] | {"calaret_1": "C2", "calaret_2": ""} and after[1] == before[1]
    assert path.read_bytes().splitlines()[0] == b"nr,clip,durata_s,calaret_1,calaret_2,cal_1,cal_2,observatii"
    assert b"\r\n" in path.read_bytes()
    assert CN.read_riders(path)["a (1).mp4"] == ("C2",)


@pytest.mark.parametrize("first, second", [("", "C1"), ("C1;C2", ""), ("C,1", ""), ("C1", "C1"),
                                           ("X" * (CN.RIDER_CODE_MAX + 1), "")])
def test_bad_rider_codes_are_refused_and_nothing_is_written(folder, first, second):
    path = folder / CN.INVENTORY_FILE
    before = path.read_bytes()
    with pytest.raises(ValueError):
        CN.save_riders("a (1).mp4", first, second, path)
    assert path.read_bytes() == before


def test_a_clip_missing_from_the_inventory_is_refused(folder):
    with pytest.raises(ValueError):
        CN.save_riders("nu-exista.mp4", "C1", "", folder / CN.INVENTORY_FILE)


def test_an_empty_first_rider_clears_both_only_when_the_second_is_empty_too(folder):
    path = folder / CN.INVENTORY_FILE
    CN.save_riders("b.mp4", "", "", path)
    assert CN.read_riders(path)["b.mp4"] == ()
