"""Filmul cu lanțul călărețului ales: textul benzii, desenul, opțiunea de fundal estompat și codarea
H.264 doar în memorie. Clipurile sunt sintetice, fără persoane, iar rezultatele stau în foldere temporare."""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from ecvestru import masurare as M
from ecvestru import schelet as Z
from ecvestru import stocare as ST
from ecvestru import suprapunere as S
from ecvestru.rezultat import FrameMeasure

POINTS = {"shoulder": (100.0, 80.0), "hip": (100.0, 180.0), "heel": (80.0, 280.0), "facing": 1.0}


def measure(frame=5, trunk=None, leg=None, reason=None):
    return FrameMeasure(frame=frame, trunk_deg=trunk, leg_deg=leg, reason=reason)


def blank(h=360, w=640):
    return np.full((h, w, 3), 90, dtype=np.uint8)


def has_colour(img: np.ndarray, colour: tuple[int, int, int]) -> bool:
    return bool((np.abs(img.astype(int) - colour).sum(axis=2) < 30).any())


def synthetic_clip(path: Path, frames: int) -> Path:
    rng = np.random.default_rng(1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


def probe(data: bytes) -> str:
    done = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                           "stream=codec_name,pix_fmt,nb_read_frames", "-of", "csv=p=0", "-i", "pipe:0"],
                          input=data, capture_output=True, check=True)
    return done.stdout.decode().strip()


def top_level_atoms(data: bytes) -> list[str]:
    atoms, at = [], 0
    while at + 8 <= len(data):
        size = int.from_bytes(data[at:at + 4], "big")
        atoms.append(data[at + 4:at + 8].decode("latin-1"))
        if size < 8:
            break
        at += size
    return atoms


def test_band_for_a_measured_frame_uses_the_report_rounding():
    assert S.band_text(measure(152, -8.2, -12.6)) == "#152 · trunchi -10° · picior -15°"
    assert S.band_text(measure(3, 1.0, 3.0)) == "#3 · trunchi 0° · picior +5°"


def test_band_for_a_missing_heel_keeps_the_trunk_and_names_the_reason():
    text = S.band_text(measure(7, 6.0, None, M.REASON_HEEL))
    assert text == f"#7 · trunchi +5° · Nemăsurat: {M.REASON_TEXTS[M.REASON_HEEL]}"


def test_band_for_an_unmeasured_frame_names_only_the_reason():
    assert S.band_text(measure(9, reason="nu_e_pe_cal")) == "#9 · Nemăsurat: persoana găsită nu e pe cal"


def test_band_text_with_diacritics_fits_a_narrow_clip():
    font = S.band_font(S.band_text(measure(9, reason="trunchi_peste_45")), 352)
    assert font.getlength(S.band_text(measure(9, reason="trunchi_peste_45"))) <= 352


def test_measured_frame_shows_trunk_leg_vertical_and_band():
    img = S.overlay_frame(blank(), POINTS, measure(5, 0.0, -11.3))
    assert has_colour(img, S.TRUNK_BGR) and has_colour(img, S.LEG_BGR)
    assert has_colour(img, S.VERTICAL_BGR)
    assert not np.array_equal(img[:S.band_height(img)], blank()[:S.band_height(img)])


def test_missing_heel_draws_only_the_trunk():
    img = S.overlay_frame(blank(), {**POINTS, "heel": None}, measure(5, 0.0, None, M.REASON_HEEL))
    assert has_colour(img, S.TRUNK_BGR) and not has_colour(img, S.LEG_BGR)


def test_unmeasured_frame_has_no_chain_even_if_points_exist():
    img = S.overlay_frame(blank(), POINTS, measure(5, reason="nu_e_pe_cal"))
    assert not has_colour(img, S.TRUNK_BGR) and not has_colour(img, S.LEG_BGR)
    assert not np.array_equal(img[:S.band_height(img)], blank()[:S.band_height(img)])


def test_blurred_background_keeps_the_chain_sharp():
    noise = np.random.default_rng(1).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    plain = S.overlay_frame(noise, POINTS, measure(5, 0.0, -11.3))
    soft = S.overlay_frame(noise, POINTS, measure(5, 0.0, -11.3), blur=True)
    assert soft[300:].std() < plain[300:].std() / 2
    assert has_colour(soft, S.TRUNK_BGR) and has_colour(soft, S.LEG_BGR)


def test_blur_grows_with_the_rider_and_never_falls_under_the_frame_minimum():
    small = S.blur_sigma(360, 640, {**POINTS, "hip": (100.0, 90.0)})
    large = S.blur_sigma(360, 640, {**POINTS, "hip": (100.0, 300.0)})
    assert large > small >= S.blur_sigma(360, 640, None) > 0


def test_film_is_h264_with_one_frame_per_interval_frame_and_moov_first(tmp_path):
    clip = synthetic_clip(tmp_path / "c.mp4", 12)
    frames = [measure(i, 0.0, 0.0) for i in range(2, 10)]
    data = S.film_bytes(clip, frames, {i: {"points": POINTS, "motiv": None} for i in range(12)}, 30.0)
    assert probe(data) == "h264,yuv420p,8"
    atoms = top_level_atoms(data)
    assert atoms.index("moov") < atoms.index("mdat")


def test_film_marks_frames_without_points_instead_of_failing(tmp_path):
    clip = synthetic_clip(tmp_path / "c.mp4", 4)
    frames = [measure(i, reason="calaret_negasit") for i in range(4)]
    assert probe(S.film_bytes(clip, frames, {}, 30.0)) == "h264,yuv420p,4"


def test_film_writes_no_file(tmp_path, monkeypatch):
    clip = synthetic_clip(tmp_path / "c.mp4", 4)
    work = tmp_path / "lucru"
    work.mkdir()
    monkeypatch.chdir(work)
    S.film_bytes(clip, [measure(i, 0.0, 0.0) for i in range(4)], {}, 30.0)
    assert not list(work.iterdir()) and [p.name for p in tmp_path.iterdir() if p.is_file()] == ["c.mp4"]


def test_failed_encoder_raises_a_clear_error(tmp_path, monkeypatch):
    clip = synthetic_clip(tmp_path / "c.mp4", 4)
    monkeypatch.setattr(S, "FFMPEG", "false")
    with pytest.raises(RuntimeError, match="film"):
        S.film_bytes(clip, [measure(i, 0.0, 0.0) for i in range(4)], {}, 30.0)


def test_upload_film_uses_the_saved_points_and_leaves_only_the_two_json_files(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    data = synthetic_clip(tmp_path / "c.mp4", 6).read_bytes()
    predictor = lambda path: {i: {"points": POINTS, "motiv": None} for i in range(6)}  # noqa: E731
    result = M.analyze_upload(data, "incarcat.mp4", "pas", predictor=predictor)
    saved = sorted(p.name for p in (tmp_path / "rezultate").rglob("*") if p.is_file())
    film = S.upload_film(data, "incarcat.mp4", result)
    assert probe(film) == "h264,yuv420p,6"
    assert sorted(p.name for p in (tmp_path / "rezultate").rglob("*") if p.is_file()) == saved
    assert len(saved) == 2 and all(json.loads(p.read_text()) for p in (tmp_path / "rezultate").rglob("*.json"))


def _bones():
    """33 de puncte în pixeli, vizibile, pe o linie orizontală departe de lanț."""
    xy = np.stack([np.linspace(20, 300, 33), np.full(33, 230.0)], axis=1)
    return xy, np.full(33, 0.9)


def test_the_full_skeleton_is_drawn_on_a_measured_frame():
    plain = S.overlay_frame(blank(), POINTS, measure(5, 0.0, -11.3))
    drawn = S.overlay_frame(blank(), POINTS, measure(5, 0.0, -11.3), bones=_bones())
    assert (drawn != plain).any()


def _one_point(visibility):
    """Un singur punct, în (320, 230); restul lipsesc (NaN), ca pe un cadru cu date parțiale."""
    xy, seen = np.full((33, 2), np.nan), np.full(33, np.nan)
    xy[0], seen[0] = (320.0, 230.0), visibility
    return xy, seen


def test_a_rider_without_a_measure_gets_the_skeleton_the_reason_and_no_chain():
    xy, seen = np.full((33, 2), np.nan), np.full(33, np.nan)
    xy[11], xy[12], seen[11], seen[12] = (200.0, 230.0), (500.0, 230.0), 0.9, 0.9     # o linie lungă, între două puncte
    img = S.overlay_frame(blank(), POINTS, measure(5, reason="trunchi_peste_45"), bones=(xy, seen))
    assert has_colour(img, S.SKELETON_BGR)
    assert not has_colour(img, S.TRUNK_BGR) and not has_colour(img, S.LEG_BGR) and not has_colour(img, S.VERTICAL_BGR)
    plain = S.overlay_frame(blank(), POINTS, measure(5, reason="trunchi_peste_45"))
    band = S.band_height(img)
    assert (img[:band] == plain[:band]).all()                                # banda cu motivul rămâne


def test_a_frame_without_a_rider_has_only_the_band():
    empty = np.full((33, 2), np.nan), np.full(33, np.nan)
    plain = S.overlay_frame(blank(), None, measure(5, reason="calaret_negasit"))
    assert (S.overlay_frame(blank(), None, measure(5, reason="calaret_negasit"), bones=empty) == plain).all()


def test_the_measured_chain_stays_on_top_of_the_skeleton():
    xy = np.stack([np.linspace(100, 100, 33), np.linspace(80, 180, 33)], axis=1)   # puncte peste trunchi
    img = S.overlay_frame(blank(), POINTS, measure(5, 0.0, -11.3), bones=(xy, np.full(33, 0.9)))
    assert np.abs(img[130, 100].astype(int) - S.TRUNK_BGR).sum() < 30
    assert has_colour(img, S.LEG_BGR) and has_colour(img, S.VERTICAL_BGR)


@pytest.mark.parametrize("visibility,colour", [(0.9, "HIGH_BGR"), (0.8, "MID_BGR"), (0.5, "MID_BGR"), (0.3, "LOW_BGR")])
def test_each_point_has_the_colour_of_its_tier(visibility, colour):
    img = S.overlay_frame(blank(), None, measure(5, reason="nu_e_pe_cal"), bones=_one_point(visibility))
    assert has_colour(img, getattr(S, colour))


def test_a_low_point_is_an_empty_ring_so_the_tiers_do_not_depend_only_on_colour():
    low = S.overlay_frame(blank(), None, measure(5, reason="nu_e_pe_cal"), bones=_one_point(0.3))
    high = S.overlay_frame(blank(), None, measure(5, reason="nu_e_pe_cal"), bones=_one_point(0.9))
    assert (low[230, 320] == blank()[0, 0]).all() and not (high[230, 320] == blank()[0, 0]).all()


def test_a_point_without_data_is_not_drawn():
    plain = S.overlay_frame(blank(), None, measure(5, reason="nu_e_pe_cal"))
    assert (S.overlay_frame(blank(), None, measure(5, reason="nu_e_pe_cal"), bones=_one_point(np.nan)) == plain).all()


def test_lines_join_only_visible_points():
    xy, _ = _bones()
    img = S.overlay_frame(blank(), None, measure(5, reason="nu_e_pe_cal"), bones=(xy, np.full(33, 0.1)))
    assert not has_colour(img, S.SKELETON_BGR) and has_colour(img, S.LOW_BGR)


def test_the_legend_names_both_thresholds_and_each_mark():
    assert "0,5" in S.VISIBILITY_LEGEND and "0,8" in S.VISIBILITY_LEGEND
    assert all(word in S.VISIBILITY_LEGEND for word in ("verde", "mov", "roșu"))


def test_a_line_with_one_missing_end_is_skipped():
    xy, seen = np.full((33, 2), np.nan), np.full(33, np.nan)
    xy[11], seen[11] = (100.0, 100.0), 0.9
    S.overlay_frame(blank(), None, measure(5, reason="nu_e_pe_cal"), bones=(xy, seen))


def _person():
    xy, seen = np.full((33, 2), np.nan), np.full(33, np.nan)
    xy[11], xy[12], seen[11], seen[12] = (200.0, 230.0), (500.0, 230.0), 0.9, 0.9
    return xy, seen, "nu_e_pe_cal"


def test_a_rejected_person_is_drawn_dashed_in_one_neutral_colour_without_a_chain():
    img = S.overlay_frame(blank(), POINTS, measure(5, reason="nu_e_pe_cal"), rejected=[_person()])
    row = img[230, 200:500]
    drawn = (np.abs(row.astype(int) - S.REJECTED_BGR).sum(axis=1) < 30).mean()
    assert 0.2 < drawn < 0.8                                                 # linie punctată, nu continuă
    assert not has_colour(img, S.SKELETON_BGR) and not has_colour(img, S.TRUNK_BGR)
    assert not has_colour(img, S.HIGH_BGR)                                   # fără treptele călărețului ales


def test_a_rejected_person_gets_a_label_with_the_reason():
    assert S.rejected_label("nu_e_pe_cal") == f"persoană respinsă: {M.REASON_TEXTS['nu_e_pe_cal']}"
    assert "respinsă" in S.rejected_label(Z.REASON_NOT_CHOSEN)
    plain = S.overlay_frame(blank(), None, measure(5, reason="nu_e_pe_cal"), rejected=[_person()])
    assert has_colour(plain, S.LABEL_BACK_BGR)


def test_rejected_people_are_not_drawn_over_a_measured_frame():
    plain = S.overlay_frame(blank(), POINTS, measure(5, 0.0, -11.3))
    assert (S.overlay_frame(blank(), POINTS, measure(5, 0.0, -11.3), rejected=[_person()]) == plain).all()


def test_a_rejected_person_with_hidden_points_has_no_lines():
    xy, _, reason = _person()
    img = S.overlay_frame(blank(), None, measure(5, reason=reason), rejected=[(xy, np.full(33, 0.1), reason)])
    assert not has_colour(img, S.REJECTED_BGR) and not has_colour(img, S.LABEL_BACK_BGR)


def test_the_legend_describes_the_style_of_rejected_people():
    assert "punctate" in S.REJECTED_LEGEND and "instructorul" in S.REJECTED_LEGEND


# ---------- lizibilitate ----------

def _pair(a, b, visibility=0.9):
    xy, seen = np.full((33, 2), np.nan), np.full(33, np.nan)
    xy[a], xy[b], seen[a], seen[b] = (150.0, 200.0), (500.0, 200.0), visibility, visibility
    return xy, seen


def _no_rider():
    return measure(5, reason="nu_e_pe_cal")


def test_points_outside_the_list_are_not_drawn():
    xy, seen = np.full((33, 2), np.nan), np.full(33, np.nan)
    xy[3], seen[3] = (320.0, 230.0), 0.9                                     # un ochi
    plain = S.overlay_frame(blank(), None, _no_rider())
    assert (S.overlay_frame(blank(), None, _no_rider(), bones=(xy, seen)) == plain).all()
    xy[17], xy[31] = (320.0, 230.0), (320.0, 230.0)                          # degetul mic, vârful piciorului
    seen[17], seen[31] = 0.9, 0.9
    assert (S.overlay_frame(blank(), None, _no_rider(), bones=(xy, seen)) == plain).all()


def test_the_lines_have_a_dark_outline_on_a_light_background():
    light = np.full((360, 640, 3), 235, np.uint8)
    img = S.overlay_frame(light, None, _no_rider(), bones=_pair(11, 12))
    column = img[190:211, 320].astype(int)
    assert np.abs(column - S.SKELETON_BGR).sum(axis=1).min() < 30             # linia deschisă
    assert (np.abs(column - S.OUTLINE_BGR).sum(axis=1) < 60).any()            # conturul închis


def test_the_outline_stays_under_the_chain():
    xy, seen = np.full((33, 2), np.nan), np.full(33, np.nan)
    xy[11], xy[23], seen[11], seen[23] = POINTS["shoulder"], POINTS["hip"], 0.9, 0.9
    img = S.overlay_frame(blank(), POINTS, measure(5, 0.0, -11.3), bones=(xy, seen))
    assert np.abs(img[130, 100].astype(int) - S.TRUNK_BGR).sum() < 30


def test_the_side_of_the_candidate_is_stronger_than_the_opposite_side():
    near = S.overlay_frame(blank(), None, _no_rider(), bones=(*_pair(11, 13), "left"))
    far = S.overlay_frame(blank(), None, _no_rider(), bones=(*_pair(11, 13), "right"))
    assert has_colour(near[190:211, 200:450], S.SKELETON_BGR)
    assert not has_colour(far[190:211, 200:450], S.SKELETON_BGR)
    width = lambda img: int((np.abs(img[150:250, 320].astype(int) - 90).sum(axis=1) > 0).sum())  # noqa: E731
    assert 0 < width(far) < width(near)                                      # subțire și estompată, dar vizibilă


def test_without_a_side_nothing_is_faded():
    both = S.overlay_frame(blank(), None, _no_rider(), bones=_pair(11, 13))
    assert has_colour(both[190:211, 200:450], S.SKELETON_BGR)


def test_the_far_points_keep_their_tier_but_are_faded():
    xy, seen = np.full((33, 2), np.nan), np.full(33, np.nan)
    xy[12], seen[12] = (320.0, 230.0), 0.9
    near = S.overlay_frame(blank(), None, _no_rider(), bones=(xy, seen, "right"))
    far = S.overlay_frame(blank(), None, _no_rider(), bones=(xy, seen, "left"))
    assert has_colour(near, S.HIGH_BGR) and not has_colour(far, S.HIGH_BGR) and (far != blank()).any()


def test_a_film_passes_the_side_of_each_frame(tmp_path, monkeypatch):
    clip = synthetic_clip(tmp_path / "c.mp4", 3)
    seen = []
    real = S.overlay_frame
    monkeypatch.setattr(S, "overlay_frame", lambda *a: seen.append(a[4][2]) or real(*a))

    class Bones:
        xy_px = np.zeros((3, 33, 2))
        visibility = np.zeros((3, 33))
        side = np.array(["left", "right", ""])

        def shown_rejected_at(self, index):
            return []

    S.film_bytes(clip, [measure(i) for i in range(3)], {}, 30.0, skeleton=Bones())
    assert seen == ["left", "right", ""]


def test_the_legend_explains_the_two_sides():
    assert "plin" in S.SIDE_LEGEND and "estompat" in S.SIDE_LEGEND


def test_the_band_shows_the_warnings_of_the_frame():
    frame = FrameMeasure(frame=7, trunk_deg=6.0, leg_deg=-11.0, warnings=["salt", "proportii"])
    assert S.band_text(frame) == "#7 · trunchi +5° · picior -10° · Atenție: salt, proporții"


def test_a_frame_without_warnings_keeps_its_band():
    assert "Atenție" not in S.band_text(measure(7, 6.0, -11.0))


def test_the_gated_leg_has_its_reason_in_the_band():
    assert "călcâi pe piciorul calului" in S.band_text(measure(7, 6.0, None, M.REASON_HEEL_ON_HORSE))


def test_the_candidate_film_uses_the_gate(tmp_path, monkeypatch):
    clip = synthetic_clip(tmp_path / "c.mp4", 4)
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    monkeypatch.setattr(ST, "run_candidate", lambda path: {})
    seen = []
    monkeypatch.setattr(M, "apply_quality", lambda *a: seen.append(a) or (a[-1], False))
    S.candidate_film(clip)
    assert seen
