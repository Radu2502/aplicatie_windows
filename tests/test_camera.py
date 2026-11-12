"""Rotirea camerei, estimată din fundal: pe imagini sintetice cu rotire și deplasare cunoscute. Zona calului
se exclude, iar un clip fără fundal utilizabil spune asta, nu ghicește."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from ecvestru import camera as K
from ecvestru import masurare as M
from ecvestru import raport as R
from ecvestru import stocare as ST
from ecvestru import suprapunere as S

SIZE = (360, 640)


def scene(seed=3):
    rng = np.random.default_rng(seed)
    big = cv2.resize(rng.integers(0, 255, (45, 80), dtype=np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(big, (0, 0), 1.2)


def view(image, angle=0.0, dx=0.0, dy=0.0):
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    matrix[:, 2] += (dx, dy)
    warped = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(warped[90:450, 160:800], cv2.COLOR_GRAY2BGR)


def write(path: Path, frames) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (SIZE[1], SIZE[0]))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def test_rotation_between_two_frames_is_recovered():
    base = scene()
    step = K.frame_motion(cv2.cvtColor(view(base), cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(view(base, 1.0), cv2.COLOR_BGR2GRAY), [])
    assert step.angle == pytest.approx(1.0, abs=0.1)


def test_translation_between_two_frames_is_recovered():
    base = scene()
    step = K.frame_motion(cv2.cvtColor(view(base), cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(view(base, 0.0, 6.0, -3.0), cv2.COLOR_BGR2GRAY), [])
    assert step.dx == pytest.approx(6.0, abs=0.3) and step.dy == pytest.approx(-3.0, abs=0.3)


def test_blank_frames_give_no_estimate():
    blank = np.full(SIZE, 90, dtype=np.uint8)
    assert K.frame_motion(blank, blank, []) is None


def test_moving_object_inside_the_excluded_box_does_not_disturb_the_camera():
    base = cv2.cvtColor(view(scene()), cv2.COLOR_BGR2GRAY)
    moved = base.copy()
    box = (200.0, 120.0, 240.0, 150.0)
    moved[120:270, 200:440] = np.roll(base[120:270, 200:440], 25, axis=1)
    assert K.frame_motion(base, moved, [box]).dx == pytest.approx(0.0, abs=0.2)


def test_excluded_zone_also_covers_the_rider_above_the_horse():
    assert K.excluded_box((100.0, 100.0, 50.0, 40.0)) == (100.0, 100.0 - K.RIDER_ABOVE * 40.0, 50.0,
                                                          40.0 + K.RIDER_ABOVE * 40.0)


def steady(n, **fields):
    return {f: K.Motion(**fields) for f in range(n)}


def test_amplitude_is_the_range_of_the_accumulated_rotation():
    motion = {f: K.Motion(angle=0.5 if f < 6 else -0.2, dx=0.0, dy=0.0, points=50) for f in range(1, 11)}
    estimate = K.rotation_amplitude(motion, (0, 10))
    assert estimate.amplitude_deg == pytest.approx(2.5) and estimate.amplitude_rounded == 2.5


def test_amplitude_uses_only_the_interval():
    motion = {f: K.Motion(angle=2.0 if f > 5 else 0.0, dx=0.0, dy=0.0, points=50) for f in range(1, 11)}
    assert K.rotation_amplitude(motion, (0, 5)).amplitude_deg == pytest.approx(0.0)


def test_too_many_frames_without_estimate_means_no_number():
    motion = {f: K.Motion(angle=0.1, dx=0.0, dy=0.0, points=50) for f in range(1, 4)}
    estimate = K.rotation_amplitude(motion, (0, 10))
    assert estimate.amplitude_deg is None and estimate.frames_estimated == 3 and estimate.frames_total == 10


def test_clip_rotation_end_to_end(tmp_path):
    base = scene()
    path = write(tmp_path / "c.avi", [view(base, 0.3 * i) for i in range(11)])
    motion = K.clip_motion(path, {})
    assert K.rotation_amplitude(motion, (0, 10)).amplitude_deg == pytest.approx(3.0, abs=0.3)


def test_steady_clip_has_almost_no_rotation(tmp_path):
    path = write(tmp_path / "c.avi", [view(scene())] * 8)
    assert K.rotation_amplitude(K.clip_motion(path, {}), (0, 7)).amplitude_deg == pytest.approx(0.0, abs=0.05)


@pytest.fixture
def analysed(tmp_path, monkeypatch):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    monkeypatch.setattr(ST, "run_ratios", lambda path: {})
    monkeypatch.setattr(ST, "detect_horses", lambda path: {})
    return tmp_path


def predictor(path):
    return {f: {"points": {"shoulder": (10.0, 10.0), "hip": (10.0, 60.0), "heel": (10.0, 120.0), "facing": 1.0},
                "motiv": None} for f in range(11)}


def test_report_states_the_rotation_and_passes_the_number_check(analysed):
    base = scene()
    path = write(analysed / "c.avi", [view(base, 0.3 * i) for i in range(11)])
    result = M.analyze(path, "pas", (0, 10), predictor=predictor, extended=True)
    text = R.report_text(result)
    assert f"Camera s-a rotit cu până la {result.camera.amplitude_rounded}°" in text
    assert "unghiurile pot fi deplasate cu atât" in text and R.report_passes_checks(text, result)


def test_report_says_when_the_rotation_cannot_be_estimated(analysed):
    path = write(analysed / "c.avi", [np.full((*SIZE, 3), 90, dtype=np.uint8)] * 11)
    result = M.analyze(path, "pas", (0, 10), predictor=predictor, extended=True)
    assert result.camera.amplitude_deg is None
    assert "Rotirea camerei nu a putut fi estimată" in R.report_text(result)


def test_plain_analysis_has_no_camera_field_and_no_cache(analysed):
    path = write(analysed / "c.avi", [view(scene())] * 11)
    result = M.analyze(path, "pas", (0, 10), predictor=predictor)
    assert result.camera is None and '"camera"' not in result.to_json()
    assert "Camera" not in R.report_text(result)
    assert not (analysed / "rezultate" / "analiza" / "camera").exists()


def test_film_and_film_module_do_not_depend_on_the_camera(analysed):
    path = write(analysed / "c.avi", [view(scene())] * 11)
    result = M.analyze(path, "pas", (0, 10), predictor=predictor, extended=True)
    assert S.clip_film(path, result)
