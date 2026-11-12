"""Punctele calului (AP-10K, RTMPose-m): preprocesarea 256×256, decodarea SimCC, alegerea casetei calului, rularea pe
clip, cache-ul versionat și acoperirea. Fără model: sesiunea ONNX e înlocuită cu o funcție care știe unde sunt
punctele."""
import cv2
import numpy as np
import pytest

from ecvestru import cal_puncte as CP
from ecvestru import masurare as M


def test_the_seventeen_ap10k_points_are_in_the_order_of_the_dataset():
    assert len(CP.AP10K) == 17
    assert [CP.AP10K.index(n) for n in ("nose", "neck", "root_of_tail")] == [2, 3, 4]
    assert [CP.AP10K.index(n) for n in CP.PAWS] == [7, 10, 13, 16]
    assert CP.REPORTED == ("nose", "neck", "root_of_tail", *CP.PAWS)


def test_the_model_is_pinned_by_url_and_sha256():
    assert CP.MODEL_URL.startswith("https://download.openmmlab.com/") and CP.MODEL_NAME in CP.MODEL_URL
    assert len(CP.MODEL_ZIP_SHA256) == 64 and int(CP.MODEL_ZIP_SHA256, 16) >= 0
    assert (CP.INPUT_W, CP.INPUT_H) == (256, 256)


def test_the_box_becomes_a_padded_square_around_its_centre():
    center, scale = CP.center_scale(np.array([100.0, 200.0, 300.0, 300.0]))
    assert center == pytest.approx([200.0, 250.0])
    assert scale == pytest.approx([200 * CP.PADDING, 200 * CP.PADDING])   # pătrat: latura cea mai mare


def test_input_and_image_coordinates_are_inverse():
    center, scale = CP.center_scale(np.array([10.0, 20.0, 110.0, 70.0]))
    points = np.array([[10.0, 20.0], [60.0, 45.0], [110.0, 70.0]])
    assert CP.to_image(CP.to_input(points, center, scale), center, scale) == pytest.approx(points)
    assert CP.to_input(np.array([[60.0, 45.0]]), center, scale) == pytest.approx(np.array([[128.0, 128.0]]))


def test_prepare_gives_the_normalized_rgb_tensor_of_the_crop():
    image = np.zeros((100, 200, 3), np.uint8)
    image[..., 2] = 255                                      # roșu în BGR
    tensor, _, _ = CP.prepare(image, np.array([50.0, 25.0, 150.0, 75.0]))
    assert tensor.shape == (1, 3, 256, 256) and tensor.dtype == np.float32
    assert tensor[0, 0, 128, 128] == pytest.approx((255 - CP.MEAN[0]) / CP.STD[0])   # canalul R primul
    assert tensor[0, 2, 128, 128] == pytest.approx((0 - CP.MEAN[2]) / CP.STD[2])


def simcc(points_input: np.ndarray, peak: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Ieșirea SimCC cu maximul exact la punctele date (pe intrarea modelului)."""
    k = len(points_input)
    x = np.zeros((1, k, CP.INPUT_W * CP.SPLIT_RATIO), np.float32)
    y = np.zeros((1, k, CP.INPUT_H * CP.SPLIT_RATIO), np.float32)
    for i, (px, py) in enumerate(points_input):
        x[0, i, round(px * CP.SPLIT_RATIO)] = peak
        y[0, i, round(py * CP.SPLIT_RATIO)] = peak
    return x, y


def test_simcc_is_decoded_to_the_peak_divided_by_the_split_ratio_with_its_score():
    x, y = simcc(np.array([[10.0, 20.5], [255.5, 0.0]]))
    x[0, 1] = 0
    y[0, 1] = 0                                              # punct fără răspuns
    locs, scores = CP.decode_simcc(x, y)
    assert locs[0] == pytest.approx([10.0, 20.5]) and scores[0] == pytest.approx(0.9)
    assert np.isnan(locs[1]).all() and scores[1] == 0


def test_the_estimator_puts_the_points_back_in_image_pixels():
    box = np.array([100.0, 50.0, 300.0, 250.0])
    center, scale = CP.center_scale(box)
    wanted = np.array([[150.0, 100.0]] * 17)
    x, y = simcc(CP.to_input(wanted, center, scale))
    estimator = CP.HorsePose.from_run(lambda tensor: (x, y))
    points, scores = estimator(np.zeros((300, 400, 3), np.uint8), box)
    assert points.shape == (17, 2) and points == pytest.approx(wanted, abs=CP.pixel_step(scale))
    assert scores == pytest.approx(np.full(17, 0.9))


def test_the_pixel_step_is_half_an_input_pixel_in_the_image():
    assert CP.pixel_step(np.array([512.0, 512.0])) == pytest.approx(1.0)


# ---------- caseta calului ----------

def test_without_a_rider_the_largest_horse_box_is_taken_as_corners():
    boxes = [[0.0, 0.0, 10.0, 10.0], [50.0, 60.0, 100.0, 40.0]]
    assert CP.select_box(boxes, None) == pytest.approx([50.0, 60.0, 150.0, 100.0])
    assert CP.select_box([], None) is None


def test_with_a_rider_the_box_under_the_hip_is_taken():
    boxes = [[0.0, 100.0, 50.0, 50.0], [200.0, 100.0, 300.0, 200.0]]   # a doua e mai mare
    assert CP.select_box(boxes, np.array([25.0, 110.0])) == pytest.approx([0.0, 100.0, 50.0, 150.0])


def test_with_a_rider_on_no_box_there_is_no_horse():
    assert CP.select_box([[0.0, 100.0, 50.0, 50.0]], np.array([500.0, 500.0])) is None


def test_a_rider_hip_with_nan_counts_as_no_rider():
    boxes = [[0.0, 0.0, 10.0, 10.0], [50.0, 60.0, 100.0, 40.0]]
    assert CP.select_box(boxes, np.array([np.nan, np.nan])) == pytest.approx([50.0, 60.0, 150.0, 100.0])


# ---------- clipul și cache-ul ----------

def write_clip(path, frames=6, size=(160, 120)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 10, np.uint8))
    writer.release()
    return path


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, image, box):
        self.calls.append((int(image[0, 0, 0]), tuple(box)))
        return np.full((17, 2), box[0]), np.full(17, 0.5)


def test_run_clip_runs_only_on_frames_with_a_horse_box(tmp_path):
    video = write_clip(tmp_path / "c.mp4")
    boxes = {0: [[10.0, 20.0, 30.0, 40.0]], 2: [], 3: [[5.0, 5.0, 10.0, 10.0]]}
    estimator = Recorder()
    run = CP.run_clip(video, boxes, None, estimator)
    assert [c[1] for c in estimator.calls] == [(10.0, 20.0, 40.0, 60.0), (5.0, 5.0, 15.0, 15.0)]
    assert run.points.shape == (6, 17, 2) and run.scores.shape == (6, 17) and run.boxes.shape == (6, 4)
    assert run.points[0, 0, 0] == 10.0 and np.isnan(run.points[1]).all() and np.isnan(run.scores[2]).all()
    assert run.boxes[3] == pytest.approx([5.0, 5.0, 15.0, 15.0]) and np.isnan(run.boxes[4]).all()
    assert run.seconds >= 0


def test_run_clip_reads_the_frame_with_the_same_index(tmp_path):
    video = write_clip(tmp_path / "c.mp4")
    estimator = Recorder()
    CP.run_clip(video, {4: [[1.0, 1.0, 5.0, 5.0]]}, None, estimator)
    assert estimator.calls[0][0] == pytest.approx(40, abs=3)


def test_the_cache_is_versioned_and_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path / "rezultate"))
    video = write_clip(tmp_path / "c.mp4")
    estimator = Recorder()
    first = CP.load_points(video, "c.mp4", "abc", {0: [[1.0, 1.0, 5.0, 5.0]]}, None, estimator)
    again = CP.load_points(video, "c.mp4", "abc", {0: [[1.0, 1.0, 5.0, 5.0]]}, None, estimator)
    assert len(estimator.calls) == 1
    assert np.array_equal(first.points, again.points, equal_nan=True) and again.seconds == first.seconds
    path = CP.points_cache_path("c.mp4", "abc")
    assert path.exists() and path.parent == M.results_dir() / "cal"
    assert f"a{CP.CAL_VERSION}_{CP.MODEL_ZIP_SHA256[:8]}" in path.name and f"c{M.BOXES_VERSION}" in path.name


def test_coverage_is_the_share_of_frames_with_a_box_where_each_point_passes_the_threshold():
    scores = np.full((4, 17), np.nan)
    scores[0] = 0.9
    scores[1] = 0.1
    scores[2] = 0.9
    scores[2, CP.AP10K.index("nose")] = 0.2
    cover = CP.coverage(scores)
    assert cover.frames == 4 and cover.with_box == 3
    assert cover.share["nose"] == pytest.approx(1 / 3) and cover.share["neck"] == pytest.approx(2 / 3)
    assert set(cover.share) == set(CP.REPORTED)


def test_coverage_without_any_box_is_zero():
    cover = CP.coverage(np.full((3, 17), np.nan))
    assert cover.with_box == 0 and all(v == 0 for v in cover.share.values())


# ---------- modelul: descărcat la prima folosire ----------

def test_the_model_is_found_in_its_own_folder(tmp_path):
    assert CP.model_path(tmp_path) is None
    (tmp_path / "x" / "y").mkdir(parents=True)
    (tmp_path / "x" / "y" / CP.MODEL_FILE).write_bytes(b"")
    assert CP.model_path(tmp_path) == tmp_path / "x" / "y" / CP.MODEL_FILE


def test_the_model_is_downloaded_at_first_use_with_the_pinned_sha(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, sha256, folder):
        calls.append((url, sha256, folder))
        (folder / "m").mkdir(parents=True)
        (folder / "m" / CP.MODEL_FILE).write_bytes(b"")
        return folder / "m" / CP.MODEL_FILE

    monkeypatch.setattr(CP.R, "download_model", fake_download)
    assert CP.ensure_model(tmp_path) == tmp_path / "m" / CP.MODEL_FILE
    assert CP.ensure_model(tmp_path) == tmp_path / "m" / CP.MODEL_FILE
    assert calls == [(CP.MODEL_URL, CP.MODEL_ZIP_SHA256, tmp_path)]


def test_a_point_with_score_zero_stays_missing_in_image_pixels_not_a_plausible_pixel():
    x, y = simcc(np.array([[10.0, 20.5], [255.5, 0.0]]))
    x[0, 1] = 0
    y[0, 1] = 0
    estimator = CP.HorsePose.from_run(lambda tensor: (x, y))
    points, scores = estimator(np.zeros((480, 640, 3), dtype=np.uint8), np.array([100.0, 100.0, 300.0, 250.0]))
    assert np.isnan(points[1]).all() and scores[1] == 0
    assert np.isfinite(points[0]).all()


def test_the_cache_version_is_2_after_the_missing_point_fix():
    assert CP.CAL_VERSION == 2
