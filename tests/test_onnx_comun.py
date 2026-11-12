"""Modulul comun al modelelor ONNX (ecvestru/onnx_comun.py): decupajul, coordonatele și decodarea SimCC pentru
RTMPose-m, RTMPose-l și AP-10K dau EXACT tablourile de referință (sume sha256 din
tests/date/onnx_comun_referinta.json); găsirea și descărcarea modelului, o singură dată."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ecvestru import cal_puncte as CP
from ecvestru import onnx_comun as OC
from ecvestru import rtmpose as R
from ecvestru import rtmpose_l as L

REFERENCE = Path(__file__).parent / "date" / "onnx_comun_referinta.json"


def digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(f"{array.dtype}|{array.shape}|".encode() + array.tobytes()).hexdigest()


def outputs() -> dict[str, np.ndarray]:
    """Aceleași intrări, în aceeași ordine a extragerilor, ca la generarea referinței."""
    rng = np.random.default_rng(20261107)
    image = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    boxes = np.array([[25.0, 12.5, 75.0, 100.0], [2.5, 5.25, 157.5, 117.5], [50.0, 50.0, 65.0, 55.0]])
    points = rng.uniform(0, 160, (26, 2))
    out = {}
    for tag, module in (("m", R), ("l", L), ("cal", CP)):
        for i, bbox in enumerate(boxes):
            tensor, center, scale = module.prepare(image, bbox)
            out[f"{tag}_prepare_{i}"], out[f"{tag}_center_{i}"], out[f"{tag}_scale_{i}"] = tensor, center, scale
            out[f"{tag}_to_input_{i}"] = module.to_input(points, center, scale)
            out[f"{tag}_to_image_{i}"] = module.to_image(points, center, scale)
    for tag, module, k, w, h in (("m", R, 26, R.INPUT_W, R.INPUT_H), ("cal", CP, 17, CP.INPUT_W, CP.INPUT_H)):
        simcc_x = rng.normal(size=(1, k, int(w * 2)))
        simcc_y = rng.normal(size=(1, k, int(h * 2)))
        simcc_x[0, 3] = -1.0
        simcc_y[0, 3] = -1.0
        locs, scores = module.decode_simcc(simcc_x, simcc_y)
        out[f"{tag}_simcc_x"], out[f"{tag}_simcc_y"] = simcc_x, simcc_y
        out[f"{tag}_decode_locs"], out[f"{tag}_decode_scores"] = locs, scores
    return out


def test_the_three_models_give_exactly_the_arrays_of_before():
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    got = {k: digest(v) for k, v in outputs().items()}
    assert got.keys() == expected.keys()
    assert [k for k in expected if got[k] != expected[k]] == []


def test_the_models_use_the_common_steps():
    assert R.decode_simcc.__module__ == CP.decode_simcc.__module__ == OC.__name__
    for module in (R, L, CP):
        assert module.prepare.__module__ == module.__name__          # învelișul păstrează numele de până acum
    assert (R.MEAN, R.STD, R.PADDING, R.MODEL_FILE) == (OC.MEAN, OC.STD, OC.PADDING, OC.MODEL_FILE)
    assert CP.MEAN == OC.MEAN and CP.STD == OC.STD


def test_a_missing_point_has_no_position():
    simcc = np.full((1, 2, 4), -1.0)
    simcc[0, 0, 2] = 3.0
    locs, scores = OC.decode_simcc(simcc, simcc)
    assert locs[0].tolist() == [1.0, 1.0] and np.isnan(locs[1]).all() and scores.tolist() == [3.0, -1.0]


def test_the_model_is_found_in_a_folder_or_downloaded_there(tmp_path, monkeypatch):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / OC.MODEL_FILE).write_bytes(b"x")
    assert OC.find_model(tmp_path) == nested / OC.MODEL_FILE
    assert OC.find_model(tmp_path / "gol") is None
    calls = []
    download = lambda folder: calls.append(folder) or folder / OC.MODEL_FILE  # noqa: E731
    empty = tmp_path / "gol"
    assert OC.ensure_model("Test", tmp_path / "models", download, empty) == empty / OC.MODEL_FILE
    assert calls == [empty]                                                  # lipsă: se descarcă acolo
    assert OC.ensure_model("Test", tmp_path / "models", download, tmp_path) == nested / OC.MODEL_FILE
    assert calls == [empty]                                                  # găsit: nu se descarcă


def test_a_bad_archive_is_refused(tmp_path, monkeypatch):
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("../afara.onnx", b"x")
    data = buffer.getvalue()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(OC.urllib.request, "urlopen", lambda url: Response(data))
    with pytest.raises(ValueError, match="sha256"):
        OC.download_model("u", "0" * 64, tmp_path)
    with pytest.raises(ValueError, match="cale nepermisă"):
        OC.download_model("u", hashlib.sha256(data).hexdigest(), tmp_path)
