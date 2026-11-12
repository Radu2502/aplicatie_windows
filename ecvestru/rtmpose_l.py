"""RTMPose-l Halpe26 384×288: modelul opțional al stratului „călcâi stabil” (`masurare.analyze(..., heel_stable=True,
heel_model="l")`, oprit implicit). Aceiași pași ca ecvestru/rtmpose.py (caseta din punctele MediaPipe, padding 1,25,
RGB normalizat, SimCC cu raportul 2), cu intrarea 288 × 384. Modelul se descarcă la prima folosire în models/rtmpose_l/,
cu suma sha256 fixată. Licența e a lui RTMPose-m (Apache-2.0, antrenat pe Body7). Fără Streamlit."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecvestru import onnx_comun as OC
from ecvestru import rtmpose as R
from ecvestru import stocare as ST
from ecvestru.rezultat import HEEL_MODEL_L, HEEL_MODEL_M

VERSION = 2   # 2: scor ≤ 0 → NaN (R.decode_simcc), scorurile în cache
MODEL_NAME = "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605"
MODEL_URL = f"https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/{MODEL_NAME}.zip"
MODEL_ZIP_SHA256 = "03a66ae0aba2130723f96ab30542226e2a48782bd121ff1d8948bd50252e14c2"
MODEL_DIR = ST.ROOT / "models" / "rtmpose_l"
INPUT_W, INPUT_H = 288, 384


def ensure_model(folder: Path | None = None) -> Path:
    """Modelul din `folder`, dacă e dat; altfel căutat cu ecvestru/modele.py (ECVESTRU_MODELE, ~/modele-ecvestru/,
    models/) și, dacă lipsește peste tot, descărcat în models/; altfel ModelNotFound, cu locurile căutate."""
    return OC.ensure_model("RTMPose-l", MODEL_DIR,
                           lambda target: R.download_model(MODEL_URL, MODEL_ZIP_SHA256, target), folder)


def to_input(points: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return OC.to_input(points, center, scale, INPUT_W, INPUT_H)


def to_image(points: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return OC.to_image(points, center, scale, INPUT_W, INPUT_H)


def prepare(image: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return OC.prepare(image, bbox, INPUT_W, INPUT_H)


class RtmPoseL:
    """Un călăreț pe cadru → 26 de puncte Halpe în pixeli și scorurile; ține timpul total și numărul de cadre."""

    def __init__(self, model: Path | None = None, threads: int = R.THREADS):
        self._run = OC.session_runner(model or ensure_model(), threads)
        self.seconds, self.frames = 0.0, 0

    @classmethod
    def from_run(cls, run: Callable) -> RtmPoseL:
        estimator = cls.__new__(cls)
        estimator._run, estimator.seconds, estimator.frames = run, 0.0, 0
        return estimator

    def __call__(self, image: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        start = time.perf_counter()
        tensor, center, scale = prepare(image, bbox)
        simcc_x, simcc_y = self._run(tensor)
        locs, scores = R.decode_simcc(simcc_x, simcc_y)
        self.seconds += time.perf_counter() - start
        self.frames += 1
        return to_image(locs, center, scale), scores


@dataclass(frozen=True)
class Points:
    points: np.ndarray       # (N, 26, 2)
    seconds: float           # timpul modelului, pe cadrele rulate
    frames: int


def points_cache_path(name: str, video_hash: str) -> Path:
    return ST.results_dir() / "rtmpose_l" / f"{ST.clip_key(name, video_hash)}__l{VERSION}_{MODEL_ZIP_SHA256[:8]}.npz"


def load_points(video_path, name: str, video_hash: str, skeleton, estimator: Callable | None = None,
                path: Path | None = None) -> Points:
    """Punctele pe cadrele cu călăreț, din cache sau calculate cu R.run_clip (aceeași casetă ca la m), cu timpul
    modelului. `path` = alt fișier de cache."""
    path = path or points_cache_path(name, video_hash)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return Points(data["points"], float(data["seconds"]), int(data["frames"]))
    estimator = estimator or RtmPoseL()
    seconds, frames = getattr(estimator, "seconds", 0.0), getattr(estimator, "frames", 0)
    points, scores = R.run_clip_scored(Path(video_path), skeleton, estimator)
    result = Points(points, getattr(estimator, "seconds", 0.0) - seconds, getattr(estimator, "frames", 0) - frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=result.points, seconds=result.seconds, frames=result.frames, scores=scores)
    return result


def points(video_path, name: str, video_hash: str, skeleton, estimator: Callable | None = None,
           heel_model: str = HEEL_MODEL_M) -> np.ndarray:
    """Punctele Halpe26 ale căii RTMPose: m (ecvestru/rtmpose.py) sau l (acest modul, doar cu stratul „călcâi
    stabil”). `estimator` e estimatorul modelului ales."""
    if heel_model == HEEL_MODEL_L:
        return load_points(video_path, name, video_hash, skeleton, estimator).points
    return R.load_points(video_path, name, video_hash, skeleton, estimator)
