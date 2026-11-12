"""
Punctele calului: 17 puncte AP-10K, cu RTMPose-m AP-10K (ONNX), pe calul de sub călăreț. Implementare proprie a
preprocesării (decupaj pătrat din caseta calului, padding 1,25, 256×256, normalizare RGB) și a decodării SimCC, ca în
`ecvestru/rtmpose.py`. Caseta vine din detectorul de cai existent (cache-ul cai/ al aplicației); dintre casete se ia
cea de sub șoldul călărețului ales (regula calitate.rider_box) sau, fără călăreț, cea mai mare. Nu schimbă nicio
valoare a analizei implicite și nu decide nimic despre vedere. Fără Streamlit.

Licență: cod și greutăți Apache-2.0; AP-10K CC-BY-4.0 (Yu et al., NeurIPS 2021); preantrenare pe AIC (Challenger) și
MS COCO. Uz științific, necomercial.

Rulare:
  uv run python -m ecvestru.cal_puncte descarca     # modelul, cu suma sha256 fixată
  uv run python -m ecvestru.cal_puncte ruleaza      # clipurile dev: timp pe cadru și acoperire
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ecvestru import calitate as C
from ecvestru import modele as MO
from ecvestru import onnx_comun as OC
from ecvestru import rtmpose as R
from ecvestru import stocare as ST
from ecvestru import verificare as V

CAL_VERSION = 2              # se mărește când se schimbă pașii de calcul (invalidează cache-ul)
MODEL_NAME = "rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206"
MODEL_URL = f"https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/{MODEL_NAME}.zip"
MODEL_ZIP_SHA256 = "2d75445331cf2f21d6e164430f96ffa765cd874872965ae1736932dda03987f0"
MODEL_FILE = R.MODEL_FILE
MODEL_DIR = ST.ROOT / "models" / "ap10k"          # models/ e ignorat de git; separat de modelul călărețului
THREADS = R.THREADS

# Ordinea din MMPose, configs/_base_/datasets/ap10k.py
AP10K = ("left_eye", "right_eye", "nose", "neck", "root_of_tail", "left_shoulder", "left_elbow", "left_front_paw",
         "right_shoulder", "right_elbow", "right_front_paw", "left_hip", "left_knee", "left_back_paw", "right_hip",
         "right_knee", "right_back_paw")
PAWS = ("left_front_paw", "right_front_paw", "left_back_paw", "right_back_paw")
REPORTED = ("nose", "neck", "root_of_tail", *PAWS)
SCORE_MIN = 0.3              # pragul de afișare al punctelor din MMPose și rtmlib (kpt_thr)

# Intrarea modelului: config rtmpose-m_8xb64-210e_ap10k-256x256 și forma intrării din end2end.onnx (1, 3, 256, 256)
INPUT_W, INPUT_H = 256, 256
PADDING = OC.PADDING                     # pașii comuni ai modelelor ONNX: ecvestru/onnx_comun.py
MEAN = OC.MEAN
STD = OC.STD
SPLIT_RATIO = int(OC.SPLIT_RATIO)        # întreg: mărimea ieșirilor SimCC (INPUT_W * SPLIT_RATIO)


# ---------- modelul: unde e, descărcat la prima folosire ----------

def model_path(folder: Path | None = None) -> Path | None:
    """Modelul din `folder`, dacă e dat; altfel primul găsit de ecvestru/modele.py (ECVESTRU_MODELE,
    ~/modele-ecvestru/, models/). None dacă lipsește."""
    if folder is None:
        return MO.find_model(MODEL_DIR.name, MODEL_FILE, MODEL_DIR)
    return OC.find_model(folder, MODEL_FILE)


def ensure_model(folder: Path | None = None) -> Path:
    """Modelul din `folder` (sau căutat ca în model_path) ori, la prima folosire, descărcat în models/ și verificat
    cu suma sha256 fixată; dacă nici descărcarea nu merge, ModelNotFound cu locurile căutate."""
    return OC.ensure_model("AP-10K (punctele calului)", MODEL_DIR,
                           lambda target: R.download_model(MODEL_URL, MODEL_ZIP_SHA256, target), folder)


# ---------- pașii modelului ----------

def center_scale(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Caseta x0, y0, x1, y1 → centru și mărime, cu padding și raportul de aspect al intrării (pătrat)."""
    return OC.center_scale(bbox, INPUT_W, INPUT_H)


def pixel_step(scale: np.ndarray) -> float:
    """Pasul decodării (1 / SPLIT_RATIO pixeli de intrare), în pixeli ai imaginii."""
    return 1 / SPLIT_RATIO / OC.factor(scale, INPUT_W)


def to_input(points: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return OC.to_input(points, center, scale, INPUT_W, INPUT_H)


def to_image(points: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return OC.to_image(points, center, scale, INPUT_W, INPUT_H)


def prepare(image: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Imaginea BGR → tensorul (1, 3, 256, 256) RGB normalizat, plus centrul și mărimea casetei."""
    return OC.prepare(image, bbox, INPUT_W, INPUT_H)


decode_simcc = OC.decode_simcc


class HorsePose:
    """Un cal pe cadru: imaginea BGR și caseta lui (x0, y0, x1, y1) → 17 puncte în pixeli și scorurile lor."""

    def __init__(self, model: Path | None = None, threads: int = THREADS):
        self._run = OC.session_runner(model or ensure_model(), threads)

    @classmethod
    def from_run(cls, run: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]) -> HorsePose:
        """Cu o funcție tensor → (simcc_x, simcc_y) în locul sesiunii ONNX (pentru teste)."""
        estimator = cls.__new__(cls)
        estimator._run = run
        return estimator

    def __call__(self, image: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, center, scale = prepare(image, np.asarray(bbox, dtype=float))
        simcc_x, simcc_y = self._run(tensor)
        locs, scores = decode_simcc(simcc_x, simcc_y)
        return to_image(locs, center, scale), scores


# ---------- caseta calului ----------

def select_box(boxes: list, hip: np.ndarray | None) -> np.ndarray | None:
    """Caseta calului (x, y, w, h din detector) → colțurile ei. Cu șoldul călărețului: caseta de sub el (regula
    candidatului, calitate.rider_box), iar dacă nu e niciuna, niciun cal; fără călăreț: cea mai mare."""
    if hip is not None and not np.isnan(hip).any():
        box = C.rider_box(hip, boxes)
    else:
        box = max(boxes, key=lambda b: b[2] * b[3]) if boxes else None
    return None if box is None else np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=float)


# ---------- clipul și cache-ul ----------

@dataclass(frozen=True)
class ClipRun:
    points: np.ndarray       # (N, 17, 2) pixeli; NaN pe cadrele fără casetă de cal
    scores: np.ndarray       # (N, 17); NaN pe cadrele fără casetă
    boxes: np.ndarray        # (N, 4) caseta folosită, x0, y0, x1, y1; NaN fără casetă
    seconds: float           # timpul modelului (preprocesare + rulare + decodare), pe tot clipul

    @property
    def inferred(self) -> int:
        return int((~np.isnan(self.boxes[:, 0])).sum())


def run_clip(video_path: Path, boxes: dict, hips: np.ndarray | None, estimator: Callable) -> ClipRun:
    """Punctele calului pe fiecare cadru cu casetă de cal; cadrele se citesc secvențial (indicele OpenCV)."""
    n_frames = ST.video_info(Path(video_path))[0]
    points = np.full((n_frames, len(AP10K), 2), np.nan)
    scores = np.full((n_frames, len(AP10K)), np.nan)
    used = np.full((n_frames, 4), np.nan)
    seconds = 0.0
    capture = cv2.VideoCapture(str(video_path))
    try:
        for index in range(n_frames):
            box = select_box(boxes.get(index, []), None if hips is None else hips[index])
            if box is None:
                if not capture.grab():
                    break
                continue
            ok, image = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            points[index], scores[index] = estimator(image, box)
            seconds += time.perf_counter() - start
            used[index] = box
    finally:
        capture.release()
    return ClipRun(points, scores, used, seconds)


def points_cache_path(name: str, video_hash: str) -> Path:
    candidate = ST.candidate_label().replace("@", "_")
    return (ST.results_dir() / "cal" /
            f"{ST.clip_key(name, video_hash)}__{candidate}__a{CAL_VERSION}_{MODEL_ZIP_SHA256[:8]}"
            f"__c{ST.BOXES_VERSION}.npz")


def load_points(video_path: Path, name: str, video_hash: str, boxes: dict, hips: np.ndarray | None,
                estimator: Callable | None = None) -> ClipRun:
    """Punctele calului din cache sau calculate (și salvate). Cheia: clipul, candidatul (șoldurile), versiunea
    codului, modelul și versiunea casetelor."""
    path = points_cache_path(name, video_hash)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return ClipRun(data["points"], data["scores"], data["boxes"], float(data["seconds"]))
    run = run_clip(video_path, boxes, hips, estimator or HorsePose())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=run.points, scores=run.scores, boxes=run.boxes, seconds=run.seconds)
    return run


# ---------- acoperirea ----------

@dataclass(frozen=True)
class Coverage:
    frames: int
    with_box: int
    share: dict[str, float]  # din cadrele cu casetă, proporția cu scorul punctului peste SCORE_MIN


def coverage(scores: np.ndarray, threshold: float = SCORE_MIN) -> Coverage:
    has_box = ~np.isnan(scores).all(axis=1)
    n = int(has_box.sum())
    share = {name: float((scores[has_box, AP10K.index(name)] > threshold).sum() / n) if n else 0.0
             for name in REPORTED}
    return Coverage(frames=len(scores), with_box=n, share=share)


# ---------- rularea pe clipurile dev ----------

def run_dev_clip(video: str) -> ClipRun:
    path = V.VIDEOS / video
    video_hash, candidate = ST.file_hash(path), ST.candidate_label()
    boxes = ST.load_boxes(path, video, video_hash)
    skeleton = ST.load_rider_skeleton(path, video, video_hash, candidate)
    hips = None if skeleton is None else C.midpoint(skeleton, "hip")
    return load_points(path, video, video_hash, boxes, hips)


def report_markdown(runs: dict[str, ClipRun]) -> str:
    names = " | ".join(REPORTED)
    lines = [f"| clip | cadre | cu casetă | ms/cadru | {names} |", "|---" * (4 + len(REPORTED)) + "|"]
    for video, run in runs.items():
        cover = coverage(run.scores)
        ms = 1000 * run.seconds / run.inferred if run.inferred else float("nan")
        cells = " | ".join(f"{100 * cover.share[n]:.0f}%" for n in REPORTED)
        lines.append(f"| {video} | {cover.frames} | {cover.with_box} | {ms:.1f} | {cells} |")
    scores = np.concatenate([r.scores for r in runs.values()])
    total = coverage(scores)
    inferred = sum(r.inferred for r in runs.values())
    ms = 1000 * sum(r.seconds for r in runs.values()) / inferred if inferred else float("nan")
    cells = " | ".join(f"**{100 * total.share[n]:.0f}%**" for n in REPORTED)
    lines.append(f"| **toate** | {total.frames} | {total.with_box} | {ms:.1f} | {cells} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ecvestru.cal_puncte", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("descarca", help="descarcă și verifică modelul AP-10K")
    commands.add_parser("ruleaza", help="punctele calului pe cele 15 clipuri dev (cache), timp și acoperire")
    args = parser.parse_args(argv)
    if args.command == "descarca":
        print(f"Model: {ensure_model()}")
        return 0
    runs = {}
    for video in ST.dev_videos():
        runs[video] = run_dev_clip(video)
        print(f"{video}: {runs[video].inferred} cadre cu cal, {runs[video].seconds:.0f} s", flush=True)
    print(report_markdown(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
