"""Pașii comuni ai modelelor ONNX din familia RTMPose (SimCC), folosiți de RTMPose-m (ecvestru/rtmpose.py), RTMPose-l
(ecvestru/rtmpose_l.py) și AP-10K, punctele calului (ecvestru/cal_puncte.py): caseta cu padding și raportul de aspect
al intrării, decupajul RGB normalizat, trecerea punctelor între imagine și intrarea modelului, decodarea SimCC,
sesiunea onnxruntime, găsirea și descărcarea modelului (cu suma sha256 fixată). Modelele diferă doar prin mărimea
intrării. Fără Streamlit; onnxruntime se importă abia când se creează o sesiune."""
from __future__ import annotations

import hashlib
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ecvestru import modele as MO

MODEL_FILE = "end2end.onnx"          # numele modelului în pachetele onnx_sdk ale MMPose
PADDING = 1.25                       # caseta se mărește cu atât (pipeline.json din pachet)
MEAN = (123.675, 116.28, 103.53)     # normalizarea RGB a intrării
STD = (58.395, 57.12, 57.375)
SPLIT_RATIO = 2.0                    # SimCC: atâtea poziții pe pixel de intrare
DOWNLOAD_CHUNK = 1 << 20


# ---------- caseta și coordonatele (intrarea are `width` × `height` pixeli) ----------

def center_scale(bbox: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Caseta x0, y0, x1, y1 → centru și mărime, cu padding și raportul de aspect al intrării."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    box_w, box_h = (x1 - x0) * PADDING, (y1 - y0) * PADDING
    aspect = width / height
    if box_w > box_h * aspect:
        box_h = box_w / aspect
    else:
        box_w = box_h * aspect
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2]), np.array([box_w, box_h])


def factor(scale: np.ndarray, width: int) -> float:
    """Pixeli de intrare pe pixel de imagine."""
    return width / float(scale[0])


def to_input(points: np.ndarray, center: np.ndarray, scale: np.ndarray, width: int, height: int) -> np.ndarray:
    return (points - center) * factor(scale, width) + (width / 2, height / 2)


def to_image(points: np.ndarray, center: np.ndarray, scale: np.ndarray, width: int, height: int) -> np.ndarray:
    return (points - (width / 2, height / 2)) / factor(scale, width) + center


def prepare(image: np.ndarray, bbox: np.ndarray, width: int, height: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Imaginea BGR → tensorul (1, 3, height, width) RGB normalizat, plus centrul și mărimea casetei."""
    center, scale = center_scale(bbox, width, height)
    k = factor(scale, width)
    matrix = np.array([[k, 0, width / 2 - k * center[0]], [0, k, height / 2 - k * center[1]]])
    crop = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)[..., ::-1]
    tensor = (crop - np.array(MEAN)) / np.array(STD)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32), center, scale


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximul pe fiecare axă → punctele pe intrarea modelului (K, 2) și scorul; scor ≤ 0 → punct lipsă (NaN), care
    rămâne NaN și după transformarea în pixeli, deci fuziunea îl tratează ca pe un punct lipsă."""
    simcc_x, simcc_y = simcc_x[0], simcc_y[0]
    locs = np.stack([simcc_x.argmax(axis=1), simcc_y.argmax(axis=1)], axis=1).astype(np.float64) / SPLIT_RATIO
    scores = 0.5 * (simcc_x.max(axis=1) + simcc_y.max(axis=1))
    locs[scores <= 0] = np.nan
    return locs, scores


# ---------- sesiunea și modelul ----------

def session_runner(model: Path, threads: int) -> Callable[[np.ndarray], list]:
    """Sesiunea onnxruntime pe CPU, cu `threads` fire într-o operație și unul între operații → funcția care rulează
    modelul pe un tensor."""
    import onnxruntime as ort  # abia aici: importul aplicației nu depinde de onnxruntime
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return lambda tensor: session.run(None, {name: tensor})


def find_model(folder: Path, model_file: str = MODEL_FILE) -> Path | None:
    """Primul model găsit oriunde sub `folder` (ordinea căilor), sau None."""
    found = sorted(folder.glob(f"**/{model_file}"))
    return found[0] if found else None


def download_model(url: str, sha256: str, folder: Path, model_file: str = MODEL_FILE) -> Path:
    """Descarcă pachetul, îi verifică suma sha256 și abia apoi îl despachetează în `folder`."""
    with tempfile.TemporaryDirectory(prefix="ecvestru_rtmpose_") as tmp:
        archive, digest = Path(tmp) / "model.zip", hashlib.sha256()
        with urllib.request.urlopen(url) as response, archive.open("wb") as out:
            while chunk := response.read(DOWNLOAD_CHUNK):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f"Suma sha256 a modelului nu se potrivește (primit {digest.hexdigest()}).")
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                target = (folder / member).resolve()
                if folder.resolve() not in target.parents and target != folder.resolve():
                    raise ValueError(f"Arhiva are o cale nepermisă: {member}")
            z.extractall(folder)
    found = find_model(folder, model_file)
    if found is None:
        raise ValueError(f"Arhiva nu conține {model_file}.")
    return found


def ensure_model(label: str, model_dir: Path, download: Callable[[Path], Path], folder: Path | None = None) -> Path:
    """Modelul din `folder`, dacă e dat (descărcat acolo dacă lipsește); altfel căutat cu ecvestru/modele.py
    (ECVESTRU_MODELE, ~/modele-ecvestru/, models/) și, dacă lipsește peste tot, descărcat în `model_dir`; altfel
    ModelNotFound, cu locurile căutate. `download(folder)` descarcă pachetul modelului în `folder`."""
    if folder is not None:
        return find_model(folder) or download(folder)
    return MO.locate(label, model_dir.name, MODEL_FILE, model_dir, lambda: download(model_dir))
