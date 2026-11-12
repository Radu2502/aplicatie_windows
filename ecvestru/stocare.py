"""
Cache-ul etapelor analizei, în data/results/analiza/ (sau ECVESTRU_RESULTS/analiza/): căile și cheile (numele
clipului, hash-ul conținutului, eticheta candidatului, versiunea fiecărei etape) și încărcarea fiecărei etape din
cache sau calculată o dată și salvată: brutul candidatului, scheletul, raportul umeri/trunchi, casetele cailor,
mișcarea camerei. Modelele se apelează doar prin cele patru funcții `run_*` / `extract_*` / `detect_*` de mai jos.
Fără Streamlit; nu importă straturile de deasupra (calitate, RTMPose, analiza).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ecvestru import camera as K
from ecvestru import schelet
from ecvestru import tinta as T
from ecvestru import vedere as W
from ecvestru import verificare as V
from ecvestru.rezumat import Raw

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "research" / "candidate.py"
SPLIT_CSV = ROOT / "data" / "split.csv"

HASH_CHARS = 12
BOXES_VERSION = 1      # casetele de cal: se mărește când se schimbă detectorul sau pragul lui
UPLOAD_LABEL = "incarcat"  # numele unui clip încărcat în rezultat; pe disc, UPLOAD_KEY: fără numele original
UPLOAD_KEY = "upload"
DEFAULT_FPS = 30.0      # când clipul nu are fps în metadate


# ---------- candidatul ----------

@lru_cache(maxsize=1)
def candidate_module():
    spec = importlib.util.spec_from_file_location("candidate_aplicatie", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_label() -> str:
    return "candidate@" + hashlib.sha1(CANDIDATE.read_bytes()).hexdigest()[:8]


def run_candidate(video_path: Path) -> Raw:
    """O singură trecere: aceleași puncte ca predict_points_with_reasons, plus scheletul complet."""
    return schelet.extract_pass(video_path, candidate_label())


def extract_skeleton(video_path: Path) -> schelet.RiderSkeleton | None:
    return schelet.extract_pass(video_path, candidate_label()).skeleton


def run_ratios(video_path: Path) -> dict:
    return W.shoulder_ratios(video_path, candidate_module()._landmarks)


def detect_horses(video_path: Path) -> dict:
    return T.detect_horses(video_path, candidate_module().detector_path())


# ---------- fișiere, cache ----------

def results_dir() -> Path:
    return Path(os.environ.get("ECVESTRU_RESULTS", ROOT / "data" / "results")) / "analiza"


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:HASH_CHARS]


def bytes_hash(data: bytes) -> str:
    """Același hash ca `file_hash`, pentru un clip încărcat, aflat încă în memorie."""
    return hashlib.sha1(data).hexdigest()[:HASH_CHARS]


def upload_name(video_hash: str) -> str:
    """Numele unui clip încărcat în rezultat: identifică fișierul, nu persoana."""
    return f"{UPLOAD_LABEL}-{video_hash}"


def clip_key(name: str, video_hash: str) -> str:
    return f"{UPLOAD_KEY}-{video_hash}" if name == upload_name(video_hash) else f"{Path(name).stem}-{video_hash}"


def skeleton_cache_path(name: str, video_hash: str, candidate: str) -> Path:
    return (results_dir() / "schelet" /
            f"{clip_key(name, video_hash)}__{candidate.replace('@', '_')}__s{schelet.SCHEMA_VERSION}.npz")


def raw_cache_path(name: str, video_hash: str, candidate: str) -> Path:
    return results_dir() / "brut" / f"{clip_key(name, video_hash)}__{candidate.replace('@', '_')}.json"


def ratios_cache_path(name: str, video_hash: str, candidate: str) -> Path:
    return results_dir() / "vedere" / f"{clip_key(name, video_hash)}__{candidate.replace('@', '_')}.json"


def boxes_cache_path(name: str, video_hash: str) -> Path:
    return results_dir() / "cai" / f"{clip_key(name, video_hash)}__c{BOXES_VERSION}.json"


def motion_cache_path(name: str, video_hash: str) -> Path:
    return results_dir() / "camera" / f"{clip_key(name, video_hash)}__v{K.MOTION_VERSION}.json"


def video_info(path: Path) -> tuple[int, float]:
    """(număr de cadre după indicele OpenCV, fps). Cadrele se numără citindu-le, nu din metadate."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n, float(fps)


def video_props(path: Path) -> tuple[float, float, float]:
    """(fps, lățime, înălțime) din metadatele clipului, fără să decodeze vreun cadru."""
    cap = cv2.VideoCapture(str(path))
    props = (float(cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS), float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return props


def read_frame(path: Path, frame: int) -> np.ndarray:
    """Un cadru, cu indicele OpenCV (citire secvențială, ca în restul proiectului)."""
    cap = cv2.VideoCapture(str(path))
    image = None
    for _ in range(frame + 1):
        ok, image = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Cadrul {frame} nu există în clip.")
    cap.release()
    return image


def dev_videos(videos_dir: Path = V.VIDEOS, split_csv: Path = SPLIT_CSV) -> list[str]:
    """Doar clipurile din setul de dezvoltare (data/split.csv) care există local."""
    with split_csv.open(newline="", encoding="utf-8") as f:
        dev = [r["video"] for r in csv.DictReader(f) if r["set"] == "dev"]
    return sorted(v for v in dev if (videos_dir / v).exists())


# ---------- etapele, din cache sau calculate ----------

def load_raw(video_path: Path, name: str, video_hash: str, candidate: str,
             predictor: Callable[[Path], Raw] | None = None) -> Raw:
    """Ieșirea candidatului pe toate cadrele, din cache sau calculată (și salvată)."""
    path = raw_cache_path(name, video_hash, candidate)
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    raw = (predictor or run_candidate)(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in raw.items()}), encoding="utf-8")
    if isinstance(raw, schelet.PassResult):
        _store_skeleton(raw.skeleton, name, video_hash, candidate)
    return raw


def _store_skeleton(skeleton: schelet.RiderSkeleton, name: str, video_hash: str,
                    candidate: str) -> schelet.RiderSkeleton:
    stamped = replace(skeleton, meta={**skeleton.meta, "video": name, "video_hash": video_hash})
    schelet.save_skeleton(skeleton_cache_path(name, video_hash, candidate), stamped)
    return stamped


def load_rider_skeleton(video_path: Path, name: str, video_hash: str, candidate: str) -> schelet.RiderSkeleton | None:
    """Scheletul din cache sau, pentru un clip care are doar brut/, calculat o dată și salvat."""
    path = skeleton_cache_path(name, video_hash, candidate)
    if path.exists():
        return schelet.load_skeleton(path)
    skeleton = extract_skeleton(video_path)
    return None if skeleton is None else _store_skeleton(skeleton, name, video_hash, candidate)


def load_ratios(video_path: Path, name: str, video_hash: str, candidate: str) -> dict:
    """Raportul umeri/trunchi pe cadre, din cache sau calculat (și salvat). Nu se păstrează punctele."""
    path = ratios_cache_path(name, video_hash, candidate)
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    ratios = run_ratios(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in ratios.items()}), encoding="utf-8")
    return ratios


def load_motion(video_path: Path, name: str, video_hash: str) -> dict[int, K.Motion]:
    """Mișcarea camerei pe cadre, din cache sau calculată (și salvată)."""
    path = motion_cache_path(name, video_hash)
    if path.exists():
        return {int(k): K.Motion(**v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    motion = K.clip_motion(video_path, load_boxes(video_path, name, video_hash))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): asdict(v) for k, v in motion.items()}), encoding="utf-8")
    return motion


def load_boxes(video_path: Path, name: str, video_hash: str) -> dict:
    """Casetele de cal pe cadre, din cache sau calculate (și salvate)."""
    path = boxes_cache_path(name, video_hash)
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    boxes = detect_horses(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in boxes.items()}), encoding="utf-8")
    return {k: [list(b) for b in v] for k, v in boxes.items()}
