"""
Scheletul calului, opțiune oprită implicit („Scheletul calului” în pagina „Analiză”): cele 17 puncte AP-10K
(ecvestru/cal_puncte.py, RTMPose-m AP-10K) ale calului țintă, pe cadrele analizate, în film și în JSON
(`horse_keypoints`, schema 1.6). Doar afișare: nicio valoare a analizei (unghiuri, medianele, raportul) nu vine din
ele. Cu opțiunea oprită nu rulează nimic de aici, iar rezultatul și filmul sunt cele obișnuite.

Calul țintă:
  - metoda veche (ecvestru/masurare.py): calul de sub călărețul ales, cu regula existentă (cal_puncte.select_box /
    calitate.rider_box pe șoldul scheletului candidatului; fără schelet, cea mai mare casetă). Cache-ul etapei e cel
    al punctelor calului (cal_puncte.load_points: data/results/analiza/cal/, cheia: clipul, candidatul, CAL_VERSION,
    modelul, versiunea casetelor), comun cu stratul „călcâi stabil”;
  - urmărirea țintei (ecvestru/analiza_urmarita.py): caseta calului urmărit pe fiecare cadru. Cache-ul etapei:
    data/results/analiza/cal_urmarire/, cheia: clipul, varianta urmăririi, ținta (click-ul), VERSION, CAL_VERSION și
    modelul.
Punctele se calculează o dată pe tot clipul (fiecare cadru cu casetă), apoi se iau doar cadrele intervalului.
Fără Streamlit.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ecvestru import cal_puncte as CP
from ecvestru import calitate as C
from ecvestru import stocare as ST
from ecvestru import tinta as T
from ecvestru.rezultat import HORSE_SCHEMA_VERSION, AnalysisResult, HorseFrame, HorseKeypoints, HorsePoint, TargetSeed

VERSION = 1                       # se mărește când se schimbă pașii etapei (invalidează cache-ul urmăririi)
APP_DEFAULT_HORSE_SKELETON = False  # opțiunea e oprită implicit în aplicație
SCORE_MIN = CP.SCORE_MIN          # sub prag, punctul nu se desenează și n-are x, y în JSON
COORD_DECIMALS = 1                # pixeli, în JSON
SCORE_DECIMALS = 3
TRACKED_FOLDER = "cal_urmarire"


# ---------- etapa: punctele pe tot clipul, din cache sau calculate ----------

def old_method_points(video_path: Path, name: str, video_hash: str, candidate: str,
                      estimator: Callable | None = None) -> CP.ClipRun:
    """Metoda veche: calul de sub șoldul călărețului ales de candidat (cache-ul cal/ al punctelor calului)."""
    skeleton = ST.load_rider_skeleton(Path(video_path), name, video_hash, candidate)
    hips = None if skeleton is None else C.midpoint(skeleton, "hip")
    boxes = ST.load_boxes(Path(video_path), name, video_hash)
    return CP.load_points(Path(video_path), name, video_hash, boxes, hips, estimator)


def tracked_cache_path(name: str, video_hash: str, label: str, seed: TargetSeed) -> Path:
    return (ST.results_dir() / TRACKED_FOLDER /
            f"{ST.clip_key(name, video_hash)}__{label.replace('@', '_')}__t{T.seed_key(seed)}__h{VERSION}"
            f"_a{CP.CAL_VERSION}_{CP.MODEL_ZIP_SHA256[:8]}.npz")


def tracked_points(video_path: Path, name: str, video_hash: str, label: str, seed: TargetSeed,
                   boxes: Callable[[], np.ndarray], estimator: Callable | None = None) -> CP.ClipRun:
    """Urmărirea țintei: punctele pe caseta calului urmărit, din cache sau calculate (și salvate). `boxes` dă casetele
    (N, 4) x0, y0, x1, y1 (NaN = fără cal) și se apelează doar când cache-ul lipsește."""
    path = tracked_cache_path(name, video_hash, label, seed)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return CP.ClipRun(data["points"], data["scores"], data["boxes"], float(data["seconds"]))
    run = CP.run_boxes(Path(video_path), boxes(), estimator or CP.HorsePose())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=run.points, scores=run.scores, boxes=run.boxes, seconds=run.seconds)
    return run


# ---------- în rezultat ----------

def _finite(value: float) -> bool:
    return value is not None and math.isfinite(value)


def point(name: str, xy: np.ndarray, score: float, width: float, height: float) -> HorsePoint:
    """Un punct: x, y doar cu scorul ≥ SCORE_MIN și în imagine; scorul rămâne (0 dacă lipsește)."""
    score = float(score) if _finite(score) else 0.0
    x, y = float(xy[0]), float(xy[1])
    shown = score >= SCORE_MIN and _finite(x) and _finite(y) and 0 <= x <= width and 0 <= y <= height
    return HorsePoint(name=name, x=round(x, COORD_DECIMALS) if shown else None,
                      y=round(y, COORD_DECIMALS) if shown else None, score=round(score, SCORE_DECIMALS))


def keypoints(run: CP.ClipRun, frames: list[int], width: float, height: float) -> HorseKeypoints:
    """Punctele pe cadrele cerute; un cadru fără casetă de cal (sau dincolo de clip) are `points` None."""
    out = []
    for frame in frames:
        if frame >= len(run.boxes) or np.isnan(run.boxes[frame]).any():
            out.append(HorseFrame(frame=frame, points=None))
            continue
        out.append(HorseFrame(frame=frame, points=[point(n, run.points[frame, i], run.scores[frame, i], width, height)
                                                   for i, n in enumerate(CP.AP10K)]))
    return HorseKeypoints(version=VERSION, model=CP.MODEL_ZIP_SHA256[:8], score_min=SCORE_MIN, frames=out)


def attach(result: AnalysisResult, run: CP.ClipRun, width: float, height: float) -> AnalysisResult:
    """Rezultatul cu scheletul calului pe cadrele lui (schema 1.6); restul câmpurilor rămâne neschimbat."""
    horse = keypoints(run, [f.frame for f in result.frames], width, height)
    return result.model_copy(update={"schema_version": HORSE_SCHEMA_VERSION, "horse_keypoints": horse})


def with_old_method(video_path: Path, result: AnalysisResult, estimator: Callable | None = None) -> AnalysisResult:
    """Metoda veche: rezultatul cu punctele calului de sub călărețul ales."""
    run = old_method_points(video_path, result.video, result.video_hash, result.candidate, estimator)
    _, width, height = ST.video_props(Path(video_path))
    return attach(result, run, width, height)


def drawn_points(horse: HorseKeypoints | None) -> dict[int, list[tuple[float, float] | None]]:
    """Pentru film: cadru → cele 17 puncte (x, y) sau None (sub prag sau în afara imaginii); doar cadrele cu cal."""
    if horse is None:
        return {}
    return {f.frame: [None if p.x is None or p.y is None else (p.x, p.y) for p in f.points]
            for f in horse.frames if f.points is not None}
