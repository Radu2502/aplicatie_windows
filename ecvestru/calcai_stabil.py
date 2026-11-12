"""
Stratul „călcâi stabil”, opțional și oprit implicit (`masurare.analyze(..., heel_stable=True)`), doar pe calea RTMPose.
După fuziune și interval, scoate din măsurarea piciorului (cu motiv, fără să netezească):
  a) saltul călcâiului: Hampel pe poziția călcâiului față de genunchi, în lungimi de trunchi (fereastra 2h + 1 cadre,
     h = floor(0,25 · fps), k = 3, abaterea minimă 0,05 trunchi, cel puțin 5 cadre măsurate în fereastră);
  b) călcâiul pe piciorul calului: la mai puțin de 0,05 trunchi de un segment cot–copită (față) sau genunchi–copită
     (spate) al calului de sub călăreț, cu punctele AP-10K (ecvestru/cal_puncte.py) cu scor > 0,3.
Modelul RTMPose al stratului: m (implicit) sau, opțional, l (`heel_model="l"`, ecvestru/rtmpose_l.py), care înlocuiește
punctele m pe toată calea RTMPose (orientarea, fuziunea, intervalul). Cu stratul oprit, nimic nu se schimbă.
Fără Streamlit.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ecvestru import cal_puncte as CP
from ecvestru import calitate as C
from ecvestru import rezumat as RZ
from ecvestru import rtmpose as R
from ecvestru import rtmpose_l as RL
from ecvestru import stocare as ST
from ecvestru.rezultat import HEEL_MODEL_M, FrameMeasure

VERSION = 1                  # se mărește când se schimbă un pas sau un prag (invalidează rezultatele cu stratul)

HAMPEL_SECONDS = 0.25        # h = floor(0,25 · fps): fereastra de 15 cadre la ~30 fps
HAMPEL_K = 3.0
HAMPEL_FLOOR = 0.05          # trunchi: zgomotul click-ului
HAMPEL_MIN_FRAMES = 5
MAD_SIGMA = 1.4826

HORSE_LEGS = (("left_elbow", "left_front_paw"), ("right_elbow", "right_front_paw"),
              ("left_knee", "left_back_paw"), ("right_knee", "right_back_paw"))
HORSE_SCORE = CP.SCORE_MIN
HORSE_LEG_TORSO = 0.05


# ---------- a) Hampel ----------

def hampel_half_window(fps: float) -> int:
    return math.floor(HAMPEL_SECONDS * fps)


def hampel_flags(signal: np.ndarray, measured: np.ndarray, half: int, k: float = HAMPEL_K,
                 floor: float = HAMPEL_FLOOR, min_frames: int = HAMPEL_MIN_FRAMES) -> np.ndarray:
    """Cadrele măsurate care se abat de mediana ferestrei cu peste max(k·σ, floor) pe oricare coordonată
    (σ = 1,4826·MAD, pe cadrele măsurate ale ferestrei, cel puțin `min_frames`). O singură trecere."""
    n = len(signal)
    flags = np.zeros(n, bool)
    for i in np.where(measured)[0]:
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = signal[lo:hi][measured[lo:hi]]
        if len(window) < min_frames:
            continue
        middle = np.median(window, axis=0)
        sigma = MAD_SIGMA * np.median(np.abs(window - middle), axis=0)
        flags[i] = bool((np.abs(signal[i] - middle) > np.maximum(k * sigma, floor)).any())
    return flags


# ---------- b) călcâiul pe piciorul calului ----------

def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    length = float(ab @ ab)
    t = 0.0 if length == 0 else min(max(float((p - a) @ ab) / length, 0.0), 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def horse_leg_flags(heels: np.ndarray, horse_points: np.ndarray, horse_scores: np.ndarray, torso: np.ndarray,
                    threshold: float = HORSE_LEG_TORSO) -> np.ndarray:
    """Cadrele cu călcâiul la mai puțin de `threshold` trunchi de un segment de picior al calului (ambele capete cu
    scor > 0,3)."""
    flags = np.zeros(len(heels), bool)
    for i, heel in enumerate(heels):
        if np.isnan(heel).any() or np.isnan(torso[i]):
            continue
        for top, paw in HORSE_LEGS:
            a, b = CP.AP10K.index(top), CP.AP10K.index(paw)
            sa, sb = horse_scores[i, a], horse_scores[i, b]
            if np.isnan(sa) or np.isnan(sb) or sa <= HORSE_SCORE or sb <= HORSE_SCORE:
                continue
            if point_segment_distance(heel, horse_points[i, a], horse_points[i, b]) / torso[i] < threshold:
                flags[i] = True
                break
    return flags


# ---------- stratul ----------

def stabilize(frames: list[FrameMeasure], heels: dict[int, tuple[float, float]], knee: np.ndarray, torso: np.ndarray,
              fps: float, horse_points: np.ndarray, horse_scores: np.ndarray) -> list[FrameMeasure]:
    """Pașii a) și b) pe cadrele date (un segment, în ordine). `heels` = călcâiul folosit pe cadrele cu picior
    măsurat; `knee`, `torso`, punctele calului sunt pe tot clipul, după indicele cadrului. Cadrele scoase pierd doar
    piciorul, cu motiv; restul rămân neschimbate."""
    index = np.array([f.frame for f in frames])
    measured = np.array([f.leg_deg is not None for f in frames])
    if not measured.any():
        return frames
    heel = np.array([heels.get(int(i), (np.nan, np.nan)) for i in index], dtype=float)
    torso_int = float(np.nanmedian(torso[index][measured]))
    signal = (heel - knee[index]) / torso_int
    jump = hampel_flags(signal, measured, hampel_half_window(fps)) & measured
    on_leg = horse_leg_flags(heel, horse_points[index], horse_scores[index], torso[index]) & measured & ~jump
    out = []
    for k, frame in enumerate(frames):
        if jump[k]:
            frame = frame.model_copy(update={"leg_deg": None, "reason": RZ.REASON_HEEL_JUMP})
        elif on_leg[k]:
            frame = frame.model_copy(update={"leg_deg": None, "reason": RZ.REASON_HEEL_ON_HORSE_LEG})
        out.append(frame)
    return out


def apply(video_path: Path, name: str, video_hash: str, candidate: str, raw: RZ.Raw, frames: list[FrameMeasure],
          fps: float, heel_estimator: Callable | None = None, horse_estimator: Callable | None = None,
          heel_model: str = HEEL_MODEL_M) -> list[FrameMeasure]:
    """Stratul pe cadrele intervalului, din cache-urile căii RTMPose (modelul ales, m sau l) și punctele calului
    (calculate la nevoie)."""
    skeleton = ST.load_rider_skeleton(video_path, name, video_hash, candidate)
    boxes = ST.load_boxes(video_path, name, video_hash)
    points = RL.points(video_path, name, video_hash, skeleton, heel_estimator, heel_model)
    used = R.film_raw(raw, frames, R.side_heels(points, skeleton), R.horse_boxes(skeleton, boxes))
    heels = {f.frame: tuple(used[f.frame]["points"]["heel"]) for f in frames if f.leg_deg is not None}
    horse = CP.load_points(video_path, name, video_hash, boxes, C.midpoint(skeleton, "hip"), horse_estimator)
    return stabilize(frames, heels, C.side_part(skeleton, "knee"), C.segment(skeleton, "shoulder", "hip"), fps,
                     horse.points, horse.scores)
