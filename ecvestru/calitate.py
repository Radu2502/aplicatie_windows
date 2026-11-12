"""
Calitatea scheletului, doar din cifre: indicatori pe cadru și pe clip și poarta care scoate piciorul pe cadrele
semnalate. Fără Streamlit.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ecvestru import rezumat as RZ
from ecvestru import schelet as Z
from ecvestru import stocare as ST
from ecvestru import verificare as V
from ecvestru.rezultat import QUALITY_VERSION, FrameMeasure, QualitySummary, QualityWarning

LOWER_ZONE = 2 / 3          # treimea de jos a casetei calului: zona picioarelor calului
LONG_SHIN = 1.2             # tibie „lungă”, față de mediana clipului
HIP_ABOVE_BOX, HIP_INTO_BOX = 0.2, 0.4   # șoldul pe spatele calului, ca în candidat
SAME_POINT_TORSOS = 0.15    # toleranța pentru „același punct”
CHAIN = ("shoulder", "hip", "heel")
LEG_CHAIN = ("shoulder", "hip", "knee", "ankle", "heel")
LEG_PARTS = ("knee", "ankle")
PROPORTION_LOW, PROPORTION_HIGH = 0.6, 1.4   # un segment față de mediana lui pe clip
JUMP_TORSOS = 0.5           # salt brusc între cadre vecine, în lungimi de trunchi
SOURCE_JUMP_TORSOS = 0.2    # la schimbarea sursei: peste P90 al săriturilor obișnuite (tabloul de calitate)
LEG_ANGLE_MIN, LEG_ANGLE_MAX = -50.0, 40.0   # grade: intervalul plauzibil al piciorului
SIGNAL_LEG = "picior_cal"
SIGNAL_PROPORTIONS = "proportii"
SIGNAL_VISIBILITY = "vizibilitate"
SIGNAL_JUMP = "salt"
SIGNAL_SIDE = "side_izolat"
SIGNAL_TEXTS = {SIGNAL_LEG: "călcâi pe zona picioarelor calului, cu tibia lungă",
                SIGNAL_PROPORTIONS: "proporții departe de mediana clipului",
                SIGNAL_VISIBILITY: "visibility sub 0,5 pe lanț",
                SIGNAL_JUMP: "salt brusc față de cadrul anterior",
                SIGNAL_SIDE: "partea folosită se schimbă izolat"}
SIGNAL_SHORT = {SIGNAL_PROPORTIONS: "proporții", SIGNAL_VISIBILITY: "vizibilitate", SIGNAL_JUMP: "salt",
                SIGNAL_SIDE: "parte"}
WARNING_SIGNALS = tuple(SIGNAL_SHORT)   # doar avertismente; SIGNAL_LEG exclude piciorul


@dataclass(frozen=True, eq=False)
class ClipData:
    video: str
    skeleton: Z.RiderSkeleton
    raw: dict
    frames: list
    boxes: dict


def load_clip(video: str) -> ClipData:
    path = V.VIDEOS / video
    video_hash, candidate = ST.file_hash(path), ST.candidate_label()
    raw = ST.load_raw(path, video, video_hash, candidate)
    skeleton = ST.load_rider_skeleton(path, video, video_hash, candidate)
    frames = RZ.frame_measures(raw, len(skeleton.frame))
    return ClipData(video, skeleton, raw, frames, ST.load_boxes(path, video, video_hash))


def side_part(skeleton: Z.RiderSkeleton, part: str) -> np.ndarray:
    """Un punct al părții folosite de candidat, în pixeli, pe toate cadrele (N, 2); NaN fără călăreț."""
    left, right = (Z.LANDMARK_NAMES.index(f"{side}_{part}") for side in ("left", "right"))
    return skeleton.xy_px[np.arange(len(skeleton.frame)), np.where(skeleton.side == "left", left, right)]


def side_visibility(skeleton: Z.RiderSkeleton, part: str) -> np.ndarray:
    left, right = (Z.LANDMARK_NAMES.index(f"{side}_{part}") for side in ("left", "right"))
    return skeleton.visibility[np.arange(len(skeleton.frame)), np.where(skeleton.side == "left", left, right)]


def midpoint(skeleton: Z.RiderSkeleton, part: str) -> np.ndarray:
    return (skeleton.point(f"left_{part}") + skeleton.point(f"right_{part}")) / 2


def segment(skeleton: Z.RiderSkeleton, first: str, second: str) -> np.ndarray:
    return np.linalg.norm(side_part(skeleton, first) - side_part(skeleton, second), axis=1)


def rider_box(hip, boxes: list) -> list | None:
    """Caseta calului de sub călăreț: șoldul pe spatele lui, ca în candidat; cea mai mare, dacă sunt mai multe."""
    near = [b for b in boxes if b[0] <= hip[0] <= b[0] + b[2]
            and b[1] - HIP_ABOVE_BOX * b[3] <= hip[1] <= b[1] + HIP_INTO_BOX * b[3]]
    return max(near, key=lambda b: b[2] * b[3]) if near else None


def heel_in_lower_zone(heel, box: list | None) -> bool:
    if box is None:
        return False
    return bool(box[1] + LOWER_ZONE * box[3] <= heel[1] and box[0] <= heel[0] <= box[0] + box[2])


def measured_leg(clip: ClipData) -> np.ndarray:
    return np.array([f.leg_deg is not None for f in clip.frames])


def low_heel(clip: ClipData) -> np.ndarray:
    """Cadre măsurate cu călcâiul în zona picioarelor calului."""
    hips = midpoint(clip.skeleton, "hip")
    out = np.zeros(len(clip.frames), bool)
    for i in np.where(measured_leg(clip) & clip.skeleton.has_rider)[0]:
        out[i] = heel_in_lower_zone(clip.raw[i]["points"]["heel"], rider_box(hips[i], clip.boxes.get(i, [])))
    return out


def long_shin(skeleton: Z.RiderSkeleton) -> np.ndarray:
    shin = segment(skeleton, "knee", "ankle")
    with np.errstate(invalid="ignore"):
        return shin > LONG_SHIN * np.nanmedian(shin)


def indicator_a(clip: ClipData) -> np.ndarray:
    """Călcâiul în zona picioarelor calului și tibia lungă: pare un om făcut din călăreț și piciorul calului."""
    return low_heel(clip) & long_shin(clip.skeleton)


# ---------- indicatori pe clip ----------

def jitter(skeleton: Z.RiderSkeleton) -> dict[str, float]:
    """Deplasarea medie pe cadru și a doua diferență (`acc`), în lungimi de trunchi; pe partea din cadrul curent."""
    rider, frames = skeleton.has_rider, len(skeleton.frame)
    torso = np.nanmedian(segment(skeleton, "shoulder", "hip"))
    out, pairs = {}, np.where(rider[:-1] & rider[1:])[0]
    triples = np.where(rider[:-2] & rider[1:-1] & rider[2:])[0]
    accs = []
    for part in ("shoulder", "hip", "knee", "ankle", "heel"):
        left, right = (Z.LANDMARK_NAMES.index(f"{side}_{part}") for side in ("left", "right"))
        index = np.where(skeleton.side == "left", left, right)
        px = skeleton.xy_px
        moved = np.linalg.norm(px[pairs + 1, index[pairs]] - px[pairs, index[pairs]], axis=1) / torso
        out[part] = float(moved.mean()) if len(moved) else float("nan")
        at = index[np.minimum(triples + 1, frames - 1)]
        accs.append(np.linalg.norm(px[triples + 2, at] - 2 * px[triples + 1, at] + px[triples, at], axis=1) / torso)
    out["acc"] = float(np.mean(accs)) if len(triples) else float("nan")
    return out


def variation(values: np.ndarray) -> float:
    return float(np.nanstd(values) / np.nanmean(values))


def source_jumps(skeleton: Z.RiderSkeleton) -> dict[str, np.ndarray]:
    """Saltul mijlocului umerilor și al șoldurilor între cadre consecutive, pe tipuri de tranziție."""
    shoulders, hips = midpoint(skeleton, "shoulder"), midpoint(skeleton, "hip")
    torso = np.nanmedian(np.linalg.norm(shoulders - hips, axis=1))
    pairs = np.where(skeleton.has_rider[:-1] & skeleton.has_rider[1:])[0]
    jump = np.maximum(np.linalg.norm(shoulders[pairs + 1] - shoulders[pairs], axis=1),
                      np.linalg.norm(hips[pairs + 1] - hips[pairs], axis=1)) / torso
    a, b = skeleton.source[pairs], skeleton.source[pairs + 1]
    return {f"{p}>{q}": jump[(a == p) & (b == q)] for p in (Z.SOURCE_FULL, Z.SOURCE_CROP)
            for q in (Z.SOURCE_FULL, Z.SOURCE_CROP)}


def reason_counts(clip: ClipData) -> Counter:
    return Counter(f.reason for f in clip.frames if f.reason)


# ---------- semnale pe cadru ----------

def bad_proportions(skeleton: Z.RiderSkeleton) -> np.ndarray:
    """Trunchiul, coapsa sau tibia sunt departe de mediana lor pe clip."""
    out = np.zeros(len(skeleton.frame), bool)
    for first, second in (("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")):
        length = segment(skeleton, first, second)
        ratio = length / np.nanmedian(length)
        with np.errstate(invalid="ignore"):
            out |= (ratio < PROPORTION_LOW) | (ratio > PROPORTION_HIGH)
    return out


def low_chain_visibility(skeleton: Z.RiderSkeleton) -> np.ndarray:
    out = np.zeros(len(skeleton.frame), bool)
    for part in LEG_CHAIN:
        with np.errstate(invalid="ignore"):
            out |= side_visibility(skeleton, part) < V.MIN_VISIBILITY
    return out & skeleton.has_rider


def sudden_jump(skeleton: Z.RiderSkeleton) -> np.ndarray:
    """Cadrul sare față de cel anterior; la schimbarea sursei pragul e mai mic."""
    shoulders, hips = midpoint(skeleton, "shoulder"), midpoint(skeleton, "hip")
    torso = np.nanmedian(np.linalg.norm(shoulders - hips, axis=1))
    out = np.zeros(len(skeleton.frame), bool)
    for t in np.where(skeleton.has_rider[:-1] & skeleton.has_rider[1:])[0]:
        jump = max(np.linalg.norm(shoulders[t + 1] - shoulders[t]), np.linalg.norm(hips[t + 1] - hips[t])) / torso
        out[t + 1] = jump > (SOURCE_JUMP_TORSOS if skeleton.source[t] != skeleton.source[t + 1] else JUMP_TORSOS)
    return out


def isolated_side(skeleton: Z.RiderSkeleton) -> np.ndarray:
    side, rider = skeleton.side, skeleton.has_rider
    out = np.zeros(len(side), bool)
    middle = rider[:-2] & rider[1:-1] & rider[2:] & (side[1:-1] != side[:-2]) & (side[1:-1] != side[2:])
    out[1:-1] = middle
    return out


def frame_signals(clip: ClipData) -> dict[str, np.ndarray]:
    sk = clip.skeleton
    return {SIGNAL_LEG: indicator_a(clip), SIGNAL_PROPORTIONS: bad_proportions(sk),
            SIGNAL_VISIBILITY: low_chain_visibility(sk), SIGNAL_JUMP: sudden_jump(sk), SIGNAL_SIDE: isolated_side(sk)}


def any_signal(signals: dict[str, np.ndarray]) -> np.ndarray:
    return np.any(list(signals.values()), axis=0)


def clean_frames(clip: ClipData, signals: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Cadre cu călăreț și fără niciun semnal."""
    return clip.skeleton.has_rider & ~any_signal(signals or frame_signals(clip))


def frame_reasons(signals: dict[str, np.ndarray], frame: int) -> list[str]:
    return [SIGNAL_TEXTS[name] for name, flags in signals.items() if flags[frame]]


# ---------- poarta A ----------

def gate_frames(clip: ClipData, leg_gate: bool = True) -> list[FrameMeasure]:
    """Indicatorul A și unghiul neplauzibil scot piciorul din măsurare, cu motiv; trunchiul rămâne. Celelalte semnale
    doar avertizează. Cu `leg_gate=False` (calea RTMPose, unde piciorul a fost deja decis) doar avertizează."""
    signals = frame_signals(clip)
    out = []
    for frame in clip.frames:
        i = frame.frame
        update = {"warnings": [name for name in WARNING_SIGNALS if signals[name][i]]}
        if not leg_gate:
            pass
        elif signals[SIGNAL_LEG][i]:
            update |= {"leg_deg": None, "reason": RZ.REASON_HEEL_ON_HORSE}
        elif frame.leg_deg is not None and not LEG_ANGLE_MIN <= frame.leg_deg <= LEG_ANGLE_MAX:
            update |= {"leg_deg": None, "reason": RZ.REASON_IMPLAUSIBLE_ANGLE}
        out.append(frame.model_copy(update=update))
    return out


def summarize_quality(frames: list[FrameMeasure]) -> QualitySummary:
    counts = Counter(code for frame in frames for code in frame.warnings)
    return QualitySummary(version=QUALITY_VERSION,
                          excluded_frames=sum(frame.reason == RZ.REASON_HEEL_ON_HORSE for frame in frames),
                          excluded_reason=RZ.REASON_TEXTS[RZ.REASON_HEEL_ON_HORSE],
                          implausible_frames=sum(frame.reason == RZ.REASON_IMPLAUSIBLE_ANGLE for frame in frames),
                          warnings=[QualityWarning(code=code, text=SIGNAL_TEXTS[code], frames=counts[code])
                                    for code in WARNING_SIGNALS if counts[code]])
