"""
Urmărirea după aspect: recuperarea calului țintă după culoare, peste urmărirea cu ancora pe cal
(ecvestru/urmarire_multi.py, `follow_anchor`), care rămâne neschimbată.

- Șabloanele, pe cadrul click-ului (al pornirii): histograma de culoare HSV (H și S) a casetei calului țintă și a
  călărețului de pe el, apoi la fel pentru celălalt cal (un alt cal cu un călăreț pe el, din cel mai apropiat cadru).
- Pe fiecare cadru, pentru fiecare cal (și călărețul de pe el, după `_on_horse`): distanța la șablonul țintei și la
  șablonul celuilalt. Scorul relativ = distanța la celălalt − distanța la țintă (mai mare = mai asemănător cu ținta).
- La recuperare (ținta lipsește și apare un ID de cal încă nevăzut) se alege, dintre caii de pe cadru, cel cu scorul
  relativ cel mai mare (argmax, fără prag). Varianta „doar ID-uri noi” alege doar dintre cei nevăzuți.
- Varianta „schimb de ID” (oprită implicit): dacă ținta are scor relativ negativ (mai asemănătoare cu celălalt)
  `swap_frames` cadre la rând, ținta trece pe calul cu scorul relativ cel mai mare.

Histograma: 50 × 60 de coșuri pe H [0, 180) și S [0, 256), normalizată min–max, comparată cu distanța Bhattacharyya,
ca în exemplul „Histogram Comparison” din documentația OpenCV. Se citesc din video doar pixelii casetelor; detectorul,
urmăritorul și RTMPose nu rulează din nou (urmele și scheletele vin din cache). Fără Streamlit.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecvestru import urmarire_multi as U

VERSION = 1
HIST_BINS = (50, 60)                 # H, S: exemplul OpenCV „Histogram Comparison”
HIST_RANGES = (0, 180, 0, 256)
SWAP_SECONDS = 1.0                   # varianta „schimb de ID”: valoarea aleasă (k ≈ 1 s), oprită implicit
CACHE_VARIANT = "aspect"
SHOULDERS, HIPS = (5, 6), (U.LEFT_HIP, U.RIGHT_HIP)     # Halpe26


# ---------- histograma și distanța ----------

def histogram(image: np.ndarray, box: tuple) -> np.ndarray | None:
    """Histograma H–S a casetei (ținută în cadru), normalizată min–max; None pentru o casetă goală."""
    import cv2
    h, w = image.shape[:2]
    x0, y0 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
    x1, y1 = min(w, int(math.ceil(box[2]))), min(h, int(math.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), list(HIST_RANGES))
    return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    import cv2
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


@dataclass
class Look:
    """Aspectul unui cal și al călărețului de pe el (rider None dacă nu stă nimeni pe el)."""
    horse: np.ndarray
    rider: np.ndarray | None = None


def look_distance(a: Look, b: Look) -> float:
    """Distanța cal–cal, plus călăreț–călăreț când ambii au călăreț."""
    d = distance(a.horse, b.horse)
    return d + (distance(a.rider, b.rider) if a.rider is not None and b.rider is not None else 0.0)


def relative(a: Look, target: Look, other: Look) -> float:
    """Mai mare = mai asemănător cu ținta decât cu celălalt."""
    return look_distance(a, other) - look_distance(a, target)


def look_at(image: np.ndarray, tracks: U.ClipTracks, frame: int, hid: int) -> Look | None:
    horse = histogram(image, tracks.horses[frame][hid])
    if horse is None:
        return None
    pid = U.rider_on(tracks.horses[frame][hid], tracks.people[frame], tracks.keypoints[frame])
    return Look(horse, None if pid is None else histogram(image, tracks.people[frame][pid]))


# ---------- șabloanele și scorurile pe cadre ----------

def other_horse(tracks: U.ClipTracks, frame: int, target_id: int) -> int | None:
    """Un alt cal decât ținta, cu un călăreț pe el; la mai mulți, cel cu caseta cea mai mare."""
    found = [hid for hid in tracks.horses[frame] if hid != target_id
             and U.rider_on(tracks.horses[frame][hid], tracks.people[frame], tracks.keypoints[frame]) is not None]
    area = lambda h: (tracks.horses[frame][h][2] - tracks.horses[frame][h][0]) * (  # noqa: E731
        tracks.horses[frame][h][3] - tracks.horses[frame][h][1])
    return max(found, key=area, default=None)


def other_frame(tracks: U.ClipTracks, seed_frame: int, target_ids: list[int | None]) -> tuple[int, int] | None:
    """Cel mai apropiat cadru de pornire cu celălalt cal (cu călăreț) lângă ținta urmărită cu ancora pe cal:
    (cadru, ID)."""
    for frame in sorted(range(tracks.n_frames), key=lambda f: abs(f - seed_frame)):
        if target_ids[frame] is not None:
            hid = other_horse(tracks, frame, target_ids[frame])
            if hid is not None:
                return frame, hid
    return None


@dataclass
class Scores:
    """Pe fiecare cadru: ID-ul calului → scorul relativ (mai mare = mai asemănător cu ținta). Plus de unde vin
    șabloanele și cât a durat citirea pixelilor."""
    relative: list[dict[int, float]]
    target_frame: int
    target_id: int
    other_frame: int
    other_id: int
    seconds: float = 0.0
    cpu_seconds: float = 0.0


def compute_scores(video_path: Path, tracks: U.ClipTracks, seed_frame: int, target_id: int,
                   anchor_ids: list[int | None]) -> Scores:
    """Șabloanele (ținta pe cadrul pornirii, celălalt pe cel mai apropiat cadru cu el), apoi o singură citire
    secvențială a clipului: scorul relativ al fiecărui cal pe fiecare cadru."""
    from ecvestru import clipuri_noi as CN
    found = other_frame(tracks, seed_frame, anchor_ids)
    if found is None:
        raise ValueError(f"{video_path.name}: niciun alt cal cu călăreț lângă țintă; fără șablonul celuilalt.")
    started, cpu_started = time.perf_counter(), time.process_time()
    reader = CN.FrameReader(video_path)
    try:
        firsts = sorted({seed_frame, found[0]})
        images = {f: reader.read(f) for f in firsts}
        target = look_at(images[seed_frame], tracks, seed_frame, target_id)
        other = look_at(images[found[0]], tracks, found[0], found[1])
        reader.close()
        reader = CN.FrameReader(video_path)
        out: list[dict[int, float]] = []
        for frame in range(tracks.n_frames):
            image = reader.read(frame)
            row = {}
            for hid in tracks.horses[frame]:
                look = look_at(image, tracks, frame, hid)
                if look is not None:
                    row[hid] = relative(look, target, other)
            out.append(row)
    finally:
        reader.close()
    return Scores(out, seed_frame, target_id, found[0], found[1], time.perf_counter() - started,
                  time.process_time() - cpu_started)


def save_scores(path: Path, scores: Scores) -> None:
    rows = [(f, hid, v) for f, row in enumerate(scores.relative) for hid, v in row.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows=np.array(rows, np.float64).reshape(-1, 3),
                        meta=np.array([len(scores.relative), scores.target_frame, scores.target_id, scores.other_frame,
                                       scores.other_id, scores.seconds, scores.cpu_seconds]))


def load_scores(path: Path) -> Scores:
    with np.load(path) as data:
        rows, meta = data["rows"], data["meta"]
    relative_rows: list[dict[int, float]] = [{} for _ in range(int(meta[0]))]
    for f, hid, v in rows:
        relative_rows[int(f)][int(hid)] = float(v)
    return Scores(relative_rows, int(meta[1]), int(meta[2]), int(meta[3]), int(meta[4]), float(meta[5]),
                  float(meta[6]))


# ---------- urmărirea calului cu aspectul ----------

def follow_aspect(boxes: list[dict[int, tuple]], start: int, start_id: int, scores: list[dict[int, float]],
                  all_visible: bool = True, swap_frames: int | None = None) -> U.Follow:
    """Ca `urmarire_multi.follow_anchor` (aceleași momente de recuperare: ținta lipsește și apare un ID încă nevăzut),
    dar la recuperare se alege calul cu scorul relativ cel mai mare: dintre toți caii de pe cadru (`all_visible`) sau
    doar dintre cei nevăzuți. Cu `swap_frames`, ținta mai trece pe calul cel mai asemănător și când scorul ei rămâne
    negativ atâtea cadre la rând. `switches` = recuperări + schimburi."""
    ids: list[int | None] = [None] * len(boxes)
    ids[start] = start_id
    switches = 0
    for step in (1, -1):
        current, last_box, seen, bad = start_id, boxes[start][start_id], set(boxes[start]), 0
        frame = start + step
        while 0 <= frame < len(boxes):
            here, score = boxes[frame], scores[frame]
            if current not in here:
                new = [tid for tid in here if tid not in seen]
                if new:
                    pool = list(here) if all_visible else new
                    current = max(pool, key=lambda t: (score.get(t, -math.inf),
                                                       -math.dist(U.center(here[t]), U.center(last_box))))
                    switches, bad = switches + 1, 0
            elif swap_frames is not None:
                bad = bad + 1 if score.get(current, 0.0) < 0 else 0
                if bad >= swap_frames:
                    best = max(here, key=lambda t: score.get(t, -math.inf))
                    if best != current:
                        current, switches = best, switches + 1
                    bad = 0
            if current in here:
                ids[frame], last_box = current, here[current]
            seen |= set(here)
            frame += step
    return U.Follow(ids, None, switches)


def choose(tracks: U.ClipTracks, seed_frame: int, start_id: int, scores: Scores, all_visible: bool = True,
           swap_frames: int | None = None) -> U.Choice:
    """Calul cu `follow_aspect`, omul de pe el pe fiecare cadru (ca la ancora pe cal), aceleași categorii."""
    horse = follow_aspect(tracks.horses, seed_frame, start_id, scores.relative, all_visible, swap_frames)
    riders = U.riders_on_target(tracks, horse.ids)
    categories = [U.LOST if h is None else U.NO_SKELETON if p is None else U.ON_C1
                  for h, p in zip(horse.ids, riders, strict=True)]
    return U.Choice(categories, horse.ids, riders, seed_frame, None, horse.switches, 0)


# ---------- același om? (pentru moștenirea răspunsurilor omului) ----------

def torso(points: np.ndarray) -> float:
    return math.dist(points[list(SHOULDERS)].mean(axis=0), points[list(HIPS)].mean(axis=0))


def same_person(a: np.ndarray, b: np.ndarray, max_torsos: float) -> bool:
    """Șoldurile la cel mult `max_torsos` lungimi de trunchi (ale lui `a`)."""
    length = torso(a)
    return length > 0 and math.dist(U.hip_mid(a), U.hip_mid(b)) / length <= max_torsos
