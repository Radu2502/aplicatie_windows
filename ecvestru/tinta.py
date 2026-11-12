"""
Călărețul țintă: din click-ul utilizatorului, urmărește caseta calului înainte și înapoi, doar din
geometrie și mișcare (poziție, viteză, mărime). Punctele rămân cele ale candidatului: un cadru
se păstrează dacă șoldul e pe caseta țintei; altfel e nemăsurat, cu motiv. Fără Streamlit.

Când două casete sunt la fel de plauzibile, identitatea nu se ghicește: cadrele sunt „țintă nesigură”
până la o separare clară. După un gol prea lung, ținta e pierdută până la capătul clipului.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ecvestru import constante_candidat as KC
from ecvestru.rezultat import TargetSeed

Box = tuple[float, float, float, float]   # x, y, lățime, înălțime

REASON_UNSURE = "tinta_nesigura"
REASON_LOST = "tinta_pierduta"
REASON_OTHER = "alt_calaret"
DETECTOR_SCORE = KC.HORSE_SCORE      # aceleași valori ca în candidat (ecvestru/constante_candidat.py)
DETECTOR_RESULTS = KC.HORSE_RESULTS
# Parametrii urmăririi (ficși).
SEED_MAX_DISTANCE = 1.0    # click-ul față de casetă, în lățimi de casetă
STEP_GATE = 1.0            # distanța față de poziția prezisă, în lățimi de casetă
AMBIGUITY_MARGIN = 0.15    # două casete mai apropiate de atât între ele, ca distanță: ținta e nesigură
MAX_SIZE_RATIO = 1.6       # o casetă mult mai mare sau mai mică nu e același cal
MAX_GAP_SECONDS = 2.0      # după atâtea cadre fără potrivire sigură, ținta e pierdută
VELOCITY_SMOOTHING = 0.5
ON_BOX_ABOVE = 0.2         # șoldul călărețului față de casetă, ca în candidat, în înălțimi de casetă
ON_BOX_BELOW = 0.4


@dataclass(frozen=True)
class TargetState:
    box: Box | None
    status: str | None     # None = sigur; altfel motivul


def max_gap_frames(fps: float) -> int:
    """Golul permis, în cadre: regula e în secunde, ca să fie aceeași la orice fps."""
    return round(MAX_GAP_SECONDS * fps)


def seed_key(target: TargetSeed) -> str:
    return f"{target.frame}-{round(target.x)}-{round(target.y)}"


def _centre(box: Box) -> tuple[float, float]:
    x, y, width, height = box
    return x + width / 2, y + height / 2


def _point_distance(box: Box, x: float, y: float) -> float:
    left, top, width, height = box
    return math.hypot(max(left - x, 0.0, x - left - width), max(top - y, 0.0, y - top - height)) / width


def seed_box(boxes: list[Box], target: TargetSeed) -> Box:
    best = min(boxes, key=lambda b: _point_distance(b, target.x, target.y), default=None)
    if best is None or _point_distance(best, target.x, target.y) > SEED_MAX_DISTANCE:
        raise ValueError("Nu am găsit un cal lângă punctul ales. Alege alt cadru sau dă click pe cal.")
    return best


def _similar_size(a: Box, b: Box) -> bool:
    _, _, width_a, height_a = a
    _, _, width_b, height_b = b
    return max(width_a / width_b, width_b / width_a, height_a / height_b, height_b / height_a) <= MAX_SIZE_RATIO


def _follow(start: Box, first: int, frames, boxes: dict, motion: dict, max_gap: int) -> dict[int, TargetState]:
    """Urmărește caseta. `drift` e mișcarea camerei acumulată de la ultima potrivire: scena se mută cu
    atât în imagine, deci și poziția prezisă."""
    states, box, last = {}, start, first
    velocity, drift = (0.0, 0.0), (0.0, 0.0)
    lost = False
    for frame in frames:
        forward = frame > last
        step = motion.get(frame if forward else frame + 1)
        if step is not None:
            sign = 1 if forward else -1
            drift = (drift[0] + sign * step.dx, drift[1] + sign * step.dy)
        if lost:
            states[frame] = TargetState(None, REASON_LOST)
            continue
        steps = frame - last
        centre = _centre(box)
        predicted = (centre[0] + velocity[0] * steps + drift[0], centre[1] + velocity[1] * steps + drift[1])
        _, _, width, height = box
        options = sorted((math.hypot(_centre(b)[0] - predicted[0], _centre(b)[1] - predicted[1]) / width, tuple(b))
                         for b in boxes.get(frame, []) if _similar_size(box, b))
        options = [o for o in options if o[0] <= STEP_GATE]
        unsure = len(options) > 1 and options[1][0] - options[0][0] < AMBIGUITY_MARGIN
        if options and not unsure:
            found = options[0][1]
            now = _centre(found)
            velocity = tuple(VELOCITY_SMOOTHING * (n - c - d) / steps + (1 - VELOCITY_SMOOTHING) * v
                             for n, c, d, v in zip(now, centre, drift, velocity, strict=True))
            box, last, drift = found, frame, (0.0, 0.0)
            states[frame] = TargetState(found, None)
        elif abs(steps) > max_gap:
            lost = True
            states[frame] = TargetState(None, REASON_LOST)
        elif unsure:
            states[frame] = TargetState(None, REASON_UNSURE)
        else:
            shifted = (predicted[0] - width / 2, predicted[1] - height / 2, width, height)
            states[frame] = TargetState(shifted, None)
    return states


def _start(boxes: dict, n_frames: int, target: TargetSeed, max_gap: int) -> tuple[int, Box]:
    """Cadrul de pornire și caseta; dacă în cadrul ales nu e detectat calul, ia cel mai apropiat cadru."""
    nearby = sorted(range(max(0, target.frame - max_gap), min(n_frames, target.frame + max_gap + 1)),
                    key=lambda f: abs(f - target.frame))
    first = next((f for f in nearby if boxes.get(f)), target.frame)
    return first, tuple(seed_box([tuple(b) for b in boxes.get(first, [])], target))


def start_of(boxes: dict, n_frames: int, target: TargetSeed, *, fps: float) -> tuple[int, Box]:
    """Cadrul de pornire și caseta calului pentru un click, ca la urmărire (ValueError dacă nu e lângă un cal)."""
    return _start(boxes, n_frames, target, max_gap_frames(fps))


def _merge(left: TargetState | None, right: TargetState | None) -> TargetState:
    """Între două click-uri: o potrivire sigură câștigă; altfel, nesigur înaintea pierdut."""
    options = [s for s in (left, right) if s is not None]
    for wanted in (None, REASON_UNSURE):
        for state in options:
            if state.status == wanted:
                return state
    return TargetState(None, REASON_LOST)


def track_target(boxes: dict, n_frames: int, seeds, motion: dict | None = None, *,
                 fps: float) -> dict[int, TargetState]:
    """Starea țintei pe fiecare cadru, din unul sau mai multe click-uri și din mișcarea camerei."""
    seeds = [seeds] if isinstance(seeds, TargetSeed) else seeds
    motion = motion or {}
    max_gap = max_gap_frames(fps)
    starts = sorted(_start(boxes, n_frames, seed, max_gap) for seed in seeds)
    forward, backward = {}, {}
    for index, (first, box) in enumerate(starts):
        next_first = starts[index + 1][0] if index + 1 < len(starts) else n_frames
        previous_first = starts[index - 1][0] if index else -1
        forward |= _follow(box, first, range(first + 1, next_first), boxes, motion, max_gap)
        backward |= _follow(box, first, range(first - 1, previous_first, -1), boxes, motion, max_gap)
    states = {f: _merge(forward.get(f), backward.get(f)) for f in range(n_frames)}
    return states | {first: TargetState(box, None) for first, box in starts}


def lost_intervals(states: dict[int, TargetState], segment: tuple[int, int]) -> list[tuple[int, int]]:
    """Unde s-a pierdut ținta, ca utilizatorul să știe unde mai dă un click."""
    out: list[tuple[int, int]] = []
    start, end = segment
    for frame in range(start, end + 1):
        state = states.get(frame)
        if state is None or state.status != REASON_LOST:
            continue
        if out and out[-1][1] == frame - 1:
            out[-1] = (out[-1][0], frame)
        else:
            out.append((frame, frame))
    return out


def on_box(hip, box: Box) -> bool:
    left, top, width, height = box
    hip_x, hip_y = hip
    return left <= hip_x <= left + width and top - ON_BOX_ABOVE * height <= hip_y <= top + ON_BOX_BELOW * height


def filter_raw(raw: dict, states: dict[int, TargetState], n_frames: int) -> dict:
    """Păstrează doar răspunsurile de pe țintă; un cadru deja fără răspuns își păstrează motivul."""
    out = {}
    for frame in range(n_frames):
        answer = raw.get(frame)
        state = states.get(frame, TargetState(None, REASON_LOST))
        if answer is None and state.status != REASON_LOST:
            continue
        hip = ((answer or {}).get("points") or {}).get("hip")
        if state.status == REASON_LOST:
            out[frame] = {"points": None, "motiv": REASON_LOST}
        elif hip is None:
            out[frame] = answer
        elif state.status is not None:
            out[frame] = {"points": None, "motiv": state.status}
        elif not on_box(hip, state.box):
            out[frame] = {"points": None, "motiv": REASON_OTHER}
        else:
            out[frame] = answer
    return out


def detect_horses(video_path: Path, model_path: str) -> dict[int, list[Box]]:
    """Casetele de cal pe fiecare cadru, cu detectorul candidatului."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    options = vision.ObjectDetectorOptions(base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                                           score_threshold=DETECTOR_SCORE, category_allowlist=[KC.HORSE_CATEGORY],
                                           max_results=DETECTOR_RESULTS)
    out: dict[int, list[Box]] = {}
    cap = cv2.VideoCapture(str(video_path))
    with vision.ObjectDetector.create_from_options(options) as detector:
        frame = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            found = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).detections
            out[frame] = [(float(d.bounding_box.origin_x), float(d.bounding_box.origin_y),
                           float(d.bounding_box.width), float(d.bounding_box.height)) for d in found]
            frame += 1
    cap.release()
    return out
