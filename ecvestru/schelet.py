"""
Scheletul complet al călărețului ales: cele 33 de puncte MediaPipe pe fiecare cadru, păstrate pentru
măsurători ulterioare. Fără Streamlit.

Candidatul nu dă mai departe punctele 3D și, pe decupaj, păstrează doar x, y și visibility.
De aceea rulează o instanță privată a lui, cu un strat care doar înregistrează rezultatele detectorilor:
research/ nu se modifică, iar alegerile rămân ale candidatului. Persoana aleasă se regăsește în
rezultatele brute; dacă pe decupaj nu se regăsește una singură, cadrul rămâne fără puncte (nu se ghicește).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe
import numpy as np
from mediapipe.tasks.python import vision

from ecvestru import verificare as V

SCHEMA_VERSION = 3
LANDMARK_NAMES = tuple(point.name.lower() for point in vision.PoseLandmark)
N_POINTS = len(LANDMARK_NAMES)
SOURCE_NONE, SOURCE_FULL, SOURCE_CROP = 0, 1, 2
REASON_UNIDENTIFIED = "schelet_neidentificat"
REASON_HIDDEN = "puncte_invizibile"   # același cod ca în candidat și în brut/
REASON_NOT_CHOSEN = "neselectata"     # persoană respinsă pentru care candidatul nu dă un motiv al ei
CROP_TOLERANCE = 1e-9   # cât se poate abate reconstrucția din decupaj față de punctele candidatului
SPACES = ("px", "norm", "world")
SAME_RIDER_TORSOS = 0.3     # până aici, umerii și șoldurile sunt ale călărețului ales (în lungimi de trunchi)
UNSURE_RIDER_TORSOS = 1.0   # până aici, persoana e prea aproape ca să fie sigur alta
UNSURE_RIDER_IOU = 0.5      # casete suprapuse cel puțin atât: tot nesigur
BODY_POINTS = slice(11, 33)  # fără față
VISIBILITY_MID = V.MIN_VISIBILITY   # sub el, punctul nu intră în lanț (ca în candidat și în brut/)
VISIBILITY_HIGH = 0.8               # treptele sunt doar pentru afișare
TIER_LOW, TIER_MID, TIER_HIGH = "scazuta", "medie", "ridicata"
NOSE = 0
# doar în film: nasul, urechile, brațele, șoldurile și picioarele
DRAWN_POINTS = (NOSE, 7, 8, *range(11, 17), *range(23, 31))
DRAWN_LINES = ((NOSE, 7), (NOSE, 8), *(line for line in V.CONNECTIONS if set(line) <= set(DRAWN_POINTS)))
ARRAYS = {"frame": "i4", "time_s": "f8", "xy_norm": "f8", "z": "f8", "visibility": "f4", "presence": "f4",
          "world_xyz": "f4", "source": "i1", "reason": "U30", "side": "U5", "facing": "f4", "crop_box": "f4",
          "n_poses": "i1", "rejected_count": "i1", "rejected_xy_norm": "f8", "rejected_z": "f8",
          "rejected_visibility": "f4", "rejected_presence": "f4", "rejected_world_xyz": "f4", "rejected_source": "i1",
          "rejected_reason": "U30", "rejected_crop_box": "f4", "rejected_is_rider": "?", "rejected_unsure": "?"}


@dataclass(frozen=True, eq=False)
class RiderSkeleton:
    """Toate cadrele clipului (N). NaN pe punctele cadrelor fără călăreț identificat."""
    frame: np.ndarray
    time_s: np.ndarray
    xy_norm: np.ndarray     # (N, 33, 2)
    z: np.ndarray           # (N, 33)
    visibility: np.ndarray  # (N, 33)
    presence: np.ndarray    # (N, 33)
    world_xyz: np.ndarray   # (N, 33, 3)
    source: np.ndarray
    reason: np.ndarray
    side: np.ndarray
    facing: np.ndarray
    crop_box: np.ndarray    # (N, 4): x0, y0, x1, y1 în pixeli
    n_poses: np.ndarray
    rejected_count: np.ndarray       # (N): câte persoane respinse are cadrul
    rejected_xy_norm: np.ndarray     # (N, R, 33, 2), R = cele mai multe persoane respinse pe un cadru al clipului
    rejected_z: np.ndarray           # (N, R, 33)
    rejected_visibility: np.ndarray  # (N, R, 33)
    rejected_presence: np.ndarray    # (N, R, 33)
    rejected_world_xyz: np.ndarray   # (N, R, 33, 3)
    rejected_source: np.ndarray      # (N, R); 0 = loc gol
    rejected_reason: np.ndarray      # (N, R)
    rejected_crop_box: np.ndarray    # (N, R, 4)
    rejected_is_rider: np.ndarray    # (N, R): e călărețul ales, găsit a doua oară
    rejected_unsure: np.ndarray      # (N, R): nu se poate spune; rejected_is_rider rămâne False
    meta: dict

    @property
    def xy_px(self) -> np.ndarray:
        return self.xy_norm * (self.meta["width"], self.meta["height"])

    @property
    def has_rider(self) -> np.ndarray:
        return self.source != SOURCE_NONE

    def rejected_at(self, index: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """Persoanele respinse de pe un cadru: (puncte în pixeli, visibility, motiv)."""
        scale = (self.meta["width"], self.meta["height"])
        return [(self.rejected_xy_norm[index, k] * scale, self.rejected_visibility[index, k],
                 str(self.rejected_reason[index, k])) for k in range(int(self.rejected_count[index]))]

    def shown_rejected_at(self, index: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """Persoanele respinse de desenat: fără dublurile călărețului ales."""
        return [person for k, person in enumerate(self.rejected_at(index)) if not self.rejected_is_rider[index, k]]

    def point(self, name: str, space: str = "px") -> np.ndarray:
        """Un punct pe toate cadrele: (N, 2) pentru px și norm, (N, 3) pentru world."""
        if name not in LANDMARK_NAMES:
            raise ValueError(f"Punct necunoscut: {name}.")
        if space not in SPACES:
            raise ValueError(f"Spațiu necunoscut: {space}; sunt {', '.join(SPACES)}.")
        data = {"px": self.xy_px, "norm": self.xy_norm, "world": self.world_xyz}[space]
        return data[:, LANDMARK_NAMES.index(name)]

    def chain(self) -> dict[str, np.ndarray]:
        """Umăr, șold, călcâi în pixeli, pe partea aleasă de candidat; NaN sub pragul de vizibilitate, ca în brut/."""
        out = {}
        for part in V.CHAIN:
            left, right = (LANDMARK_NAMES.index(f"{side}_{part}") for side in ("left", "right"))
            index = np.where(self.side == "left", left, right)
            rows = np.arange(len(self.frame))
            xy = self.xy_px[rows, index].copy()
            xy[~(self.visibility[rows, index] >= V.MIN_VISIBILITY)] = np.nan
            out[part] = xy
        return out


class PassResult(dict):
    """Ieșirea candidatului (ca în brut/), împreună cu scheletul din aceeași trecere."""

    def __init__(self, raw: dict, skeleton: RiderSkeleton):
        super().__init__(raw)
        self.skeleton = skeleton


def is_far_point(point: int, side: str) -> bool:
    """Punctul e pe cealaltă parte decât cea folosită de candidat (MediaPipe: impar = stânga, par = dreapta)."""
    if side not in ("left", "right") or point == NOSE:
        return False
    return (point % 2 == 0) == (side == "left")


def is_far_line(line: tuple[int, int], side: str) -> bool:
    """Linia ține de cealaltă parte: nu atinge deloc partea folosită (umerii și șoldurile o ating)."""
    return any(is_far_point(point, side) for point in line) and not any(
        point != NOSE and not is_far_point(point, side) for point in line)


def visibility_tier(value: float) -> str | None:
    """Treapta de încredere a unui punct: sub 0,5 / 0,5–0,8 / peste 0,8; None dacă punctul lipsește."""
    if np.isnan(value):
        return None
    if value < VISIBILITY_MID:
        return TIER_LOW
    return TIER_MID if value <= VISIBILITY_HIGH else TIER_HIGH


# ---------- stratul de înregistrare ----------

class Recording:
    """Rezultatele brute ale detectorilor pe cadrul curent."""

    def __init__(self):
        self.single, self.crops, self.backs = None, [], []   # backs: pozele din crops, pe cadrul întreg


class _RecordingLandmarker:
    def __init__(self, landmarker, log: Recording):
        self._landmarker, self._log = landmarker, log

    def __enter__(self):
        self._landmarker.__enter__()
        return self

    def __exit__(self, *details):
        return self._landmarker.__exit__(*details)

    def detect_for_video(self, image, timestamp):
        self._log.single = self._landmarker.detect_for_video(image, timestamp)
        self._log.crops, self._log.backs = [], []
        return self._log.single

    def detect(self, image):
        result = self._landmarker.detect(image)
        self._log.crops.append(result)
        return result


def _recording_candidate(log: Recording):
    """O instanță nouă a candidatului, doar pentru această trecere; fișierul lui rămâne neatins."""
    spec = importlib.util.spec_from_file_location("candidat_schelet", V.CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real, pose_on_crop = module.vision, module._pose_on_crop

    def recorded_pose_on_crop(detector, rgb, box):
        before = len(log.crops)
        poses = pose_on_crop(detector, rgb, box)
        if len(log.crops) > before:
            log.backs.append(poses)
        return poses

    def create(options):
        return _RecordingLandmarker(real.PoseLandmarker.create_from_options(options), log)

    module.vision = SimpleNamespace(PoseLandmarker=SimpleNamespace(create_from_options=create),
                                    PoseLandmarkerOptions=real.PoseLandmarkerOptions, RunningMode=real.RunningMode,
                                    ObjectDetector=real.ObjectDetector,
                                    ObjectDetectorOptions=real.ObjectDetectorOptions)
    module._pose_on_crop = recorded_pose_on_crop
    return module


# ---------- identificarea persoanei alese ----------

def _optional(value) -> float:
    return float("nan") if value is None else float(value)


def _axis_box(raw: np.ndarray, full: np.ndarray, size: int) -> tuple[int, int] | None:
    top, bottom = int(raw.argmax()), int(raw.argmin())
    if raw[top] == raw[bottom]:
        return None
    span = (full[top] - full[bottom]) * size / (raw[top] - raw[bottom])
    low, high = round(full[top] * size - raw[top] * span), round(full[top] * size - raw[top] * span + span)
    if high <= low or np.abs((low + raw * (high - low)) / size - full).max() > CROP_TOLERANCE:
        return None
    return low, high


def _crop_box(raw_xy: np.ndarray, full_xy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Decupajul (x0, y0, x1, y1), reconstruit din cele două seturi de coordonate; None dacă nu se potrivesc."""
    across, down = _axis_box(raw_xy[:, 0], full_xy[:, 0], width), _axis_box(raw_xy[:, 1], full_xy[:, 1], height)
    return None if across is None or down is None else (across[0], down[0], across[1], down[1])


def _matches_on_crops(xy: np.ndarray, visibility: np.ndarray, crops: list, width: int, height: int) -> list:
    hits = []
    for result in crops:
        for raw, world in zip(result.pose_landmarks, result.pose_world_landmarks, strict=True):
            if not np.array_equal([p.visibility for p in raw], visibility):
                continue
            box = _crop_box(np.array([(p.x, p.y) for p in raw]), xy, width, height)
            if box is not None:
                hits.append((raw, world, box))
    return hits


def _rejected_people(chosen, reason, single, crops, backs, hit_raw, unidentified: bool, width: int,
                     height: int) -> list:
    """Toate pozele cadrului, în afară de cea aleasă: (poză, world, xy, sursă, motiv, decupaj, scara lui z).
    Pe un cadru neidentificat nu se știe care poză de pe decupaj e cea aleasă, deci acelea nu se trec."""
    people = []
    if single and single.pose_landmarks and chosen is not single.pose_landmarks[0]:
        pose = single.pose_landmarks[0]
        people.append((pose, single.pose_world_landmarks[0], np.array([(p.x, p.y) for p in pose]), SOURCE_FULL,
                       reason if not chosen else REASON_NOT_CHOSEN, None, 1.0))
    if unidentified:
        return people
    for result, back in zip(crops, backs, strict=True):
        for raw, world, full in zip(result.pose_landmarks, result.pose_world_landmarks, back, strict=True):
            if raw is hit_raw:
                continue
            xy = np.array([(p.x, p.y) for p in full])
            box = _crop_box(np.array([(p.x, p.y) for p in raw]), xy, width, height)
            scale = float("nan") if box is None else (box[2] - box[0]) / width
            people.append((raw, world, xy, SOURCE_CROP, REASON_NOT_CHOSEN, box, scale))
    return people


def _midpoint(xy: np.ndarray, part: str) -> np.ndarray:
    return (xy[LANDMARK_NAMES.index(f"left_{part}")] + xy[LANDMARK_NAMES.index(f"right_{part}")]) / 2


def _box_overlap(first: np.ndarray, second: np.ndarray) -> float:
    (ax0, ay0), (ax1, ay1) = first.min(axis=0), first.max(axis=0)
    (bx0, by0), (bx1, by1) = second.min(axis=0), second.max(axis=0)
    common = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - common
    return float(common / union) if union > 0 else 0.0


def same_as_rider(rider: np.ndarray, other: np.ndarray) -> bool | None:
    """Persoana `other` (33 de puncte în pixeli) e călărețul ales? True / False / None = nesigur. Se compară
    mijlocul umerilor și al șoldurilor, în lungimi de trunchi: picioarele unei poze respinse sunt adesea greșite."""
    torso = np.linalg.norm(_midpoint(rider, "shoulder") - _midpoint(rider, "hip"))
    if torso == 0:
        return None
    gap = max(np.linalg.norm(_midpoint(rider, part) - _midpoint(other, part)) for part in ("shoulder", "hip")) / torso
    if gap <= SAME_RIDER_TORSOS:
        return True
    close = gap <= UNSURE_RIDER_TORSOS or _box_overlap(rider[BODY_POINTS], other[BODY_POINTS]) >= UNSURE_RIDER_IOU
    return None if close else False


def _mark_doubles(rows: dict) -> None:
    size = np.array(rows["size"])
    rows["rejected_is_rider"] = np.zeros(rows["rejected_source"].shape, ARRAYS["rejected_is_rider"])
    rows["rejected_unsure"] = np.zeros(rows["rejected_source"].shape, ARRAYS["rejected_unsure"])
    for i in np.where((rows["source"] != SOURCE_NONE) & (rows["rejected_count"] > 0))[0]:
        for k in range(rows["rejected_count"][i]):
            same = same_as_rider(rows["xy_norm"][i] * size, rows["rejected_xy_norm"][i, k] * size)
            rows["rejected_is_rider"][i, k], rows["rejected_unsure"][i, k] = same is True, same is None


def _add_rejected(rows: dict, people: list[list]) -> None:
    frames, capacity = len(people), max(map(len, people), default=0)
    for name, shape in (("rejected_xy_norm", (frames, capacity, N_POINTS, 2)),
                        ("rejected_z", (frames, capacity, N_POINTS)),
                        ("rejected_visibility", (frames, capacity, N_POINTS)),
                        ("rejected_presence", (frames, capacity, N_POINTS)),
                        ("rejected_world_xyz", (frames, capacity, N_POINTS, 3)),
                        ("rejected_crop_box", (frames, capacity, 4))):
        rows[name] = np.full(shape, np.nan, ARRAYS[name])
    rows["rejected_source"] = np.full((frames, capacity), SOURCE_NONE, ARRAYS["rejected_source"])
    rows["rejected_reason"] = np.full((frames, capacity), "", ARRAYS["rejected_reason"])
    rows["rejected_count"] = np.array([len(found) for found in people], ARRAYS["rejected_count"])
    for i, found in enumerate(people):
        for k, (pose, world, xy, source, reason, box, scale) in enumerate(found):
            rows["rejected_xy_norm"][i, k], rows["rejected_source"][i, k] = xy, source
            rows["rejected_z"][i, k] = [p.z * scale for p in pose]
            rows["rejected_visibility"][i, k] = [_optional(p.visibility) for p in pose]
            rows["rejected_presence"][i, k] = [_optional(p.presence) for p in pose]
            rows["rejected_world_xyz"][i, k] = [(p.x, p.y, p.z) for p in world]
            rows["rejected_reason"][i, k] = reason
            if box is not None:
                rows["rejected_crop_box"][i, k] = box


def _empty_rows(frames: int) -> dict:
    shapes = {"xy_norm": (frames, N_POINTS, 2), "z": (frames, N_POINTS), "visibility": (frames, N_POINTS),
              "presence": (frames, N_POINTS), "world_xyz": (frames, N_POINTS, 3), "crop_box": (frames, 4)}
    rows = {name: np.full(shape, np.nan, ARRAYS[name]) for name, shape in shapes.items()}
    rows["facing"] = np.full(frames, np.nan, ARRAYS["facing"])
    rows["source"] = np.full(frames, SOURCE_NONE, ARRAYS["source"])
    rows["n_poses"] = np.zeros(frames, ARRAYS["n_poses"])
    rows["reason"] = np.full(frames, "", ARRAYS["reason"])
    rows["side"] = np.full(frames, "", ARRAYS["side"])
    rows["frame"] = np.arange(frames, dtype=ARRAYS["frame"])
    return rows


def _fill(rows: dict, i: int, raw_pose, world_pose, source: int, scale: float, xy: np.ndarray, points: dict) -> None:
    rows["xy_norm"][i], rows["source"][i] = xy, source
    rows["z"][i] = [p.z * scale for p in raw_pose]
    rows["visibility"][i] = [_optional(p.visibility) for p in raw_pose]
    rows["presence"][i] = [_optional(p.presence) for p in raw_pose]
    rows["world_xyz"][i] = [(p.x, p.y, p.z) for p in world_pose]
    left, right = (LANDMARK_NAMES.index(f"{side}_hip") for side in ("left", "right"))
    rows["side"][i] = "left" if rows["visibility"][i, left] >= rows["visibility"][i, right] else "right"
    rows["facing"][i] = points["facing"]


def assemble(source, points_for_frame, log: Recording) -> tuple[dict, dict, dict]:
    """Din trecerea candidatului: rândurile scheletului, ieșirea pentru brut/ (ca predict_points_with_reasons)
    și numărătoarea identificării. `source` dă ce dă `_landmarks`; `log` e completat în timpul iterării."""
    frames, raw = [], {}
    size = (0, 0)
    for index, chosen, width, height, reason in source:
        size = (width, height)
        points = points_for_frame(chosen, width, height) if chosen else None
        if points is not None and (points["shoulder"] is None or points["hip"] is None):
            reason = REASON_HIDDEN
        raw[index] = {"points": points, "motiv": reason}
        frames.append((index, chosen, reason, points, log.single, list(log.crops), list(log.backs), width, height))
    rows = _empty_rows(len(frames))
    rows["size"] = size
    stats = {"riders": 0, "identified": 0, "unmatched": 0, "ambiguous": 0}
    people = []
    for index, chosen, reason, points, single, crops, backs, width, height in frames:
        rows["reason"][index] = reason or ""
        counts = [len(result.pose_landmarks) for result in ([single] if single else []) + crops]
        rows["n_poses"][index] = max(counts, default=0)
        if not chosen:
            people.append(_rejected_people(chosen, reason, single, crops, backs, None, False, width, height))
            continue
        stats["riders"] += 1
        xy = np.array([(p.x, p.y) for p in chosen])
        if single and single.pose_landmarks and chosen is single.pose_landmarks[0]:
            _fill(rows, index, chosen, single.pose_world_landmarks[0], SOURCE_FULL, 1.0, xy, points)
            stats["identified"] += 1
            people.append([])
            continue
        hits = _matches_on_crops(xy, np.array([p.visibility for p in chosen]), crops, width, height)
        if len(hits) != 1:
            rows["reason"][index] = REASON_UNIDENTIFIED
            stats["unmatched" if not hits else "ambiguous"] += 1
            people.append(_rejected_people(chosen, reason, single, crops, backs, None, True, width, height))
            continue
        raw_pose, world_pose, box = hits[0]
        _fill(rows, index, raw_pose, world_pose, SOURCE_CROP, (box[2] - box[0]) / width, xy, points)
        rows["crop_box"][index] = box
        stats["identified"] += 1
        people.append(_rejected_people(chosen, reason, single, crops, backs, raw_pose, False, width, height))
    _add_rejected(rows, people)
    _mark_doubles(rows)
    return rows, raw, stats


def make_skeleton(rows: dict, time_s: np.ndarray, meta: dict) -> RiderSkeleton:
    data = {name: rows[name] for name in ARRAYS if name != "time_s"}
    return RiderSkeleton(time_s=np.asarray(time_s, ARRAYS["time_s"]), meta=meta, **data)


# ---------- trecerea pe clip ----------

def _timestamps(video_path: Path, frames: int, fps: float) -> np.ndarray:
    """Timpul real al cadrelor, față de primul; cel nominal, dacă clipul nu dă câte un timp pe cadru."""
    cap = cv2.VideoCapture(str(video_path))
    times = []
    while cap.grab():
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    cap.release()
    if len(times) != frames:
        return np.arange(frames) / fps
    return np.array(times) - times[0]


@lru_cache(maxsize=4)
def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_pass(video_path: Path, candidate: str) -> PassResult:
    """O singură trecere a candidatului: dă ieșirea pentru brut/ și scheletul."""
    log = Recording()
    module = _recording_candidate(log)
    rows, raw, stats = assemble(module._landmarks(str(video_path)), module.points_for_frame, log)
    if not raw:
        raise ValueError(f"Clipul nu are cadre: {Path(video_path).name}")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    meta = {"schema_version": SCHEMA_VERSION, "video": Path(video_path).name, "video_hash": "", "candidate": candidate,
            "fps": float(fps), "width": rows["size"][0], "height": rows["size"][1], "n_frames": len(raw),
            "mediapipe_version": mediapipe.__version__, "pose_model": module.MODEL_VARIANT,
            "pose_model_sha256": _file_sha256(module.model_path()),
            "detector_sha256": _file_sha256(module.detector_path()),
            "identification": stats, "rejected_people": int(rows["rejected_count"].sum()),
            "rejected_doubles": int(rows["rejected_is_rider"].sum()),
            "rejected_unsure": int(rows["rejected_unsure"].sum())}
    return PassResult(raw, make_skeleton(rows, _timestamps(video_path, len(raw), fps), meta))


# ---------- citire, scriere, unghiul de exemplu ----------

def save_skeleton(path: Path, skeleton: RiderSkeleton) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    with partial.open("wb") as f:
        np.savez_compressed(f, meta=np.array(json.dumps(skeleton.meta)),
                            **{name: getattr(skeleton, name) for name in ARRAYS})
    partial.replace(path)


def load_skeleton(path: Path) -> RiderSkeleton:
    """Singura funcție de citire a formatului."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Fișierul de schelet are formatul {meta.get('schema_version')}, iar cel curent e "
                             f"{SCHEMA_VERSION}: nu se citește, trebuie regenerat.")
        return RiderSkeleton(meta=meta, **{name: data[name] for name in ARRAYS})


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Exemplu: unghiul din b, în grade, între b→a și b→c, pe cadre (2D sau 3D). NaN dacă lipsește un punct."""
    first, second = np.asarray(a, float) - b, np.asarray(c, float) - b
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (first * second).sum(axis=-1) / (np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
