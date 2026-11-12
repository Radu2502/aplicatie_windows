"""
Urmărirea multi-obiect: „detectezi pe toți, urmărești pe toți, apoi alegi”.

Pe fiecare cadru:
  1. YOLOX-S (ONNX, Apache-2.0) detectează toți oamenii și toți caii (clasele COCO person și horse);
  2. ByteTrack (MIT, portat aici, fără dependențe grele) dă ID-uri stabile, separat pentru oameni și pentru cai;
  3. RTMPose-m (ecvestru/rtmpose.py) dă scheletul fiecărui om urmărit;
  4. „călăreț” = omul cu șoldul pe caseta unui cal, cu regula `_on_horse` din candidat.
Click-ul alege ID-ul calului de sub click (aceeași pornire ca `tinta._start`) și omul de pe el; pe cadrele următoare
(și anterioare) se păstrează acele ID-uri. La pierderea unui ID: se așteaptă întoarcerea lui sau un ID nou de aceeași
clasă (apărut după pierdere, cel mai apropiat de ultima poziție), cel mult `tinta.MAX_GAP_SECONDS`; altfel ținta e
pierdută până la capăt, în acel sens (ca în `tinta`).

Detectorul și urmăritorul folosesc valorile lor implicite (constantele de mai jos, cu sursa).
Trecerea grea (detector + urmăritor + RTMPose) se salvează în data/results/urmarire_multi/; alegerea e rapidă.
Fără Streamlit.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ecvestru import stocare as ST
from ecvestru import tinta as T
from ecvestru.rezultat import TargetSeed

VERSION = 1      # se mărește când se schimbă trecerea grea (invalidează cache-ul)

# ---------- detectorul: YOLOX-S, valorile din demo/ONNXRuntime/onnx_inference.py (YOLOX 0.1.1rc0) ----------
DETECTOR_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
DETECTOR_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
DETECTOR_FILE = "yolox_s.onnx"
DETECTOR_DIR = ST.ROOT / "models" / "yolox"      # models/ e ignorat de git
DETECTOR_INPUT = (640, 640)       # înălțime, lățime (--input_shape implicit)
PAD_VALUE = 114.0                 # yolox.data.data_augment.preproc
NMS_IOU = 0.45                    # multiclass_nms(nms_thr=0.45, ...) din demo
SCORE_MIN = 0.1                   # multiclass_nms(..., score_thr=0.1) din demo
STRIDES = (8, 16, 32)             # demo_postprocess (fără p6)
PERSON, HORSE = 0, 17             # indicii COCO
CLASSES = (PERSON, HORSE)
CLASS_NAMES = {PERSON: "om", HORSE: "cal"}

# ---------- urmăritorul: ByteTrack, valorile implicite din demo_track.py și yolox/tracker/ (ByteTrack) ----------
TRACK_THRESH = 0.5                # --track_thresh
TRACK_BUFFER = 30                 # --track_buffer (cadre la 30 fps)
MATCH_THRESH = 0.8                # --match_thresh
LOW_SCORE = 0.1                   # byte_tracker.py: inds_low = scores > 0.1
SECOND_MATCH = 0.5                # a doua asociere (casetele cu scor mic)
UNCONFIRMED_MATCH = 0.7           # urmele neconfirmate
NEW_TRACK_MARGIN = 0.1            # det_thresh = track_thresh + 0.1
DUPLICATE_DISTANCE = 0.15         # remove_duplicate_stracks: iou_distance < 0.15
BASE_FPS = 30.0                   # buffer_size = int(frame_rate / 30 * track_buffer)
STD_POSITION, STD_VELOCITY = 1 / 20, 1 / 160    # kalman_filter.py
# În demo_track.py, după urmăritor, casetele cu lățime/înălțime > 1,6 sunt aruncate (pietoni în picioare); aici nu,
# pentru că ar arunca toți caii. Filtrul `min_box_area` (10 px²) e tot doar al afișării din demo; nu se aplică.

# ---------- stările unui cadru ----------
ON_C1, OTHER_RIDER, NO_SKELETON, LOST = "pe_c1", "alt_calaret", "fara_schelet", "pierdut"
CATEGORIES = (ON_C1, OTHER_RIDER, LOST, NO_SKELETON)
LEFT_HIP, RIGHT_HIP = 11, 12      # Halpe26 (= COCO) în ieșirea RTMPose
MEDIAPIPE_HIPS = (23, 24)         # indicii pe care îi citește `_on_horse`

CACHE_DIR = "urmarire_multi"
POSE_META = 4                     # meta: fps, cadre, secunde, CPU; apoi timpii celei de-a doua treceri
ANCHOR_VARIANT = "ancora"         # numele cache-ului cu ancora pe cal (urmat de cheia click-ului)


# =====================================================================================================================
# Detectorul
# =====================================================================================================================

def letterbox(image: np.ndarray, size: tuple[int, int] = DETECTOR_INPUT) -> tuple[np.ndarray, float]:
    """Ca `preproc` din YOLOX: micșorare cu raportul păstrat, colțul stânga-sus, restul 114; BGR, CHW, float32."""
    import cv2
    ratio = min(size[0] / image.shape[0], size[1] / image.shape[1])
    resized = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded = np.full((size[0], size[1], 3), PAD_VALUE, np.float32)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1))[None], ratio


def decode(output: np.ndarray, size: tuple[int, int] = DETECTOR_INPUT) -> np.ndarray:
    """`demo_postprocess`: ieșirea (N, 85) a rețelei → centru și mărime în pixelii intrării."""
    grids, strides = [], []
    for stride in STRIDES:
        h, w = size[0] // stride, size[1] // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((h * w, 1), stride))
    grid, stride = np.concatenate(grids), np.concatenate(strides)
    out = output.astype(np.float64).copy()
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> list[int]:
    """`nms` din yolox/utils/demo_utils.py (cu convenția +1 pixel)."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= iou)[0] + 1]
    return keep


def detections(predictions: np.ndarray, ratio: float) -> dict[int, np.ndarray]:
    """`multiclass_nms` implicit (fără clasă, `class_agnostic=True`): scor = obiect × clasă, apoi doar oamenii și
    caii. Rezultat: clasa → (k, 5) x0, y0, x1, y1, scor în pixelii cadrului."""
    cxcywh, scores = predictions[:, :4], predictions[:, 4:5] * predictions[:, 5:]
    boxes = np.column_stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                             cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2]) / ratio
    classes, best = scores.argmax(1), scores.max(1)
    valid = best > SCORE_MIN
    boxes, best, classes = boxes[valid], best[valid], classes[valid]
    keep = nms(boxes, best) if len(best) else []
    out = {}
    for cls in CLASSES:
        chosen = [k for k in keep if classes[k] == cls]
        out[cls] = np.column_stack([boxes[chosen], best[chosen]]) if chosen else np.zeros((0, 5))
    return out


def detector_path(folder: Path | None = None) -> Path:
    return (folder or DETECTOR_DIR) / DETECTOR_FILE


def download_detector(folder: Path | None = None) -> Path:
    """Descarcă YOLOX-S o singură dată și verifică SHA-256."""
    import hashlib
    import urllib.request
    path = detector_path(folder)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        urllib.request.urlretrieve(DETECTOR_URL, temporary)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DETECTOR_SHA256:
        raise ValueError(f"{path}: SHA-256 {digest}, aștepta {DETECTOR_SHA256}")
    return path


def session(model: Path, threads: int):
    """O sesiune onnxruntime pe CPU, cu `threads` fire și fără așteptare activă (firele nu se învârt în gol când
    rulează mai multe clipuri deodată). Nu schimbă rezultatul, doar împărțirea procesorului."""
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


class Detector:
    """YOLOX-S pe un cadru BGR → clasa → casete (k, 5)."""

    def __init__(self, model: Path | None = None, threads: int = 1):
        self.session = session(model or detector_path(), threads)
        self.input = self.session.get_inputs()[0].name

    def __call__(self, image: np.ndarray) -> dict[int, np.ndarray]:
        tensor, ratio = letterbox(image)
        output = self.session.run(None, {self.input: tensor})[0][0]
        return detections(decode(output), ratio)


# =====================================================================================================================
# Urmăritorul: ByteTrack (Zhang et al., 2022; github.com/ifzhang/ByteTrack, MIT), port fidel, mic
# =====================================================================================================================

TRACKED, LOST_TRACK, REMOVED = 1, 2, 3


class Kalman:
    """`yolox/tracker/kalman_filter.py`: starea x, y, a, h și vitezele; viteză constantă."""

    def __init__(self):
        self.motion = np.eye(8)
        for i in range(4):
            self.motion[i, 4 + i] = 1.0
        self.observe = np.eye(4, 8)

    @staticmethod
    def _std(h: float, position: float, velocity: float, aspect: float, aspect_velocity: float) -> np.ndarray:
        return np.square([position * h, position * h, aspect, position * h,
                          velocity * h, velocity * h, aspect_velocity, velocity * h])

    def initiate(self, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = xyah[3]
        return np.r_[xyah, np.zeros(4)], np.diag(self._std(h, 2 * STD_POSITION, 10 * STD_VELOCITY, 1e-2, 1e-5))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        noise = np.diag(self._std(mean[3], STD_POSITION, STD_VELOCITY, 1e-2, 1e-5))
        return self.motion @ mean, self.motion @ cov @ self.motion.T + noise

    def update(self, mean: np.ndarray, cov: np.ndarray, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        innovation_cov = np.diag(np.square([STD_POSITION * h, STD_POSITION * h, 1e-1, STD_POSITION * h]))
        projected_mean = self.observe @ mean
        projected_cov = self.observe @ cov @ self.observe.T + innovation_cov
        gain = np.linalg.solve(projected_cov, (cov @ self.observe.T).T).T
        return mean + (xyah - projected_mean) @ gain.T, cov - gain @ projected_cov @ gain.T


def tlbr_to_xyah(tlbr: np.ndarray) -> np.ndarray:
    w, h = tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]
    return np.array([tlbr[0] + w / 2, tlbr[1] + h / 2, w / h, h], float)


@dataclass(eq=False)
class Track:
    tlbr_start: np.ndarray
    score: float
    track_id: int = 0
    mean: np.ndarray | None = None
    cov: np.ndarray | None = None
    state: int = 0
    activated: bool = False
    frame_id: int = 0
    start_frame: int = 0

    @property
    def tlbr(self) -> np.ndarray:
        if self.mean is None:
            return self.tlbr_start.copy()
        x, y, a, h = self.mean[:4]
        w = a * h
        return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def iou_matrix(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    """IoU cu convenția +1 pixel (`cython_bbox.bbox_overlaps`, folosit de ByteTrack)."""
    out = np.zeros((len(a), len(b)))
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            iw = min(p[2], q[2]) - max(p[0], q[0]) + 1
            ih = min(p[3], q[3]) - max(p[1], q[1]) + 1
            if iw > 0 and ih > 0:
                union = (p[2] - p[0] + 1) * (p[3] - p[1] + 1) + (q[2] - q[0] + 1) * (q[3] - q[1] + 1) - iw * ih
                out[i, j] = iw * ih / union
    return out


def iou_distance(tracks: list[Track], dets: list[Track]) -> np.ndarray:
    return 1 - iou_matrix([t.tlbr for t in tracks], [d.tlbr for d in dets])


def fuse_score(cost: np.ndarray, dets: list[Track]) -> np.ndarray:
    if cost.size == 0:
        return cost
    return 1 - (1 - cost) * np.array([d.score for d in dets])[None, :]


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Ca `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)`: asignarea optimă în care a lăsa un rând sau o coloană
    nepotrivită costă thresh/2 (matricea extinsă a lui lapjv), rezolvată cu scipy."""
    from scipy.optimize import linear_sum_assignment
    n, m = cost.shape
    if cost.size == 0:
        return [], list(range(n)), list(range(m))
    big = thresh * 2 + 1.0
    extended = np.full((n + m, n + m), big)
    extended[:n, :m] = np.where(cost > thresh, big, cost)
    extended[:n, m:] = np.where(np.eye(n), thresh / 2, big)
    extended[n:, :m] = np.where(np.eye(m), thresh / 2, big)
    extended[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(extended)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if r < n and c < m and cost[r, c] <= thresh]
    matched_a, matched_b = {r for r, _ in matches}, {c for _, c in matches}
    return matches, [i for i in range(n) if i not in matched_a], [j for j in range(m) if j not in matched_b]


class ByteTracker:
    """`BYTETracker.update` din yolox/tracker/byte_tracker.py, cu numărătorul de ID-uri al fiecărui urmăritor.
    Intrare: casete (k, 5) x0, y0, x1, y1, scor, deja în pixelii cadrului. Ieșire: (ID, x0, y0, x1, y1, scor)."""

    def __init__(self, frame_rate: float = BASE_FPS):
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.removed: list[Track] = []
        self.frame_id = 0
        self.next_id = 0
        self.max_time_lost = int(frame_rate / BASE_FPS * TRACK_BUFFER)
        self.kalman = Kalman()

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def _predict(self, tracks: list[Track]) -> None:
        for t in tracks:
            mean = t.mean.copy()
            if t.state != TRACKED:
                mean[7] = 0
            t.mean, t.cov = self.kalman.predict(mean, t.cov)

    def _update(self, t: Track, det: Track) -> None:
        t.mean, t.cov = self.kalman.update(t.mean, t.cov, tlbr_to_xyah(det.tlbr))
        t.state, t.activated, t.frame_id, t.score = TRACKED, True, self.frame_id, det.score

    def update(self, dets: np.ndarray) -> list[tuple[int, float, float, float, float, float]]:
        self.frame_id += 1
        activated, refound, lost, removed = [], [], [], []
        scores = dets[:, 4] if len(dets) else np.zeros(0)
        high = [Track(d[:4].astype(float), float(d[4])) for d in dets[scores > TRACK_THRESH]]
        second = [Track(d[:4].astype(float), float(d[4]))
                  for d in dets[(scores > LOW_SCORE) & (scores < TRACK_THRESH)]]
        unconfirmed = [t for t in self.tracked if not t.activated]
        tracked = [t for t in self.tracked if t.activated]

        pool = _join(tracked, self.lost)
        self._predict(pool)
        matches, u_track, u_det = linear_assignment(fuse_score(iou_distance(pool, high), high), MATCH_THRESH)
        for i, j in matches:
            (activated if pool[i].state == TRACKED else refound).append(pool[i])
            self._update(pool[i], high[j])

        remaining = [pool[i] for i in u_track if pool[i].state == TRACKED]
        matches, u_remaining, _ = linear_assignment(iou_distance(remaining, second), SECOND_MATCH)
        for i, j in matches:
            (activated if remaining[i].state == TRACKED else refound).append(remaining[i])
            self._update(remaining[i], second[j])
        for i in u_remaining:
            if remaining[i].state != LOST_TRACK:
                remaining[i].state = LOST_TRACK
                lost.append(remaining[i])

        high = [high[j] for j in u_det]
        matches, u_unconfirmed, u_det = linear_assignment(fuse_score(iou_distance(unconfirmed, high), high),
                                                          UNCONFIRMED_MATCH)
        for i, j in matches:
            self._update(unconfirmed[i], high[j])
            activated.append(unconfirmed[i])
        for i in u_unconfirmed:
            unconfirmed[i].state = REMOVED
            removed.append(unconfirmed[i])

        for j in u_det:
            t = high[j]
            if t.score < TRACK_THRESH + NEW_TRACK_MARGIN:
                continue
            t.track_id = self._new_id()
            t.mean, t.cov = self.kalman.initiate(tlbr_to_xyah(t.tlbr_start))
            t.state, t.activated = TRACKED, self.frame_id == 1
            t.frame_id = t.start_frame = self.frame_id
            activated.append(t)
        for t in self.lost:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.state = REMOVED
                removed.append(t)

        self.tracked = _join(_join([t for t in self.tracked if t.state == TRACKED], activated), refound)
        self.lost = _sub(_sub(self.lost, self.tracked) + lost, self.removed)   # ca în original: removed de dinainte
        self.removed += removed
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)
        return [(t.track_id, *map(float, t.tlbr), t.score) for t in self.tracked if t.activated]


def _join(a: list[Track], b: list[Track]) -> list[Track]:
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen and not seen.add(t.track_id)]


def _sub(a: list[Track], b: list[Track]) -> list[Track]:
    gone = {t.track_id for t in b}
    return [t for t in a if t.track_id not in gone]


def _remove_duplicates(a: list[Track], b: list[Track]) -> tuple[list[Track], list[Track]]:
    distance = iou_distance(a, b)
    drop_a, drop_b = set(), set()
    for p, q in zip(*np.where(distance < DUPLICATE_DISTANCE), strict=True):
        if a[p].frame_id - a[p].start_frame > b[q].frame_id - b[q].start_frame:
            drop_b.add(q)
        else:
            drop_a.add(p)
    return [t for i, t in enumerate(a) if i not in drop_a], [t for i, t in enumerate(b) if i not in drop_b]


# =====================================================================================================================
# Trecerea grea pe clip: cadru → urme de cai și de oameni (cu scheletul RTMPose)
# =====================================================================================================================

@dataclass
class ClipTracks:
    """Pe fiecare cadru: caii (ID → x0, y0, x1, y1) și oamenii (ID → caseta, plus șoldul RTMPose și cele 26 de
    puncte). `seconds` / `cpu_seconds`: durata trecerii grele."""
    fps: float
    n_frames: int
    horses: list[dict[int, tuple]] = field(default_factory=list)
    people: list[dict[int, tuple]] = field(default_factory=list)
    keypoints: list[dict[int, np.ndarray]] = field(default_factory=list)
    seconds: float = 0.0
    cpu_seconds: float = 0.0
    pose_seconds: float = 0.0        # a doua trecere (RTMPose doar lângă calul țintă), inclusă în `seconds`
    pose_cpu_seconds: float = 0.0


def hip_mid(points: np.ndarray) -> tuple[float, float]:
    mid = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2
    return float(mid[0]), float(mid[1])


class Pose:
    """RTMPose-m (același model și aceiași pași ca `rtmpose.RtmPose`), cu sesiunea de mai sus."""

    def __init__(self, threads: int = 1):
        from ecvestru import rtmpose
        model = rtmpose.model_path()
        if model is None:
            raise FileNotFoundError("Modelul RTMPose nu e descărcat: uv run python -m ecvestru.rtmpose descarca")
        self.rtm, self.session = rtmpose, session(model, threads)
        self.input = self.session.get_inputs()[0].name

    def __call__(self, image: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, center_xy, scale = self.rtm.prepare(image, bbox)
        locs, scores = self.rtm.decode_simcc(*self.session.run(None, {self.input: tensor}))
        return self.rtm.to_image(locs, center_xy, scale), scores


def run_clip(video_path: Path, detector=None, pose=None, max_frames: int | None = None, threads: int = 1,
             pose_people: bool = True) -> ClipTracks:
    """Detectorul, urmăritorul (unul pentru oameni, unul pentru cai) și RTMPose-m pe fiecare om urmărit;
    cu `pose_people=False`, fără RTMPose (prima trecere a urmăririi cu ancora pe cal)."""
    import cv2
    cv2.setNumThreads(1)
    detector = detector or Detector(threads=threads)
    pose = pose or (Pose(threads=threads) if pose_people else None)
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or BASE_FPS
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = total if max_frames is None else min(total, max_frames)
    trackers = {cls: ByteTracker(fps) for cls in CLASSES}
    out = ClipTracks(fps=fps, n_frames=0)
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        while out.n_frames < limit:
            ok, image = capture.read()
            if not ok:
                break
            found = detector(image)
            horses = {t[0]: tuple(t[1:5]) for t in trackers[HORSE].update(found[HORSE])}
            people = {t[0]: tuple(t[1:5]) for t in trackers[PERSON].update(found[PERSON])}
            points = ({pid: pose(image, np.array(box))[0].astype(np.float32) for pid, box in people.items()}
                      if pose_people else {})
            out.horses.append(horses)
            out.people.append(people)
            out.keypoints.append(points)
            out.n_frames += 1
    finally:
        capture.release()
    out.seconds, out.cpu_seconds = time.perf_counter() - started, time.process_time() - cpu_started
    return out


def cache_path(name: str, video_hash: str, max_frames: int | None = None, variant: str = "") -> Path:
    part = ("" if max_frames is None else f"__primele{max_frames}") + (f"__{variant}" if variant else "")
    return ST.results_dir().parent / CACHE_DIR / f"{Path(name).stem}-{video_hash}__v{VERSION}{part}.npz"


def save_tracks(path: Path, tracks: ClipTracks) -> None:
    """Un rând pe (cadru, urmă): cadru, clasă, ID, caseta; pentru oameni și cele 26 de puncte RTMPose."""
    rows, points = [], []
    for frame in range(tracks.n_frames):
        for tid, box in tracks.horses[frame].items():
            rows.append((frame, HORSE, tid, *box))
            points.append(np.full((26, 2), np.nan, np.float32))
        for tid, box in tracks.people[frame].items():
            rows.append((frame, PERSON, tid, *box))
            points.append(tracks.keypoints[frame].get(tid, np.full((26, 2), np.nan, np.float32)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows=np.array(rows, np.float64).reshape(-1, 7),
                        points=np.array(points, np.float32).reshape(-1, 26, 2),
                        meta=np.array([tracks.fps, tracks.n_frames, tracks.seconds, tracks.cpu_seconds,
                                       tracks.pose_seconds, tracks.pose_cpu_seconds]))


def load_tracks(path: Path) -> ClipTracks:
    with np.load(path) as data:
        rows, points, meta = data["rows"], data["points"], data["meta"]
    out = ClipTracks(fps=float(meta[0]), n_frames=int(meta[1]), seconds=float(meta[2]), cpu_seconds=float(meta[3]),
                     pose_seconds=float(meta[4]) if len(meta) > POSE_META else 0.0,
                     pose_cpu_seconds=float(meta[5]) if len(meta) > POSE_META else 0.0)
    out.horses = [{} for _ in range(out.n_frames)]
    out.people = [{} for _ in range(out.n_frames)]
    out.keypoints = [{} for _ in range(out.n_frames)]
    for row, pts in zip(rows, points, strict=True):
        frame, cls, tid, box = int(row[0]), int(row[1]), int(row[2]), tuple(map(float, row[3:7]))
        if cls == HORSE:
            out.horses[frame][tid] = box
        else:
            out.people[frame][tid] = box
            if not np.isnan(pts).all():          # oamenii fără RTMPose (departe de calul țintă)
                out.keypoints[frame][tid] = pts
    return out


# =====================================================================================================================
# Alegerea: click → ID-ul calului și al omului de pe el, păstrate pe cadre (ușor, pe date în memorie)
# =====================================================================================================================

def to_xywh(box: tuple) -> T.Box:
    return box[0], box[1], box[2] - box[0], box[3] - box[1]


def center(box: tuple) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def on_horse(hip: tuple[float, float], horse_box: tuple, width: int = 1, height: int = 1) -> bool:
    """Regula `_on_horse` din research/candidate.py, apelată neschimbată: mijlocul șoldurilor dat ca două puncte
    MediaPipe identice (normalizate), caseta calului ca obiect cu origin_x, origin_y, width, height."""
    rule = ST.candidate_module()._on_horse
    lm = [SimpleNamespace(x=0.0, y=0.0) for _ in range(max(MEDIAPIPE_HIPS) + 1)]
    for index in MEDIAPIPE_HIPS:
        lm[index] = SimpleNamespace(x=hip[0] / width, y=hip[1] / height)
    x, y, w, h = to_xywh(horse_box)
    return bool(rule(lm, [SimpleNamespace(origin_x=x, origin_y=y, width=w, height=h)], width, height))


def seed_horse(horses: list[dict[int, tuple]], click: TargetSeed, fps: float) -> tuple[int, int]:
    """Cadrul și ID-ul calului de sub click: aceeași pornire ca `tinta._start` (cadrul cel mai apropiat cu cai, cel
    mult MAX_GAP_SECONDS, caseta cea mai apropiată de click, cel mult SEED_MAX_DISTANCE lățimi)."""
    boxes = {f: [to_xywh(b) for b in h.values()] for f, h in enumerate(horses) if h}
    frame, box = T._start(boxes, len(horses), click, T.max_gap_frames(fps))
    return frame, next(tid for tid, b in horses[frame].items() if to_xywh(b) == box)


def rider_on(horse_box: tuple, people: dict[int, tuple], keypoints: dict[int, np.ndarray]) -> int | None:
    """Omul cu șoldul pe caseta calului (`on_horse`); dacă sunt mai mulți, cel cu șoldul cel mai aproape de mijlocul
    marginii de sus a casetei."""
    top = ((horse_box[0] + horse_box[2]) / 2, horse_box[1])
    riders = [pid for pid in people if pid in keypoints and on_horse(hip_mid(keypoints[pid]), horse_box)]
    return min(riders, key=lambda p: math.dist(hip_mid(keypoints[p]), top), default=None)


@dataclass
class Follow:
    """ID-ul urmărit pe fiecare cadru (None = lipsă), starea definitivă „pierdut” și schimbările de ID."""
    ids: list[int | None]
    lost_from: int | None          # primul cadru (în sensul înainte) de la care ținta e pierdută definitiv
    switches: int


def follow(boxes: list[dict[int, tuple]], start: int, start_id: int, max_gap: int) -> Follow:
    """Un ID de la cadrul `start`, înainte și înapoi. Dacă lipsește: se așteaptă întoarcerea lui sau un ID nou
    (absent pe ultimul cadru unde ținta era văzută), cel mai apropiat de ultima ei poziție; după mai mult de `max_gap`
    cadre fără niciuna, ținta e pierdută până la capăt, în acel sens."""
    ids: list[int | None] = [None] * len(boxes)
    ids[start] = start_id
    switches, lost_from = 0, None
    for step in (1, -1):
        current, last_box, last_seen = start_id, boxes[start][start_id], start
        before = set(boxes[start])
        frame = start + step
        while 0 <= frame < len(boxes):
            here = boxes[frame]
            if current in here:
                ids[frame], last_box, last_seen, before = current, here[current], frame, set(here)
            else:
                new = [tid for tid in here if tid not in before]
                if new:
                    current = min(new, key=lambda t: math.dist(center(here[t]), center(last_box)))
                    ids[frame], last_box, last_seen, before = current, here[current], frame, set(here)
                    switches += 1
                elif abs(frame - last_seen) > max_gap:
                    if step == 1:
                        lost_from = frame
                    break
            frame += step
    return Follow(ids, lost_from, switches)


@dataclass
class Choice:
    """Rezultatul alegerii pe un clip: categoria fiecărui cadru, ID-urile țintei și schimbările lor."""
    categories: list[str]
    horse_ids: list[int | None]
    rider_ids: list[int | None]
    seed_frame: int
    lost_from: int | None
    horse_switches: int
    rider_switches: int


def choose_riders(tracks: ClipTracks, horse_ids: list[int | None], gap: int) -> tuple[list[int | None], int]:
    """Omul țintă pe cadre: de la primul cadru (de la începutul clipului) cu un om pe calul țintă, ID-ul lui se
    păstrează cu `follow` (doar înainte) până se pierde; apoi se alege din nou omul de pe calul țintă."""
    rider_ids: list[int | None] = [None] * tracks.n_frames
    switches, frame = 0, 0
    while frame < tracks.n_frames:
        hid = horse_ids[frame]
        pid = None if hid is None else rider_on(tracks.horses[frame][hid], tracks.people[frame],
                                                tracks.keypoints[frame])
        if pid is None:
            frame += 1
            continue
        segment = follow(tracks.people[frame:], 0, pid, gap)
        end = frame + (segment.lost_from if segment.lost_from is not None else len(segment.ids))
        rider_ids[frame:end] = segment.ids[:end - frame]
        switches += segment.switches
        frame = end
    return rider_ids, switches


def choose(tracks: ClipTracks, click: TargetSeed, recover: bool = False, rider_by_horse: bool = False) -> Choice:
    """Click → calul și omul de pe el, urmăriți; categoria fiecărui cadru, ca în măsurarea veche:
    pierdut (fără caseta calului țintă) → fără schelet (fără omul țintă) → alt călăreț (șoldul nu e pe caseta
    țintei) → pe Călărețul 1. Cu `recover` și `rider_by_horse` (ancora pe cal): calul cu `follow_anchor`
    (recuperare prin eliminare, fără pierdere definitivă; `horse_switches` = recuperările) și, pe fiecare cadru,
    omul de pe calul țintă (`rider_on`), fără ID de om. Separat, fiecare arată contribuția pasului lui."""
    gap = T.max_gap_frames(tracks.fps)
    seed_frame, horse_id = seed_horse(tracks.horses, click, tracks.fps)
    horse = follow_anchor(tracks.horses, seed_frame, horse_id) if recover else follow(tracks.horses, seed_frame,
                                                                                    horse_id, gap)
    if rider_by_horse:
        rider_ids, rider_switches = riders_on_target(tracks, horse.ids), 0
    else:
        rider_ids, rider_switches = choose_riders(tracks, horse.ids, gap)
    categories = []
    for f in range(tracks.n_frames):
        hid, pid = horse.ids[f], rider_ids[f]
        if hid is None:
            categories.append(LOST)
        elif pid is None or pid not in tracks.people[f]:
            categories.append(NO_SKELETON)
        elif on_horse(hip_mid(tracks.keypoints[f][pid]), tracks.horses[f][hid]):
            categories.append(ON_C1)
        else:
            categories.append(OTHER_RIDER)
    return Choice(categories, horse.ids, rider_ids, seed_frame, horse.lost_from, horse.switches, rider_switches)


def follow_anchor(boxes: list[dict[int, tuple]], start: int, start_id: int) -> Follow:
    """Ancora = calul: un ID de cal de la `start`, înainte și înapoi, fără pierdere definitivă. Se ține
    evidența tuturor ID-urilor de cai văzute (în sensul parcurgerii): cât ținta e văzută, celelalte sunt „alți cai”.
    Când ținta lipsește, primul ID încă nevăzut care apare devine ținta (prin eliminare nu e unul dintre ceilalți cai);
    la mai mulți deodată, cel mai apropiat de ultima poziție a țintei. Cadrele dintre ele rămân fără țintă."""
    ids: list[int | None] = [None] * len(boxes)
    ids[start] = start_id
    recoveries = 0
    for step in (1, -1):
        current, last_box, seen = start_id, boxes[start][start_id], set(boxes[start])
        frame = start + step
        while 0 <= frame < len(boxes):
            here = boxes[frame]
            if current not in here:
                new = [tid for tid in here if tid not in seen]
                if new:
                    current = min(new, key=lambda t: math.dist(center(here[t]), center(last_box)))
                    recoveries += 1
            if current in here:
                ids[frame], last_box = current, here[current]
            seen |= set(here)
            frame += step
    return Follow(ids, None, recoveries)


def riders_on_target(tracks: ClipTracks, horse_ids: list[int | None]) -> list[int | None]:
    """Pe fiecare cadru, omul de pe calul țintă (`rider_on`, aceeași departajare); ID-ul omului nu contează."""
    return [None if hid is None else rider_on(tracks.horses[f][hid], tracks.people[f], tracks.keypoints[f])
            for f, hid in enumerate(horse_ids)]


def overlaps(a: tuple, b: tuple) -> bool:
    """Două casete x0, y0, x1, y1 au o arie comună (orice arie: fără prag)."""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def run_target_poses(video_path: Path, tracks: ClipTracks, horse_ids: list[int | None], pose=None,
                     threads: int = 1) -> ClipTracks:
    """A doua trecere a urmăririi cu ancora pe cal: RTMPose-m doar pe oamenii a căror casetă se suprapune cu caseta
    calului țintă."""
    import cv2
    cv2.setNumThreads(1)
    pose = pose or Pose(threads=threads)
    keypoints: list[dict[int, np.ndarray]] = [{} for _ in range(tracks.n_frames)]
    capture = cv2.VideoCapture(str(video_path))
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        for frame in range(tracks.n_frames):
            ok, image = capture.read()
            if not ok:
                break
            hid = horse_ids[frame]
            if hid is None:
                continue
            horse_box = tracks.horses[frame][hid]
            keypoints[frame] = {pid: pose(image, np.array(box))[0].astype(np.float32)
                                for pid, box in tracks.people[frame].items() if overlaps(box, horse_box)}
    finally:
        capture.release()
    seconds, cpu = time.perf_counter() - started, time.process_time() - cpu_started
    return ClipTracks(fps=tracks.fps, n_frames=tracks.n_frames, horses=tracks.horses, people=tracks.people,
                      keypoints=keypoints, seconds=tracks.seconds + seconds, cpu_seconds=tracks.cpu_seconds + cpu,
                      pose_seconds=seconds, pose_cpu_seconds=cpu)


def run_anchor(video_path: Path, click: TargetSeed, threads: int = 1, max_frames: int | None = None) -> ClipTracks:
    """Urmărirea cu ancora pe cal, pe un clip: detector + urmăritor (fără RTMPose), calul țintă cu `follow_anchor`,
    apoi RTMPose doar lângă el."""
    first = run_clip(video_path, max_frames=max_frames, threads=threads, pose_people=False)
    seed_frame, horse_id = seed_horse(first.horses, click, first.fps)
    return run_target_poses(video_path, first, follow_anchor(first.horses, seed_frame, horse_id).ids,
                            threads=threads)


def shares(categories: list[str]) -> dict[str, float]:
    n = len(categories) or 1
    return {c: 100 * categories.count(c) / n for c in CATEGORIES}
