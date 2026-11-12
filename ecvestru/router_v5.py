"""
Vederea călărețului cu routerul V5 (opțiunea „Vederea călărețului (router V5, experimental)” din „Analiză”, oprită
implicit): doar afișare. Nu schimbă selecția cadrelor, unghiurile, abținerea sau raportul; intervalul de profil ales de
utilizator rămâne cel care decide.

Modelul (`router_v5_model.json`, lângă acest fișier, fără pickle): regresia logistică multinomială cu ponderi
echilibrate pe 16 semnale ale țintei urmărite, apoi netezirea în timp cu un HMM (Viterbi). Experimental: vederea poate
fi greșită, mai ales între profil și oblic.

Semnalele, pe fiecare cadru al clipului (NaN fără țintă):
  - 8 semnale din scheletul RTMPose al călărețului urmărit și din AP-10K pe calul țintă
    (`semnale_tinta.target_signals`);
  - fața călărețului (scorurile RTMPose ale nasului și ochilor; vizibilitatea MediaPipe, doar când persoana MediaPipe e
    ținta), umeri / trunchi, capul și coada calului (AP-10K), mișcarea calului față de fundal pe ~1 s.
Ținta e cea a urmăririi (click-ul dat, cel salvat sau ținta automată); fără țintă, vederea lipsește
(`ViewsUnavailable`). Calculul greu se face o dată pe clip și se ține în cache (trecerea urmăririi, AP-10K pe calul
țintă, trecerea suplimentară a acestui modul). Fără Streamlit.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ecvestru import analiza_urmarita as AU
from ecvestru import cal_puncte as CP
from ecvestru import progres as P
from ecvestru import router_vedere as RV
from ecvestru import schelet as Z
from ecvestru import schelet_cal as SC
from ecvestru import semnale_tinta as SEM
from ecvestru import stocare as ST
from ecvestru import tinta as T
from ecvestru import urmarire_multi as U
from ecvestru.rezultat import RIDER_VIEW_SCHEMA_VERSION, AnalysisResult, RiderView, RiderViews, TargetSeed

VERSION = 1                          # pașii acestui modul (cache-ul trecerii suplimentare și câmpul din JSON)
APP_DEFAULT_RIDER_VIEW = False       # opțiunea e oprită implicit în aplicație
MODEL_FILE = Path(__file__).with_name("router_v5_model.json")
CACHE_FOLDER = "router_v5"
VIEWS = ("profil", "fata", "spate", "oblic", "nu_se_vede")

NEW_SIGNALS = ("fata_rtm", "fata_mp", "umeri_trunchi", "cal_cap", "cal_coada", "miscare_laterala", "miscare_abs",
               "miscare_scala")
ALL_SIGNALS = (*RV.SIGNALS, *NEW_SIGNALS)
HALPE_FACE = (0, 1, 2)                     # nas, ochiul stâng, ochiul drept
MEDIAPIPE_FACE = (0, 2, 5)
HALPE_SHOULDERS, HALPE_HIPS = (5, 6), (11, 12)
MEDIAPIPE_SHOULDERS, MEDIAPIPE_HIPS = (11, 12), (23, 24)
HORSE_HEAD = tuple(CP.AP10K.index(n) for n in ("nose", "left_eye", "right_eye"))
HORSE_TAIL = CP.AP10K.index("root_of_tail")
N_HALPE = 26
WINDOW_SECONDS = 1.0                       # fereastra mișcării calului față de fundal
MIN_FLOW_SHARE = 0.8                       # cât din fereastră trebuie să aibă mișcarea fundalului estimată
MIN_LENGTH_PX = 1.0
# mișcarea fundalului: colțuri în afara casetelor (cai și oameni, lărgite), urmărite cu Lucas–Kanade, mediana
MAX_CORNERS, CORNER_QUALITY, CORNER_DISTANCE_PX = 300, 0.01, 8
MIN_POINTS = 20
BOX_MARGIN = 0.1
HMM_EPS = 1e-9


class ViewsUnavailable(ValueError):
    """Vederea V5 nu se poate calcula pe acest clip (de exemplu, fără țintă); mesajul spune de ce."""


# ---------- modelul ----------

@dataclass(frozen=True)
class Model:
    classes: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray            # (2 · semnale + 1, clase): semnalele standardizate, „lipsește”, termenul liber
    stay: float                    # HMM: probabilitatea de a rămâne în aceeași vedere de la un cadru la altul
    sha: str                       # primele 8 caractere din sha256-ul fișierului
    info: dict

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        missing = np.isnan(X)
        z = np.where(missing, 0.0, (X - self.mean) / self.std)
        logits = np.hstack([z, missing.astype(float), np.ones((len(X), 1))]) @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


def load_model(path: Path = MODEL_FILE) -> Model:
    data = path.read_bytes()
    m = json.loads(data.decode("utf-8"))
    if tuple(m["semnale"]) != ALL_SIGNALS:
        raise ValueError("Modelul V5 are alte semnale decât acest cod.")
    return Model(classes=tuple(m["clase"]), mean=np.array(m["medie"], float), std=np.array(m["abatere"], float),
                 weights=np.array(m["ponderi"], float), stay=float(m["hmm_ramane"]),
                 sha=hashlib.sha256(data).hexdigest()[:8],
                 info={k: m[k] for k in ("versiune", "antrenare", "evaluare") if k in m})


def balanced_accuracy(model: Model) -> float | None:
    """Acuratețea echilibrată a modelului, dacă e scrisă în el (altfel None)."""
    return (model.info.get("evaluare") or {}).get("echilibrata")


# ---------- netezirea ----------

def hmm(proba: np.ndarray, classes: tuple[str, ...], stay: float) -> np.ndarray:
    """Calea Viterbi; rândurile NaN (fără semnale) au emisie uniformă."""
    p = np.asarray(proba, float)
    n, k = p.shape
    emit = np.where(np.isnan(p).any(axis=1, keepdims=True), 1.0 / k, p)
    log_e = np.log(emit + HMM_EPS)
    move = np.full((k, k), (1.0 - stay) / max(k - 1, 1))
    np.fill_diagonal(move, stay)
    log_t = np.log(move)
    score = log_e[0] - math.log(k)
    back = np.zeros((n, k), int)
    for i in range(1, n):
        cand = score[:, None] + log_t
        back[i] = cand.argmax(axis=0)
        score = cand.max(axis=0) + log_e[i]
    path = np.zeros(n, int)
    path[-1] = int(score.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return np.array(classes, dtype=object)[path]


def predict(model: Model, X: np.ndarray) -> np.ndarray:
    """Vederea pe fiecare cadru; None pe cadrele fără semnale (fără țintă)."""
    X = np.asarray(X, float)
    usable = ~np.isnan(X).all(axis=1)
    out = np.full(len(X), None, dtype=object)
    if not usable.any():
        return out
    proba = np.full((len(X), len(model.classes)), np.nan)
    proba[usable] = model.predict_proba(X[usable])
    views = hmm(proba, model.classes, model.stay)
    out[usable] = views[usable]
    return out


# ---------- semnalele noi ----------

def face_rtm(scores: np.ndarray) -> np.ndarray:
    return np.asarray(scores, float)[:, HALPE_FACE].mean(axis=1)


def shoulders_over_torso(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, float)
    width = np.linalg.norm(p[:, HALPE_SHOULDERS[0]] - p[:, HALPE_SHOULDERS[1]], axis=1)
    height = np.abs(p[:, HALPE_SHOULDERS, 1].mean(axis=1) - p[:, HALPE_HIPS, 1].mean(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(height >= MIN_LENGTH_PX, width / height, np.nan)


def same_person(rtm_points: np.ndarray, mp_xy: np.ndarray, limit: float = Z.SAME_RIDER_TORSOS) -> np.ndarray:
    rtm_hips = np.asarray(rtm_points, float)[:, HALPE_HIPS].mean(axis=1)
    mp = np.asarray(mp_xy, float)
    mp_hips, mp_shoulders = mp[:, MEDIAPIPE_HIPS].mean(axis=1), mp[:, MEDIAPIPE_SHOULDERS].mean(axis=1)
    torso = np.linalg.norm(mp_shoulders - mp_hips, axis=1)
    distance = np.linalg.norm(rtm_hips - mp_hips, axis=1)
    with np.errstate(invalid="ignore"):
        return (torso >= MIN_LENGTH_PX) & (distance <= limit * torso)


def horse_head_tail(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, float)
    head = np.where(np.isnan(s[:, HORSE_HEAD]).all(axis=1), np.nan,
                    np.nanmax(np.where(np.isnan(s[:, HORSE_HEAD]), -np.inf, s[:, HORSE_HEAD]), axis=1))
    return head, s[:, HORSE_TAIL]


def motion(boxes: np.ndarray, flow: np.ndarray, fps: float, seconds: float = WINDOW_SECONDS,
           min_flow_share: float = MIN_FLOW_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """Mișcarea calului față de fundal pe [f − h, f + h]: laterală (pe lățimea medie a casetei) și scala (log)."""
    b, f = np.asarray(boxes, float), np.asarray(flow, float)
    n = len(b)
    half = max(int(round(seconds * fps / 2)), 1)
    centre, width, height = (b[:, 0] + b[:, 2]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    lateral, scale = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(half, n - half):
        a, z = i - half, i + half
        if np.isnan(centre[a]) or np.isnan(centre[z]) or height[a] <= 0 or height[z] <= 0:
            continue
        steps = f[a + 1: z + 1, 0]
        known = ~np.isnan(steps)
        if known.sum() < min_flow_share * len(steps):
            continue
        camera = steps[known].mean() * len(steps)
        mean_width = np.nanmean(width[a: z + 1])
        if not mean_width > 0:
            continue
        lateral[i] = (centre[z] - centre[a] - camera) / mean_width
        scale[i] = np.log(height[z] / height[a])
    return lateral, scale


def new_signals(rtm_scores: np.ndarray, rtm_points: np.ndarray, mp_xy: np.ndarray | None,
                mp_visibility: np.ndarray | None, horse_scores: np.ndarray, horse_boxes: np.ndarray,
                flow: np.ndarray, fps: float, known: np.ndarray) -> np.ndarray:
    n = len(known)
    out = np.full((n, len(NEW_SIGNALS)), np.nan)
    col = {name: i for i, name in enumerate(NEW_SIGNALS)}
    out[:, col["fata_rtm"]] = face_rtm(rtm_scores)
    if mp_xy is not None and mp_visibility is not None:
        m = min(n, len(mp_xy))
        same = np.zeros(n, bool)
        same[:m] = same_person(rtm_points[:m], mp_xy[:m])
        vis = np.full((n, mp_visibility.shape[1]), np.nan)
        vis[:m] = mp_visibility[:m]
        out[:, col["fata_mp"]] = np.where(same, vis[:, MEDIAPIPE_FACE].mean(axis=1), np.nan)
    out[:, col["umeri_trunchi"]] = shoulders_over_torso(rtm_points)
    head, tail = horse_head_tail(horse_scores[:n])
    out[:, col["cal_cap"]], out[:, col["cal_coada"]] = head, tail
    lateral, scale = motion(horse_boxes[:n], flow[:n], fps)
    out[:, col["miscare_laterala"]], out[:, col["miscare_abs"]] = lateral, np.abs(lateral)
    out[:, col["miscare_scala"]] = scale
    out[~np.asarray(known, bool)] = np.nan
    return out


def rider_rtm_points(tracks: U.ClipTracks, rider_ids: list) -> np.ndarray:
    out = np.full((tracks.n_frames, N_HALPE, 2), np.nan)
    for f, pid in enumerate(rider_ids):
        if pid is not None and pid in tracks.keypoints[f]:
            out[f] = tracks.keypoints[f][pid]
    return out


# ---------- trecerea suplimentară: mișcarea fundalului și scorurile RTMPose ale țintei ----------

@dataclass
class Extra:
    flow: np.ndarray           # (N, 2) deplasarea fundalului de la cadrul precedent (px); NaN fără estimare
    rtm_scores: np.ndarray     # (N, 26) scorurile RTMPose ale călărețului țintă; NaN fără țintă
    seconds: float


def background_mask(shape: tuple[int, int], boxes: list[tuple], margin: float = BOX_MARGIN) -> np.ndarray:
    mask = np.full(shape, 255, np.uint8)
    h, w = shape
    for x0, y0, x1, y1 in boxes:
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        a, b = max(0, int(x0 - mx)), min(w, int(np.ceil(x1 + mx)))
        c, d = max(0, int(y0 - my)), min(h, int(np.ceil(y1 + my)))
        if a < b and c < d:
            mask[c:d, a:b] = 0
    return mask


def median_shift(start: np.ndarray, end: np.ndarray, found: np.ndarray, min_points: int = MIN_POINTS) -> np.ndarray:
    found = np.asarray(found, bool).ravel()
    moves = (np.asarray(end, float) - np.asarray(start, float)).reshape(-1, 2)[found]
    return np.full(2, np.nan) if len(moves) < min_points else np.median(moves, axis=0)


def background_flow(previous: np.ndarray, current: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(previous, MAX_CORNERS, CORNER_QUALITY, CORNER_DISTANCE_PX,
                                      mask=background_mask(previous.shape, boxes))
    if corners is None:
        return np.full(2, np.nan)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
    return median_shift(corners, moved, status.ravel() == 1)


def extra_pass(video_path: Path, tracks: U.ClipTracks, rider_ids: list, pose=None) -> Extra:
    pose = pose or U.Pose(threads=1)
    n = tracks.n_frames
    flow, scores = np.full((n, 2), np.nan), np.full((n, N_HALPE), np.nan)
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    previous = None
    try:
        for frame in range(n):
            ok, image = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                boxes = list(tracks.horses[frame - 1].values()) + list(tracks.people[frame - 1].values())
                flow[frame] = background_flow(previous, gray, boxes)
            pid = rider_ids[frame]
            if pid is not None and pid in tracks.people[frame]:
                scores[frame] = pose(image, np.array(tracks.people[frame][pid]))[1]
            previous = gray
    finally:
        capture.release()
    return Extra(flow, scores, time.perf_counter() - started)


def extra_cache_path(name: str, video_hash: str, seed: TargetSeed) -> Path:
    return ST.results_dir() / CACHE_FOLDER / f"{ST.clip_key(name, video_hash)}__t{T.seed_key(seed)}__x{VERSION}.npz"


def load_extra(video_path: Path, name: str, video_hash: str, seed: TargetSeed, tracks: U.ClipTracks,
               rider_ids: list, pose=None) -> Extra:
    path = extra_cache_path(name, video_hash, seed)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return Extra(data["flow"], data["rtm_scores"], float(data["seconds"]))
    extra = extra_pass(video_path, tracks, rider_ids, pose)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, flow=extra.flow, rtm_scores=extra.rtm_scores, seconds=extra.seconds)
    return extra


# ---------- vederea pe clip și în rezultat ----------

def signals(tracks: U.ClipTracks, choice: U.Choice, horse: CP.ClipRun, extra: Extra, skeleton) -> np.ndarray:
    """(N, 16) în ordinea `ALL_SIGNALS`; NaN pe cadrele fără țintă."""
    base = SEM.target_signals(tracks, choice, horse).values
    known = ~np.isnan(base).all(axis=1)
    n = tracks.n_frames
    mp_xy = mp_vis = None
    if skeleton is not None:
        mp_xy, mp_vis = np.full((n, 33, 2), np.nan), np.full((n, 33), np.nan)
        frames = np.asarray(skeleton.frame, int)
        ok = frames < n
        mp_xy[frames[ok]], mp_vis[frames[ok]] = np.asarray(skeleton.xy_px)[ok], np.asarray(skeleton.visibility)[ok]
    rtm_points = rider_rtm_points(tracks, choice.rider_ids)
    new = new_signals(extra.rtm_scores, rtm_points, mp_xy, mp_vis, horse.scores, horse.boxes, extra.flow, tracks.fps,
                      known)
    return np.hstack([base, new])


def clip_views(video_path: Path, name: str, video_hash: str, click: TargetSeed | None = None,
               stages: P.Stages | None = None, model: Model | None = None, pose=None,
               horse_estimator=None) -> tuple[np.ndarray, TargetSeed]:
    """(vederea pe fiecare cadru al clipului, ținta). ViewsUnavailable fără țintă."""
    video_path = Path(video_path)
    stages = stages or P.Stages()
    model = model or load_model()
    tracks, _ = AU.heavy_pass(video_path, name, video_hash, stages)
    try:
        seed, _ = AU.resolve_target(tracks, video_hash, click)
    except AU.TargetNeeded as error:
        raise ViewsUnavailable(f"Vederea V5 are nevoie de ținta urmăririi: {error}") from error
    with stages.stage(P.RIDER_VIEW):
        choice = AU.choose(video_path, tracks, seed)
        horse = SC.tracked_points(video_path, name, video_hash, AU.CANDIDATE_LABEL, seed,
                                  lambda: AU.target_horse_boxes(tracks, choice), horse_estimator)
        extra = load_extra(video_path, name, video_hash, seed, tracks, choice.rider_ids, pose)
        skeleton = ST.load_rider_skeleton(video_path, name, video_hash, ST.candidate_label())
        views = predict(model, signals(tracks, choice, horse, extra, skeleton))
    return views, seed


def attach(result: AnalysisResult, views: np.ndarray, model: Model) -> AnalysisResult:
    """Rezultatul cu vederea pe cadrele lui (schema 1.7); restul câmpurilor rămâne neschimbat."""
    frames = [RiderView(frame=f.frame, view=views[f.frame] if f.frame < len(views) else None) for f in result.frames]
    field = RiderViews(version=VERSION, model=model.sha, balanced_accuracy=balanced_accuracy(model), frames=frames)
    return result.model_copy(update={"schema_version": RIDER_VIEW_SCHEMA_VERSION, "rider_views": field})


def with_views(video_path: Path, result: AnalysisResult, click: TargetSeed | None = None,
               stages: P.Stages | None = None, model: Model | None = None, pose=None,
               horse_estimator=None) -> AnalysisResult:
    """Rezultatul (al oricărei metode) cu vederea V5 pe cadrele analizate. Ținta: click-ul dat, ținta rezultatului
    (urmărirea), cea salvată sau cea automată."""
    model = model or load_model()
    seed = click or (result.target if result.candidate == AU.CANDIDATE_LABEL else None)
    views, _ = clip_views(video_path, result.video, result.video_hash, seed, stages, model, pose, horse_estimator)
    return attach(result, views, model)
