"""
Routerul de vedere pe ținta urmărită: aceleași semnale, același clasificator și același vot ca în
ecvestru/router_vedere.py, dar calculate pe călărețul ales de urmărire (aspect și schimb de ID, k = 1 s:
ecvestru/urmarire_multi.py + ecvestru/aspect_urmarire.py), nu pe persoana candidatului. Modul pe candidat
(`router_vedere.load_clip_signals`) rămâne neschimbat.

- Călărețul: cele 26 de puncte RTMPose-m (din urme); semnalele de umeri și coate cu aceleași formule
  (`router_vedere.rider_signals`), cu punctele Halpe mutate pe indicii MediaPipe. Orientarea 3D cere punctele world ale
  MediaPipe, pe care RTMPose nu le dă: rămâne NaN (clasificatorul are un indicator „lipsește” pentru fiecare semnal).
- Calul: AP-10K (ecvestru/cal_puncte.py) pe caseta calului țintă, apoi `router_vedere.horse_signals`.
- Cadrele fără schelet al țintei (calul pierdut sau nimeni pe el) sunt „necunoscut”: nu votează, nu merg nicăieri și se
  raportează separat.
- Evaluarea: validare încrucișată „un clip lăsat deoparte” pe clipurile etichetate.

Oprit implicit (ROUTER_PILOT_DEFAULT): analiza implicită nu se schimbă; în „Analiză” doar se afișează vederea, dacă e
bifat. Fără Streamlit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ecvestru import aspect_urmarire as AS
from ecvestru import cal_puncte as CP
from ecvestru import evaluare_router as ER
from ecvestru import intervale as IV
from ecvestru import router_vedere as RV
from ecvestru import stocare as ST
from ecvestru import urmarire_multi as U

ROUTER_PILOT_DEFAULT = False
VERSION = 1
TRACKING = "aspect + schimb de ID (k = 1 s)"
UNKNOWN = "necunoscut"
PILOT_DIR = "router_pilot"
HORSE_VARIANT = "ap10k_tinta"
# Halpe26 (RTMPose) → indicii MediaPipe folosiți de router_vedere.rider_signals
HALPE_TO_MEDIAPIPE = {RV.LEFT_SHOULDER: 5, RV.RIGHT_SHOULDER: 6, RV.LEFT_ELBOW: 7, RV.RIGHT_ELBOW: 8,
                      RV.LEFT_HIP: U.LEFT_HIP, RV.RIGHT_HIP: U.RIGHT_HIP}
MEDIAPIPE_POINTS = 33
TRUTH_VIEWS = (IV.PROFILE, IV.FRONT, IV.BACK, IV.OBLIQUE, IV.HIDDEN)
ROUTER_VIEWS = (*TRUTH_VIEWS, RV.UNDECIDED, UNKNOWN)


# ---------- ținta urmărită ----------

def target_choice(tracks: U.ClipTracks, scores: AS.Scores, seed_frame: int, horse_id: int) -> U.Choice:
    """Urmărirea cu aspect și schimb de ID, cu k = AS.SWAP_SECONDS (1 s)."""
    return AS.choose(tracks, seed_frame, horse_id, scores, swap_frames=round(AS.SWAP_SECONDS * tracks.fps))


def known_frames(choice: U.Choice) -> np.ndarray:
    """Cadrele cu un schelet al țintei (un om pe calul țintă)."""
    return np.array([r is not None for r in choice.rider_ids], bool)


# ---------- semnalele ----------

def rider_xy(tracks: U.ClipTracks, choice: U.Choice) -> np.ndarray:
    """(N, 33, 2) în indicii MediaPipe, cu umerii, coatele și șoldurile din RTMPose; NaN în rest și fără țintă."""
    xy = np.full((tracks.n_frames, MEDIAPIPE_POINTS, 2), np.nan)
    for frame, pid in enumerate(choice.rider_ids):
        if pid is not None:
            points = tracks.keypoints[frame][pid]
            for mp_index, halpe_index in HALPE_TO_MEDIAPIPE.items():
                xy[frame, mp_index] = points[halpe_index]
    return xy


def target_boxes(tracks: U.ClipTracks, horse_ids: list[int | None]) -> np.ndarray:
    """(N, 4) caseta calului țintă, x0, y0, x1, y1; NaN pe cadrele fără el."""
    boxes = np.full((tracks.n_frames, 4), np.nan)
    for frame, hid in enumerate(horse_ids):
        if hid is not None:
            boxes[frame] = tracks.horses[frame][hid]
    return boxes


def run_target_horse(video_path: Path, boxes: np.ndarray, estimator=None) -> CP.ClipRun:
    """AP-10K pe caseta calului țintă, pe fiecare cadru cu ea (citire secvențială, indicii OpenCV)."""
    import cv2
    estimator = estimator or CP.HorsePose()
    n = len(boxes)
    points, scores = np.full((n, len(CP.AP10K), 2), np.nan), np.full((n, len(CP.AP10K)), np.nan)
    seconds = 0.0
    capture = cv2.VideoCapture(str(video_path))
    try:
        for frame in range(n):
            if np.isnan(boxes[frame, 0]):
                if not capture.grab():
                    break
                continue
            ok, image = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            points[frame], scores[frame] = estimator(image, boxes[frame])
            seconds += time.perf_counter() - start
    finally:
        capture.release()
    return CP.ClipRun(points, scores, boxes, seconds)


def horse_cache_path(name: str, video_hash: str, seed_key: str) -> Path:
    return U.cache_path(name, video_hash, variant=f"{HORSE_VARIANT}{VERSION}_cal{CP.CAL_VERSION}_t{seed_key}")


def save_horse(path: Path, run: CP.ClipRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=run.points, scores=run.scores, boxes=run.boxes, seconds=run.seconds)


def load_horse(path: Path) -> CP.ClipRun:
    with np.load(path, allow_pickle=False) as data:
        return CP.ClipRun(data["points"], data["scores"], data["boxes"], float(data["seconds"]))


def target_signals(tracks: U.ClipTracks, choice: U.Choice, horse: CP.ClipRun) -> RV.FrameSignals:
    """Semnalele routerului pe ținta urmărită; rândurile cadrelor fără țintă sunt NaN (necunoscut)."""
    values = np.full((tracks.n_frames, len(RV.SIGNALS)), np.nan)
    world = np.full((tracks.n_frames, MEDIAPIPE_POINTS, 3), np.nan)
    with np.errstate(invalid="ignore"):
        for name, v in RV.rider_signals(rider_xy(tracks, choice), world).items():
            values[:, RV.SIGNALS.index(name)] = v
        for name, v in RV.horse_signals(horse).items():
            values[:, RV.SIGNALS.index(name)] = v[:tracks.n_frames]
    values[~known_frames(choice)] = np.nan
    return RV.FrameSignals(names=RV.SIGNALS, values=values)


# ---------- predicția, validarea încrucișată, confuzia ----------

def predict(model, values: np.ndarray, fps: float) -> np.ndarray:
    """Vederea votată (ca `router_vedere.predict_clip`); cadrele fără niciun semnal, adică fără țintă, devin
    „necunoscut”."""
    views = RV.predict_clip(model, values, fps)
    views[np.isnan(np.asarray(values, float)).all(axis=1)] = UNKNOWN
    return views


def leave_one_out(clips: dict[str, ER.ClipData], kind: str = RV.LOGISTIC) -> dict[str, np.ndarray]:
    """Fiecare clip ținut deoparte pe rând: modelul se antrenează pe cadrele etichetate (cu țintă) ale celorlalte."""
    preds = {}
    for held in clips:
        model = RV.make_classifier(kind).fit(*ER._training(clips, [c for c in clips if c != held]))
        preds[held] = predict(model, clips[held].values, clips[held].fps)
    return preds


def train_all(clips: dict[str, ER.ClipData], kind: str = RV.LOGISTIC):
    return RV.make_classifier(kind).fit(*ER._training(clips, list(clips)))


def confusion(truth: np.ndarray, voted: np.ndarray) -> dict[str, dict[str, int]]:
    """Cadrele etichetate: vederea omului → vederea routerului → câte cadre."""
    truth, voted = np.asarray(truth, dtype=object), np.asarray(voted, dtype=object)
    return {t: {v: int(((truth == t) & (voted == v)).sum()) for v in ROUTER_VIEWS} for t in TRUTH_VIEWS}


def evaluate(truth: np.ndarray, voted: np.ndarray) -> dict:
    """R1–R3 pe cadrele cu țintă (principal) și pe toate cadrele etichetate (cu necunoscutele ca netrimise)."""
    truth, voted = np.asarray(truth, dtype=object), np.asarray(voted, dtype=object)
    known = voted != UNKNOWN
    labeled = truth != ER.UNLABELED
    return {"cu_tinta": ER.criteria(truth[known], voted[known]).model_dump(),
            "toate": ER.criteria(truth, voted).model_dump(),
            "necunoscut": {t: int(((truth == t) & ~known).sum()) for t in TRUTH_VIEWS},
            "cadre_etichetate": int(labeled.sum()), "cadre_etichetate_cu_tinta": int((labeled & known).sum()),
            "confuzie": confusion(truth[known], voted[known])}


# ---------- vederea pe intervale, pentru afișare ----------

def intervals(views: np.ndarray) -> list[tuple[int, int, str]]:
    """Cadre consecutive cu aceeași vedere: (primul, ultimul, vederea)."""
    out: list[tuple[int, int, str]] = []
    for frame, view in enumerate(views):
        if out and out[-1][2] == view and out[-1][1] == frame - 1:
            out[-1] = (out[-1][0], frame, view)
        else:
            out.append((frame, frame, str(view)))
    return out


def views_dir() -> Path:
    return ST.results_dir().parent / PILOT_DIR


def views_path(video_hash: str) -> Path:
    return views_dir() / f"{video_hash}__v{VERSION}.json"


def save_views(video_hash: str, clip: str, fps: float, views: np.ndarray, note: str) -> Path:
    path = views_path(video_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"clip": clip, "fps": fps, "urmarire": TRACKING, "nota": note,
                                "intervale": [list(i) for i in intervals(views)]}, ensure_ascii=False),
                    encoding="utf-8")
    return path


def load_views(video_hash: str) -> dict | None:
    """Vederea pilot a unui clip (după amprenta conținutului), dacă a fost calculată; altfel None."""
    path = views_path(video_hash)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


VIEW_NAMES = {**IV.VIEWS, RV.UNDECIDED: "Nedecis", UNKNOWN: "Necunoscut (fără țintă)"}


def view_rows(data: dict) -> list[dict]:
    """Rândurile tabelului din „Analiză”: de la, până la (secunde), vederea și traseul."""
    fps = data["fps"]
    route = {IV.PROFILE: "măsurarea de profil", IV.FRONT: "modulul față/spate", IV.BACK: "modulul față/spate"}
    return [{"de la (s)": round(a / fps, 2), "până la (s)": round((b + 1) / fps, 2), "cadre": f"{a}–{b}",
             "vedere": VIEW_NAMES.get(v, v), "merge la": route.get(v, "nicăieri")} for a, b, v in data["intervale"]]


def view_at(data: dict, frame: int) -> str | None:
    return next((VIEW_NAMES.get(v, v) for a, b, v in data["intervale"] if a <= frame <= b), None)
