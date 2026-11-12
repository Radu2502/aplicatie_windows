"""
Routerul de vedere: pe fiecare cadru, vederea călărețului principal (profil, față, spate, oblic, nu se vede), ca doar
cadrele din profil să intre în măsurarea de profil, iar cele din față sau spate în modulul față/spate. Trei pași, toți
aici:

1. **Semnalele pe cadru** (NaN unde lipsesc): raportul lățime umeri / trunchi; semnul stânga–dreapta al umerilor și al
   coatelor în imagine (stânga − dreapta pe x, raportat la trunchi: din față stânga călărețului e în dreapta imaginii);
   orientarea 3D a liniei umerilor din punctele world ale MediaPipe (cosinus și sinus); direcția calului din
   ecvestru/directie_cal.py (lungimea cap–coadă față de casetă și cât de orizontală e); forma casetei calului.
2. **Clasificatorul mic**, cu tipul ca parametru (regresia logistică multinomială, în numpy).
3. **Votul pe ~1 s**: vederea majoritară din fereastră, altfel „nedecis” (nu merge nicăieri).

Stratul e opțional și oprit implicit (ROUTER_LAYER_DEFAULT): calea implicită a aplicației nu se schimbă. Antrenarea și
evaluarea sunt în ecvestru/evaluare_router.py. Fără Streamlit.

Rulare (doar umple cache-ul semnalelor; nu raportează nicio valoare din ele):
  uv run python -m ecvestru.router_vedere semnale --procese 6
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecvestru import cal_puncte as CP
from ecvestru import directie_cal as DC
from ecvestru import intervale as IV
from ecvestru import stocare as ST

ROUTER_LAYER_DEFAULT = False         # stratul router e oprit implicit
SIGNALS_VERSION = 1                  # se mărește când se schimbă calculul semnalelor (invalidează cache-ul)

# Indicii MediaPipe Pose
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_HIP, RIGHT_HIP = 11, 12, 13, 14, 23, 24
MIN_TORSO_PX = 1.0                   # sub atât trunchiul nu e o lungime: raportul e NaN

SIGNALS = ("raport_umeri", "umeri_stanga_dreapta", "coate_stanga_dreapta", "orientare_cos", "orientare_sin",
           "cal_lungime", "cal_orizontal", "cal_caseta")

# Traseul cadrului
UNDECIDED = "nedecis"
NO_PREDICTION = ""
ROUTE_PROFILE, ROUTE_FRONT_BACK, ROUTE_NONE = "profil", "fata_spate", "niciunul"
FRONT_BACK_VIEWS = (IV.FRONT, IV.BACK)

# Votul (ca la B4: fereastra în secunde, centrată pe cadru)
VOTE_SECONDS = 1.0
VOTE_SHARE = 0.6                     # vederea majoritară trebuie să aibă cel puțin atât din voturi
VOTE_MIN_FILL = 0.5                  # și să voteze cel puțin atâtea dintre cadrele ferestrei

# Clasificatorul
LOGISTIC = "logistica"
LOGISTIC_L2 = 1e-2
LOGISTIC_STEPS = 600
LOGISTIC_RATE = 0.5
MIN_STD = 1e-9


# ---------- semnalele călărețului ----------

def _signed_width(xy: np.ndarray, left: int, right: int, torso: np.ndarray) -> np.ndarray:
    return (xy[:, left, 0] - xy[:, right, 0]) / torso


def rider_signals(xy_px: np.ndarray, world_xyz: np.ndarray) -> dict[str, np.ndarray]:
    """Semnalele scheletului pe cadre: (N, 33, 2) pixeli și (N, 33, 3) world → câte un vector (N,) pe semnal."""
    mid_shoulder = (xy_px[:, LEFT_SHOULDER] + xy_px[:, RIGHT_SHOULDER]) / 2
    mid_hip = (xy_px[:, LEFT_HIP] + xy_px[:, RIGHT_HIP]) / 2
    torso = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    torso = np.where(torso >= MIN_TORSO_PX, torso, np.nan)
    shoulders = _signed_width(xy_px, LEFT_SHOULDER, RIGHT_SHOULDER, torso)
    line = world_xyz[:, RIGHT_SHOULDER] - world_xyz[:, LEFT_SHOULDER]
    yaw = np.arctan2(line[:, 2], line[:, 0])
    return {"raport_umeri": np.abs(shoulders), "umeri_stanga_dreapta": shoulders,
            "coate_stanga_dreapta": _signed_width(xy_px, LEFT_ELBOW, RIGHT_ELBOW, torso),
            "orientare_cos": np.cos(yaw), "orientare_sin": np.sin(yaw)}


# ---------- semnalele calului ----------

def horse_signals(run: CP.ClipRun) -> dict[str, np.ndarray]:
    features = DC.direction_features(run)
    return {"cal_lungime": features.length_ratio, "cal_orizontal": np.abs(np.cos(np.radians(features.angle_deg))),
            "cal_caseta": features.aspect}


# ---------- semnalele pe cadrele clipului ----------

@dataclass(frozen=True)
class FrameSignals:
    names: tuple[str, ...]
    values: np.ndarray       # (n_frames, len(names)), NaN unde lipsește


def frame_signals(skeleton, horse: CP.ClipRun | None, n_frames: int) -> FrameSignals:
    """Semnalele pe toate cadrele clipului (indicii OpenCV). `skeleton` are `frame`, `xy_px`, `world_xyz`."""
    values = np.full((n_frames, len(SIGNALS)), np.nan)
    if skeleton is not None:
        frames = np.asarray(skeleton.frame, dtype=int)
        keep = (frames >= 0) & (frames < n_frames)
        for name, v in rider_signals(np.asarray(skeleton.xy_px, float), np.asarray(skeleton.world_xyz, float)).items():
            values[frames[keep], SIGNALS.index(name)] = v[keep]
    if horse is not None:
        n = min(n_frames, len(horse.boxes))
        for name, v in horse_signals(horse).items():
            values[:n, SIGNALS.index(name)] = v[:n]
    return FrameSignals(names=SIGNALS, values=values)


def save_signals(path: Path, signals: FrameSignals) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, names=np.array(signals.names), values=signals.values)


def load_signals(path: Path) -> FrameSignals:
    with np.load(path, allow_pickle=False) as data:
        return FrameSignals(names=tuple(str(n) for n in data["names"]), values=data["values"])


# ---------- clasificatorul ----------

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LogisticClassifier:
    """Regresia logistică multinomială cu L2, în numpy, deterministă. Semnalele se standardizează pe antrenare;
    un NaN devine media (0 după standardizare) și are un indicator „lipsește” separat."""

    kind = LOGISTIC

    def __init__(self, l2: float = LOGISTIC_L2, steps: int = LOGISTIC_STEPS, rate: float = LOGISTIC_RATE):
        self.l2, self.steps, self.rate = l2, steps, rate
        self.classes: tuple[str, ...] = ()
        self.mean = self.std = self.weights = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        missing = np.isnan(X)
        z = np.where(missing, 0.0, (X - self.mean) / self.std)
        return np.hstack([z, missing.astype(float), np.ones((len(X), 1))])

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticClassifier:
        y = np.asarray(y, dtype=object)
        self.classes = tuple(sorted({str(v) for v in y}))
        if len(self.classes) < 2:
            raise ValueError("Clasificatorul are nevoie de cel puțin două vederi în antrenare.")
        X = np.asarray(X, float)
        with np.errstate(all="ignore"):
            mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        self.mean = np.nan_to_num(mean)
        self.std = np.where(np.nan_to_num(std) > MIN_STD, np.nan_to_num(std), 1.0)
        D = self._design(X)
        Y = (y[:, None] == np.array(self.classes, dtype=object)[None, :]).astype(float)
        W = np.zeros((D.shape[1], len(self.classes)))
        for _ in range(self.steps):
            grad = D.T @ (_softmax(D @ W) - Y) / len(D)
            grad[:-1] += self.l2 * W[:-1]           # fără penalizare pe termenul liber
            W -= self.rate * grad
        self.weights = W
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self._design(X) @ self.weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(self.classes, dtype=object)[self.predict_proba(X).argmax(axis=1)]

    def to_dict(self) -> dict:
        return {"tip": self.kind, "l2": self.l2, "steps": self.steps, "rate": self.rate,
                "classes": list(self.classes), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> LogisticClassifier:
        model = cls(l2=data["l2"], steps=data["steps"], rate=data["rate"])
        model.classes = tuple(data["classes"])
        model.mean, model.std = np.array(data["mean"]), np.array(data["std"])
        model.weights = np.array(data["weights"])
        return model


CLASSIFIERS: dict[str, Callable] = {LOGISTIC: LogisticClassifier}


def make_classifier(kind: str = LOGISTIC, **params):
    if kind not in CLASSIFIERS:
        raise ValueError(f"Tip de clasificator necunoscut: {kind} (cunoscute: {', '.join(CLASSIFIERS)}).")
    return CLASSIFIERS[kind](**params)


def classifier_from_dict(data: dict):
    return CLASSIFIERS[data["tip"]].from_dict(data)


# ---------- votul pe ~1 s ----------

def vote(views: np.ndarray, fps: float) -> np.ndarray:
    """Vederea majoritară în fereastra de VOTE_SECONDS centrată pe cadru, dacă are cel puțin VOTE_SHARE din voturi
    și dacă votează cel puțin VOTE_MIN_FILL din cadrele ferestrei; altfel UNDECIDED. Cadrele fără predicție
    (NO_PREDICTION) nu votează."""
    views = np.asarray(views, dtype=object)
    half = max(int(round(VOTE_SECONDS * fps / 2)), 0)
    out = np.full(len(views), UNDECIDED, dtype=object)
    for i in range(len(views)):
        window = views[max(0, i - half): i + half + 1]
        cast = window[window != NO_PREDICTION]
        if len(cast) < math.ceil(VOTE_MIN_FILL * len(window)):
            continue
        labels, counts = np.unique(cast.astype(str), return_counts=True)
        best = counts.argmax()
        if counts[best] >= VOTE_SHARE * len(cast) and (counts == counts[best]).sum() == 1:
            out[i] = labels[best]
    return out


def predict_clip(model, values: np.ndarray, fps: float) -> np.ndarray:
    """Vederea votată pe fiecare cadru al clipului; cadrele fără niciun semnal nu au predicție."""
    values = np.asarray(values, float)
    views = np.full(len(values), NO_PREDICTION, dtype=object)
    usable = ~np.isnan(values).all(axis=1)
    if usable.any():
        views[usable] = model.predict(values[usable])
    return vote(views, fps)


def routes(views: np.ndarray) -> np.ndarray:
    """Profil → măsurarea de profil; față sau spate → modulul față/spate; restul (oblic, nu se vede, nedecis)
    nicăieri."""
    views = np.asarray(views, dtype=object)
    return np.where(views == IV.PROFILE, ROUTE_PROFILE,
                    np.where(np.isin(views, FRONT_BACK_VIEWS), ROUTE_FRONT_BACK, ROUTE_NONE)).astype(object)


# ---------- cache-ul semnalelor pe clipurile noi (fără nicio valoare raportată) ----------

def signals_cache_path(name: str, video_hash: str) -> Path:
    candidate = ST.candidate_label().replace("@", "_")
    return (ST.results_dir() / "router" /
            f"{ST.clip_key(name, video_hash)}__{candidate}__s{SIGNALS_VERSION}__a{CP.CAL_VERSION}"
            f"__c{ST.BOXES_VERSION}.npz")


def load_clip_signals(path: Path) -> FrameSignals:
    """Semnalele unui clip, din cache sau calculate din cache-urile scheletului și calului (și salvate)."""
    path = Path(path)
    video_hash = ST.file_hash(path)
    cache = signals_cache_path(path.name, video_hash)
    if cache.exists():
        return load_signals(cache)
    from ecvestru import calitate as C
    skeleton = ST.load_rider_skeleton(path, path.name, video_hash, ST.candidate_label())
    boxes = ST.load_boxes(path, path.name, video_hash)
    hips = None if skeleton is None else C.midpoint(skeleton, "hip")
    horse = CP.load_points(path, path.name, video_hash, boxes, hips)
    signals = frame_signals(skeleton, horse, ST.video_info(path)[0])
    save_signals(cache, signals)
    return signals


@dataclass(frozen=True)
class SignalsRun:
    clip: str
    frames: int
    seconds: float
    error: str | None


def fill_signals(path: Path) -> SignalsRun:
    start = time.perf_counter()
    try:
        frames, error = len(load_clip_signals(path).values), None
    except Exception as exc:  # o eroare pe un clip nu oprește celelalte; se raportează
        frames, error = 0, f"{type(exc).__name__}: {exc}"
    return SignalsRun(clip=Path(path).name, frames=frames, seconds=time.perf_counter() - start, error=error)


def main(argv: list[str] | None = None) -> int:
    from ecvestru import clipuri_noi as CN
    parser = argparse.ArgumentParser(prog="python -m ecvestru.router_vedere", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fill = commands.add_parser("semnale", help="cache-ul semnalelor routerului pe clipurile noi; doar timpul")
    fill.add_argument("--procese", type=int, default=CN.DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    folder = CN.clips_dir()
    paths = sorted((folder / c for c in CN.list_clips(folder)), key=lambda p: p.stat().st_size, reverse=True)
    with ProcessPoolExecutor(max_workers=args.procese) as pool:
        runs = list(pool.map(fill_signals, paths))
    for r in sorted(runs, key=lambda r: r.clip):
        print(f"{r.clip}: {r.frames} cadre, {r.seconds:.0f} s{'' if r.error is None else ' – ' + r.error}")
    print(f"Erori: {sum(r.error is not None for r in runs)}; cache: {ST.results_dir() / 'router'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
