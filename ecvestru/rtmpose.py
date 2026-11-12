"""
Stratul RTMPose, calea implicită a aplicației: călcâiul vine din RTMPose-m (Halpe26), pe caseta călărețului găsit de
MediaPipe, în loc de călcâiul MediaPipe. Fuziunea: se ia călcâiul RTMPose; dacă e în treimea de jos a casetei calului,
se ia cel MediaPipe; dacă și acela e acolo, cadrul nu e măsurat pentru picior. Apoi se aplică intervalul plauzibil al
unghiului. Poarta A nu se aplică pe această cale.
Tot aici, orientarea (regula B4): în intervalul analizat, sensul de mers (care dă semnul unghiurilor) vine din votul
direcției labei pe cadrele vecine, iar dacă orientarea minoritară trece de 5%, raportul text se ascunde („profil
neclar”).

`onnxruntime` (dependență a aplicației) se importă abia când se creează sesiunea. Modelul se descarcă la prima analiză
(`masurare.rtmpose_problem`) sau cu `uv run python -m ecvestru.rtmpose descarca`, cu suma de control fixată mai jos.
Fără el, analiza rulează linia de bază și spune de ce.

Licență: codul și greutățile RTMPose sunt Apache-2.0, dar modelul e antrenat pe Body7, care include seturi de date
doar pentru cercetare sau uz necomercial (vezi README.md, „Licențe”). Fără Streamlit.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ecvestru import calitate as C
from ecvestru import onnx_comun as OC
from ecvestru import rezumat as RZ
from ecvestru import schelet as Z
from ecvestru import stocare as ST
from ecvestru.rezultat import FACING_FOOT, FACING_NOSE_EAR, FrameMeasure, OrientationSummary

RTMPOSE_VERSION = 2          # se mărește când se schimbă pașii de calcul (invalidează cache-ul)
                             # 2: scor ≤ 0 → NaN (nu −1 trecut prin transformarea afină); scorurile în cache
MODEL_NAME = "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605"
MODEL_URL = f"https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/{MODEL_NAME}.zip"
MODEL_ZIP_SHA256 = "55b81170e236040b59fc792ad0a8315301ac4c079a3bdb1095d838aad3088d18"
MODEL_FILE = OC.MODEL_FILE
MODEL_DIR = ST.ROOT / "models" / "rtmpose"        # models/ e ignorat de git
THREADS = 1                                       # un fir: timp comparabil cu MediaPipe

HALPE26 = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
           "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
           "right_knee", "left_ankle", "right_ankle", "head", "neck", "hip", "left_big_toe", "right_big_toe",
           "left_small_toe", "right_small_toe", "left_heel", "right_heel")
HEEL_INDEX = {"left": HALPE26.index("left_heel"), "right": HALPE26.index("right_heel")}
TOE_INDEX = {side: (HALPE26.index(f"{side}_big_toe"), HALPE26.index(f"{side}_small_toe")) for side in ("left", "right")}
INPUT_W, INPUT_H = 192, 256
PADDING = OC.PADDING                              # pașii comuni ai modelelor ONNX: ecvestru/onnx_comun.py
MEAN = OC.MEAN
STD = OC.STD
SPLIT_RATIO = OC.SPLIT_RATIO
MIN_BOX_POINTS = 5                                # puncte MediaPipe găsite, pentru caseta călărețului
SOURCE_RTMPOSE, SOURCE_ORIGINAL, SOURCE_EXCLUDED = "rtmpose", "original", "exclus"

# Regula B4: orientarea după direcția labei, cu vot pe cadrele vecine
ORIENTATION_VERSION = 1   # se mărește când se schimbă regula (invalidează rezultatele căii RTMPose)
VOTES = 9                 # cele mai apropiate cadre cu direcția labei
MAX_SECONDS = 0.5         # cât de departe pot fi, în timp
MIN_VOTES = 5
SHARE = 2 / 3             # acordul necesar ca orientarea să se schimbe
PROFILE_UNCLEAR_SHARE = 0.05   # orientarea minoritară peste atât → „profil neclar” (ales pe dev, marjă mică)
PROFILE_UNCLEAR_TEXT = "profil neclar: orientarea se schimbă pe {pct}% din cadre"


# ---------- model: unde e, cum se descarcă ----------

def runtime_available() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


def model_path(folder: Path | None = None) -> Path | None:
    return OC.find_model(folder or MODEL_DIR, MODEL_FILE)


def available() -> bool:
    """Stratul poate rula: există onnxruntime și modelul descărcat."""
    return runtime_available() and model_path() is not None


def download_model(url: str = MODEL_URL, sha256: str = MODEL_ZIP_SHA256, folder: Path | None = None) -> Path:
    """Descarcă pachetul, verifică suma sha256 și abia apoi îl despachetează în `folder` (implicit models/rtmpose)."""
    return OC.download_model(url, sha256, folder or MODEL_DIR, MODEL_FILE)


# ---------- pașii modelului (după pipeline.json din pachet) ----------

def center_scale(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Caseta x0, y0, x1, y1 → centru și mărime, cu padding și raportul de aspect al intrării."""
    return OC.center_scale(bbox, INPUT_W, INPUT_H)


def to_input(points: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return OC.to_input(points, center, scale, INPUT_W, INPUT_H)


def to_image(points: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return OC.to_image(points, center, scale, INPUT_W, INPUT_H)


def prepare(image: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Imaginea BGR → tensorul (1, 3, 256, 192) RGB al modelului, plus centrul și mărimea casetei."""
    return OC.prepare(image, bbox, INPUT_W, INPUT_H)


decode_simcc = OC.decode_simcc


class RtmPose:
    """Un călăreț pe cadru: imaginea BGR și caseta lui → 26 de puncte în pixeli și scorurile lor."""

    def __init__(self, model: Path | None = None, threads: int = THREADS):
        import onnxruntime as ort  # abia aici: importul aplicației nu depinde de onnxruntime
        model = model or model_path()
        if model is None:
            raise FileNotFoundError("Modelul RTMPose nu e descărcat: uv run python -m ecvestru.rtmpose descarca")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, image: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, center, scale = prepare(image, bbox)
        simcc_x, simcc_y = self.session.run(None, {self.input: tensor})
        locs, scores = decode_simcc(simcc_x, simcc_y)
        return to_image(locs, center, scale), scores


# ---------- rularea pe clip ----------

def rider_box(points_px: np.ndarray) -> np.ndarray | None:
    """Caseta x0, y0, x1, y1 a punctelor MediaPipe găsite; None dacă sunt prea puține."""
    found = points_px[~np.isnan(points_px).any(axis=1)]
    if len(found) < MIN_BOX_POINTS:
        return None
    return np.array([*found.min(axis=0), *found.max(axis=0)])


def run_clip(video_path: Path, skeleton: Z.RiderSkeleton, estimator: Callable, only: set[int] | None = None
             ) -> np.ndarray:
    """Punctele RTMPose (N, 26, 2) pe cadrele cu călăreț (NaN în rest), cu caseta din scheletul MediaPipe."""
    return run_clip_scored(video_path, skeleton, estimator, only)[0]


def run_clip_scored(video_path: Path, skeleton: Z.RiderSkeleton, estimator: Callable, only: set[int] | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Ca run_clip, plus scorurile (N, 26) ale modelului (NaN pe cadrele nerulate). Scorurile doar se păstrează."""
    frames = len(skeleton.frame)
    points = np.full((frames, len(HALPE26), 2), np.nan)
    scores = np.full((frames, len(HALPE26)), np.nan)
    rider, px = skeleton.has_rider, skeleton.xy_px
    capture = cv2.VideoCapture(str(video_path))
    try:
        for index in range(frames):
            wanted = rider[index] and (only is None or index in only)
            if not wanted:
                if not capture.grab():
                    break
                continue
            ok, image = capture.read()
            if not ok:
                break
            box = rider_box(px[index])
            if box is not None:
                points[index], scores[index] = estimator(image, box)
    finally:
        capture.release()
    return points, scores


def points_cache_path(name: str, video_hash: str) -> Path:
    return (ST.results_dir() / "rtmpose" /
            f"{ST.clip_key(name, video_hash)}__r{RTMPOSE_VERSION}_{MODEL_ZIP_SHA256[:8]}.npz")


def load_points(video_path: Path, name: str, video_hash: str, skeleton: Z.RiderSkeleton,
                estimator: Callable | None = None) -> np.ndarray:
    """Punctele RTMPose din cache sau calculate (și salvate). Cheia: clipul, versiunea codului și modelul."""
    path = points_cache_path(name, video_hash)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return data["points"]
    points, scores = run_clip_scored(video_path, skeleton, estimator or RtmPose())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points, scores=scores)
    return points


def load_scores(name: str, video_hash: str) -> np.ndarray | None:
    """Scorurile RTMPose (N, 26) din cache; None dacă punctele nu sunt calculate. Nu se folosesc în nicio regulă."""
    path = points_cache_path(name, video_hash)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return data["scores"] if "scores" in data.files else None


def side_heels(points: np.ndarray, skeleton: Z.RiderSkeleton) -> np.ndarray:
    """Călcâiul RTMPose de pe partea aleasă de candidat, (N, 2)."""
    index = np.where(skeleton.side == "left", HEEL_INDEX["left"], HEEL_INDEX["right"])
    return points[np.arange(len(points)), index]


def horse_boxes(skeleton: Z.RiderSkeleton, boxes: dict) -> dict[int, list | None]:
    """Caseta calului de sub călăreț pe fiecare cadru cu călăreț (regula candidatului)."""
    hips = C.midpoint(skeleton, "hip")
    return {int(i): C.rider_box(hips[i], boxes.get(int(i), [])) for i in np.where(skeleton.has_rider)[0]}


# ---------- regula: fuziune + interval ----------

def fuse_heel(rtm: tuple[float, float] | None, original: tuple[float, float] | None, box: list | None
              ) -> tuple[tuple[float, float] | None, str]:
    """RTMPose; în treimea de jos a casetei calului (sau lipsă) → originalul; dacă și el e acolo → exclus."""
    def usable(heel) -> bool:
        return heel is not None and not np.isnan(heel[0]) and not C.heel_in_lower_zone(heel, box)

    if usable(rtm):
        return (float(rtm[0]), float(rtm[1])), SOURCE_RTMPOSE
    if usable(original):
        return (float(original[0]), float(original[1])), SOURCE_ORIGINAL
    return None, SOURCE_EXCLUDED


def _fused(index: int, raw: RZ.Raw, heels: np.ndarray, boxes: dict) -> tuple[float, float] | None:
    """Călcâiul folosit pe cadrul `index` (fuziunea), None dacă cadrul e exclus."""
    rtm = None if np.isnan(heels[index][0]) else tuple(heels[index])
    return fuse_heel(rtm, tuple(raw[index]["points"]["heel"]), boxes.get(index))[0]


def measure_legs(frames: list[FrameMeasure], raw: RZ.Raw, heels: np.ndarray, boxes: dict) -> list[FrameMeasure]:
    """Piciorul pe calea RTMPose, pe cadrele unde MediaPipe a măsurat piciorul: fuziunea, apoi intervalul.
    Trunchiul și cadrele cu piciorul nemăsurat rămân neschimbate."""
    out = []
    for frame in frames:
        if frame.leg_deg is None:
            out.append(frame)
            continue
        heel = _fused(frame.frame, raw, heels, boxes)
        if heel is None:
            out.append(frame.model_copy(update={"leg_deg": None, "reason": RZ.REASON_HEEL_ON_HORSE}))
            continue
        p = raw[frame.frame]["points"]
        angle = RZ.angle_from_vertical(p["hip"], heel, p["facing"])
        if not C.LEG_ANGLE_MIN <= angle <= C.LEG_ANGLE_MAX:
            out.append(frame.model_copy(update={"leg_deg": None, "reason": RZ.REASON_IMPLAUSIBLE_ANGLE}))
        else:
            out.append(frame.model_copy(update={"leg_deg": angle}))
    return out


def heel_sources(raw: RZ.Raw, frames: list[FrameMeasure], heels: np.ndarray, boxes: dict) -> dict[int, str]:
    """De unde vine călcâiul folosit pe fiecare cadru cu picior măsurat (fuziunea): rtmpose sau original."""
    out = {}
    for frame in frames:
        if frame.leg_deg is None:
            continue
        i = frame.frame
        rtm = None if np.isnan(heels[i][0]) else tuple(heels[i])
        heel, source = fuse_heel(rtm, tuple(raw[i]["points"]["heel"]), boxes.get(i))
        if heel is not None:
            out[i] = source
    return out


def film_raw(raw: RZ.Raw, frames: list[FrameMeasure], heels: np.ndarray, boxes: dict) -> RZ.Raw:
    """Copie a brutului cu călcâiul folosit în măsurare, pe cadrele cu picior măsurat (pentru desenul din film)."""
    out = dict(raw)
    for frame in frames:
        heel = None if frame.leg_deg is None else _fused(frame.frame, raw, heels, boxes)
        if heel is not None:
            out[frame.frame] = {**raw[frame.frame], "points": {**raw[frame.frame]["points"], "heel": heel}}
    return out


# ---------- regula B4: orientarea după direcția labei ----------

def _sign(dx: np.ndarray) -> np.ndarray:
    """±1 după semnul diferenței pe x; NaN dacă e zero sau lipsește."""
    out = np.sign(dx).astype(float)
    out[(dx == 0) | np.isnan(dx)] = np.nan
    return out


def current_facing(raw: RZ.Raw, n_frames: int) -> np.ndarray:
    """Orientarea candidatului (nas față de ureche) pe cadrele măsurate; NaN în rest."""
    out = np.full(n_frames, np.nan)
    for i in range(n_frames):
        r = raw.get(i) or {}
        if r.get("points") and r.get("motiv") is None:
            out[i] = r["points"]["facing"]
    return out


def _measured_side(raw: RZ.Raw, skeleton: Z.RiderSkeleton, i: int) -> str:
    """Partea măsurată de candidat: cea al cărei șold e șoldul din brut; altfel partea scheletului."""
    hip = (raw.get(i) or {}).get("points", {}).get("hip")
    for side in ("left", "right"):
        if hip is not None and np.allclose(skeleton.xy_px[i, Z.LANDMARK_NAMES.index(f"{side}_hip")], hip, atol=1e-3):
            return side
    return str(skeleton.side[i])


def foot_direction(raw: RZ.Raw, skeleton: Z.RiderSkeleton, points: np.ndarray) -> np.ndarray:
    """Direcția labei pe cadru, x(vârf) − x(călcâi): întâi RTMPose (călcâiul și mijlocul degetelor mare și mic, pe
    partea scheletului), unde lipsește MediaPipe (călcâiul și `foot_index`, pe partea măsurată). NaN pe cadrele fără
    orientare."""
    n = len(points)
    rows = np.arange(n)
    side = skeleton.side[:n]
    heel = points[rows, np.where(side == "left", HEEL_INDEX["left"], HEEL_INDEX["right"])]
    big, small = (points[rows, np.where(side == "left", TOE_INDEX["left"][k], TOE_INDEX["right"][k])] for k in (0, 1))
    rtm = _sign((big[:, 0] + small[:, 0]) / 2 - heel[:, 0])
    current = current_facing(raw, n)
    mp = np.full(n, np.nan)
    for i in np.where(~np.isnan(current))[0]:
        measured = _measured_side(raw, skeleton, i)
        toe, back = (skeleton.xy_px[i, Z.LANDMARK_NAMES.index(f"{measured}_{part}")] for part in ("foot_index", "heel"))
        mp[i] = _sign(np.array([toe[0] - back[0]]))[0]
    foot = np.where(np.isnan(rtm), mp, rtm)
    foot[np.isnan(current)] = np.nan
    return foot


def vote_facing(current: np.ndarray, foot: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Votul labei pe cele mai apropiate VOTES cadre în ±MAX_SECONDS: cu cel puțin MIN_VOTES voturi și SHARE acord,
    orientarea e cea votată (decis = True); altfel rămâne cea curentă."""
    out, decided = current.copy(), np.zeros(len(current), bool)
    have = np.where(~np.isnan(foot))[0]
    reach = round(MAX_SECONDS * fps)
    for i in np.where(~np.isnan(current))[0]:
        near = have[np.abs(have - i) <= reach]
        votes = foot[near[np.argsort(np.abs(near - i), kind="stable")][:VOTES]]
        if len(votes) < MIN_VOTES:
            continue
        plus = float(np.mean(votes == 1))
        if plus >= SHARE:
            out[i], decided[i] = 1.0, True
        elif 1 - plus >= SHARE:
            out[i], decided[i] = -1.0, True
    return out, decided


def minority_share(values: np.ndarray) -> float:
    """Ponderea orientării mai rare printre cadrele cu orientare (NaN dacă nu există niciunul). Din numărători, nu din
    1 − pondere: altfel 1 din 20 ar ieși puțin peste 5%."""
    known = values[~np.isnan(values)]
    if not len(known):
        return float("nan")
    plus = int(np.sum(known == 1))
    return min(plus, len(known) - plus) / len(known)


def profile_unclear(share: float) -> bool:
    return bool(share > PROFILE_UNCLEAR_SHARE)


def orient(raw: RZ.Raw, skeleton: Z.RiderSkeleton, points: np.ndarray, fps: float, segment: tuple[int, int]
           ) -> tuple[RZ.Raw, OrientationSummary, dict[int, str]]:
    """Regula B4, doar în `segment`: copie a brutului cu orientarea după vot, rezumatul și sursa orientării pe fiecare
    cadru cu orientare din interval. Votul folosește și cadrele vecine din afara intervalului; în afara intervalului
    orientarea nu se schimbă. Brutul primit rămâne neatins."""
    current = current_facing(raw, len(points))
    voted, decided = vote_facing(current, foot_direction(raw, skeleton, points), fps)
    a, b = segment
    inside = np.zeros(len(current), bool)
    inside[a:b + 1] = True
    known = inside & ~np.isnan(current)
    changed = known & (voted != current)
    out = dict(raw)
    for i in np.where(changed)[0]:
        out[int(i)] = {**raw[int(i)], "points": {**raw[int(i)]["points"], "facing": float(voted[i])}}
    share = minority_share(voted[known]) if known.any() else 0.0
    pct = round(100 * share)
    unclear = profile_unclear(share)
    reason = PROFILE_UNCLEAR_TEXT.format(pct=pct) if unclear else None
    summary = OrientationSummary(changed_frames=int(changed.sum()), foot_frames=int((known & decided).sum()),
                                 frames_with_facing=int(known.sum()), minority_share=share, minority_pct=pct,
                                 profile_unclear=unclear, reason=reason)
    sources = {int(i): FACING_FOOT if decided[i] else FACING_NOSE_EAR for i in np.where(known)[0]}
    return out, summary, sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ecvestru.rtmpose", description=__doc__.split("\n\n")[0])
    parser.add_subparsers(dest="command", required=True).add_parser("descarca", help="descarcă și verifică modelul")
    parser.parse_args(argv)
    print(f"Model: {download_model()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
