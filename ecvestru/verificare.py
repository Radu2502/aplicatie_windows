"""
Unghiurile pe fiecare cadru din punctele MediaPipe (aceeași regulă ca research/candidate.py), motivele de excludere,
segmentele filmate din profil (data/profil.csv), video cu schelet și căile comune (clipurile, adnotările).
Landmark-urile vin din cache-ul data/results/poses/ (MediaPipe Pose `full`, `num_poses=1`, mod VIDEO, identic cu
linia de bază din candidate.py). Folderul adnotărilor se poate schimba cu ECVESTRU_ANNOTATIONS.
"""
from __future__ import annotations

import csv
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ecvestru import constante_candidat as KC
from ecvestru import inventar

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "videos"
POSES = ROOT / "data" / "results" / "poses"
SKELETON_VIDEOS = ROOT / "data" / "results" / "verificare"
CANDIDATE = ROOT / "research" / "candidate.py"
PROFILE_CSV = ROOT / "data" / "profil.csv"

MIN_VISIBILITY = KC.MIN_VISIBILITY   # ca în candidat (ecvestru/constante_candidat.py)
IDX = {"nose": 0, "left_ear": 7, "right_ear": 8, "left_shoulder": 11, "right_shoulder": 12,
       "left_hip": 23, "right_hip": 24, "left_heel": 29, "right_heel": 30}
CHAIN = ("shoulder", "hip", "heel")

M_NO_PERSON = "persoană negăsită"
M_TRUNK = "umăr sau șold invizibil"
M_LEG = "șold sau călcâi invizibil"

# Conexiunile scheletului MediaPipe Pose (33 de puncte), fără fața.
CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
               (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30),
               (30, 32), (28, 32)]


def annotations_dir() -> Path:
    return Path(os.environ.get("ECVESTRU_ANNOTATIONS", ROOT / "data" / "annotations"))


# ---------- unghiuri și excluderi ----------

def angle_from_vertical(top, bottom, facing: float) -> float:
    return math.degrees(math.atan2((top[0] - bottom[0]) * facing, bottom[1] - top[1]))


def frame_angles(lm: np.ndarray) -> dict:
    """Pentru un cadru (33, 4: x_px, y_px, z, visibility): partea, punctele lanțului, unghiurile
    și motivele de excludere. Aceeași regulă ca angles_for_frame din candidate.py."""
    if np.isnan(lm[0, 0]):
        return {"found": False, "side": None, "facing": None, "points": {}, "trunk_deg": None,
                "leg_deg": None, "motive": [M_NO_PERSON]}
    vis = lambda name: float(lm[IDX[name], 3])  # noqa: E731
    xy = lambda name: (float(lm[IDX[name], 0]), float(lm[IDX[name], 1]))  # noqa: E731
    side = "left" if vis("left_hip") >= vis("right_hip") else "right"
    facing = 1.0 if xy("nose")[0] >= xy(f"{side}_ear")[0] else -1.0
    pts = {k: (*xy(f"{side}_{k}"), vis(f"{side}_{k}")) for k in CHAIN}
    trunk_ok = min(pts["shoulder"][2], pts["hip"][2]) >= MIN_VISIBILITY
    leg_ok = min(pts["hip"][2], pts["heel"][2]) >= MIN_VISIBILITY
    motive = ([] if trunk_ok else [M_TRUNK]) + ([] if leg_ok else [M_LEG])
    return {"found": True, "side": side, "facing": facing, "points": pts,
            "trunk_deg": angle_from_vertical(pts["shoulder"], pts["hip"], facing) if trunk_ok else None,
            "leg_deg": angle_from_vertical(pts["hip"], pts["heel"], facing) if leg_ok else None,
            "motive": motive}


def load_poses(video: str) -> dict:
    """Landmark-urile din cache; dacă lipsesc, rulează MediaPipe (ca în inventar)."""
    path = POSES / f"{Path(video).stem}.npz"
    if not path.exists():
        inventar.extract_poses(VIDEOS / video)
    data = np.load(path)
    return {k: data[k] for k in data.files}


def series(landmarks: np.ndarray) -> list[dict]:
    return [frame_angles(lm) for lm in landmarks]


def exclusion_intervals(rows: list[dict]) -> list[dict]:
    """Cadre consecutive cu același motiv de excludere, grupate în intervale."""
    out: list[dict] = []
    for i, r in enumerate(rows):
        for m in r["motive"]:
            last = next((o for o in reversed(out) if o["motiv"] == m), None)
            if last and last["pana_la"] == i - 1:
                last["pana_la"] = i
                last["cadre"] += 1
            else:
                out.append({"de_la": i, "pana_la": i, "cadre": 1, "motiv": m})
    return sorted(out, key=lambda o: (o["de_la"], o["motiv"]))


# ---------- segmentele din profil ----------

def profile_segments(path: Path = PROFILE_CSV) -> dict[str, list[tuple[int, int]]]:
    """Clip → segmentele filmate din profil (cadre, inclusiv capetele), din data/profil.csv."""
    out = {}
    for r in _rows(path):
        out[r["video"]] = [tuple(int(x) for x in s.split("-")) for s in r["segmente"].split(";") if s.strip()]
    return out


# ---------- desen și video ----------

def draw(frame_bgr: np.ndarray, lm: np.ndarray | None, info: dict, label: str | None = None) -> np.ndarray:
    """Scheletul complet (dacă există landmark-uri) și lanțul umăr–șold–călcâi."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    t = max(1, round(min(h, w) / 300))
    if info["found"] and lm is not None:
        for a, b in CONNECTIONS:
            if min(lm[a, 3], lm[b, 3]) >= MIN_VISIBILITY:
                cv2.line(img, (round(lm[a, 0]), round(lm[a, 1])), (round(lm[b, 0]), round(lm[b, 1])),
                         (220, 220, 220), t, cv2.LINE_AA)
    if info["found"]:
        p = info["points"]
        for a, b in (("shoulder", "hip"), ("hip", "heel")):
            if p[a] is None or p[b] is None:
                continue
            ok = min(p[a][2], p[b][2]) >= MIN_VISIBILITY
            cv2.line(img, (round(p[a][0]), round(p[a][1])), (round(p[b][0]), round(p[b][1])),
                     (0, 255, 255) if ok else (0, 0, 255), 2 * t + 1, cv2.LINE_AA)
        for k in CHAIN:
            if p[k] is None:
                continue
            x, y, v = p[k]
            cv2.circle(img, (round(x), round(y)), 3 * t + 1, (0, 200, 0) if v >= MIN_VISIBILITY else (0, 0, 255),
                       -1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (8, 8 + 18 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (0, 0, 0), 3 * t, cv2.LINE_AA)
        cv2.putText(img, label, (8, 8 + 18 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (255, 255, 255), t, cv2.LINE_AA)
    return img


def frame_label(i: int, info: dict) -> str:
    fmt = lambda v: "-" if v is None else f"{v:+.0f}"  # noqa: E731
    return f"#{i}  trunchi {fmt(info['trunk_deg'])}  picior {fmt(info['leg_deg'])}"


def crop_around(img: np.ndarray, info: dict, margin: float = 1.2) -> np.ndarray:
    """Decupaj în jurul lanțului umăr–șold–călcâi, cu loc pentru cal (pentru călăreții mici)."""
    if not info["found"]:
        return img
    xs = [p[0] for p in info["points"].values() if p is not None]
    ys = [p[1] for p in info["points"].values() if p is not None]
    size = max(max(ys) - min(ys), max(xs) - min(xs), 40) * (1 + 2 * margin)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    h, w = img.shape[:2]
    x0, y0 = int(max(cx - size / 2, 0)), int(max(cy - size / 2, 0))
    x1, y1 = int(min(cx + size / 2, w)), int(min(cy + size / 2, h))
    return img[y0:y1, x0:x1]


def skeleton_video(video: str, out_dir: Path = SKELETON_VIDEOS) -> Path:
    """Video H.264 (redabil în browser) cu scheletul desenat; refăcut doar dacă s-a schimbat cache-ul."""
    poses = POSES / f"{Path(video).stem}.npz"
    data = load_poses(video)
    out = out_dir / f"{Path(video).stem}.mp4"
    if out.exists() and out.stat().st_mtime >= poses.stat().st_mtime:
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    lms, fps = data["landmarks"], float(data["fps"])
    w, h = int(data["width"]), int(data["height"])
    tmp = out.with_suffix(".tmp.mp4")
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps:.3f}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(VIDEOS / video))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(lms):
                break
            info = frame_angles(lms[i])
            proc.stdin.write(draw(frame, lms[i], info, frame_label(i, info)).tobytes())
            i += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg a eșuat pentru {video}")
    tmp.replace(out)
    return out


# ---------- citirea CSV-urilor ----------

def _rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


