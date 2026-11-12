"""
Mișcarea camerei ținute în mână, estimată din fundal: puncte urmărite între cadre, în afara zonei
caselor de cal și a călăreților de deasupra lor. Se folosește doar mișcarea relativă dintre cadre, nu o
înclinare absolută: liniile verticale ale scenei nu dau un rezultat sigur. Fără praguri: raportul
arată amplitudinea, iar dacă estimarea nu e posibilă, spune asta. Fără Streamlit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ecvestru.rezultat import CameraEstimate

MAX_CORNERS = 300
CORNER_QUALITY = 0.01
CORNER_DISTANCE_PX = 8
MIN_POINTS = 20               # sub atâtea puncte de fundal, cadrul nu are estimare
RANSAC_THRESHOLD_PX = 2.0
MIN_ESTIMATED_SHARE = 0.9     # sub această parte din interval, amplitudinea nu se raportează
RIDER_ABOVE = 0.8             # zona exclusă urcă deasupra casetei, cât decupajul candidatului
DISPLAY_DECIMALS = 1
MASK_ON = 255
MOTION_VERSION = 1            # se mărește când se schimbă estimarea, ca să nu se folosească un cache vechi


@dataclass(frozen=True)
class Motion:
    angle: float   # grade, între cadrul precedent și acesta; pozitiv = imaginea se rotește spre dreapta
    dx: float
    dy: float
    points: int


def excluded_box(box) -> tuple[float, float, float, float]:
    left, top, width, height = box
    above = RIDER_ABOVE * height
    return left, top - above, width, height + above


def _background_mask(shape, boxes) -> np.ndarray:
    mask = np.full(shape, MASK_ON, dtype=np.uint8)
    for box in boxes:
        left, top, width, height = excluded_box(box)
        mask[max(0, int(top)):max(0, int(top + height)), max(0, int(left)):max(0, int(left + width))] = 0
    return mask


def frame_motion(previous: np.ndarray, current: np.ndarray, boxes) -> Motion | None:
    """Mișcarea fundalului între două cadre gri, sau None dacă nu sunt destule puncte sigure."""
    corners = cv2.goodFeaturesToTrack(previous, MAX_CORNERS, CORNER_QUALITY, CORNER_DISTANCE_PX,
                                      mask=_background_mask(previous.shape, boxes))
    if corners is None or len(corners) < MIN_POINTS:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
    found = status.ravel() == 1
    if found.sum() < MIN_POINTS:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(corners[found], moved[found], method=cv2.RANSAC,
                                                  ransacReprojThreshold=RANSAC_THRESHOLD_PX)
    if matrix is None or int(inliers.sum()) < MIN_POINTS:
        return None
    return Motion(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])), float(matrix[0, 2]), float(matrix[1, 2]),
                  int(inliers.sum()))


def clip_motion(video_path: Path, boxes: dict) -> dict[int, Motion]:
    """Mișcarea spre fiecare cadru, față de cel precedent. Cadrele fără estimare lipsesc."""
    out, previous, frame = {}, None, 0
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ok, image = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            step = frame_motion(previous, gray, boxes.get(frame, []))
            if step is not None:
                out[frame] = step
        previous, frame = gray, frame + 1
    cap.release()
    return out


def rotation_amplitude(motion: dict[int, Motion], segment: tuple[int, int]) -> CameraEstimate:
    """Cât s-a rotit camera în interval: max–min al rotirii acumulate, sau fără număr dacă nu se poate."""
    start, end = segment
    wanted = range(start + 1, end + 1)
    known = [motion[f].angle for f in wanted if f in motion]
    total = len(wanted)
    if not known or len(known) < MIN_ESTIMATED_SHARE * total:
        return CameraEstimate(frames_estimated=len(known), frames_total=total)
    path = np.cumsum([0.0, *known])
    amplitude = float(path.max() - path.min())
    return CameraEstimate(amplitude_deg=amplitude, amplitude_rounded=round(amplitude, DISPLAY_DECIMALS),
                          frames_estimated=len(known), frames_total=total)
