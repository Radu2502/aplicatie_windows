"""
Alegerea călărețului prin click în pagina „Analiză”: zona afișată din cadru, conversia click → pixeli în cadrul
original (și invers) și imaginea afișată.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

LABELS = {"shoulder": "Umăr", "hip": "Șold", "heel": "Călcâi"}
COLORS = {"shoulder": (255, 80, 80), "hip": (80, 220, 80), "heel": (80, 160, 255)}  # RGB


@dataclass(frozen=True)
class View:
    """Zona afișată din cadrul original (x0, y0, w, h) și mărimea ei pe ecran."""
    x0: float
    y0: float
    w: float
    h: float
    disp_w: int
    disp_h: int


def make_view(frame_w: int, frame_h: int, zoom: float = 1.0, cx: float | None = None,
              cy: float | None = None, max_w: int = 900, max_h: int = 700) -> View:
    """Decupajul de mărime frame/zoom centrat în (cx, cy), ținut în cadru, scalat să încapă."""
    w, h = frame_w / zoom, frame_h / zoom
    cx = frame_w / 2 if cx is None else cx
    cy = frame_h / 2 if cy is None else cy
    x0 = min(max(cx - w / 2, 0.0), frame_w - w)
    y0 = min(max(cy - h / 2, 0.0), frame_h - h)
    scale = min(max_w / w, max_h / h)
    return View(x0, y0, w, h, max(1, round(w * scale)), max(1, round(h * scale)))


def to_original(view: View, x: float, y: float, shown_w: float | None = None,
                shown_h: float | None = None) -> tuple[float, float]:
    """Click (în pixelii imaginii afișate) → pixeli în cadrul original.

    shown_w/shown_h: mărimea reală afișată în browser (o întoarce componenta); implicit view.disp_*.
    """
    shown_w = shown_w or view.disp_w
    shown_h = shown_h or view.disp_h
    return view.x0 + x * view.w / shown_w, view.y0 + y * view.h / shown_h


def to_display(view: View, x: float, y: float) -> tuple[float, float]:
    return (x - view.x0) * view.disp_w / view.w, (y - view.y0) * view.disp_h / view.h


def render(frame_bgr: np.ndarray, view: View, points: dict, active: str | None = None) -> np.ndarray:
    """Imaginea afișată (RGB): decupajul scalat, cu liniile umăr–șold–călcâi și punctele."""
    x0, y0 = int(math.floor(view.x0)), int(math.floor(view.y0))
    x1 = int(math.ceil(view.x0 + view.w))
    y1 = int(math.ceil(view.y0 + view.h))
    crop = frame_bgr[y0:y1, x0:x1]
    # Scalare exactă a zonei [x0, x0+w) → [0, disp_w): folosim o transformare afină, nu resize,
    # ca desenul să cadă exact pe pixelii pe care îi întoarce to_original.
    sx, sy = view.disp_w / view.w, view.disp_h / view.h
    m = np.float32([[sx, 0, (x0 - view.x0) * sx], [0, sy, (y0 - view.y0) * sy]])
    img = cv2.warpAffine(crop, m, (view.disp_w, view.disp_h), flags=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    disp = {k: to_display(view, *p) for k, p in points.items() if p is not None}
    for a, b in (("shoulder", "hip"), ("hip", "heel")):
        if a in disp and b in disp:
            cv2.line(img, tuple(map(round, disp[a])), tuple(map(round, disp[b])), (255, 255, 0), 2,
                     cv2.LINE_AA)
    for k, (x, y) in disp.items():
        c = (round(x), round(y))
        cv2.circle(img, c, 9 if k == active else 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, c, 7 if k == active else 4, COLORS[k], -1, cv2.LINE_AA)
        cv2.putText(img, LABELS[k], (c[0] + 10, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, LABELS[k], (c[0] + 10, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLORS[k], 1, cv2.LINE_AA)
    return img
