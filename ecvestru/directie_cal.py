"""
Mărimi pentru direcția calului, din punctele AP-10K (ecvestru/cal_puncte.py) și caseta calului, pe fiecare cadru:
- unghiul cap–coadă: vectorul rădăcina cozii → cap, în grade, măsurat de la dreapta imaginii, în sens trigonometric
  (90° = capul spre sus în imagine; y-ul imaginii crește în jos); capul e nasul sau, dacă nasul e sub prag, gâtul;
- lungimea aparentă cap–coadă, raportată la înălțimea casetei calului (lungă din profil, scurtă din față sau spate);
- raportul lățime / înălțime al casetei calului.
Doar mărimile: nu decid vederea. NaN unde lipsește un punct sau caseta. Fără Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecvestru import cal_puncte as CP

HEAD_NOSE, HEAD_NECK, HEAD_NONE = "nas", "gat", ""
NOSE, NECK, TAIL = (CP.AP10K.index(n) for n in ("nose", "neck", "root_of_tail"))
HALF_TURN = 180.0


def _usable(scores: np.ndarray, index: int, threshold: float) -> np.ndarray:
    values = scores[:, index]
    return ~np.isnan(values) & (np.nan_to_num(values, nan=-np.inf) > threshold)


def head_points(points: np.ndarray, scores: np.ndarray, threshold: float = CP.SCORE_MIN
                ) -> tuple[np.ndarray, np.ndarray]:
    """Capul pe cadre (N, 2): nasul peste prag, altfel gâtul peste prag, altfel NaN; și sursa lui."""
    nose, neck = _usable(scores, NOSE, threshold), _usable(scores, NECK, threshold)
    heads = np.full((len(points), 2), np.nan)
    heads[neck] = points[neck, NECK]
    heads[nose] = points[nose, NOSE]
    sources = np.where(nose, HEAD_NOSE, np.where(neck, HEAD_NECK, HEAD_NONE))
    return heads, sources


def tail_points(points: np.ndarray, scores: np.ndarray, threshold: float = CP.SCORE_MIN) -> np.ndarray:
    tails = np.full((len(points), 2), np.nan)
    usable = _usable(scores, TAIL, threshold)
    tails[usable] = points[usable, TAIL]
    return tails


def head_tail_angle(heads: np.ndarray, tails: np.ndarray) -> np.ndarray:
    """Unghiul vectorului coadă → cap, în grade, în (-180°, 180°]: 0° spre dreapta imaginii, 90° spre sus."""
    d = heads - tails
    angle = np.degrees(np.arctan2(-d[:, 1], d[:, 0]))
    return np.where(angle <= -HALF_TURN, angle + 2 * HALF_TURN, angle)   # -180° (din -0) devine 180°


def _heights(boxes: np.ndarray) -> np.ndarray:
    heights = boxes[:, 3] - boxes[:, 1]
    return np.where(heights > 0, heights, np.nan)


def head_tail_length(heads: np.ndarray, tails: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Distanța cap–coadă în imagine, împărțită la înălțimea casetei calului (x0, y0, x1, y1)."""
    return np.linalg.norm(heads - tails, axis=1) / _heights(boxes)


def box_aspect(boxes: np.ndarray) -> np.ndarray:
    """Lățimea casetei împărțită la înălțimea ei."""
    return (boxes[:, 2] - boxes[:, 0]) / _heights(boxes)


@dataclass(frozen=True)
class DirectionFeatures:
    angle_deg: np.ndarray      # (N,)
    length_ratio: np.ndarray   # (N,)
    aspect: np.ndarray         # (N,)
    head_source: np.ndarray    # (N,) HEAD_NOSE / HEAD_NECK / HEAD_NONE


def direction_features(run: CP.ClipRun, threshold: float = CP.SCORE_MIN) -> DirectionFeatures:
    heads, sources = head_points(run.points, run.scores, threshold)
    tails = tail_points(run.points, run.scores, threshold)
    return DirectionFeatures(angle_deg=head_tail_angle(heads, tails), length_ratio=head_tail_length(heads, tails,
                             run.boxes), aspect=box_aspect(run.boxes), head_source=sources)
