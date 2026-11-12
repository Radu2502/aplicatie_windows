"""
Semnalele vederii pe ținta urmărită (călărețul ales de urmărire și calul lui), pentru routerul V5
(ecvestru/router_v5.py): aceleași formule ca în ecvestru/router_vedere.py (`rider_signals`, `horse_signals`), pe
cele 26 de puncte RTMPose-m ale țintei (mutate pe indicii MediaPipe) și pe AP-10K al calului țintă. Orientarea 3D cere
punctele world ale MediaPipe, pe care RTMPose nu le dă: rămâne NaN. Cadrele fără țintă au toate semnalele NaN.
Fără Streamlit.
"""
from __future__ import annotations

import numpy as np

from ecvestru import cal_puncte as CP
from ecvestru import router_vedere as RV
from ecvestru import urmarire_multi as U

# Halpe26 (RTMPose) → indicii MediaPipe folosiți de router_vedere.rider_signals
HALPE_TO_MEDIAPIPE = {RV.LEFT_SHOULDER: 5, RV.RIGHT_SHOULDER: 6, RV.LEFT_ELBOW: 7, RV.RIGHT_ELBOW: 8,
                      RV.LEFT_HIP: U.LEFT_HIP, RV.RIGHT_HIP: U.RIGHT_HIP}
MEDIAPIPE_POINTS = 33


def known_frames(choice: U.Choice) -> np.ndarray:
    """Cadrele cu un schelet al țintei (un om pe calul țintă)."""
    return np.array([r is not None for r in choice.rider_ids], bool)


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
