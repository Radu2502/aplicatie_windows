"""Semnalele vederii pe ținta urmărită (ecvestru/semnale_tinta.py): din RTMPose și AP-10K pe ținta aleasă, NaN fără
țintă. Date inventate, fără video real."""
import numpy as np
import pytest

from ecvestru import cal_puncte as CP
from ecvestru import router_vedere as RV
from ecvestru import semnale_tinta as SEM
from ecvestru import urmarire_multi as U


def rider(shoulder_dx: float) -> np.ndarray:
    points = np.zeros((26, 2), np.float32)
    points[5], points[6] = (100 + shoulder_dx, 100), (100 - shoulder_dx, 100)       # umerii stâng, drept
    points[7], points[8] = (100 + shoulder_dx, 150), (100 - shoulder_dx, 150)       # coatele
    points[11], points[12] = (100, 200), (100, 200)                                 # șoldurile
    return points


def test_signals_come_from_the_target_rider_and_horse_and_are_unknown_without_a_target():
    tracks = U.ClipTracks(fps=30.0, n_frames=3, horses=[{1: (0.0, 150.0, 200.0, 250.0)}] * 3,
                          people=[{4: (50.0, 50.0, 150.0, 250.0)}] * 3, keypoints=[{4: rider(20.0)}] * 3)
    choice = U.Choice([U.ON_C1, U.ON_C1, U.NO_SKELETON], [1, 1, 1], [4, 4, None], 0, None, 0, 0)
    xy = SEM.rider_xy(tracks, choice)
    assert xy.shape == (3, 33, 2) and tuple(xy[0, RV.LEFT_SHOULDER]) == (120.0, 100.0) and np.isnan(xy[2]).all()
    horse = CP.ClipRun(np.full((3, 17, 2), np.nan), np.full((3, 17), np.nan), SEM.target_boxes(tracks, [1, 1, 1]), 0.0)
    signals = SEM.target_signals(tracks, choice, horse)
    row = dict(zip(signals.names, signals.values[0], strict=True))
    assert row["raport_umeri"] == pytest.approx(0.4) and row["umeri_stanga_dreapta"] == pytest.approx(0.4)
    assert row["cal_caseta"] == pytest.approx(2.0) and np.isnan(row["orientare_cos"])       # fără world pe RTMPose
    assert np.isnan(signals.values[2]).all()                                                 # fără țintă
    assert list(SEM.known_frames(choice)) == [True, True, False]
