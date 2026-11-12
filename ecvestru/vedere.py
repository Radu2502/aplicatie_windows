"""
Vederea din profil: pe un cadru oblic, umerii apar lați față de trunchi. Raportul lățime umeri / înălțime
trunchi vine din scheletul complet al persoanei alese de candidat, dar se păstrează doar raportul pe cadru,
nu cele 33 de puncte. Pragul e fix. Fără Streamlit.
"""
from __future__ import annotations

import math

LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 11, 12, 23, 24
PROFILE_MAX_RATIO = 0.5      # taie 1% din cadrele corecte și 9% din cele greșite
REASON_NOT_PROFILE = "nu_e_din_profil"


def shoulder_ratio(landmarks, width: int, height: int) -> float:
    def pixels(index):
        return landmarks[index].x * width, landmarks[index].y * height

    left, right, left_hip, right_hip = (pixels(i) for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP))
    mid_shoulder = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    mid_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
    return abs(left[0] - right[0]) / max(math.dist(mid_shoulder, mid_hip), 1.0)


def shoulder_ratios(video_path, source) -> dict[int, float]:
    """Raportul pe fiecare cadru cu persoană aleasă. `source` e `_landmarks` din candidat, doar citit."""
    return {frame: shoulder_ratio(landmarks, width, height)
            for frame, landmarks, width, height, _ in source(str(video_path)) if landmarks}


def apply_profile(raw: dict, ratios: dict) -> dict:
    """Scoate cadrele oblice; un cadru fără răspuns își păstrează motivul."""
    out = {}
    for frame, answer in raw.items():
        ratio = ratios.get(frame)
        oblique = answer["points"] is not None and ratio is not None and ratio >= PROFILE_MAX_RATIO
        out[frame] = {"points": None, "motiv": REASON_NOT_PROFILE} if oblique else answer
    return out
