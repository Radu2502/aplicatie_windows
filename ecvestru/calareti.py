"""
Identitatea călărețului-țintă al fiecărui clip: descrierea din inventar și o imagine de referință
decupată (data/calareti.csv), ca să se știe pe cine trebuie să fie scheletul când în cadru sunt mai
mulți călăreți. Decupajele se scriu doar local, în data/results/calareti/ (nu intră în git).
"""
from __future__ import annotations

import csv
import os
import textwrap
from pathlib import Path

import cv2
import numpy as np

from ecvestru import inventar

ROOT = Path(__file__).resolve().parents[1]
RIDERS_CSV = ROOT / "data" / "calareti.csv"
CLIPS_CSV = ROOT / "data" / "clips.csv"
VIDEOS = ROOT / "data" / "videos"
NO_REASON = "motiv nenotat în inventar"


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def riders() -> dict[str, dict]:
    return {r["calaret"]: r for r in _read(RIDERS_CSV)}


def target(video: str) -> dict | None:
    """Călărețul-țintă al clipului (rândul din calareti.csv), după coloana `calaret` din clips.csv."""
    code = next((r["calaret"] for r in _read(CLIPS_CSV) if r["video"] == video), "")
    return riders().get(code.rstrip("?"))


def unusable_reason(video: str) -> str | None:
    """Motivul din inventar pentru un clip cu utilizabil=nu; None pentru celelalte."""
    row = next((r for r in _read(CLIPS_CSV) if r["video"] == video), None)
    if row is None or row["utilizabil"] != "nu":
        return None
    return row["motiv"] or NO_REASON


def unclear_clips() -> set[str]:
    """Clipurile în care nu e clar cine e călărețul-țintă (coloana `tinta_neclara`)."""
    return {v.strip() for r in riders().values() for v in r.get("tinta_neclara", "").split(";") if v.strip()}


def _cache_dir() -> Path:
    return Path(os.environ.get("ECVESTRU_RESULTS", ROOT / "data" / "results")) / "calareti"


def reference(code: str) -> np.ndarray:
    """Decupajul de referință (BGR) al călărețului, din cadrul și caseta din calareti.csv."""
    path = _cache_dir() / f"{code}.png"
    if path.exists():
        return cv2.imread(str(path))
    r = riders()[code]
    frame = int(r["frame"])
    img = inventar.read_frames(VIDEOS / r["video"], [frame])[frame]
    x0, y0, x1, y1 = (int(r[k]) for k in ("x0", "y0", "x1", "y1"))
    crop = img[y0:y1, x0:x1].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    return crop


def panel(code: str, height: int, width: int | None = None) -> np.ndarray:
    """Panoul „călărețul-țintă”: titlu, referința și descrierea (ASCII, fără codul călărețului)."""
    width = width or round(height * 0.75)
    out = np.full((height, width, 3), 40, np.uint8)
    lines = textwrap.wrap(riders()[code]["descriere"], 34)
    title_h, text_h = 28, 18 * len(lines) + 10
    ref = reference(code)
    avail_h = height - title_h - text_h
    s = min((width - 8) / ref.shape[1], avail_h / ref.shape[0])
    ref = cv2.resize(ref, (max(1, round(ref.shape[1] * s)), max(1, round(ref.shape[0] * s))),
                     interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    x = (width - ref.shape[1]) // 2
    out[title_h:title_h + ref.shape[0], x:x + ref.shape[1]] = ref
    cv2.putText(out, "TINTA (alt moment):", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (6, title_h + ref.shape[0] + 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out
