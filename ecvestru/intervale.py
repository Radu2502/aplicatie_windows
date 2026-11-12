"""
Intervalele de vedere pe clipuri: adnotatorul marchează primul și ultimul cadru (inclusiv) și vederea. Fișierul din
folderul adnotărilor e doar adăugat: o corectură sau o ștergere e un rând nou, cu același id; contează ultimul rând al
fiecărui (adnotator, id). Al doilea adnotator sau a doua completare nu suprascrie nimic. Fără Streamlit.
"""
from __future__ import annotations

import csv
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ecvestru import tabel_csv as TC
from ecvestru import verificare as V

ANSWER_FILE = "clipuri_noi_vedere.csv"
PROFILE, FRONT, BACK, OBLIQUE, HIDDEN = "profil", "fata", "spate", "oblic", "nu_se_vede"
VIEWS = {PROFILE: "Profil", FRONT: "Față", BACK: "Spate", OBLIQUE: "Oblic", HIDDEN: "Nu se vede"}
ADDED, CORRECTED, DELETED = "adaugat", "corectat", "sters"
FIELDS = ["id", "video", "start", "stop", "vedere", "actiune", "observatii", "adnotator", "data"]
ID_CHARS = 8

# Banda de timp a paginii: o coloană de pixeli pe fracțiune de clip, colorată după vedere
TIMELINE_WIDTH, TIMELINE_HEIGHT = 800, 18
VIEW_RGB = {PROFILE: (46, 139, 87), FRONT: (65, 105, 225), BACK: (218, 112, 214), OBLIQUE: (255, 165, 0),
            HIDDEN: (128, 128, 128)}
UNLABELED_RGB = (235, 235, 235)
CURRENT_RGB = (220, 20, 60)
VIEW_COLOR_NAMES = {PROFILE: "verde", FRONT: "albastru", BACK: "mov", OBLIQUE: "portocaliu", HIDDEN: "gri"}
TIMELINE_LEGEND = (" · ".join(f"{VIEWS[v]}: {VIEW_COLOR_NAMES[v]}" for v in VIEWS)
                   + " · neetichetat: alb · cadrul curent: roșu")


@dataclass(frozen=True, order=True)
class Interval:
    video: str
    start: int
    stop: int              # inclusiv
    view: str
    id: str
    note: str
    annotator: str

    @property
    def frames(self) -> int:
        return self.stop - self.start + 1


def answers_path() -> Path:
    return V.annotations_dir() / ANSWER_FILE


def same_annotator(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _latest(path: Path, annotator: str) -> dict[str, dict]:
    """Ultimul rând al fiecărui id al adnotatorului (inclusiv ștergerile)."""
    latest: dict[str, dict] = {}
    for row in read_rows(path):
        if same_annotator(row["adnotator"], annotator):
            latest[row["id"]] = row
    return latest


def _interval(row: dict) -> Interval:
    return Interval(video=row["video"], start=int(row["start"]), stop=int(row["stop"]), view=row["vedere"],
                    id=row["id"], note=row["observatii"], annotator=row["adnotator"])


def current_intervals(path: Path, annotator: str, video: str | None = None) -> list[Interval]:
    """Intervalele în vigoare ale adnotatorului (fără cele șterse), sortate după clip și început."""
    found = [_interval(r) for r in _latest(path, annotator).values() if r["actiune"] != DELETED]
    return sorted(i for i in found if video is None or i.video == video)


def annotators(path: Path) -> list[str]:
    """Adnotatorii care au cel puțin un interval în vigoare, cu numele din primul lor rând."""
    names: dict[str, str] = {}
    for row in read_rows(path):
        names.setdefault(row["adnotator"].strip().casefold(), row["adnotator"].strip())
    return sorted(name for name in names.values() if current_intervals(path, name))


def covered_frames(intervals: list[Interval]) -> int:
    return len({f for i in intervals for f in range(i.start, i.stop + 1)})


def view_at(intervals: list[Interval], frame: int) -> str | None:
    return next((i.view for i in intervals if i.start <= frame <= i.stop), None)


def timeline_image(intervals: list[Interval], n_frames: int, current: int, width: int = TIMELINE_WIDTH
                   ) -> np.ndarray:
    """Banda intervalelor pe tot clipul (RGB), cu cadrul curent marcat."""
    image = np.empty((TIMELINE_HEIGHT, width, 3), np.uint8)
    image[:] = UNLABELED_RGB
    column = np.arange(width) * n_frames // width          # primul cadru al fiecărei coloane
    for interval in intervals:
        image[:, (column >= interval.start) & (column <= interval.stop)] = VIEW_RGB[interval.view]
    image[:, min(current * width // n_frames, width - 1)] = CURRENT_RGB
    return image


def _append(path: Path, row: dict) -> None:
    TC.append_row(path, FIELDS, row)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save_interval(path: Path, video: str, start: int, stop: int, view: str, annotator: str, note: str = "",
                  interval_id: str | None = None, n_frames: int | None = None) -> Interval:
    """Adaugă un interval nou sau, cu `interval_id`, corectura unuia al aceluiași adnotator. Refuză adnotatorul gol,
    vederea necunoscută, capetele din afara clipului și suprapunerea cu alt interval în vigoare din același clip."""
    if not annotator.strip():
        raise ValueError("Completează numele adnotatorului.")
    if view not in VIEWS:
        raise ValueError("Alege o vedere din cele de pe pagină.")
    if start < 0 or stop < start or (n_frames is not None and stop >= n_frames):
        raise ValueError(f"Interval greșit: {start}–{stop} (începutul cel mult cât sfârșitul, ambele în clip).")
    latest = _latest(path, annotator)
    if interval_id is not None and (interval_id not in latest or latest[interval_id]["actiune"] == DELETED):
        raise ValueError("Intervalul de corectat nu e al tău sau a fost șters.")
    for other in current_intervals(path, annotator, video):
        if other.id != interval_id and other.start <= stop and start <= other.stop:
            raise ValueError(f"Intervalul se suprapune cu {other.start}–{other.stop} ({VIEWS[other.view]}). "
                             "Corectează-l pe acela sau alege alte capete.")
    action = ADDED if interval_id is None else CORRECTED
    interval_id = interval_id or uuid.uuid4().hex[:ID_CHARS]
    row = {"id": interval_id, "video": video, "start": start, "stop": stop, "vedere": view, "actiune": action,
           "observatii": note.strip(), "adnotator": annotator.strip(), "data": _now()}
    _append(path, row)
    return _interval(row)


def delete_interval(path: Path, interval_id: str, annotator: str) -> None:
    """Ștergerea e un rând nou; rândurile vechi rămân."""
    latest = _latest(path, annotator)
    if interval_id not in latest or latest[interval_id]["actiune"] == DELETED:
        raise ValueError("Intervalul de șters nu e al tău sau a fost deja șters.")
    _append(path, latest[interval_id] | {"actiune": DELETED, "adnotator": annotator.strip(), "data": _now()})
