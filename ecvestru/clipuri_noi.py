"""
Clipurile din clipuri-noi/: inventarul fără imagini, cititorul exact de cadre și umplerea cache-ului scheletelor
(MediaPipe, apoi RTMPose). Umplerea cache-ului raportează doar timpul și erorile. Fără Streamlit.

Rulare:
  uv run python -m ecvestru.clipuri_noi inventariaza        # scrie inventarul clipurilor
  uv run python -m ecvestru.clipuri_noi umple --procese 4   # cache-ul MediaPipe + RTMPose, doar timpul pe clip
  uv run python -m ecvestru.clipuri_noi umple-cal --procese 6   # casetele calului + punctele AP-10K, doar timpul
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from ecvestru import masurare as M
from ecvestru import rtmpose
from ecvestru import stocare as ST

ROOT = ST.ROOT
INVENTORY_FILE = "inventar-calareti.csv"
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".avi")
RIDER_COLUMNS = ("calaret_1", "calaret_2")
RIDER_CODE_MAX = 12           # caractere într-un cod de călăreț (ex. C1)
RIDER_FORBIDDEN = ",;"         # separatorii din CSV și din coloana `calareti` a împărțirii
NOT_FILLED = "necompletat în CSV"
NO_SKELETON = "scheletul MediaPipe lipsește: RTMPose nu are caseta călărețului"
INVENTORY_DOC = ROOT / "docs" / "inventar-clipuri-noi.md"
CACHE_DOC = ROOT / "docs" / "clipuri-noi-cache.md"
HORSE_CACHE_DOC = ROOT / "docs" / "clipuri-noi-cache-cal.md"
DEFAULT_WORKERS = 4
SECONDS_PER_MINUTE = 60


def clips_dir() -> Path:
    return Path(os.environ.get("ECVESTRU_CLIPURI_NOI", ROOT / "clipuri-noi"))


def list_clips(folder: Path | None = None) -> list[str]:
    folder = clips_dir() if folder is None else folder
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Probe(_Model):
    frames: int               # numărate citind (indicele OpenCV), nu din metadate
    fps: float
    width: int                # mărimea cadrului decodat (OpenCV aplică rotația din metadate)
    height: int

    @property
    def seconds(self) -> float:
        return self.frames / self.fps


class ClipInfo(_Model):
    nr: int | None
    clip: str
    frames: int
    fps: float
    width: int
    height: int
    seconds: float
    riders: tuple[str, ...]


class CacheRun(_Model):
    clip: str
    seconds: float
    rtmpose: bool
    error: str | None


# ---------- inventarul ----------

def probe(path: Path) -> Probe:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ok, first = cap.read()
        if not ok:
            raise ValueError(f"Clipul nu poate fi citit: {Path(path).name}")
        frames = 1
        while cap.grab():
            frames += 1
    finally:
        cap.release()
    return Probe(frames=frames, fps=float(fps), width=first.shape[1], height=first.shape[0])


def _csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_riders(csv_path: Path) -> dict[str, tuple[str, ...]]:
    """Călăreții fiecărui clip, din CSV-ul de inventar; celulele goale nu contează."""
    return {r["clip"]: tuple(r[c].strip() for c in RIDER_COLUMNS if (r.get(c) or "").strip())
            for r in _csv_rows(csv_path)}


def save_riders(clip: str, first: str, second: str, csv_path: Path | None = None) -> tuple[str, ...]:
    """Scrie codurile călăreților clipului în CSV-ul de inventar (doar cele două celule; restul rămâne la fel).
    Refuză un cod prea lung sau cu separatori, al doilea fără primul, același cod de două ori, un clip lipsă."""
    csv_path = clips_dir() / INVENTORY_FILE if csv_path is None else Path(csv_path)
    codes = (first.strip(), second.strip())
    for code in codes:
        if len(code) > RIDER_CODE_MAX or any(c in code for c in RIDER_FORBIDDEN):
            raise ValueError(f"Codul „{code}” nu e bun: cel mult {RIDER_CODE_MAX} caractere, fără virgulă sau „;”.")
    if codes[1] and not codes[0]:
        raise ValueError("Completează întâi primul călăreț.")
    if codes[1] and codes[0].casefold() == codes[1].casefold():
        raise ValueError("Cei doi călăreți au același cod.")
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    row = next((r for r in rows if r["clip"] == clip), None)
    if row is None:
        raise ValueError(f"Clipul {clip} nu e în {csv_path.name}.")
    row.update(dict(zip(RIDER_COLUMNS, codes, strict=True)))
    temporary = csv_path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return tuple(c for c in codes if c)


def inventory(folder: Path | None = None) -> list[ClipInfo]:
    folder = clips_dir() if folder is None else folder
    rows = _csv_rows(folder / INVENTORY_FILE)
    numbers = {r["clip"]: int(r["nr"]) for r in rows if (r.get("nr") or "").strip()}
    riders = read_riders(folder / INVENTORY_FILE)
    out = []
    for clip in list_clips(folder):
        info = probe(folder / clip)
        out.append(ClipInfo(nr=numbers.get(clip), clip=clip, frames=info.frames, fps=info.fps, width=info.width,
                            height=info.height, seconds=info.seconds, riders=riders.get(clip, ())))
    return out


def inventory_markdown(rows: list[ClipInfo]) -> str:
    lines = ["| nr | clip | durata (s) | fps | rezoluție | cadre | călăreț |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        nr = "–" if r.nr is None else str(r.nr)
        rider = ", ".join(r.riders) if r.riders else NOT_FILLED
        lines.append(f"| {nr} | {r.clip} | {r.seconds:.1f} | {r.fps:.2f} | {r.width}×{r.height} | {r.frames} | "
                     f"{rider} |")
    frames = sum(r.frames for r in rows)
    minutes = sum(r.seconds for r in rows) / SECONDS_PER_MINUTE
    lines += ["", f"Total: {len(rows)} clipuri, {frames} cadre, {minutes:.1f} min."]
    return "\n".join(lines) + "\n"


INVENTORY_INTRO = """# Inventarul clipurilor noi

Generat cu `uv run python -m ecvestru.clipuri_noi inventariaza`, fără să se privească vreo imagine. Clipurile
stau în `clipuri-noi/` (ignorat de git).

- **Cadre:** numărate citind clipul cu OpenCV (indicele folosit peste tot în proiect), nu din metadate.
- **fps:** cel raportat de OpenCV; durata = cadre / fps.
- **Rezoluția:** a cadrului decodat (OpenCV aplică rotația din metadate).
- **Călărețul:** din `clipuri-noi/inventar-calareti.csv` (coloanele `calaret_1`, `calaret_2`).
- Inventarul nu calculează și nu raportează nicio măsurătoare pe aceste clipuri.

"""


def inventory_doc(rows: list[ClipInfo]) -> str:
    return INVENTORY_INTRO + inventory_markdown(rows)


# ---------- cititorul exact de cadre ----------

class FrameReader:
    """Cadrul cu indicele OpenCV, citit secvențial (exact). Saltul cu CAP_PROP_POS_FRAMES greșește cu 1–2 cadre pe
    clipurile noi (frecvență variabilă), deci se citește înainte și se redeschide clipul doar la întoarcere."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None
        self._next = 0          # indicele cadrului pe care îl dă următorul grab()
        self._last: tuple[int, np.ndarray] | None = None

    def _open(self) -> None:
        self.close()
        self._cap = cv2.VideoCapture(str(self.path))
        self._next = 0

    def read(self, index: int) -> np.ndarray:
        if index < 0:
            raise ValueError(f"Cadrul {index} nu există în clip.")
        if self._last is not None and self._last[0] == index:
            return self._last[1]
        if self._cap is None or index < self._next:
            self._open()
        while self._next <= index:
            if not self._cap.grab():
                self._open()
                raise ValueError(f"Cadrul {index} nu există în clip.")
            self._next += 1
        ok, image = self._cap.retrieve()
        if not ok:
            raise ValueError(f"Cadrul {index} nu poate fi decodat.")
        self._last = (index, image)
        return image

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------- cache-ul: MediaPipe, apoi RTMPose; doar timpul și erorile ----------

def prepare_rtmpose_points(path: Path, name: str, video_hash: str, skeleton) -> None:
    problem = M.rtmpose_problem()
    if problem is not None:
        raise RuntimeError(problem)
    rtmpose.load_points(path, name, video_hash, skeleton)


def fill_cache(path: Path) -> CacheRun:
    """Scheletul MediaPipe (brut/ și schelet/) și punctele RTMPose (rtmpose/), din cache sau calculate. Nu întoarce
    nicio valoare din ele: doar cât a durat și ce eroare a apărut."""
    path = Path(path)
    start = time.perf_counter()
    rtm, error = False, None
    try:
        video_hash, candidate = ST.file_hash(path), ST.candidate_label()
        ST.load_raw(path, path.name, video_hash, candidate)
        skeleton = ST.load_rider_skeleton(path, path.name, video_hash, candidate)
        if skeleton is None:
            error = NO_SKELETON
        else:
            prepare_rtmpose_points(path, path.name, video_hash, skeleton)
            rtm = True
    except Exception as exc:  # o eroare pe un clip nu oprește celelalte; se raportează
        error = f"{type(exc).__name__}: {exc}"
    return CacheRun(clip=path.name, seconds=time.perf_counter() - start, rtmpose=rtm, error=error)


def cache_markdown(runs: list[CacheRun], workers: int) -> str:
    lines = [f"Rulare cu {workers} procese în paralel (timpul pe clip include concurența).", "",
             "| clip | timp (s) | RTMPose | eroare |", "|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r.clip} | {r.seconds:.0f} | {'da' if r.rtmpose else 'nu'} | {r.error or '–'} |")
    return "\n".join(lines) + "\n"


CACHE_INTRO = """# Cache-ul scheletelor pe clipurile noi

Rulat cu `uv run python -m ecvestru.clipuri_noi umple`, de pe commitul `{commit}`: scheletul MediaPipe (candidatul;
`data/results/analiza/brut/` și `schelet/`), apoi punctele RTMPose (`data/results/analiza/rtmpose/`). Doar ca să
umple cache-ul: nu s-a calculat și nu se raportează nicio măsurătoare. Timpul e de perete, pe clip, cu procesele
rulând în paralel pe aceeași mașină.

"""


def cache_doc(runs: list[CacheRun], workers: int, commit: str) -> str:
    total = sum(r.seconds for r in runs)
    return (CACHE_INTRO.format(commit=commit) + cache_markdown(runs, workers)
            + f"\nTotal: {total:.0f} s de calcul pe clipuri; erori: {sum(r.error is not None for r in runs)}.\n")


# ---------- cache-ul calului: casetele, apoi punctele AP-10K; doar cadrele, timpul și erorile ----------

class HorseCacheRun(_Model):
    """O rulare a cache-ului calului pe un clip: câte cadre are, pe câte s-a rulat modelul (cu casetă), timpul și
    eroarea. Nicio valoare din puncte."""
    clip: str
    frames: int
    with_box: int
    seconds: float
    error: str | None


def rider_hips(skeleton) -> np.ndarray:
    from ecvestru import calitate as C
    return C.midpoint(skeleton, "hip")


def prepare_horse_points(path: Path, name: str, video_hash: str, boxes: dict, hips):
    from ecvestru import cal_puncte as CP
    return CP.load_points(path, name, video_hash, boxes, hips)


def fill_horse_cache(path: Path) -> HorseCacheRun:
    """Casetele calului (cai/) și punctele AP-10K (cal/), din cache sau calculate. Caseta e cea de sub șoldurile
    călărețului din cache-ul MediaPipe (fără schelet: cea mai mare), ca pe clipurile dev."""
    path = Path(path)
    start = time.perf_counter()
    try:
        video_hash = ST.file_hash(path)
        boxes = ST.load_boxes(path, path.name, video_hash)
        skeleton = ST.load_rider_skeleton(path, path.name, video_hash, ST.candidate_label())
        hips = None if skeleton is None else rider_hips(skeleton)
        run = prepare_horse_points(path, path.name, video_hash, boxes, hips)
        frames, with_box, error = len(run.boxes), run.inferred, None
    except Exception as exc:  # o eroare pe un clip nu oprește celelalte; se raportează
        frames, with_box, error = 0, 0, f"{type(exc).__name__}: {exc}"
    return HorseCacheRun(clip=path.name, frames=frames, with_box=with_box, seconds=time.perf_counter() - start,
                         error=error)


HORSE_CACHE_INTRO = """# Cache-ul calului pe clipurile noi

Rulat cu `uv run python -m ecvestru.clipuri_noi umple-cal`, de pe commitul `{commit}`: casetele calului
(`data/results/analiza/cai/`), apoi punctele AP-10K (`data/results/analiza/cal/`). Doar ca să umple cache-ul: nu s-a
calculat și nu se raportează nicio măsurătoare. Timpul e de perete, pe clip, cu
procesele rulând în paralel pe aceeași mașină.

"""


def horse_cache_doc(runs: list[HorseCacheRun], workers: int, commit: str) -> str:
    lines = [f"Rulare cu {workers} procese în paralel (timpul pe clip include concurența).", "",
             "| clip | cadre | cadre cu casetă (procesate) | timp (s) | eroare |", "|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r.clip} | {r.frames} | {r.with_box} | {r.seconds:.0f} | {r.error or '–'} |")
    total = sum(r.seconds for r in runs)
    return (HORSE_CACHE_INTRO.format(commit=commit) + "\n".join(lines) + "\n"
            + f"\nTotal: {sum(r.frames for r in runs)} cadre, {sum(r.with_box for r in runs)} procesate, "
            f"{total:.0f} s de calcul pe clipuri; erori: {sum(r.error is not None for r in runs)}.\n")


def _run_all(folder: Path, workers: int, job: Callable) -> list:
    paths = [folder / c for c in list_clips(folder)]
    # Cele mai lungi întâi, ca procesele să termine aproape deodată.
    paths.sort(key=lambda p: p.stat().st_size, reverse=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = []
        for run in pool.map(job, paths):
            print(f"{run.clip}: {run.seconds:.0f} s{'' if run.error is None else ' – ' + run.error}", flush=True)
            runs.append(run)
    return sorted(runs, key=lambda r: r.clip)


def fill_all_horses(folder: Path, workers: int) -> list[HorseCacheRun]:
    return _run_all(folder, workers, fill_horse_cache)


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def fill_all(folder: Path, workers: int) -> list[CacheRun]:
    return _run_all(folder, workers, fill_cache)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ecvestru.clipuri_noi", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventariaza", help="durata, fps, rezoluția, cadrele și călărețul, fără imagini")
    fill = commands.add_parser("umple", help="cache-ul MediaPipe + RTMPose; raportează doar timpul și erorile")
    fill.add_argument("--procese", type=int, default=DEFAULT_WORKERS)
    fill_horse = commands.add_parser("umple-cal", help="casetele calului + punctele AP-10K; doar timpul și erorile")
    fill_horse.add_argument("--procese", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    if args.command == "umple-cal":
        horse_runs = fill_all_horses(clips_dir(), args.procese)
        HORSE_CACHE_DOC.write_text(horse_cache_doc(horse_runs, args.procese, _commit()), encoding="utf-8")
        print(HORSE_CACHE_DOC)
        return 0
    if args.command == "inventariaza":
        INVENTORY_DOC.write_text(inventory_doc(inventory()), encoding="utf-8")
        print(INVENTORY_DOC)
    else:
        runs = fill_all(clips_dir(), args.procese)
        CACHE_DOC.write_text(cache_doc(runs, args.procese, _commit()), encoding="utf-8")
        print(cache_markdown(runs, args.procese))
    return 0


if __name__ == "__main__":
    sys.exit(main())
