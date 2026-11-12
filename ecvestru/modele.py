"""
Unde se caută modelele mari care nu stau în repo (AP-10K pentru cal, RTMPose-l): întâi folderul din variabila de
mediu ECVESTRU_MODELE, apoi ~/modele-ecvestru/, apoi calea veche din proiect (models/). În fiecare loc, modelul e în
subfolderul lui (`ap10k/`, `rtmpose_l/`), oricât de adânc. Fără Streamlit.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

ENV_VAR = "ECVESTRU_MODELE"
HOME_FOLDER = "modele-ecvestru"
NOT_FOUND = ("Modelul {label} nu a fost găsit. L-am căutat (fișierul {filename}) în: {places}. "
             "Pune folderul `{subdir}/` într-unul din aceste locuri (de exemplu în ~/{home}/) sau setează "
             "{env} cu folderul care conține `{subdir}/`.")
DOWNLOAD_FAILED = " Descărcarea automată în {old} nu a mers: {error}"


class ModelNotFound(FileNotFoundError):
    """Modelul lipsește din toate locurile căutate (și descărcarea, dacă s-a încercat, n-a mers)."""


def search_dirs(subdir: str, old_dir: Path) -> list[Path]:
    """Locurile, în ordinea căutării: ECVESTRU_MODELE/subdir (dacă e setată), ~/modele-ecvestru/subdir, calea veche."""
    dirs = []
    if env := os.environ.get(ENV_VAR):
        dirs.append(Path(env).expanduser() / subdir)
    dirs.append(Path.home() / HOME_FOLDER / subdir)
    dirs.append(Path(old_dir))
    return dirs


def find_model(subdir: str, filename: str, old_dir: Path) -> Path | None:
    """Primul fișier `filename` găsit, în ordinea din search_dirs; None dacă nu e nicăieri."""
    for folder in search_dirs(subdir, old_dir):
        found = sorted(folder.glob(f"**/{filename}")) if folder.is_dir() else []
        if found:
            return found[0]
    return None


def not_found_message(label: str, subdir: str, filename: str, old_dir: Path) -> str:
    places = "; ".join(str(d) for d in search_dirs(subdir, old_dir))
    return NOT_FOUND.format(label=label, filename=filename, places=places, subdir=subdir, home=HOME_FOLDER,
                            env=ENV_VAR)


def locate(label: str, subdir: str, filename: str, old_dir: Path, download: Callable[[], Path] | None = None
           ) -> Path:
    """Modelul găsit; altfel, dacă `download` e dat, descărcarea în models/ din proiect;
    altfel, sau dacă descărcarea nu merge, ModelNotFound cu toate locurile căutate."""
    found = find_model(subdir, filename, old_dir)
    if found is not None:
        return found
    message = not_found_message(label, subdir, filename, old_dir)
    if download is None:
        raise ModelNotFound(message)
    try:
        return download()
    except (OSError, ValueError) as error:
        raise ModelNotFound(message + DOWNLOAD_FAILED.format(old=old_dir, error=error)) from error
