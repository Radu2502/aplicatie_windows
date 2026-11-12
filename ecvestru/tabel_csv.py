"""Adăugarea unui rând într-un CSV doar adăugat (etichete, ținte, click-uri, verdicte): un singur ajutor. Antetul se
scrie dacă fișierul lipsește sau e gol; altfel doar rândul. Octeții sunt cei ai lui csv.DictWriter (UTF-8, CRLF).
Fără Streamlit."""
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path


def append_row(path: Path, fields: Sequence[str], row: Mapping, extrasaction: str = "raise") -> None:
    """Adaugă `row` la `path`, cu coloanele `fields`; creează folderul și antetul când e nevoie."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction=extrasaction)
        if new:
            writer.writeheader()
        writer.writerow(row)
