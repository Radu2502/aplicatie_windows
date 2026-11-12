"""Etapele analizei: anunță începutul fiecărei etape (pentru bara de progres) și îi măsoară durata. Fără Streamlit.

Doar observă: nu schimbă ordinea, parametrii sau rezultatul calculului, iar duratele nu intră în JSON-ul
rezultatului (acesta trebuie să iasă identic cu sau fără etape, tests/test_masurare_etape.py)."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

SKELETON = "Scheletul călărețului (MediaPipe)"
VIEW = "Vederea din profil"
HORSES = "Casetele cailor"
CAMERA = "Mișcarea camerei"
TRACKING = "Urmărirea țintei"
RTMPOSE = "RTMPose (călcâiul și sensul de mers)"
MEASURE = "Măsurarea și calitatea"
STABLE = "Călcâi stabil"
HORSE_SKELETON = "Scheletul calului (AP-10K)"
RIDER_VIEW = "Vederea călărețului (router V5)"
FILM = "Filmul"
FROM_CACHE = "Rezultat din cache: nicio etapă nu a rulat din nou."
SECONDS_PER_MINUTE = 60

OnStart = Callable[[str, int, int], None]      # (etapa, câte etape au început înaintea ei, câte sunt în total)


@dataclass(frozen=True)
class StageTime:
    name: str
    seconds: float


class Stages:
    """Etapele unei analize: lista (ca bara să știe totalul), începutul fiecăreia și durata, adunată pe etapă (o etapă
    care rulează de două ori, de exemplu mișcarea camerei, se numără o dată)."""

    def __init__(self, on_start: OnStart | None = None, clock: Callable[[], float] = time.perf_counter):
        self._on_start = on_start
        self._clock = clock
        self._planned: list[str] = []
        self._seconds: dict[str, float] = {}

    def plan(self, names: Iterable[str]) -> None:
        for name in names:
            if name not in self._planned:
                self._planned.append(name)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in self._seconds:
            started = len(self._seconds)
            self._seconds[name] = 0.0
            if self._on_start is not None:
                self._on_start(name, started, max(len(self._planned), len(self._seconds)))
        start = self._clock()
        try:
            yield
        finally:
            self._seconds[name] += self._clock() - start

    def durations(self) -> list[StageTime]:
        return [StageTime(name, seconds) for name, seconds in self._seconds.items()]

    def total_seconds(self) -> float:
        return sum(self._seconds.values())


def fraction(rider: int, riders: int, started: int, total: int) -> float:
    """Cât din bară e plin: călăreții se analizează pe rând, fiecare cu etapele lui."""
    return min(1.0, (rider + min(started, total) / max(total, 1)) / max(riders, 1))


def bar_text(label: str, name: str, started: int, total: int) -> str:
    step = f"tapa {started + 1} din {total}: {name}…"
    return f"{label} · e{step}" if label else f"E{step}"


def seconds_text(seconds: float) -> str:
    if round(seconds, 1) < SECONDS_PER_MINUTE:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(round(seconds), SECONDS_PER_MINUTE)
    return f"{minutes} min {rest} s"


def durations_text(times: list[StageTime]) -> str:
    """Duratele etapelor; dacă nu a rulat nicio etapă a analizei (rezultatul a venit din cache, eventual doar cu filmul
    refăcut de pagină), spune asta în loc de durate."""
    if not [t for t in times if t.name != FILM]:
        film = next((t for t in times if t.name == FILM), None)
        return FROM_CACHE if film is None else f"{FROM_CACHE} {FILM}: {seconds_text(film.seconds)}."
    parts = [f"{t.name} {seconds_text(t.seconds)}" for t in times]
    return f"Durata etapelor: {' · '.join(parts)} · total {seconds_text(sum(t.seconds for t in times))}"
