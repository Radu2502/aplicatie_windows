"""Filmele din teste nu rulează MediaPipe: fără schelet în cache, desenează doar lanțul."""
from pathlib import Path

import pytest

from ecvestru import analiza_urmarita as AU
from ecvestru import stocare as ST

# Testele paginii „Analiză” pe clipurile de dezvoltare citesc data/split.csv la import; fără date, se ignoră.
NEEDS_DEV_DATA = ["test_app_analiza.py", "test_app_analiza_intrari.py", "test_app_analiza_progres.py",
                  "test_app_analiza_urmarita.py", "test_app_calcai_stabil.py", "test_app_rtmpose.py",
                  "test_calcai_stabil.py"]
collect_ignore = [] if (Path(__file__).resolve().parents[1] / "data" / "split.csv").exists() else NEEDS_DEV_DATA

if not ST.SPLIT_CSV.exists():
    # Doar în teste: fără setul de dezvoltare, lista clipurilor dev e goală, deci testele pe clipuri se sar.
    from ecvestru import masurare as M

    ST.dev_videos = M.dev_videos = lambda *args, **kwargs: []


@pytest.fixture(autouse=True)
def no_skeleton_extraction(monkeypatch):
    monkeypatch.setattr(ST, "extract_skeleton", lambda path: None)


@pytest.fixture(autouse=True)
def one_ridden_horse(request, monkeypatch):
    """Metoda automată din „Analiză” numără caii călăriți cu modelele de detecție; în teste, implicit 1 cal (deci
    metoda veche), fără model. Testele regulii însăși au marcajul `metoda_reala`."""
    if request.node.get_closest_marker("metoda_reala") is None:
        monkeypatch.setattr(AU, "method_choice", lambda *a, **k: AU.MethodChoice(1))


def pytest_configure(config):
    config.addinivalue_line("markers", "metoda_reala: fără înlocuirea numărării cailor călăriți (AU.method_choice)")
