"""„Analiză” cu alegerea metodei: metoda implicită e cea din AU.APP_DEFAULT_METHOD, metoda veche rămâne
disponibilă, iar urmărirea țintei afișează rezultatul sau cere un click. Analiza urmărită e
înlocuită aici cu una falsă (din referință); nicio rulare de model."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ecvestru import analiza_urmarita as AU
from ecvestru import masurare as M
from ecvestru import suprapunere as S
from ecvestru.rezultat import AnalysisResult

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "app" / "pages" / "analiza.py")
REFERENCE = ROOT / "tests" / "referinte" / "urmarire" / "clip_150054.json"
pytestmark = pytest.mark.skipif(not M.dev_videos(), reason="clipurile dev nu sunt locale")


@pytest.fixture
def page(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_method_choice_starts_on_the_app_default(page):
    radio = page.radio(key="an-method")
    assert radio.value == AU.APP_DEFAULT_METHOD and set(radio.options) == set(AU.METHOD_LABELS.values())


def test_the_tracked_method_asks_for_a_click_when_the_target_is_not_clear(page, monkeypatch):
    def needs(*args, **kwargs):
        raise AU.TargetNeeded("doi călăreți")
    monkeypatch.setattr(AU, "analyze", needs)
    page.radio(key="an-method").set_value(AU.METHOD_TRACKED).run()
    assert any("Timp estimat" in c.value for c in page.caption)
    page.button(key="an-run").click().run()
    assert any(AU.CLICK_NEEDED_TEXT in w.value for w in page.warning) and not page.exception


def test_the_tracked_method_shows_the_result_on_the_tracked_rider(page, monkeypatch):
    result = AnalysisResult.model_validate(json.loads(REFERENCE.read_text(encoding="utf-8"))["json"])
    calls = []
    monkeypatch.setattr(AU, "analyze", lambda *a, **k: calls.append(k.get("click")) or result)
    page.radio(key="an-method").set_value(AU.METHOD_TRACKED).run()
    page.button(key="an-run").click().run()
    assert calls == [None] and not page.exception
    assert any("Filmul cu scheletul nu e încă disponibil" in c.value for c in page.caption)
    assert page.metric[0].label == "Trunchi (mediană)"


# ---------- metoda automată după numărul de cai călăriți ----------

def test_the_default_is_the_automatic_method(page):
    assert page.radio(key="an-method").value == AU.METHOD_AUTO


def test_two_ridden_horses_choose_the_tracking_and_ask_for_a_click(page, monkeypatch):
    def needs(*args, **kwargs):
        raise AU.TargetNeeded("doi călăreți")
    monkeypatch.setattr(AU, "method_choice", lambda *a, **k: AU.MethodChoice(2))
    monkeypatch.setattr(AU, "analyze", needs)
    monkeypatch.setattr(M, "analyze", lambda *a, **k: pytest.fail("metoda veche nu trebuia să ruleze"))
    page.button(key="an-run").click().run()
    assert not page.exception
    assert any("2 cai călăriți în primele 2 s → urmărirea țintei" in i.value for i in page.info)
    assert any(AU.CLICK_NEEDED_TEXT in w.value for w in page.warning)


@pytest.mark.parametrize("n_ridden", [0, 1])
def test_fewer_ridden_horses_keep_the_old_method(page, monkeypatch, n_ridden):
    reference = ROOT / "tests" / "referinte" / "curenta" / "clip_150054.json"
    result = AnalysisResult.model_validate(json.loads(reference.read_text(encoding="utf-8"))["json"])
    calls = []
    monkeypatch.setattr(AU, "method_choice", lambda *a, **k: AU.MethodChoice(n_ridden))
    monkeypatch.setattr(AU, "analyze", lambda *a, **k: pytest.fail("urmărirea nu trebuia să ruleze"))
    monkeypatch.setattr(M, "analyze", lambda *a, **k: calls.append(a) or result)
    monkeypatch.setattr(S, "clip_film", lambda *a, **k: None)
    page.button(key="an-run").click().run()
    assert len(calls) == 1 and not page.exception
    assert any(f"{AU.horses_text(n_ridden)} în primele 2 s → metoda veche" in i.value for i in page.info)


def test_a_manual_choice_says_so_and_skips_the_count(page, monkeypatch):
    monkeypatch.setattr(AU, "method_choice", lambda *a, **k: pytest.fail("fără numărare la alegerea manuală"))
    monkeypatch.setattr(AU, "analyze", lambda *a, **k: (_ for _ in ()).throw(AU.TargetNeeded("x")))
    page.radio(key="an-method").set_value(AU.METHOD_TRACKED).run()
    assert any("Metoda aleasă manual" in c.value for c in page.caption)
    page.button(key="an-run").click().run()
    assert not page.exception
