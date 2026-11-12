"""Pagina „Analiză” pornește, arată doar clipurile dev, folosește intervalul din profil și afișează
rezultatul. MediaPipe e înlocuit de un predictor fals; rezultatele se scriu doar în foldere temporare."""
import csv
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ecvestru import calareti
from ecvestru import masurare as M
from ecvestru import raport as T
from ecvestru import stocare as ST
from ecvestru import suprapunere as S
from ecvestru.rezultat import HEEL_MEDIAPIPE

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "app" / "pages" / "analiza.py")
CLIPS = M.dev_videos()
pytestmark = pytest.mark.skipif(not CLIPS, reason="clipurile nu sunt locale")


def fake(path):
    return {i: {"points": {"shoulder": (110.0, 100.0), "hip": (100.0, 200.0), "heel": (110.0, 300.0),
                           "facing": 1.0}, "motiv": None} if i % 5 else {"points": None, "motiv": "nu_e_pe_cal"}
            for i in range(2000)}


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("ECVESTRU_RESULTS", str(tmp_path))
    monkeypatch.setattr(ST, "run_candidate", fake)
    # Predictorul fals nu are schelet complet, deci calea RTMPose (implicită) ar cădea pe linia de bază, cu
    # avertisment; testele de aici sunt despre restul paginii, pe linia de bază (căderea: test_app_rtmpose.py).
    monkeypatch.setattr(M, "APP_DEFAULT_HEEL_SOURCE", HEEL_MEDIAPIPE)
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_only_dev_clips_and_default_segment_from_profile(app):
    test_clips = {r["video"] for r in csv.DictReader((ROOT / "data" / "split.csv").open()) if r["set"] == "test"}
    options = app.selectbox(key="an-clip").options
    assert options and not set(options) & test_clips
    name = app.selectbox(key="an-clip").value
    n, fps = M.video_info(M.V.VIDEOS / name)
    a, b = M.default_segment(name, n)
    lo, hi = app.slider(key=f"an-seg-{name}").value
    assert (round(lo * fps), round(hi * fps)) == (a, b)


def test_analyze_shows_metrics_report_table_and_writes_json(app, tmp_path):
    app.radio(key="an-gait").set_value("trap").run()
    app.button(key="an-run").click().run()
    assert not app.exception, app.exception
    labels = [m.label for m in app.metric]
    assert labels == ["Trunchi (mediană)", "Picior (mediană)", "Măsurat: trunchi", "Măsurat: picior"]
    assert app.metric[0].value == "+5°" and app.metric[1].value == "-5°" and app.metric[2].value == "80%"
    text = " ".join(m.value for m in app.markdown)
    assert "Trunchiul: aplecat înainte cu aproximativ 5°" in text and "alura declarată: trap" in text
    assert any(s.value == "Nemăsurat" for s in app.subheader) and len(app.dataframe) == 1
    assert list((tmp_path / "analiza").glob("*__trap__q2__v1.4.json"))         # rezultatul e salvat ca JSON
    assert "Precizia unghiurilor" in app.info[-1].value
    assert not app.warning                                                 # raportul a trecut verificările


def analyzed_text(app):
    app.button(key="an-run").click().run()
    assert not app.exception, app.exception
    return " ".join(m.value for m in app.markdown)


def assert_only_report_hidden(app, tmp_path, text):
    assert len(app.warning) == 1 and "Raportul text nu e afișat" in app.warning[0].value
    assert "Trunchiul" not in text and "Clipul" not in text
    assert len(app.metric) == 4 and len(app.dataframe) == 1                # valorile și tabelul rămân
    assert app.get("vega_lite_chart") and app.get("download_button")       # graficul și descărcarea rămân
    assert "Precizia unghiurilor" in app.info[-1].value
    assert list((tmp_path / "analiza").glob("*.json"))


def test_report_hidden_on_unknown_number(app, tmp_path, monkeypatch):
    monkeypatch.setattr(T, "report_text", lambda r: "Clipul are trunchiul la 999999.5°.")
    assert_only_report_hidden(app, tmp_path, analyzed_text(app))


def test_report_hidden_on_forbidden_label(app, tmp_path, monkeypatch):
    original = T.report_text
    monkeypatch.setattr(T, "report_text", lambda r: original(r).replace("Trunchiul:", "Trunchiul, perfect:"))
    assert_only_report_hidden(app, tmp_path, analyzed_text(app))


def test_report_hidden_on_forbidden_label_on_first_line_outside_clip_name(app, tmp_path, monkeypatch):
    original = T.report_text
    monkeypatch.setattr(T, "report_text", lambda r: original(r).replace("alura declarată", "alură perfect declarată"))
    assert_only_report_hidden(app, tmp_path, analyzed_text(app))


def test_report_shown_when_clip_name_contains_forbidden_label(app, monkeypatch):
    original = M.analyze
    monkeypatch.setattr(M, "analyze",
                        lambda *a, **k: original(*a, **k).model_copy(update={"video": "perfect_trot.mp4"}))
    text = analyzed_text(app)
    assert not app.warning and "Clipul perfect_trot.mp4" in text and "Trunchiul" in text


def test_result_json_is_shown_in_an_expander(app):
    app.button(key="an-run").click().run()
    assert not app.exception, app.exception
    assert [e.label for e in app.expander] == ["Rezultatul (JSON)"]
    shown = json.loads(app.expander[0].json[0].value)
    assert shown == json.loads(app.session_state["an-result"][1].to_json())


def test_film_with_legend_is_shown_after_analysis(app):
    app.button(key="an-run").click().run()
    assert not app.exception, app.exception
    assert len(app.get("video")) == 1
    assert any("verticala prin șold" in c.value for c in app.caption)


def test_failed_film_leaves_everything_else_on_the_page(app, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("Nu am putut face filmul.")

    monkeypatch.setattr(S, "clip_film", fail)
    app.button(key="an-run").click().run()
    assert not app.exception, app.exception
    assert not app.get("video") and len(app.warning) == 1 and "Filmul" in app.warning[0].value
    assert len(app.metric) == 4 and len(app.dataframe) == 1


def test_clip_with_unclear_target_shows_a_warning_before_analysis(app):
    unclear = next(iter(calareti.unclear_clips() & set(CLIPS)))
    app.selectbox(key="an-clip").set_value(unclear).run()
    assert len(app.warning) == 1 and "țintă neclară" in app.warning[0].value


def test_clip_with_clear_target_has_no_warning(app):
    clear = next(v for v in CLIPS if v not in calareti.unclear_clips())
    app.selectbox(key="an-clip").set_value(clear).run()
    assert not app.warning


def test_unusable_clip_shows_the_reason_before_any_result(app):
    clip = next(c for c in CLIPS if calareti.unusable_reason(c))
    app.selectbox(key="an-clip").set_value(clip).run()
    assert not app.exception, app.exception
    assert not app.metric
    assert any(calareti.unusable_reason(clip) in w.value for w in app.warning)


def test_usable_clip_has_no_unusable_warning(app):
    clip = next(c for c in CLIPS if calareti.unusable_reason(c) is None and c not in calareti.unclear_clips())
    app.selectbox(key="an-clip").set_value(clip).run()
    assert not app.warning


def test_unevaluated_clip_shows_the_neutral_text(app):
    clip = next(c for c in CLIPS if c not in M.V.evaluated_clips())
    app.selectbox(key="an-clip").set_value(clip).run()
    app.button(key="an-run").click().run()
    assert not app.exception, app.exception
    assert app.info[-1].value == M.LIMITATION_UNVERIFIED


def test_upload_page_says_what_stays_on_disk(app):
    app.radio(key="an-source").set_value("upload").run()
    assert app.info[0].value.endswith(M.UPLOAD_KEPT)


def test_the_legend_of_the_film_explains_the_visibility_tiers(app):
    app.button(key="an-run").click().run()
    assert any(S.VISIBILITY_LEGEND in c.value for c in app.caption)


def test_the_legend_of_the_film_describes_rejected_people(app):
    app.button(key="an-run").click().run()
    assert any(S.REJECTED_LEGEND in c.value for c in app.caption)


def test_the_legend_of_the_film_explains_the_two_sides(app):
    app.button(key="an-run").click().run()
    assert any(S.SIDE_LEGEND in c.value for c in app.caption)
