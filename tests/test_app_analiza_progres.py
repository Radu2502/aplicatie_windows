"""Pagina „Analiză”: bara de progres pe etape (pe fiecare călăreț, apoi filmul), durata fiecărei etape după analiză și
saltul la primul cadru în care ținta s-a pierdut, ca omul să mai dea un click (ancorele existente). Clipuri sintetice,
predictori falși, foldere temporare (ca în test_app_analiza_intrari.py)."""
import streamlit
from test_app_analiza_intrari import alternating_points, boxes, page, synthetic_clip, upload  # noqa: F401

from ecvestru import masurare as M
from ecvestru import progres as P
from ecvestru import stocare as ST
from ecvestru import tinta as T
from ecvestru.rezultat import TargetSeed


class Bar:
    def __init__(self, log: list):
        self.log = log

    def progress(self, value, text=None):
        self.log.append((value, text))

    def empty(self):
        self.log.append("gol")


def record_bar(monkeypatch) -> list:
    log: list = []

    def progress(value=0, text=None, **kwargs):
        log.append((value, text))
        return Bar(log)

    monkeypatch.setattr(streamlit, "progress", progress)
    return log


def two_riders(page, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ST, "run_candidate", alternating_points)
    monkeypatch.setattr(ST, "detect_horses", boxes)
    data = synthetic_clip(tmp_path / "a.mp4", 20, seed=1)
    upload(page, "doi.mp4", data)
    page.checkbox(key="an-pick").check().run()
    page.session_state["an-targets"] = {M.bytes_hash(data): {
        "Călărețul 1": [TargetSeed(frame=0, x=120.0, y=100.0)], "Călărețul 2": [TargetSeed(frame=0, x=520.0, y=100.0)]}}
    page.run()
    return page


def test_the_bar_goes_through_the_stages_of_each_rider_then_disappears(page, tmp_path, monkeypatch):  # noqa: F811
    at = two_riders(page, tmp_path, monkeypatch)
    log = record_bar(monkeypatch)
    at.button(key="an-run").click().run()
    assert not at.exception, at.exception
    texts = [text for item in log if item != "gol" for _, text in [item] if text]
    assert any(t.startswith("Călărețul 1 · etapa 1 din ") and P.SKELETON in t for t in texts)
    assert any(t.startswith("Călărețul 2 · etapa ") and P.TRACKING in t for t in texts)
    assert sum(P.FILM in t for t in texts) == 2                              # filmul fiecărui călăreț e o etapă
    values = [item[0] for item in log if item != "gol"]
    assert values == sorted(values) and values[-1] < 1.0 and log[-1] == "gol"
    assert min(v for (v, t) in (i for i in log if i != "gol") if t and t.startswith("Călărețul 2")) >= 0.5


def test_after_the_analysis_each_rider_shows_how_long_each_stage_took(page, tmp_path, monkeypatch):  # noqa: F811
    at = two_riders(page, tmp_path, monkeypatch)
    at.button(key="an-run").click().run()
    assert not at.exception, at.exception
    lines = [c.value for c in at.caption if c.value.startswith("Durata etapelor:")]
    assert len(lines) == 2 and all(P.SKELETON in line and P.FILM in line and "total" in line for line in lines)


def test_a_result_from_the_cache_says_so_instead_of_durations(page, tmp_path, monkeypatch):  # noqa: F811
    at = two_riders(page, tmp_path, monkeypatch)
    at.button(key="an-run").click().run()
    at.button(key="an-run").click().run()                                  # a doua oară: din cache, doar filmul
    assert not at.exception, at.exception
    assert not [c.value for c in at.caption if c.value.startswith("Durata etapelor:")]
    lines = [c.value for c in at.caption if c.value.startswith(P.FROM_CACHE)]
    assert len(lines) == 2 and all(P.SKELETON not in line and P.FILM in line for line in lines)


def lost_clip(page, tmp_path, monkeypatch):  # noqa: F811
    """Calul se vede tot timpul, dar de la cadrul 2 caseta lui e de două ori mai mare (ca pe un manej rotund, când calul
    vine spre cameră): urmărirea nu-l mai recunoaște și, după golul maxim, ținta e pierdută până la capăt."""
    n = 2 * T.max_gap_frames(30.0) + 20
    horse, near = (70.0, 60.0, 100.0, 80.0), (20.0, 20.0, 200.0, 160.0)       # același centru, (120, 100)
    monkeypatch.setattr(ST, "run_candidate", alternating_points)
    monkeypatch.setattr(ST, "detect_horses", lambda path: {f: [horse] if f < 2 else [near] for f in range(n)})
    data = synthetic_clip(tmp_path / "a.mp4", n, seed=1)
    upload(page, "doi.mp4", data)
    page.checkbox(key="an-pick").check().run()
    page.session_state["an-targets"] = {M.bytes_hash(data): {"Călărețul 1": [TargetSeed(frame=0, x=120.0, y=100.0)]}}
    page.run()
    page.button(key="an-run").click().run()
    assert not page.exception, page.exception
    return page, M.bytes_hash(data), n


def test_first_lost_frame_is_the_start_of_the_first_lost_interval():
    from ecvestru.rezultat import AnalysisResult
    result = AnalysisResult.model_construct(unmeasured=[])
    assert M.first_lost_frame(result) is None


def test_the_jump_opens_the_first_lost_frame_for_the_same_rider(page, tmp_path, monkeypatch):  # noqa: F811
    at, content, n = lost_clip(page, tmp_path, monkeypatch)
    result = at.session_state["an-result"][2][0][1]
    first = M.lost_target_intervals(result)[0][0]
    assert M.first_lost_frame(result) == first and 0 < first < n - 1
    at.session_state["an-slot"] = "Călărețul 2"                         # alt slot ales înainte: saltul îl schimbă
    at.button(key="an-jump-Călărețul 1").click().run()
    assert not at.exception, at.exception
    assert at.session_state["an-slot"] == "Călărețul 1"
    assert at.slider(key=f"an-pick-frame-{content}").value == first
    assert any(f"cadrul {first}" in i.value and "Călărețul 1" in i.value for i in at.info)


def test_one_more_click_on_the_jumped_frame_becomes_an_anchor(page, tmp_path, monkeypatch):  # noqa: F811
    at, content, n = lost_clip(page, tmp_path, monkeypatch)
    at.button(key="an-jump-Călărețul 1").click().run()
    frame = at.slider(key=f"an-pick-frame-{content}").value
    seeds = at.session_state["an-targets"][content]["Călărețul 1"]
    at.session_state["an-targets"] = {content: {"Călărețul 1": [*seeds, TargetSeed(frame=frame, x=120.0, y=100.0)]}}
    at.run()
    at.button(key="an-run").click().run()
    assert not at.exception, at.exception
    result = at.session_state["an-result"][2][0][1]
    assert [a.frame for a in result.anchors] == [frame]                  # analiza a folosit click-ul ca ancoră
    assert M.lost_target_intervals(result) == []                         # și ținta nu se mai pierde
