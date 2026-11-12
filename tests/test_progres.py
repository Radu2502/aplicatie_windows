"""Etapele analizei (ecvestru/progres.py): începutul fiecărei etape cu numărul ei, duratele adunate pe etapă,
textul barei și al duratelor. Ceasul e fals, deci testele nu depind de timp."""
import pytest

from ecvestru import progres as P


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_each_stage_reports_its_number_and_the_total_and_its_duration():
    clock, seen = FakeClock(), []
    stages = P.Stages(on_start=lambda name, k, total: seen.append((name, k, total)), clock=clock)
    stages.plan([P.SKELETON, P.MEASURE])
    stages.plan([P.FILM])                                   # pagina adaugă filmul; etapele nu se dublează
    stages.plan([P.SKELETON])
    with stages.stage(P.SKELETON):
        clock.now += 300.0
    with stages.stage(P.MEASURE):
        clock.now += 2.5
    with stages.stage(P.FILM):
        clock.now += 60.0
    assert seen == [(P.SKELETON, 0, 3), (P.MEASURE, 1, 3), (P.FILM, 2, 3)]
    assert stages.durations() == [P.StageTime(P.SKELETON, 300.0), P.StageTime(P.MEASURE, 2.5),
                                  P.StageTime(P.FILM, 60.0)]
    assert stages.total_seconds() == pytest.approx(362.5)


def test_a_stage_that_runs_twice_is_counted_once_and_its_times_add_up():
    clock, seen = FakeClock(), []
    stages = P.Stages(on_start=lambda name, k, total: seen.append((name, k, total)), clock=clock)
    stages.plan([P.HORSES, P.CAMERA, P.MEASURE])
    for name, seconds in ((P.CAMERA, 4.0), (P.MEASURE, 1.0), (P.CAMERA, 0.5)):
        with stages.stage(name):
            clock.now += seconds
    assert [k for _, k, _ in seen] == [0, 1]                  # a doua oară camera nu mai anunță nimic
    assert stages.durations() == [P.StageTime(P.CAMERA, 4.5), P.StageTime(P.MEASURE, 1.0)]


def test_a_plan_with_a_repeated_stage_counts_it_once():
    seen = []
    stages = P.Stages(on_start=lambda name, k, total: seen.append(total), clock=FakeClock())
    stages.plan([P.CAMERA, P.MEASURE, P.CAMERA])
    with stages.stage(P.CAMERA):
        pass
    assert seen == [2]


def test_the_duration_is_kept_even_when_the_stage_fails():
    clock = FakeClock()
    stages = P.Stages(clock=clock)
    with pytest.raises(ValueError), stages.stage(P.TRACKING):
        clock.now += 1.0
        raise ValueError("click departe de cal")
    assert stages.durations() == [P.StageTime(P.TRACKING, 1.0)]


def test_without_a_plan_the_total_grows_with_the_stages():
    seen = []
    stages = P.Stages(on_start=lambda name, k, total: seen.append((k, total)), clock=FakeClock())
    with stages.stage(P.SKELETON):
        pass
    with stages.stage(P.MEASURE):
        pass
    assert seen == [(0, 1), (1, 2)]


def test_the_bar_moves_over_all_the_riders():
    assert P.fraction(0, 2, 0, 4) == 0.0
    assert P.fraction(0, 2, 2, 4) == pytest.approx(0.25)
    assert P.fraction(1, 2, 0, 4) == pytest.approx(0.5)
    assert P.fraction(1, 2, 3, 4) == pytest.approx(0.875)
    assert P.fraction(0, 1, 5, 4) == 1.0                    # niciodată peste capăt


def test_the_bar_text_names_the_rider_and_the_stage():
    assert P.bar_text("Călărețul 1", P.RTMPOSE, 4, 7) == f"Călărețul 1 · etapa 5 din 7: {P.RTMPOSE}…"
    assert P.bar_text("", P.SKELETON, 0, 3) == f"Etapa 1 din 3: {P.SKELETON}…"


def test_seconds_are_written_for_people():
    assert P.seconds_text(0.04) == "0.0 s"
    assert P.seconds_text(12.34) == "12.3 s"
    assert P.seconds_text(59.96) == "1 min 0 s"
    assert P.seconds_text(612.0) == "10 min 12 s"


def test_the_durations_text_has_every_stage_and_the_total():
    times = [P.StageTime(P.SKELETON, 300.0), P.StageTime(P.FILM, 62.5)]
    assert P.durations_text(times) == (f"Durata etapelor: {P.SKELETON} 5 min 0 s · {P.FILM} 1 min 2 s · "
                                       "total 6 min 2 s")
    assert P.durations_text([]) == P.FROM_CACHE


def test_when_only_the_film_ran_the_result_came_from_the_cache():
    """Din cache, analiza nu rulează nicio etapă; pagina face doar filmul. Mesajul spune că rezultatul e din cache."""
    assert P.durations_text([P.StageTime(P.FILM, 2.0)]) == f"{P.FROM_CACHE} {P.FILM}: 2.0 s."
    assert P.durations_text([P.StageTime(P.SKELETON, 1.0), P.StageTime(P.FILM, 2.0)]).startswith("Durata etapelor:")
