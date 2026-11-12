"""Etapele analizei (bara de progres și duratele) doar observă: JSON-ul rezultatului e identic cu și fără ele, pe
calea cu țintă și pe calea RTMPose cu „călcâi stabil”. Fiecare rulare pornește fără cache. Clipuri sintetice,
estimatori falși, foldere temporare."""
import shutil

from test_calcai_stabil import HorseOnHeel, SpikeEstimator, tmp_results  # noqa: F401 (tmp_results e fixture)
from test_calcai_stabil import analyze as analyze_rtmpose
from test_tinta import alternating, clip, seed  # noqa: F401 (clip e fixture)

from ecvestru import masurare as M
from ecvestru import progres as P


def fresh(results) -> None:
    shutil.rmtree(results, ignore_errors=True)


def recorder():
    seen = []
    return P.Stages(on_start=lambda name, k, total: seen.append((name, k, total))), seen


def test_with_a_target_the_json_is_the_same_with_and_without_stages(clip, tmp_path):  # noqa: F811
    plain = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(0, 100))
    fresh(tmp_path / "rezultate")
    stages, seen = recorder()
    observed = M.analyze(clip, "pas", (0, 19), predictor=alternating, target=seed(0, 100), stages=stages)
    assert observed.to_json() == plain.to_json()
    order = [P.SKELETON, P.VIEW, P.HORSES, P.CAMERA, P.TRACKING, P.MEASURE]
    assert seen == [(name, k, len(order)) for k, name in enumerate(order)]
    assert [t.name for t in stages.durations()] == order


def test_on_the_rtmpose_path_with_the_stable_heel_the_json_is_the_same(tmp_results, tmp_path, monkeypatch):  # noqa: F811
    results = tmp_results                                      # ECVESTRU_RESULTS în folderul temporar
    plain = analyze_rtmpose(tmp_path, monkeypatch, SpikeEstimator(spike=6), heel_stable=True,
                            horse_estimator=HorseOnHeel(3))
    fresh(results)
    stages, seen = recorder()
    observed = analyze_rtmpose(tmp_path, monkeypatch, SpikeEstimator(spike=6), heel_stable=True,
                               horse_estimator=HorseOnHeel(3), stages=stages)
    assert observed.to_json() == plain.to_json()
    assert [name for name, _, _ in seen] == [P.SKELETON, P.RTMPOSE, P.MEASURE, P.STABLE]


def test_a_cached_result_runs_no_stage(clip):  # noqa: F811
    M.analyze(clip, "pas", (0, 19), predictor=alternating)
    stages, seen = recorder()
    M.analyze(clip, "pas", (0, 19), predictor=alternating, stages=stages)
    assert seen == [] and P.durations_text(stages.durations()) == P.FROM_CACHE
