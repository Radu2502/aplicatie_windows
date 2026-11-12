"""Modelul de date: JSON dus-întors, versiunea schemei, valori invalide respinse."""
import pytest
from pydantic import ValidationError

from ecvestru.rezultat import SCHEMA_VERSION, AnalysisResult, FrameMeasure, MetricSummary, UnmeasuredInterval


def example(**over) -> AnalysisResult:
    m = MetricSummary(n_frames=30, coverage=0.75, coverage_pct=75, median_deg=8.2, q25_deg=4.9, q75_deg=12.6,
                      median_rounded=10, q25_rounded=5, q75_rounded=15)
    base = dict(video="clip_x.mp4", video_hash="abc", gait="trap", fps=30.0, n_frames=100, segment=(10, 49),
                candidate="candidate@ff", trunk=m, leg=m,
                unmeasured=[UnmeasuredInterval(start=12, end=13, frames=2, reason="nu_e_pe_cal",
                                               reason_text="persoana găsită nu e pe cal")],
                frames=[FrameMeasure(frame=10, trunk_deg=8.2, leg_deg=-3.1)])
    return AnalysisResult(**(base | over))


def test_json_round_trip_and_schema_version():
    r = example()
    assert r.schema_version == SCHEMA_VERSION == "1.4"
    assert AnalysisResult.from_json(r.to_json()) == r
    assert "Precizia unghiurilor" in r.limitation


def test_invalid_values_are_rejected():
    with pytest.raises(ValidationError):
        example(gait="galop rapid")
    with pytest.raises(ValidationError):
        AnalysisResult.from_json(example().to_json().replace('"1.4"', '"2.0"', 1))
    with pytest.raises(ValidationError):
        MetricSummary(n_frames=1, coverage=1.5, coverage_pct=150)
    with pytest.raises(ValidationError):
        example(neașteptat=1)


def test_the_excluded_frames_description_says_what_is_counted_on_each_path():
    """Pe calea RTMPose câmpul numără excluderile fuziunii, nu ale indicatorului A (arhitectura, 2B.1.4)."""
    from ecvestru.rezultat import QualitySummary
    text = QualitySummary.model_fields["excluded_frames"].description
    assert "indicatorul A" in text and "fuziun" in text and "AP-10K" in text
