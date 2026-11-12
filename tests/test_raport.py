"""Raportul din șabloane pe exemple fixe; orice număr din text există în JSON; fără etichete interzise."""
import pytest

from ecvestru import raport as T
from ecvestru.rezultat import AnalysisResult, MetricSummary, UnmeasuredInterval


def metric(med, lo, hi, n=30, pct=75):
    return MetricSummary(n_frames=n, coverage=pct / 100, coverage_pct=pct, median_deg=med + 0.3, q25_deg=lo + 0.2,
                         q75_deg=hi - 0.1, median_rounded=med, median_abs_rounded=abs(med), q25_rounded=lo,
                         q75_rounded=hi)


def result(trunk, leg, unmeasured=()):
    return AnalysisResult(video="clip_150054.mp4", video_hash="abc", gait="trap", fps=30.0, n_frames=470,
                          segment=(150, 420), candidate="candidate@ff", trunk=trunk, leg=leg,
                          unmeasured=list(unmeasured), frames=[])


NOT_ON_HORSE = UnmeasuredInterval(start=160, end=170, frames=11, reason="nu_e_pe_cal",
                                  reason_text="persoana găsită nu e pe cal")
LEANING = UnmeasuredInterval(start=200, end=201, frames=2, reason="trunchi_peste_45",
                             reason_text="aplecare peste 45° sau persoană greșită")


def test_report_fixed_example():
    r = result(metric(10, 5, 15), metric(-10, -15, -5, pct=60), [NOT_ON_HORSE, LEANING, NOT_ON_HORSE])
    assert T.report_text(r) == (
        "Clipul clip_150054.mp4, alura declarată: trap, cadrele 150–420.\n"
        "Trunchiul: aplecat înainte cu aproximativ 10° față de verticală (jumătate din cadre între +5° și +15°; "
        "+ înseamnă aplecat înainte), măsurat pe 75% din cadrele intervalului.\n"
        "Piciorul: călcâiul în fața șoldului, cu aproximativ 10° față de verticala prin șold (jumătate din cadre "
        "între -15° și -5°; - înseamnă călcâi în față), măsurat pe 60% din cadrele intervalului.\n"
        "Cadre nemăsurate, după motiv: persoana găsită nu e pe cal; aplecare peste 45° sau persoană greșită.\n"
        + r.limitation)


@pytest.mark.parametrize("med,word", [(0, "aproape vertical"), (-5, "aplecat pe spate cu aproximativ 5°")])
def test_trunk_directions(med, word):
    assert word in T.trunk_sentence(metric(med, med - 5, med + 5))


def test_equal_quartiles_are_stated_as_one_value():
    r = result(metric(-10, -10, -10), metric(5, 5, 5))
    text = T.report_text(r)
    assert "între -10° și -10°" not in text and "între +5° și +5°" not in text
    assert "cel puțin jumătate din cadre la -10°" in text and "cel puțin jumătate din cadre la +5°" in text
    assert not T.unknown_numbers(text, r)


def test_leg_directions_and_too_few_frames():
    assert "călcâiul în spatele șoldului, cu aproximativ 5°" in T.leg_sentence(metric(5, 0, 10))
    assert "călcâiul aproape sub șold" in T.leg_sentence(metric(0, -5, 5))
    few = T.leg_sentence(metric(5, 0, 10, n=T.MIN_FRAMES - 1))
    assert "prea puține cadre măsurate (9)" in few and "°" not in few


def test_every_number_in_text_is_in_json_and_no_forbidden_labels():
    for r in (result(metric(10, 5, 15), metric(-10, -15, -5), [LEANING]),
              result(metric(-15, -20, -10), metric(-5, -10, 0)),             # mărimea 15 apare doar ca mărime
              result(metric(0, -5, 5), metric(5, 0, 10, n=3, pct=4))):
        text = T.report_text(r)
        assert T.unknown_numbers(text, r) == set()
        assert T.forbidden_labels(text) == []
    r = result(metric(10, 5, 15), metric(-10, -15, -5))
    assert T.unknown_numbers("Trunchiul e aplecat cu 12°.", r) == {12.0}       # 12 nu e în JSON
    assert T.forbidden_labels("Are un chair seat clar.") == ["chair seat"]


def test_report_passes_checks_on_template_text():
    r = result(metric(10, 5, 15), metric(-10, -15, -5), [LEANING])
    assert T.report_passes_checks(T.report_text(r), r)


def test_report_fails_checks_on_unknown_number_or_forbidden_label():
    r = result(metric(10, 5, 15), metric(-10, -15, -5))
    assert not T.report_passes_checks("Trunchiul e aplecat cu 12°.", r)
    assert not T.report_passes_checks(T.report_text(r) + "\nAre un chair seat clar.", r)


def test_clip_name_is_not_judged_as_label():
    r = result(metric(10, 5, 15), metric(-10, -15, -5)).model_copy(update={"video": "perfect_trot.mp4"})
    text = T.report_text(r)
    assert T.forbidden_labels(text) == ["perfect"]                             # fără excepție, ar pica
    assert T.report_passes_checks(text, r)


def test_label_on_first_line_outside_clip_name_fails():
    r = result(metric(10, 5, 15), metric(-10, -15, -5)).model_copy(update={"video": "perfect_trot.mp4"})
    text = T.report_text(r).replace("alura declarată", "alură perfect declarată")
    assert "Clipul perfect_trot.mp4" in text.splitlines()[0]
    assert not T.report_passes_checks(text, r)


def test_clip_name_digits_still_count_for_numbers():
    r = result(metric(10, 5, 15), metric(-10, -15, -5))                        # 150054 vine din numele clipului
    assert T.report_passes_checks(T.report_text(r), r) and 150054.0 in T.numbers_in(T.report_text(r))
    assert not T.report_passes_checks("Clipul are 150055°.", r)


def test_numbers_in_parses_signs_and_decimals():
    assert T.numbers_in("între -15° și +5°, 3,5 și 7.25") == {-15.0, 5.0, 3.5, 7.25}


def _plain():
    return result(metric(5, 0, 10), metric(-5, -10, 0))


def _with_quality(excluded=2, warnings=None):
    from ecvestru.rezultat import QualitySummary, QualityWarning
    warned = warnings if warnings is not None else [QualityWarning(code="salt", text="salt brusc", frames=3)]
    return _plain().model_copy(update={"quality": QualitySummary(
        version=1, excluded_frames=excluded, excluded_reason="călcâi pe piciorul calului", warnings=warned)})


def test_the_report_says_how_many_frames_were_excluded_and_why():
    text = T.report_text(_with_quality())
    assert "2 cadre" in text and "călcâi pe piciorul calului" in text
    assert "salt brusc (3 cadre)" in text


def test_the_numbers_of_the_quality_sentences_are_in_the_json():
    gated = _with_quality()
    assert T.report_passes_checks(T.report_text(gated), gated)


def test_without_exclusions_or_warnings_the_report_stays_silent():
    text = T.report_text(_with_quality(0, []))
    assert "excluse" not in text and "Avertismente" not in text


def test_without_the_quality_layer_the_report_stays_silent():
    assert "excluse" not in T.report_text(_plain())


# ---------- sursa călcâiului ----------

def test_the_report_names_rtmpose_when_the_heel_comes_from_it():
    rtm = _plain().model_copy(update={"heel_source": "rtmpose"})
    text = T.report_text(rtm)
    assert "călcâi estimat cu rtmpose (calea implicită)" in text.lower() and "experimental" not in text.lower()
    assert T.report_passes_checks(text, rtm)


def test_the_report_stays_silent_about_the_heel_source_on_the_default_path():
    assert "RTMPose" not in T.report_text(_plain()) and "RTMPose" not in T.report_text(_with_quality())


# ---------- orientarea după labă (stratul RTMPose) ----------

def _oriented(changed=3, foot=40, share=0.0, unclear=False):
    from ecvestru.rezultat import OrientationSummary
    pct = round(100 * share)
    reason = f"profil neclar: orientarea se schimbă pe {pct}% din cadre" if unclear else None
    summary = OrientationSummary(changed_frames=changed, foot_frames=foot, frames_with_facing=50,
                                 minority_share=share, minority_pct=pct, profile_unclear=unclear, reason=reason)
    return _plain().model_copy(update={"heel_source": "rtmpose", "orientation": summary})


def test_the_report_says_where_the_facing_comes_from_the_foot():
    r = _oriented()
    text = T.report_text(r)
    assert "direcția labei" in text and "40 de cadre" in text and "3" in text
    assert T.report_passes_checks(text, r) and T.hidden_reason(r) is None


def test_an_unclear_profile_hides_the_report_with_the_reason():
    r = _oriented(changed=30, share=0.12, unclear=True)
    assert T.hidden_reason(r) == "profil neclar: orientarea se schimbă pe 12% din cadre"


def test_the_default_path_has_no_orientation_sentence_and_nothing_hidden():
    assert "labei" not in T.report_text(_plain()) and T.hidden_reason(_plain()) is None


def test_the_report_says_when_the_analysis_fell_back_to_the_baseline():
    fell = _plain().model_copy(update={"fallback_reason": "onnxruntime lipsește"})
    text = T.report_text(fell)
    assert "linia de bază" in text and "RTMPose nu a putut rula" in text and T.report_passes_checks(text, fell)
    assert "linia de bază" not in T.report_text(_plain())
