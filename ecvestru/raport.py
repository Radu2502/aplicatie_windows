"""
Raportul din șabloane: descrie strict ce s-a măsurat, din câmpurile rotunjite ale AnalysisResult,
fără etichete de tip „chair seat” și fără să sugereze o precizie în grade care nu a fost măsurată.

`numbers_in` și `allowed_numbers` verifică faptul că orice număr din text există în JSON; aceeași
verificare se poate aplica oricărui text de raport, nu doar celui din șabloane.
"""
from __future__ import annotations

import json
import re

from ecvestru.rezultat import HEEL_RTMPOSE, AnalysisResult, MetricSummary

MIN_FRAMES = 10   # sub atâtea cadre măsurate, un unghi nu se descrie
FORBIDDEN = ("chair seat", "fork seat", "scaun", "perfect")   # etichete și judecăți pe care nu le punem
NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _signed(v: int) -> str:
    return f"+{v}" if v > 0 else str(v)


def _range(m: MetricSummary) -> str:
    if m.q25_rounded == m.q75_rounded:
        return f"cel puțin jumătate din cadre la {_signed(m.q25_rounded)}°"
    return f"jumătate din cadre între {_signed(m.q25_rounded)}° și {_signed(m.q75_rounded)}°"


def trunk_sentence(m: MetricSummary) -> str:
    if m.n_frames < MIN_FRAMES:
        return f"Trunchiul: prea puține cadre măsurate ({m.n_frames}) pentru o descriere."
    v = m.median_rounded
    where = ("aproape vertical" if v == 0 else
             f"aplecat {'înainte' if v > 0 else 'pe spate'} cu aproximativ {m.median_abs_rounded}° față de verticală")
    return (f"Trunchiul: {where} ({_range(m)}; + înseamnă aplecat înainte), "
            f"măsurat pe {m.coverage_pct}% din cadrele intervalului.")


def leg_sentence(m: MetricSummary) -> str:
    if m.n_frames < MIN_FRAMES:
        return f"Piciorul: prea puține cadre măsurate ({m.n_frames}) pentru o descriere."
    v = m.median_rounded
    where = ("călcâiul aproape sub șold" if v == 0 else
             f"călcâiul {'în fața' if v < 0 else 'în spatele'} șoldului, cu aproximativ {m.median_abs_rounded}° "
             "față de verticala prin șold")
    return (f"Piciorul: {where} ({_range(m)}; - înseamnă călcâi în față), "
            f"măsurat pe {m.coverage_pct}% din cadrele intervalului.")


def unmeasured_sentence(r: AnalysisResult) -> str | None:
    reasons = list(dict.fromkeys(u.reason_text for u in r.unmeasured))
    return ("Cadre nemăsurate, după motiv: " + "; ".join(reasons) + ".") if reasons else None


def camera_sentence(r: AnalysisResult) -> str | None:
    if r.camera is None:
        return None
    if r.camera.amplitude_rounded is None:
        return "Rotirea camerei nu a putut fi estimată pe acest clip."
    return (f"Camera s-a rotit cu până la {r.camera.amplitude_rounded}° în interval; "
            "unghiurile pot fi deplasate cu atât.")


RTMPOSE_SENTENCE = "Călcâi estimat cu RTMPose (calea implicită)."
FALLBACK_SENTENCE = "Analiza a rulat pe linia de bază (MediaPipe): RTMPose nu a putut rula."
ORIENTATION_SENTENCE = ("Sensul de mers vine din direcția labei (RTMPose, regula B4) pe {foot} de cadre din interval; "
                        "pe {changed} dintre ele diferă de cel după nas și ureche.")


def orientation_sentence(r: AnalysisResult) -> str | None:
    """Pe calea RTMPose, de unde vine sensul de mers (care dă semnul unghiurilor); altfel nimic."""
    if r.orientation is None or not r.orientation.foot_frames:
        return None
    return ORIENTATION_SENTENCE.format(foot=r.orientation.foot_frames, changed=r.orientation.changed_frames)


def hidden_reason(r: AnalysisResult) -> str | None:
    """Motivul pentru care raportul text nu se afișează deloc („profil neclar”); None = se poate afișa."""
    return r.orientation.reason if r.orientation is not None and r.orientation.profile_unclear else None


def quality_sentences(r: AnalysisResult) -> list[str]:
    """Câte cadre a exclus poarta și ce avertismente există; fără strat de calitate, nimic."""
    if r.quality is None:
        return []
    lines = []
    if r.quality.excluded_frames:
        lines.append(f"Piciorul: {r.quality.excluded_frames} cadre excluse din măsurare, motiv: "
                     f"{r.quality.excluded_reason}. Trunchiul rămâne măsurat pe ele.")
    if r.quality.warnings:
        listed = "; ".join(f"{w.text} ({w.frames} cadre)" for w in r.quality.warnings)
        lines.append(f"Avertismente, fără excludere: {listed}.")
    return lines


def report_text(r: AnalysisResult) -> str:
    chosen = f", alegere manuală, cadrul {r.target.frame}" if r.target else ""
    lines = [f"Clipul {r.video}, alura declarată: {r.gait}, cadrele {r.segment[0]}–{r.segment[1]}{chosen}.",
             trunk_sentence(r.trunk), leg_sentence(r.leg)]
    if r.heel_source == HEEL_RTMPOSE:
        lines.append(RTMPOSE_SENTENCE)
    if r.fallback_reason is not None:     # motivul exact e în pagină și în JSON; aici fără cifre (verificarea)
        lines.append(FALLBACK_SENTENCE)
    if (s := orientation_sentence(r)) is not None:
        lines.append(s)
    if (s := unmeasured_sentence(r)) is not None:
        lines.append(s)
    lines += quality_sentences(r)
    if (s := camera_sentence(r)) is not None:
        lines.append(s)
    lines.append(r.limitation)
    return "\n".join(lines)


# ---------- verificarea numerelor ----------

def numbers_in(text: str) -> set[float]:
    return {float(n.replace(",", ".")) for n in NUMBER.findall(text)}


def allowed_numbers(r: AnalysisResult) -> set[float]:
    """Numerele din JSON-ul rezultatului, fără lista pe cadre (ar permite aproape orice întreg)."""
    data = json.loads(r.model_dump_json(exclude={"frames"}))
    return numbers_in(json.dumps(data, ensure_ascii=False))


def unknown_numbers(text: str, r: AnalysisResult) -> set[float]:
    """Numerele din text care nu există în JSON (gol = textul trece verificarea)."""
    return numbers_in(text) - allowed_numbers(r)


def forbidden_labels(text: str) -> list[str]:
    low = text.lower()
    return [w for w in FORBIDDEN if w in low]


def report_passes_checks(text: str, r: AnalysisResult) -> bool:
    """Textul se poate afișa: orice număr e în JSON și nu are etichete interzise. Numele clipului
    nu e judecat ca etichetă, pentru că îl alege utilizatorul (de exemplu perfect_trot.mp4)."""
    return not unknown_numbers(text, r) and not forbidden_labels(text.replace(r.video, ""))
