"""
Unghiurile pe cadre și rezumatul unei analize: motivele cadrelor nemăsurate, unghiul față de verticală, mediana
rotunjită, intervalele nemăsurate. Pur (fără fișiere, fără modele), deci îl pot folosi toate straturile analizei.
Fără Streamlit.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ecvestru import schelet
from ecvestru import tinta as T
from ecvestru import vedere as W
from ecvestru.rezultat import AnalysisResult, FrameMeasure, MetricSummary, UnmeasuredInterval

ROUND_DEG = 5          # unghiurile din text se rotunjesc la 5°: precizia în grade nu e măsurată
QUANTILES = (25, 75)   # intervalul arătat în jurul medianei (percentile)

# Motivele cadrelor nemăsurate: codurile candidatului + călcâiul invizibil (doar piciorul).
REASON_HEEL = "calcai_invizibil"
REASON_HEEL_ON_HORSE = "calcai_pe_piciorul_calului"   # poarta A: doar piciorul, trunchiul rămâne
REASON_IMPLAUSIBLE_ANGLE = "unghi_neplauzibil"        # poarta B: unghi de picior în afara intervalului plauzibil
# Stratul opțional „călcâi stabil” (ecvestru/calcai_stabil.py, oprit implicit)
REASON_HEEL_JUMP = "salt_calcai"
REASON_HEEL_ON_HORSE_LEG = "calcai_pe_piciorul_calului_ap10k"
REASON_VIEW_NOT_PROFILE = "vedere_ne_profil"   # doar în analiza cu vederi (ecvestru/analiza_vederi.py)

REASON_TEXTS = {
    "calaret_negasit": "călărețul nu a fost găsit",
    "nu_e_pe_cal": "persoana găsită nu e pe cal",
    "trunchi_peste_45": "aplecare peste 45° sau persoană greșită",
    "puncte_invizibile": "umăr sau șold invizibil",
    REASON_HEEL: "călcâi invizibil (doar piciorul)",
    REASON_HEEL_ON_HORSE: "călcâi pe piciorul calului",
    REASON_IMPLAUSIBLE_ANGLE: "unghi neplauzibil",
    REASON_HEEL_JUMP: "salt al călcâiului",
    REASON_HEEL_ON_HORSE_LEG: "călcâi pe piciorul calului (puncte ale calului)",
    W.REASON_NOT_PROFILE: "nu e din profil (umerii apar lați)",
    REASON_VIEW_NOT_PROFILE: "nu e din profil (după vedere: router sau intervale)",
    T.REASON_UNSURE: "țintă nesigură: alt cal e prea aproape",
    T.REASON_LOST: "ținta s-a pierdut",
    T.REASON_OTHER: "metoda a ales alt călăreț",
    schelet.REASON_NOT_CHOSEN: "neselectată de candidat",
}

Raw = dict[int, dict]   # cadru → {"points": {...} | None, "motiv": str | None}


def angle_from_vertical(top, bottom, facing: float) -> float:
    """Convenția semnelor: trunchi > 0 = aplecat în sensul de mers; picior < 0 =
    călcâi în fața șoldului."""
    return math.degrees(math.atan2((top[0] - bottom[0]) * facing, bottom[1] - top[1]))


def frame_measures(raw: Raw, n_frames: int) -> list[FrameMeasure]:
    out = []
    for i in range(n_frames):
        r = raw.get(i) or {"points": None, "motiv": "calaret_negasit"}
        p, reason = r["points"], r["motiv"]
        trunk = leg = None
        if reason is None and p is not None:
            f = p.get("facing")
            trunk = angle_from_vertical(p["shoulder"], p["hip"], f)
            if p.get("heel") is not None:
                leg = angle_from_vertical(p["hip"], p["heel"], f)
            else:
                reason = REASON_HEEL
        out.append(FrameMeasure(frame=i, trunk_deg=trunk, leg_deg=leg, reason=reason))
    return out


def round_to(value: float, step: int = ROUND_DEG) -> int:
    """Rotunjire simetrică la cel mai apropiat multiplu de `step` (−7.5 → −10, 7.4 → 5)."""
    return int(math.copysign(math.floor(abs(value) / step + 0.5) * step, value)) or 0


def summarize(values: list[float | None]) -> MetricSummary:
    """Rezumatul unui unghi pe cadrele din interval (None = nemăsurat)."""
    got = np.array([v for v in values if v is not None], float)
    n_all = len(values)
    coverage = len(got) / n_all if n_all else 0.0
    base = {"n_frames": len(got), "coverage": coverage, "coverage_pct": round(100 * coverage)}
    if not len(got):
        return MetricSummary(**base)
    med = float(np.median(got))
    lo, hi = (float(np.percentile(got, q)) for q in QUANTILES)
    return MetricSummary(**base, median_deg=med, q25_deg=lo, q75_deg=hi, median_rounded=round_to(med),
                         median_abs_rounded=abs(round_to(med)), q25_rounded=round_to(lo), q75_rounded=round_to(hi))


def unmeasured_intervals(frames: list[FrameMeasure]) -> list[UnmeasuredInterval]:
    """Cadre consecutive cu același motiv (trunchi sau picior nemăsurat), grupate în intervale."""
    out: list[UnmeasuredInterval] = []
    for fm in frames:
        if fm.reason is None:
            continue
        last = out[-1] if out else None
        if last and last.reason == fm.reason and last.end == fm.frame - 1:
            out[-1] = last.model_copy(update={"end": fm.frame, "frames": last.frames + 1})
        else:
            out.append(UnmeasuredInterval(start=fm.frame, end=fm.frame, frames=1, reason=fm.reason,
                                          reason_text=REASON_TEXTS.get(fm.reason, fm.reason)))
    return out


def lost_target_intervals(result: AnalysisResult) -> list[tuple[int, int]]:
    """Intervalele în care ținta s-a pierdut, ca pagina să arate unde mai trebuie un click."""
    return [(u.start, u.end) for u in result.unmeasured if u.reason == T.REASON_LOST]


def first_lost_frame(result: AnalysisResult) -> int | None:
    """Primul cadru în care ținta s-a pierdut: pagina îl deschide, ca omul să mai dea un click pe același călăreț."""
    lost = lost_target_intervals(result)
    return lost[0][0] if lost else None


def frame_summary(result: AnalysisResult) -> tuple[int, list[tuple[str, int]]]:
    """Cadrele cu trunchi măsurat și, pentru celelalte, câte are fiecare motiv, în ordine descrescătoare."""
    measured = sum(f.trunk_deg is not None for f in result.frames)
    reasons = Counter(REASON_TEXTS.get(f.reason, f.reason) for f in result.frames if f.trunk_deg is None)
    return measured, reasons.most_common()
