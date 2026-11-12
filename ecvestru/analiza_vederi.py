"""
Analiza cu vederi: modulul față/spate lipit de măsurarea de profil, într-un singur rezultat (CombinedResult,
JSON 1.5) și un singur raport. Vederea pe cadru vine din router (ecvestru/router_vedere.py) sau din intervale (ale
omului, ori intervalul din profil). Partea de profil e analiza obișnuită (masurare.analyze, JSON 1.4), din care
cadrele cu altă vedere ies cu motivul lor; dacă toate cadrele intervalului sunt din profil, ea e identică cu analiza
obișnuită. Pe cadrele din față sau din spate se măsoară înclinarea liniei umerilor și a bazinului din scheletul
MediaPipe, ca dreaptă (modulo 180°); valorile sunt nevalidate.

Stratul e oprit implicit (VIEWS_LAYER_DEFAULT): aplicația și JSON-ul implicit nu se schimbă. Fără Streamlit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ecvestru import constante_candidat as KC
from ecvestru import intervale as IV
from ecvestru import masurare as M
from ecvestru import raport as RP
from ecvestru import stocare as ST
from ecvestru.rezultat import (
    AnalysisResult,
    CombinedResult,
    FrontBackFrame,
    FrontBackSummary,
    MetricSummary,
    ViewSpan,
)

VIEWS_LAYER_DEFAULT = False
SOURCE_ROUTER, SOURCE_INTERVALS = "router", "intervale"
SOURCES = (SOURCE_ROUTER, SOURCE_INTERVALS)
SOURCE_TEXTS = {SOURCE_ROUTER: "router (automat, nevalidat)", SOURCE_INTERVALS: "intervale"}
NO_VIEW = ""
NO_VIEW_TEXT = "fără vedere"
REASON_VIEW = M.REASON_VIEW_NOT_PROFILE
FRONT_BACK = (IV.FRONT, IV.BACK)
HALF_TURN = 180.0
QUARTER_TURN = 90.0

LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 11, 12, 23, 24
MIN_VISIBILITY = KC.MIN_VISIBILITY   # ca în candidat (ecvestru/constante_candidat.py)

FRONT_BACK_LIMITATION = ("Înclinările din față și din spate nu sunt încă verificate față de click-urile omului "
                         "(criteriile pentru umeri și bazin).")


# ---------- vederea pe cadre ----------

def views_from_segment(segment: tuple[int, int], n_frames: int) -> np.ndarray:
    """Intervalul vechi din profil: profil pe el, fără vedere în rest."""
    views = np.full(n_frames, NO_VIEW, dtype=object)
    views[max(segment[0], 0): min(segment[1], n_frames - 1) + 1] = IV.PROFILE
    return views


def views_from_intervals(intervals: list[IV.Interval], n_frames: int) -> np.ndarray:
    views = np.full(n_frames, NO_VIEW, dtype=object)
    for interval in intervals:
        views[max(interval.start, 0): min(interval.stop, n_frames - 1) + 1] = interval.view
    return views


def view_spans(views: np.ndarray, segment: tuple[int, int]) -> list[ViewSpan]:
    spans: list[ViewSpan] = []
    for frame in range(segment[0], segment[1] + 1):
        view = str(views[frame]) if frame < len(views) else NO_VIEW
        if spans and spans[-1].view == view and spans[-1].end == frame - 1:
            spans[-1] = spans[-1].model_copy(update={"end": frame})
        else:
            spans.append(ViewSpan(start=frame, end=frame, view=view))
    return spans


# ---------- înclinările din față / spate ----------

def tilt_deg(a, b) -> float:
    """Unghiul dreptei prin a și b față de orizontala imaginii, în (-90°, 90°]; + = partea din dreapta imaginii mai
    jos (y crește în jos). Ordinea punctelor nu contează."""
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    while angle > QUARTER_TURN:
        angle -= HALF_TURN
    while angle <= -QUARTER_TURN:
        angle += HALF_TURN
    return angle


def _pair_tilt(xy: np.ndarray, visibility: np.ndarray, left: int, right: int) -> float | None:
    points = xy[[left, right]]
    if np.isnan(points).any() or (visibility[[left, right]] < MIN_VISIBILITY).any():
        return None
    return tilt_deg(points[0], points[1])


def front_back_frames(skeleton, views: np.ndarray, segment: tuple[int, int]) -> list[FrontBackFrame]:
    """Cadrele din față sau din spate ale intervalului, cu înclinările din schelet (None unde nu se văd)."""
    rows = {} if skeleton is None else {int(f): i for i, f in enumerate(skeleton.frame)}
    out = []
    for frame in range(segment[0], min(segment[1], len(views) - 1) + 1):
        view = views[frame]
        if view not in FRONT_BACK:
            continue
        shoulders = pelvis = None
        if frame in rows:
            xy, vis = np.asarray(skeleton.xy_px[rows[frame]], float), np.asarray(skeleton.visibility[rows[frame]])
            shoulders = _pair_tilt(xy, vis, LEFT_SHOULDER, RIGHT_SHOULDER)
            pelvis = _pair_tilt(xy, vis, LEFT_HIP, RIGHT_HIP)
        out.append(FrontBackFrame(frame=frame, view=view, shoulder_tilt_deg=shoulders, pelvis_tilt_deg=pelvis))
    return out


def front_back_summary(frames: list[FrontBackFrame]) -> FrontBackSummary | None:
    if not frames:
        return None
    return FrontBackSummary(frames_front=sum(f.view == IV.FRONT for f in frames),
                            frames_back=sum(f.view == IV.BACK for f in frames),
                            shoulders=M.summarize([f.shoulder_tilt_deg for f in frames]),
                            pelvis=M.summarize([f.pelvis_tilt_deg for f in frames]), frames=frames)


# ---------- profilul restrâns la cadrele din profil ----------

def restrict_profile(result: AnalysisResult, views: np.ndarray) -> AnalysisResult:
    """Cadrele intervalului care nu sunt din profil ies din măsurare, cu motivul REASON_VIEW. Fără niciun astfel de
    cadru, rezultatul e întors neschimbat (identic cu analiza obișnuită). Rezumatul de calitate se recalculează pe
    cadrele rămase în profil; None rămâne None (stratul n-a rulat)."""
    def is_profile(frame: int) -> bool:
        return frame < len(views) and views[frame] == IV.PROFILE

    if all(is_profile(f.frame) for f in result.frames):
        return result
    frames = [f if is_profile(f.frame) else f.model_copy(update={"trunk_deg": None, "leg_deg": None,
                                                                 "reason": REASON_VIEW})
              for f in result.frames]
    kept = [f for f in frames if is_profile(f.frame)]
    quality = None if result.quality is None else M._quality_summary(kept)
    return result.model_copy(update={"frames": frames, "trunk": M.summarize([f.trunk_deg for f in frames]),
                                     "leg": M.summarize([f.leg_deg for f in frames]),
                                     "unmeasured": M.unmeasured_intervals(frames), "quality": quality})


def profile_skeleton(video_path: Path, profile: AnalysisResult):
    return ST.load_rider_skeleton(video_path, profile.video, profile.video_hash, profile.candidate)


def analyze_views(video_path: Path, gait: str, views: np.ndarray, source: str,
                  segment: tuple[int, int] | None = None, **analyze_args) -> CombinedResult:
    """Analiza cu vederi pe `segment`: profilul (masurare.analyze, cu aceleași argumente) restrâns la cadrele din
    profil, plus modulul față/spate pe cadrele din față sau din spate."""
    if source not in SOURCES:
        raise ValueError(f"Sursă de vedere necunoscută: {source} (cunoscute: {', '.join(SOURCES)}).")
    video_path = Path(video_path)
    profile = M.analyze(video_path, gait, segment, **analyze_args)
    a, b = profile.segment
    views = np.asarray(views, dtype=object)
    skeleton = profile_skeleton(video_path, profile)
    spans = view_spans(views, (a, b))
    counts: dict[str, int] = {}
    for span in spans:
        counts[span.view] = counts.get(span.view, 0) + span.end - span.start + 1
    return CombinedResult(view_source=source, views=spans, frames_by_view=counts,
                          profile=restrict_profile(profile, views),
                          front_back=front_back_summary(front_back_frames(skeleton, views, (a, b))))


# ---------- raportul unic ----------

def _signed(v: int) -> str:
    return f"+{v}" if v > 0 else str(v)


def _tilt_sentence(label: str, m: MetricSummary) -> str:
    if m.n_frames < RP.MIN_FRAMES:
        return f"{label}: prea puține cadre măsurate ({m.n_frames}) pentru o descriere."
    where = ("aproape orizontală" if m.median_rounded == 0 else
             f"înclinată cu aproximativ {m.median_abs_rounded}°, partea din "
             f"{'dreapta' if m.median_rounded > 0 else 'stânga'} imaginii mai jos")
    spread = (f"cel puțin jumătate din cadre la {_signed(m.q25_rounded)}°" if m.q25_rounded == m.q75_rounded else
              f"jumătate din cadre între {_signed(m.q25_rounded)}° și {_signed(m.q75_rounded)}°")
    return (f"{label}: linia {where} ({spread}; + înseamnă partea din dreapta imaginii mai jos), "
            f"măsurată pe {m.coverage_pct}% din cadrele din față și din spate.")


def views_sentence(c: CombinedResult) -> str:
    parts = [f"{IV.VIEWS.get(view, NO_VIEW_TEXT if view == NO_VIEW else view).lower()} pe {n} cadre"
             for view, n in c.frames_by_view.items()]
    return f"Vederea vine din {SOURCE_TEXTS[c.view_source]}: " + ", ".join(parts) + "."


def report_text(c: CombinedResult) -> str:
    lines = [RP.report_text(c.profile), views_sentence(c)]
    if c.front_back is not None:
        lines += [_tilt_sentence("Umerii (din față și din spate)", c.front_back.shoulders),
                  _tilt_sentence("Bazinul (din față și din spate)", c.front_back.pelvis), FRONT_BACK_LIMITATION]
    return "\n".join(lines)


def allowed_numbers(c: CombinedResult) -> set[float]:
    """Numerele din JSON-ul 1.5, fără listele pe cadre (ar permite aproape orice întreg)."""
    data = json.loads(c.to_json())
    data["profile"].pop("frames", None)
    if "front_back" in data:
        data["front_back"].pop("frames", None)
    return RP.numbers_in(json.dumps(data, ensure_ascii=False))


def report_passes_checks(text: str, c: CombinedResult) -> bool:
    return (not (RP.numbers_in(text) - allowed_numbers(c))
            and not RP.forbidden_labels(text.replace(c.profile.video, "")))
