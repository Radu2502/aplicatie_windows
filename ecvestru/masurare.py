"""
Orchestratorul analizei: video → schelet (cu candidatul, research/candidate.py, doar citit) → unghiuri pe
cadre → rezumat (AnalysisResult). Fără Streamlit.

Trei module, fără ciclu de importuri:
  - ecvestru/stocare.py: cache-ul etapelor (căile, cheile, încărcarea din cache sau calculul o dată);
  - ecvestru/rezumat.py: unghiurile pe cadre, motivele, mediana și intervalele nemăsurate (pur);
  - acesta: ordinea etapelor (`analyze`), straturile (calitate, RTMPose, călcâi stabil) și cheia rezultatului,
    din versiunile etapelor (`result_cache_path`). Numele din stocare și rezumat rămân disponibile și aici.

Cache-ul, în data/results/analiza/ (sau ECVESTRU_RESULTS/analiza/):
  - brut/<clip>-<hash>__<candidat>.json: ieșirea candidatului pe toate cadrele (rularea durează);
  - <clip>-<hash>__<candidat>__<a>-<b>__<alură>__v<schemă>.json: rezultatul pentru un interval;
  - schelet/<clip>-<hash>__<candidat>__s<format>.npz: cele 33 de puncte.
Hash-ul e al conținutului fișierului, deci un clip încărcat cu același nume, dar alt conținut,
nu folosește rezultatul vechi. Clipurile încărcate stau doar într-un folder temporar.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ecvestru import cal_puncte, calcai_stabil, calitate, rtmpose, rtmpose_l, schelet_cal
from ecvestru import camera as K
from ecvestru import progres as P
from ecvestru import stocare as ST
from ecvestru import tinta as T
from ecvestru import vedere as W
from ecvestru import verificare as V
from ecvestru.rezultat import (
    HEEL_MEDIAPIPE,
    HEEL_MODEL_L,
    HEEL_MODEL_M,
    HEEL_RTMPOSE,
    LIMITATION_UNVERIFIED,
    PIPELINE_BASELINE,
    PIPELINE_RTMPOSE,
    QUALITY_VERSION,
    SCHEMA_VERSION,
    AnalysisResult,
    FrameMeasure,
    LayerVersions,
    OrientationSummary,
    TargetSeed,
)
from ecvestru.rezumat import (  # noqa: F401  (păstrate și aici: paginile, uneltele și testele le citesc din masurare)
    QUANTILES,
    REASON_HEEL,
    REASON_HEEL_JUMP,
    REASON_HEEL_ON_HORSE,
    REASON_HEEL_ON_HORSE_LEG,
    REASON_IMPLAUSIBLE_ANGLE,
    REASON_TEXTS,
    REASON_VIEW_NOT_PROFILE,
    ROUND_DEG,
    Raw,
    angle_from_vertical,
    first_lost_frame,
    frame_measures,
    frame_summary,
    lost_target_intervals,
    round_to,
    summarize,
    unmeasured_intervals,
)
from ecvestru.stocare import (  # noqa: F401  (idem)
    BOXES_VERSION,
    CANDIDATE,
    HASH_CHARS,
    ROOT,
    SPLIT_CSV,
    UPLOAD_KEY,
    UPLOAD_LABEL,
    boxes_cache_path,
    bytes_hash,
    candidate_label,
    candidate_module,
    clip_key,
    dev_videos,
    file_hash,
    load_boxes,
    load_motion,
    load_ratios,
    load_raw,
    load_rider_skeleton,
    motion_cache_path,
    ratios_cache_path,
    raw_cache_path,
    read_frame,
    results_dir,
    skeleton_cache_path,
    upload_name,
    video_info,
)

EXTENDED_VERSION = 5   # analizele noi (țintă sau clip încărcat): se mărește când se schimbă verificările lor
MIN_UPLOAD_FRAMES = 2  # un interval are nevoie de două capete distincte
TEMP_STEM = "clip"
UPLOAD_SUFFIXES = (".mp4", ".mov", ".avi", ".m4v")   # ca în pagina Analiză; altfel copia temporară primește .mp4
DEFAULT_SUFFIX = ".mp4"
UPLOAD_KEPT = ("Rămân pe disc, în data/results/analiza/: rezultatul (JSON), brut/ (punctele și motivele pe cadre), "
               "vedere/ (raportul umeri/trunchi pe cadre), cai/ (casetele cailor pe cadre), camera/ (mișcarea "
               "camerei pe cadre), schelet/ (cele 33 de puncte pe cadre, pentru călăreț și pentru alte persoane "
               "găsite în cadru, de exemplu instructorul). Numele original al fișierului nu se "
               "păstrează: fișierele poartă doar hash-ul conținutului.")


# ---------- cheia rezultatului: din versiunile etapelor ----------


def result_cache_path(name: str, video_hash: str, candidate: str, segment: tuple[int, int], gait: str,
                      target: TargetSeed | None = None, extended: bool = False,
                      anchors: Sequence[TargetSeed] = (), heel_source: str = HEEL_MEDIAPIPE,
                      heel_stable: bool = False, heel_model: str = HEEL_MODEL_M) -> Path:
    chosen = "".join(f"__t{T.seed_key(seed)}" for seed in ([target] if target else []) + list(anchors))
    checks = f"__x{EXTENDED_VERSION}" if extended else ""
    if heel_source == HEEL_RTMPOSE:
        checks += f"__rtm{rtmpose.RTMPOSE_VERSION}{rtmpose.MODEL_ZIP_SHA256[:8]}o{rtmpose.ORIENTATION_VERSION}"
        if heel_stable:
            checks += f"__cs{calcai_stabil.VERSION}{cal_puncte.MODEL_ZIP_SHA256[:8]}"
            if heel_model == HEEL_MODEL_L:
                checks += f"l{rtmpose_l.VERSION}{rtmpose_l.MODEL_ZIP_SHA256[:8]}"
    return ST.results_dir() / (f"{ST.clip_key(name, video_hash)}__{candidate.replace('@', '_')}__"
                            f"{segment[0]}-{segment[1]}__{gait}{chosen}{checks}__q{QUALITY_VERSION}__v{SCHEMA_VERSION}.json")


def has_profile_interval(name: str) -> bool:
    """Clipul are un interval din profil verificat manual (în data/profil.csv)."""
    return bool(V.profile_segments().get(name))


def default_segment(name: str, n_frames: int) -> tuple[int, int]:
    """Intervalul implicit: cel mai lung segment din profil (data/profil.csv), altfel tot clipul."""
    segments = V.profile_segments().get(name, [])
    if not segments:
        return (0, n_frames - 1)
    a, b = max(segments, key=lambda s: s[1] - s[0])
    return (max(0, a), min(n_frames - 1, b))


# ---------- analiza ----------


def _quality_summary(frames: list[FrameMeasure]):
    return calitate.summarize_quality(frames)


def limitation_for(name: str, uploaded: bool = False, target: TargetSeed | None = None) -> str:
    """Textul de limitare al analizei: fără verdicte pe cadre, afirmația despre verificare nu se poate face."""
    return LIMITATION_UNVERIFIED


def rtmpose_points(video_path: Path, name: str, video_hash: str, skeleton, heel_estimator: Callable | None = None,
                   heel_model: str = HEEL_MODEL_M) -> np.ndarray:
    """Punctele Halpe26 ale căii RTMPose (m sau l): `rtmpose_l.points`."""
    return rtmpose_l.points(video_path, name, video_hash, skeleton, heel_estimator, heel_model)


def heel_model_of(layers: LayerVersions | None) -> str:
    """Modelul RTMPose cu care s-a măsurat un rezultat, din versiunile straturilor: l doar cu stratul „călcâi stabil”
    și sha256-ul pachetului RTMPose-l; altfel m."""
    if layers is None or layers.heel_stable is None or layers.rtmpose_model is None:
        return HEEL_MODEL_M
    return HEEL_MODEL_L if layers.rtmpose_model == rtmpose_l.MODEL_ZIP_SHA256[:8] else HEEL_MODEL_M


def apply_quality(video_path: Path, name: str, video_hash: str, candidate: str, raw: Raw,
                  frames: list[FrameMeasure], heel_source: str = HEEL_MEDIAPIPE,
                  heel_estimator: Callable | None = None, heel_model: str = HEEL_MODEL_M
                  ) -> tuple[list[FrameMeasure], bool]:
    """Poarta A, intervalul și avertismentele, peste măsurători. Cu `heel_source=rtmpose` (calea implicită), piciorul
    din fuziune + interval, fără poarta A. Fără schelet complet, cadrele rămân cum sunt (False)."""
    skeleton = ST.load_rider_skeleton(video_path, name, video_hash, candidate)
    if skeleton is None:
        return frames, False
    boxes = ST.load_boxes(video_path, name, video_hash)
    if heel_source == HEEL_RTMPOSE:
        points = rtmpose_points(video_path, name, video_hash, skeleton, heel_estimator, heel_model)
        frames = rtmpose.measure_legs(frames, raw, rtmpose.side_heels(points, skeleton),
                                      rtmpose.horse_boxes(skeleton, boxes))
        return calitate.gate_frames(calitate.ClipData(name, skeleton, raw, frames, boxes), leg_gate=False), True
    return calitate.gate_frames(calitate.ClipData(name, skeleton, raw, frames, boxes)), True


def orient_rtmpose(video_path: Path, name: str, video_hash: str, candidate: str, raw: Raw, fps: float,
                   segment: tuple[int, int], heel_estimator: Callable | None = None, heel_model: str = HEEL_MODEL_M
                   ) -> tuple[Raw, OrientationSummary | None, dict[int, str]]:
    """Calea RTMPose: orientarea după regula B4, doar în `segment`, înainte de măsurare. Fără schelet complet,
    brutul rămâne (și analiza cade pe MediaPipe)."""
    skeleton = ST.load_rider_skeleton(video_path, name, video_hash, candidate)
    if skeleton is None:
        return raw, None, {}
    points = rtmpose_points(video_path, name, video_hash, skeleton, heel_estimator, heel_model)
    return rtmpose.orient(raw, skeleton, points, fps, segment)


APP_DEFAULT_HEEL_SOURCE = HEEL_RTMPOSE   # aplicația cere RTMPose; linia de bază doar la cerere sau ca rezervă
APP_DEFAULT_HEEL_STABLE = False          # stratul „călcâi stabil” e oprit implicit în aplicație
HEEL_MODEL_LABELS = {HEEL_MODEL_M: "RTMPose-m (implicit)", HEEL_MODEL_L: "RTMPose-l (mai mare, mai lent)"}
FALLBACK_NO_RUNTIME = "onnxruntime lipsește"
FALLBACK_DOWNLOAD = "modelul RTMPose nu s-a putut descărca ({error})"
FALLBACK_NO_SKELETON = "lipsește scheletul complet al clipului"
FALLBACK_DOWNLOAD_L = "modelul RTMPose-l nu s-a putut descărca ({error})"


def rtmpose_problem(heel_estimator: Callable | None = None, heel_model: str = HEEL_MODEL_M) -> str | None:
    """De ce nu poate rula calea RTMPose (None = poate). Modelul lipsă se descarcă acum, o singură dată, cu suma
    sha256 verificată. Un estimator dat din afară (în teste) nu cere nici onnxruntime, nici model."""
    if heel_estimator is not None:
        return None
    if not rtmpose.runtime_available():
        return FALLBACK_NO_RUNTIME
    if rtmpose.model_path() is None:
        try:
            rtmpose.download_model()
        except (OSError, ValueError) as error:   # rețea, disc, sumă sha256 greșită, arhivă stricată
            return FALLBACK_DOWNLOAD.format(error=error)
    if heel_model == HEEL_MODEL_L:
        try:
            rtmpose_l.ensure_model()
        except (OSError, ValueError) as error:
            return FALLBACK_DOWNLOAD_L.format(error=error)
    return None


def _layers(pipeline: str, gated: bool, heel_stable: bool = False, heel_model: str = HEEL_MODEL_M) -> LayerVersions:
    if pipeline != PIPELINE_RTMPOSE:
        return LayerVersions(quality=QUALITY_VERSION if gated else None)
    stable, model = None, rtmpose.MODEL_ZIP_SHA256[:8]
    if heel_stable:
        stable = calcai_stabil.VERSION
        if heel_model == HEEL_MODEL_L:
            model = rtmpose_l.MODEL_ZIP_SHA256[:8]
    return LayerVersions(quality=QUALITY_VERSION, rtmpose=rtmpose.RTMPOSE_VERSION,
                         orientation=rtmpose.ORIENTATION_VERSION, rtmpose_model=model,
                         heel_stable=stable)


def analyze(video_path: Path, gait: str, segment: tuple[int, int] | None = None, name: str | None = None,
            predictor: Callable[[Path], Raw] | None = None, target: TargetSeed | None = None,
            extended: bool = False, anchors: Sequence[TargetSeed] = (), uploaded: bool = False,
            heel_source: str = HEEL_MEDIAPIPE, heel_estimator: Callable | None = None, heel_stable: bool = False,
            horse_estimator: Callable | None = None, heel_model: str = HEEL_MODEL_M,
            stages: P.Stages | None = None, horse_skeleton: bool = False) -> AnalysisResult:
    """Analiza unui clip pe intervalul din profil `segment` (cadre, inclusiv capetele). `name` e
    numele afișat (pentru un clip încărcat, numele original, nu cel al fișierului temporar).
    `extended` cere verificările analizelor noi; o țintă le cere întotdeauna. Fără ele, rezultatul și
    fișierul din cache rămân cele de dinainte. Textul limitării nu vine din cache: se schimbă când clipul
    primește verdicte. `heel_source=rtmpose` cere calea RTMPose (implicită în aplicație: APP_DEFAULT_HEEL_SOURCE).
    Dacă ea nu poate rula (modul, onnxruntime sau model lipsă; schelet incomplet), rulează linia de bază, iar rezultatul
    spune calea (`pipeline`) și motivul (`fallback_reason`). Căderea din lipsa modulului, a onnxruntime sau a modelului
    nu se păstrează în cache ca rezultat RTMPose. `heel_stable` pornește stratul opțional „călcâi stabil”
    (ecvestru/calcai_stabil.py), doar pe calea RTMPose; oprit (implicit), rezultatul și JSON-ul sunt cele de
    dinainte. `heel_model` alege modelul RTMPose al stratului: m (implicit) sau l (RTMPose-l, pe toată calea RTMPose);
    l cere stratul pornit. `stages` (ecvestru/progres.py) primește începutul și durata fiecărei etape, pentru bara de
    progres; doar observă, rezultatul e același cu sau fără. `horse_skeleton` (oprit implicit) adaugă scheletul calului
    de sub călăreț pe cadrele analizate (ecvestru/schelet_cal.py, schema 1.6); rezultatul din cache rămâne cel fără el.
    `horse_estimator` e estimatorul AP-10K (al stratului „călcâi stabil” și al scheletului calului)."""
    if heel_model not in (HEEL_MODEL_M, HEEL_MODEL_L):
        raise ValueError(f"Model RTMPose necunoscut: {heel_model}")
    if heel_model == HEEL_MODEL_L and not heel_stable:
        raise ValueError("RTMPose-l se folosește doar cu stratul „călcâi stabil” (heel_stable=True).")
    extended = extended or target is not None
    fallback = None
    if heel_source == HEEL_RTMPOSE and (fallback := rtmpose_problem(heel_estimator, heel_model)) is not None:
        heel_source = HEEL_MEDIAPIPE
    video_path = Path(video_path)
    name = name or video_path.name
    video_hash, candidate = ST.file_hash(video_path), ST.candidate_label()
    n_frames, fps = ST.video_info(video_path)
    segment = segment or default_segment(name, n_frames)
    a, b = max(0, segment[0]), min(n_frames - 1, segment[1])
    if a > b:
        raise ValueError(f"Interval gol: {segment}")
    cached = result_cache_path(name, video_hash, candidate, (a, b), gait, target, extended, anchors, heel_source,
                               heel_stable and heel_source == HEEL_RTMPOSE, heel_model)
    limitation = limitation_for(name, uploaded, target)
    if cached.exists():
        saved = AnalysisResult.from_json(cached.read_text(encoding="utf-8"))
        saved = saved.model_copy(update={"limitation": limitation,
                                         "fallback_reason": fallback or saved.fallback_reason})
        return _with_horse(horse_skeleton, video_path, saved, horse_estimator, stages)
    stages = stages or P.Stages()
    stages.plan([P.SKELETON] + [P.VIEW] * extended + [P.HORSES, P.CAMERA, P.TRACKING] * bool(target)
                + [P.RTMPOSE] * (heel_source == HEEL_RTMPOSE) + [P.MEASURE] + [P.CAMERA] * extended
                + [P.STABLE] * (heel_stable and heel_source == HEEL_RTMPOSE) + [P.HORSE_SKELETON] * horse_skeleton)
    with stages.stage(P.SKELETON):
        raw = ST.load_raw(video_path, name, video_hash, candidate, predictor)
    if extended:
        with stages.stage(P.VIEW):
            raw = W.apply_profile(raw, ST.load_ratios(video_path, name, video_hash, candidate))
    if target:
        with stages.stage(P.HORSES):
            boxes = ST.load_boxes(video_path, name, video_hash)
        with stages.stage(P.CAMERA):
            motion = ST.load_motion(video_path, name, video_hash)
        with stages.stage(P.TRACKING):
            states = T.track_target(boxes, n_frames, [target, *anchors], motion, fps=fps)
            raw = T.filter_raw(raw, states, n_frames)
    orientation, facing_sources = None, {}
    if heel_source == HEEL_RTMPOSE:
        with stages.stage(P.RTMPOSE):
            raw, orientation, facing_sources = orient_rtmpose(video_path, name, video_hash, candidate, raw, fps,
                                                              (a, b), heel_estimator, heel_model)
    with stages.stage(P.MEASURE):
        frames, gated = apply_quality(video_path, name, video_hash, candidate, raw, frame_measures(raw, n_frames),
                                      heel_source, heel_estimator, heel_model)
    frames = frames[a:b + 1]
    if orientation is not None:
        frames = [f.model_copy(update={"facing": raw[f.frame]["points"]["facing"],
                                       "facing_source": facing_sources[f.frame]}) if f.frame in facing_sources else f
                  for f in frames]
    camera = None
    if extended:
        with stages.stage(P.CAMERA):
            camera = K.rotation_amplitude(ST.load_motion(video_path, name, video_hash), (a, b))
    ran_rtmpose = heel_source == HEEL_RTMPOSE and gated
    ran_stable = heel_stable and ran_rtmpose
    if ran_stable:
        with stages.stage(P.STABLE):
            frames = calcai_stabil.apply(video_path, name, video_hash, candidate, raw, frames, fps, heel_estimator,
                                         horse_estimator, heel_model)
    pipeline = PIPELINE_RTMPOSE if ran_rtmpose else PIPELINE_BASELINE
    no_skeleton = FALLBACK_NO_SKELETON if heel_source == HEEL_RTMPOSE and not gated else None
    result = AnalysisResult(
        video=name, video_hash=video_hash, gait=gait, fps=fps, n_frames=n_frames, segment=(a, b),
        candidate=candidate, trunk=summarize([f.trunk_deg for f in frames]),
        leg=summarize([f.leg_deg for f in frames]), unmeasured=unmeasured_intervals(frames), frames=frames,
        target=target, camera=camera, anchors=list(anchors), limitation=limitation,
        quality=_quality_summary(frames) if gated else None,
        heel_source=HEEL_RTMPOSE if ran_rtmpose else HEEL_MEDIAPIPE, orientation=orientation if gated else None,
        pipeline=pipeline, fallback_reason=no_skeleton, layers=_layers(pipeline, gated, ran_stable, heel_model))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(result.to_json(), encoding="utf-8")
    result = result.model_copy(update={"fallback_reason": fallback or no_skeleton})
    return _with_horse(horse_skeleton, video_path, result, horse_estimator, stages)


def _with_horse(horse_skeleton: bool, video_path: Path, result: AnalysisResult, horse_estimator: Callable | None,
                stages: P.Stages | None) -> AnalysisResult:
    """Cu opțiunea oprită, rezultatul neschimbat; pornită, cu scheletul calului (etapa are cache-ul ei)."""
    if not horse_skeleton:
        return result
    with (stages or P.Stages()).stage(P.HORSE_SKELETON):
        return schelet_cal.with_old_method(video_path, result, horse_estimator)


# ---------- clipuri încărcate: doar într-un folder temporar, care se șterge ----------

ROOT_NOT_ASCII = ("Folderul aplicației are în cale diacritice sau alte caractere speciale ({root}). Modelele MediaPipe "
                  "s-ar putea să nu se deschidă. Mută folderul într-o cale simplă, de exemplu C:\\postura, și "
                  "pornește din nou.")


def is_windows() -> bool:
    return sys.platform == "win32"


def root_path_problem(root: Path | None = None) -> str | None:
    """Pe Windows, mesajul pentru un folder al aplicației cu o cale non-ASCII: research/candidate.py dă
    modelele lui MediaPipe după cale. Pe Linux și Mac căile sunt UTF-8, deci nu e o problemă."""
    root = ST.ROOT if root is None else root
    return ROOT_NOT_ASCII.format(root=root) if is_windows() and not str(root).isascii() else None


def has_dev_set() -> bool:
    """Setul de dezvoltare există doar în proiect, nu și în copia dată altcuiva."""
    return ST.SPLIT_CSV.exists()


def ascii_fallback_temp() -> Path:
    return ST.results_dir().parent / "tmp"


def temp_root() -> Path:
    """Folderul temporar al sistemului, dacă are o cale ASCII (OpenCV și MediaPipe pot rata căile cu diacritice pe
    Windows, de exemplu C:\\Users\\Ștefan\\AppData\\Local\\Temp); altfel unul din folderul rezultatelor."""
    system = Path(tempfile.gettempdir())
    if str(system).isascii():
        return system
    fallback = ascii_fallback_temp()
    if not str(fallback).isascii():
        return system
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def upload_suffix(filename: str) -> str:
    suffix = Path(filename.strip()).suffix
    return suffix if suffix.lower() in UPLOAD_SUFFIXES else DEFAULT_SUFFIX


def with_temp_file(data: bytes, filename: str, fn):
    """Clipul încărcat, scris sub un nume ASCII simplu (clip + extensie), într-un folder temporar care se șterge."""
    with tempfile.TemporaryDirectory(prefix="ecvestru_", dir=temp_root(), ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / f"{TEMP_STEM}{upload_suffix(filename)}"   # OpenCV are nevoie doar de extensie
        path.write_bytes(data)
        return fn(path)


def upload_info(data: bytes, filename: str) -> tuple[int, float]:
    """(număr de cadre, fps) pentru un clip încărcat; fișierul temporar se șterge imediat."""
    return with_temp_file(data, filename, ST.video_info)


def upload_problem(n_frames: int) -> str | None:
    """Mesajul pentru un fișier încărcat care nu se poate analiza (None = se poate)."""
    if n_frames == 0:
        return "Fișierul nu poate fi citit ca video."
    if n_frames < MIN_UPLOAD_FRAMES:
        return f"Clipul are un singur cadru; analiza are nevoie de cel puțin {MIN_UPLOAD_FRAMES} cadre."
    return None


def analyze_upload(data: bytes, filename: str, gait: str, segment: tuple[int, int] | None = None,
                   predictor: Callable[[Path], Raw] | None = None,
                   target: TargetSeed | None = None, extended: bool = False,
                   anchors: Sequence[TargetSeed] = (), heel_source: str = HEEL_MEDIAPIPE,
                   heel_estimator: Callable | None = None, heel_stable: bool = False,
                   heel_model: str = HEEL_MODEL_M, stages: P.Stages | None = None, horse_skeleton: bool = False,
                   horse_estimator: Callable | None = None) -> AnalysisResult:
    """Analiza unui clip încărcat: rulează pe o copie temporară, care se șterge, deci videoul nu rămâne.
    Rămân rezultatul JSON și cache-urile pe cadre din brut/, vedere/, cai/, camera/ și schelet/ (UPLOAD_KEPT), toate
    după hash-ul conținutului: numele original nu ajunge pe disc."""
    def run(path: Path) -> AnalysisResult:
        # Intervalul implicit e tot clipul: numele unui fișier încărcat nu spune nimic despre conținut.
        whole_clip = (0, ST.video_info(path)[0] - 1)
        return analyze(path, gait, segment or whole_clip, name=ST.upload_name(ST.bytes_hash(data)), predictor=predictor,
                       target=target, extended=extended, anchors=anchors, uploaded=True, heel_source=heel_source,
                       heel_estimator=heel_estimator, heel_stable=heel_stable, heel_model=heel_model, stages=stages,
                       horse_skeleton=horse_skeleton, horse_estimator=horse_estimator)

    return with_temp_file(data, filename, run)
