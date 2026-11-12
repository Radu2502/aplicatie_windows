"""
Analiza pe călărețul țintă urmărit (urmărirea cu aspect și schimb de ID, k = 1 s), cu scheletul RTMPose-m al țintei.
Metoda veche (`masurare.analyze`, candidatul, un călăreț) rămâne disponibilă ca opțiune.

Etapele (pentru bara de progres): trecerea grea, cu o singură citire a clipului (pe fiecare cadru: detecția YOLOX-S
a oamenilor și a cailor, urmărirea ByteTrack, RTMPose-m pe oamenii a căror casetă atinge caseta unui cal) → ținta
→ unghiurile.

**Ținta** (`TargetNeeded` dacă nu se poate alege singură):
1. un click dat (în sesiune) → calul de sub click;
2. altfel, dacă în primele AUTO_SECONDS (= tinta.MAX_GAP_SECONDS, 2 s) există cel puțin un cadru cu un om pe un cal și
   pe niciun cadru nu sunt doi cai cu câte un om pe ei → ținta automată, calul acela;
3. altfel analiza cere un click pe calul călărețului.

**Punctele pe cadru** (același format ca brutul candidatului, apoi `masurare.frame_measures`): pe călărețul țintă,
partea cu șoldul RTMPose mai sigur (regula candidatului, cu scorul în loc de vizibilitate); umăr, șold, călcâi de pe
acea parte, fiecare lipsă sub CP.SCORE_MIN (0,3, pragul de afișare MMPose / rtmlib); sensul de mers nas față de
urechea părții (regula candidatului), apoi votul direcției labei RTMPose (aceleași constante) în interval; trunchiul
peste 45° (regula `_upright` din candidat) iese ca „trunchi_peste_45”; piciorul în afara intervalului plauzibil
(calitate.LEG_ANGLE_MIN … LEG_ANGLE_MAX) iese ca „unghi_neplauzibil”. Poarta A și fuziunea cu MediaPipe nu există aici
(nu e niciun schelet MediaPipe). Fără Streamlit.
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ecvestru import aspect_urmarire as AS
from ecvestru import cal_puncte as CP
from ecvestru import calitate as C
from ecvestru import masurare as M
from ecvestru import progres as P
from ecvestru import rtmpose as R
from ecvestru import stocare as ST
from ecvestru import tinta as T
from ecvestru import urmarire_multi as U
from ecvestru.rezultat import FACING_FOOT, FACING_NOSE_EAR, AnalysisResult, OrientationSummary, TargetSeed

TRACKING_VERSION = 1                 # se mărește când se schimbă urmărirea sau punctele (cheile de cache)
METHOD_AUTO, METHOD_TRACKED, METHOD_OLD = "automata", "urmarire", "veche"
AUTO_SECONDS = T.MAX_GAP_SECONDS
METHOD_LABELS = {METHOD_AUTO: f"Automată, după numărul de cai călăriți în primele {AUTO_SECONDS:g} s",
                 METHOD_TRACKED: "Urmărirea țintei (aspect + schimb de ID, k = 1 s)",
                 METHOD_OLD: "Metoda veche, un călăreț"}
# Metoda automată se alege după numărul de cai călăriți în primele AUTO_SECONDS ale clipului. Cu cel puțin
# RIDDEN_FOR_TRACKING → urmărirea țintei (ținta: click-ul dat, altfel se cere un click); altfel (unul sau
# niciunul) → metoda veche.
APP_DEFAULT_METHOD = METHOD_AUTO
RIDDEN_FOR_TRACKING = 2
CANDIDATE_LABEL = f"urmarire_aspect_k1@{TRACKING_VERSION}"
SCORE_MIN = CP.SCORE_MIN
CACHE_VARIANT = "final"
SHOULDER, HIP, HEEL, EAR = {"left": 5, "right": 6}, {"left": 11, "right": 12}, R.HEEL_INDEX, {"left": 3, "right": 4}
NOSE = 0
# Indicii MediaPipe citiți de regula `_upright` a candidatului
MP_SHOULDERS, MP_HIPS = (11, 12), (23, 24)

PASS = "Detecția, urmărirea și scheletele (YOLOX-S, ByteTrack, RTMPose-m), o singură citire a clipului"
TRACKING = "Ținta (calul ales, urmărit)"
ANGLES = "Unghiurile și calitatea"
STAGES = (PASS, TRACKING, ANGLES)
# Timpul estimat al trecerii grele: mediana măsurată pe 15 clipuri (în paralel, pe 16 nuclee): 1,05 s pe cadru
# (0,89–1,29 s). Doar pentru afișare.
ESTIMATE_SECONDS_PER_FRAME = 1.05
CLICK_NEEDED_TEXT = ("Ținta nu se poate alege singură (mai mulți călăreți sau niciunul în primele "
                     f"{AUTO_SECONDS:g} s). Bifează „Aleg călăreții dintr-un cadru” și dă click pe calul "
                     "Călărețului 1, apoi apasă din nou „Analizează”.")


def estimate_text(n_frames: int, cached: bool) -> str:
    if cached:
        return "Detecțiile, urmele și scheletele sunt deja calculate pentru acest clip: analiza durează câteva secunde."
    return (f"Timp estimat la prima analiză a clipului: ~{n_frames * ESTIMATE_SECONDS_PER_FRAME / 60:.0f} min "
            f"({n_frames} cadre).")


class TargetNeeded(ValueError):
    """Clipul are mai mulți călăreți (sau niciunul la început): analiza cere un click pe cal."""


# ---------- trecerea grea: detecții, urme, schelete (cache pe amprentă și versiune) ----------

def read_frames(video_path: Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Cadrele clipului, citite o singură dată, în ordine; cu `max_frames`, doar primele."""
    import cv2
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(video_path))
    try:
        count = 0
        while max_frames is None or count < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            count += 1
            yield image
    finally:
        capture.release()


def video_fps(video_path: Path) -> float:
    return ST.video_props(Path(video_path))[0] or U.BASE_FPS


def near_horse(tracks: U.ClipTracks, frame: int) -> list[int]:
    """Oamenii a căror casetă are o arie comună cu caseta unui cal (singurii care pot sta pe un cal)."""
    return [pid for pid, box in tracks.people[frame].items()
            if any(U.overlaps(box, h) for h in tracks.horses[frame].values())]


def run_pass(images: Iterable[np.ndarray], fps: float, detector, pose
             ) -> tuple[U.ClipTracks, list[dict[int, np.ndarray]]]:
    """Pe fiecare cadru, o dată: detecția (oameni și cai), urmărirea (ByteTrack, separat pentru oameni și cai; merge
    doar înainte, deci urma unui cadru depinde doar de cadrele de până la el) și RTMPose-m pe oamenii de lângă un cal.
    Puncte în `tracks.keypoints`; scorurile întoarse (cadru → ID → (26,))."""
    trackers = {cls: U.ByteTracker(fps) for cls in U.CLASSES}
    tracks = U.ClipTracks(fps=fps, n_frames=0)
    scores: list[dict[int, np.ndarray]] = []
    for image in images:
        found = detector(image)
        tracks.horses.append({t[0]: tuple(t[1:5]) for t in trackers[U.HORSE].update(found[U.HORSE])})
        tracks.people.append({t[0]: tuple(t[1:5]) for t in trackers[U.PERSON].update(found[U.PERSON])})
        tracks.keypoints.append({})
        scores.append({})
        frame = tracks.n_frames
        tracks.n_frames += 1
        for pid in near_horse(tracks, frame):
            points, score = pose(image, np.array(tracks.people[frame][pid]))
            tracks.keypoints[frame][pid] = np.asarray(points, np.float32)
            scores[frame][pid] = np.asarray(score, np.float32).reshape(-1)
    return tracks, scores


def one_pass(video_path: Path, threads: int = 1, max_frames: int | None = None
             ) -> tuple[U.ClipTracks, list[dict[int, np.ndarray]]]:
    """Trecerea grea cu o singură citire a clipului (YOLOX-S, ByteTrack, RTMPose-m)."""
    detector, pose = U.Detector(threads=threads), U.Pose(threads=threads)
    return run_pass(read_frames(video_path, max_frames), video_fps(video_path), detector, pose)


def cache_path(name: str, video_hash: str) -> Path:
    return U.cache_path(name, video_hash, variant=f"{CACHE_VARIANT}{TRACKING_VERSION}")


def save_pass(path: Path, tracks: U.ClipTracks, scores: list[dict[int, np.ndarray]]) -> None:
    U.save_tracks(path, tracks)
    rows = [(f, pid, *s) for f, row in enumerate(scores) for pid, s in row.items()]
    np.savez_compressed(path.with_suffix(".scoruri.npz"), rows=np.array(rows, np.float32).reshape(-1, 28))


def load_pass(path: Path) -> tuple[U.ClipTracks, list[dict[int, np.ndarray]]]:
    tracks = U.load_tracks(path)
    scores: list[dict[int, np.ndarray]] = [{} for _ in range(tracks.n_frames)]
    with np.load(path.with_suffix(".scoruri.npz")) as data:
        for row in data["rows"]:
            scores[int(row[0])][int(row[1])] = row[2:]
    return tracks, scores


def heavy_pass(video_path: Path, name: str, video_hash: str, stages: P.Stages, threads: int = 1
               ) -> tuple[U.ClipTracks, list[dict[int, np.ndarray]]]:
    path = cache_path(name, video_hash)
    if path.exists():
        return load_pass(path)
    started, cpu = time.perf_counter(), time.process_time()
    with stages.stage(PASS):
        tracks, scores = one_pass(video_path, threads)
    tracks.seconds, tracks.cpu_seconds = time.perf_counter() - started, time.process_time() - cpu
    save_pass(path, tracks, scores)
    return tracks, scores


# ---------- ținta ----------

def auto_target(tracks: U.ClipTracks) -> TargetSeed | None:
    """Regula 2: în primele AUTO_SECONDS, cel puțin un cadru cu un om pe un cal și niciun cadru cu doi cai călăriți;
    ținta = centrul calului de pe primul cadru cu călăreț. Altfel None (trebuie un click)."""
    last = min(tracks.n_frames, round(AUTO_SECONDS * tracks.fps) + 1)
    first = None
    for frame in range(last):
        ridden = [hid for hid, box in tracks.horses[frame].items()
                  if U.rider_on(box, tracks.people[frame], tracks.keypoints[frame]) is not None]
        if len(ridden) > 1:
            return None
        if ridden and first is None:
            first = (frame, tracks.horses[frame][ridden[0]])
    if first is None:
        return None
    frame, box = first
    return TargetSeed(frame=frame, x=(box[0] + box[2]) / 2, y=(box[1] + box[3]) / 2)


def first_frames(n_frames: int, fps: float) -> int:
    """Câte cadre intră în primele AUTO_SECONDS (aceeași fereastră ca la ținta automată)."""
    return min(n_frames, round(AUTO_SECONDS * fps) + 1)


def ridden_counts(tracks: U.ClipTracks) -> list[int]:
    """Numărul de cai cu un om pe ei (`U.rider_on`), pe fiecare cadru din primele AUTO_SECONDS."""
    return [sum(U.rider_on(box, tracks.people[f], tracks.keypoints[f]) is not None
                for box in tracks.horses[f].values())
            for f in range(first_frames(tracks.n_frames, tracks.fps))]


def ridden_horses(tracks: U.ClipTracks) -> int:
    """Caii călăriți în primele AUTO_SECONDS: cei mai mulți de pe un singur cadru."""
    return max(ridden_counts(tracks), default=0)


def method_for(ridden: int) -> str:
    """Metoda automată: urmărirea țintei cu cel puțin RIDDEN_FOR_TRACKING cai călăriți, altfel metoda veche."""
    return METHOD_TRACKED if ridden >= RIDDEN_FOR_TRACKING else METHOD_OLD


def horses_text(n: int) -> str:
    return "1 cal călărit" if n == 1 else f"{n} cai călăriți"


@dataclass(frozen=True)
class MethodChoice:
    """Metoda aleasă automat și motivul ei (numărul de cai călăriți în primele AUTO_SECONDS)."""
    ridden: int

    @property
    def method(self) -> str:
        return method_for(self.ridden)

    @property
    def text(self) -> str:
        then = "urmărirea țintei" if self.method == METHOD_TRACKED else "metoda veche (un călăreț)"
        return f"{horses_text(self.ridden)} în primele {AUTO_SECONDS:g} s → {then}"


def early_cache_path(name: str, video_hash: str, frames: int) -> Path:
    return U.cache_path(name, video_hash, max_frames=frames, variant=f"{CACHE_VARIANT}{TRACKING_VERSION}")


def early_pass(video_path: Path, frames: int, threads: int = 1
               ) -> tuple[U.ClipTracks, list[dict[int, np.ndarray]]]:
    """Trecerea grea doar pe primele `frames` cadre (detecția, urmărirea, scheletul); ByteTrack merge înainte, deci
    urmele sunt aceleași ca pe începutul trecerii întregi."""
    return one_pass(video_path, threads, max_frames=frames)


def method_choice(video_path: Path, name: str | None = None, threads: int = 1) -> MethodChoice:
    """Numără caii călăriți în primele AUTO_SECONDS: din trecerea întreagă salvată, altfel din trecerea scurtă (salvată
    și ea, ca să nu se repete)."""
    video_path = Path(video_path)
    name = name or video_path.name
    video_hash = ST.file_hash(video_path)
    full = cache_path(name, video_hash)
    if full.exists():
        return MethodChoice(ridden_horses(load_pass(full)[0]))
    n_frames, fps = ST.video_info(video_path)
    frames = first_frames(n_frames, fps or U.BASE_FPS)
    path = early_cache_path(name, video_hash, frames)
    if path.exists():
        return MethodChoice(ridden_horses(load_pass(path)[0]))
    tracks, scores = early_pass(video_path, frames, threads)
    save_pass(path, tracks, scores)
    return MethodChoice(ridden_horses(tracks))


def resolve_target(tracks: U.ClipTracks, video_hash: str, click: TargetSeed | None) -> tuple[TargetSeed, str]:
    """(ținta, de unde vine): click-ul dat, ținta automată; altfel TargetNeeded."""
    if click is not None:
        return click, "click"
    if (auto := auto_target(tracks)) is not None:
        return auto, "automata"
    raise TargetNeeded("Pe acest clip sunt mai mulți călăreți (sau niciunul în primele secunde): dă un click pe "
                       "calul călărețului.")


def choose(video_path: Path, tracks: U.ClipTracks, seed: TargetSeed) -> U.Choice:
    """Urmărirea cu aspect și schimb de ID (k = 1 s); fără un al doilea cal călărit nu există șablonul celuilalt,
    deci rămâne recuperarea prin eliminare (ancora pe cal), singura posibilă cu un singur călăreț."""
    seed_frame, horse_id = U.seed_horse(tracks.horses, seed, tracks.fps)
    anchored = U.choose(tracks, seed, recover=True, rider_by_horse=True)
    try:
        scores = AS.compute_scores(Path(video_path), tracks, seed_frame, horse_id, anchored.horse_ids)
    except ValueError:
        return anchored
    return AS.choose(tracks, seed_frame, horse_id, scores, swap_frames=round(AS.SWAP_SECONDS * tracks.fps))


# ---------- punctele pe cadru și orientarea ----------

def _upright(points: np.ndarray, width: float, height: float) -> bool:
    """Regula `_upright` a candidatului (trunchiul sub 45°), pe umerii și șoldurile RTMPose."""
    lm = [SimpleNamespace(x=0.0, y=0.0) for _ in range(max(MP_HIPS) + 1)]
    for mp, halpe in zip((*MP_SHOULDERS, *MP_HIPS), (5, 6, 11, 12), strict=True):
        lm[mp] = SimpleNamespace(x=float(points[halpe, 0]) / width, y=float(points[halpe, 1]) / height)
    return bool(ST.candidate_module()._upright(lm, width, height))


def frame_points(points: np.ndarray, score: np.ndarray) -> tuple[dict, str]:
    """Punctele unui cadru (umăr, șold, călcâi de pe partea cu șoldul mai sigur; sensul nas–ureche) și partea."""
    side = "left" if score[HIP["left"]] >= score[HIP["right"]] else "right"
    out = {"facing": 1.0 if points[NOSE, 0] >= points[EAR[side], 0] else -1.0}
    for name, index in (("shoulder", SHOULDER), ("hip", HIP), ("heel", HEEL)):
        i = index[side]
        out[name] = (float(points[i, 0]), float(points[i, 1])) if score[i] >= SCORE_MIN else None
    return out, side


def build_raw(tracks: U.ClipTracks, scores: list[dict[int, np.ndarray]], choice: U.Choice, width: float,
              height: float) -> tuple[M.Raw, list[str | None]]:
    """Brutul (format candidat) pe călărețul țintă și partea pe fiecare cadru."""
    cand = ST.candidate_module()
    raw, sides = {}, []
    for frame in range(tracks.n_frames):
        hid, pid = choice.horse_ids[frame], choice.rider_ids[frame]
        if hid is None:
            raw[frame], side = {"points": None, "motiv": T.REASON_LOST}, None
        elif pid is None or pid not in scores[frame]:
            raw[frame], side = {"points": None, "motiv": cand.REASON_NO_PERSON}, None
        else:
            points = tracks.keypoints[frame][pid]
            pts, side = frame_points(points, scores[frame][pid])
            reason = None
            if pts["shoulder"] is None or pts["hip"] is None:
                reason = cand.REASON_POINTS
            elif not _upright(points, width, height):
                reason = cand.REASON_TRUNK
            raw[frame] = {"points": pts, "motiv": reason}
        sides.append(side)
    return raw, sides


def foot_direction(tracks: U.ClipTracks, choice: U.Choice, sides: list[str | None], raw: M.Raw) -> np.ndarray:
    """Direcția labei RTMPose (mijlocul degetelor mare și mic − călcâiul, pe partea țintei), ca partea RTMPose a
    `rtmpose.foot_direction`; NaN pe cadrele fără orientare."""
    out = np.full(tracks.n_frames, np.nan)
    current = R.current_facing(raw, tracks.n_frames)
    for frame, side in enumerate(sides):
        if side is None or np.isnan(current[frame]):
            continue
        p = tracks.keypoints[frame][choice.rider_ids[frame]]
        big, small = (p[R.TOE_INDEX[side][k]] for k in (0, 1))
        out[frame] = R._sign(np.array([(big[0] + small[0]) / 2 - p[HEEL[side], 0]]))[0]
    return out


def orient(raw: M.Raw, foot: np.ndarray, fps: float, segment: tuple[int, int]
           ) -> tuple[M.Raw, OrientationSummary, dict[int, str]]:
    """Regula B4 (aceleași constante și același vot ca `rtmpose.orient`), doar în `segment`."""
    current = R.current_facing(raw, len(foot))
    voted, decided = R.vote_facing(current, foot, fps)
    inside = np.zeros(len(current), bool)
    inside[segment[0]:segment[1] + 1] = True
    known = inside & ~np.isnan(current)
    changed = known & (voted != current)
    out = dict(raw)
    for i in np.where(changed)[0]:
        out[int(i)] = {**raw[int(i)], "points": {**raw[int(i)]["points"], "facing": float(voted[i])}}
    share = R.minority_share(voted[known]) if known.any() else 0.0
    share = 0.0 if np.isnan(share) else share
    pct = round(100 * share)
    unclear = R.profile_unclear(share)
    summary = OrientationSummary(changed_frames=int(changed.sum()), foot_frames=int((known & decided).sum()),
                                 frames_with_facing=int(known.sum()), minority_share=share, minority_pct=pct,
                                 profile_unclear=unclear, reason=R.PROFILE_UNCLEAR_TEXT.format(pct=pct) if unclear
                                 else None)
    return out, summary, {int(i): FACING_FOOT if decided[i] else FACING_NOSE_EAR for i in np.where(known)[0]}


def plausible_legs(frames: list) -> list:
    """Intervalul plauzibil al piciorului (poarta B, constantele din calitate)."""
    return [f.model_copy(update={"leg_deg": None, "reason": M.REASON_IMPLAUSIBLE_ANGLE})
            if f.leg_deg is not None and not C.LEG_ANGLE_MIN <= f.leg_deg <= C.LEG_ANGLE_MAX else f for f in frames]


# ---------- analiza ----------

def result_cache_path(name: str, video_hash: str, segment: tuple[int, int], gait: str, seed: TargetSeed) -> Path:
    return M.result_cache_path(name, video_hash, CANDIDATE_LABEL, segment, gait, seed, False, (), M.HEEL_RTMPOSE)


def analyze(video_path: Path, gait: str, segment: tuple[int, int] | None = None, name: str | None = None,
            click: TargetSeed | None = None, stages: P.Stages | None = None, uploaded: bool = False,
            threads: int = 1) -> AnalysisResult:
    """Analiza pe ținta urmărită; TargetNeeded dacă trebuie un click. Același model de rezultat ca metoda veche."""
    video_path = Path(video_path)
    name = name or video_path.name
    video_hash = ST.file_hash(video_path)
    n_frames, fps = ST.video_info(video_path)
    segment = segment or M.default_segment(name, n_frames)
    a, b = max(0, segment[0]), min(n_frames - 1, segment[1])
    if a > b:
        raise ValueError(f"Interval gol: {segment}")
    stages = stages or P.Stages()
    stages.plan(STAGES)
    tracks, scores = heavy_pass(video_path, name, video_hash, stages, threads)
    seed, _ = resolve_target(tracks, video_hash, click)
    cached = result_cache_path(name, video_hash, (a, b), gait, seed)
    limitation = M.limitation_for(name, uploaded, seed)
    if cached.exists():
        saved = AnalysisResult.from_json(cached.read_text(encoding="utf-8"))
        return saved.model_copy(update={"limitation": limitation})
    _, width, height = ST.video_props(video_path)        # doar metadatele, fără cadre
    with stages.stage(TRACKING):
        choice = choose(video_path, tracks, seed)
    with stages.stage(ANGLES):
        raw, sides = build_raw(tracks, scores, choice, width, height)
        raw, orientation, sources = orient(raw, foot_direction(tracks, choice, sides, raw), fps, (a, b))
        frames = plausible_legs(M.frame_measures(raw, n_frames))[a:b + 1]
        frames = [f.model_copy(update={"facing": raw[f.frame]["points"]["facing"], "facing_source": sources[f.frame]})
                  if f.frame in sources else f for f in frames]
    result = AnalysisResult(
        video=name, video_hash=video_hash, gait=gait, fps=fps, n_frames=n_frames, segment=(a, b),
        candidate=CANDIDATE_LABEL, trunk=M.summarize([f.trunk_deg for f in frames]),
        leg=M.summarize([f.leg_deg for f in frames]), unmeasured=M.unmeasured_intervals(frames), frames=frames,
        target=seed, limitation=limitation, quality=M._quality_summary(frames), heel_source=M.HEEL_RTMPOSE,
        orientation=orientation, pipeline=M.PIPELINE_RTMPOSE, layers=M._layers(M.PIPELINE_RTMPOSE, True))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(result.to_json(), encoding="utf-8")
    return result


def needs_click(video_path: Path) -> bool:
    """Doar din cache-ul trecerii grele, fără să ruleze vreun model: True dacă ținta nu se poate alege singură (fără
    țintă automată). Fără cache: False, pentru că decizia vine abia după trecere."""
    video_path = Path(video_path)
    video_hash = ST.file_hash(video_path)
    path = cache_path(video_path.name, video_hash)
    if not path.exists():
        return False
    tracks, _ = load_pass(path)
    return auto_target(tracks) is None
