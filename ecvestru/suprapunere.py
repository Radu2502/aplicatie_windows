"""
Filmul analizat: peste fiecare cadru din interval se desenează scheletul complet al călărețului ales (dacă
există în cache), cu punctele colorate după visibility, și o bandă cu unghiurile rotunjite sau cu motivul
cadrului nemăsurat. Lanțul umăr–șold–călcâi și verticala apar doar peste unghiurile măsurate. Pe cadrele
nemăsurate, persoanele respinse de candidat apar punctat, într-o singură culoare, cu o etichetă.
Același desen, cu `blur=True`, estompează tot cadrul (pentru figuri fără fețe). Cu `smooth=True`, lanțul desenat
vine din punctele netezite pentru afișare (A2, mai jos); valorile din bandă, rezultatul și JSON-ul rămân cele
măsurate. Fără Streamlit.

Filmul se codează cu ffmpeg (din pachetul imageio-ffmpeg, altfel din PATH) într-un fișier dintr-un folder temporar,
se citește ca octeți și folderul se șterge imediat: merge la fel pe Linux, Windows și Mac. Un clip încărcat are nevoie
doar de copia temporară scurtă din care îl citește OpenCV. Orice eroare la film devine FilmError (un RuntimeError),
ca pagina să arate analiza și fără film.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont

from ecvestru import cal_puncte as CP
from ecvestru import calitate as C
from ecvestru import masurare as M
from ecvestru import rtmpose
from ecvestru import schelet as Z
from ecvestru import schelet_cal as SC
from ecvestru import stocare as ST
from ecvestru.rezultat import HEEL_RTMPOSE, AnalysisResult, FrameMeasure

FFMPEG: str | None = None   # None: ffmpeg din imageio-ffmpeg, apoi din PATH; un șir: exact acel program
FILM_NAME = "film.mp4"
H264_PRESET, H264_CRF = "veryfast", "23"
FONT_PATH = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"   # are diacritice și semnul de grad

TRUNK_BGR = (0, 220, 255)      # galben
LEG_BGR = (0, 140, 255)        # portocaliu
VERTICAL_BGR = (255, 200, 120) # albastru deschis
SKELETON_BGR = (220, 220, 220) # gri deschis, sub lanț
OUTLINE_BGR = (20, 20, 20)     # conturul închis: linia se vede și pe fond deschis
FAR_OPACITY = 0.45             # cealaltă parte decât cea folosită de candidat
SIDE_LEGEND = ("Partea folosită de candidat e desenată plin; cealaltă parte e subțire și estompată. Se desenează doar "
               "capul, brațele, trunchiul și picioarele.")
HIGH_BGR = (60, 200, 0)        # verde, punct plin
MID_BGR = (255, 0, 200)        # mov, punct plin
LOW_BGR = (0, 0, 255)          # roșu, cerc gol: treapta se vede și fără culoare
_MID, _HIGH = (str(limit).replace(".", ",") for limit in (Z.VISIBILITY_MID, Z.VISIBILITY_HIGH))
VISIBILITY_LEGEND = (f"Punctele, după visibility: verde = peste {_HIGH}, mov = {_MID}–{_HIGH}, "
                     f"cerc roșu gol = sub {_MID}.")
REJECTED_BGR = (160, 160, 160) # o singură culoare neutră, doar linii punctate
LABEL_BACK_BGR = (40, 40, 40)
DASH_PX = 8
REJECTED_LABEL = "persoană respinsă: {}"
REJECTED_LEGEND = ("Linii punctate gri, cu etichetă = persoană respinsă de candidat (poate fi instructorul sau alt "
                   "călăreț), pe cadrele nemăsurate; nu e măsurată.")
# Netezirea, doar pentru afișare: Butterworth fără întârziere, pe segmente
SMOOTH_ORDER = 4
SMOOTH_CUTOFF_HZ = 6.0
SMOOTH_MIN_FRAMES = 15         # segmentele mai scurte rămân nenetezite
SMOOTH_PADLEN = 15             # extensia impară la capete, ca filtfilt din scipy (3 × max(len(a), len(b)))
SMOOTH_NOTE = ("Netezire pentru afișare: lanțul din film și curbele din grafic sunt netezite (Butterworth, 6 Hz, pe "
               "segmente fără goluri și fără schimbări de sursă sau de orientare). Valorile din bandă, medianele, "
               "raportul și JSON-ul nu se schimbă.")
HORSE_BGR = (255, 255, 0)      # turcoaz: scheletul calului (opțiunea „Scheletul calului”), altă culoare decât călărețul
HORSE_LEGEND = (f"Turcoaz = scheletul calului (AP-10K), doar punctele cu scor ≥ {str(SC.SCORE_MIN).replace('.', ',')}; "
                "doar afișare, nicio valoare nu vine din el.")
_HORSE_BONES = tuple((CP.AP10K.index(a), CP.AP10K.index(b)) for a, b in CP.AP10K_BONES)
BAND_BGR = (0, 0, 0)
VIEW_TEXT = "Vederea (router V5, experimental): {}"   # banda de jos, doar cu opțiunea „Vederea călărețului”
VIEW_NAMES = {"profil": "profil", "fata": "față", "spate": "spate", "oblic": "oblic", "nu_se_vede": "nu se vede"}
LINE_DIVISOR = 300             # grosimea liniilor: latura mică a cadrului / atât
BAND_FRACTION = 0.05           # înălțimea benzii, din latura mică a cadrului
BAND_MIN_PX = 22
BAND_MARGIN_PX = 6
FONT_MIN_PX = 8
BLUR_FRAME_FRACTION = 0.03     # estomparea minimă, din latura mică a cadrului
BLUR_TORSO_FRACTION = 0.25     # estomparea crește cu trunchiul: un călăreț aproape are fața mai mare


def _degrees(value: float) -> str:
    rounded = M.round_to(value)
    return f"{rounded:+d}°" if rounded else "0°"


def band_text(measure: FrameMeasure) -> str:
    parts = [f"#{measure.frame}"]
    if measure.trunk_deg is not None:
        parts.append(f"trunchi {_degrees(measure.trunk_deg)}")
    if measure.leg_deg is not None:
        parts.append(f"picior {_degrees(measure.leg_deg)}")
    if measure.reason is not None:
        parts.append(f"Nemăsurat: {M.REASON_TEXTS.get(measure.reason, measure.reason)}")
    if measure.warnings:
        parts.append("Atenție: " + ", ".join(C.SIGNAL_SHORT.get(code, code) for code in measure.warnings))
    return " · ".join(parts)


def band_height(img: np.ndarray) -> int:
    return max(BAND_MIN_PX, round(min(img.shape[:2]) * BAND_FRACTION))


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def band_font(text: str, width: int, height: int = BAND_MIN_PX) -> ImageFont.FreeTypeFont:
    """Cel mai mare font care încape în bandă; textul lung de pe clipurile înguste se micșorează."""
    usable = width - 2 * BAND_MARGIN_PX
    for size in range(max(FONT_MIN_PX, height - BAND_MARGIN_PX), FONT_MIN_PX, -1):
        if _font(size).getlength(text) <= usable:
            return _font(size)
    return _font(FONT_MIN_PX)


def _draw_band(img: np.ndarray, text: str) -> None:
    h, w = band_height(img), img.shape[1]
    strip = Image.new("L", (w, h), 0)
    font = band_font(text, w, h)
    ImageDraw.Draw(strip).text((BAND_MARGIN_PX, h / 2), text, fill=255, font=font, anchor="lm")
    img[:h] = BAND_BGR
    img[:h][np.asarray(strip) > 0] = (255, 255, 255)


def _draw_view(img: np.ndarray, view: str) -> None:
    """Banda de jos cu vederea routerului V5 (doar afișare)."""
    text = VIEW_TEXT.format(VIEW_NAMES.get(view, view))
    h, w = band_height(img), img.shape[1]
    strip = Image.new("L", (w, h), 0)
    ImageDraw.Draw(strip).text((BAND_MARGIN_PX, h / 2), text, fill=255, font=band_font(text, w, h), anchor="lm")
    img[-h:] = BAND_BGR
    img[-h:][np.asarray(strip) > 0] = (255, 255, 255)


def blur_sigma(height: int, width: int, points: dict | None) -> float:
    sigma = min(height, width) * BLUR_FRAME_FRACTION
    if points and points.get("shoulder") is not None and points.get("hip") is not None:
        sigma = max(sigma, math.dist(points["shoulder"], points["hip"]) * BLUR_TORSO_FRACTION)
    return sigma


def _pixel(point) -> tuple[int, int]:
    return round(point[0]), round(point[1])


def _draw_point(img: np.ndarray, point, value: float, thick: int, size: int) -> None:
    tier = Z.visibility_tier(value)
    if tier == Z.TIER_LOW:
        cv2.circle(img, _pixel(point), size + 1, LOW_BGR, thick + 1, cv2.LINE_AA)
    elif tier is not None:
        cv2.circle(img, _pixel(point), size, HIGH_BGR if tier == Z.TIER_HIGH else MID_BGR, -1, cv2.LINE_AA)


def _draw_bones(img: np.ndarray, bones: tuple) -> None:
    """`bones`: puncte în pixeli, visibility și, opțional, partea folosită de candidat. Cealaltă parte se
    desenează subțire, pe un strat estompat, iar partea folosită plin, cu contur, peste ea."""
    xy, visibility, *rest = bones
    side = rest[0] if rest else ""
    thick = max(1, round(min(img.shape[:2]) / LINE_DIVISOR))
    seen = [(a, b) for a, b in Z.DRAWN_LINES
            if visibility[a] >= M.V.MIN_VISIBILITY and visibility[b] >= M.V.MIN_VISIBILITY]   # False și pe NaN
    far_lines = [line for line in seen if Z.is_far_line(line, side)]
    far_points = [i for i in Z.DRAWN_POINTS if Z.is_far_point(i, side) and not np.isnan(visibility[i])]
    if far_lines or far_points:
        faded = img.copy()
        for a, b in far_lines:
            cv2.line(faded, _pixel(xy[a]), _pixel(xy[b]), SKELETON_BGR, thick, cv2.LINE_AA)
        for i in far_points:
            _draw_point(faded, xy[i], visibility[i], thick, thick + 1)
        cv2.addWeighted(faded, FAR_OPACITY, img, 1 - FAR_OPACITY, 0, dst=img)
    near = [line for line in seen if line not in far_lines]
    for a, b in near:
        cv2.line(img, _pixel(xy[a]), _pixel(xy[b]), OUTLINE_BGR, thick + 4, cv2.LINE_AA)
    for a, b in near:
        cv2.line(img, _pixel(xy[a]), _pixel(xy[b]), SKELETON_BGR, thick + 1, cv2.LINE_AA)
    for i in Z.DRAWN_POINTS:
        if i not in far_points:
            _draw_point(img, xy[i], visibility[i], thick, 2 * thick + 1)


def rejected_label(reason: str) -> str:
    return REJECTED_LABEL.format(M.REASON_TEXTS.get(reason, reason))


def _dashed(img: np.ndarray, start: tuple[int, int], end: tuple[int, int], thick: int) -> None:
    steps = max(1, round(math.dist(start, end) / DASH_PX))
    for step in range(0, steps, 2):
        a = (round(start[0] + (end[0] - start[0]) * step / steps), round(start[1] + (end[1] - start[1]) * step / steps))
        last = min(step + 1, steps)
        b = (round(start[0] + (end[0] - start[0]) * last / steps), round(start[1] + (end[1] - start[1]) * last / steps))
        cv2.line(img, a, b, OUTLINE_BGR, thick + 2, cv2.LINE_AA)
        cv2.line(img, a, b, REJECTED_BGR, thick, cv2.LINE_AA)


def _draw_label(img: np.ndarray, text: str, x: int, bottom: int) -> None:
    """Eticheta deasupra punctului cel mai de sus; rămâne în cadru și sub bandă."""
    height, width = band_height(img), img.shape[1]
    font = band_font(text, width, height)
    wide = min(width, round(font.getlength(text)) + 2 * BAND_MARGIN_PX)
    left = min(max(0, x), width - wide)
    top = min(max(height, bottom - height), img.shape[0] - height)
    mask = Image.new("L", (wide, height), 0)
    ImageDraw.Draw(mask).text((BAND_MARGIN_PX, height / 2), text, fill=255, font=font, anchor="lm")
    img[top:top + height, left:left + wide] = LABEL_BACK_BGR
    img[top:top + height, left:left + wide][np.asarray(mask) > 0] = (255, 255, 255)


def _draw_rejected(img: np.ndarray, people: list) -> None:
    thick = max(1, round(min(img.shape[:2]) / LINE_DIVISOR)) + 1
    for xy, visibility, reason in people:
        drawn = []
        for a, b in Z.DRAWN_LINES:
            if visibility[a] >= M.V.MIN_VISIBILITY and visibility[b] >= M.V.MIN_VISIBILITY:
                _dashed(img, _pixel(xy[a]), _pixel(xy[b]), thick)
                drawn += [_pixel(xy[a]), _pixel(xy[b])]
        if drawn:
            x, y = min(drawn, key=lambda point: point[1])
            _draw_label(img, rejected_label(reason), x, y)


def _draw_chain(img: np.ndarray, points: dict, measure: FrameMeasure) -> None:
    thick = max(1, round(min(img.shape[:2]) / LINE_DIVISOR))
    hip = _pixel(points["hip"])
    shoulder, heel = _pixel(points["shoulder"]), None
    if measure.leg_deg is not None:
        heel = _pixel(points["heel"])
    bottom = heel[1] if heel else hip[1] + (hip[1] - shoulder[1])
    cv2.line(img, (hip[0], shoulder[1]), (hip[0], bottom), VERTICAL_BGR, thick, cv2.LINE_AA)
    cv2.line(img, shoulder, hip, TRUNK_BGR, 2 * thick + 1, cv2.LINE_AA)
    if heel:
        cv2.line(img, hip, heel, LEG_BGR, 2 * thick + 1, cv2.LINE_AA)
        cv2.circle(img, heel, 3 * thick + 1, LEG_BGR, -1, cv2.LINE_AA)
    cv2.circle(img, shoulder, 3 * thick + 1, TRUNK_BGR, -1, cv2.LINE_AA)
    cv2.circle(img, hip, 3 * thick + 1, TRUNK_BGR, -1, cv2.LINE_AA)


def _draw_horse(img: np.ndarray, points: list) -> None:
    """Scheletul calului (17 puncte AP-10K, None = nedesenat), sub călăreț, în HORSE_BGR, cu contur închis."""
    thick = max(1, round(min(img.shape[:2]) / LINE_DIVISOR))
    lines = [(points[a], points[b]) for a, b in _HORSE_BONES if points[a] is not None and points[b] is not None]
    for a, b in lines:
        cv2.line(img, _pixel(a), _pixel(b), OUTLINE_BGR, thick + 3, cv2.LINE_AA)
    for a, b in lines:
        cv2.line(img, _pixel(a), _pixel(b), HORSE_BGR, thick, cv2.LINE_AA)
    for p in points:
        if p is not None:
            cv2.circle(img, _pixel(p), 2 * thick + 1, HORSE_BGR, -1, cv2.LINE_AA)


def overlay_frame(frame: np.ndarray, points: dict | None, measure: FrameMeasure, blur: bool = False,
                  bones: tuple | None = None, rejected: list | tuple = (), horse: list | None = None,
                  view: str | None = None) -> np.ndarray:
    """Cadrul cu scheletul (`bones`, vezi `_draw_bones`), pe orice cadru cu călăreț, și banda.
    Lanțul apare doar pe unghiurile măsurate, niciodată pe cele respinse, deci nimic nu pare măsurat.
    `rejected` (puncte în pixeli, visibility, motiv) se desenează doar pe cadrele nemăsurate. `horse` (cele 17 puncte
    ale calului, None = nedesenat) se desenează primul, sub călăreț."""
    h, w = frame.shape[:2]
    img = cv2.GaussianBlur(frame, (0, 0), blur_sigma(h, w, points)) if blur else frame.copy()
    if horse is not None:
        _draw_horse(img, horse)
    if bones is not None:
        _draw_bones(img, bones)
    if measure.trunk_deg is None:
        _draw_rejected(img, rejected)
    elif points:
        _draw_chain(img, points, measure)
    _draw_band(img, band_text(measure))
    if view is not None:
        _draw_view(img, view)
    return img


# ---------- netezirea pentru afișare (A2) ----------

def butter_lowpass(cutoff_hz: float, fps: float, order: int = SMOOTH_ORDER) -> tuple[np.ndarray, np.ndarray] | None:
    """Coeficienții (b, a) ai unui Butterworth trece-jos digital (transformarea biliniară cu predistorsionare, ca
    scipy.signal.butter); None dacă frecvența de tăiere nu e sub Nyquist."""
    wn = cutoff_hz / (fps / 2)
    if not 0 < wn < 1:
        return None
    warped = 4 * math.tan(math.pi * wn / 2)                     # fs = 2, ca în scipy
    analog = warped * np.exp(1j * math.pi * (2 * np.arange(order) + order + 1) / (2 * order))
    poles = (4 + analog) / (4 - analog)
    a = np.real(np.poly(poles))
    b = np.real(np.poly(-np.ones(order)))
    return b * a.sum() / b.sum(), a                             # câștig 1 la frecvența zero


def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray, zi: np.ndarray) -> np.ndarray:
    """Forma directă II transpusă, cu starea inițială `zi`."""
    z, out = zi.copy(), np.empty_like(x)
    for n, value in enumerate(x):
        y = b[0] * value + z[0]
        z[:-1] = z[1:] + b[1:-1] * value - a[1:-1] * y
        z[-1] = b[-1] * value - a[-1] * y
        out[n] = y
    return out


def _steady_state(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Starea filtrului pentru un semnal constant egal cu 1 (ca scipy.signal.lfilter_zi)."""
    n = len(a) - 1
    companion = np.zeros((n, n))
    companion[0] = -a[1:]
    companion[1:, :-1] = np.eye(n - 1)
    return np.linalg.solve(np.eye(n) - companion.T, b[1:] - a[1:] * b[0])


def zero_phase(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Filtrare înainte și înapoi, fără întârziere (ca scipy.signal.filtfilt, extensie impară la capete)."""
    pad = min(SMOOTH_PADLEN, len(x) - 1)
    ext = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2:-pad - 2:-1]])
    zi = _steady_state(b, a)
    y = _lfilter(b, a, ext, zi * ext[0])
    y = _lfilter(b, a, y[::-1], zi * y[-1])[::-1]
    return y[pad:len(y) - pad]


def display_segments(keys: list) -> list[tuple[int, int]]:
    """Cadre consecutive cu aceeași cheie (sursă, orientare); None = gol, care taie segmentul."""
    out: list[tuple[int, int]] = []
    for i, key in enumerate(keys):
        if key is None:
            continue
        if out and out[-1][1] == i - 1 and keys[i - 1] == key:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def smooth_track(values: np.ndarray, keys: list, fps: float) -> np.ndarray:
    """Punctele (N, 2) netezite pe fiecare segment de cel puțin SMOOTH_MIN_FRAMES cadre; restul rămâne cum e."""
    out = np.array(values, dtype=float)
    coefficients = butter_lowpass(SMOOTH_CUTOFF_HZ, fps)
    if coefficients is None:
        return out
    for start, end in display_segments(keys):
        if end - start + 1 >= SMOOTH_MIN_FRAMES:
            for axis in range(out.shape[1]):
                out[start:end + 1, axis] = zero_phase(*coefficients, out[start:end + 1, axis])
    return out


@dataclass(frozen=True)
class DisplayTrack:
    """Ce se afișează cu netezirea pornită: brutul cu lanțul netezit (pentru film) și unghiurile din el (pentru grafic),
    câte unul pe fiecare element din `frames` (None = nemăsurat)."""
    raw: dict
    trunk: list
    leg: list


def smooth_display(raw: M.Raw, frames: list[FrameMeasure], fps: float, skeleton_source: np.ndarray | None = None,
                   heel_sources: dict[int, str] | None = None) -> DisplayTrack:
    """Netezirea pentru afișare, pe cadrele din `frames`. Umărul și șoldul pe segmentele cu trunchi măsurat, călcâiul pe
    cele cu picior măsurat; un segment se taie la goluri, la schimbarea trecerii MediaPipe (`skeleton_source`), a sursei
    călcâiului (`heel_sources`, calea RTMPose) și a orientării (cea a cadrului, altfel cea din brut). Nimic din ce
    primește nu se schimbă."""
    def facing(f: FrameMeasure) -> float:
        return f.facing if f.facing is not None else raw[f.frame]["points"]["facing"]

    def source(f: FrameMeasure):
        return None if skeleton_source is None else int(skeleton_source[f.frame])

    def track(name: str, measured) -> np.ndarray:
        return np.array([raw[f.frame]["points"][name] if measured(f) else (np.nan, np.nan) for f in frames], float)

    has_trunk = lambda f: f.trunk_deg is not None  # noqa: E731
    has_leg = lambda f: f.leg_deg is not None  # noqa: E731
    trunk_keys = [(source(f), facing(f)) if has_trunk(f) else None for f in frames]
    leg_keys = [(source(f), facing(f), (heel_sources or {}).get(f.frame)) if has_leg(f) else None for f in frames]
    shoulder = smooth_track(track("shoulder", has_trunk), trunk_keys, fps)
    hip = smooth_track(track("hip", has_trunk), trunk_keys, fps)
    heel = smooth_track(track("heel", has_leg), leg_keys, fps)
    out, trunk, leg = dict(raw), [], []
    for k, f in enumerate(frames):
        if not has_trunk(f):
            trunk.append(None)
            leg.append(None)
            continue
        points = {**raw[f.frame]["points"], "shoulder": tuple(map(float, shoulder[k])),
                  "hip": tuple(map(float, hip[k]))}
        if has_leg(f):
            points["heel"] = tuple(map(float, heel[k]))
        out[f.frame] = {**raw[f.frame], "points": points}
        trunk.append(M.angle_from_vertical(points["shoulder"], points["hip"], facing(f)))
        leg.append(M.angle_from_vertical(points["hip"], points["heel"], facing(f)) if has_leg(f) else None)
    return DisplayTrack(out, trunk, leg)


class FilmError(RuntimeError):
    """Filmul nu s-a putut face; analiza rămâne valabilă."""


def ffmpeg_path() -> str:
    """Programul ffmpeg: cel ales explicit, apoi cel din imageio-ffmpeg, apoi cel din PATH."""
    if FFMPEG:
        return FFMPEG
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:   # pachet lipsă sau fără binar pentru această platformă
        found = shutil.which("ffmpeg")
        if found:
            return found
    raise FilmError("Nu am putut face filmul: ffmpeg lipsește.")


def _encoder(width: int, height: int, fps: float, target: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [ffmpeg_path(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
         "-r", f"{fps:.3f}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
         "-preset", H264_PRESET, "-crf", H264_CRF, "-g", str(max(1, round(fps))), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-f", "mp4", str(target)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def film_bytes(video_path: Path, frames: list[FrameMeasure], raw: M.Raw, fps: float, blur: bool = False,
               skeleton=None, horse: dict | None = None, views: dict | None = None) -> bytes:
    """MP4 H.264 cu câte un cadru pentru fiecare element din `frames`; fișierul temporar se șterge imediat. `horse`:
    cadru → cele 17 puncte ale calului (schelet_cal.drawn_points); None sau gol = fără scheletul calului."""
    wanted = {f.frame: f for f in frames}
    pixels = None if skeleton is None else skeleton.xy_px
    with tempfile.TemporaryDirectory(prefix="ecvestru_film_", dir=M.temp_root(), ignore_cleanup_errors=True) as tmp:
        return _encode(video_path, wanted, raw, fps, blur, skeleton, pixels, Path(tmp) / FILM_NAME, horse or {},
                       views or {})


def _encode(video_path: Path, wanted: dict, raw: M.Raw, fps: float, blur: bool, skeleton, pixels,
            target: Path, horse: dict, views: dict | None = None) -> bytes:
    views = views or {}
    cap = cv2.VideoCapture(str(video_path))
    encoder = None
    try:
        for index in range(max(wanted) + 1):
            if not cap.grab():
                break
            if index not in wanted:
                continue
            frame = cap.retrieve()[1]
            encoder = encoder or _encoder(frame.shape[1], frame.shape[0], fps, target)
            points = (raw.get(index) or {}).get("points")
            bones = None if pixels is None else (pixels[index], skeleton.visibility[index], str(skeleton.side[index]))
            others = () if skeleton is None else skeleton.shown_rejected_at(index)
            encoder.stdin.write(overlay_frame(frame, points, wanted[index], blur, bones, others,
                                              horse.get(index), views.get(index)).tobytes())
        if encoder is None:
            raise FilmError("Nu am putut face filmul: clipul nu are cadre în interval.")
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise FilmError("Nu am putut face filmul: ffmpeg a eșuat.")
        return target.read_bytes()
    except BrokenPipeError as error:
        raise FilmError("Nu am putut face filmul: ffmpeg s-a oprit.") from error
    except (OSError, ValueError, subprocess.SubprocessError) as error:   # ffmpeg lipsă, fișier, proces
        raise FilmError(f"Nu am putut face filmul: {error}") from error
    finally:
        cap.release()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()


def _shown_raw(video_path: Path, result: AnalysisResult, smooth: bool):
    """Brutul desenat (pe calea RTMPose, cu călcâiul folosit în măsurare), scheletul și, cu `smooth`, netezirea."""
    raw = ST.load_raw(Path(video_path), result.video, result.video_hash, result.candidate)
    skeleton = ST.load_rider_skeleton(Path(video_path), result.video, result.video_hash, result.candidate)
    sources = None
    if result.heel_source == HEEL_RTMPOSE and skeleton is not None:
        raw, sources = _rtmpose_raw(Path(video_path), result, raw, skeleton)
    shown = None
    if smooth:
        shown = smooth_display(raw, result.frames, result.fps, None if skeleton is None else skeleton.source,
                               sources)
    return raw, skeleton, shown


def clip_film(video_path: Path, result: AnalysisResult, blur: bool = False, smooth: bool = False) -> bytes:
    """Filmul unui rezultat, din punctele deja salvate în cache-ul brut; cu `smooth`, lanțul netezit pentru afișare. Cu
    scheletul calului în rezultat (opțiunea pornită), îl desenează și pe el."""
    raw, skeleton, shown = _shown_raw(video_path, result, smooth)
    horse = SC.drawn_points(result.horse_keypoints)
    views = drawn_views(result)
    extra = {k: v for k, v in (("horse", horse), ("views", views)) if v}   # fără opțiuni: apelul de dinainte
    return film_bytes(Path(video_path), result.frames, raw if shown is None else shown.raw, result.fps, blur, skeleton,
                      **extra)


def drawn_views(result: AnalysisResult) -> dict[int, str]:
    """Pentru film: cadru → vederea routerului V5 (doar cu opțiunea pornită și doar cadrele cu vedere)."""
    if result.rider_views is None:
        return {}
    return {f.frame: f.view for f in result.rider_views.frames if f.view is not None}


def display_series(video_path: Path, result: AnalysisResult, smooth: bool) -> tuple[list, list]:
    """Unghiurile trunchiului și piciorului pentru grafic: cele măsurate sau, cu `smooth`, cele netezite pentru
    afișare."""
    if not smooth:
        return [f.trunk_deg for f in result.frames], [f.leg_deg for f in result.frames]
    _, _, shown = _shown_raw(video_path, result, smooth)
    return shown.trunk, shown.leg


def upload_display_series(data: bytes, filename: str, result: AnalysisResult, smooth: bool) -> tuple[list, list]:
    """Ca display_series, pentru un clip încărcat (copia temporară se șterge)."""
    return M.with_temp_file(data, filename, lambda path: display_series(path, result, smooth))


def _rtmpose_raw(video_path: Path, result: AnalysisResult, raw: M.Raw, skeleton) -> tuple[M.Raw, dict[int, str]]:
    """Pe calea RTMPose filmul desenează călcâiul folosit în măsurare, nu cel MediaPipe (din modelul m sau l al
    rezultatului); plus sursa lui pe cadru."""
    try:
        points = M.rtmpose_points(video_path, result.video, result.video_hash, skeleton,
                                  heel_model=M.heel_model_of(result.layers))
    except (ImportError, OSError) as error:
        raise FilmError(f"Nu am putut face filmul: punctele RTMPose lipsesc ({error}).") from error
    boxes = rtmpose.horse_boxes(skeleton, ST.load_boxes(video_path, result.video, result.video_hash))
    heels = rtmpose.side_heels(points, skeleton)
    return (rtmpose.film_raw(raw, result.frames, heels, boxes),
            rtmpose.heel_sources(raw, result.frames, heels, boxes))


def candidate_film(video_path: Path) -> bytes:
    """Filmul candidatului pe tot clipul, fără rezultat salvat (fără alură și fără interval)."""
    video_path = Path(video_path)
    name, video_hash, candidate = video_path.name, ST.file_hash(video_path), ST.candidate_label()
    n_frames, fps = ST.video_info(video_path)
    raw = ST.load_raw(video_path, name, video_hash, candidate)
    skeleton = ST.load_rider_skeleton(video_path, name, video_hash, candidate)
    frames, _ = M.apply_quality(video_path, name, video_hash, candidate, raw, M.frame_measures(raw, n_frames))
    return film_bytes(video_path, frames, raw, fps, skeleton=skeleton)


def upload_film(data: bytes, filename: str, result: AnalysisResult, blur: bool = False,
                smooth: bool = False) -> bytes:
    """Filmul unui clip încărcat: citit din copia temporară, care se șterge; rămâne doar în memorie."""
    return M.with_temp_file(data, filename, lambda path: clip_film(path, result, blur, smooth))
