"""
CANDIDATUL: metoda veche care găsește călărețul și punctele umăr, șold, călcâi.

predict(video_path, frames) primește un clip și lista cadrelor adnotate (indici de la 0) și
întoarce, pentru fiecare cadru, unghiurile față de verticală (grade, cu semn):

    trunk_deg: > 0 = trunchi aplecat în sensul de mers, < 0 = pe spate
    leg_deg:   < 0 = călcâi în fața șoldului ("chair seat"), > 0 = călcâi în spate

None = „nu știu”.

predict_points(video_path) întoarce, pentru fiecare cadru,
umărul, șoldul și călcâiul în pixeli (None sub pragul de vizibilitate) și direcția de mers.

Linia de bază: MediaPipe Pose Landmarker pe fiecare cadru, partea mai vizibilă, fără netezire.
"""
from __future__ import annotations

import math
import urllib.request
from types import SimpleNamespace
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
MODEL_VARIANT = "full"  # lite / full / heavy
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_{MODEL_VARIANT}/float16/latest/pose_landmarker_{MODEL_VARIANT}.task"
)
MODEL_PATH = ROOT / "models" / f"pose_landmarker_{MODEL_VARIANT}.task"
MIN_VISIBILITY = 0.5

# Indicii celor 33 de puncte MediaPipe Pose
IDX = {
    "nose": 0, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_heel": 29, "right_heel": 30,
}


def model_path() -> str:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


def angle_from_vertical(top, bottom, facing: float) -> float:
    dx = (top[0] - bottom[0]) * facing
    dy = bottom[1] - top[1]
    return math.degrees(math.atan2(dx, dy))


def angles_for_frame(landmarks, width: int, height: int):
    def pt(name):
        p = landmarks[IDX[name]]
        return (p.x * width, p.y * height), float(getattr(p, "visibility", None) or 0.0)

    side = "left" if pt("left_hip")[1] >= pt("right_hip")[1] else "right"
    (shoulder, v_sh), (hip, v_hip), (heel, v_heel) = (
        pt(f"{side}_shoulder"), pt(f"{side}_hip"), pt(f"{side}_heel"))
    nose, ear = pt("nose")[0], pt(f"{side}_ear")[0]
    facing = 1.0 if nose[0] >= ear[0] else -1.0

    trunk = angle_from_vertical(shoulder, hip, facing) if min(v_sh, v_hip) >= MIN_VISIBILITY else None
    leg = angle_from_vertical(hip, heel, facing) if min(v_hip, v_heel) >= MIN_VISIBILITY else None
    return trunk, leg


KNEE = {"left": (23, 25, 27), "right": (24, 26, 28)}   # șold, genunchi, gleznă
STRAIGHT_KNEE = 165                                     # peste: persoană în picioare, nu pe cal
MULTI_POSES = 4


def _knee_angle(lm, width: int, height: int) -> float | None:
    """Unghiul genunchiului pe partea mai vizibilă (180 = picior întins)."""
    vis = lambda i: float(getattr(lm[i], "visibility", 0) or 0)  # noqa: E731
    a, b, c = (lm[i] for i in KNEE["left" if vis(23) >= vis(24) else "right"])
    v1 = ((a.x - b.x) * width, (a.y - b.y) * height)
    v2 = ((c.x - b.x) * width, (c.y - b.y) * height)
    n = math.hypot(*v1) * math.hypot(*v2)
    return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n)))) if n else None


def _is_rider(lm, width: int, height: int) -> bool:
    knee = _knee_angle(lm, width, height)
    return knee is not None and knee < STRAIGHT_KNEE


def _hip_gap(a, b, width: int, height: int) -> float:
    """Distanța dintre șoldurile a două persoane, în lungimi de trunchi ale lui b."""
    hip = lambda lm: ((lm[23].x + lm[24].x) / 2 * width, (lm[23].y + lm[24].y) / 2 * height)  # noqa: E731
    sh = lambda lm: ((lm[11].x + lm[12].x) / 2 * width, (lm[11].y + lm[12].y) / 2 * height)  # noqa: E731
    return math.dist(hip(a), hip(b)) / max(math.dist(hip(b), sh(b)), 1.0)


DETECTOR_URL = ("https://storage.googleapis.com/mediapipe-models/object_detector/"
                "efficientdet_lite0/float32/latest/efficientdet_lite0.tflite")
DETECTOR_PATH = ROOT / "models" / "efficientdet_lite0.tflite"


def detector_path() -> str:
    if not DETECTOR_PATH.exists():
        DETECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DETECTOR_URL, DETECTOR_PATH)
    return str(DETECTOR_PATH)


def _on_horse(lm, horses, width: int, height: int) -> bool:
    """Șoldul persoanei stă pe spatele unui cal: în dreptul casetei, aproape de marginea ei de sus."""
    hx = (lm[23].x + lm[24].x) / 2 * width
    hy = (lm[23].y + lm[24].y) / 2 * height
    return any(b.origin_x <= hx <= b.origin_x + b.width
               and b.origin_y - 0.2 * b.height <= hy <= b.origin_y + 0.4 * b.height for b in horses)


def _upright(lm, width: int, height: int) -> bool:
    """Trunchiul e aproape vertical (sub 45°): un schelet întins de-a lungul calului nu e călărețul."""
    dx = ((lm[11].x + lm[12].x) - (lm[23].x + lm[24].x)) / 2 * width
    dy = ((lm[23].y + lm[24].y) - (lm[11].y + lm[12].y)) / 2 * height
    return dy > 0 and abs(dx) < math.tan(math.radians(45)) * dy


CROP_SIZE = 512
MAX_GAP = 30   # cadre peste care se ține minte poziția călărețului


def _pose_on_crop(detector, rgb, box) -> list:
    """Pose pe zona de deasupra unui cal (unde stă călărețul), mărită; punctele revin în coordonatele
    cadrului întreg (normalizate), ca la detecția pe tot cadrul."""
    h, w = rgb.shape[:2]
    x0, x1 = max(0, int(box.origin_x - 0.1 * box.width)), min(w, int(box.origin_x + 1.1 * box.width))
    y0, y1 = max(0, int(box.origin_y - 0.8 * box.height)), min(h, int(box.origin_y + box.height))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    s = CROP_SIZE / max(x1 - x0, y1 - y0)
    crop = cv2.resize(np.ascontiguousarray(rgb[y0:y1, x0:x1]), (round((x1 - x0) * s), round((y1 - y0) * s)),
                      interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    found = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)).pose_landmarks
    back = lambda p: SimpleNamespace(x=(x0 + p.x * (x1 - x0)) / w, y=(y0 + p.y * (y1 - y0)) / h,  # noqa: E731
                                     visibility=p.visibility)
    return [[back(p) for p in lm] for lm in found]


# Motivul pentru care un cadru nu are răspuns (doar explicație: deciziile nu depind de el).
REASON_NO_PERSON = "calaret_negasit"        # nimeni găsit, nici pe decupajul de deasupra calului
REASON_NOT_ON_HORSE = "nu_e_pe_cal"         # persoana găsită nu stă pe un cal (sau stă în picioare)
REASON_TRUNK = "trunchi_peste_45"           # pe cal, dar trunchiul e aplecat peste 45° (sau e alt schelet)
REASON_POINTS = "puncte_invizibile"         # călărețul e găsit, dar umărul sau șoldul nu se văd


def _landmarks(video_path: str, last: int | None = None):
    """Rulează Pose pe cadre, în ordine (indice OpenCV). Dă (index, landmarks | None, lățime, înălțime,
    motiv | None); motivul spune de ce cadrul nu are persoană aleasă.
    Persoana vine din detectorul cu o singură persoană (cel mai stabil). Doar când aceasta stă în
    picioare (genunchi întins) și un al doilea detector (mai multe persoane) găsește pe cineva cu
    genunchiul îndoit, se trece pe acela."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    make = lambda n, mode: vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(  # noqa: E731
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path()), running_mode=mode, num_poses=n))
    try:
        # detectorul multiplu în modul IMAGE: fără urmărire, pierde mai rar persoanele
        objects = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=detector_path()), score_threshold=0.3,
            category_allowlist=["horse"], max_results=5))
        with make(1, vision.RunningMode.VIDEO) as single, make(MULTI_POSES, vision.RunningMode.IMAGE) as multi, \
                objects:
            index, prev, prev_at = 0, None, -1
            while last is None or index <= last:
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int(index * 1000 / fps)
                result = single.detect_for_video(image, ts)
                chosen = result.pose_landmarks[0] if result.pose_landmarks else None
                horses = [d.bounding_box for d in objects.detect(image).detections]
                # cu un cal în cadru: călăreț = trunchi drept, cu șoldul pe spatele calului;
                # fără cal: genunchi îndoit
                rider = (lambda lm: _upright(lm, width, height) and _on_horse(lm, horses, width, height)) \
                    if horses else (lambda lm: _is_rider(lm, width, height))
                reason = None
                if chosen is None or not rider(chosen):
                    if chosen is None:
                        reason = REASON_NO_PERSON
                    elif horses and _on_horse(chosen, horses, width, height):
                        reason = REASON_TRUNK
                    else:
                        reason = REASON_NOT_ON_HORSE
                    riders = []
                    # Pose pe decupajul mărit de deasupra fiecărui cal (cel mai mare întâi)
                    recent = prev is not None and index - prev_at <= MAX_GAP
                    for k, b in enumerate(sorted(horses, key=lambda b: -b.width * b.height)):
                        # lângă călărețul recent (fără salturi între persoane); fără unul recent,
                        # doar pe calul cel mai mare (cel mai aproape de cameră)
                        riders = [lm for lm in _pose_on_crop(multi, rgb, b) if rider(lm)
                                  and (_hip_gap(lm, prev, width, height) < 1.0 if recent else k == 0)]
                        if riders:
                            break
                    # cineva în picioare nu e niciodată călărețul: fără alternativă, nu răspundem
                    chosen = riders[0] if riders else None
                    if chosen is not None:
                        reason = None
                if chosen is not None:   # ultima poziție a călărețului, ținută peste goluri scurte
                    prev, prev_at = chosen, index
                yield index, chosen, width, height, reason
                index += 1
    finally:
        cap.release()


def predict(video_path: str, frames: list[int]) -> dict[int, dict]:
    """Interfața v1: unghiurile pe cadrele cerute."""
    wanted = set(frames)
    out = {f: {"trunk_deg": None, "leg_deg": None} for f in frames}
    for index, landmarks, width, height, _ in _landmarks(video_path, max(frames)):
        if index in wanted and landmarks:
            trunk, leg = angles_for_frame(landmarks, width, height)
            out[index] = {"trunk_deg": trunk, "leg_deg": leg}
    return out


def points_for_frame(landmarks, width: int, height: int) -> dict:
    """Umăr, șold, călcâi (pixeli, partea mai vizibilă) și direcția; None sub pragul de vizibilitate."""
    def pt(name):
        p = landmarks[IDX[name]]
        return (p.x * width, p.y * height), float(getattr(p, "visibility", None) or 0.0)

    side = "left" if pt("left_hip")[1] >= pt("right_hip")[1] else "right"
    facing = 1.0 if pt("nose")[0][0] >= pt(f"{side}_ear")[0][0] else -1.0
    out = {"facing": facing}
    for name in ("shoulder", "hip", "heel"):
        xy, v = pt(f"{side}_{name}")
        out[name] = xy if v >= MIN_VISIBILITY else None
    return out


def predict_points_with_reasons(video_path: str) -> dict[int, dict]:
    """Pentru FIECARE cadru: punctele (ca în predict_points) sau None, plus „motiv”: None dacă umărul
    și șoldul sunt măsurabile, altfel unul din REASON_*. Aceleași decizii ca predict_points."""
    out = {}
    for index, landmarks, width, height, reason in _landmarks(video_path):
        points = points_for_frame(landmarks, width, height) if landmarks else None
        if points is not None and (points["shoulder"] is None or points["hip"] is None):
            reason = REASON_POINTS
        out[index] = {"points": points, "motiv": reason}
    return out


def predict_points(video_path: str) -> dict[int, dict]:
    """Pentru fiecare cadru al clipului, punctele lanțului și direcția de mers.
    Cadrele fără persoană lipsesc din rezultat."""
    return {i: r["points"] for i, r in predict_points_with_reasons(video_path).items() if r["points"] is not None}
