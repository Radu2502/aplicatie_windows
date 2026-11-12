"""
Inventarul clipurilor: orientare, rezoluție, durată, persoane, profil și vizibilitatea
punctelor umăr / șold / călcâi (MediaPipe, ca în research/candidate.py).

Coloanele automate (orientare, rezoluție, durată, procente) vin din cod. Persoanele, profilul,
utilizabilitatea și motivul vin din verificarea vizuală a colajelor de cadre, în data/inventar_vizual.csv:
MediaPipe nu numără sigur oamenii mici din fundal și se agață uneori de instructor, iar clipurile
sunt filmate în manej, cu profilul doar pe segmente. Coloanele lui `num_poses` sunt doar indicii.

Rulare:
    uv run python -m ecvestru.inventar            # scrie data/clips.csv și colajele de cadre
    uv run python -m ecvestru.inventar --refa     # rulează din nou MediaPipe, ignoră cache-ul
    uv run python -m ecvestru.inventar --refa --clip clip_152846.mp4   # doar pe un clip

Cadrele se numără cu indicele citit prin OpenCV (care aplică automat rotația din metadate),
niciodată prin timp. Landmark-urile pe fiecare cadru se păstrează în data/results/poses/.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ecvestru import constante_candidat as KC

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "videos"
CLIPS_CSV = ROOT / "data" / "clips.csv"
VISUAL_CSV = ROOT / "data" / "inventar_vizual.csv"
RESULTS = ROOT / "data" / "results"
POSES = RESULTS / "poses"
MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"

MIN_VISIBILITY = KC.MIN_VISIBILITY   # ca în candidat (ecvestru/constante_candidat.py)
# Doar un reper pentru cine citește coloana „umeri” a inventarului (lățime umeri / lungime trunchi): din profil
# umerii se suprapun, deci sub ~0,35 clipul arată a profil. Nicio regulă nu-l folosește (regula e în vedere.py).
INVENTORY_PROFILE_SHOULDER_RATIO_HINT = 0.35
MAX_POSES = 4

COLUMNS = ["video", "orientare", "rezolutie", "durata_s", "persoane", "profil",
           "procent_trunchi", "procent_picior", "utilizabil", "motiv",
           "calaret", "cal", "sesiune", "alura"]

IDX = {"left_shoulder": 11, "right_shoulder": 12, "left_hip": 23, "right_hip": 24,
       "left_heel": 29, "right_heel": 30}


def orientation(width: int, height: int) -> str:
    if width > height:
        return "orizontal"
    if height > width:
        return "vertical"
    return "patrat"


def frame_info(points: dict[str, tuple[float, float, float]]) -> dict:
    """Din punctele (x_px, y_px, visibility) ale unui cadru: partea dinspre cameră și ce se vede.

    Partea se alege ca în candidate.py: șoldul mai vizibil.
    """
    side = "left" if points["left_hip"][2] >= points["right_hip"][2] else "right"
    sh, hip, heel = (points[f"{side}_{n}"] for n in ("shoulder", "hip", "heel"))
    trunk_ok = min(sh[2], hip[2]) >= MIN_VISIBILITY
    leg_ok = min(hip[2], heel[2]) >= MIN_VISIBILITY

    ratio = None
    if trunk_ok:
        torso = float(np.hypot(sh[0] - hip[0], sh[1] - hip[1]))
        if torso > 1e-6:
            ls, rs = points["left_shoulder"], points["right_shoulder"]
            ratio = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])) / torso
    return {"side": side, "trunk_ok": trunk_ok, "leg_ok": leg_ok, "shoulder_ratio": ratio}


def typical_count(counts, min_share: float = 0.10) -> int:
    """Cel mai mare număr de persoane care apare în cel puțin `min_share` din cadre."""
    counts = list(counts)
    if not counts:
        return 0
    n = len(counts)
    return max((k for k in set(counts) if sum(c >= k for c in counts) >= min_share * n), default=0)


def load_visual(path: Path = VISUAL_CSV) -> dict[str, dict]:
    """Verificarea vizuală: video -> {calaret, alura, persoane, profil, utilizabil, motiv}."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["video"]: r for r in csv.DictReader(f)}


def points_from_array(lm: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """lm: (33, 4) cu x_px, y_px, z, visibility."""
    return {name: (float(lm[i, 0]), float(lm[i, 1]), float(lm[i, 3])) for name, i in IDX.items()}


def extract_poses(path: Path) -> Path:
    """Rulează MediaPipe pe tot clipul și salvează landmark-urile în data/results/poses/<clip>.npz."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    def make(num_poses):
        return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO, num_poses=num_poses))

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarks, counts = [], []
    width = height = 0
    with make(1) as single, make(MAX_POSES) as multi:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            ts = int(len(counts) * 1000 / fps)
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = single.detect_for_video(image, ts)
            counts.append(len(multi.detect_for_video(image, ts).pose_landmarks))
            lm = np.full((33, 4), np.nan, np.float32)
            if res.pose_landmarks:
                for i, p in enumerate(res.pose_landmarks[0]):
                    lm[i] = (p.x * width, p.y * height, p.z, getattr(p, "visibility", None) or 0.0)
            landmarks.append(lm)
    cap.release()

    POSES.mkdir(parents=True, exist_ok=True)
    out = POSES / f"{path.stem}.npz"
    np.savez_compressed(out, landmarks=np.array(landmarks), counts=np.array(counts),
                        fps=fps, width=width, height=height)
    return out


def summarize(video: str, data, visual: dict | None = None) -> tuple[dict, dict]:
    lms, counts = data["landmarks"], data["counts"]
    fps, width, height = float(data["fps"]), int(data["width"]), int(data["height"])
    n = len(lms)
    found = [frame_info(points_from_array(lm)) for lm in lms if not np.isnan(lm[0, 0])]

    def pct(k: str) -> float:
        return round(100.0 * sum(i[k] for i in found) / n, 1) if n else 0.0

    ratios = [i["shoulder_ratio"] for i in found if i["shoulder_ratio"] is not None]
    sides = [i["side"] for i in found]
    visual = visual or {}
    row = {
        "video": video, "orientare": orientation(width, height),
        "rezolutie": f"{width}x{height}", "durata_s": round(n / fps, 1),
        "persoane": visual.get("persoane", ""), "profil": visual.get("profil", ""),
        "procent_trunchi": pct("trunk_ok"), "procent_picior": pct("leg_ok"),
        "utilizabil": visual.get("utilizabil", ""), "motiv": visual.get("motiv", ""),
        "calaret": visual.get("calaret", ""), "cal": "", "sesiune": "",
        "alura": visual.get("alura", ""),
    }
    extra = {"frames": n, "shoulder_ratio": float(np.median(ratios)) if ratios else None,
             "mp_persons": typical_count(counts), "max_persons": int(max(counts, default=0)),
             "pct_found": 100.0 * len(found) / n if n else 0.0,
             "side": max(set(sides), key=sides.count) if sides else None}
    return row, extra


def read_frames(path: Path, indices) -> dict[int, np.ndarray]:
    wanted, out = set(indices), {}
    cap = cv2.VideoCapture(str(path))
    index = 0
    while index <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            out[index] = frame
        index += 1
    cap.release()
    return out


def draw_chain(img: np.ndarray, lm: np.ndarray) -> None:
    if np.isnan(lm[0, 0]):
        return
    info = frame_info(points_from_array(lm))
    pts = points_from_array(lm)
    chain = [pts[f"{info['side']}_{n}"] for n in ("shoulder", "hip", "heel")]
    r = max(3, img.shape[1] // 120)
    color = (0, 255, 255)
    for a, b in zip(chain, chain[1:], strict=False):   # perechi consecutive
        if min(a[2], b[2]) >= MIN_VISIBILITY:
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, r // 2 + 1)
    for x, y, v in chain:
        cv2.circle(img, (int(x), int(y)), r, (0, 220, 0) if v >= MIN_VISIBILITY else (0, 0, 255), -1)


def fit(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    tile = np.full((size, size, 3), 30, np.uint8)
    y0, x0 = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    tile[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return tile


def label(tile: np.ndarray, lines: list[str], height: int = 44) -> np.ndarray:
    band = np.full((height, tile.shape[1], 3), 30, np.uint8)
    for k, text in enumerate(lines):
        cv2.putText(band, text, (6, 17 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5 - 0.05 * k,
                    (255, 255, 255) if k == 0 else (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([tile, band])


def grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    blank = np.zeros_like(tiles[0])
    rows = [tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols]))
            for i in range(0, len(tiles), cols)]
    return np.vstack([np.hstack(r) for r in rows])


def sheets(rows: list[dict], datas: dict, per_clip: int = 6) -> list[Path]:
    """Colajul principal (cadrul din mijloc) și un colaj cu `per_clip` cadre pe clip."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    main_tiles, strips = [], []
    for row in rows:
        lms = datas[row["video"]]["landmarks"]
        n = len(lms)
        picks = [int(n * (k + 0.5) / per_clip) for k in range(per_clip)]
        frames = read_frames(VIDEOS / row["video"], picks + [n // 2])
        mid = frames[n // 2].copy()
        draw_chain(mid, lms[n // 2])
        main_tiles.append(label(fit(mid, 320), [
            f"{row['video'][:-4]}  #{n // 2}  {row['utilizabil']}",
            f"tr {row['procent_trunchi']:.0f}%  pic {row['procent_picior']:.0f}%  pers {row['persoane']}"]))
        strip = []
        for i in picks:
            img = frames[i].copy()
            draw_chain(img, lms[i])
            strip.append(label(fit(img, 200), [f"{row['video'][5:-4]} #{i}"], 24))
        strips.append(np.hstack(strip))
    out = [RESULTS / "inventar_cadre.jpg"]
    cv2.imwrite(str(out[0]), grid(main_tiles, 6))
    for k in range(0, len(strips), 7):
        p = RESULTS / f"inventar_detaliu_{k // 7 + 1}.jpg"
        cv2.imwrite(str(p), np.vstack(strips[k:k + 7]))
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--refa", action="store_true", help="rulează din nou MediaPipe")
    ap.add_argument("--clip", action="append", default=[],
                    help="cu --refa: doar aceste clipuri (de ex. clip_152846.mp4); se poate repeta")
    args = ap.parse_args()

    paths = sorted(VIDEOS.glob("*.mp4"))
    redo = {p.name for p in paths} if args.refa and not args.clip else set(args.clip)
    todo = [p for p in paths if p.name in redo or not (POSES / f"{p.stem}.npz").exists()]
    if todo:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(extract_poses, todo))

    datas = {p.name: dict(np.load(POSES / f"{p.stem}.npz")) for p in paths}
    visual = load_visual()
    results = [summarize(p.name, datas[p.name], visual.get(p.name)) for p in paths]
    rows = [r for r, _ in results]
    with CLIPS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    images = sheets(rows, datas)

    for row, ex in results:
        ratio = "-" if ex["shoulder_ratio"] is None else f"{ex['shoulder_ratio']:.2f}"
        print(f"{row['video']:16} {row['rezolutie']:>8} {row['durata_s']:5.1f}s "
              f"mp_pers={ex['mp_persons']}(max {ex['max_persons']}) umeri={ratio:>4} "
              f"gasit={ex['pct_found']:3.0f}% tr={row['procent_trunchi']:5.1f}% "
              f"pic={row['procent_picior']:5.1f}% | {row['persoane']} {row['profil']} "
              f"{row['utilizabil']}: {row['motiv']}")
    print("\nScris: " + ", ".join(str(p.relative_to(ROOT)) for p in [CLIPS_CSV, *images]))


if __name__ == "__main__":
    main()
