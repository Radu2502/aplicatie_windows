"""
Evaluarea routerului de vedere: etichetele pe cadre (din intervale), criteriile R1–R3 cu pragurile lor, intervalul
Wilson de 95% și validarea încrucișată pe clipuri (un clip, sau un grup de clipuri cu același conținut, ținut deoparte
pe rând).

Datele vin din `data/clipuri_noi/impartire.csv` (împărțirea dezvoltare / verificare); fără el nu se citește nimic.
Evaluarea pe verificare (antrenare pe dezvoltare, metoda fixată printr-un commit) folosește aceleași funcții
(`evaluate_holdout`), prin comanda `verificare`. Comanda are protecții: refuză fără `impartire.csv`, refuză dacă
SHA-256 al împărțirii nu e cel dat, refuză un commit al metodei care nu e strămoș al lui HEAD și refuză a doua rulare
(jurnalul `data/clipuri_noi/verificare_jurnal.json` se scrie înainte de calcul, iar rezultatul o singură dată).
Cadrele aceluiași clip nu sunt independente: intervalul Wilson e doar orientativ. Fără Streamlit.

Rulare:
  uv run python -m ecvestru.evaluare_router dezvoltare
  uv run python -m ecvestru.evaluare_router verificare --hash-impartire <SHA-256> --commit-metoda <commit>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from ecvestru import intervale as IV
from ecvestru import router_vedere as RV
from ecvestru import stocare as ST

# Pragurile criteriilor; trecerea se decide pe estimarea punctuală
R1_MAX = 0.05     # cadre ne-profil intrate în profil, din cele trimise la profil
R2_MAX = 0.15     # cadre de profil pierdute
R3_MIN = 0.90     # vederea corectă, din cadrele față/spate trimise la modulul față/spate
WILSON_Z = 1.96   # 95%

UNLABELED = ""
NON_PROFILE = (IV.FRONT, IV.BACK, IV.OBLIQUE, IV.HIDDEN)
DEV_SET, CHECK_SET = "dezvoltare", "verificare"
SETS = (DEV_SET, CHECK_SET)
SPLIT_CSV = ST.ROOT / "data" / "clipuri_noi" / "impartire.csv"
HOLDOUT_JOURNAL = ST.ROOT / "data" / "clipuri_noi" / "verificare_jurnal.json"
HOLDOUT_REPORT = ST.ROOT / "data" / "clipuri_noi" / "verificare_rezultat.md"
REFERENCE_ANNOTATOR = "adnotator"
RIDER_SEPARATOR = ";"
PERCENT = 100


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Share(_Model):
    """k din n, cu intervalul Wilson; value e None când n = 0."""
    k: int
    n: int
    value: float | None
    low: float | None
    high: float | None


class Criterion(Share):
    threshold: float
    at_most: bool          # True: trece cu value ≤ prag; False: cu value ≥ prag
    passes: bool


class RouterCriteria(_Model):
    r1: Criterion
    r2: Criterion
    r3: Criterion
    non_profile_into_profile: Share     # din toate cadrele ne-profil, câte au intrat în profil
    front_back_routed: Share            # din cadrele față/spate, câte au ajuns la modulul față/spate

    @property
    def passes(self) -> bool:
        return self.r1.passes and self.r2.passes and self.r3.passes


class RouterEvaluation(_Model):
    set_name: str
    kind: str
    pooled: RouterCriteria
    per_clip: dict[str, RouterCriteria]
    per_rider: dict[str, RouterCriteria]


# ---------- etichetele pe cadre ----------

def frame_labels(intervals: list[IV.Interval], n_frames: int) -> np.ndarray:
    labels = np.full(n_frames, UNLABELED, dtype=object)
    for interval in intervals:
        labels[max(interval.start, 0): min(interval.stop, n_frames - 1) + 1] = interval.view
    return labels


# ---------- criteriile ----------

def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def share(k: int, n: int) -> Share:
    low, high = wilson(k, n)
    return Share(k=k, n=n, value=k / n if n else None, low=low, high=high)


def criterion(k: int, n: int, threshold: float, at_most: bool) -> Criterion:
    s = share(k, n)
    passes = s.value is not None and (s.value <= threshold if at_most else s.value >= threshold)
    return Criterion(**s.model_dump(), threshold=threshold, at_most=at_most, passes=passes)


def criteria(truth: np.ndarray, voted: np.ndarray) -> RouterCriteria:
    """R1–R3 pe cadrele etichetate; `voted` e vederea routerului după vot (sau UNDECIDED)."""
    truth, voted = np.asarray(truth, dtype=object), np.asarray(voted, dtype=object)
    labeled = truth != UNLABELED
    route = RV.routes(voted)
    to_profile = labeled & (route == RV.ROUTE_PROFILE)
    non_profile = labeled & np.isin(truth, NON_PROFILE)
    profile = labeled & (truth == IV.PROFILE)
    front_back = labeled & np.isin(truth, RV.FRONT_BACK_VIEWS)
    fb_routed = front_back & (route == RV.ROUTE_FRONT_BACK)
    return RouterCriteria(
        r1=criterion(int((to_profile & non_profile).sum()), int(to_profile.sum()), R1_MAX, at_most=True),
        r2=criterion(int((profile & (route != RV.ROUTE_PROFILE)).sum()), int(profile.sum()), R2_MAX, at_most=True),
        r3=criterion(int((fb_routed & (voted == truth)).sum()), int(fb_routed.sum()), R3_MIN, at_most=False),
        non_profile_into_profile=share(int((to_profile & non_profile).sum()), int(non_profile.sum())),
        front_back_routed=share(int(fb_routed.sum()), int(front_back.sum())))


# ---------- datele unui clip și validarea încrucișată ----------

@dataclass
class ClipData:
    values: np.ndarray        # (n_frames, n_semnale)
    labels: np.ndarray        # (n_frames,) vederea omului sau UNLABELED
    fps: float


def _training(clips: dict[str, ClipData], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([clips[c].values for c in names])
    y = np.concatenate([clips[c].labels for c in names])
    keep = (y != UNLABELED) & ~np.isnan(X).all(axis=1)
    return X[keep], y[keep]


def _evaluation(set_name: str, kind: str, truths: dict, preds: dict, riders: dict | None) -> RouterEvaluation:
    per_rider = {}
    for rider in sorted({r for rs in (riders or {}).values() for r in rs}):
        names = [c for c in truths if rider in (riders or {}).get(c, ())]
        per_rider[rider] = criteria(np.concatenate([truths[c] for c in names]),
                                    np.concatenate([preds[c] for c in names]))
    pooled = criteria(np.concatenate(list(truths.values())), np.concatenate(list(preds.values())))
    return RouterEvaluation(set_name=set_name, kind=kind, pooled=pooled,
                            per_clip={c: criteria(truths[c], preds[c]) for c in truths}, per_rider=per_rider)


def cross_validate(clips: dict[str, ClipData], kind: str = RV.LOGISTIC, groups: dict[str, str] | None = None,
                   riders: dict[str, tuple[str, ...]] | None = None) -> RouterEvaluation:
    """Validarea încrucișată pe dezvoltare: fiecare grup (implicit, fiecare clip) e ținut deoparte pe rând; modelul
    se antrenează pe cadrele etichetate ale celorlalte, apoi predicția votată se compară cu omul."""
    groups = {c: (groups or {}).get(c, c) for c in clips}
    preds = {}
    for group in sorted(set(groups.values())):
        held = [c for c in clips if groups[c] == group]
        model = RV.make_classifier(kind).fit(*_training(clips, [c for c in clips if groups[c] != group]))
        for c in held:
            preds[c] = RV.predict_clip(model, clips[c].values, clips[c].fps)
    return _evaluation(DEV_SET, kind, {c: clips[c].labels for c in clips}, preds, riders)


def evaluate_holdout(train: dict[str, ClipData], test: dict[str, ClipData], kind: str = RV.LOGISTIC,
                     riders: dict[str, tuple[str, ...]] | None = None) -> RouterEvaluation:
    """Antrenare pe `train`, evaluare o singură dată pe `test` (verificarea, cu metoda fixată)."""
    model = RV.make_classifier(kind).fit(*_training(train, list(train)))
    preds = {c: RV.predict_clip(model, d.values, d.fps) for c, d in test.items()}
    return _evaluation(CHECK_SET, kind, {c: d.labels for c, d in test.items()}, preds, riders)


# ---------- raportul ----------

def _pct(x: float | None) -> str:
    return "–" if x is None else f"{PERCENT * x:.1f}%"


def _cell(c: Share) -> str:
    return f"{_pct(c.value)} ({c.k}/{c.n})"


def _verdict(c: Criterion) -> str:
    return "trece" if c.passes else "nu trece"


def _row(name: str, c: RouterCriteria) -> str:
    return f"| {name} | {_cell(c.r1)} | {_cell(c.r2)} | {_cell(c.r3)} |"


def report_markdown(result: RouterEvaluation) -> str:
    p = result.pooled
    dev = result.set_name == DEV_SET
    lines = [f"# Routerul față-spate: {result.set_name} ({result.kind})", ""]
    if dev:
        lines += ["Cifre de **dezvoltare** (validare încrucișată pe clipuri): nu decid trecerea.", ""]
    lines += ["| Criteriu | Valoare | Prag | Wilson 95% | Rezultat |", "|---|---|---|---|---|"]
    for label, c, sign in (("R1. Cadre ne-profil intrate în profil", p.r1, "≤"),
                           ("R2. Cadre de profil pierdute", p.r2, "≤"),
                           ("R3. Față vs spate", p.r3, "≥")):
        lines.append(f"| {label} | {_cell(c)} | {sign} {PERCENT * c.threshold:.0f}% | "
                     f"{_pct(c.low)}–{_pct(c.high)} | {_verdict(c)} |")
    lines += ["", f"Cadre ne-profil intrate în profil, din toate cele ne-profil: {_cell(p.non_profile_into_profile)}.",
              f"Cadre față/spate trimise la modulul față/spate: {_cell(p.front_back_routed)}.",
              "Intervalul Wilson tratează cadrele ca independente; cele ale aceluiași clip nu sunt.", "",
              "| Clip | R1 | R2 | R3 |", "|---|---|---|---|"]
    lines += [_row(c, v) for c, v in result.per_clip.items()]
    if result.per_rider:
        lines += ["", "| Călăreț | R1 | R2 | R3 |", "|---|---|---|---|"]
        lines += [_row(r, v) for r, v in result.per_rider.items()]
    return "\n".join(lines) + "\n"


# ---------- datele după împărțire ----------

@dataclass(frozen=True)
class SplitClip:
    video: str
    group: str                 # hash-ul conținutului: clipurile identice sunt ținute deoparte împreună
    riders: tuple[str, ...]


def split_clips(path: Path, set_name: str) -> list[SplitClip]:
    if set_name not in SETS:
        raise ValueError(f"Set necunoscut: {set_name} (cunoscute: {', '.join(SETS)}).")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lipsește împărțirea ({path}): evaluarea routerului rulează doar după ea.")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [SplitClip(video=r["video"], group=r["hash"],
                      riders=tuple(x for x in r["calareti"].split(RIDER_SEPARATOR) if x))
            for r in rows if r["set"] == set_name]


def load_clip_data(clip: SplitClip, clips_dir: Path, answers: Path, annotator: str) -> ClipData:
    path = clips_dir / clip.video
    signals = RV.load_clip_signals(path)
    n_frames, fps = len(signals.values), ST.video_info(path)[1]
    return ClipData(values=signals.values, fps=fps,
                    labels=frame_labels(IV.current_intervals(answers, annotator, clip.video), n_frames))


# ---------- evaluarea unică pe verificare (cu protecții) ----------

class HoldoutRefused(RuntimeError):
    """Evaluarea pe verificare nu pornește: motivul e în mesaj."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_is_ancestor(commit: str) -> bool:
    """Commitul există și e strămoș al lui HEAD (sau chiar HEAD)."""
    run = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ST.ROOT, capture_output=True)
    return run.returncode == 0


def run_holdout(split_csv: Path, split_hash: str, method_commit: str, journal: Path, report: Path,
                kind: str = RV.LOGISTIC, loader=None, is_ancestor=_git_is_ancestor) -> RouterEvaluation:
    """Antrenare pe dezvoltare, evaluare o singură dată pe verificare. Refuză (HoldoutRefused) înainte să citească
    vreun clip dacă: lipsește împărțirea, hash-ul ei nu e cel dat, commitul metodei nu e strămoș al lui HEAD, sau
    jurnalul arată că evaluarea a rulat deja. Jurnalul se creează înainte de calcul, deci și o rulare întreruptă
    consumă verificarea."""
    split_csv, journal, report = Path(split_csv), Path(journal), Path(report)
    if not split_csv.exists():
        raise HoldoutRefused(f"Lipsește împărțirea ({split_csv}): verificarea rulează doar după ea.")
    actual = file_sha256(split_csv)
    if actual != split_hash.strip().lower():
        raise HoldoutRefused(f"SHA-256 al împărțirii e {actual}, nu {split_hash}: împărțirea s-a schimbat.")
    if not is_ancestor(method_commit):
        raise HoldoutRefused(f"Commitul metodei ({method_commit}) nu există sau nu e strămoș al lui HEAD.")
    if journal.exists() or report.exists():
        raise HoldoutRefused(f"Verificarea a rulat deja ({journal}): a doua rulare nu e permisă.")
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("x", encoding="utf-8") as f:
        json.dump({"pornit": datetime.now(timezone.utc).isoformat(timespec="seconds"), "impartire_sha256": actual,
                   "commit_metoda": method_commit, "tip": kind}, f, ensure_ascii=False, indent=2)
    if loader is None:
        from ecvestru import clipuri_noi as CN

        def loader(clip: SplitClip) -> ClipData:
            return load_clip_data(clip, CN.clips_dir(), IV.answers_path(), REFERENCE_ANNOTATOR)
    train_clips, test_clips = split_clips(split_csv, DEV_SET), split_clips(split_csv, CHECK_SET)
    riders = {c.video: c.riders for c in train_clips + test_clips}
    result = evaluate_holdout({c.video: loader(c) for c in train_clips}, {c.video: loader(c) for c in test_clips},
                              kind=kind, riders=riders)
    with report.open("x", encoding="utf-8") as f:
        f.write(report_markdown(result))
    return result


def main(argv: list[str] | None = None) -> int:
    from ecvestru import clipuri_noi as CN
    parser = argparse.ArgumentParser(prog="python -m ecvestru.evaluare_router", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    dev = commands.add_parser("dezvoltare", help="validarea încrucișată pe dezvoltare (după împărțire)")
    dev.add_argument("--tip", default=RV.LOGISTIC)
    check = commands.add_parser("verificare", help="evaluarea unică pe verificare (o singură dată, cu metoda fixată)")
    check.add_argument("--hash-impartire", required=True, help="SHA-256 al lui data/clipuri_noi/impartire.csv")
    check.add_argument("--commit-metoda", required=True, help="commitul care îngheață metoda")
    check.add_argument("--tip", default=RV.LOGISTIC)
    args = parser.parse_args(argv)
    if args.command == "verificare":
        try:
            result = run_holdout(SPLIT_CSV, args.hash_impartire, args.commit_metoda, HOLDOUT_JOURNAL, HOLDOUT_REPORT,
                                 kind=args.tip)
        except HoldoutRefused as refused:
            print(f"Refuzat: {refused}", file=sys.stderr)
            return 2
        print(report_markdown(result))
        return 0
    clips = split_clips(SPLIT_CSV, DEV_SET)
    data = {c.video: load_clip_data(c, CN.clips_dir(), IV.answers_path(), REFERENCE_ANNOTATOR) for c in clips}
    result = cross_validate(data, kind=args.tip, groups={c.video: c.group for c in clips},
                            riders={c.video: c.riders for c in clips})
    print(report_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
