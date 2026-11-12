"""Constantele candidatului folosite de aplicație (ecvestru/constante_candidat.py): o singură sursă în
ecvestru/, cu aceleași valori ca research/candidate.py (citit cu ast, nemodificat); nicio copie scrisă de mână în
alte module."""
import ast
import re
from pathlib import Path

from ecvestru import analiza_vederi as AV
from ecvestru import constante_candidat as KC
from ecvestru import inventar as INV
from ecvestru import tinta as T
from ecvestru import verificare as V

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "research" / "candidate.py"


def candidate_tree() -> ast.Module:
    return ast.parse(CANDIDATE.read_text(encoding="utf-8"))


def test_min_visibility_is_the_one_of_the_frozen_candidate():
    values = [node.value.value for node in ast.walk(candidate_tree()) if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "MIN_VISIBILITY" for t in node.targets)]
    assert values == [KC.MIN_VISIBILITY]


def test_the_horse_detector_settings_are_the_ones_of_the_frozen_candidate():
    calls = [node for node in ast.walk(candidate_tree()) if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "ObjectDetectorOptions"]
    assert len(calls) == 1
    keywords = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords if k.arg != "base_options"}
    assert keywords == {"score_threshold": KC.HORSE_SCORE, "category_allowlist": [KC.HORSE_CATEGORY],
                        "max_results": KC.HORSE_RESULTS}


def test_the_app_modules_take_them_from_the_single_source():
    assert V.MIN_VISIBILITY is KC.MIN_VISIBILITY and INV.MIN_VISIBILITY is KC.MIN_VISIBILITY
    assert AV.MIN_VISIBILITY is KC.MIN_VISIBILITY
    assert (T.DETECTOR_SCORE, T.DETECTOR_RESULTS) == (KC.HORSE_SCORE, KC.HORSE_RESULTS)
    pattern = re.compile(r"^\s*(MIN_VISIBILITY|DETECTOR_SCORE|DETECTOR_RESULTS)\s*=\s*[0-9]", re.M)
    copies = [p.name for p in (ROOT / "ecvestru").glob("*.py")
              if p.name != "constante_candidat.py" and pattern.search(p.read_text(encoding="utf-8"))]
    assert copies == []


def test_the_inventory_ratio_has_a_name_that_says_what_it_is():
    assert not hasattr(INV, "PROFILE_MAX_RATIO")
    assert INV.INVENTORY_PROFILE_SHOULDER_RATIO_HINT == 0.35
