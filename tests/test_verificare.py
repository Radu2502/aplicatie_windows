import importlib.util
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from ecvestru import verificare as V

SHORT = "clip_151902.mp4"   # cel mai scurt clip (129 de cadre)
have_cache = (V.POSES / "clip_152846.npz").exists() and (V.VIDEOS / SHORT).exists()


def _lm(**over):
    lm = np.zeros((33, 4), np.float32)
    lm[:, 3] = 0.9
    for name, (x, y, v) in over.items():
        lm[V.IDX[name]] = (x, y, 0, v)
    return lm


def test_frame_angles_signs_and_exclusions():
    # mers spre dreapta (nasul în dreapta urechii), trunchi aplecat înainte, călcâi în fața șoldului
    lm = _lm(nose=(110, 90, .9), left_ear=(100, 90, .9), left_shoulder=(110, 100, .9),
             left_hip=(100, 200, .95), right_hip=(100, 200, .3), left_heel=(130, 330, .9))
    r = V.frame_angles(lm)
    assert r["side"] == "left" and r["facing"] == 1.0
    assert r["trunk_deg"] > 0 and r["leg_deg"] < 0 and r["motive"] == []
    lm[V.IDX["left_heel"], 3] = 0.2
    r = V.frame_angles(lm)
    assert r["leg_deg"] is None and r["motive"] == [V.M_LEG]
    missing = np.full((33, 4), np.nan, np.float32)
    assert V.frame_angles(missing)["motive"] == [V.M_NO_PERSON]


@pytest.mark.skipif(not have_cache, reason="cache-ul de landmark-uri nu e local")
def test_angles_identical_to_candidate_rule():
    spec = importlib.util.spec_from_file_location("candidate", V.CANDIDATE)
    cand = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cand)
    d = V.load_poses("clip_152846.mp4")
    w, h = int(d["width"]), int(d["height"])
    checked = 0
    for lm in d["landmarks"]:
        if np.isnan(lm[0, 0]):
            continue
        objs = [SimpleNamespace(x=p[0] / w, y=p[1] / h, visibility=p[3]) for p in lm]
        trunk, leg = cand.angles_for_frame(objs, w, h)
        mine = V.frame_angles(lm)
        for a, b in ((trunk, mine["trunk_deg"]), (leg, mine["leg_deg"])):
            assert (a is None) == (b is None)
            if a is not None:
                assert a == pytest.approx(b, abs=1e-6)
        checked += 1
    assert checked > 100


def test_exclusion_intervals_group_consecutive_frames():
    ok = {"motive": []}
    rows = [ok, {"motive": [V.M_NO_PERSON]}, {"motive": [V.M_NO_PERSON]}, ok,
            {"motive": [V.M_LEG]}, {"motive": [V.M_TRUNK, V.M_LEG]}]
    assert V.exclusion_intervals(rows) == [
        {"de_la": 1, "pana_la": 2, "cadre": 2, "motiv": V.M_NO_PERSON},
        {"de_la": 4, "pana_la": 5, "cadre": 2, "motiv": V.M_LEG},
        {"de_la": 5, "pana_la": 5, "cadre": 1, "motiv": V.M_TRUNK},
    ]


@pytest.mark.skipif(not have_cache, reason="clipurile nu sunt locale")
def test_skeleton_video_is_h264_with_all_frames(tmp_path):
    out = V.skeleton_video(SHORT, out_dir=tmp_path)
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=codec_name,pix_fmt,nb_read_frames,width,height", "-of", "json", str(out)],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    assert probe["codec_name"] == "h264" and probe["pix_fmt"] == "yuv420p"
    assert int(probe["nb_read_frames"]) == len(V.load_poses(SHORT)["landmarks"])
    assert (probe["width"], probe["height"]) == (640, 352)


