import numpy as np

from ecvestru.inventar import frame_info, load_visual, orientation, summarize, typical_count


def pts(**over):
    base = {
        "left_shoulder": (100, 100, 0.9), "right_shoulder": (110, 100, 0.3),
        "left_hip": (100, 200, 0.9), "right_hip": (108, 200, 0.4),
        "left_heel": (105, 300, 0.8), "right_heel": (110, 350, 0.2),
    }
    base.update(over)
    return base


def test_orientation():
    assert orientation(720, 1280) == "vertical"
    assert orientation(848, 480) == "orizontal"


def test_frame_info_profile_left_side_visible():
    info = frame_info(pts())
    assert info["side"] == "left"
    assert info["trunk_ok"] and info["leg_ok"]
    assert info["shoulder_ratio"] < 0.35


def test_frame_info_hidden_heel_and_frontal_shoulders():
    info = frame_info(pts(left_heel=(105, 350, 0.1),
                          left_shoulder=(40, 100, 0.9), right_shoulder=(160, 100, 0.9)))
    assert info["trunk_ok"] and not info["leg_ok"]
    assert info["shoulder_ratio"] > 1.0


def test_frame_info_picks_more_visible_hip():
    info = frame_info(pts(right_hip=(108, 200, 0.95)))
    assert info["side"] == "right"
    assert not info["trunk_ok"]  # umărul drept are visibility 0.3


def test_typical_count_ignores_rare_spikes():
    assert typical_count([1] * 95 + [2] * 5) == 1
    assert typical_count([1] * 70 + [2] * 30) == 2
    assert typical_count([0] * 10) == 0
    assert typical_count([]) == 0


def test_load_visual_missing_file(tmp_path):
    assert load_visual(tmp_path / "nu_exista.csv") == {}


def test_summarize_counts_visible_frames_and_merges_visual():
    lm = np.zeros((33, 4), np.float32)
    for name, i in {"left_shoulder": 11, "right_shoulder": 12, "left_hip": 23,
                    "right_hip": 24, "left_heel": 29, "right_heel": 30}.items():
        lm[i] = (*pts()[name][:2], 0, pts()[name][2])
    no_heel = lm.copy()
    no_heel[29, 3] = 0.1
    missing = np.full((33, 4), np.nan, np.float32)
    data = {"landmarks": np.stack([lm, no_heel, missing, missing]), "counts": np.array([1, 1, 0, 2]),
            "fps": 30.0, "width": 480, "height": 848}
    row, extra = summarize("c.mp4", data, {"calaret": "R1", "alura": "trap", "persoane": "2", "profil": "da",
                                            "utilizabil": "da", "motiv": "x"})
    assert row["orientare"] == "vertical" and row["rezolutie"] == "480x848"
    assert row["procent_trunchi"] == 50.0 and row["procent_picior"] == 25.0
    assert row["persoane"] == "2" and row["utilizabil"] == "da"
    assert row["calaret"] == "R1" and row["alura"] == "trap" and row["cal"] == "" and extra["pct_found"] == 50.0
