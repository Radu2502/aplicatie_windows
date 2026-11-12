"""Configurația finală (ecvestru/analiza_urmarita.py): alegerea țintei (click, click salvat, automată, cere click),
punctele pe cadru din RTMPose (partea, pragul, sensul, trunchiul peste 45°), orientarea B4, intervalul plauzibil,
trecerea grea salvată. Date inventate, fără video real; metoda veche (masurare.analyze) nu e atinsă."""
import numpy as np
import pytest

from ecvestru import analiza_urmarita as AU
from ecvestru import calitate as C
from ecvestru import masurare as M
from ecvestru import stocare as ST
from ecvestru import tinta as T
from ecvestru import urmarire_multi as U
from ecvestru.rezultat import FrameMeasure, TargetSeed

FPS = 30.0
HORSE_A, HORSE_B = (100.0, 200.0, 300.0, 300.0), (500.0, 200.0, 700.0, 300.0)


def test_the_configuration_and_its_labels():
    assert AU.METHOD_LABELS[AU.METHOD_OLD] == "Metoda veche, un călăreț"
    assert AU.APP_DEFAULT_METHOD in AU.METHOD_LABELS and AU.AUTO_SECONDS == T.MAX_GAP_SECONDS
    assert AU.STAGES == (AU.PASS, AU.TRACKING, AU.ANGLES) and str(AU.TRACKING_VERSION) in \
        AU.CANDIDATE_LABEL


def rider_points(hip=(200.0, 210.0), lean_x: float = 0.0, facing_right: bool = True) -> np.ndarray:
    p = np.zeros((26, 2), np.float32)
    p[5] = p[6] = (hip[0] + lean_x, hip[1] - 80)       # umerii
    p[11] = p[12] = hip                                # șoldurile
    p[24] = p[25] = (hip[0], hip[1] + 90)              # călcâiele
    p[3] = p[4] = (hip[0], hip[1] - 110)               # urechile
    p[0] = (hip[0] + (10 if facing_right else -10), hip[1] - 105)
    p[20] = p[21] = p[22] = p[23] = (hip[0] + (15 if facing_right else -15), hip[1] + 92)   # degetele
    return p


def scores(left_hip: float = 0.9, right_hip: float = 0.8, low: tuple[int, ...] = ()) -> np.ndarray:
    s = np.full(26, 0.9, np.float32)
    s[11], s[12] = left_hip, right_hip
    for i in low:
        s[i] = 0.1
    return s


def test_frame_points_follow_the_candidate_rules_with_rtmpose_scores():
    pts, side = AU.frame_points(rider_points(), scores(left_hip=0.4, right_hip=0.9))
    assert side == "right" and pts["facing"] == 1.0 and pts["hip"] == (200.0, 210.0)
    pts, _ = AU.frame_points(rider_points(facing_right=False), scores(low=(24,)))
    assert pts["facing"] == -1.0 and pts["heel"] is None and pts["shoulder"] is not None   # sub CP.SCORE_MIN (0,3)


def clip(horses, people, keypoints):
    return U.ClipTracks(fps=FPS, n_frames=len(horses), horses=horses, people=people, keypoints=keypoints)


def test_build_raw_gives_the_candidate_format_with_reasons():
    person = (150.0, 100.0, 250.0, 320.0)
    tracks = clip([{1: HORSE_A}] * 4, [{5: person}] * 4,
                  [{5: rider_points()}, {5: rider_points()}, {5: rider_points(lean_x=200.0)}, {5: rider_points()}])
    sc = [{5: scores()}, {5: scores(low=(5, 6))}, {5: scores()}, {}]
    choice = U.Choice([], [1, 1, 1, 1], [5, 5, 5, 5], 0, None, 0, 0)
    raw, sides = AU.build_raw(tracks, sc, choice, 848.0, 480.0)
    cand = M.candidate_module()
    assert raw[0]["motiv"] is None and raw[1]["motiv"] == cand.REASON_POINTS and raw[2]["motiv"] == cand.REASON_TRUNK
    assert raw[3] == {"points": None, "motiv": cand.REASON_NO_PERSON} and sides[:3] == ["left"] * 3
    lost = U.Choice([], [None] * 4, [None] * 4, 0, None, 0, 0)
    assert AU.build_raw(tracks, sc, lost, 848.0, 480.0)[0][0]["motiv"] == T.REASON_LOST
    frames = M.frame_measures(raw, 4)
    assert frames[0].trunk_deg == pytest.approx(0.0) and frames[1].trunk_deg is None


def test_auto_target_needs_exactly_one_ridden_horse_in_the_first_seconds():
    person, on = (150.0, 100.0, 250.0, 320.0), rider_points()
    other = rider_points(hip=(600.0, 210.0))
    n = 10
    one = clip([{1: HORSE_A, 2: HORSE_B}] * n, [{5: person}] * n, [{5: on}] * n)
    seed = AU.auto_target(one)
    assert seed == TargetSeed(frame=0, x=200.0, y=250.0)
    two = clip([{1: HORSE_A, 2: HORSE_B}] * n, [{5: person, 6: (550.0, 100.0, 650.0, 320.0)}] * n,
               [{5: on, 6: other}] * n)
    assert AU.auto_target(two) is None
    nobody = clip([{1: HORSE_A}] * n, [{}] * n, [{}] * n)
    assert AU.auto_target(nobody) is None


def test_b4_orientation_votes_with_the_rtmpose_foot_inside_the_segment():
    n = 20
    raw = {f: {"points": {"facing": -1.0, "shoulder": (0, 0), "hip": (0, 1), "heel": (0, 2)}, "motiv": None}
           for f in range(n)}
    foot = np.ones(n)
    out, summary, sources = AU.orient(raw, foot, FPS, (5, 14))
    assert all(out[f]["points"]["facing"] == 1.0 for f in range(5, 15)) and out[0]["points"]["facing"] == -1.0
    assert summary.changed_frames == 10 and summary.frames_with_facing == 10 and set(sources) == set(range(5, 15))
    assert raw[5]["points"]["facing"] == -1.0                                     # brutul primit rămâne neatins


def test_the_foot_direction_uses_the_target_toes_and_heel():
    person = (150.0, 100.0, 250.0, 320.0)
    tracks = clip([{1: HORSE_A}] * 2, [{5: person}] * 2, [{5: rider_points()}, {5: rider_points(facing_right=False)}])
    choice = U.Choice([], [1, 1], [5, 5], 0, None, 0, 0)
    raw, sides = AU.build_raw(tracks, [{5: scores()}] * 2, choice, 848.0, 480.0)
    assert AU.foot_direction(tracks, choice, sides, raw).tolist() == [1.0, -1.0]


def test_implausible_legs_are_removed_with_the_reason():
    frames = [FrameMeasure(frame=0, trunk_deg=1.0, leg_deg=C.LEG_ANGLE_MAX + 1, reason=None),
              FrameMeasure(frame=1, trunk_deg=1.0, leg_deg=0.0, reason=None)]
    out = AU.plausible_legs(frames)
    assert out[0].leg_deg is None and out[0].reason == M.REASON_IMPLAUSIBLE_ANGLE and out[1].leg_deg == 0.0


def det(x):
    return np.array([[x, 100.0, x + 40, 180.0, 0.9]])


class FakePose:
    def __init__(self):
        self.frames = []

    def __call__(self, image, bbox):
        self.frames.append(int(image[0]))
        return rider_points(), scores()


def test_one_pass_detects_tracks_and_poses_each_frame_once_and_the_pass_is_saved(tmp_path):
    found = [{U.PERSON: det(310 - k), U.HORSE: det(300 - k)} for k in range(6)]     # omul pe cal
    found[2] = {U.PERSON: det(900.0), U.HORSE: det(298.0)}              # cadrul 2: omul e departe de cal
    images = [np.array([k]) for k in range(6)]
    seen, pose = [], FakePose()

    def detector(image):
        seen.append(int(image[0]))
        return found[int(image[0])]
    tracks, sc = AU.run_pass(iter(images), FPS, detector, pose)
    assert seen == list(range(6)) and tracks.n_frames == 6 and tracks.fps == FPS
    assert all(set(h) == {1} for h in tracks.horses)
    assert pose.frames == [0, 1, 3, 4, 5] and sc[2] == {} and set(tracks.keypoints[2]) == set()
    assert np.allclose(tracks.keypoints[0][1], rider_points())
    AU.save_pass(tmp_path / "t.npz", tracks, sc)
    back, back_scores = AU.load_pass(tmp_path / "t.npz")
    assert back.horses == tracks.horses and np.allclose(back_scores[0][1], sc[0][1]) and back_scores[2] == {}


def test_the_frames_are_read_once_and_only_the_first_ones_when_asked(tmp_path):
    import cv2
    path = tmp_path / "c.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (32, 24))
    for k in range(5):
        writer.write(np.full((24, 32, 3), 40 * k, np.uint8))
    writer.release()
    assert len(list(AU.read_frames(path))) == 5 and len(list(AU.read_frames(path, max_frames=3))) == 3
    assert AU.video_fps(path) == pytest.approx(FPS)
    fps, width, height = ST.video_props(path)                   # doar metadatele
    assert fps == pytest.approx(FPS) and (width, height) == (32.0, 24.0)


def test_only_people_touching_a_horse_get_a_skeleton():
    tracks = clip([{1: HORSE_A}], [{5: (150.0, 100.0, 250.0, 320.0), 6: (800.0, 0.0, 840.0, 50.0)}], [{}])
    assert AU.near_horse(tracks, 0) == [5]


# ---------- metoda după numărul de cai călăriți în primele 2 s ----------

def riders(n_ridden: int, n: int = 10):
    """Un clip cu doi cai și `n_ridden` dintre ei cu un om pe ei, pe toate cele n cadre."""
    people = {5: (150.0, 100.0, 250.0, 320.0), 6: (550.0, 100.0, 650.0, 320.0)}
    points = {5: rider_points(), 6: rider_points(hip=(600.0, 210.0))}
    keep = [5, 6][:n_ridden]
    return clip([{1: HORSE_A, 2: HORSE_B}] * n, [{k: people[k] for k in keep}] * n,
                [{k: points[k] for k in keep}] * n)


@pytest.mark.parametrize(("n_ridden", "method"), [(0, AU.METHOD_OLD), (1, AU.METHOD_OLD), (2, AU.METHOD_TRACKED)])
def test_the_method_follows_the_number_of_ridden_horses(n_ridden, method):
    tracks = riders(n_ridden)
    assert AU.ridden_horses(tracks) == n_ridden and AU.method_for(n_ridden) == method
    assert AU.MethodChoice(n_ridden).method == method


def test_no_horse_at_all_keeps_the_old_method():
    empty = clip([{}] * 5, [{}] * 5, [{}] * 5)
    assert AU.ridden_horses(empty) == 0 and AU.MethodChoice(0).method == AU.METHOD_OLD


def test_only_the_first_two_seconds_count():
    late = riders(2, n=200)
    window = AU.first_frames(late.n_frames, late.fps)
    assert window == round(AU.AUTO_SECONDS * FPS) + 1
    for f in range(window):
        late.people[f], late.keypoints[f] = {5: late.people[f][5]}, {5: late.keypoints[f][5]}
    assert AU.ridden_horses(late) == 1 and AU.method_for(AU.ridden_horses(late)) == AU.METHOD_OLD


def test_the_choice_explains_itself():
    assert AU.MethodChoice(2).text == "2 cai călăriți în primele 2 s → urmărirea țintei"
    assert AU.MethodChoice(1).text == "1 cal călărit în primele 2 s → metoda veche (un călăreț)"
    assert AU.MethodChoice(0).text == "0 cai călăriți în primele 2 s → metoda veche (un călăreț)"
    assert AU.APP_DEFAULT_METHOD == AU.METHOD_AUTO and AU.RIDDEN_FOR_TRACKING == 2


@pytest.mark.metoda_reala
def test_the_count_comes_from_the_saved_full_pass_then_the_short_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "results_dir", lambda: tmp_path / "rezultate")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"nu e un film")
    monkeypatch.setattr(ST, "video_info", lambda path: (100, FPS))
    calls = []

    def short(path, frames, threads=1):
        calls.append(frames)
        tracks = riders(2, n=frames)
        return tracks, [{} for _ in range(frames)]
    monkeypatch.setattr(AU, "early_pass", short)
    assert AU.method_choice(video).ridden == 2 and calls == [61]
    assert AU.method_choice(video).ridden == 2 and calls == [61]          # a doua oară, din trecerea scurtă salvată
    AU.save_pass(AU.cache_path("clip.mp4", M.file_hash(video)), riders(1, n=100), [{} for _ in range(100)])
    assert AU.method_choice(video).ridden == 1 and calls == [61]          # trecerea întreagă are întâietate


def test_the_short_pass_reads_only_the_first_frames(monkeypatch):
    seen = []
    monkeypatch.setattr(AU, "one_pass", lambda path, threads=1, max_frames=None: seen.append(max_frames) or "trecere")
    assert AU.early_pass("clip.mp4", 3) == "trecere" and seen == [3]
