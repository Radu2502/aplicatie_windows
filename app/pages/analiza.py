"""Pagina „Analiză”: un clip (din setul dev sau încărcat), alura declarată, intervalul din profil →
unghiuri, cadre nemăsurate, raportul din șabloane, JSON. Doar afișare: logica e în ecvestru/."""
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from ecvestru import analiza_urmarita as AU
from ecvestru import (
    calareti,
    raport,
)
from ecvestru import click_tinta as CT
from ecvestru import masurare as M
from ecvestru import progres as P
from ecvestru import router_v5 as RV5
from ecvestru import schelet_cal as SC
from ecvestru import stocare as ST
from ecvestru import suprapunere as S
from ecvestru.rezultat import GAITS, HEEL_MEDIAPIPE, HEEL_MODEL_M, TargetSeed

ss = st.session_state
REPORT_HIDDEN = ("Raportul text nu e afișat, pentru că nu a trecut verificările automate. "
                 "Valorile, graficul și JSON-ul rămân disponibile.")
FILM_LEGEND = ("Galben = trunchi măsurat, portocaliu = picior măsurat, gri = scheletul călărețului ales, albastru = "
               "verticala prin șold. Cadrele nemăsurate "
               "nu au lanț; banda arată motivul. Filmul nu se salvează.")
BLUR_NOTE = ("Estomparea ascunde detaliile, dar nu e anonimizare: o persoană poate rămâne recognoscibilă. "
             "Verifică filmul înainte să-l folosești.")
BLUR_WAIT = "Estompez filmul (pe clipurile mari poate dura ~30 s)…"
SMOOTH_WAIT = "Netezesc lanțul din film…"
FILM_FAILED = "Filmul nu a putut fi generat. Valorile, graficul și raportul rămân disponibile."
TARGET_UNCLEAR = ("Clip cu țintă neclară: nu e clar cine e călărețul urmărit, deci verdictele nu acoperă acest clip. "
                  "Rezultatul poate fi pe alt călăreț.")
TARGET_LABELS = ("Călărețul 1", "Călărețul 2")
PICK_LABEL = "Aleg călăreții dintr-un cadru (clip cu mai mulți călăreți)"
TARGET_HELP = ("Alege un cadru în care se văd călăreții, apoi dă click pe fiecare. Ținta se urmărește după "
               "poziția și mișcarea calului. Unde metoda a ales alt călăreț, cadrul rămâne nemăsurat, cu motivul.")
MARK_RGB, MARK_RADIUS = (255, 255, 0), 8
JSON_DEPTH = 1   # nivelurile deschise: câmpurile principale; lista pe cadre rămâne restrânsă
UNUSABLE_CLIP = "Clip marcat neutilizabil în inventar, deci rezultatul poate fi greșit. Motivul:"
INTERVAL_CHECKED = "Intervalul implicit vine din data/profil.csv."
PROFILE_UNCLEAR = ("Raportul text nu e afișat: {reason} (regula B4, strat RTMPose). Valorile, graficul și JSON-ul "
                   "rămân disponibile.")
RTMPOSE_ACTIVE = ("Călcâi RTMPose (calea implicită), cu fuziune și interval plauzibil; sensul de mers din "
                  "intervalul analizat după direcția labei (regula B4).")
RTMPOSE_FALLBACK = "RTMPose nu a putut rula ({reason}); rezultatul e calculat pe linia de bază (MediaPipe)."
LICENSE_NOTE = ("RTMPose: codul e Apache-2.0; modelul e antrenat și pe seturi de date doar pentru uz necomercial. "
                "Aplicația e folosită necomercial, pentru cercetare (README.md, „Licențe”).")
HEEL_STABLE_LABEL = "Călcâi stabil (experimental)"
HEEL_STABLE_NOTE = ("Stratul opțional „călcâi stabil”, doar pe calea RTMPose: scoate salturile izolate ale călcâiului "
                    "(Hampel) și călcâiul pus pe piciorul calului. Oprit implicit; neverificat pe clipurile noi. "
                    "Filmul desenează călcâiul din modelul folosit la măsurare.")
HEEL_MODEL_TITLE = "Modelul RTMPose al stratului"
INTERVAL_UNCHECKED = "Intervalul implicit e tot clipul: nu există interval verificat manual."
BAR_START = "Pornesc analiza…"
HORSE_SKELETON_LABEL = "Scheletul calului"
HORSE_SKELETON_HELP = ("Opțional, oprit implicit: cele 17 puncte AP-10K ale calului țintă (calul de sub călăreț pe "
                       "metoda veche, calul urmărit pe urmărirea țintei), desenate în film și adăugate în JSON "
                       "(`horse_keypoints`, schema 1.6). Doar afișare: unghiurile, medianele și raportul nu se "
                       "schimbă. "
                       "Prima rulare pe un clip calculează punctele calului și durează mai mult.")
RIDER_VIEW_LABEL = "Vederea călărețului (router V5, experimental)"
RIDER_VIEW_HELP = ("Vederea călărețului țintă (profil, față, spate, oblic, nu se vede) pe fiecare cadru analizat, în "
                   "film (banda de jos, pe metoda veche) și în JSON (`rider_views`, schema 1.7). Nu schimbă cadrele "
                   "analizate, unghiurile, abținerea sau raportul: intervalul de profil ales mai sus decide. Cere "
                   "ținta urmăririi; prima rulare pe un clip durează mai mult.")
RIDER_VIEW_NOTE = "experimental, doar afișare"
RIDER_VIEW_MISSING = "Vederea V5 nu s-a putut calcula: {reason}"
RIDER_VIEW_TITLE = "Vederea călărețului (router V5, experimental, doar afișare)"
METHOD_TITLE = "Metoda"
METHOD_HELP = ("Automată (implicit): cu doi sau mai mulți cai călăriți în primele secunde ale clipului, urmărirea "
               "țintei (cere un click pe calul călărețului, dacă nu există unul salvat); altfel, metoda veche. "
               "Poți alege și manual una dintre cele două metode.")
COUNT_WAIT = "Număr caii călăriți din primele secunde ale clipului…"
METHOD_CHOSEN = "Metoda aleasă automat: {text}."
METHOD_MANUAL = "Metoda aleasă manual: {label}."
TRACKED_FILM_NOTE = ("Filmul cu scheletul nu e încă disponibil pentru urmărirea țintei (scheletul RTMPose-m al "
                     "țintei); valorile, graficul și raportul sunt pe ținta urmărită.")
JUMP_LABEL = "Deschide cadrul {frame} (primul pierdut) pentru încă un click pe {label}"
JUMP_NOTE = ("Am deschis cadrul {frame}, primul în care ținta s-a pierdut. Dă click pe {label} în imagine, apoi apasă "
             "din nou „Analizează”: click-ul se adaugă la alegerile de până acum.")


@st.cache_data(show_spinner=False)
def clip_info(name: str) -> tuple[int, float]:
    return ST.video_info(M.V.VIDEOS / name)


@st.cache_data(show_spinner=False)
def clip_hash(name: str) -> str:
    return ST.file_hash(M.V.VIDEOS / name)


def make_film(result, name: str, data: bytes | None, blur: bool = False, smooth: bool = False) -> bytes | None:
    try:
        if data is None:
            return S.clip_film(M.V.VIDEOS / name, result, blur, smooth)
        return S.upload_film(data, name, result, blur, smooth)
    except RuntimeError:
        return None


def shown_film(key, label: str, result, film: bytes | None, name: str, data: bytes | None) -> bytes | None:
    """Filmul normal sau o variantă (estompat, netezit pentru afișare); variantele se fac la cerere și se păstrează doar
    cât e aleasă una dintre ele."""
    blur, smooth = bool(ss.get("an-blur")), bool(ss.get("an-smooth"))
    if not (blur or smooth) or film is None:
        ss.pop("an-variants", None)
        return film
    if ss.get("an-variants", (None,))[0] != key:
        ss["an-variants"] = (key, {})
    films = ss["an-variants"][1]
    if (label, blur, smooth) not in films:
        with st.spinner(BLUR_WAIT if blur else SMOOTH_WAIT):
            films[(label, blur, smooth)] = make_film(result, name, data, blur, smooth)
    if blur:
        st.caption(BLUR_NOTE)
    return films[(label, blur, smooth)]


def chart_series(result, name: str, data: bytes | None) -> tuple[list, list]:
    """Unghiurile din grafic: cele măsurate sau, cu netezirea pentru afișare, cele netezite (doar în grafic). Pe ținta
    urmărită, doar cele măsurate (netezirea pentru afișare citește scheletul metodei vechi)."""
    smooth = bool(ss.get("an-smooth")) and result.candidate != AU.CANDIDATE_LABEL
    try:
        if data is None:
            return S.display_series(M.V.VIDEOS / name, result, smooth)
        return S.upload_display_series(data, name, result, smooth)
    except RuntimeError:     # punctele lipsesc: graficul rămâne cu valorile măsurate
        return S.display_series(None, result, False)


def make_analysis(name: str, data: bytes | None, gait: str, segment: tuple[int, int], seeds=(),
                  heel_source: str = HEEL_MEDIAPIPE, heel_stable: bool = False, heel_model: str = HEEL_MODEL_M,
                  stages: P.Stages | None = None, horse_skeleton: bool = False):
    target, anchors = (seeds[0], seeds[1:]) if seeds else (None, ())
    layer = {"heel_stable": True, "heel_model": heel_model} if heel_stable else {}   # oprit: fără argumentele stratului
    layer |= {"horse_skeleton": True} if horse_skeleton else {}
    if data is None:
        return M.analyze(M.V.VIDEOS / name, gait, segment, target=target, anchors=anchors, heel_source=heel_source,
                         stages=stages, **layer)
    return M.analyze_upload(data, name, gait, segment, target=target, anchors=anchors, extended=True,
                            heel_source=heel_source, stages=stages, **layer)


def run_tracked(name: str, data: bytes | None, gait: str, segment: tuple[int, int], click,
                horse_skeleton: bool = False) -> tuple[list, dict]:
    """Analiza pe ținta urmărită, cu bara de progres pe cele 4 etape; fără film încă."""
    bar = st.progress(0.0, text=BAR_START)
    horse = {"horse_skeleton": True} if horse_skeleton else {}   # oprit: fără argumente noi

    def started(stage: str, k: int, total: int) -> None:
        bar.progress(P.fraction(0, 1, k, total), text=P.bar_text("", stage, k, total))
    stages = P.Stages(on_start=started)
    try:
        if data is None:
            res = AU.analyze(M.V.VIDEOS / name, gait, segment, click=click, stages=stages, **horse)
        else:
            res = M.with_temp_file(data, name, lambda path: AU.analyze(path, gait, segment, name=name, click=click,
                                                                      stages=stages, uploaded=True, **horse))
    finally:
        bar.empty()
    return [("", res, None)], {"": stages.durations()}


def auto_choice(name: str, data: bytes | None, content: str) -> AU.MethodChoice:
    """Numărul de cai călăriți în primele secunde (regula metodei automate), pe clipul din set sau pe cel încărcat."""
    if data is None:
        return AU.method_choice(M.V.VIDEOS / name)
    return M.with_temp_file(data, name, lambda path: AU.method_choice(path, name=ST.upload_name(content)))


def rider_view_note() -> str:
    """Textul de lângă bifa V5."""
    return RIDER_VIEW_NOTE


def add_views(result, name: str, data: bytes | None, click=None, stages: P.Stages | None = None):
    """Rezultatul cu vederea V5 (pe clipul din set sau pe cel încărcat); fără țintă, rezultatul neschimbat și un
    avertisment."""
    try:
        if data is None:
            return RV5.with_views(M.V.VIDEOS / name, result, click, stages)
        return M.with_temp_file(data, name, lambda path: RV5.with_views(path, result, click, stages))
    except RV5.ViewsUnavailable as error:
        st.warning(RIDER_VIEW_MISSING.format(reason=error))
        return result


def show_rider_views(result) -> None:
    """Vederea V5 pe intervale de cadre consecutive cu aceeași vedere."""
    if result.rider_views is None:
        return
    rows, start = [], None
    frames = result.rider_views.frames
    for i, f in enumerate(frames):
        if start is None:
            start = f
        last = i == len(frames) - 1 or frames[i + 1].view != f.view or frames[i + 1].frame != f.frame + 1
        if last:
            rows.append({"de la cadrul": start.frame, "până la cadrul": f.frame,
                         "vederea": S.VIEW_NAMES.get(f.view, "fără țintă") if f.view else "fără țintă"})
            start = None
    st.markdown(f"**{RIDER_VIEW_TITLE}**")
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def run_analyses(name: str, data: bytes | None, gait: str, segment: tuple[int, int], targets: dict,
                 heel_source: str, heel_stable: bool, heel_model: str, horse_skeleton: bool = False,
                 rider_view: bool = False) -> tuple[list, dict]:
    """Analiza fiecărui călăreț ales (sau a clipului, fără alegeri) și filmul lui, cu bara de progres pe etape.
    Întoarce [(eticheta, rezultatul, filmul)] și duratele etapelor pe etichetă."""
    runs = list(targets.items()) or [("", ())]
    bar = st.progress(0.0, text=BAR_START)
    done, times = [], {}
    try:
        for i, (label, seeds) in enumerate(runs):
            def started(stage: str, k: int, total: int, i=i, label=label) -> None:
                bar.progress(P.fraction(i, len(runs), k, total), text=P.bar_text(label, stage, k, total))
            stages = P.Stages(on_start=started)
            stages.plan([P.FILM])
            res = make_analysis(name, data, gait, segment, seeds, heel_source, heel_stable, heel_model, stages,
                                horse_skeleton)
            if rider_view:
                res = add_views(res, name, data, None, stages)
            with stages.stage(P.FILM):
                film = make_film(res, name, data)
            done.append((label, res, film))
            times[label] = stages.durations()
    finally:
        bar.empty()
    return done, times


def jump_to_lost(content: str, label: str, frame: int) -> None:
    """Alegerea călăreților, deschisă pe primul cadru pierdut, pentru același călăreț.

    Callback: rulează înaintea widget-urilor, deci le poate schimba valoarea."""
    ss["an-pick"] = True
    ss["an-slot"] = label
    ss[f"an-pick-frame-{content}"] = frame
    ss["an-jump-note"] = JUMP_NOTE.format(frame=frame, label=label)


def clip_frame(name: str, data: bytes | None, frame: int):
    if data is None:
        return ST.read_frame(M.V.VIDEOS / name, frame)
    return M.with_temp_file(data, name, lambda path: ST.read_frame(path, frame))


def pick_targets(name: str, data: bytes | None, content: str, segment: tuple[int, int]) -> dict:
    """Click-urile utilizatorului: câte unul pentru fiecare călăreț ales."""
    targets = ss.setdefault("an-targets", {}).setdefault(content, {})
    st.caption(TARGET_HELP)
    if note := ss.pop("an-jump-note", None):
        st.info(note)
    slot = st.radio("Click pentru", TARGET_LABELS, horizontal=True, key="an-slot")
    frame = segment[0]
    if segment[0] < segment[1]:
        frame_key = f"an-pick-frame-{content}"
        if not segment[0] <= ss.get(frame_key, segment[0]) <= segment[1]:
            ss.pop(frame_key)                        # alt interval: cadrul vechi nu mai e în el
        ss.setdefault(frame_key, segment[0])
        frame = st.slider("Cadrul", segment[0], segment[1], key=frame_key)
    image = clip_frame(name, data, frame)
    view = CT.make_view(image.shape[1], image.shape[0])
    shown = CT.render(image, view, {})
    for label, seeds in targets.items():
        for target in (seed for seed in seeds if seed.frame == frame):
            centre = tuple(map(round, CT.to_display(view, target.x, target.y)))
            cv2.circle(shown, centre, MARK_RADIUS, MARK_RGB, 2, cv2.LINE_AA)
            cv2.putText(shown, label[-1], centre, cv2.FONT_HERSHEY_SIMPLEX, 0.7, MARK_RGB, 2, cv2.LINE_AA)
    click = streamlit_image_coordinates(shown, key=f"an-pick-{content}-{frame}", cursor="crosshair")
    if click and click.get("unix_time") != ss.get("an-last-click"):
        ss["an-last-click"] = click["unix_time"]
        x, y = CT.to_original(view, click["x"], click["y"], click.get("width"), click.get("height"))
        targets.setdefault(slot, []).append(TargetSeed(frame=frame, x=x, y=y))
        st.rerun()
    if st.button("Șterge alegerile", key="an-clear", disabled=not targets):
        targets.clear()
        st.rerun()
    return targets


def show_summary(result) -> None:
    measured, reasons = M.frame_summary(result)
    st.markdown(f"**Măsurate: {measured} din {len(result.frames)} de cadre** (trunchi). Restul:")
    for text, count in reasons:
        st.markdown(f"- {text}: {count}")


def show(result, film: bytes | None, label: str = "", series: tuple[list, list] | None = None,
         times: list | None = None, content: str = "") -> None:
    if label:
        show_summary(result)
    if result.pipeline == "rtmpose":
        st.caption(RTMPOSE_ACTIVE)
    if lost := M.lost_target_intervals(result):
        where = ", ".join(f"{a}–{b}" for a, b in lost)
        st.warning(f"Ținta s-a pierdut în cadrele {where}. Alege un cadru de acolo, dă click pe același călăreț "
                   "și analizează din nou.")
        first, rider = M.first_lost_frame(result), label or TARGET_LABELS[0]
        st.button(JUMP_LABEL.format(frame=first, label=rider), key=f"an-jump-{rider}", on_click=jump_to_lost,
                  args=(content, rider, first))
    if times is not None:
        st.caption(P.durations_text(times))
    fmt = lambda v: "—" if v is None else (f"{v:+d}°" if v else "0°")  # noqa: E731
    m = st.columns(4)
    m[0].metric("Trunchi (mediană)", fmt(result.trunk.median_rounded), help="+ = aplecat înainte")
    m[1].metric("Picior (mediană)", fmt(result.leg.median_rounded), help="- = călcâi în fața șoldului")
    m[2].metric("Măsurat: trunchi", f"{result.trunk.coverage_pct}%")
    m[3].metric("Măsurat: picior", f"{result.leg.coverage_pct}%")

    trunk, leg = series or ([f.trunk_deg for f in result.frames], [f.leg_deg for f in result.frames])
    df = pd.DataFrame({"cadru": [f.frame for f in result.frames], "trunchi (°)": trunk,
                       "picior (°)": leg}).set_index("cadru")
    left, right = st.columns([2, 3])
    with left:
        if film is None and result.candidate == AU.CANDIDATE_LABEL:
            st.caption(TRACKED_FILM_NOTE)
        elif film is None:
            st.warning(FILM_FAILED)
        else:
            st.video(film, format="video/mp4")
            st.caption(FILM_LEGEND)
            st.caption(S.SIDE_LEGEND)
            st.caption(S.VISIBILITY_LEGEND)
            st.caption(S.REJECTED_LEGEND)
            if result.horse_keypoints is not None:
                st.caption(S.HORSE_LEGEND)
    with right:
        st.line_chart(df, x_label="cadru (indice OpenCV)", y_label="grade față de verticală")
        if ss.get("an-smooth"):
            st.caption(S.SMOOTH_NOTE)
        else:
            st.caption("Golurile sunt cadre nemăsurate. Valorile pe cadre nu sunt netezite; în raport, unghiurile "
                       f"sunt rotunjite la {M.ROUND_DEG}°.")

    st.subheader("Raport")
    text = raport.report_text(result)
    if hidden := raport.hidden_reason(result):
        st.warning(PROFILE_UNCLEAR.format(reason=hidden))
    elif raport.report_passes_checks(text, result):
        for line in text.splitlines():
            st.markdown(line)
    else:
        st.warning(REPORT_HIDDEN)

    st.subheader("Nemăsurat")
    if result.unmeasured:
        st.dataframe(pd.DataFrame([{"de la": u.start, "până la": u.end, "cadre": u.frames, "motiv": u.reason_text}
                                   for u in result.unmeasured]), hide_index=True, width="stretch")
    else:
        st.write("Toate cadrele din interval sunt măsurate.")
    with st.expander("Rezultatul (JSON)"):
        st.json(result.to_json(), expanded=JSON_DEPTH)
    json_name = f"{Path(result.video).stem}{label.replace(' ', '-')}.json"
    st.download_button("Descarcă rezultatul (JSON)", result.to_json(), file_name=json_name,
                       mime="application/json", key=f"an-json-{label}")
    st.info(result.limitation)


st.title("Analiză")
if problem := M.root_path_problem():
    st.error(problem)
    st.stop()
st.caption("Călărețul e găsit cu varianta fixă a candidatului "
           f"({ST.candidate_label()}); unghiurile se măsoară doar în intervalul ales, filmat din profil.")

source = st.radio("Clipul", ["set", "upload"] if M.has_dev_set() else ["upload"], horizontal=True, key="an-source",
                  format_func=lambda s: "Din setul de dezvoltare" if s == "set" else "Încarc un clip")
if source == "set":
    name = st.selectbox("Clip", ST.dev_videos(), key="an-clip",
                        help="Doar clipurile din setul de dezvoltare.")
    n_frames, fps = clip_info(name)
    data = None
    if name in calareti.unclear_clips():
        st.warning(TARGET_UNCLEAR)
    if reason := calareti.unusable_reason(name):
        st.warning(f"{UNUSABLE_CLIP} {reason}")
    content = clip_hash(name)
else:
    up = st.file_uploader("Clip video (filmat din profil)", type=[s[1:] for s in M.UPLOAD_SUFFIXES], key="an-upload")
    if up is None:
        st.info(f"Încarcă un clip. Videoul stă doar într-un folder temporar, care se șterge după analiză. "
                f"{M.UPLOAD_KEPT}")
        st.stop()
    name, data = up.name, up.getvalue()
    n_frames, fps = M.upload_info(data, name)
    if problem := M.upload_problem(n_frames):
        st.error(problem)
        st.stop()
    content = ST.bytes_hash(data)

methods = list(AU.METHOD_LABELS)
method = st.radio(METHOD_TITLE, methods, index=methods.index(AU.APP_DEFAULT_METHOD), format_func=AU.METHOD_LABELS.get,
                  horizontal=True, key="an-method", help=METHOD_HELP)
choice_box = st.empty()
known = ss["an-choice"][1] if ss.get("an-choice", (None,))[0] == content else None
if method != AU.METHOD_AUTO:
    choice_box.caption(METHOD_MANUAL.format(label=AU.METHOD_LABELS[method]))
elif known is not None:
    choice_box.info(METHOD_CHOSEN.format(text=known.text))
tracked = method == AU.METHOD_TRACKED or (method == AU.METHOD_AUTO and known is not None
                                          and known.method == AU.METHOD_TRACKED)
gait = st.radio("Alura declarată", GAITS, horizontal=True, key="an-gait")
a, b = M.default_segment(name, n_frames) if data is None else (0, n_frames - 1)
last = (n_frames - 1) / fps
start_s, end_s = st.slider("Intervalul în care călărețul e filmat din profil (secunde)", 0.0, last,
                           (a / fps, b / fps), step=1 / fps, format="%.2f s",
                           key=f"an-seg-{name if data is None else content}")
segment = (round(start_s * fps), min(n_frames - 1, round(end_s * fps)))
st.caption(f"Cadrele {segment[0]}–{segment[1]} din {n_frames} ({fps:.0f} cadre/s). "
           + ("" if data is not None else INTERVAL_CHECKED if M.has_profile_interval(name) else INTERVAL_UNCHECKED))

targets = pick_targets(name, data, content, segment) if st.checkbox(PICK_LABEL, key="an-pick") else {}
heel_source = M.APP_DEFAULT_HEEL_SOURCE
stable = st.checkbox(HEEL_STABLE_LABEL, value=M.APP_DEFAULT_HEEL_STABLE, key="an-heel-stable", help=HEEL_STABLE_NOTE)
heel_model = HEEL_MODEL_M
if stable:
    labels = {label: model for model, label in M.HEEL_MODEL_LABELS.items()}
    heel_model = labels[st.radio(HEEL_MODEL_TITLE, list(labels), horizontal=True, key="an-heel-model")]
heel_stable = stable
horse_skeleton = st.checkbox(HORSE_SKELETON_LABEL, value=SC.APP_DEFAULT_HORSE_SKELETON, key="an-horse-skeleton",
                             help=HORSE_SKELETON_HELP)
rider_view = st.checkbox(RIDER_VIEW_LABEL, value=RV5.APP_DEFAULT_RIDER_VIEW, key="an-rider-view",
                         help=RIDER_VIEW_HELP)
st.caption(rider_view_note())
st.caption(LICENSE_NOTE)
key = (content, segment, gait, heel_source, heel_stable, heel_model,
       tuple((k, s.frame, s.x, s.y) for k, seeds in sorted(targets.items()) for s in seeds), method, horse_skeleton,
       rider_view)
if tracked:
    cached = AU.cache_path(name if data is None else ST.upload_name(content), content).exists()
    st.caption(AU.estimate_text(n_frames, cached))
if st.button("Analizează", type="primary", key="an-run"):
    try:
        run = method
        if method == AU.METHOD_AUTO:
            with st.spinner(COUNT_WAIT):
                known = auto_choice(name, data, content)
            ss["an-choice"] = (content, known)
            choice_box.info(METHOD_CHOSEN.format(text=known.text))
            run = known.method
        if run == AU.METHOD_TRACKED:
            click = next(iter(targets.get(TARGET_LABELS[0], ())), None)
            done, times = run_tracked(name if data is None else ST.upload_name(content), data, gait, segment, click,
                                      horse_skeleton)
            if rider_view:
                done = [(label, add_views(res, name, data, click), film) for label, res, film in done]
        else:
            done, times = run_analyses(name, data, gait, segment, targets, heel_source, heel_stable, heel_model,
                                       horse_skeleton, rider_view)
    except AU.TargetNeeded:
        st.warning(AU.CLICK_NEEDED_TEXT)
    except ValueError as e:
        st.error(str(e))
    else:
        ss["an-result"] = (key, done[0][1], done)
        ss["an-times"] = (key, times)
if ss.get("an-result") and ss["an-result"][0] == key:
    analyses = ss["an-result"][2]
    stage_times = ss["an-times"][1] if ss.get("an-times", (None,))[0] == key else {}
    if analyses[0][1].fallback_reason is not None:
        st.warning(RTMPOSE_FALLBACK.format(reason=analyses[0][1].fallback_reason))
    if analyses[0][0]:
        for tab, (label, res, film) in zip(st.tabs([a[0] for a in analyses]), analyses, strict=True):
            with tab:
                show(res, shown_film(key, label, res, film, name, data), label, chart_series(res, name, data),
                     stage_times.get(label), content)
                show_rider_views(res)
    else:
        show(analyses[0][1], shown_film(key, "", analyses[0][1], analyses[0][2], name, data), "",
             chart_series(analyses[0][1], name, data), stage_times.get(""), content)
        show_rider_views(analyses[0][1])
