# Aplicație de analiză a posturii călărețului

Aplicația măsoară postura călărețului dintr-un clip video filmat din profil. Găsește călărețul și calul, extrage
scheletul și calculează, cadru cu cadru, unghiul trunchiului și unghiul piciorului (umăr–șold–călcâi) față de
verticală. Pe cadrele unde măsurarea nu e sigură se abține și spune de ce. La final arată medianele, un grafic, un
film cu scheletul desenat, un raport scurt și un JSON cu toate valorile. Rulează local, în browser; clipul nu se
trimite nicăieri.

## Instalare pe Windows 11

### 1. Programul uv (o singură dată)

uv pregătește Python-ul și bibliotecile aplicației.

1. Apasă tasta Windows, scrie **PowerShell** și deschide „Windows PowerShell”.
2. Lipește rândul de mai jos și apasă Enter:

   ```
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. Când scrie că s-a terminat, închide fereastra.

### 2. Folderul aplicației: C:\postura

1. Descarcă arhiva (pe GitHub: **Code → Download ZIP**).
2. Click dreapta pe zip → **Properties** (Proprietăți). Jos, bifează **Unblock** (Deblochează), apoi **OK**.
   Altfel Windows poate bloca fișierele dezarhivate.
3. Dezarhivează și redenumește folderul astfel încât să existe **C:\postura**, cu fișierul **porneste.bat** direct
   în el.

Calea folderului **nu** trebuie să aibă diacritice sau caractere speciale (modelele nu se deschid). Nu-l pune pe
Desktop, în Documente sau în OneDrive.

### 3. Pornirea

Dă **dublu click pe C:\postura\porneste.bat**.

- **La prima pornire**, fereastra neagră instalează Python și bibliotecile (câteva sute de MB). Poate dura câteva
  minute; nu o închide.
- Apoi se deschide browserul cu aplicația, la http://localhost:8501.
- Dacă Windows spune „Windows protected your PC”, apasă **More info** (Mai multe informații) → **Run anyway**
  (Execută oricum).
- Dacă apare o fereastră Windows Defender Firewall pentru Python, poți apăsa **Cancel** (Anulare): aplicația se vede
  doar pe acest calculator.

Oprirea: închide fereastra neagră sau apasă în ea **Ctrl+C**.

## Instalare pe Linux și macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv, o singură dată
cd postura                                            # folderul aplicației
uv sync --frozen --no-dev
uv run --no-dev streamlit run app/app.py
```

Pe Mac cu procesor Intel aplicația nu e suportată.

## Folosirea

1. În pagina „Analiză”, la **Încarc un clip**, alege fișierul video (mp4, mov, avi sau m4v), filmat din profil.
2. Alege alura și intervalul în care călărețul se vede din profil.
3. Lasă **Metoda** pe „Automată” sau alege una manual. Metoda automată numără caii călăriți în primele 2 secunde:
   cu 2 sau mai mulți, folosește urmărirea țintei; altfel, metoda veche (un călăreț).
4. Cu mai mulți călăreți în cadru, bifează **Aleg călăreții dintr-un cadru**, alege un cadru și dă click pe calul
   călărețului. Click-urile se țin doar în sesiunea paginii.
5. Apasă **Analizează**. Dacă ținta se pierde, pagina oferă un buton care deschide primul cadru pierdut, pentru încă un
   click pe același călăreț.
6. Opțional, bifează **Scheletul calului** (oprită implicit): cele 17 puncte ale calului țintă se desenează în film
   (doar pe metoda veche) și se adaugă în JSON (`horse_keypoints`, schema 1.6); unghiurile și raportul nu se schimbă,
   iar prima analiză a unui clip durează mai mult.

## Modelele

Modelele se descarcă singure la prima analiză, deci prima analiză are nevoie de internet: MediaPipe Pose, detectorul
de cai EfficientDet, RTMPose-m și, pentru urmărire, YOLOX-S; la nevoie, RTMPose-l și AP-10K. Se pun în `models/`
(sau în `~/modele-ecvestru/`; altă cale se dă cu variabila `ECVESTRU_MODELE`). Sumele sha256 ale modelelor ONNX sunt
fixate în cod. După descărcare, analizele merg și fără internet.

## Structura

- `porneste.bat`: pornirea pe Windows (instalează mediul la prima rulare, apoi pornește aplicația).
- `app/app.py`: aplicația Streamlit (o singură pagină).
- `app/pages/analiza.py`: pagina „Analiză”. Doar afișează; logica e în `ecvestru/`.
- `research/candidate.py`: metoda veche. Găsește călărețul cu MediaPipe Pose (persoana cu șoldul pe caseta calului) și
  dă punctele umăr/șold/călcâi pe fiecare cadru.
- `ecvestru/`, logica, fără Streamlit:
  - `masurare.py`: orchestratorul analizei cu metoda veche (etape, cache, intervalul implicit);
  - `stocare.py`: cache-ul etapelor (căi, chei, citirea cadrelor, încărcarea candidatului);
  - `schelet.py`: scheletul MediaPipe pe cadru (33 de puncte, partea folosită, vizibilitatea);
  - `rtmpose.py`: călcâiul RTMPose-m, fuziunea cu MediaPipe și orientarea după direcția labei;
  - `rtmpose_l.py`: varianta RTMPose-l, folosită doar de stratul „călcâi stabil”;
  - `calcai_stabil.py`: stratul opțional „călcâi stabil”;
  - `onnx_comun.py`: pașii comuni ai modelelor ONNX (descărcare, verificare sha256, preprocesare, decodare);
  - `modele.py`: unde se caută modelele;
  - `calitate.py`: indicatorii de calitate pe cadru și poarta care scoate piciorul pe cadrele suspecte;
  - `rezumat.py`: unghiurile, motivele de abținere și rezumatele (mediane, acoperire);
  - `vedere.py`: regula „cadrul e din profil” (lățimea umerilor față de trunchi);
  - `rezultat.py`: modelul de date al rezultatului (Pydantic), același cu JSON-ul;
  - `raport.py`: raportul text din șabloane și verificarea numerelor din text;
  - `suprapunere.py`: filmul cu scheletul desenat și seriile pentru grafic;
  - `progres.py`: etapele analizei și bara de progres;
  - `click_tinta.py`: alegerea călărețului prin click (conversia click → pixeli în cadrul original);
  - `analiza_urmarita.py`: metoda cu urmărire și alegerea automată a metodei;
  - `urmarire_multi.py`: detectorul YOLOX-S, urmăritorul ByteTrack și regula „om pe cal”;
  - `aspect_urmarire.py`: recuperarea calului țintă după aspect (culoare) și schimbul de ID;
  - `tinta.py`: urmărirea calului țintă pornind de la un click;
  - `camera.py`: mișcarea camerei între cadre;
  - `directie_cal.py`: direcția calului din punctele AP-10K;
  - `cal_puncte.py`: punctele calului (AP-10K);
  - `schelet_cal.py`: opțiunea „Scheletul calului” (punctele calului țintă în film și în JSON);
  - `calareti.py`: identitatea călărețului-țintă pe clipurile dintr-un set local;
  - `inventar.py`: citirea cadrelor și extragerea posturilor MediaPipe;
  - `verificare.py`: unghiurile pe cadru din punctele MediaPipe, segmentele din profil, desenul scheletului;
  - `constante_candidat.py`: constantele citite din candidat;
  - `tabel_csv.py`: scrierea comună a CSV-urilor;
  - `router_vedere.py`, `router_pilot.py`: routerul de vedere (profil / față / spate / oblic / nu se vede);
  - `analiza_vederi.py`: modulul față/spate, lipit de măsurarea din profil;
  - `intervale.py`, `evaluare_router.py`, `clipuri_noi.py`: intervalele de vedere, evaluarea routerului și citirea
    clipurilor pentru router.
- `tests/`: testele (pytest).
- `.streamlit/config.toml`: fără statistici de folosire; aplicația ascultă doar pe localhost.

## Lanțul de prelucrare

1. **Detecție:** MediaPipe Pose pe fiecare cadru (metoda veche) sau YOLOX-S pentru oameni și cai (urmărirea).
2. **Urmărire:** metoda veche ia pe fiecare cadru omul cu șoldul pe caseta calului. Urmărirea ține calul țintă cu
   ByteTrack (cu recuperare după aspect) și ia călărețul de pe el.
3. **Schelet:** punctele MediaPipe și, pe calea implicită, călcâiul RTMPose-m (cu fuziune și interval plauzibil); la
   urmărire, scheletul RTMPose-m al țintei.
4. **Unghiuri:** trunchiul (umăr–șold) și piciorul (șold–călcâi) față de verticală, cu semnul dat de sensul de mers.
5. **Abținere:** un cadru rămâne nemăsurat, cu motivul scris: călăreț negăsit, nu e pe cal, trunchi peste 45°, puncte
   invizibile, călcâi pe piciorul calului, unghi neplauzibil, cadru care nu e din profil.
6. **Raport:** medianele și acoperirea, raportul text din șabloane, filmul și JSON-ul (`AnalysisResult`).

Rezultatele și cache-ul stau în `data/results/` (sau în folderul din variabila `ECVESTRU_RESULTS`).

## Testele

```bash
uv sync --frozen            # cu pytest și ruff
uv run pytest -q
uv run ruff check .
```

Testele care cer clipuri locale (`data/videos/` și `data/split.csv`) se sar sau se ignoră când lipsesc.

## Probleme cunoscute

1. **Alegerea călărețului corect cu mai mulți călăreți în cadru.** Metoda veche ia omul cu șoldul pe caseta unui cal
   și poate sări pe alt călăreț. Urmărirea are nevoie de un click pe calul țintă și poate pierde ținta când călăreții
   se apropie.
2. **Viteza urmăririi pe CPU:** aproximativ 1 s pe cadru pe un laptop fără placă video; un clip cu doi călăreți poate
   dura 25–40 de minute. După aceea, totul vine din cache.
3. **Routerul de vedere nu e activat în analiză.** Apare doar ca opțiune de afișare („Router vedere (pilot)”,
   debifată implicit) și nu schimbă valorile sau raportul.

## Licențe

Aplicația e folosită **necomercial, pentru cercetare**. Componentele externe:

| Componentă | Licență | Cum ajunge pe calculator |
|---|---|---|
| Modelul `pose_landmarker_full.task` (MediaPipe BlazePose GHUM 3D, Google) | Apache-2.0 | descărcat la prima analiză |
| Biblioteca MediaPipe | Apache-2.0 | bibliotecă Python, de pe PyPI |
| Modelul `efficientdet_lite0.tflite` (detectorul de cai, Google, antrenat pe COCO) | neconfirmată oficial de Google | descărcat la prima analiză |
| Modelul RTMPose-m Halpe26 (MMPose) | cod și greutăți Apache-2.0; antrenat pe Body7 (AI Challenger, MS COCO, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18), seturi dintre care unele sunt doar pentru cercetare sau uz necomercial | descărcat la prima analiză, cu suma sha256 verificată |
| Modelul RTMPose-l Halpe26 (doar cu stratul „călcâi stabil”) | ca RTMPose-m | descărcat la prima folosire, cu suma sha256 verificată |
| Modelul RTMPose-m AP-10K (punctele calului) | cod și greutăți Apache-2.0; setul AP-10K CC-BY-4.0 (Yu et al., NeurIPS 2021); preantrenat pe AI Challenger și MS COCO | descărcat la prima folosire, cu suma sha256 verificată |
| Modelul YOLOX-S (Megvii), detectorul de oameni și cai al urmăririi | Apache-2.0 | descărcat la prima folosire, cu suma sha256 verificată |
| ByteTrack (urmăritorul, portat în `ecvestru/urmarire_multi.py`) | MIT (Yifu Zhang) | cod în aplicație |
| `onnxruntime` | MIT | bibliotecă Python, de pe PyPI |
| `scipy` | BSD-3-Clause | bibliotecă Python, de pe PyPI |
| `imageio-ffmpeg` | BSD-2-Clause | bibliotecă Python, de pe PyPI |
| ffmpeg (din `imageio-ffmpeg`, altfel din PATH), pentru filmul cu scheletul | GPLv3 | rulează doar ca proces separat; aplicația nu îl include și nu se leagă de el. Fără ffmpeg, analiza merge, fără film. |
| Celelalte biblioteci Python (Streamlit, OpenCV, NumPy, pandas, Pydantic etc.) | licența fiecăreia | de pe PyPI |
