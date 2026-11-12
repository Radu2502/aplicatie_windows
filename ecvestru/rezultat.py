"""
Modelul de date al unei analize: același obiect se afișează în pagina „Analiză” și se salvează ca
JSON. Valorile rotunjite care apar în textul raportului sunt
câmpuri separate (`*_rounded`, `coverage_pct`), ca verificarea numerelor din text să compare exact.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.4"
COMBINED_SCHEMA_VERSION = "1.5"   # rezultatul cu vederi (profil + față/spate); calea implicită rămâne 1.4
QUALITY_VERSION = 2   # stratul de calitate: se mărește când se schimbă indicatorii sau poarta
GAITS = ("pas", "trap", "galop")
HEEL_MEDIAPIPE, HEEL_RTMPOSE = "mediapipe", "rtmpose"   # sursa călcâiului
HEEL_MODEL_M, HEEL_MODEL_L = "m", "l"   # modelul RTMPose: m (implicit) sau l (doar cu stratul „călcâi stabil”)
PIPELINE_RTMPOSE, PIPELINE_BASELINE = "rtmpose", "baseline"   # calea care a rulat (RTMPose = implicită în aplicație)
FACING_NOSE_EAR, FACING_FOOT = "nas_ureche", "laba"     # de unde vine orientarea unui cadru (laba: regula B4)

LIMITATION = ("Precizia unghiurilor în grade nu e măsurată față de o referință umană. Verificarea a "
              "arătat doar că aplicația găsește călărețul și pune punctele aproximativ corect.")
# Pentru clipurile fără verdicte, cele încărcate și țintele alese: pe ele nu s-a verificat nimic.
LIMITATION_UNVERIFIED = ("Precizia unghiurilor în grade nu e măsurată față de o referință umană. Pentru această "
                         "analiză nu există verdicte: nu știm dacă aplicația a găsit călărețul și a pus "
                         "punctele corect.")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetricSummary(_Model):
    """Rezumatul unui unghi pe cadrele măsurate din intervalul ales (grade față de verticală)."""
    n_frames: int = Field(ge=0, description="cadre măsurate în interval")
    coverage: float = Field(ge=0, le=1, description="cadre măsurate / cadre din interval")
    coverage_pct: int = Field(ge=0, le=100, description="acoperirea rotunjită, ca în text")
    median_deg: float | None = None
    q25_deg: float | None = None
    q75_deg: float | None = None
    median_rounded: int | None = Field(None, description="mediana rotunjită, cu semn")
    median_abs_rounded: int | None = Field(None, ge=0, description="mărimea medianei rotunjite, ca în text")
    q25_rounded: int | None = None
    q75_rounded: int | None = None


class FrameMeasure(_Model):
    """Un cadru: unghiurile (None = nemăsurat) și motivul, dacă un unghi lipsește."""
    frame: int = Field(ge=0)
    trunk_deg: float | None = None
    leg_deg: float | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list, description="semnale de calitate, fără excludere")
    facing: Literal[-1.0, 1.0] | None = Field(
        None, description="sensul de mers folosit la semnul unghiurilor (+1 = dreapta); doar pe calea RTMPose")
    facing_source: Literal["nas_ureche", "laba"] | None = Field(
        None, description="nas_ureche = candidatul; laba = regula B4 (votul direcției labei); doar pe calea RTMPose")


class UnmeasuredInterval(_Model):
    """Cadre consecutive nemăsurate din același motiv."""
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    frames: int = Field(ge=1)
    reason: str
    reason_text: str


class TargetSeed(_Model):
    """Călărețul ales de utilizator: un click într-un cadru, în pixeli."""
    frame: int = Field(ge=0)
    x: float
    y: float


class CameraEstimate(_Model):
    """Rotirea relativă a camerei în interval; fără număr când fundalul nu permite estimarea."""
    amplitude_deg: float | None = None
    amplitude_rounded: float | None = None
    frames_estimated: int = Field(ge=0)
    frames_total: int = Field(ge=0)


class QualityWarning(_Model):
    """Un semnal de calitate pe intervalul analizat: doar avertisment, cadrele rămân măsurate."""
    code: str
    text: str
    frames: int = Field(ge=0)


class QualitySummary(_Model):
    """Stratul de calitate al analizei. None în rezultat = stratul n-a rulat (fără schelet complet)."""
    version: int
    excluded_frames: int = Field(ge=0, description=(
        "cadre cu piciorul exclus pentru călcâi pe piciorul calului: de indicatorul A pe linia de bază, de fuziunea "
        "RTMPose pe calea implicită; fără excluderile stratului „călcâi stabil” (AP-10K), care au alt motiv"))
    excluded_reason: str
    implausible_frames: int = Field(0, ge=0, description="cadre cu piciorul exclus pentru unghi neplauzibil")
    warnings: list[QualityWarning] = Field(default_factory=list)


class OrientationSummary(_Model):
    """Orientarea pe calea RTMPose, în intervalul analizat (regula B4). None pe calea implicită."""
    rule: str = Field("B4: vot pe direcția labei, 9 cadre în ±0,5 s, cel puțin 5 voturi și 2/3 acord",
                      description="regula aplicată, doar în intervalul analizat")
    changed_frames: int = Field(ge=0, description="cadre cu orientarea schimbată față de nas–ureche")
    foot_frames: int = Field(ge=0, description="cadre pe care votul labei a decis orientarea")
    frames_with_facing: int = Field(ge=0, description="cadre cu orientare în interval")
    minority_share: float = Field(ge=0, le=0.5, description="ponderea orientării minoritare după regulă")
    minority_pct: int = Field(ge=0, le=50, description="ponderea minoritară rotunjită, ca în text")
    profile_unclear: bool = Field(description="minoritara > 5%: raportul text se ascunde")
    reason: str | None = Field(None, description="motivul ascunderii raportului")


class LayerVersions(_Model):
    """Versiunile straturilor care au produs rezultatul; None = stratul n-a rulat."""
    quality: int | None = Field(None, description="stratul de calitate (poarta A, intervalul, avertismentele)")
    rtmpose: int | None = Field(None, description="stratul RTMPose (călcâiul, fuziunea)")
    orientation: int | None = Field(None, description="regula B4 și „profil neclar”")
    rtmpose_model: str | None = Field(None, description="primele 8 caractere din sha256-ul pachetului modelului")
    heel_stable: int | None = Field(None, description="stratul opțional „călcâi stabil”; lipsește din JSON, oprit")


class AnalysisResult(_Model):
    schema_version: Literal["1.4"] = SCHEMA_VERSION
    video: str
    video_hash: str
    gait: Literal["pas", "trap", "galop"]
    fps: float = Field(gt=0)
    n_frames: int = Field(ge=1)
    segment: tuple[int, int] = Field(description="intervalul din profil (cadre, inclusiv capetele)")
    candidate: str = Field(description="varianta de calcul, candidate@<hash>")
    trunk: MetricSummary
    leg: MetricSummary
    unmeasured: list[UnmeasuredInterval]
    frames: list[FrameMeasure]
    limitation: str = LIMITATION
    target: TargetSeed | None = None
    camera: CameraEstimate | None = None
    anchors: list[TargetSeed] = Field(default_factory=list, description="click-uri în plus, după pierderea țintei")
    quality: QualitySummary | None = None
    heel_source: Literal["mediapipe", "rtmpose"] = Field(HEEL_MEDIAPIPE, description="de unde vine călcâiul")
    pipeline: Literal["rtmpose", "baseline"] = Field(
        PIPELINE_BASELINE, description="calea care a rulat: rtmpose (implicită în aplicație) sau baseline (MediaPipe)")
    fallback_reason: str | None = Field(
        None, description="de ce a rulat linia de bază deși s-a cerut RTMPose; None = a rulat calea cerută")
    layers: LayerVersions | None = Field(None, description="versiunile straturilor")
    orientation: OrientationSummary | None = Field(None, description="orientarea după labă; doar pe calea RTMPose")

    def to_json(self) -> str:
        """Câmpurile goale lipsesc; fără bloc `orientation` (linia de bază) lipsesc și `facing*` de pe cadre, deci
        JSON-ul liniei de bază are câmpurile din schema 1.2, plus `pipeline`, `fallback_reason` și `layers`."""
        exclude: dict = {name: True for name in ("target", "camera", "anchors", "orientation")
                         if not getattr(self, name)}
        if self.orientation is None:
            exclude["frames"] = {"__all__": {"facing", "facing_source"}}
        if self.layers is not None and self.layers.heel_stable is None:
            exclude["layers"] = {"heel_stable"}         # cu stratul oprit, câmpul lipsește din JSON
        return self.model_dump_json(indent=1, exclude=exclude or None)

    @classmethod
    def from_json(cls, text: str) -> AnalysisResult:
        return cls.model_validate_json(text)


# ---------- rezultatul cu vederi (schema 1.5): profilul de mai sus, plus modulul față/spate ----------

class ViewSpan(_Model):
    """Cadre consecutive cu aceeași vedere, în intervalul analizat. `view` = "" (fără vedere) sau nedecis."""
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    view: str


class FrontBackFrame(_Model):
    """Un cadru din față sau din spate: înclinarea liniei umerilor și a bazinului față de orizontala imaginii, ca
    dreaptă (modulo 180°, deci o inversare stânga–dreapta nu o schimbă); + = partea din dreapta imaginii mai jos."""
    frame: int = Field(ge=0)
    view: Literal["fata", "spate"]
    shoulder_tilt_deg: float | None = None
    pelvis_tilt_deg: float | None = None


class FrontBackSummary(_Model):
    """Modulul față/spate pe cadrele din față sau din spate ale intervalului; valorile sunt nevalidate."""
    frames_front: int = Field(ge=0)
    frames_back: int = Field(ge=0)
    shoulders: MetricSummary
    pelvis: MetricSummary
    frames: list[FrontBackFrame]


class CombinedResult(_Model):
    """O analiză cu vederi: partea de profil e exact AnalysisResult (1.4), restrânsă la cadrele din profil; partea
    față/spate lipsește când intervalul n-are cadre din față sau din spate."""
    schema_version: Literal["1.5"] = COMBINED_SCHEMA_VERSION
    view_source: Literal["router", "intervale"]
    views: list[ViewSpan]
    frames_by_view: dict[str, int]
    profile: AnalysisResult
    front_back: FrontBackSummary | None = None

    def to_json(self) -> str:
        data = json.loads(self.model_dump_json(exclude={"profile"}, exclude_none=False))
        data["profile"] = json.loads(self.profile.to_json())
        if self.front_back is None:
            data.pop("front_back")
        return json.dumps(data, indent=1, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> CombinedResult:
        return cls.model_validate_json(text)
