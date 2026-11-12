"""Căutarea modelelor mari: ECVESTRU_MODELE, apoi ~/modele-ecvestru/, apoi models/; mesaj clar dacă lipsesc."""
from pathlib import Path

import pytest

from ecvestru import cal_puncte as CP
from ecvestru import modele as MO
from ecvestru import rtmpose_l as RL


def put(folder: Path, subdir: str, filename: str = "end2end.onnx") -> Path:
    path = folder / subdir / "20230831" / "pachet" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"onnx")
    return path


@pytest.fixture
def places(tmp_path, monkeypatch):
    home, env, old = tmp_path / "acasa", tmp_path / "env", tmp_path / "proiect" / "models" / "ap10k"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv(MO.ENV_VAR, raising=False)
    return home, env, old


def test_order_is_env_then_home_then_old(places, monkeypatch):
    home, env, old = places
    monkeypatch.setenv(MO.ENV_VAR, str(env))
    assert MO.search_dirs("ap10k", old) == [env / "ap10k", home / MO.HOME_FOLDER / "ap10k", old]


def test_without_env_the_home_folder_comes_first(places):
    home, _, old = places
    assert MO.search_dirs("ap10k", old) == [home / MO.HOME_FOLDER / "ap10k", old]


def test_env_wins_over_home_and_old(places, monkeypatch):
    home, env, old = places
    put(home / MO.HOME_FOLDER, "ap10k")
    put(old.parent, "ap10k")
    wanted = put(env, "ap10k")
    monkeypatch.setenv(MO.ENV_VAR, str(env))
    assert MO.find_model("ap10k", "end2end.onnx", old) == wanted


def test_home_wins_over_old(places):
    home, _, old = places
    put(old.parent, "ap10k")
    wanted = put(home / MO.HOME_FOLDER, "ap10k")
    assert MO.find_model("ap10k", "end2end.onnx", old) == wanted


def test_old_path_is_the_last_resort(places):
    _, _, old = places
    wanted = put(old.parent, "ap10k")
    assert MO.find_model("ap10k", "end2end.onnx", old) == wanted


def test_missing_everywhere_gives_a_clear_error_with_every_place(places, monkeypatch):
    home, env, old = places
    monkeypatch.setenv(MO.ENV_VAR, str(env))
    with pytest.raises(MO.ModelNotFound) as error:
        MO.locate("AP-10K", "ap10k", "end2end.onnx", old)
    text = str(error.value)
    assert "AP-10K" in text and MO.ENV_VAR in text
    assert all(str(d) in text for d in (env / "ap10k", home / MO.HOME_FOLDER / "ap10k", old))


def test_failed_download_keeps_the_places_and_the_reason(places):
    _, _, old = places

    def fail():
        raise OSError("fără rețea")

    with pytest.raises(MO.ModelNotFound, match="fără rețea") as error:
        MO.locate("AP-10K", "ap10k", "end2end.onnx", old, download=fail)
    assert str(old) in str(error.value)
    assert isinstance(error.value, OSError)   # rtmpose_problem prinde OSError: aplicația afișează mesajul


def test_download_is_only_the_fallback(places):
    _, _, old = places
    assert MO.locate("x", "ap10k", "end2end.onnx", old, download=lambda: Path("/descarcat")) == Path("/descarcat")


def test_horse_and_rtmpose_l_use_the_search(places, monkeypatch):
    home, _, _ = places
    horse = put(home / MO.HOME_FOLDER, "ap10k")
    large = put(home / MO.HOME_FOLDER, "rtmpose_l")
    monkeypatch.setattr(CP, "MODEL_DIR", Path("/nu/exista/ap10k"))
    monkeypatch.setattr(RL, "MODEL_DIR", Path("/nu/exista/rtmpose_l"))
    assert CP.model_path() == horse and CP.ensure_model() == horse
    assert RL.ensure_model() == large


def test_explicit_folder_keeps_the_old_behaviour(tmp_path):
    path = put(tmp_path, "m")
    assert CP.model_path(tmp_path) == path and RL.ensure_model(tmp_path) == path
