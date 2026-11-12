"""Adăugarea unui rând în CSV (ecvestru/tabel_csv.py), un singur ajutor pentru toate fișierele doar adăugate: antetul
se scrie dacă fișierul lipsește SAU e gol; octeții sunt cei de până acum (csv.DictWriter, UTF-8, CRLF)."""
import csv
import io
from pathlib import Path

import pytest

from ecvestru import intervale as IV
from ecvestru import tabel_csv as TC

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["a", "b"]


def old_bytes(rows: list[dict], header: bool = True) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    if header:
        writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def test_a_new_file_gets_the_header_then_the_row(tmp_path):
    path = tmp_path / "sub" / "x.csv"
    TC.append_row(path, FIELDS, {"a": "ă", "b": 1})
    assert path.read_bytes() == old_bytes([{"a": "ă", "b": 1}]) == "a,b\r\nă,1\r\n".encode()


def test_an_empty_file_gets_the_header_too(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes(b"")
    TC.append_row(path, FIELDS, {"a": 1, "b": 2})
    assert path.read_bytes() == b"a,b\r\n1,2\r\n"


def test_an_existing_file_is_only_appended(tmp_path):
    path = tmp_path / "x.csv"
    TC.append_row(path, FIELDS, {"a": 1, "b": 2})
    before = path.read_bytes()
    TC.append_row(path, FIELDS, {"a": "x,y", "b": ""})
    assert path.read_bytes() == before + b'"x,y",\r\n'


def test_extra_keys_are_refused_unless_ignored(tmp_path):
    path = tmp_path / "x.csv"
    with pytest.raises(ValueError):
        TC.append_row(path, FIELDS, {"a": 1, "b": 2, "c": 3})
    TC.append_row(path, FIELDS, {"a": 1, "b": 2, "c": 3}, extrasaction="ignore")
    assert path.read_bytes().endswith(b"1,2\r\n")


def test_the_views_file_keeps_its_bytes(tmp_path):
    path = tmp_path / IV.ANSWER_FILE
    IV.save_interval(path, "a.avi", 0, 9, IV.PROFILE, "adnotator1")
    lines = path.read_bytes().split(b"\r\n")
    assert lines[0] == ",".join(IV.FIELDS).encode() and lines[-1] == b""


