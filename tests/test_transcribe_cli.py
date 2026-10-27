#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pytest==9.1.1",
#     "click==8.3.3",
#     "numpy==2.4.6; python_version < '3.12'",
#     "numpy==2.5.4; python_version >= '3.12'",
#     "scipy==1.17.1; python_version < '3.12'",
#     "scipy==1.18.1; python_version >= '3.12'",
#     "soundfile==0.14.0",
# ]
#
# [tool.uv]
# exclude-newer = "2026-10-27T00:00:00Z"
# ///
"""Pure helpers and source adapters of transcribe-cli, with the external CLIs stubbed.

Run: uv run --script tests/test_transcribe_cli.py
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import transcribe_cli as tc


def wav_bytes(samples: list[int], extensible: bool = False) -> bytes:
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else 1, 1, 16000, 32000, 2, 16)
    if extensible:
        fmt += struct.pack("<HHI", 22, 16, 4) + b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    data = struct.pack(f"<{len(samples)}h", *samples)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def completed(stdout: str, code: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=stdout, stderr="")


def test_pcm_scales_to_unit_range():
    assert tc.pcm_from_wav(wav_bytes([0, 16384, -32768])).tolist() == pytest.approx([0.0, 0.5, -1.0])


def test_pcm_reads_extensible_wav():
    assert len(tc.pcm_from_wav(wav_bytes([1, 2, 3], extensible=True))) == 3


def test_mono_16k_downmixes_and_resamples():
    assert tc.to_mono_16k(np.zeros((48000, 2), dtype=np.float32), 48000).shape == (16000,)


def test_coredata_epoch():
    assert tc.coredata_to_datetime(0) == dt.datetime(2001, 1, 1, tzinfo=dt.timezone.utc)


def test_voicememos_rows_become_items(tmp_path):
    db = tmp_path / "CloudRecordings.db"
    import sqlite3
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE ZCLOUDRECORDING (ZUNIQUEID, ZENCRYPTEDTITLE, ZCUSTOMLABEL, ZDATE, ZDURATION, ZPATH)")
    con.execute("INSERT INTO ZCLOUDRECORDING VALUES ('UID-1', 'Mémo', NULL, 810113737.9, 26.1, 'memo.m4a')")
    con.commit(); con.close()
    (tmp_path / "memo.m4a").write_bytes(b"x")
    items = tc.voicememos_items(tmp_path)
    assert items[0]["id"] == "UID-1" and items[0]["available"] is True


def test_voicememos_since_filters_old_rows(tmp_path):
    import sqlite3
    con = sqlite3.connect(tmp_path / "CloudRecordings.db")
    con.execute("CREATE TABLE ZCLOUDRECORDING (ZUNIQUEID, ZENCRYPTEDTITLE, ZCUSTOMLABEL, ZDATE, ZDURATION, ZPATH)")
    con.execute("INSERT INTO ZCLOUDRECORDING VALUES ('OLD', 'x', NULL, 0.0, 1.0, 'old.m4a')")
    con.commit(); con.close()
    assert tc.voicememos_items(tmp_path, since=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)) == []


WA_READ = {"chat": "c", "name": "", "messages": [
    {"ts": 10, "msg_id": "V", "text": "[audio]", "quoted_id": "", "from_me": True, "media": {"kind": "audio", "seconds": 3}},
    {"ts": 11, "msg_id": "D", "text": "[document]", "quoted_id": "", "from_me": True, "media": {"kind": "document", "mimetype": "audio/ogg; codecs=opus", "file_name": "WhatsApp Audio.opus"}},
    {"ts": 12, "msg_id": "P", "text": "[document]", "quoted_id": "", "from_me": True, "media": {"kind": "document", "mimetype": "application/pdf"}},
    {"ts": 13, "msg_id": "T", "text": "hello", "quoted_id": "", "from_me": True, "media": None},
]}


def test_whatsapp_items_keep_voice_notes_and_audio_documents(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: completed(json.dumps(WA_READ)))
    assert [it["id"] for it in tc.whatsapp_items("c", 80, None)] == ["V", "D"]


def test_whatsapp_items_since_uses_message_ts(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: completed(json.dumps(WA_READ)))
    assert [it["id"] for it in tc.whatsapp_items("c", 80, dt.datetime.fromtimestamp(11, dt.timezone.utc))] == ["D"]


def test_whatsapp_items_fail_loud_when_wa_fails(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: completed("", 1))
    import click
    with pytest.raises(click.ClickException):
        tc.whatsapp_items("c", 80, None)


def test_telegram_items_keep_audio_only(monkeypatch, tmp_path):
    saved = [{"ts": "2026-09-03T10:00:00", "id": 175270, "reply_to": None, "file": str(tmp_path / "audio_1.ogg"), "caption": ""}, {"ts": "2026-09-03T10:01:00", "id": 175271, "file": str(tmp_path / "photo.jpg"), "caption": ""}]
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: completed(json.dumps(saved)))
    assert [it["id"] for it in tc.telegram_items("me", 10, None, tmp_path)] == ["175270"]


def test_telegram_items_fall_back_to_file_name_without_id(monkeypatch, tmp_path):
    saved = [{"ts": "2026-09-03T10:00:00", "file": str(tmp_path / "audio_1.ogg"), "caption": ""}]
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: completed(json.dumps(saved)))
    assert tc.telegram_items("me", 10, None, tmp_path)[0]["id"] == "audio_1.ogg"


def test_auto_backend_is_whisper_on_apple_silicon():
    assert tc.resolve_backend("auto", apple_silicon=True) == "whisper"


def test_auto_backend_is_parakeet_on_cpu_box():
    assert tc.resolve_backend(None, apple_silicon=False) == "parakeet"


def test_explicit_backend_is_kept():
    assert tc.resolve_backend("parakeet", apple_silicon=True) == "parakeet"


def test_unknown_backend_fails_loud():
    import click
    with pytest.raises(click.ClickException):
        tc.resolve_backend("vosk")


def test_default_parakeet_model_per_platform():
    assert (tc.default_model("parakeet", True), tc.default_model("parakeet", False)) == (tc.PARAKEET_MLX_MODEL, tc.PARAKEET_ONNX_MODEL)


def test_default_whisper_model_per_platform():
    assert (tc.default_model("whisper", True), tc.default_model("whisper", False)) == (tc.MLX_MODEL, tc.CT2_MODEL)


def test_glossary_inline_or_file(tmp_path):
    f = tmp_path / "g.txt"
    f.write_text("Zama, Slack\n")
    assert tc.load_glossary(str(f)) == "Zama, Slack" and tc.load_glossary("a, b") == "a, b"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
