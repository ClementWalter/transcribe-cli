#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click>=8.1",
#     "numpy",
#     "soundfile",
#     "scipy",
#     "mlx-whisper>=0.4; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "parakeet-mlx>=0.5; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "faster-whisper>=1.1; sys_platform != 'darwin' or platform_machine != 'arm64'",
#     "onnx-asr[cpu,hub]>=0.12; sys_platform != 'darwin' or platform_machine != 'arm64'",
# ]
# ///
"""Speech to text for the audio the user records or receives: one CLI, several sources.

`transcribe file <audio>` runs Whisper locally on any audio file. The source
subcommands know where voice recordings live and hand them to the same
pipeline, so a heartbeat or an agent never deals with formats or stores:

  voicememos   Apple Voice Memos on this Mac (CloudRecordings.db + iCloud .m4a)
  whatsapp     voice notes and audio files of a WhatsApp chat, via the `wa` CLI
  telegram     voice messages and audio files of a Telegram chat, via the `tg` CLI

Each source lists items as JSON ({id, source, ts, duration, title, path}) and
`--transcribe` adds {text, language} to every item, fetching the audio first
when needed. Ids are stable per source (Voice Memos uuid, WhatsApp msg_id,
Telegram file name) so a caller can keep its own "seen" state.

Two speech models, `--backend whisper|parakeet|auto`:

  whisper    Whisper large-v3-turbo. Honours --glossary (domain words as the
             initial prompt), fuller transcripts. mlx-whisper on Apple Silicon,
             faster-whisper (CTranslate2, CPU int8) elsewhere.
  parakeet   NVIDIA parakeet-tdt-0.6b-v3, 25 European languages. Several times
             faster, better punctuation, no glossary. parakeet-mlx on Apple
             Silicon, onnx-asr (ONNX Runtime, int8) elsewhere.
  auto       whisper on Apple Silicon where both take seconds, parakeet on a
             CPU-only box where Whisper is several times slower than real time.

Decoding: CoreAudio's afconvert on macOS, libsndfile (soundfile) for Ogg/Opus
voice notes and on Linux. A glossary is a file or an inline comma-separated
list; set TRANSCRIBE_GLOSSARY or pass --glossary.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pathlib
import platform
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
from math import gcd

import click
import numpy as np

SAMPLE_RATE = 16_000
IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
CT2_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
PARAKEET_MLX_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
PARAKEET_ONNX_MODEL = "nemo-parakeet-tdt-0.6b-v3"
BACKENDS = ("auto", "whisper", "parakeet")
AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".ogg", ".oga", ".opus", ".flac", ".aac", ".mp4", ".caf", ".aiff"}
LIBSNDFILE_SUFFIXES = {".ogg", ".oga", ".opus", ".flac", ".wav", ".aiff", ".mp3"}

HOME = pathlib.Path.home()
VOICEMEMOS_DIR = HOME / "Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings"
COREDATA_EPOCH = dt.datetime(2001, 1, 1, tzinfo=dt.timezone.utc)

log = logging.getLogger("transcribe")


# --- decoding ---------------------------------------------------------------

def pcm_from_wav(data: bytes) -> np.ndarray:
    """Mono float32 samples from 16-bit PCM WAV bytes, walking RIFF chunks (afconvert writes WAVE_FORMAT_EXTENSIBLE)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        (size,) = struct.unpack("<I", data[offset + 4 : offset + 8])
        body = offset + 8
        if chunk_id == b"data":
            return np.frombuffer(data[body : body + size], dtype="<i2").astype(np.float32) / 32768.0
        offset = body + size + (size & 1)
    raise ValueError("no data chunk in WAV")


def to_mono_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import resample_poly

    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    if rate != SAMPLE_RATE:
        g = gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, rate // g)
    return mono.astype(np.float32)


def decode_with_afconvert(path: pathlib.Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        wav = pathlib.Path(tmp) / "audio.wav"
        subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", str(path), str(wav)], check=True, capture_output=True)
        return pcm_from_wav(wav.read_bytes())


def decode_with_soundfile(path: pathlib.Path) -> np.ndarray:
    import soundfile as sf

    samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_16k(samples, rate)


def decode_audio(path: pathlib.Path) -> np.ndarray:
    """16 kHz mono float32. afconvert for CoreAudio formats on macOS, libsndfile otherwise (it reads Ogg/Opus, not m4a)."""
    if sys.platform == "darwin" and path.suffix.lower() not in LIBSNDFILE_SUFFIXES:
        return decode_with_afconvert(path)
    if path.suffix.lower() in LIBSNDFILE_SUFFIXES or sys.platform != "darwin":
        try:
            return decode_with_soundfile(path)
        except Exception as exc:  # m4a on Linux: fall back to ffmpeg when present
            if shutil.which("ffmpeg") is None:
                raise RuntimeError(f"cannot decode {path.name}: {exc}; install ffmpeg for this format") from exc
            with tempfile.TemporaryDirectory() as tmp:
                wav = pathlib.Path(tmp) / "audio.wav"
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", str(wav)], check=True)
                return decode_with_soundfile(wav)
    return decode_with_afconvert(path)


# --- transcription -----------------------------------------------------------

def resolve_backend(option: str | None, apple_silicon: bool = IS_APPLE_SILICON) -> str:
    """`auto` picks the model that is fast enough on this hardware."""
    if option in (None, "auto"):
        return "whisper" if apple_silicon else "parakeet"
    if option not in BACKENDS:
        raise click.ClickException(f"unknown backend {option!r}; choose from {', '.join(BACKENDS)}")
    return option


def default_model(backend: str, apple_silicon: bool = IS_APPLE_SILICON) -> str:
    if backend == "parakeet":
        return PARAKEET_MLX_MODEL if apple_silicon else PARAKEET_ONNX_MODEL
    return MLX_MODEL if apple_silicon else CT2_MODEL


def transcribe_whisper(audio: np.ndarray, model: str, glossary: str | None) -> dict:
    if IS_APPLE_SILICON and not model.startswith("deepdml/"):
        import mlx_whisper

        options = {"initial_prompt": glossary} if glossary else {}
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=model, **options)
        segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result.get("segments", [])]
        return {"text": result["text"].strip(), "language": result.get("language"), "segments": segments, "backend": "mlx-whisper", "model": model}
    from faster_whisper import WhisperModel

    engine = WhisperModel(model, device="cpu", compute_type="int8")
    raw_segments, info = engine.transcribe(audio, initial_prompt=glossary, vad_filter=True)
    segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in raw_segments]
    return {"text": " ".join(s["text"] for s in segments).strip(), "language": info.language, "segments": segments, "backend": "faster-whisper", "model": model}


def transcribe_parakeet(audio: np.ndarray, model: str) -> dict:
    """Parakeet TDT v3: no prompt, no language id in the output; sentence timings on MLX only."""
    if IS_APPLE_SILICON and not model.startswith("nemo-"):
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        engine = from_pretrained(model)
        # float32 in: the mel front end mis-shapes bfloat16 input.
        result = engine.generate(get_logmel(mx.array(audio), engine.preprocessor_config))[0]
        segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in getattr(result, "sentences", [])]
        return {"text": result.text.strip(), "language": None, "segments": segments, "backend": "parakeet-mlx", "model": model}
    import onnx_asr

    engine = onnx_asr.load_model(model, quantization="int8", providers=["CPUExecutionProvider"])
    text = engine.recognize(audio.astype(np.float32), sample_rate=SAMPLE_RATE)
    return {"text": str(text).strip(), "language": None, "segments": [], "backend": "onnx-asr", "model": model}


def transcribe_audio(audio: np.ndarray, backend: str, model: str | None, glossary: str | None) -> dict:
    """16 kHz mono samples in, the same result shape out whatever the model."""
    model = model or default_model(backend)
    if backend == "parakeet":
        if glossary:
            log.warning("parakeet has no prompt: the glossary is ignored")
        return transcribe_parakeet(audio, model)
    return transcribe_whisper(audio, model, glossary)


def transcribe_file(path: pathlib.Path, model: str | None = None, glossary: str | None = None, backend: str | None = None) -> dict:
    audio = decode_audio(path)
    result = transcribe_audio(audio, resolve_backend(backend), model, glossary)
    result["duration"] = round(len(audio) / SAMPLE_RATE, 1)
    result["path"] = str(path)
    return result


def load_glossary(option: str | None) -> str | None:
    source = option or os.environ.get("TRANSCRIBE_GLOSSARY")
    if not source:
        return None
    path = pathlib.Path(source).expanduser()
    return path.read_text().strip() if path.is_file() else source


# --- sources -----------------------------------------------------------------

def coredata_to_datetime(seconds: float) -> dt.datetime:
    return COREDATA_EPOCH + dt.timedelta(seconds=seconds)


def voicememos_items(recordings_dir: pathlib.Path = VOICEMEMOS_DIR, since: dt.datetime | None = None) -> list[dict]:
    """Apple Voice Memos, from a snapshot copy of CloudRecordings.db so the app's writes are never disturbed."""
    db = recordings_dir / "CloudRecordings.db"
    if not db.is_file():
        raise click.ClickException(f"Voice Memos database not found at {db} (macOS only; needs Full Disk Access)")
    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("", "-wal", "-shm"):
            src = db.with_name(db.name + suffix)
            if src.exists():
                shutil.copy2(src, pathlib.Path(tmp) / src.name)
        con = sqlite3.connect(pathlib.Path(tmp) / db.name)
        try:
            rows = con.execute(
                "SELECT ZUNIQUEID, ZENCRYPTEDTITLE, ZCUSTOMLABEL, ZDATE, ZDURATION, ZPATH FROM ZCLOUDRECORDING WHERE ZPATH IS NOT NULL ORDER BY ZDATE"
            ).fetchall()
        finally:
            con.close()
    items = []
    for uid, title, label, date, duration, rel in rows:
        when = coredata_to_datetime(date)
        if since and when < since:
            continue
        path = recordings_dir / rel
        items.append({
            "id": uid, "source": "voicememos", "ts": int(when.timestamp()), "duration": round(duration or 0, 1),
            "title": title or label or uid, "path": str(path), "available": path.is_file(),
        })
    return items


def cli_path(name: str) -> str:
    return shutil.which(name) or str(HOME / ".local" / "bin" / name)


def whatsapp_items(chat: str, limit: int, since: dt.datetime | None) -> list[dict]:
    """Audio messages of a WhatsApp chat: voice-note bubbles and audio files shared as documents."""
    result = subprocess.run([cli_path("wa"), "read", chat, "--limit", str(limit), "--json", "--no-extend"], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise click.ClickException(f"wa read failed: {result.stderr.strip()[-300:]}")
    items = []
    for m in json.loads(result.stdout)["messages"]:
        media = m.get("media") or {}
        is_audio = media.get("kind") == "audio" or (media.get("kind") == "document" and (media.get("mimetype") or "").startswith("audio/"))
        if not is_audio or (since and m["ts"] < since.timestamp()):
            continue
        items.append({
            "id": m["msg_id"], "source": "whatsapp", "chat": chat, "ts": m["ts"], "duration": float(media.get("seconds") or 0),
            "title": media.get("file_name") or "Vocal WhatsApp", "path": None, "quoted_id": m.get("quoted_id", ""), "from_me": m.get("from_me"),
        })
    return items


def whatsapp_fetch(item: dict, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([cli_path("wa"), "download", item["id"], "-o", str(out_dir)], capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise click.ClickException(f"wa download {item['id']} failed: {result.stderr.strip()[-300:]}")
    return pathlib.Path(result.stdout.strip().splitlines()[-1])


def telegram_items(chat: str, limit: int, since: dt.datetime | None, out_dir: pathlib.Path) -> list[dict]:
    """Audio attachments of a Telegram chat. `tg download` saves them; the file name is the stable id."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [cli_path("tg"), "download", chat, "--limit", str(limit), "--out", str(out_dir), "--json"]
    if since:
        cmd += ["--since", since.isoformat()]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise click.ClickException(f"tg download failed: {result.stderr.strip()[-300:]}")
    items = []
    for saved in json.loads(result.stdout):
        path = pathlib.Path(saved["file"])
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        when = dt.datetime.fromisoformat(saved["ts"])
        # The Telegram message id is the stable handle (tg-cli >= 2026-09-04); older tg falls back to the file name.
        items.append({
            "id": str(saved.get("id") or path.name), "source": "telegram", "chat": chat, "ts": int(when.timestamp()), "duration": 0.0,
            "title": saved.get("caption") or "Vocal Telegram", "path": str(path), "reply_to": saved.get("reply_to"),
        })
    return items


# --- CLI --------------------------------------------------------------------

def parse_since(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    when = dt.datetime.fromisoformat(value)
    return when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)


def emit(items: list[dict], as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(items, ensure_ascii=False, indent=2))
        return
    for it in items:
        when = dt.datetime.fromtimestamp(it["ts"]).strftime("%Y-%m-%d %H:%M")
        line = f"{when}  {it['source']:<10} {it['id'][:24]:<24} {it['duration']:>5.0f}s  {it['title']}"
        click.echo(line)
        if it.get("text"):
            click.echo(f"    {it['text']}")


def transcribe_items(items: list[dict], fetch, model: str | None, glossary: str | None, backend: str | None = None) -> None:
    for it in items:
        if not it.get("path"):
            it["path"] = str(fetch(it))
        result = transcribe_file(pathlib.Path(it["path"]), model, glossary, backend)
        it.update(text=result["text"], language=result["language"], duration=result["duration"] or it["duration"], backend=result["backend"])


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", is_flag=True)
def cli(verbose: bool) -> None:
    """Local speech to text over Voice Memos, WhatsApp and Telegram recordings."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(name)s %(levelname)s %(message)s", stream=sys.stderr)


backend_option = click.option("--backend", default="auto", type=click.Choice(BACKENDS), show_default=True, help="whisper (glossary, fuller) or parakeet (faster); auto = whisper on Apple Silicon, parakeet elsewhere")
model_option = click.option("--model", default=None, help="weights to use instead of the backend's default (an HF repo for MLX/CTranslate2, an onnx-asr name on Linux)")
glossary_option = click.option("--glossary", default=None, help="file or inline comma-separated domain words for Whisper's initial prompt (or TRANSCRIBE_GLOSSARY)")
json_option = click.option("--json", "as_json", is_flag=True, help="machine-readable output")
since_option = click.option("--since", default=None, help="ISO date/datetime; only items at/after it (UTC if no tz)")
transcribe_option = click.option("--transcribe/--no-transcribe", default=False, help="also fetch and transcribe each item")


@cli.command()
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@backend_option
@model_option
@glossary_option
@json_option
def file(audio: pathlib.Path, backend: str, model: str | None, glossary: str | None, as_json: bool) -> None:
    """Transcribe one audio file (m4a, ogg/opus, mp3, wav, ...)."""
    result = transcribe_file(audio, model, load_glossary(glossary), backend)
    click.echo(json.dumps(result, ensure_ascii=False, indent=2) if as_json else result["text"])


@cli.command()
@since_option
@transcribe_option
@backend_option
@model_option
@glossary_option
@json_option
def voicememos(since: str | None, transcribe: bool, backend: str, model: str | None, glossary: str | None, as_json: bool) -> None:
    """List Apple Voice Memos on this Mac (macOS only), optionally transcribed."""
    items = [it for it in voicememos_items(since=parse_since(since)) if it["available"] or not transcribe]
    if transcribe:
        transcribe_items(items, lambda it: pathlib.Path(it["path"]), model, load_glossary(glossary), backend)
    emit(items, as_json)


@cli.command()
@click.argument("chat")
@click.option("--limit", default=80, help="recent messages to scan")
@click.option("--out", default=None, help="where downloaded audio goes (default: a temp dir)")
@since_option
@transcribe_option
@backend_option
@model_option
@glossary_option
@json_option
def whatsapp(chat: str, limit: int, out: str | None, since: str | None, transcribe: bool, backend: str, model: str | None, glossary: str | None, as_json: bool) -> None:
    """List the audio messages of a WhatsApp chat (name or JID), optionally downloaded and transcribed."""
    items = whatsapp_items(chat, limit, parse_since(since))
    if transcribe:
        out_dir = pathlib.Path(out) if out else pathlib.Path(tempfile.mkdtemp(prefix="transcribe-wa-"))
        transcribe_items(items, lambda it: whatsapp_fetch(it, out_dir), model, load_glossary(glossary), backend)
    emit(items, as_json)


@cli.command()
@click.argument("chat")
@click.option("--limit", default=50, help="recent messages to scan")
@click.option("--out", default=None, help="where downloaded audio goes (default: a temp dir)")
@since_option
@transcribe_option
@backend_option
@model_option
@glossary_option
@json_option
def telegram(chat: str, limit: int, out: str | None, since: str | None, transcribe: bool, backend: str, model: str | None, glossary: str | None, as_json: bool) -> None:
    """List the audio attachments of a Telegram chat (downloaded by `tg`), optionally transcribed."""
    out_dir = pathlib.Path(out) if out else pathlib.Path(tempfile.mkdtemp(prefix="transcribe-tg-"))
    items = telegram_items(chat, limit, parse_since(since), out_dir)
    if transcribe:
        transcribe_items(items, lambda it: pathlib.Path(it["path"]), model, load_glossary(glossary), backend)
    emit(items, as_json)


if __name__ == "__main__":
    cli()
