---
name: transcribe-cli
description: Local speech to text over the audio the user records or receives, via the bundled `transcribe` command. `transcribe file <audio>` runs Whisper locally on any file (m4a, ogg/opus, mp3, wav); `transcribe voicememos` lists Apple Voice Memos on the Mac, `transcribe whatsapp <chat>` the voice notes and audio files of a WhatsApp chat (through `wa`), `transcribe telegram <chat>` the audio attachments of a Telegram chat (through `tg`), each with `--transcribe` to fetch and transcribe and `--json` for stable ids. Two local models via `--backend`: Whisper large-v3-turbo (glossary, default on Apple Silicon) and NVIDIA parakeet-tdt-0.6b-v3 (several times faster, default on a CPU-only box), each running on MLX on the Mac and on CTranslate2 / ONNX Runtime on Linux. Use when the user wants a recording, voice memo or voice note transcribed, or a heartbeat needs new recordings as text.
---

# transcribe-cli

## How to invoke

`transcribe` is on PATH (linked by claudine's `link_clis.py`); otherwise run
`bin/transcribe` from this repo. Every command has `-h`.

```bash
transcribe file ~/Downloads/note.m4a                    # text on stdout
transcribe file note.ogg --json                         # {text, language, duration, segments, backend, model, path}
transcribe voicememos --since 2026-09-01                # list Apple Voice Memos (macOS only)
transcribe voicememos --since 2026-09-01 --transcribe --json
transcribe whatsapp "Clément Walter" --transcribe --out ~/audio   # voice notes + audio documents of a chat
transcribe telegram "Marguerite" --limit 50 --transcribe --json
```

## Items

Source commands emit one item per recording:

```json
{"id": "...", "source": "whatsapp", "chat": "...", "ts": 1788427224, "duration": 3.0,
 "title": "Vocal WhatsApp", "path": "/.../3ABFA8.ogg", "text": "...", "language": "fr"}
```

`id` is stable per source: the Voice Memos uuid, the WhatsApp `msg_id`, the
Telegram file name. WhatsApp items also carry `quoted_id` and `from_me`.
Without `--transcribe`, `path` is null for WhatsApp (not downloaded yet) and
`text` is absent. `--since` filters on the recording time (UTC if no tz).

## Backends and models

`--backend whisper|parakeet|auto` (default `auto`):

| backend  | model                          | Apple Silicon  | Linux CPU            | traits                                             |
| -------- | ------------------------------ | -------------- | -------------------- | -------------------------------------------------- |
| whisper  | Whisper large-v3-turbo         | mlx-whisper    | faster-whisper int8  | honours `--glossary`, fuller transcripts, slower   |
| parakeet | NVIDIA parakeet-tdt-0.6b-v3    | parakeet-mlx   | onnx-asr int8        | 2-5x faster, better punctuation, no glossary       |

`auto` is whisper on Apple Silicon (both take seconds there) and parakeet on a
CPU-only box (Whisper turbo ran at 3x real time on 2 vCPU). Measured on the same
French memos: equal accuracy overall, different failures. Whisper spells glossary
words right (T'choupi, Zama); Parakeet once dropped a memo's closing sentence.
`--model` overrides the weights of either backend. Weights are cached in
`~/.cache/huggingface`. On Linux, `.m4a` needs `ffmpeg` on PATH; Ogg/Opus voice
notes do not.

## Glossary

Whisper spells domain words right when told them: pass `--glossary` a file or
an inline comma-separated list, or set `TRANSCRIBE_GLOSSARY`. The dictaphone
heartbeat keeps its list in `heartbeats/dictaphone/dictaphone-glossary.txt`.

## Requirements per source

- voicememos: macOS, and Full Disk Access for the calling process (launchd jobs
  must run under an interpreter that holds it).
- whatsapp: the `wa` CLI paired on this machine; a chat name or JID; the
  self-chat is the `@lid` JID `wa list` shows under the user's own name.
- telegram: the `tg` CLI logged in on this machine; `tg download` fetches every
  attachment of the scanned window, `transcribe` keeps the audio ones.

## Not this tool

Deciding what to do with a transcript. Heartbeats own the "seen" state and the
agent step; this CLI only turns recordings into text with stable ids.
