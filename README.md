# transcribe-cli

**Visibility: public.**

**Extension: not shipped** as a Claude Desktop extension: its local speech models are large downloads meant for a terminal setup.

Local speech to text over the audio you record or receive: Apple Voice Memos,
WhatsApp voice notes, Telegram voice messages, or any file. Two local models,
`--backend whisper|parakeet|auto`: Whisper large-v3-turbo (glossary) and NVIDIA
parakeet-tdt-0.6b-v3 (faster), on MLX (Apple Silicon) or CTranslate2 / ONNX
Runtime (Linux CPU).

```bash
bin/transcribe file note.m4a
bin/transcribe whatsapp "Alice Example" --transcribe --json
```

See `SKILL.md` for the full command reference. Tests:

```bash
uv run --script tests/test_transcribe_cli.py
```
