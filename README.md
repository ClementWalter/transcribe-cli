# transcribe-cli

Local speech to text over the audio you record or receive: Apple Voice Memos,
WhatsApp voice notes, Telegram voice messages, or any file. Whisper
large-v3-turbo on Apple Silicon (mlx-whisper) and Linux (faster-whisper).

```bash
bin/transcribe file note.m4a
bin/transcribe whatsapp "Clément Walter" --transcribe --json
```

See `SKILL.md` for the full command reference. Tests:

```bash
PYTHONPATH=. uv run --with pytest --with click --with numpy --with scipy --with soundfile pytest -q tests
```
