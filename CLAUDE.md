# yt-extract-text

Python CLI tools for YouTube/local-audio transcription on Windows with CUDA (Quadro P5000, Pascal sm_61).

## Scripts

- `extract-text.py` — downloads YouTube audio (yt-dlp), transcribes with openai-whisper (large, CUDA), optionally translates to Spanish via DeepL. Outputs `transcription.md` / `transcription_es.md` (gitignored).
- `transcribe_local.py` — transcribes a local audio file with faster-whisper: `python transcribe_local.py <audio> <out.txt> [out.srt] [model] [device] [compute]`. On Pascal use `int8` compute (fp16 is slow on sm_61); the script preloads torch's bundled cuBLAS/cuDNN DLLs.

## Environment

- Venv at `.venv/`; deps in `requirements.txt` (torch from the cu121 extra index). `requirements.txt` is the authoritative dependency list.
- Whisper models cache in `cache/`, downloaded audio in `temp/` — both gitignored; never commit them.
- Requires FFmpeg on PATH and an NVIDIA GPU (falls back to CPU, much slower).

## Review loop

PR review procedure, domain invariants for reviewers, and branch-naming convention live in `AGENTS.md` (read by codex/agy too) — that file is canonical for them.

## Security & Data Safety

- **Secrets**: never read, print, log, or commit `.env` (holds `DEEPL_API_KEY`). It is gitignored; keep it that way. Never echo the key into commits, PRs, or artifacts.
- **Injection surfaces**: user input reaches yt-dlp (URLs) and file paths taken from argv — never shell-interpolate them; pass as list args / library calls.
- **Dependencies**: before adding a package, verify the canonical name (typosquat risk) and prefer what's already in `requirements.txt`; flag low-adoption installs for confirmation.
- **Destructive ops**: `git push --force`, `git reset --hard`, `rm -rf` require explicit user confirmation.
