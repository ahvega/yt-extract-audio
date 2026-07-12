# yt-extract-text

Python CLI tools for YouTube/local-audio transcription on Windows with CUDA (Quadro P5000, Pascal sm_61).

## Scripts

- `extract-text.py` — downloads YouTube audio (yt-dlp), transcribes with openai-whisper (large, CUDA), optionally translates to Spanish via DeepL. Outputs `transcription.md` / `transcription_es.md` (gitignored).
- `transcribe_local.py` — transcribes a local audio file with faster-whisper: `python transcribe_local.py <audio> <out.txt> [out.srt] [model] [device] [compute] [language]` (pass `""` to skip `out.srt`; language defaults to auto-detect). On Pascal use `int8` compute (fp16 is slow on sm_61); the script preloads torch's bundled cuBLAS/cuDNN DLLs.

## Environment

- Venv at `.venv/`; deps in `requirements.txt` (torch pinned on the cu128 extra index). `requirements.txt` is the authoritative dependency list.
- When pinning/updating deps, read the working venv first (`.venv/Scripts/pip list`) — it is the empirically working set; requirements.txt has drifted from it before (cu121 vs installed cu128).
- Verification gate: `py_compile` alone is insufficient for CUDA/DLL changes. GPU smoke test: `ffmpeg -y -f lavfi -i "sine=frequency=440:duration=3" -ar 16000 t.wav` then `.venv/Scripts/python transcribe_local.py t.wav t.txt "" tiny cuda int8` — exit 0 proves model load + decode on the P5000.
- Whisper models cache in `cache/`, downloaded audio in `temp/` — both gitignored; never commit them.
- Requires FFmpeg on PATH and an NVIDIA GPU. `extract-text.py` falls back to CPU (much slower); `transcribe_local.py` defaults to CUDA — pass `cpu` as the device argument to run without a GPU.

## Review loop

PR review procedure, domain invariants for reviewers, and branch-naming convention live in `AGENTS.md` (read by codex/agy too) — that file is canonical for them.

## Security & Data Safety

- **Secrets**: never read, print, log, or commit `.env` (holds `DEEPL_API_KEY`). It is gitignored; keep it that way. Never echo the key into commits, PRs, or artifacts.
- **Injection surfaces**: user input reaches yt-dlp (URLs) and file paths taken from argv — never shell-interpolate them; pass as list args / library calls.
- **Dependencies**: before adding a package, verify the canonical name (typosquat risk) and prefer what's already in `requirements.txt`; flag low-adoption installs for confirmation.
- **Destructive ops**: `git push --force`, `git reset --hard`, `rm -rf` require explicit user confirmation.
