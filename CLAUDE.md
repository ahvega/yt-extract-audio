# yt-extract-text

Python CLI tools for YouTube/local-audio transcription on Windows with CUDA (Quadro P5000, Pascal sm_61).

## Scripts

- `extract-text.py` — downloads YouTube audio (yt-dlp), transcribes with **faster-whisper** (`large-v3`, CUDA int8), optionally translates via DeepL. `--url` is required; see `--help` for model/device/output flags. Outputs `transcription.md` / `transcription_es.md` (gitignored). Exits non-zero on failure.
- `transcribe_local.py` — transcribes a local audio file with faster-whisper: `python transcribe_local.py <audio> <out.txt> [out.srt] [model] [device] [compute] [language]` (pass `""` to skip `out.srt`; language defaults to auto-detect). On Pascal use `int8` compute (fp16 is slow on sm_61).
- `cuda_dlls.py` — shared helper that puts torch's bundled cuBLAS/cuDNN directory on the Windows DLL search path. **Import it before `faster_whisper`/`ctranslate2`** or CUDA fails at import time.

## GPU reality check

torch cannot run kernels on this machine's Quadro P5000 — the pinned cu128 build targets sm_75+, the card is sm_61, and `torch.cuda.is_available()` returns `True` anyway, so the failure is silent. **CTranslate2 does all GPU compute**; torch is present only as a source of cuBLAS/cuDNN DLLs. Never route inference through torch here.

## Environment

- Venv at `.venv/`; deps in `requirements.txt` (torch pinned on the cu128 extra index). `requirements.txt` is the authoritative dependency list.
- When pinning/updating deps, read the working venv first (`.venv/Scripts/pip list`) — it is the empirically working set; requirements.txt has drifted from it before (cu121 vs installed cu128).
- Verification gate: `py_compile` only parses — it has already passed a script whose every dependency was missing. Always follow it with an **import check** (`find_spec` over `deepl,dotenv,yt_dlp,faster_whisper,ctranslate2,torch`), and for CUDA/DLL/dependency changes the GPU smoke test: `ffmpeg -y -f lavfi -i "sine=frequency=440:duration=3" -ar 16000 t.wav` then `.venv/Scripts/python transcribe_local.py t.wav t.txt "" tiny cuda int8` — exit 0 proves model load + decode on the P5000. Full commands in `AGENTS.md`.
- Whisper models cache in `cache/`, downloaded audio in `temp/` — both gitignored; never commit them.
- Requires FFmpeg on PATH and an NVIDIA GPU. `extract-text.py` falls back to CPU (much slower); `transcribe_local.py` defaults to CUDA — pass `cpu` as the device argument to run without a GPU.

## Review loop

PR review procedure, domain invariants for reviewers, and branch-naming convention live in `AGENTS.md` (read by codex/agy too) — that file is canonical for them.

## Security & Data Safety

- **Secrets**: never read, print, log, or commit `.env` (holds `DEEPL_API_KEY`). It is gitignored; keep it that way. Never echo the key into commits, PRs, or artifacts.
- **Injection surfaces**: user input reaches yt-dlp (URLs) and file paths taken from argv — never shell-interpolate them; pass as list args / library calls.
- **Dependencies**: before adding a package, verify the canonical name (typosquat risk) and prefer what's already in `requirements.txt`; flag low-adoption installs for confirmation.
- **Destructive ops**: `git push --force`, `git reset --hard`, `rm -rf` require explicit user confirmation.
