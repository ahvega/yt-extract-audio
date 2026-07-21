# AGENTS.md — yt-extract-text (repo: ahvega/yt-extract-audio)

Python CLI tools for YouTube/local-audio transcription on Windows with an NVIDIA Quadro P5000 (Pascal, sm_61). Public repo, solo maintainer.

| Module | Purpose | Status (2026-07-20) |
|---|---|---|
| `extract-text.py` | YouTube download (yt-dlp) → faster-whisper transcription → optional DeepL translation | repaired 2026-07-20: was unrunnable (openai-whisper uninstallable on py3.14, and its torch kernels cannot execute on sm_61) |
| `transcribe_local.py` | Local audio → faster-whisper (CUDA int8) → txt + SRT | working; reviewed + merged (PR #2), documented in README |
| `cuda_dlls.py` | Shared Windows CUDA DLL discovery for ctranslate2 (extracted from `transcribe_local.py`) | new 2026-07-20 |
| `requirements.txt` | Authoritative dependency list (torch pinned to the cu128 PyTorch index) | pinned set |

## Domain-critical invariants (reviewers: treat violations as blocking)

- **torch is NOT an inference engine here.** Verified 2026-07-20: `torch.cuda.get_arch_list()` on the pinned 2.11.0+cu128 build returns `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` — the Quadro P5000 is sm_61, so **no torch kernel can run on this GPU**. `torch.cuda.is_available()` still returns `True`, which makes this failure silent. torch is depended on *solely* for the cuBLAS/cuDNN binaries in `torch/lib` that ctranslate2 loads. **All GPU compute goes through CTranslate2**, which does support Pascal. Any proposal to route inference through torch on this hardware is wrong.
- **Dependency pins are load-bearing.** `torch` MUST stay pinned with the `+cu128` local version on `--extra-index-url https://download.pytorch.org/whl/cu128`. A bare `torch>=X` resolves a newer **CPU-only** wheel from PyPI (extra-index picks the highest version across indexes) and silently drops the DLLs ctranslate2 needs. 2.11.0+cu128 is the newest build on that index — PyPI showing a higher `torch` version is not a reason to bump. `faster-whisper`/`ctranslate2` pins must keep cuDNN majors compatible with those torch DLLs. When updating pins, read the working venv first (`.venv/Scripts/pip list`) — it is the empirically working set and requirements.txt has drifted from it in **both** directions (listing packages the venv lacks, and lagging what it has).
- **Pascal (sm_61) quirk:** `int8` compute is fast (DP4A); `fp16` is slow. Don't "upgrade" defaults to fp16.
- **Both scripts reuse torch's bundled cuBLAS/cuDNN DLLs** via `cuda_dlls.py` (`os.add_dll_directory` derived from the installed torch package) — never reintroduce hardcoded machine paths. `import cuda_dlls` MUST stay above any `faster_whisper`/`ctranslate2` import; reordering it breaks CUDA at import time.
- **Secrets:** `.env` holds `DEEPL_API_KEY`; never read, print (even partially), log, or commit it. `.env` is gitignored and has never been in history — keep it that way.
- **Public repo:** transcription outputs (`transcription*.md`, `*.srt`) and media files are gitignored because they may contain copyrighted third-party content. Never force-add them.
- **User input reaches yt-dlp (URLs) and argv file paths** — library calls only, never shell interpolation.

## Build / test

- Venv: `.venv/` (Python 3.14). Install: `pip install -r requirements.txt`.
- No test suite. Gates, in order — **`py_compile` alone is not enough and has already let a fully unrunnable script through** (it only parses; it never resolves an import):
  1. Compile: `.venv/Scripts/python -m py_compile extract-text.py transcribe_local.py cuda_dlls.py`
  2. **Import** (catches missing/uninstallable deps — the failure `py_compile` misses): `.venv/Scripts/python -c "import importlib.util as u; m=['deepl','dotenv','yt_dlp','faster_whisper','ctranslate2','torch']; b=[x for x in m if not u.find_spec(x)]; print('missing:', b or 'none'); raise SystemExit(1 if b else 0)"`
  3. GPU smoke test, for any CUDA/DLL/dependency change: `ffmpeg -y -f lavfi -i "sine=frequency=440:duration=3" -ar 16000 t.wav` then `.venv/Scripts/python transcribe_local.py t.wav t.txt "" tiny cuda int8` (exit 0 = model load + decode verified on the P5000).
- Never add a dependency that routes GPU inference through torch — see the sm_61 invariant above.
- Branch naming (enforced by hook): `<type>/<id>--<source>`, type ∈ {feat,fix,docs,chore,refactor,test,perf}, source ∈ {rm,spec,plan,issue-N,user} (rm/spec/plan ids must include the ws-code).

## Review procedure

**Per-task PR + review loop (required for every implementation task).** This procedure is
authoritative and self-contained. Review sources: **Sourcery** (automated, in-PR) +
**codex** (adversarial, GPT CLI — required gate on dependency-pin/CUDA and secret-handling
changes) + **agy** (Gemini CLI, optional third viewpoint).

1. Implement on a branch off `main` (dedicated worktree if multiple sessions share the
   checkout); `python -m py_compile extract-text.py transcribe_local.py` green; commit.
2. Push the branch and open a PR. Agent-authored PRs always require **human** merge
   approval — never merge your own PR.
3. **Wait for the Sourcery review. In parallel**, run codex over the exact PR diff
   (`--base main` = the PR's Files-changed), backgrounded, from the branch worktree, with
   focus: "challenge approach/design/tradeoffs AND find concrete defects". For substantial
   features run codex at **spec, plan, AND code** stages.
   If agy is available: `agy --add-dir . --new-project -p "<review prompt>"` — the
   `--add-dir` attachment is mandatory (a bare `agy -p` reads neither the repo nor
   AGENTS.md/GEMINI.md); `--new-project` avoids cross-project memory bleed.
4. **Triage all sources with the landing rule:**
   - Concrete file+line defect (bug / security issue / measurable perf) → post to the PR
     (`gh pr comment`, labeled with the tool) and commit accepted fixes as
     `fix(...): ... (Sourcery/Codex)`.
   - Approach-level critique with no single fix site (design/API shape, architecture
     tradeoffs) → stays in the session; decide, don't auto-apply.
   - Borderline: names a fix at a location → PR; questions the whole direction → session.
   Skip/justify findings that don't apply. Iterate until reviews are settled.
5. **Don't over-wait** for Sourcery's confirmation re-review (throttled/unreliable): if
   outstanding feedback is non-blocking and addressed, checks are green, and the PR is
   mergeable → request human merge and continue.
6. Expect codex to escalate to ever-narrower edges: fix the concrete asymmetries, then
   **draw the line** — PR + Sourcery + human merge are the remaining gates.

### Adjudicated findings (do not re-flag; see PR #2 triage)

- **Discarded `os.add_dll_directory()` handle**: NOT a bug — CPython's `_AddedDllDirectory`
  has no finalizer (verified against the 3.14 stdlib); the handle is retained anyway as
  hardening. Claims that refcounting removes the search path are wrong.
- **Incremental flush instead of atomic temp-file writes** in `transcribe_local.py`:
  deliberate — live progress and salvageable partial transcripts. An input/output
  path-collision guard covers the destructive case.
- **Positional CLI (no argparse)** and **`--extra-index-url` (not `--index-url`)**:
  accepted tradeoffs — the non-torch deps only exist on PyPI, and the at-risk packages
  carry exact local-version pins.
- **`requested_downloads[0]['filepath']` is "the pre-postprocessed path"**: NOT a bug —
  agy raised this as BLOCKING in PR #6, claiming `download_audio()` would crash on every
  download. Disproven empirically (yt-dlp 2026.7.4, `FFmpegExtractAudio` → wav, served
  over localhost): the field reads `temp\clip.wav` and the file exists. yt-dlp mutates
  those entries in place when postprocessors run. The proposed fix — `info.get('filepath')`
  — is actively wrong: that key is `None` on the top-level info dict, so it would fall
  through to the guessed path. Re-verify with a local HTTP-served audio file before
  accepting any future claim here.
- **Whitespace collapse in DeepL chunk reassembly**: fixed in PR #6 — `chunk_text()`
  returns `(separator, chunk)` pairs so rejoining reproduces the source exactly. Do not
  "simplify" it back to a plain list joined with `" "`; that reintroduces both the
  mid-word space injection and the newline loss.

*Data note:* the loop sends only diffs (never gitignored secrets); the repo must hold no
real PII/PHI. External reviewer CLIs are authorized dev tools with the same repo access as
the primary agent.

## Security

- Never read, print, or commit `.env` or any credential file; external reviewers receive diffs only, never gitignored files.
- No real PII in the repo or in reviewer prompts.
- Destructive git ops (`push --force`, `reset --hard`, `clean`, `rm -rf`) require explicit user confirmation.
