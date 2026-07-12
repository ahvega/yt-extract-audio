# AGENTS.md — yt-extract-text (repo: ahvega/yt-extract-audio)

Python CLI tools for YouTube/local-audio transcription on Windows with an NVIDIA Quadro P5000 (Pascal, sm_61). Public repo, solo maintainer.

| Module | Purpose | Status (2026-07-11) |
|---|---|---|
| `extract-text.py` | YouTube download (yt-dlp) → openai-whisper transcription → optional DeepL translation to Spanish | working, stable |
| `transcribe_local.py` | Local audio → faster-whisper (CUDA int8) → txt + SRT | new, post-review fixes applied |
| `requirements.txt` | Authoritative dependency list (torch pinned to the cu128 PyTorch index) | pinned set |

## Domain-critical invariants (reviewers: treat violations as blocking)

- **Dependency pins are load-bearing.** `torch` MUST stay pinned with the `+cu128` local version on `--extra-index-url https://download.pytorch.org/whl/cu128`. A bare `torch>=X` resolves a newer **CPU-only** wheel from PyPI (extra-index picks the highest version across indexes) and silently kills CUDA. `faster-whisper`/`ctranslate2` pins must keep cuDNN majors compatible with the bundled torch DLLs.
- **Pascal (sm_61) quirk:** `int8` compute is fast (DP4A); `fp16` is slow. Don't "upgrade" defaults to fp16.
- **`transcribe_local.py` reuses torch's bundled cuBLAS/cuDNN DLLs** via `os.add_dll_directory` derived from the installed torch package — never reintroduce hardcoded machine paths.
- **Secrets:** `.env` holds `DEEPL_API_KEY`; never read, print (even partially), log, or commit it. `.env` is gitignored and has never been in history — keep it that way.
- **Public repo:** transcription outputs (`transcription*.md`, `*.srt`) and media files are gitignored because they may contain copyrighted third-party content. Never force-add them.
- **User input reaches yt-dlp (URLs) and argv file paths** — library calls only, never shell interpolation.

## Build / test

- Venv: `.venv/` (Python 3.14). Install: `pip install -r requirements.txt`.
- No test suite. Minimum gate: `.venv/Scripts/python -m py_compile extract-text.py transcribe_local.py` plus a manual smoke run when behavior changes.
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

*Data note:* the loop sends only diffs (never gitignored secrets); the repo must hold no
real PII/PHI. External reviewer CLIs are authorized dev tools with the same repo access as
the primary agent.

## Security

- Never read, print, or commit `.env` or any credential file; external reviewers receive diffs only, never gitignored files.
- No real PII in the repo or in reviewer prompts.
- Destructive git ops (`push --force`, `reset --hard`, `clean`, `rm -rf`) require explicit user confirmation.
