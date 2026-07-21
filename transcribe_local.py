import os
import sys
import time
from contextlib import ExitStack

if len(sys.argv) < 3:
    print(
        f"Usage: python {os.path.basename(sys.argv[0])} <audio> <out.txt> [out.srt] [model] [device] [compute] [language]",
        file=sys.stderr,
    )
    sys.exit(2)

audio = sys.argv[1]
out_txt = sys.argv[2]
out_srt = sys.argv[3] if len(sys.argv) > 3 else None
model_name = sys.argv[4] if len(sys.argv) > 4 else "large-v3"
device = sys.argv[5] if len(sys.argv) > 5 else "cuda"
compute = sys.argv[6] if len(sys.argv) > 6 else "int8"  # Pascal: int8 (DP4A) is fast; fp16 is slow on sm_61
language = (sys.argv[7] if len(sys.argv) > 7 else None) or None  # None/"" = auto-detect

_audio_abs = os.path.abspath(audio)
if any(_out and os.path.abspath(_out) == _audio_abs for _out in (out_txt, out_srt)):
    print("Error: an output path equals the input audio path (would truncate the input)", file=sys.stderr)
    sys.exit(2)
if out_srt and os.path.abspath(out_srt) == os.path.abspath(out_txt):
    print("Error: <out.txt> and <out.srt> are the same path", file=sys.stderr)
    sys.exit(2)

# Pascal (P5000): ctranslate2 needs cuBLAS/cuDNN DLLs, reused from torch's bundle.
# This import MUST stay above faster_whisper -- it sets the DLL search path.
import cuda_dlls  # noqa: F401,E402

# isort: split
# ruff: isort: split
from faster_whisper import WhisperModel  # noqa: E402

print(f"Loading model ({model_name}, {device}, {compute})...", flush=True)
model = WhisperModel(model_name, device=device, compute_type=compute)

print(f"Transcribing: {audio}", flush=True)
t0 = time.time()
segments, info = model.transcribe(
    audio,
    language=language,
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
lang_mode = "forced" if language else "detected"
print(f"Language ({lang_mode}): {info.language} (p={info.language_probability:.2f}), duration={info.duration:.1f}s", flush=True)


def fmt(t):
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


with ExitStack() as stack:
    ft = stack.enter_context(open(out_txt, "w", encoding="utf-8"))
    srt_f = stack.enter_context(open(out_srt, "w", encoding="utf-8")) if out_srt else None
    for idx, seg in enumerate(segments, 1):
        line = seg.text.strip()
        ft.write(f"[{fmt(seg.start)} -> {fmt(seg.end)}] {line}\n")
        ft.flush()
        if srt_f:
            srt_f.write(f"{idx}\n{fmt(seg.start).replace('.', ',')} --> {fmt(seg.end).replace('.', ',')}\n{line}\n\n")
            srt_f.flush()
        pct = (seg.end / info.duration) * 100 if info.duration else 0
        elapsed = time.time() - t0
        print(f"[{pct:5.1f}%] t={fmt(seg.end)} elapsed={elapsed:6.0f}s :: {line[:70]}", flush=True)

print(f"DONE in {time.time()-t0:.0f}s -> {out_txt}", flush=True)
