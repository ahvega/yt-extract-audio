import importlib.util
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
language = sys.argv[7] if len(sys.argv) > 7 else None  # None = auto-detect

# Pascal (P5000) GPU: ctranslate2 needs cuBLAS/cuDNN DLLs. Reuse the ones bundled with torch.
_torch_spec = importlib.util.find_spec("torch")
_torch_lib = os.path.join(os.path.dirname(_torch_spec.origin), "lib") if _torch_spec and _torch_spec.origin else None
if _torch_lib and os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)
else:
    print("Warning: torch lib dir not found; ctranslate2 may fail to locate cuBLAS/cuDNN DLLs", file=sys.stderr)

from faster_whisper import WhisperModel

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
