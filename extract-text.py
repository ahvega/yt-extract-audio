"""YouTube audio -> transcript -> optional DeepL translation.

Downloads a video's audio with yt-dlp, transcribes it with faster-whisper
(CTranslate2 backend, the same engine as transcribe_local.py) and optionally
translates the transcript with DeepL.
"""

import argparse
import os
import re
import sys
import time

import deepl
from dotenv import load_dotenv
from yt_dlp import YoutubeDL

# Sets the CUDA DLL search path; must run before faster_whisper/ctranslate2 load.
# That ordering is enforced structurally rather than by comment: faster_whisper is
# imported inside load_model(), where no import sorter can hoist it above this line.
import cuda_dlls  # noqa: F401

TEMP_DIR = "temp"
CACHE_DIR = "cache"
DEFAULT_MODEL = "large-v3"
DEFAULT_COMPUTE = "int8"  # Pascal (sm_61): int8 uses fast DP4A kernels; fp16 is slow
# Bytes, not characters: DeepL's cap applies to the encoded request body, so a
# character-based bound overshoots badly on non-ASCII (100k CJK chars ~= 300 KiB).
# Held well under the 128 KiB cap to leave room for the JSON envelope.
DEEPL_CHUNK_LIMIT = 100_000
CPU_SAFE_COMPUTE = {"int8", "int8_float32", "float32"}
# Substrings that mark an exception as a CUDA/driver/library init failure rather than a
# bad model name, a network error, or a permissions problem.
CUDA_FAILURE_MARKERS = ("cuda", "cublas", "cudnn", "gpu", "device", "driver", "no kernel image")
# Languages written without spaces between words.
UNSPACED_LANGUAGES = frozenset({"ja", "zh", "yue", "th", "lo", "my", "km"})
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def ensure_dir(path):
    """Create a directory if missing and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def _progress_hook(d):
    """yt-dlp progress callback: renders a download bar on one console line."""
    if d["status"] == "downloading":
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        if total:
            pct = d.get("downloaded_bytes", 0) / total * 100
            filled = int(pct / 2)
            print(f"\r   [{'#' * filled}{'.' * (50 - filled)}] {pct:5.1f}%", end="", flush=True)
    elif d["status"] == "finished":
        print(f"\r   download complete{' ' * 44}")


def download_audio(url):
    """Download a video's audio as WAV.

    Returns the path to the extracted audio. Raises on failure -- callers are
    expected to translate that into a non-zero exit status.
    """
    ensure_dir(TEMP_DIR)
    opts = {
        "format": "bestaudio/best",
        # Keyed by video id so a crashed run cannot collide with the next one.
        "outtmpl": os.path.join(TEMP_DIR, "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "http_headers": {"User-Agent": USER_AGENT},
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
    }

    print("1. Downloading audio...")
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Ask yt-dlp where the file actually landed rather than assuming a name --
        # the postprocessor, not the template, decides the final extension.
        downloads = info.get("requested_downloads") or []
        path = downloads[0].get("filepath") if downloads else None
        if not path:
            path = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"

    print(f"   {info.get('title', 'unknown title')}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"yt-dlp reported success but no audio at {path}")
    return path


def _is_cuda_failure(exc):
    """True when an exception looks like a CUDA/driver/library initialisation failure.

    Guards the CPU fallback: without this, an invalid model name or a network error is
    reported as "CUDA unavailable" and retried on CPU, hiding the real cause and paying
    for a second model download.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in CUDA_FAILURE_MARKERS)


def load_model(model_name, device, compute, cache_dir):
    """Load a faster-whisper model, falling back to CPU if CUDA is unusable.

    The fallback covers model *construction* only. faster-whisper decodes lazily, so a
    CUDA failure raised while iterating segments is not recoverable here.

    Returns (model, device_actually_used).
    """
    cuda_dlls.preload_cuda_dlls()
    from faster_whisper import WhisperModel  # imported here: after the DLL path is set

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute, download_root=cache_dir)
        return model, device
    except Exception as exc:
        if device == "cpu" or not _is_cuda_failure(exc):
            raise
        # float16 and friends are GPU-only in CTranslate2 -- reusing the requested
        # compute type on CPU would crash the fallback it is meant to rescue.
        cpu_compute = compute if compute in CPU_SAFE_COMPUTE else "int8"
        swap = "" if cpu_compute == compute else f"; compute {compute} -> {cpu_compute} (CPU-unsupported)"
        print(f"\nCUDA unavailable ({exc}); falling back to CPU -- significantly slower{swap}.", file=sys.stderr)
        model = WhisperModel(model_name, device="cpu", compute_type=cpu_compute, download_root=cache_dir)
        return model, "cpu"


def transcribe(model, audio_path, language, use_vad=True):
    """Transcribe audio and return the joined text.

    faster-whisper yields segments lazily, so decoding happens as this loop runs --
    which is what makes the progress readout real rather than simulated.

    VAD is on by default (it skips silence and speeds long recordings up considerably)
    but can drop quiet or clipped speech, so `--no-vad` disables it. The previous
    openai-whisper path had no VAD at all.
    """
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=use_vad,
        vad_parameters=dict(min_silence_duration_ms=500) if use_vad else None,
    )
    mode = "forced" if language else "detected"
    print(
        f"   language ({mode}): {info.language} "
        f"(p={info.language_probability:.2f}), duration={info.duration:.0f}s"
    )

    parts = []
    t0 = time.time()
    for seg in segments:
        parts.append(seg.text.strip())
        pct = (seg.end / info.duration * 100) if info.duration else 0
        print(f"\r   [{pct:5.1f}%] elapsed={time.time() - t0:4.0f}s", end="", flush=True)
    print(f"\r   transcribed in {time.time() - t0:.0f}s{' ' * 24}")
    # Japanese/Chinese/Thai and friends are written without inter-word spaces; joining
    # their segments with " " would inject breaks the source text does not have.
    glue = "" if info.language in UNSPACED_LANGUAGES else " "
    return glue.join(p for p in parts if p)


def format_to_markdown(text, source_url=None):
    """Convert a transcript into markdown, promoting likely headings.

    Headings are inferred from punctuation only -- questions, short lines ending in a
    colon, and numbered points. Deliberately content-agnostic: an earlier version keyed
    off a hardcoded topic list taken from one specific video, which produced arbitrary
    headings on anything else.
    """
    md = "# Transcripción\n\n"
    if source_url:
        md += f"**Fuente:** {source_url}\n\n"

    current = []
    for paragraph in text.split(". "):
        p = paragraph.strip()
        if not p:
            continue

        is_heading = (
            ("?" in p and len(p) < 100)  # questions
            or (len(p) < 80 and p.endswith(":"))  # short lead-ins
            or (len(p) < 100 and re.match(r"^\s*(\d+[.)]|número\b)", p, re.IGNORECASE))  # numbered points
        )

        if is_heading:
            if current:
                md += "\n".join(current) + "\n\n"
                current = []
            md += f"\n## {p}\n\n"
        else:
            current.append(p if p.endswith((".", "!", "?")) else p + ".")
            if len(current) >= 4:
                md += "\n".join(current) + "\n\n"
                current = []

    if current:
        md += "\n".join(current) + "\n"
    return md


def _utf8_len(s):
    return len(s.encode("utf-8"))


def _fit(s, byte_budget):
    """Longest prefix of `s` encoding to at most `byte_budget` UTF-8 bytes.

    Truncates on the encoded form and discards a trailing partial character, so a
    multi-byte codepoint is never split down the middle.
    """
    if byte_budget <= 0:
        return ""
    return s.encode("utf-8")[:byte_budget].decode("utf-8", errors="ignore")


def _atoms(text, limit):
    """Yield (separator, token) pairs; concatenating every separator+token rebuilds `text`.

    A token larger than a whole chunk is hard-split into pieces joined by "" so that
    reassembly cannot invent a word break in the middle of it.
    """
    for ws, token in re.findall(r"(\s*)(\S+)", text):
        sep = ws
        while _utf8_len(token) > limit:
            head = _fit(token, limit)
            if not head:
                break
            yield sep, head
            token, sep = token[len(head):], ""
        if token:
            yield sep, token


def chunk_text(text, limit=DEEPL_CHUNK_LIMIT):
    """Split text into chunks whose UTF-8 encoding stays under `limit` bytes.

    Returns [(separator, chunk), ...] where `separator` is the whitespace that joined
    this chunk to the previous one in the source -- "" when a single token had to be
    split mid-word. Rejoining with those separators reproduces the original spacing
    instead of collapsing every boundary to one space.
    """
    if _utf8_len(text) <= limit:
        return [("", text)]

    chunks, current, current_sep = [], "", ""
    for sep, token in _atoms(text, limit):
        if not current:
            current, current_sep = token, sep
        elif _utf8_len(current) + _utf8_len(sep) + _utf8_len(token) > limit:
            chunks.append((current_sep, current))
            current, current_sep = token, sep
        else:
            current += sep + token
    if current:
        chunks.append((current_sep, current))
    return chunks


def translate(text, target_lang):
    """Translate text with DeepL.

    Returns the translated text, or None when no API key is configured (translation is
    an optional feature). Genuine API failures propagate to the caller.
    """
    load_dotenv()
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        return None

    # DeepL routes free (':fx') vs Pro keys itself -- pinning the free endpoint here
    # used to make Pro keys fail outright.
    client_cls = getattr(deepl, "DeepLClient", None) or deepl.Translator
    client = client_cls(api_key)

    chunks = chunk_text(text)
    print(f"3. Translating to {target_lang} ({len(chunks)} chunk(s))...")

    out = []
    for i, (sep, chunk) in enumerate(chunks, 1):
        # One request per chunk. Handing the whole list to translate_text() would put
        # every chunk into a single HTTP body -- precisely the size cap the chunking
        # exists to stay under, so batching here would defeat it entirely.
        result = client.translate_text(chunk, target_lang=target_lang)
        out.append(("" if i == 1 else sep) + result.text)
        if len(chunks) > 1:
            print(f"   chunk {i}/{len(chunks)}", flush=True)
    return "".join(out)


def translated_path(output, target_lang):
    """Derive the translation's filename from the transcript's (foo.md -> foo_es.md)."""
    stem, ext = os.path.splitext(output)
    return f"{stem}_{target_lang.lower()}{ext}"


def write_markdown(path, content):
    """Write markdown, creating the parent directory when the path is nested."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="YouTube transcription and translation tool")
    parser.add_argument("--url", "-u", required=True, help="YouTube video URL to process")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Whisper model (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"], help="Compute device (default: cuda)")
    parser.add_argument("--compute", "-c", default=DEFAULT_COMPUTE, help=f"Compute type (default: {DEFAULT_COMPUTE})")
    parser.add_argument("--language", "-l", default=None, help="Source language code (default: auto-detect)")
    parser.add_argument("--output", "-o", default="transcription.md", help="Transcript output path")
    parser.add_argument("--target-lang", default="ES", help="DeepL target language (default: ES)")
    parser.add_argument("--no-translate", action="store_true", help="Skip the DeepL translation step")
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable voice-activity filtering (slower, but keeps quiet or clipped speech)",
    )
    args = parser.parse_args(argv)
    args.language = args.language or None  # an explicit -l "" means auto-detect
    return args


def main(argv=None):
    args = parse_args(argv)
    cache_dir = ensure_dir(CACHE_DIR)

    audio_path = None
    try:
        audio_path = download_audio(args.url)

        print("2. Transcribing...")
        model, device = load_model(args.model, args.device, args.compute, cache_dir)
        print(f"   model={args.model} device={device} compute={args.compute} vad={not args.no_vad}")
        text = transcribe(model, audio_path, args.language, use_vad=not args.no_vad)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print("   temporary audio removed")

    if not text.strip():
        print("Error: transcription produced no text", file=sys.stderr)
        return 1

    try:
        write_markdown(args.output, format_to_markdown(text, source_url=args.url))
    except OSError as exc:
        print(f"Error: could not write transcript to {args.output}: {exc}", file=sys.stderr)
        return 1
    print(f"Transcription saved to {args.output}")

    if args.no_translate:
        return 0

    try:
        translated = translate(text, args.target_lang)
    except deepl.exceptions.AuthorizationException:
        print("Error: invalid DeepL API key -- check .env", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: translation failed: {exc}", file=sys.stderr)
        return 1

    if translated is None:
        print("DEEPL_API_KEY not set -- skipping translation.", file=sys.stderr)
        return 0

    out_path = translated_path(args.output, args.target_lang)
    try:
        write_markdown(out_path, format_to_markdown(translated, source_url=args.url))
    except OSError as exc:
        print(f"Error: could not write translation to {out_path}: {exc}", file=sys.stderr)
        return 1
    print(f"Translation saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
