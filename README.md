# YouTube Transcription & Translation Tool

A Python script that downloads YouTube videos, transcribes them with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend, CUDA-accelerated), and optionally translates them to Spanish using the DeepL API.

## Features

- Downloads YouTube video audio using yt-dlp
- Transcribes audio using faster-whisper (default model: `large-v3`)
- CUDA GPU acceleration, with automatic CPU fallback
- Translation via DeepL, with automatic chunking for long transcripts
- Markdown formatting for better readability
- Live per-segment transcription progress
- Caching support for the Whisper model

## Prerequisites

- Python 3.10+ (developed against 3.14)
- NVIDIA GPU with CUDA support — optional; pass `--device cpu` to run without one
- On a CPU-only machine, prefer a smaller model:

  ```bash
  python extract-text.py --url "<url>" --model small --device cpu
  ```

- FFmpeg:
  - Windows: Download from [FFmpeg official website](https://ffmpeg.org/download.html#build-windows) or install via [Chocolatey](https://chocolatey.org/):

    ```bash
    choco install ffmpeg
    ```

  - macOS: Install via Homebrew:

    ```bash
    brew install ffmpeg
    ```

  - Linux (Ubuntu/Debian):

    ```bash
    sudo apt update
    sudo apt install ffmpeg
    ```

- DeepL API key (get one from [DeepL API page](https://www.deepl.com/pro-api))

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ahvega/yt-extract-audio.git
   cd yt-extract-audio
   ```

2. Create and activate a virtual environment:
   - Windows:

     ```bash
     python -m venv .venv
     .\.venv\Scripts\activate
     ```

   - macOS/Linux:

     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory:

   ```text
   DEEPL_API_KEY=your-api-key-goes-here
   ```

   Important notes for `.env` file:
   - Don't use quotes around the API key
   - Don't add spaces around the equals sign
   - Each variable should be on its own line
   - Get your DeepL API key from:
     1. [DeepL API page](https://www.deepl.com/pro-api)
     2. Create an account
     3. Choose free or pro plan
     4. Copy API key from account settings
   - The `.env` file is automatically ignored by Git to keep your API key private

## CUDA Support

### Verifying CUDA Support

1. Check if you have an NVIDIA GPU:
   - Windows:

     ```bash
     nvidia-smi
     ```

   - Linux:

     ```bash
     lspci | grep -i nvidia
     ```

2. The script will automatically check CUDA availability when running.

### How CUDA is wired here

All GPU compute runs through **CTranslate2** (the faster-whisper backend), not PyTorch.
`torch` is installed only because ctranslate2 needs the cuBLAS/cuDNN binaries bundled in
`torch/lib`; `cuda_dlls.py` puts that directory on the Windows DLL search path at import
time. **You do not need a separate CUDA Toolkit or cuDNN installation** — a current NVIDIA
driver plus `pip install -r requirements.txt` is sufficient.

This matters on older GPUs. On a Quadro P5000 (Pascal, sm_61) the pinned torch build
reports `torch.cuda.is_available() == True` but its kernels target sm_75 and newer, so
torch cannot actually compute on that card. CTranslate2 can, which is why both scripts use
it and why `int8` is the default compute type (Pascal's DP4A integer path is fast there,
while `fp16` is slow).

Verify what your GPU supports:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("torch kernel archs:", torch.cuda.get_arch_list())
print("your GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
```

Note: if CUDA is unavailable, `extract-text.py` falls back to CPU automatically
(significantly slower). `transcribe_local.py` does not — pass `cpu` as its device
argument explicitly.

## Usage

Run the script with a YouTube URL:

```bash
python extract-text.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

Options:

- `-u, --url`: YouTube video URL to process (**required**)
- `-m, --model`: Whisper model name (default: `large-v3`)
- `-d, --device`: `cuda` or `cpu` (default: `cuda`, falls back to CPU automatically)
- `-c, --compute`: Compute type (default: `int8`)
- `-l, --language`: Source language code such as `es` (default: auto-detect)
- `-o, --output`: Transcript output path (default: `transcription.md`)
- `--target-lang`: DeepL target language (default: `ES`)
- `--no-translate`: Skip the DeepL translation step

The translation is written alongside the transcript with the language appended —
`transcription.md` produces `transcription_es.md`. Translation is skipped with a
warning when `DEEPL_API_KEY` is unset; the transcript is still written.

Exit status is `0` on success, `2` for a usage error, and `1` if the download,
transcription, or a requested translation fails.

Examples:

```bash
# Transcript only, no translation
python extract-text.py -u "https://youtu.be/VIDEO_ID" --no-translate

# Force Spanish source, translate to English, custom output path
python extract-text.py -u "https://youtu.be/VIDEO_ID" -l es --target-lang EN -o charla.md
```

## Local File Transcription (transcribe_local.py)

Transcribes a local audio file using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend) with CUDA acceleration. Writes a timestamped text file and an optional SRT subtitle file, streaming progress to the console as each segment is decoded.

```bash
python transcribe_local.py <audio> <out.txt> [out.srt] [model] [device] [compute] [language]
```

Arguments (positional; only the first two are required):

- `audio`: Path to the local audio file to transcribe
- `out.txt`: Output text file with `[HH:MM:SS.mmm -> HH:MM:SS.mmm]` timestamps
- `out.srt`: Output SRT subtitle file (optional)
- `model`: Whisper model name (default: `large-v3`)
- `device`: `cuda` or `cpu` (default: `cuda` — no automatic CPU fallback)
- `compute`: Compute type (default: `int8`)
- `language`: Language code such as `es` or `en` (default: auto-detect)

Examples:

```bash
# Transcribe with defaults (large-v3, CUDA, int8, auto-detected language)
python transcribe_local.py interview.mp3 interview.txt

# Force Spanish and also produce subtitles
python transcribe_local.py charla.wav charla.txt charla.srt large-v3 cuda int8 es

# CPU-only machine (much slower; consider a smaller model)
python transcribe_local.py audio.m4a audio.txt "" small cpu int8
```

Notes:

- On Pascal GPUs (e.g. Quadro P5000, sm_61) keep `int8` compute — it uses fast DP4A integer kernels, while `fp16` is slow on that architecture.
- On Windows the script reuses the cuBLAS/cuDNN DLLs bundled with the installed `torch` package, so no separate cuDNN installation is needed.
- Transcription output files (`transcription*.md`, `*.srt`) and media files are gitignored — this is intentional for a public repository.
