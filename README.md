# YouTube Transcription & Translation Tool

A Python script that downloads YouTube videos, transcribes them using Whisper (with CUDA support), and optionally translates them to Spanish using DeepL API.

## Features

- Downloads YouTube video audio using yt-dlp
- Transcribes audio using OpenAI's Whisper (large model)
- CUDA GPU acceleration support for faster transcription
- Translation to Spanish using DeepL API
- Markdown formatting for better readability
- Progress bars for all operations
- Caching support for the Whisper model
- Reuse of existing transcriptions and translations

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- If CUDA is not available, you can use the small model instead:
  ```python
  # Instead of:
  model = whisper.load_model("large", device="cuda", download_root=cache_dir, fp16=True)
  
  # Use:
  model = whisper.load_model("small", download_root=cache_dir)
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

