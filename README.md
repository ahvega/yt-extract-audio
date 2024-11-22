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

- FFmpeg installed and in PATH
- DeepL API key

## Installation

1. Clone the repository: 
   ```bash
   git clone https://github.com/ahvega/youtube-transcription-tool.git
   cd youtube-transcription-tool
   ```

