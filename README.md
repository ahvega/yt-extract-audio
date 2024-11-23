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

2. Create and activate a virtual environment:
   - Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     python -m venv venv
     source venv/bin/activate
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

### Setting up CUDA (if not detected)

If you have an NVIDIA GPU but CUDA is not being detected, follow these steps:

1. Install NVIDIA GPU drivers from [NVIDIA's website](https://www.nvidia.com/download/index.aspx)

2. Install CUDA Toolkit:
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
   - Ensure version compatibility with PyTorch

3. Install cuDNN:
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Follow installation instructions for your OS

4. Install PyTorch with CUDA support:
   ```bash
   # For CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

5. Verify installation:
   ```python
   import torch
   print("CUDA available:", torch.cuda.is_available())
   print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
   ```

Note: If CUDA is not available, the script will automatically fall back to CPU processing, which will be significantly slower for transcription tasks.

