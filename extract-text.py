import os
from dotenv import load_dotenv
import whisper
from yt_dlp import YoutubeDL
from tqdm import tqdm
import time
import deepl
import torch
import sys
import argparse

def create_temp_dir():
    """
    Creates a temporary directory to store downloaded audio files.
    The directory will be created if it doesn't exist already.
    """
    if not os.path.exists("temp"):
        os.makedirs("temp")

def create_cache_dir():
    """
    Creates a cache directory for storing Whisper model files.
    
    Returns:
        str: Path to the cache directory
    """
    cache_dir = os.path.join(os.getcwd(), "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def download_video(url):
    """
    Downloads audio from a YouTube video URL using yt-dlp.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Path to the downloaded audio file (.wav format)
        None: If download fails
    """
    try:
        create_temp_dir()
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp/audio',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            },
            'progress_hooks': [download_progress_hook],
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            print("\n1. File Downloading:")
            info = ydl.extract_info(url, download=True)
            print(f"Successfully extracted info for: {info.get('title', 'Unknown Title')}")
            
        return "temp/audio.wav"
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None

def download_progress_hook(d):
    """
    Progress hook callback for yt-dlp to display download progress.
    Creates a visual progress bar in the console.
    
    Args:
        d (dict): Dictionary containing download status information
    """
    if d['status'] == 'downloading':
        total_bytes = d.get('total_bytes')
        downloaded_bytes = d.get('downloaded_bytes', 0)
        if total_bytes:
            percentage = (downloaded_bytes / total_bytes) * 100
            progress = int(percentage / 2)
            bar = f"[{'#' * progress}{'.' * (50-progress)}] {percentage:.1f}%"
            print(f"\r1. File Downloading: {bar}", end='')
    elif d['status'] == 'finished':
        print("\nDownload completed")

def format_to_markdown(text):
    """
    Converts plain text to formatted markdown with topic-based headings.
    
    The function identifies potential headings based on:
    - Questions (sentences ending with '?')
    - Key topic indicators (monetizar, marketing, etc.)
    - Short statements ending with colons
    - Numbered points
    
    Args:
        text (str): Plain text to be formatted
        
    Returns:
        str: Formatted markdown text with headers and paragraphs
    """
    # Split text into paragraphs
    paragraphs = text.split('. ')
    
    # Initialize markdown content
    md_content = "# Transcripción\n\n"
    
    # Keywords that might indicate main topics
    topic_indicators = [
        'manera', 'monetizar', 'marketing', 'afiliación', 'libros', 
        'productos', 'patreon', 'patrocinios', 'crowdfunding', 
        'coaching', 'mentalidad', 'consejos', 'conclusión'
    ]
    
    current_paragraphs = []
    
    for paragraph in paragraphs:
        # Clean up the paragraph
        p = paragraph.strip()
        if not p:
            continue
            
        # Detect if paragraph might be a heading
        is_heading = (
            ('?' in p and len(p) < 100) or  # Questions
            any(f" {indicator} " in p.lower() for indicator in topic_indicators) or  # Topic keywords
            (len(p) < 80 and p.endswith(':')) or  # Short statements ending with colon
            ('número' in p.lower() and len(p) < 100)  # Numbered points
        )
        
        if is_heading:
            # Add previous paragraphs if any
            if current_paragraphs:
                md_content += '\n'.join(current_paragraphs) + '\n\n'
                current_paragraphs = []
            
            # Add as heading
            md_content += f"\n## {p}\n\n"
        else:
            # Add as regular paragraph
            current_paragraphs.append(p + '.')
            
            # Every few paragraphs, add a paragraph break
            if len(current_paragraphs) >= 4:
                md_content += '\n'.join(current_paragraphs) + '\n\n'
                current_paragraphs = []
    
    # Add any remaining paragraphs
    if current_paragraphs:
        md_content += '\n'.join(current_paragraphs) + '\n'
    
    return md_content

def translate_text(text, target_language='ES'):
    """
    Translates text using DeepL API.
    
    Requires DEEPL_API_KEY in .env file.
    Uses the free tier API endpoint by default.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code (default: 'ES' for Spanish)
        
    Returns:
        str: Translated text
        None: If translation fails
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key and verify it's loaded
    api_key = os.getenv('DEEPL_API_KEY')
    if not api_key:
        print("Error: DEEPL_API_KEY not found in .env file")
        return None
        
    print(f"Debug - API Key loaded: {api_key[:8]}...")
    
    try:
        # Initialize DeepL translator with free API
        translator = deepl.Translator(api_key, server_url="https://api-free.deepl.com")
        
        # Create a progress bar for translation
        print("\nTranslating with DeepL...")
        with tqdm(total=100, 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                 desc="3. Translation") as pbar:
            # Start translation
            result = translator.translate_text(text, target_lang=target_language)
            
            # Simulate progress
            for i in range(100):
                time.sleep(0.01)
                pbar.update(1)
                
        return result.text
        
    except deepl.exceptions.AuthorizationException:
        print("Error: Invalid DeepL API key. Please check your .env file.")
        print("Note: Make sure you're using an API key for the free tier (should end with ':fx')")
        return None
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return None

def check_cuda_availability():
    """
    Checks if CUDA is available and prints relevant information.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"\nCUDA is available. Using GPU: {device_name}")
        else:
            print("\nCUDA is not available. Using CPU. This will be significantly slower.")
            print("If you have an NVIDIA GPU, make sure you have installed the CUDA toolkit and cuDNN.")
        return cuda_available
    except Exception as e:
        print(f"\nError checking CUDA availability: {str(e)}")
        return False

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        str: YouTube video URL or default test URL if none provided
    """
    parser = argparse.ArgumentParser(description='YouTube video transcription and translation tool')
    parser.add_argument('--url', '-u', type=str, 
                       help='YouTube video URL to process',
                       default="https://www.youtube.com/watch?v=1F3siwJT6GM")  # Short test video
    
    args = parser.parse_args()
    return args.url

def main():
    """
    Main execution function that handles the workflow.
    Now accepts command line parameter for video URL.
    """
    create_temp_dir()
    cache_dir = create_cache_dir()
    
    # Get video URL from command line or use default
    url = parse_arguments()
    if url == "https://www.youtube.com/watch?v=1F3siwJT6GM":
        print("\nNo URL provided. Using sample video for testing...")
    print(f"Processing video: {url}")
    
    # Rest of your existing main() function...
    print("Starting download...")
    audio_file = download_video(url)
    # ... rest of the code ...

if __name__ == "__main__":
    main() 