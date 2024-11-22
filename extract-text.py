import os
from dotenv import load_dotenv
import whisper
from yt_dlp import YoutubeDL
from tqdm import tqdm
import time
import deepl

def create_temp_dir():
    if not os.path.exists("temp"):
        os.makedirs("temp")

def create_cache_dir():
    cache_dir = os.path.join(os.getcwd(), "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def download_video(url):
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
    """Convert plain text to formatted markdown with topic-based headings"""
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

def main():
    create_temp_dir()
    cache_dir = create_cache_dir()
    current_dir = os.getcwd()
    transcription_path = os.path.join(current_dir, "transcription.txt")
    spanish_path = os.path.join(current_dir, "transcription_spanish.txt")
    markdown_path = os.path.join(current_dir, "transcription_spanish.md")
    
    # Check if Spanish translation already exists
    if os.path.exists(spanish_path):
        print("\nExisting Spanish translation found. Creating markdown version...")
        with open(spanish_path, 'r', encoding='utf-8') as f:
            spanish_text = f.read()
        
        # Create markdown version
        print("\nCreating formatted markdown version...")
        markdown_content = format_to_markdown(spanish_text)
        
        with tqdm(total=100,
                 bar_format='Formatting: [{bar:50}] {percentage:3.1f}%',
                 desc="4. Markdown") as pbar:
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            for i in range(100):
                time.sleep(0.01)
                pbar.update(1)
        
        print(f"\nFormatted markdown version saved to {markdown_path}")
        return

    # Check if transcription exists
    if os.path.exists(transcription_path):
        print("\nExisting transcription found. Would you like to use it?")
        while True:
            reuse = input("Use existing transcription? (y/n): ").lower()
            if reuse in ['y', 'n']:
                break
            print("Please enter 'y' for yes or 'n' for no.")
        
        if reuse == 'y':
            print("Loading existing transcription...")
            with open(transcription_path, 'r', encoding='utf-8') as f:
                text = f.read()
                print("\nTranscription preview (first 500 characters):")
                print(text[:500] + "...")
                
            # Skip to translation step
            while True:
                translate = input("\nWould you like to translate this to Spanish? (y/n): ").lower()
                if translate in ['y', 'n']:
                    break
                print("Please enter 'y' for yes or 'n' for no.")
            
            if translate == 'y':
                print("\n3. Translation to Spanish:")
                spanish_text = translate_text(text)
                
                if spanish_text:
                    print("\nSaving translation...")
                    with tqdm(total=100,
                             bar_format='Saving: [{bar:50}] {percentage:3.1f}%',
                             desc="3. Saving") as pbar:
                        with open(spanish_path, "w", encoding="utf-8") as f:
                            f.write(spanish_text)
                        for i in range(100):
                            time.sleep(0.01)
                            pbar.update(1)
                    
                    # Create markdown version
                    print("\nCreating formatted markdown version...")
                    markdown_content = format_to_markdown(spanish_text)
                    
                    with tqdm(total=100,
                             bar_format='Formatting: [{bar:50}] {percentage:3.1f}%',
                             desc="4. Markdown") as pbar:
                        with open(markdown_path, "w", encoding="utf-8") as f:
                            f.write(markdown_content)
                        for i in range(100):
                            time.sleep(0.01)
                            pbar.update(1)
                    
                    print("\nSpanish translation preview (first 500 characters):")
                    print(spanish_text[:500] + "...")
                    print(f"\nFull Spanish translation saved to {spanish_path}")
                    print(f"Formatted markdown version saved to {markdown_path}")
                else:
                    print("Translation failed.")
            return

    url = "https://www.youtube.com/watch?v=1F3siwJT6GM"
    print("Starting download...")
    audio_file = download_video(url)
    print(f"Debug - File exists after download: {os.path.exists('temp/audio.wav')}")
    
    if not audio_file:
        return
    
    try:
        print("\n2. Audio Transcription:")
        # Load the Whisper large model with CUDA device and caching
        if not os.path.exists(os.path.join(cache_dir, "large")):
            print("Downloading model (first time only)...")
        
        with tqdm(total=100, 
                 bar_format='Loading Model: [{bar:50}] {percentage:3.1f}%',
                 desc="2. Model Loading") as pbar:
            model = whisper.load_model("large", download_root=cache_dir).cuda()
            for i in range(100):
                time.sleep(0.01)
                pbar.update(1)
        
        print("\nTranscribing with Whisper using CUDA...")
        with tqdm(total=100,
                 bar_format='Transcribing: [{bar:50}] {percentage:3.1f}%',
                 desc="2. Processing") as pbar:
            result = model.transcribe("temp/audio.wav", fp16=True)
            for i in range(100):
                time.sleep(0.05)
                pbar.update(1)
        
        print(f"\nSaving transcription to {transcription_path}...")
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        print("\nTranscription preview (first 500 characters):")
        print(result["text"][:500] + "...")
        print(f"\nFull transcription saved to {transcription_path}")
        
        while True:
            translate = input("\nWould you like to translate this to Spanish? (y/n): ").lower()
            if translate in ['y', 'n']:
                break
            print("Please enter 'y' for yes or 'n' for no.")
        
        if translate == 'y':
            print("\n3. Translation to Spanish:")
            spanish_text = translate_text(result["text"])
            
            if spanish_text:
                print("\nSaving translation...")
                with tqdm(total=100,
                         bar_format='Saving: [{bar:50}] {percentage:3.1f}%',
                         desc="3. Saving") as pbar:
                    with open(spanish_path, "w", encoding="utf-8") as f:
                        f.write(spanish_text)
                    for i in range(100):
                        time.sleep(0.01)
                        pbar.update(1)
                
                # Create markdown version
                print("\nCreating formatted markdown version...")
                markdown_content = format_to_markdown(spanish_text)
                
                with tqdm(total=100,
                         bar_format='Formatting: [{bar:50}] {percentage:3.1f}%',
                         desc="4. Markdown") as pbar:
                    with open(markdown_path, "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                    for i in range(100):
                        time.sleep(0.01)
                        pbar.update(1)
                
                print("\nSpanish translation preview (first 500 characters):")
                print(spanish_text[:500] + "...")
                print(f"\nFull Spanish translation saved to {spanish_path}")
                print(f"Formatted markdown version saved to {markdown_path}")
            else:
                print("Translation failed.")

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        print(f"File exists check: {os.path.exists('temp/audio.wav')}")
        
    finally:
        print("\nCleaning up temporary files...")
        if os.path.exists("temp/audio.wav"):
            os.remove("temp/audio.wav")

if __name__ == "__main__":
    main() 