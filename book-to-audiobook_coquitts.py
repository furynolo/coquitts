"""
Book to Audiobook Converter using CoquiTTS

This script converts text files to audiobook format using CoquiTTS.
The output filename matches the input filename.
"""

import os
import sys
import re
import argparse
import json
import unicodedata
from pathlib import Path
import platform
from datetime import datetime

try:
    from TTS.api import TTS
except ImportError:
    print("Error: CoquiTTS is not installed. Please run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. Audio chunks will not be combined.")
    print("  Install with: pip install pydub")
    print("  For now, each chunk will be saved separately.")

# Try to import audio playback libraries
PLAYBACK_AVAILABLE = False
PLAYBACK_METHOD = None
WINSOUND_AVAILABLE = False
PLAYSOUND_FUNC = None

# Try winsound (Windows built-in)
if platform.system() == "Windows":
    try:
        import winsound
        PLAYBACK_AVAILABLE = True
        PLAYBACK_METHOD = "winsound"
        WINSOUND_AVAILABLE = True
    except ImportError:
        pass

# Try playsound as fallback
if not PLAYBACK_AVAILABLE:
    try:
        from playsound import playsound as playsound_func
        PLAYBACK_AVAILABLE = True
        PLAYBACK_METHOD = "playsound"
        PLAYSOUND_FUNC = playsound_func
    except ImportError:
        pass

def generate_unique_filename(base_name, output_dir=None):
    """
    Generate a unique output filename with incrementing number and timestamp.
    Format: base_name_N_YYYYMMDD-HHMMSS.wav
    
    Args:
        base_name: Base name for the file (without extension)
        output_dir: Optional output directory (default: output-audio/)
        
    Returns:
        Path to unique output audio file
    """
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path("output-audio")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Find the next available number by checking existing files
    # Pattern: base_name_N_YYYYMMDD-HHMMSS.wav
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)_.*\.wav$")
    existing_numbers = []
    
    for file in output_path.glob(f"{base_name}_*.wav"):
        match = pattern.match(file.name)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    # Get the next number (start at 1 if no files exist)
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    
    # Generate unique filename
    output_file = output_path / f"{base_name}_{next_number}_{timestamp}.wav"
    
    return str(output_file)

def get_output_filename(input_file, output_dir=None):
    """
    Generate output filename based on input filename.
    Now includes incrementing number and timestamp.
    
    Args:
        input_file: Path to input text file
        output_dir: Optional output directory (default: output-audio/)
        
    Returns:
        Path to unique output audio file
    """
    input_path = Path(input_file)
    input_stem = input_path.stem  # filename without extension
    
    return generate_unique_filename(input_stem, output_dir)

def normalize_for_tts(text):
    """
    Normalize text for TTS processing.
    Normalizes Unicode compatibility forms and replaces fancy characters with ASCII equivalents.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text string
    """
    # Normalize Unicode compatibility forms (e.g., fancy characters)
    text = unicodedata.normalize("NFKC", text)
    
    # Replace curly quotes and dashes with ASCII equivalents
    replacements = {
        "'": "'",
        "'": "'",
        '"': '"',
        '"': '"',
        "—": "-",
        "–": "-",
        "…": "...",
        "\u00a0": " ",  # non-breaking space
    }
    
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    
    return text

def load_pronunciations(pronunciations_file):
    """
    Load pronunciation mappings from a JSON file.
    
    Args:
        pronunciations_file: Path to JSON file with pronunciation mappings
        
    Returns:
        Dictionary with pronunciation mappings, or None if file cannot be loaded
    """
    try:
        with open(pronunciations_file, 'r', encoding='utf-8') as f:
            pronunciations = json.load(f)
        
        if not isinstance(pronunciations, dict):
            print(f"Warning: Pronunciation file '{pronunciations_file}' does not contain a valid dictionary.")
            return None
        
        print(f"Loaded {len(pronunciations)} pronunciation mappings from '{pronunciations_file}'")
        return pronunciations
    except FileNotFoundError:
        print(f"Error: Pronunciation file '{pronunciations_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in pronunciation file '{pronunciations_file}': {e}")
        return None
    except Exception as e:
        print(f"Error loading pronunciation file '{pronunciations_file}': {e}")
        return None

def apply_pronunciations(text, pronunciations):
    """
    Apply pronunciation replacements to text.
    Replaces keys with their pronunciation values.
    Uses word boundaries to ensure whole-word matching.
    
    Args:
        text: Text to process
        pronunciations: Dictionary mapping original text to pronunciation
        
    Returns:
        Text with pronunciations applied
    """
    if not pronunciations:
        return text
    
    # Sort keys by length (longest first) to handle multi-word replacements correctly
    sorted_keys = sorted(pronunciations.keys(), key=len, reverse=True)
    
    for original in sorted_keys:
        replacement = pronunciations[original]
        # Use word boundaries to match whole words only
        # This prevents partial matches (e.g., "Azeroth" matching "Azerothian")
        pattern = r'\b' + re.escape(original) + r'\b'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def preprocess_text(text, is_short_input=False, pronunciations=None):
    """
    Preprocess text to handle special characters and normalize for TTS.
    
    Args:
        text: Raw text string
        is_short_input: If True, this is a short direct input that may need special handling
        pronunciations: Optional dictionary of pronunciation mappings to apply after normalization
        
    Returns:
        Preprocessed text string
    """
    # Step 1: Normalize Unicode and replace fancy characters with ASCII equivalents
    # This runs FIRST before pronunciations
    text = normalize_for_tts(text)
    
    # Step 2: Apply pronunciation replacements after normalization
    if pronunciations:
        text = apply_pronunciations(text, pronunciations)
    
    # Step 3: Replace asterisks used for scene breaks with newlines
    text = re.sub(r'\*{3,}', '\n\n', text)  # Multiple asterisks
    text = re.sub(r'^\*+$', '', text, flags=re.MULTILINE)  # Lines of only asterisks
    
    # Step 4: Normalize multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Step 5: Normalize multiple newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    text = text.strip()
    
    # Step 6: For very short inputs, add punctuation and ensure proper sentence structure
    # This helps prevent the model from getting stuck in infinite loops
    if is_short_input and len(text) < 20:
        # If it doesn't end with punctuation, add a period
        if not text.endswith(('.', '!', '?', ':', ';')):
            text = text + '.'
        # Ensure there's at least one space or word boundary to help the model
        # Some models get confused with very short inputs that look incomplete
        if len(text.split()) == 1 and not text.endswith('.'):
            text = text + '.'
    
    return text

def split_text_into_chunks(text, max_chunk_size=5000, min_chunk_size=100):
    """
    Split text into chunks for processing.
    Tries to split at sentence boundaries, then paragraph boundaries, then line boundaries.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk (to avoid kernel errors)
        
    Returns:
        List of text chunks
    """
    # If text is short enough, return as single chunk
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        # If paragraph is very long, try to split by sentences first
        if len(para) > max_chunk_size:
            sentences = re.split(r'([.!?]+\s+)', para)
            # If sentence splitting didn't work (only one element), try splitting by lines
            if len(sentences) <= 1:
                # Split by single newlines as fallback
                lines = para.split('\n')
                for line in lines:
                    if len(current_chunk) + len(line) + 1 > max_chunk_size and current_chunk:
                        if len(current_chunk) >= min_chunk_size:
                            chunks.append(current_chunk.strip())
                            current_chunk = line
                        else:
                            current_chunk += "\n" + line
                    else:
                        current_chunk += "\n" + line if current_chunk else line
            else:
                # Process sentence pairs
                sentence_pairs = []
                for i in range(0, len(sentences) - 1, 2):
                    if i + 1 < len(sentences):
                        sentence_pairs.append(sentences[i] + sentences[i + 1])
                    else:
                        sentence_pairs.append(sentences[i])
                
                for sentence in sentence_pairs:
                    if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                        if len(current_chunk) >= min_chunk_size:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            # If chunk is too small, append to it anyway
                            current_chunk += " " + sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
        else:
            # Check if adding this paragraph would exceed max size
            if len(current_chunk) + len(para) + 2 > max_chunk_size and current_chunk:
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Final safety check: if we somehow ended up with no chunks, split by character count
    if not chunks:
        # Force split by character count as last resort
        for i in range(0, len(text), max_chunk_size):
            chunk = text[i:i + max_chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
    
    return chunks if chunks else [text]  # Ensure we always return at least one chunk

def read_text_file(file_path):
    """
    Read text from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Text content as string, or None if file cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if not text:
            print(f"Warning: File '{file_path}' is empty.")
            return None
        return text
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None

def validate_audio_duration(file_path, text_length, max_duration_per_char=0.5):
    """
    Validate that the generated audio duration is reasonable for the text length.
    Prevents infinite loops or excessive generation.
    
    Args:
        file_path: Path to audio file
        text_length: Length of input text in characters
        max_duration_per_char: Maximum seconds per character (default: 0.5s)
        
    Returns:
        Tuple of (is_valid, duration_seconds, expected_max_duration)
    """
    if not PYDUB_AVAILABLE:
        # Can't validate without pydub, assume it's okay
        return True, None, None
    
    try:
        audio = AudioSegment.from_wav(str(file_path))
        duration_seconds = len(audio) / 1000.0  # pydub gives duration in milliseconds
        expected_max = text_length * max_duration_per_char
        
        # For very short texts, use a minimum expected duration
        if text_length < 10:
            expected_max = max(expected_max, 2.0)  # At least 2 seconds for very short texts
        
        is_valid = duration_seconds <= expected_max
        return is_valid, duration_seconds, expected_max
    except Exception as e:
        print(f"  Warning: Could not validate audio duration: {e}")
        return True, None, None  # Assume valid if we can't check

def play_audio(file_path):
    """
    Play an audio file using available playback method.
    
    Args:
        file_path: Path to audio file to play
        
    Returns:
        True if successful, False otherwise
    """
    if not PLAYBACK_AVAILABLE:
        print("Warning: No audio playback library available.")
        print("  On Windows, winsound should be available by default.")
        print("  Otherwise, install playsound: pip install playsound")
        return False
    
    try:
        if PLAYBACK_METHOD == "winsound" and WINSOUND_AVAILABLE:
            import winsound
            winsound.PlaySound(str(file_path), winsound.SND_FILENAME)
        elif PLAYBACK_METHOD == "playsound" and PLAYSOUND_FUNC:
            PLAYSOUND_FUNC(str(file_path))
        else:
            print("Error: Playback method not properly initialized.")
            return False
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def synthesize_text_chunk(tts, text, chunk_path):
    """
    Synthesize a single chunk of text.
    
    Args:
        tts: Initialized TTS model
        text: Text chunk to synthesize
        chunk_path: Path to save the chunk audio file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure minimum length to avoid kernel errors
        if len(text.strip()) < 50:
            print(f"  Warning: Chunk too short ({len(text)} chars), skipping...")
            return False
        
        tts.tts_to_file(text=text, file_path=chunk_path)
        return True
    except Exception as e:
        print(f"  Error synthesizing chunk: {e}")
        return False

def synthesize_text(text, output_path, model_name=None, chunk_size=5000, is_short_input=False, pronunciations=None):
    """
    Synthesize text to speech, handling long texts by chunking.
    
    Args:
        text: Text to synthesize
        output_path: Path to save the output audio file
        model_name: Optional model name. If None, uses a recommended default model.
        chunk_size: Maximum characters per chunk (default: 5000)
        is_short_input: If True, indicates this is a short direct text input
        pronunciations: Optional dictionary of pronunciation mappings to apply
    """
    try:
        # Preprocess text (with special handling for short inputs and pronunciations)
        original_text = text
        text = preprocess_text(text, is_short_input=is_short_input, pronunciations=pronunciations)
        
        # Initialize the TTS model
        if model_name:
            print(f"Loading model: {model_name}")
            tts = TTS(model_name=model_name, progress_bar=True)
        else:
            # Use a recommended default model for English TTS
            default_model = "tts_models/en/ljspeech/tacotron2-DDC"
            print(f"Loading default model: {default_model}")
            tts = TTS(model_name=default_model, progress_bar=True)
        
        # Show text info
        print(f"Text length: {len(text)} characters")
        
        # Split text into chunks if necessary
        chunks = split_text_into_chunks(text, max_chunk_size=chunk_size)
        
        if len(chunks) == 1:
            # Single chunk - process directly
            print(f"Synthesizing text...")
            tts.tts_to_file(text=text, file_path=output_path)
            
            # Validate audio duration to detect infinite loops
            is_valid, duration, max_expected = validate_audio_duration(output_path, len(text))
            if not is_valid:
                print(f"⚠ Warning: Generated audio is unusually long!")
                print(f"  Duration: {duration:.2f}s (expected max: {max_expected:.2f}s)")
                print(f"  This may indicate the TTS model got stuck in a loop.")
                
                # If audio is way too long (10x expected), automatically truncate it
                if duration is not None and max_expected is not None and duration > max_expected * 10:
                    if PYDUB_AVAILABLE:
                        print(f"  Auto-truncating audio to {max_expected * 2:.2f}s (safety limit)...")
                        try:
                            audio = AudioSegment.from_wav(str(output_path))
                            max_duration_ms = int(max_expected * 2 * 1000)  # 2x expected as safety limit
                            truncated = audio[:max_duration_ms]
                            truncated.export(output_path, format="wav")
                            print(f"  ✓ Audio truncated successfully.")
                        except Exception as e:
                            print(f"  ✗ Could not truncate audio: {e}")
                    else:
                        print(f"  Install pydub to enable automatic truncation: pip install pydub")
                
                print(f"  Tip: Try adding punctuation, using a different model, or rephrasing the text.")
            elif duration is not None:
                print(f"  Audio duration: {duration:.2f}s")
            
            print(f"✓ Speech generated successfully!")
            print(f"  Output saved to: {output_path}")
            return True
        else:
            # Multiple chunks - process and combine
            print(f"Splitting into {len(chunks)} chunks for processing...")
            
            output_dir = Path(output_path).parent
            output_stem = Path(output_path).stem
            chunk_files = []
            
            for i, chunk in enumerate(chunks, 1):
                chunk_path = output_dir / f"{output_stem}_chunk_{i:04d}.wav"
                print(f"  Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)...")
                
                if synthesize_text_chunk(tts, chunk, str(chunk_path)):
                    chunk_files.append(chunk_path)
                else:
                    print(f"  ✗ Failed to process chunk {i}")
            
            if not chunk_files:
                print("✗ All chunks failed to process.")
                return False
            
            # Combine audio files if pydub is available
            if PYDUB_AVAILABLE and len(chunk_files) > 1:
                print(f"\nCombining {len(chunk_files)} audio chunks...")
                combined = AudioSegment.empty()
                
                for chunk_file in chunk_files:
                    audio = AudioSegment.from_wav(str(chunk_file))
                    combined += audio
                    # Add small pause between chunks
                    combined += AudioSegment.silent(duration=500)  # 500ms pause
                
                combined.export(output_path, format="wav")
                print(f"✓ Combined audio saved to: {output_path}")
                
                # Clean up chunk files
                for chunk_file in chunk_files:
                    chunk_file.unlink()
                print(f"  Cleaned up {len(chunk_files)} temporary chunk files.")
            elif len(chunk_files) == 1:
                # Only one chunk succeeded, just rename it
                chunk_files[0].rename(output_path)
                print(f"✓ Audio saved to: {output_path}")
            else:
                print(f"✓ Processed {len(chunk_files)} chunks (not combined - install pydub to combine)")
                print(f"  Chunk files saved in: {output_dir}")
            
            return True
        
    except Exception as e:
        print(f"Error during synthesis: {e}")
        print("\nTip: Try specifying a different model or check available models with scripts/list_models.py")
        return False

def main():
    """Main function for book to audiobook conversion."""
    parser = argparse.ArgumentParser(
        description="Book to Audiobook Converter using CoquiTTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a text file:
  python book-to-audiobook_coquitts.py input-text/mybook.txt
  
  # Convert text directly:
  python book-to-audiobook_coquitts.py --text "Az-er-oth"
  
  # Convert and play audio:
  python book-to-audiobook_coquitts.py --text "Az-er-oth" --play
  
  # Use pronunciations file:
  python book-to-audiobook_coquitts.py input-text/mybook.txt --pronunciations pronunciations.json
  
  # Use a specific model:
  python book-to-audiobook_coquitts.py --text "Hello world" --model tts_models/en/ljspeech/tacotron2-DDC
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        'input_file',
        nargs='?',
        help='Path to input text file'
    )
    input_group.add_argument(
        '--text', '-t',
        type=str,
        help='Text string to convert directly (e.g., "Az-er-oth")'
    )
    
    # Other options
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='TTS model name (optional)'
    )
    parser.add_argument(
        '--play', '-p',
        action='store_true',
        help='Play the generated audio file after conversion'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (optional, defaults to output-audio/<name>.wav)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=5000,
        help='Maximum characters per chunk (default: 5000)'
    )
    parser.add_argument(
        '--pronunciations', '-pr',
        type=str,
        help='Path to JSON file with pronunciation mappings (keys: original text, values: pronunciation)'
    )
    
    args = parser.parse_args()
    
    print("Book to Audiobook Converter (CoquiTTS)\n" + "="*50)
    
    # Determine input source
    text = None
    input_source = None
    
    if args.text:
        # Text provided directly
        text = args.text
        input_source = "direct text input"
        print(f"Input: Direct text input")
    elif args.input_file:
        # File path provided
        input_file = args.input_file
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)
        input_source = input_file
        print(f"Input file: {input_file}")
        text = read_text_file(input_file)
        if text is None:
            print("Error: Could not read text from file.")
            sys.exit(1)
    else:
        # No input provided, try default file
        default_file = "input-text/example-test-passage.txt"
        if os.path.exists(default_file):
            input_file = default_file
            input_source = input_file
            print(f"Input file: {input_file}")
            text = read_text_file(input_file)
            if text is None:
                print("Error: Could not read text from file.")
                sys.exit(1)
        else:
            parser.print_help()
            sys.exit(1)
    
    # Generate output filename
    if args.output:
        output_file = args.output
    elif args.text:
        # For direct text, use a default name with unique numbering and timestamp
        # Create a safe filename from text (first 30 chars, sanitized)
        safe_name = re.sub(r'[^\w\s-]', '', args.text[:30]).strip().replace(' ', '_')
        # If safe_name is empty after sanitization, use a default
        if not safe_name:
            safe_name = "text_output"
        output_file = generate_unique_filename(safe_name)
    else:
        output_file = get_output_filename(input_source)
    
    print(f"Output file: {output_file}\n")
    
    if args.model:
        print(f"Using model: {args.model}\n")
    
    # Load pronunciations if provided
    pronunciations = None
    if args.pronunciations:
        pronunciations = load_pronunciations(args.pronunciations)
        if pronunciations is None:
            print("Warning: Could not load pronunciations file. Continuing without pronunciation replacements.")
        print()  # Add blank line for readability
    
    # Synthesize text to speech
    # Mark as short input if text was provided directly (likely a single word/phrase)
    is_short = args.text is not None
    success = synthesize_text(
        text, 
        output_file, 
        model_name=args.model,
        chunk_size=args.chunk_size,
        is_short_input=is_short,
        pronunciations=pronunciations
    )
    
    if success:
        print(f"\n{'='*50}")
        print(f"✓ Conversion complete!")
        print(f"  Input:  {input_source}")
        print(f"  Output: {output_file}")
        
        # Play audio if requested
        if args.play:
            print(f"\nPlaying audio...")
            if play_audio(output_file):
                print("✓ Audio playback complete.")
            else:
                print("✗ Audio playback failed.")
    else:
        print(f"\n{'='*50}")
        print("✗ Conversion failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

