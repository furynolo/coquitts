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
    
    # Replace curly quotes, apostrophes, dashes, and other special characters with ASCII equivalents
    # Comprehensive list of apostrophe-like characters
    replacements = {
        # Apostrophes and single quotes
        "'": "'",  # U+2019 RIGHT SINGLE QUOTATION MARK
        "'": "'",  # U+2018 LEFT SINGLE QUOTATION MARK
        "'": "'",  # U+201B SINGLE HIGH-REVERSED-9 QUOTATION MARK
        "'": "'",  # U+02BC MODIFIER LETTER APOSTROPHE
        "'": "'",  # U+02BB MODIFIER LETTER TURNED COMMA
        "'": "'",  # U+201A SINGLE LOW-9 QUOTATION MARK
        # Double quotes
        '"': '"',  # U+201C LEFT DOUBLE QUOTATION MARK
        '"': '"',  # U+201D RIGHT DOUBLE QUOTATION MARK
        '"': '"',  # U+201E DOUBLE LOW-9 QUOTATION MARK
        '"': '"',  # U+201F DOUBLE HIGH-REVERSED-9 QUOTATION MARK
        '"': '"',  # U+2E42 DOUBLE LOW-REVERSED-9 QUOTATION MARK
        # Dashes and hyphens
        "—": "-",  # U+2014 EM DASH
        "–": "-",  # U+2013 EN DASH
        "―": "-",  # U+2015 HORIZONTAL BAR
        "‐": "-",  # U+2010 HYPHEN (non-breaking hyphen)
        # Ellipsis
        "…": "...",  # U+2026 HORIZONTAL ELLIPSIS
        # Spaces
        "\u00a0": " ",  # U+00A0 NON-BREAKING SPACE
        "\u2000": " ",  # U+2000 EN QUAD
        "\u2001": " ",  # U+2001 EM QUAD
        "\u2002": " ",  # U+2002 EN SPACE
        "\u2003": " ",  # U+2003 EM SPACE
        "\u2004": " ",  # U+2004 THREE-PER-EM SPACE
        "\u2005": " ",  # U+2005 FOUR-PER-EM SPACE
        "\u2006": " ",  # U+2006 SIX-PER-EM SPACE
        "\u2007": " ",  # U+2007 FIGURE SPACE
        "\u2008": " ",  # U+2008 PUNCTUATION SPACE
        "\u2009": " ",  # U+2009 THIN SPACE
        "\u200a": " ",  # U+200A HAIR SPACE
        "\u202f": " ",  # U+202F NARROW NO-BREAK SPACE
        "\u205f": " ",  # U+205F MEDIUM MATHEMATICAL SPACE
        "\u3000": " ",  # U+3000 IDEOGRAPHIC SPACE
    }
    
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    
    # Final pass: catch any remaining apostrophe-like characters that might have slipped through
    # This uses a regex to find any remaining quotation mark or apostrophe-like characters
    # and replaces them with standard apostrophe
    apostrophe_pattern = re.compile(r'[\u2018\u2019\u201A\u201B\u02BC\u02BB]')
    text = apostrophe_pattern.sub("'", text)
    
    # Catch any remaining double quote variants
    quote_pattern = re.compile(r'[\u201C\u201D\u201E\u201F\u2E42]')
    text = quote_pattern.sub('"', text)
    
    # Catch any remaining hyphen/dash variants
    dash_pattern = re.compile(r'[\u2010\u2011\u2012\u2013\u2014\u2015]')
    text = dash_pattern.sub('-', text)
    
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
    
    # Step 3: Remove all asterisks (they're not in TTS vocabulary)
    # First, handle multiple asterisks (scene breaks) - convert to paragraph breaks
    text = re.sub(r'\*{3,}', '\n\n', text)  # Multiple asterisks
    text = re.sub(r'^\*+$', '', text, flags=re.MULTILINE)  # Lines of only asterisks
    # Remove ALL remaining asterisks (they cause TTS errors)
    text = text.replace('*', '')
    
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
    Always splits at sentence boundaries to ensure natural breaks.
    
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
    
    # Split text into sentences
    # Pattern matches: sentence text + sentence ending punctuation + optional whitespace
    # This preserves the punctuation and spacing
    sentence_pattern = r'([^.!?]+[.!?]+(?:\s+|$))'
    sentence_matches = re.finditer(sentence_pattern, text)
    
    sentences = []
    for match in sentence_matches:
        sentence = match.group(1).strip()
        if sentence:  # Only add non-empty sentences
            sentences.append(sentence)
    
    # If no sentences found (unlikely but possible), fall back to paragraph splitting
    if not sentences:
        # Fall back to paragraph-based splitting
        paragraphs = text.split('\n\n')
        sentences = [p.strip() for p in paragraphs if p.strip()]
    
    # If still no sentences, fall back to line splitting
    if not sentences:
        lines = text.split('\n')
        sentences = [l.strip() for l in lines if l.strip()]
    
    # Build chunks by adding sentences until we approach max_chunk_size
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Calculate what the chunk would be if we add this sentence
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # If adding this sentence would exceed max size (with some buffer for safety)
        # and we have enough content, start a new chunk
        if len(potential_chunk) > max_chunk_size and current_chunk:
            # Check if current chunk meets minimum size requirement
            if len(current_chunk.strip()) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Current chunk is too small, add sentence anyway
                # This prevents very small chunks (they'll be merged later if needed)
                current_chunk = potential_chunk
        else:
            # Add sentence to current chunk
            current_chunk = potential_chunk
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Final safety check: if we somehow ended up with no chunks, split by character count
    # This should rarely happen, but ensures we always return something
    if not chunks:
        # Force split by character count as last resort (but try to break at spaces)
        words = text.split()
        current_chunk = ""
        for word in words:
            potential_chunk = current_chunk + " " + word if current_chunk else word
            if len(potential_chunk) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk = potential_chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
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
        # Clean and validate text
        text = text.strip()
        
        # Count words (split by whitespace and filter out empty strings)
        words = [w for w in text.split() if w.strip()]
        word_count = len(words)
        
        # TTS models need a minimum number of words to process properly
        # Kernel size errors often occur when there aren't enough tokens
        if word_count < 10:
            print(f"  Warning: Chunk has very few words ({word_count} words, {len(text)} chars), may fail...")
        
        # Ensure minimum length to avoid kernel errors
        if len(text) < 100:
            print(f"  Warning: Chunk too short ({len(text)} chars), attempting anyway...")
        
        # Check if text has enough actual content (not just whitespace/punctuation)
        alphanumeric_count = sum(1 for c in text if c.isalnum())
        if alphanumeric_count < 20:
            print(f"  Warning: Chunk has little content ({alphanumeric_count} alphanumeric chars), attempting anyway...")
        
        tts.tts_to_file(text=text, file_path=chunk_path)
        return True
    except Exception as e:
        error_msg = str(e).lower()
        print(f"  Error synthesizing chunk: {e}")
        
        # Kernel size errors indicate insufficient tokenizable content
        if "kernel size" in error_msg:
            words = [w for w in text.split() if w.strip()]
            word_count = len(words)
            print(f"  Note: Chunk has insufficient content for TTS model ({word_count} words, {len(text)} chars)")
            print(f"  This chunk will be merged with adjacent chunks and retried.")
        
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
                print(f"[WARNING] Generated audio is unusually long!")
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
                            print(f"  [OK] Audio truncated successfully.")
                        except Exception as e:
                            print(f"  [ERROR] Could not truncate audio: {e}")
                    else:
                        print(f"  Install pydub to enable automatic truncation: pip install pydub")
                
                print(f"  Tip: Try adding punctuation, using a different model, or rephrasing the text.")
            elif duration is not None:
                print(f"  Audio duration: {duration:.2f}s")
            
            print(f"[OK] Speech generated successfully!")
            print(f"  Output saved to: {output_path}")
            return True
        else:
            # Multiple chunks - process and combine
            print(f"Splitting into {len(chunks)} chunks for processing...")
            
            # Merge small chunks with adjacent chunks to avoid losing content
            # Check both character count and word count for better validation
            merged_chunks = []
            i = 0
            while i < len(chunks):
                chunk = chunks[i]
                chunk_len = len(chunk.strip())
                words = [w for w in chunk.split() if w.strip()]
                word_count = len(words)
                alphanumeric_count = sum(1 for c in chunk if c.isalnum())
                
                # If chunk is too small (by chars, words, or content), merge with adjacent chunks
                # TTS models need sufficient words/tokens, not just characters
                if chunk_len < 100 or word_count < 10 or alphanumeric_count < 20:
                    merged = chunk
                    j = i + 1
                    
                    # Keep merging with next chunks until we have enough content
                    while j < len(chunks):
                        next_chunk = chunks[j]
                        test_merge = merged + " " + next_chunk
                        test_len = len(test_merge.strip())
                        test_words = [w for w in test_merge.split() if w.strip()]
                        test_word_count = len(test_words)
                        test_alnum = sum(1 for c in test_merge if c.isalnum())
                        
                        # If merged chunk is still too small, continue merging
                        if test_len < 100 or test_word_count < 10 or test_alnum < 20:
                            merged = test_merge
                            j += 1
                        else:
                            # Merged chunk is now large enough
                            break
                    
                    # If we still don't have enough and there's a previous chunk, merge backwards
                    merged_words = [w for w in merged.split() if w.strip()]
                    merged_word_count = len(merged_words)
                    if (len(merged.strip()) < 100 or merged_word_count < 10 or sum(1 for c in merged if c.isalnum()) < 20) and len(merged_chunks) > 0:
                        merged_chunks[-1] = merged_chunks[-1] + " " + merged
                    else:
                        merged_chunks.append(merged)
                    
                    i = j  # Move to the next unprocessed chunk
                else:
                    # Chunk is large enough, use as-is
                    merged_chunks.append(chunk)
                    i += 1
            
            if len(merged_chunks) != len(chunks):
                print(f"  Merged small chunks: {len(chunks)} -> {len(merged_chunks)} chunks")
            
            output_dir = Path(output_path).parent
            output_stem = Path(output_path).stem
            chunk_files_dict = {}  # Dict of index -> path to maintain order
            failed_indices = set()  # Track which chunk indices failed
            
            # First pass: try to process all chunks
            for i, chunk in enumerate(merged_chunks, 1):
                chunk_path = output_dir / f"{output_stem}_chunk_{i:04d}.wav"
                words = [w for w in chunk.split() if w.strip()]
                word_count = len(words)
                print(f"  Processing chunk {i}/{len(merged_chunks)} ({word_count} words, {len(chunk)} chars)...")
                
                # Chunks are already preprocessed, so use them directly
                if synthesize_text_chunk(tts, chunk, str(chunk_path)):
                    chunk_files_dict[i] = chunk_path
                else:
                    print(f"  [ERROR] Failed to process chunk {i}")
                    failed_indices.add(i)
            
            # Second pass: retry failed chunks by merging with adjacent chunks
            if failed_indices:
                print(f"\n  Retrying {len(failed_indices)} failed chunks by merging with adjacent content...")
                retry_attempts = {}
                processed_indices = set()  # Track which chunks we've already processed in retries
                
                for failed_idx in sorted(failed_indices):
                    if failed_idx in processed_indices:
                        continue  # Skip if already processed as part of another merge
                    
                    failed_chunk = merged_chunks[failed_idx - 1]  # Convert to 0-based
                    retry_path = output_dir / f"{output_stem}_retry_{failed_idx:04d}.wav"
                    merged_retry = failed_chunk
                    merged_indices = {failed_idx}
                    
                    # Try merging with next chunks first (more aggressive)
                    j = failed_idx
                    while j < len(merged_chunks) and len(merged_indices) < 3:  # Merge up to 3 chunks
                        if j < len(merged_chunks):
                            next_chunk = merged_chunks[j]  # Next chunk (0-based)
                            merged_retry = merged_retry + " " + next_chunk
                            merged_indices.add(j + 1)  # j+1 is 1-based index
                            j += 1
                            
                            words = [w for w in merged_retry.split() if w.strip()]
                            word_count = len(words)
                            print(f"    Retrying chunk {failed_idx} merged with {len(merged_indices)-1} next chunk(s) ({word_count} words, {len(merged_retry)} chars)...")
                            
                            if synthesize_text_chunk(tts, merged_retry, str(retry_path)):
                                retry_attempts[failed_idx] = retry_path
                                print(f"    [OK] Retry successful for chunk {failed_idx}")
                                processed_indices.update(merged_indices)
                                # Remove merged chunks from chunk_files_dict
                                for idx in merged_indices:
                                    if idx in chunk_files_dict:
                                        del chunk_files_dict[idx]
                                break
                    
                    # If next merge didn't work, try merging with previous chunks
                    if failed_idx not in retry_attempts and failed_idx > 1:
                        merged_retry = failed_chunk
                        merged_indices = {failed_idx}
                        j = failed_idx - 2  # Start from previous chunk (0-based)
                        
                        while j >= 0 and len(merged_indices) < 3:  # Merge up to 3 chunks
                            prev_chunk = merged_chunks[j]
                            merged_retry = prev_chunk + " " + merged_retry
                            merged_indices.add(j + 1)  # j+1 is 1-based index
                            j -= 1
                            
                            words = [w for w in merged_retry.split() if w.strip()]
                            word_count = len(words)
                            print(f"    Retrying chunk {failed_idx} merged with {len(merged_indices)-1} previous chunk(s) ({word_count} words, {len(merged_retry)} chars)...")
                            
                            if synthesize_text_chunk(tts, merged_retry, str(retry_path)):
                                retry_attempts[failed_idx] = retry_path
                                print(f"    [OK] Retry successful for chunk {failed_idx}")
                                processed_indices.update(merged_indices)
                                # Remove merged chunks from chunk_files_dict
                                for idx in merged_indices:
                                    if idx in chunk_files_dict:
                                        del chunk_files_dict[idx]
                                break
                    
                    if failed_idx not in retry_attempts:
                        print(f"    [ERROR] Retry failed for chunk {failed_idx}")
                
                # Update chunk_files_dict with successful retries
                chunk_files_dict.update(retry_attempts)
            
            # Convert to sorted list of paths
            chunk_files = [chunk_files_dict[i] for i in sorted(chunk_files_dict.keys())]
            
            if not chunk_files:
                print("[ERROR] All chunks failed to process.")
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
                print(f"[OK] Combined audio saved to: {output_path}")
                
                # Clean up chunk files
                for chunk_file in chunk_files:
                    chunk_file.unlink()
                print(f"  Cleaned up {len(chunk_files)} temporary chunk files.")
            elif len(chunk_files) == 1:
                # Only one chunk succeeded, just rename it
                chunk_files[0].rename(output_path)
                print(f"[OK] Audio saved to: {output_path}")
            else:
                print(f"[OK] Processed {len(chunk_files)} chunks (not combined - install pydub to combine)")
                print(f"  Chunk files saved in: {output_dir}")
            
            return True
        
    except Exception as e:
        # Handle encoding errors gracefully
        try:
            error_msg = str(e)
            print(f"Error during synthesis: {error_msg}")
        except UnicodeEncodeError:
            print(f"Error during synthesis: {repr(e)}")
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
        print(f"[OK] Conversion complete!")
        print(f"  Input:  {input_source}")
        print(f"  Output: {output_file}")
        
        # Play audio if requested
        if args.play:
            print(f"\nPlaying audio...")
            if play_audio(output_file):
                print("[OK] Audio playback complete.")
            else:
                print("[ERROR] Audio playback failed.")
    else:
        print(f"\n{'='*50}")
        print("[ERROR] Conversion failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

