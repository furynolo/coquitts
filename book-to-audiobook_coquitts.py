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
import shutil
from pathlib import Path
import platform
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict

try:
    from TTS.api import TTS
except ImportError:
    print("Error: CoquiTTS is not installed. Please run: pip install -r requirements.txt")
    sys.exit(1)

# Try to import NLTK for better sentence tokenization
NLTK_AVAILABLE = False
NLTK_SETUP_MESSAGE = None

try:
    import nltk
    
    # Check if punkt tokenizer is available (try both punkt and punkt_tab for compatibility)
    punkt_available = False
    punkt_tab_available = False
    
    try:
        nltk.data.find('tokenizers/punkt')
        punkt_available = True
    except LookupError:
        pass
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
        punkt_tab_available = True
    except LookupError:
        pass
    
    # If neither is available, download them
    if not punkt_available or not punkt_tab_available:
        try:
            if not punkt_available:
                print("NLTK punkt tokenizer not found. Downloading...")
                nltk.download('punkt', quiet=False)
            if not punkt_tab_available:
                print("NLTK punkt_tab tokenizer not found. Downloading...")
                nltk.download('punkt_tab', quiet=False)
            
            # Verify downloads
            try:
                nltk.data.find('tokenizers/punkt_tab')
                punkt_tab_available = True
            except LookupError:
                pass
            try:
                nltk.data.find('tokenizers/punkt')
                punkt_available = True
            except LookupError:
                pass
            
            if punkt_tab_available or punkt_available:
                NLTK_AVAILABLE = True
                print("NLTK tokenizers downloaded successfully.")
            else:
                NLTK_SETUP_MESSAGE = "Warning: NLTK tokenizer download may have failed. Using regex fallback for sentence splitting."
                print(NLTK_SETUP_MESSAGE)
        except Exception as e:
            NLTK_SETUP_MESSAGE = f"Warning: Could not download NLTK tokenizers ({e}). Using regex fallback for sentence splitting."
            print(NLTK_SETUP_MESSAGE)
            NLTK_AVAILABLE = False
    else:
        # Both are available
        NLTK_AVAILABLE = True
        
except ImportError:
    NLTK_SETUP_MESSAGE = "NLTK not installed. Using regex fallback for sentence splitting. Install with: pip install nltk"
except Exception as e:
    NLTK_SETUP_MESSAGE = f"Warning: NLTK setup error ({e}). Using regex fallback for sentence splitting."
    NLTK_AVAILABLE = False

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

def generate_unique_filename(base_name, output_dir=None, model=None, speaker=None):
    """
    Generate a unique output filename with incrementing number and timestamp.
    Format: base_name[_model][_speaker]_N_YYYYMMDD-HHMMSS.wav
    
    Args:
        base_name: Base name for the file (without extension)
        output_dir: Optional output directory (default: output-audio/)
        model: Optional model name to include in filename
        speaker: Optional speaker name to include in filename
        
    Returns:
        Path to unique output audio file
    """
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path("output-audio")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sanitize model name for filesystem (replace slashes, colons, etc. with underscores)
    safe_model = None
    if model:
        # Replace filesystem-unsafe characters with underscores
        safe_model = re.sub(r'[^\w\s-]', '_', model).strip().replace(' ', '_')
        # Replace multiple underscores with single underscore
        safe_model = re.sub(r'_+', '_', safe_model)
        # Limit length to avoid very long filenames
        if len(safe_model) > 50:
            # Use last part of model path if it's too long
            parts = safe_model.split('_')
            safe_model = '_'.join(parts[-3:]) if len(parts) > 3 else safe_model[:50]
    
    # Sanitize speaker name for filesystem
    safe_speaker = None
    if speaker:
        safe_speaker = re.sub(r'[^\w\s-]', '_', speaker).strip().replace(' ', '_')
        safe_speaker = re.sub(r'_+', '_', safe_speaker)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Build filename pattern for matching existing files
    # Pattern: base_name[_model][_speaker]_N_YYYYMMDD-HHMMSS.wav
    # We match files with the same base_name, model, and speaker (if provided)
    # Build the prefix pattern (everything before the number) first
    prefix_parts = [re.escape(base_name)]
    if safe_model:
        prefix_parts.append(re.escape(safe_model))
    if safe_speaker:
        prefix_parts.append(re.escape(safe_speaker))
    prefix_pattern = '_'.join(prefix_parts)
    
    # Now build the full pattern: prefix_(\d+)_timestamp.wav
    # The number pattern needs a single underscore before it (already in prefix_pattern)
    # and a single underscore after it (before timestamp)
    pattern = re.compile(rf'^{prefix_pattern}_(\d+)_\d{{8}}-\d{{6}}\.wav$')
    
    existing_numbers = []
    # Search for files matching the base pattern
    search_pattern = f"{base_name}_*.wav"
    for file in output_path.glob(search_pattern):
        match = pattern.match(file.name)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    # Get the next number (start at 1 if no files exist)
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    
    # Generate unique filename
    filename_parts = [base_name]
    if safe_model:
        filename_parts.append(safe_model)
    if safe_speaker:
        filename_parts.append(safe_speaker)
    filename_parts.append(str(next_number))
    filename_parts.append(timestamp)
    
    output_file = output_path / f"{'_'.join(filename_parts)}.wav"
    
    return str(output_file)

def get_output_filename(input_file, output_dir=None, model=None, speaker=None):
    """
    Generate output filename based on input filename.
    Now includes incrementing number, timestamp, model, and speaker (if applicable).
    
    Args:
        input_file: Path to input text file
        output_dir: Optional output directory (default: output-audio/)
        model: Optional model name to include in filename
        speaker: Optional speaker name to include in filename
        
    Returns:
        Path to unique output audio file
    """
    input_path = Path(input_file)
    input_stem = input_path.stem  # filename without extension
    
    return generate_unique_filename(input_stem, output_dir, model=model, speaker=speaker)

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
    
    # Remove IPA combining characters (Unicode combining diacritical marks)
    # These are not supported by TTS models and cause vocabulary errors
    # Remove combining characters but keep base characters
    combining_pattern = re.compile(r'[\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]')
    text = combining_pattern.sub('', text)
    
    return text

def load_character_voice_mapping(mapping_file):
    """
    Load character-to-voice mappings from a JSON file.
    
    Args:
        mapping_file: Path to JSON file with character-to-voice mappings
        Format: {"CharacterName": "speaker_id", ...}
        
    Returns:
        Dictionary with character-to-voice mappings, or None if file cannot be loaded
    """
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        if not isinstance(mapping, dict):
            print(f"Warning: Character voice mapping file '{mapping_file}' does not contain a valid dictionary.")
            return None
        
        print(f"Loaded {len(mapping)} character-to-voice mappings from '{mapping_file}'")
        return mapping
    except FileNotFoundError:
        print(f"Error: Character voice mapping file '{mapping_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in character voice mapping file '{mapping_file}': {e}")
        return None
    except Exception as e:
        print(f"Error loading character voice mapping file '{mapping_file}': {e}")
        return None

def load_pronunciations(pronunciations_file):
    """
    Load pronunciation mappings from a JSON file.
    Removes IPA combining characters from pronunciation values to prevent TTS errors.
    
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
        
        # Remove IPA combining characters from pronunciation values
        # These characters cause TTS vocabulary errors
        combining_pattern = re.compile(r'[\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]')
        cleaned_pronunciations = {}
        for key, value in pronunciations.items():
            if isinstance(value, str):
                # Remove combining characters from the pronunciation value
                cleaned_value = combining_pattern.sub('', value)
                cleaned_pronunciations[key] = cleaned_value
            else:
                cleaned_pronunciations[key] = value
        
        print(f"Loaded {len(cleaned_pronunciations)} pronunciation mappings from '{pronunciations_file}'")
        return cleaned_pronunciations
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
        # Remove IPA combining characters that may have been introduced by pronunciations
        # These are Unicode combining diacritical marks (U+0300-U+036F) that TTS models don't support
        # Remove combining characters but keep the base characters
        text = re.sub(r'[\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]', '', text)
    
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
    
    # Step 5.5: Final pass to remove any remaining combining characters
    # This ensures all IPA combining characters are removed before TTS processing
    combining_pattern = re.compile(r'[\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]')
    text = combining_pattern.sub('', text)
    
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
    Each chunk starts as a single sentence to ensure natural breaks and pauses in speech.
    Uses NLTK sentence tokenizer if available for better accuracy, otherwise falls back to regex.
    If a sentence exceeds max_chunk_size, it will be split further by clauses (commas, semicolons)
    or word boundaries as a last resort.
    Small chunks (below min_chunk_size) are automatically merged with subsequent sentences
    until they meet the minimum size requirement (without exceeding max_chunk_size).
    
    Args:
        text: Text to split
        max_chunk_size: Maximum characters per chunk (for sentences that exceed this, they'll be split further)
        min_chunk_size: Minimum characters per chunk (chunks smaller than this will be merged with next sentences)
        
    Returns:
        List of text chunks, where each chunk is typically a single sentence or merged sentences
    """
    # If text is short enough, return as single chunk
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split text into sentences using the best available method
    sentences = []
    use_nltk = False
    
    if NLTK_AVAILABLE:
        try:
            from nltk.tokenize import sent_tokenize
            # Use NLTK's sentence tokenizer for better accuracy
            # It handles abbreviations, decimals, ellipses, etc. correctly
            sentences = sent_tokenize(text)
            # Filter out empty sentences and strip whitespace
            sentences = [s.strip() for s in sentences if s.strip()]
            use_nltk = True
        except Exception as e:
            # If NLTK fails for any reason, fall back to regex
            print(f"  Warning: NLTK sentence tokenization failed ({e}), using regex fallback")
            use_nltk = False
    
    # Fallback to regex-based sentence splitting if NLTK not available or failed
    if not sentences and not use_nltk:
        # Improved regex pattern: matches sentence text + sentence ending punctuation + optional whitespace
        # This preserves the punctuation and spacing
        sentence_pattern = r'([^.!?]+[.!?]+(?:\s+|$))'
        sentence_matches = re.finditer(sentence_pattern, text)
        
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
    
    # Build chunks: each sentence becomes its own chunk
    # If a sentence exceeds max_chunk_size, split it further by clauses/commas
    chunks = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If sentence fits within max_chunk_size, use it as-is
        if len(sentence) <= max_chunk_size:
            chunks.append(sentence)
        else:
            # Sentence is too long - split it by clauses (commas, semicolons, colons)
            # This preserves natural speech breaks better than word boundaries
            clause_pattern = r'([^,;:]+[,;:]+(?:\s+|$))'
            clause_matches = list(re.finditer(clause_pattern, sentence))
            
            if clause_matches:
                # Split by clauses
                current_subchunk = ""
                for match in clause_matches:
                    clause = match.group(1).strip()
                    if not clause:
                        continue
                    
                    potential_subchunk = current_subchunk + " " + clause if current_subchunk else clause
                    
                    # If adding this clause would exceed max size, save current and start new
                    if len(potential_subchunk) > max_chunk_size and current_subchunk:
                        chunks.append(current_subchunk.strip())
                        current_subchunk = clause
                    else:
                        current_subchunk = potential_subchunk
                
                # Add any remaining content
                if current_subchunk.strip():
                    # If remaining is still too long, split by words
                    if len(current_subchunk) > max_chunk_size:
                        words = current_subchunk.split()
                        word_chunk = ""
                        for word in words:
                            potential_word_chunk = word_chunk + " " + word if word_chunk else word
                            if len(potential_word_chunk) > max_chunk_size and word_chunk:
                                chunks.append(word_chunk.strip())
                                word_chunk = word
                            else:
                                word_chunk = potential_word_chunk
                        if word_chunk.strip():
                            chunks.append(word_chunk.strip())
                    else:
                        chunks.append(current_subchunk.strip())
                
                # Handle any text after the last clause match
                last_match_end = clause_matches[-1].end()
                remaining = sentence[last_match_end:].strip()
                if remaining:
                    if len(remaining) > max_chunk_size:
                        # Split remaining by words
                        words = remaining.split()
                        word_chunk = ""
                        for word in words:
                            potential_word_chunk = word_chunk + " " + word if word_chunk else word
                            if len(potential_word_chunk) > max_chunk_size and word_chunk:
                                chunks.append(word_chunk.strip())
                                word_chunk = word
                            else:
                                word_chunk = potential_word_chunk
                        if word_chunk.strip():
                            chunks.append(word_chunk.strip())
                    else:
                        chunks.append(remaining)
            else:
                # No clauses found, split by words as fallback
                words = sentence.split()
                word_chunk = ""
                for word in words:
                    potential_word_chunk = word_chunk + " " + word if word_chunk else word
                    if len(potential_word_chunk) > max_chunk_size and word_chunk:
                        chunks.append(word_chunk.strip())
                        word_chunk = word
                    else:
                        word_chunk = potential_word_chunk
                if word_chunk.strip():
                    chunks.append(word_chunk.strip())
    
    # Filter out empty chunks
    filtered_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # Merge small chunks with next chunks until they meet minimum size
    merged_chunks = []
    i = 0
    while i < len(filtered_chunks):
        current_chunk = filtered_chunks[i]
        
        # If chunk is too small, merge with subsequent chunks
        if len(current_chunk) < min_chunk_size:
            merged = current_chunk
            j = i + 1
            
            # Keep merging with next chunks until we meet minimum size
            # But don't exceed max_chunk_size
            while j < len(filtered_chunks) and len(merged) < min_chunk_size:
                next_chunk = filtered_chunks[j]
                potential_merge = merged + " " + next_chunk
                
                # If merging would exceed max size, stop merging
                if len(potential_merge) > max_chunk_size:
                    break
                
                merged = potential_merge
                j += 1
            
            merged_chunks.append(merged.strip())
            i = j  # Move to the next unprocessed chunk
        else:
            # Chunk is large enough, use as-is
            merged_chunks.append(current_chunk)
            i += 1
    
    # Final safety check: if we somehow ended up with no chunks, return the original text
    return merged_chunks if merged_chunks else [text]

@dataclass
class Segment:
    """Represents a text segment (narration or dialogue)."""
    type: str  # "narration" or "dialogue"
    speaker: str  # Speaker name (e.g., "NARRATOR", "Alleria", "UNKNOWN") - post-pronunciation
    text: str  # The actual text content
    start_pos: int = 0  # Character position in original text
    end_pos: int = 0  # Character position in original text
    original_speaker: str = None  # Original speaker name before pronunciation mappings

class DialogueSegmenter:
    """Segments text into narration and dialogue blocks with speaker attribution."""
    
    # Common speech verbs for speaker attribution
    # Expanded list including more variations and less common verbs
    SPEECH_VERBS = {
        'say', 'said', 'says', 'saying',
        'ask', 'asked', 'asks', 'asking',
        'reply', 'replied', 'replies', 'replying',
        'whisper', 'whispered', 'whispers', 'whispering',
        'shout', 'shouted', 'shouts', 'shouting',
        'mutter', 'muttered', 'mutters', 'muttering',
        'cry', 'cried', 'cries', 'crying',
        'call', 'called', 'calls', 'calling',
        'exclaim', 'exclaimed', 'exclaims', 'exclaiming',
        'declare', 'declared', 'declares', 'declaring',
        'announce', 'announced', 'announces', 'announcing',
        'state', 'stated', 'states', 'stating',
        'tell', 'told', 'tells', 'telling',
        'speak', 'spoke', 'speaks', 'speaking',
        'answer', 'answered', 'answers', 'answering',
        'respond', 'responded', 'responds', 'responding',
        'bellow', 'bellowed', 'bellows', 'bellowing',
        'thunder', 'thundered', 'thunders', 'thundering',
        'growl', 'growled', 'growls', 'growling',
        'snarl', 'snarled', 'snarls', 'snarling',
        'roar', 'roared', 'roars', 'roaring',
        'yell', 'yelled', 'yells', 'yelling',
        'scream', 'screamed', 'screams', 'screaming',
        'murmur', 'murmured', 'murmurs', 'murmuring',
        'mumble', 'mumbled', 'mumbles', 'mumbling',
        'stammer', 'stammered', 'stammers', 'stammering',
        'stutter', 'stuttered', 'stutters', 'stuttering',
        'gasp', 'gasped', 'gasps', 'gasping',
        'sigh', 'sighed', 'sighs', 'sighing',
        'hiss', 'hissed', 'hisses', 'hissing',
        'snap', 'snapped', 'snaps', 'snapping',
        'bark', 'barked', 'barks', 'barking',
        'command', 'commanded', 'commands', 'commanding',
        'order', 'ordered', 'orders', 'ordering',
        'demand', 'demanded', 'demands', 'demanding',
        'insist', 'insisted', 'insists', 'insisting',
        'argue', 'argued', 'argues', 'arguing',
        'protest', 'protested', 'protests', 'protesting',
        'object', 'objected', 'objects', 'objecting',
        'agree', 'agreed', 'agrees', 'agreeing',
        'admit', 'admitted', 'admits', 'admitting',
        'confess', 'confessed', 'confesses', 'confessing',
        'claim', 'claimed', 'claims', 'claiming',
        'suggest', 'suggested', 'suggests', 'suggesting',
        'propose', 'proposed', 'proposes', 'proposing',
        'offer', 'offered', 'offers', 'offering',
        'promise', 'promised', 'promises', 'promising',
        'warn', 'warned', 'warns', 'warning',
        'threaten', 'threatened', 'threatens', 'threatening',
        'beg', 'begged', 'begs', 'begging',
        'plead', 'pleaded', 'pleads', 'pleading',
        'urge', 'urged', 'urges', 'urging',
        'encourage', 'encouraged', 'encourages', 'encouraging',
        'advise', 'advised', 'advises', 'advising',
        'explain', 'explained', 'explains', 'explaining',
        'describe', 'described', 'describes', 'describing',
        'mention', 'mentioned', 'mentions', 'mentioning',
        'note', 'noted', 'notes', 'noting',
        'observe', 'observed', 'observes', 'observing',
        'comment', 'commented', 'comments', 'commenting',
        'remark', 'remarked', 'remarks', 'remarking',
        'add', 'added', 'adds', 'adding',
        'continue', 'continued', 'continues', 'continuing',
        'conclude', 'concluded', 'concludes', 'concluding',
        'finish', 'finished', 'finishes', 'finishing',
        'interrupt', 'interrupted', 'interrupts', 'interrupting',
        'concede', 'conceded', 'concedes', 'conceding',
        'acknowledge', 'acknowledged', 'acknowledges', 'acknowledging',
        'confirm', 'confirmed', 'confirms', 'confirming',
        'deny', 'denied', 'denies', 'denying',
        'refuse', 'refused', 'refuses', 'refusing',
        'accept', 'accepted', 'accepts', 'accepting',
        'admit', 'admitted', 'admits', 'admitting',
    }
    
    def __init__(self, verbose: bool = False):
        self.unknown_count = 0
        self.verbose = verbose
        self._nltk_pos_available = False
        
        # Try to use NLTK for POS tagging if available
        if NLTK_AVAILABLE:
            try:
                from nltk import word_tokenize
                from nltk.tag import pos_tag
                from nltk.data import find as nltk_find
                
                # Try to download required data
                # The standard NLTK POS tagger resource is 'averaged_perceptron_tagger'
                tagger_available = False
                try:
                    nltk_find('taggers/averaged_perceptron_tagger')
                    tagger_available = True
                except LookupError:
                    # Try to download it
                    try:
                        if self.verbose:
                            print("    [NLTK] Downloading averaged_perceptron_tagger...")
                        nltk.download('averaged_perceptron_tagger', quiet=not self.verbose)
                        # Verify it was downloaded
                        try:
                            nltk_find('taggers/averaged_perceptron_tagger')
                            tagger_available = True
                            if self.verbose:
                                print("    [NLTK] averaged_perceptron_tagger downloaded successfully")
                        except LookupError:
                            if self.verbose:
                                print("    [NLTK] Warning: averaged_perceptron_tagger download may have failed")
                    except Exception as e:
                        if self.verbose:
                            print(f"    [NLTK] Error downloading averaged_perceptron_tagger: {e}")
                
                # Verify punkt is available
                punkt_available = False
                try:
                    nltk_find('tokenizers/punkt')
                    punkt_available = True
                except LookupError:
                    try:
                        nltk.download('punkt', quiet=True)
                        try:
                            nltk_find('tokenizers/punkt')
                            punkt_available = True
                        except LookupError:
                            pass
                    except Exception:
                        pass
                
                # Test that NLTK actually works by trying to tag a simple sentence
                if tagger_available and punkt_available:
                    try:
                        test_tokens = word_tokenize("Test sentence.")
                        test_tags = pos_tag(test_tokens)
                        # If we get here, NLTK is working
                        self._nltk_pos_available = True
                        self._pos_tag = pos_tag
                        self._word_tokenize = word_tokenize
                        if self.verbose:
                            print(f"    [NLTK SETUP] NLTK POS tagging initialized successfully")
                    except LookupError as lookup_error:
                        # If we get a LookupError, it means the tagger resource wasn't found
                        # This can happen if NLTK is looking for a different resource name
                        if self.verbose:
                            print(f"    [NLTK SETUP] Tagger resource not found: {lookup_error}")
                            print(f"    [NLTK SETUP] Attempting to download required resources...")
                        # Try downloading the tagger again
                        try:
                            nltk.download('averaged_perceptron_tagger', quiet=False)
                            # Try the test again
                            test_tokens = word_tokenize("Test sentence.")
                            test_tags = pos_tag(test_tokens)
                            self._nltk_pos_available = True
                            self._pos_tag = pos_tag
                            self._word_tokenize = word_tokenize
                            if self.verbose:
                                print(f"    [NLTK SETUP] NLTK POS tagging initialized successfully after download")
                        except Exception as retry_error:
                            if self.verbose:
                                print(f"    [NLTK SETUP] NLTK initialization failed after retry: {retry_error}")
                            self._nltk_pos_available = False
                    except Exception as test_error:
                        if self.verbose:
                            print(f"    [NLTK SETUP] NLTK resources found but test failed: {test_error}")
                        self._nltk_pos_available = False
                else:
                    self._nltk_pos_available = False
            except Exception as e:
                if self.verbose:
                    print(f"    [NLTK SETUP] Could not initialize NLTK POS tagging: {e}")
                self._nltk_pos_available = False
    
    def segment_text(self, text: str) -> List[Segment]:
        """
        Segment text into narration and dialogue blocks.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of Segment objects
        """
        segments = []
        self.unknown_count = 0
        
        # Pattern to match quoted dialogue (handles both straight and curly quotes)
        # Matches: "text" or "text" or "text"
        quote_pattern = re.compile(r'["""]([^"""]*)["""]', re.DOTALL)
        
        last_end = 0
        
        for match in quote_pattern.finditer(text):
            # Add narration before this quote
            narration_start = last_end
            narration_end = match.start()
            if narration_end > narration_start:
                narration_text = text[narration_start:narration_end].strip()
                if narration_text:
                    segments.append(Segment(
                        type="narration",
                        speaker="NARRATOR",
                        text=narration_text,
                        start_pos=narration_start,
                        end_pos=narration_end
                    ))
            
            # Extract dialogue
            dialogue_text = match.group(1).strip()
            if dialogue_text:
                # Check if this is actually narration (not dialogue)
                if self._is_narration_not_dialogue(dialogue_text):
                    # Treat as narration instead
                    segments.append(Segment(
                        type="narration",
                        speaker="NARRATOR",
                        text=dialogue_text,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
                else:
                    # Try to attribute speaker
                    speaker = self._attribute_speaker(text, match.start(), match.end(), dialogue_text)
                    segments.append(Segment(
                        type="dialogue",
                        speaker=speaker,
                        text=dialogue_text,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
            
            last_end = match.end()
        
        # Add remaining narration after last quote
        if last_end < len(text):
            narration_text = text[last_end:].strip()
            if narration_text:
                segments.append(Segment(
                    type="narration",
                    speaker="NARRATOR",
                    text=narration_text,
                    start_pos=last_end,
                    end_pos=len(text)
                ))
        
        # If no quotes found, treat entire text as narration
        if not segments:
            segments.append(Segment(
                type="narration",
                speaker="NARRATOR",
                text=text.strip(),
                start_pos=0,
                end_pos=len(text)
            ))
        
        return segments
    
    def _is_narration_not_dialogue(self, text: str) -> bool:
        """
        Check if quoted text is actually narration (not dialogue).
        
        Many books put narration or dialogue tags in quotes, which we should treat as narration.
        
        Args:
            text: The quoted text to check
            
        Returns:
            True if this should be treated as narration, False if it's dialogue
        """
        text_lower = text.lower().strip()
        
        # Parenthetical thoughts (e.g., "(Bladefist. There could be no doubt.)")
        if text.strip().startswith('(') and text.strip().endswith(')'):
            return True
        
        # Very short quotes are often dialogue tags or single words/phrases
        if len(text) < 20:
            # Single words or very short phrases that are likely not dialogue
            if len(text.split()) <= 2 and not any(c in text for c in '!?'):
                # Check if it's just a title or single word
                if text.strip() in ['Imperator,', 'Imperator', 'And,', 'And', 'Yes?', 'Yes', 'convinced.', 'convinced']:
                    return True
            
            # Check for common dialogue tag patterns (e.g., "he said", "she whispered", "he saluted")
            dialogue_tag_pattern = r'^(he|she|they|it|the|a|an)\s+\w+\s+(said|whispered|shouted|replied|asked|muttered|growled|snarled|roared|yelled|screamed|spoke|continued|interrupted|added|concluded|finished|agreed|nodded|shook|laughed|smiled|frowned|sighed|gasped|hissed|snapped|barked|commanded|ordered|demanded|insisted|argued|protested|objected|admitted|confessed|claimed|suggested|proposed|offered|promised|warned|threatened|begged|pleaded|urged|encouraged|advised|explained|described|mentioned|noted|observed|commented|remarked|conceded|acknowledged|confirmed|denied|refused|accepted|admitted|sneered|saluted|responded|snorted|seemed|held|did|was|were|is|are|flatly|with|pointing|looking|walking|turning|moving)'
            if re.match(dialogue_tag_pattern, text_lower):
                return True
            # Check for patterns like "he saluted", "Mar'gok snorted", "Ko'ragh seemed confused", "he said flatly", "the imperator said with a flourish"
            if re.match(r'^[A-Z][a-zA-Z\'-]+\s+(snorted|saluted|seemed|held|did|was|were|is|are|looked|glanced|stared|gazed|watched|saw|heard|felt|touched|reached|grabbed|turned|walked|ran|stood|sat|moved|went|came|arrived|left|entered|exited|opened|closed|started|stopped|began|ended|continued|paused|waited|hurried|rushed)', text_lower):
                return True
            # Check for patterns like "he said flatly", "the imperator said with a flourish"
            if re.search(r'\b(he|she|they|it|the|a|an|imperator|warchief|councilor|councillor)\s+\w+\s+(said|whispered|shouted|replied|asked|muttered|growled|snarled|roared|yelled|screamed|spoke|continued|interrupted|added|concluded|finished|agreed|nodded|shook|laughed|smiled|frowned|sighed|gasped|hissed|snapped|barked|commanded|ordered|demanded|insisted|argued|protested|objected|admitted|confessed|claimed|suggested|proposed|offered|promised|warned|threatened|begged|pleaded|urged|encouraged|advised|explained|described|mentioned|noted|observed|commented|remarked|conceded|acknowledged|confirmed|denied|refused|accepted|admitted)\s+(flatly|with|pointing|looking|walking|turning|moving|reaching|grabbing|holding|pushing|pulling|throwing|dropping|picking|putting|placing|opening|closing|starting|stopping|beginning|ending|continuing|pausing|waiting|hurrying|rushing)', text_lower):
                return True
        
        # Check for narration patterns (e.g., "Because he had to, Mar'gok leaned down")
        narration_patterns = [
            r'^(because|when|while|as|after|before|during|since|until|if|unless|although|though|even|despite)',
            r'^(the|a|an)\s+\w+\s+(did not|does not|will not|would not|could not|should not|may not|might not|must not|cannot|was not|were not|is not|are not|had not|has not|have not)',
            r'^(he|she|they|it|the|a|an)\s+\w+\s+(leaned|turned|walked|ran|stood|sat|looked|glanced|stared|gazed|watched|saw|heard|felt|touched|reached|grabbed|held|pushed|pulled|threw|dropped|picked|put|placed|moved|went|came|arrived|left|entered|exited|opened|closed|started|stopped|began|ended|continued|paused|waited|hurried|rushed|hit|struck|punched|kicked|slapped|squeezed|twisted|bent|straightened|raised|lowered|lifted|fell|jumped|hopped|stepped|sprinted|dashed|strolled|marched|trudged|limped|crawled|climbed|descended|ascended|flew|soared|dove|swam|floated|sank|rose|plunged|leaped|bounded|sprang|lunged|charged|attacked|shoved|yanked|tugged|dragged|hauled|carried|brought|took|seized|snatched|caught|gripped|clutched|crushed|smashed|slammed|bashed|pounded|hammered|beat|stomped|squashed|flattened|squished|compressed|compacted|packed|stuffed|filled|emptied|poured|spilled|dripped|tumbled|rolled|spun|rotated|stretched|extended|retracted|saluted|responded|snorted|seemed|held|beamed|stamped|raised|lowered)',
            # Patterns like "All feet stamped; fists were raised."
            r'^(all|some|many|few|several|most|both|each|every)\s+\w+\s+(stamped|raised|lowered|moved|went|came|did|was|were|is|are)',
            # Patterns like "It would have seemed..."
            r'^(it|this|that|these|those)\s+(would|could|should|might|may|will|can)\s+(have|be|seem|appear|look)',
            # Patterns like "He looked at Ko'ragh. The breaker beamed back."
            r'^(he|she|they|it|the|a|an)\s+\w+\s+(looked|glanced|stared|gazed|watched|saw|heard|felt|touched|reached|grabbed|held|pushed|pulled|threw|dropped|picked|put|placed|moved|went|came|arrived|left|entered|exited|opened|closed|started|stopped|began|ended|continued|paused|waited|hurried|rushed|hit|struck|punched|kicked|slapped|squeezed|twisted|bent|straightened|raised|lowered|lifted|fell|jumped|hopped|stepped|sprinted|dashed|strolled|marched|trudged|limped|crawled|climbed|descended|ascended|flew|soared|dove|swam|floated|sank|rose|plunged|leaped|bounded|sprang|lunged|charged|attacked|shoved|yanked|tugged|dragged|hauled|carried|brought|took|seized|snatched|caught|gripped|clutched|crushed|smashed|slammed|bashed|pounded|hammered|beat|stomped|squashed|flattened|squished|compressed|compacted|packed|stuffed|filled|emptied|poured|spilled|dripped|tumbled|rolled|spun|rotated|stretched|extended|retracted|saluted|responded|snorted|seemed|held|beamed)',
            # Patterns like "Mar'gok did not raise his hand"
            r'^[A-Z][a-zA-Z\'-]+\s+(did not|does not|will not|would not|could not|should not|may not|might not|must not|cannot|was not|were not|is not|are not|had not|has not|have not)',
            # Patterns like "A weighty grunt was Growmash's only response"
            r'^(a|an|the)\s+\w+\s+\w+\s+(was|were|is|are)\s+[A-Z][a-zA-Z\'-]+\'s',
            # Patterns like "The warchief was walking around the artifact"
            r'^(the|a|an)\s+\w+\s+(was|were|is|are)\s+(walking|running|standing|sitting|looking|glancing|staring|gazing|watching|seeing|hearing|feeling|touching|reaching|grabbing|holding|pushing|pulling|throwing|dropping|picking|putting|placing|moving|going|coming|arriving|leaving|entering|exiting|opening|closing|starting|stopping|beginning|ending|continuing|pausing|waiting|hurrying|rushing|hitting|striking|punching|kicking|slapping|squeezing|twisting|bending|straightening|raising|lowering|lifting|falling|jumping|hopping|stepping|sprinting|dashing|strolling|marching|trudging|limping|crawling|climbing|descending|ascending|flying|soaring|diving|swimming|floating|sinking|rising|plunging|leaping|bounding|springing|lunging|charging|attacking|shoving|yanking|tugging|dragging|hauling|carrying|bringing|taking|seizing|snatching|catching|gripping|clutching|crushing|smashing|slamming|bashing|pounding|hammering|beating|stomping|squashing|flattening|squishing|compressing|compacting|packing|stuffing|filling|emptying|pouring|spilling|dripping|tumbling|rolling|spinning|rotating|stretching|extending|retracting|saluting|responding|snorting|seeming|holding|beaming|squinting)',
        ]
        
        for pattern in narration_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Very long quotes without dialogue markers are often narration
        # Also check for very long quotes (like the 4932 char one)
        if len(text) > 500:
            return True
        if len(text) > 200 and not any(marker in text for marker in ['!', '?']):
            return True
        
        # Check for patterns like "he said, pointing at..." (dialogue tags with actions)
        if re.search(r',\s*(pointing|looking|walking|turning|moving|reaching|grabbing|holding|pushing|pulling|throwing|dropping|picking|putting|placing|opening|closing|starting|stopping|beginning|ending|continuing|pausing|waiting|hurrying|rushing|hitting|striking|punching|kicking|slapping|squeezing|twisting|bending|straightening|raising|lowering|lifting|falling|jumping|hopping|stepping|sprinting|dashing|strolling|marching|trudging|limping|crawling|climbing|descending|ascending|flying|soaring|diving|swimming|floating|sinking|rising|plunging|leaping|bounding|springing|lunging|charging|attacking|shoving|yanking|tugging|dragging|hauling|carrying|bringing|taking|seizing|snatching|catching|gripping|clutching|crushing|smashing|slamming|bashing|pounding|hammering|beating|stomping|squashing|flattening|squishing|compressing|compacting|packing|stuffing|filling|emptying|pouring|spilling|dripping|tumbling|rolling|spinning|rotating|stretching|extending|retracting|saluting|responding|snorting|seeming|holding|beaming)', text_lower):
            return True
        
        return False
    
    def _attribute_speaker(self, text: str, quote_start: int, quote_end: int, dialogue_text: str = "") -> str:
        """
        Attempt to attribute dialogue to a speaker by looking for name patterns near the quote.
        
        Args:
            text: Full text
            quote_start: Start position of quote
            quote_end: End position of quote
            dialogue_text: The dialogue text itself (for verbose output)
            
        Returns:
            Speaker name or "UNKNOWN"
        """
        # Look in a window around the quote (120 chars before and after for most patterns)
        # Use larger window for pronoun resolution and general detection (500 chars)
        window_size = 120
        pronoun_window_size = 500  # Increased from 300 to catch speakers mentioned further away
        before_context = text[max(0, quote_start - window_size):quote_start]
        before_context_for_pronouns = text[max(0, quote_start - pronoun_window_size):quote_start]
        after_context = text[quote_end:min(len(text), quote_end + window_size)]
        after_context_extended = text[quote_end:min(len(text), quote_end + pronoun_window_size)]  # Extended after context
        
        if self.verbose:
            preview = dialogue_text[:50] + "..." if len(dialogue_text) > 50 else dialogue_text
            print(f"    [VERBOSE] Attributing dialogue: \"{preview}\"")
            print(f"      Before context: {before_context[-60:] if len(before_context) > 60 else before_context}")
            print(f"      After context: {after_context[:60] if len(after_context) > 60 else after_context}")
        
        # Try to find speaker name before quote (use larger context for pronoun resolution)
        speaker = self._find_speaker_before(before_context_for_pronouns)
        if speaker:
            if self.verbose:
                print(f"      [FOUND] Speaker before quote: {speaker}")
            return speaker
        
        # Try to find speaker name after quote (use extended context for better detection)
        speaker = self._find_speaker_after(after_context_extended)
        if speaker:
            if self.verbose:
                print(f"      [FOUND] Speaker after quote: {speaker}")
            return speaker
        
        # Try to find speaker name in the dialogue itself (e.g., "I will give anything, Vareg")
        speaker = self._find_speaker_in_dialogue(dialogue_text)
        if speaker:
            if self.verbose:
                print(f"      [FOUND] Speaker in dialogue: {speaker}")
            return speaker
        
        # Try NLTK-based detection if available
        if self._nltk_pos_available:
            speaker = self._find_speaker_with_nltk(before_context, after_context)
            if speaker:
                if self.verbose:
                    print(f"      [FOUND] Speaker via NLTK: {speaker}")
                return speaker
        
        # No speaker found - provide detailed debug output
        if self.verbose:
            print(f"      [NOT FOUND] No speaker detected - using UNKNOWN")
            print(f"      [DEBUG] Full dialogue text ({len(dialogue_text)} chars): \"{dialogue_text[:200]}{'...' if len(dialogue_text) > 200 else ''}\"")
            print(f"      [DEBUG] Before context (last 300 chars): \"{before_context[-300:] if len(before_context) > 300 else before_context}\"")
            print(f"      [DEBUG] After context (first 300 chars): \"{after_context[:300] if len(after_context) > 300 else after_context}\"")
            print(f"      [DEBUG] Detection methods attempted:")
            print(f"        - Pattern matching before quote: Tried, no valid name found")
            print(f"        - Pattern matching after quote: Tried, no valid name found")
            print(f"        - Pattern matching in dialogue: Tried, no valid name found")
            if self._nltk_pos_available:
                print(f"        - NLTK POS tagging: Tried, no valid name found")
            else:
                print(f"        - NLTK POS tagging: Not available")
            
            # Show potential matches that were rejected
            print(f"      [DEBUG] Potential issues:")
            # Check if dialogue looks like narration (no quotes, very long, etc.)
            if len(dialogue_text) > 500:
                print(f"        - Dialogue text is very long ({len(dialogue_text)} chars) - might be narration")
            if not any(c in dialogue_text for c in '!?.'):
                print(f"        - Dialogue text lacks sentence-ending punctuation - might be narration")
            # Check if there are names in context that weren't detected
            import re
            potential_names = re.findall(r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,2})\b', before_context[-100:] + after_context[:100])
            if potential_names:
                print(f"        - Found potential names in context (may have been rejected): {', '.join(set(potential_names[:5]))}")
        self.unknown_count += 1
        return "UNKNOWN"
    
    def _find_speaker_before(self, text: str) -> Optional[str]:
        """Find speaker name in text before a quote."""
        if self.verbose:
            print(f"      [PATTERN] Trying patterns in text before quote...")
        
        # Pattern 1: Name said, Name, said, Name said:
        # Match up to 4 words as a name, followed by speech verb (with optional comma)
        name_pattern = r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3})(?:[,:]?\s+)(' + '|'.join(self.SPEECH_VERBS) + r')(?:[,:;]|\s+|$)'
        match = re.search(name_pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if self._validate_name(name):
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found: {name} {match.group(2)}")
                return name
        
        # Pattern 2: Name said [additional text] (e.g., "Mar'gok said, waving him off")
        # This allows text after the speech verb
        name_pattern2 = r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3})(?:[,:]?\s+)(' + '|'.join(self.SPEECH_VERBS) + r')(?:[,:;]|\s+[^.!?]*)'
        match = re.search(name_pattern2, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if self._validate_name(name):
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found (with trailing text): {name} {match.group(2)}")
                return name
        
        # Pattern 3: Handle pronouns (he/she/they) by finding last mentioned name
        # Look for "he said", "she said", etc. and try to find the last proper noun before it
        pronoun_pattern = r'\b(he|she|they|it)\s+(' + '|'.join(self.SPEECH_VERBS) + r')(?:[,:;]|\s+|$)'
        pronoun_match = re.search(pronoun_pattern, text, re.IGNORECASE)
        if pronoun_match:
            # Look backwards for the last proper noun (name)
            # Search in reverse order to find the most recent name
            before_pronoun = text[:pronoun_match.start()]
            # Find all capitalized word sequences that look like names
            name_matches = list(re.finditer(r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3})\b', before_pronoun))
            if name_matches:
                # Get the last (most recent) match
                name_match = name_matches[-1]
                name = name_match.group(1).strip()
                if self._validate_name(name):
                    if self.verbose:
                        print(f"      [PATTERN MATCH] Found via pronoun resolution: {name} (from '{pronoun_match.group(1)} {pronoun_match.group(2)}')")
                    return name
        
        # Pattern 4: Look for "He said nothing about..." or "He did not..." patterns
        # These often have the speaker mentioned earlier in a larger context
        pronoun_action_pattern = r'\b(he|she|they|it)\s+(said|did|does|will|would|could|should|may|might|must|can)\s+(nothing|not|no|never)'
        pronoun_action_match = re.search(pronoun_action_pattern, text, re.IGNORECASE)
        if pronoun_action_match:
            # Look backwards for the last proper noun (name) - use larger window
            before_pronoun = text[:pronoun_action_match.start()]
            # Find all capitalized word sequences that look like names
            name_matches = list(re.finditer(r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3})\b', before_pronoun))
            if name_matches:
                # Get the last (most recent) match
                name_match = name_matches[-1]
                name = name_match.group(1).strip()
                if self._validate_name(name):
                    if self.verbose:
                        print(f"      [PATTERN MATCH] Found via pronoun action resolution: {name} (from '{pronoun_action_match.group(1)} {pronoun_action_match.group(2)} {pronoun_action_match.group(3)}')")
                    return name
        
        if self.verbose:
            print(f"      [PATTERN] No matches found in before context")
        return None
    
    def _find_speaker_after(self, text: str) -> Optional[str]:
        """Find speaker name in text after a quote."""
        if self.verbose:
            print(f"      [PATTERN] Trying patterns in text after quote...")
        
        # Pattern 1: said Name, said Name.
        speech_verb_pattern = r'\b(' + '|'.join(self.SPEECH_VERBS) + r')(?:[,:]?\s+)([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3})(?:[,:;]|\s+|\.|$)'
        match = re.search(speech_verb_pattern, text, re.IGNORECASE)
        if match:
            name = match.group(2).strip()
            if self._validate_name(name):
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found: {match.group(1)} {name}")
                return name
        
        # Pattern 2: said Name [additional text] (e.g., "said Name, pointing at...")
        speech_verb_pattern2 = r'\b(' + '|'.join(self.SPEECH_VERBS) + r')(?:[,:]?\s+)([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3})(?:[,:;]|\s+[^.!?]*)'
        match = re.search(speech_verb_pattern2, text, re.IGNORECASE)
        if match:
            name = match.group(2).strip()
            if self._validate_name(name):
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found (with trailing text): {match.group(1)} {name}")
                return name
        
        if self.verbose:
            print(f"      [PATTERN] No matches found in after context")
        return None
    
    def _find_speaker_with_nltk(self, before_context: str, after_context: str) -> Optional[str]:
        """
        Use NLTK POS tagging to identify proper nouns (names) near speech verbs.
        
        Args:
            before_context: Text before the quote
            after_context: Text after the quote
            
        Returns:
            Speaker name if found, None otherwise
        """
        if not self._nltk_pos_available:
            return None
        
        try:
            # Combine contexts for analysis
            combined_context = (before_context + " " + after_context).strip()
            if not combined_context:
                return None
            
            # Tokenize and tag
            # Suppress stderr temporarily to avoid NLTK LookupError messages
            # (NLTK prints verbose error messages before raising LookupError)
            import sys
            import io
            old_stderr = sys.stderr
            stderr_buffer = io.StringIO()
            sys.stderr = stderr_buffer
            try:
                tokens = self._word_tokenize(combined_context)
                pos_tags = self._pos_tag(tokens)
            except LookupError:
                # NLTK resource not found - this is expected if resources aren't available
                # Restore stderr and return None silently
                sys.stderr = old_stderr
                return None
            finally:
                # Restore stderr (only if we didn't return early)
                if sys.stderr is stderr_buffer:
                    sys.stderr = old_stderr
            
            # Look for patterns: Proper noun (NNP) followed by verb (VB/VBD/VBZ/VBG)
            # or verb followed by proper noun
            for i in range(len(pos_tags) - 1):
                word1, pos1 = pos_tags[i]
                word2, pos2 = pos_tags[i + 1] if i + 1 < len(pos_tags) else ('', '')
                
                # Pattern 1: NNP (proper noun) followed by verb
                if pos1.startswith('NNP') and pos2.startswith('VB'):
                    # Check if word2 is a speech verb (case-insensitive)
                    if word2.lower() in self.SPEECH_VERBS:
                        # Extract the name (may be multi-word)
                        name_parts = [word1]
                        # Look backwards for more proper nouns
                        for j in range(i - 1, max(-1, i - 4), -1):
                            if j >= 0 and pos_tags[j][1].startswith('NNP'):
                                name_parts.insert(0, pos_tags[j][0])
                            else:
                                break
                        name = ' '.join(name_parts)
                        if self._validate_name(name):
                            if self.verbose:
                                print(f"      [NLTK] Found pattern: {name} {word2} (NNP + VB)")
                            return name
                
                # Pattern 2: Verb followed by NNP (proper noun)
                if pos1.startswith('VB') and pos2.startswith('NNP'):
                    # Check if word1 is a speech verb
                    if word1.lower() in self.SPEECH_VERBS:
                        # Extract the name (may be multi-word)
                        name_parts = [word2]
                        # Look forwards for more proper nouns
                        for j in range(i + 2, min(len(pos_tags), i + 5)):
                            if j < len(pos_tags) and pos_tags[j][1].startswith('NNP'):
                                name_parts.append(pos_tags[j][0])
                            else:
                                break
                        name = ' '.join(name_parts)
                        if self._validate_name(name):
                            if self.verbose:
                                print(f"      [NLTK] Found pattern: {word1} {name} (VB + NNP)")
                            return name
            
            # Also look for any proper nouns near speech verbs (within 3 words)
            speech_verb_indices = [i for i, (word, pos) in enumerate(pos_tags) 
                                  if word.lower() in self.SPEECH_VERBS]
            
            for verb_idx in speech_verb_indices:
                # Check 3 words before and after the verb
                for offset in range(-3, 4):
                    if offset == 0:
                        continue
                    check_idx = verb_idx + offset
                    if 0 <= check_idx < len(pos_tags):
                        word, pos = pos_tags[check_idx]
                        if pos.startswith('NNP') and self._validate_name(word):
                            if self.verbose:
                                print(f"      [NLTK] Found proper noun near speech verb: {word} (offset {offset} from '{pos_tags[verb_idx][0]}')")
                            return word
            
        except Exception as e:
            if self.verbose:
                print(f"      [NLTK ERROR] {e}")
        
        return None
    
    def _find_speaker_in_dialogue(self, dialogue_text: str) -> Optional[str]:
        """
        Try to find speaker name mentioned within the dialogue itself.
        Patterns like: "I will give anything, Vareg" or "Vareg, I will give anything"
        """
        if not dialogue_text or len(dialogue_text) < 5:
            return None
        
        if self.verbose:
            print(f"      [PATTERN] Checking dialogue text for speaker mentions...")
        
        # Pattern 1: Name at the end (e.g., "I will give anything, Vareg")
        # Look for comma or period followed by a name-like word at the end
        end_name_pattern = r'[,.]\s*([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,2})\s*[.)]?$'
        match = re.search(end_name_pattern, dialogue_text)
        if match:
            name = match.group(1).strip()
            if self._validate_name(name):
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found name at end of dialogue: {name}")
                return name
        
        # Pattern 2: Name at the beginning (e.g., "Vareg, I will give anything")
        start_name_pattern = r'^([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,2})[,:]?\s+[A-Z]'
        match = re.search(start_name_pattern, dialogue_text)
        if match:
            name = match.group(1).strip()
            if self._validate_name(name):
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found name at start of dialogue: {name}")
                return name
        
        # Pattern 3: Parenthetical mention (e.g., "(I will give anything, Vareg)")
        paren_pattern = r'\([^)]*([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,2})[^)]*\)'
        match = re.search(paren_pattern, dialogue_text)
        if match:
            name = match.group(1).strip()
            if self._validate_name(name):
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found name in parentheses: {name}")
                return name
        
        # Pattern 4: Possessive patterns (e.g., "A weighty grunt was Growmash's only response")
        # Look for patterns like "X's response", "X's voice", "X's only", etc.
        possessive_pattern = r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,2})\'s\s+(only|response|voice|reply|answer|words|statement|comment|remark|question|demand|request|order|command|threat|warning|promise|offer|suggestion|proposal|claim|admission|confession|denial|refusal|acceptance|agreement|disagreement|nod|shake|laugh|smile|frown|sigh|gasp|hiss|snap|bark|grunt|growl|snarl|roar|yell|scream|mutter|whisper|shout|reply|ask|speak|continue|interrupt|add|conclude|finish|agree|disagree)'
        match = re.search(possessive_pattern, dialogue_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if self._validate_name(name):
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found name in possessive pattern: {name}")
                return name
        
        # Pattern 5: "X was/were/is/are..." patterns (e.g., "Growmash was walking", "Mar'gok is ready")
        # But only if it's not at the very start (which might be narration)
        was_pattern = r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,2})\s+(was|were|is|are|did|does|will|would|could|should|may|might|must|can)\s+'
        match = re.search(was_pattern, dialogue_text)
        if match:
            # Only use this if it's not the first few words (likely narration)
            if match.start() > 10:  # Not at the very beginning
                name = match.group(1).strip()
                if self._validate_name(name):
                    if self.verbose:
                        print(f"      [PATTERN MATCH] Found name in 'was/is' pattern: {name}")
                    return name
        
        # Pattern 6: Look for any capitalized name-like words in the dialogue (as a last resort)
        # This is less reliable but can catch cases like "He said nothing about their armies, territories, mutual defense. Let Growmash ask..."
        # Be more strict - only look for multi-word names or names that appear in specific contexts
        # Skip single-word matches unless they're clearly names (e.g., after "Let", "Tell", etc.)
        all_names = re.findall(r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,2})\b', dialogue_text)
        for name in all_names:
            if self._validate_name(name):
                # For single-word names, only accept if they appear in specific contexts
                # (e.g., "Let Growmash ask", "Tell Vareg", etc.)
                if len(name.split()) == 1:
                    # Check if it appears after common verbs that introduce names
                    name_lower = name.lower()
                    # Skip if it's in our comprehensive rejection list
                    if name_lower in ['bile', 'by', 'call', 'crawl', 'escape', 'even', 'idiot', 'join', 
                                     'let', 'perhaps', 'pick', 'pride', 'satisfied', 'sour', 'speak', 
                                     'teach', 'tell', 'terribly', 'trust', 'very', 'wait', 'we\'ve', 
                                     'weve', 'while', 'and', 'or', 'but', 'if', 'when', 'where', 'why', 
                                     'how', 'what', 'who', 'which', 'that', 'this', 'these', 'those']:
                        continue
                    # Only accept if it appears after verbs like "let", "tell", "ask", etc.
                    name_pos = dialogue_text.find(name)
                    if name_pos > 0:
                        before_name = dialogue_text[:name_pos].lower()
                        # Check if it's after a verb that introduces a name
                        if not re.search(r'\b(let|tell|ask|call|name|know|see|meet|find|help|show|give|send|bring|take)\s+$', before_name[-20:]):
                            # Skip single-word names that don't appear in name-introducing contexts
                            continue
                
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found potential name in dialogue: {name}")
                return name
        
        if self.verbose:
            print(f"      [PATTERN] No speaker found in dialogue text")
        return None
    
    def _validate_name(self, name: str) -> bool:
        """Validate that a string looks like a proper name."""
        if not name or len(name) < 2:
            return False
        
        # Must start with capital letter
        if not name[0].isupper():
            return False
        
        # Should be mostly alphabetic (allow apostrophes, hyphens, spaces)
        if not all(c.isalnum() or c in " '-" for c in name):
            return False
        
        # Should have at least one letter
        if not any(c.isalpha() for c in name):
            return False
        
        words = name.split()
        if not words:
            return False
        
        # Reject if it contains punctuation that suggests it's a sentence fragment
        # (allow apostrophes and hyphens which are common in names)
        if any(c in name for c in ',;:!?.'):
            return False
        
        # Reject possessive forms (e.g., "Mar'gok's", "John's")
        if name.endswith("'s") or name.endswith("'s "):
            return False
        
        # Reject patterns like "X's Y" (e.g., "Mar'gok's heads", "John's hand")
        # This catches possessive + noun combinations
        if "'s " in name or " 's " in name:
            return False
        
        # Reject known non-person entities (organizations, places, etc.)
        non_person_entities = {
            'iron horde', 'highmaul', 'shattered hand', 'warsong', 'draenor', 'draenoor',
            'highmaul clan', 'iron horde', 'grommashar', 'nuh-grand', 'bladefist',
            'warchief', 'imperator', 'councilor', 'councillor', 'high councilor',
        }
        name_lower = name.lower()
        if name_lower in non_person_entities:
            return False
        
        # Reject if it contains common organization/place indicators
        org_indicators = {'horde', 'clan', 'army', 'legion', 'empire', 'kingdom', 'city', 'town'}
        if any(word.lower() in org_indicators for word in words):
            return False
        
        # Reject single-word pronouns and common words
        # Expanded list to include common words that are often capitalized but aren't names
        single_word_rejects = {
            # Pronouns
            'do', 'he', 'she', 'it', 'we', 'they', 'you', 'i', 'me', 'him', 'her',
            'us', 'them', 'this', 'that', 'these', 'those', 'who', 'what', 'which',
            'where', 'when', 'why', 'how', 'their', 'his', 'hers', 'ours', 'yours',
            'theirs', 'my', 'your', 'our', 'its', 'whose', 'whom',
            # Common verbs (often capitalized at sentence start)
            'bile', 'by', 'call', 'crawl', 'escape', 'even', 'idiot', 'join', 'let', 'perhaps',
            'pick', 'pride', 'satisfied', 'sour', 'speak', 'teach', 'tell', 'terribly', 'trust',
            'very', 'wait', 'while', 'we\'ve', 'weve',
            # More common verbs
            'ask', 'answer', 'reply', 'say', 'said', 'tell', 'told', 'speak', 'spoke', 'talk',
            'walk', 'run', 'go', 'come', 'see', 'look', 'watch', 'hear', 'listen', 'feel',
            'think', 'know', 'understand', 'believe', 'hope', 'wish', 'want', 'need', 'try',
            'give', 'take', 'get', 'put', 'set', 'make', 'do', 'did', 'done', 'have', 'has',
            'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'begin', 'start', 'stop', 'end', 'finish', 'continue', 'keep', 'stay', 'leave',
            'return', 'turn', 'move', 'stand', 'sit', 'lie', 'lay', 'fall', 'rise', 'raise',
            'lower', 'lift', 'drop', 'throw', 'catch', 'grab', 'hold', 'push', 'pull',
            'open', 'close', 'break', 'fix', 'build', 'destroy', 'create', 'make', 'find',
            'lose', 'win', 'lose', 'fight', 'attack', 'defend', 'protect', 'save', 'kill',
            'die', 'live', 'survive', 'escape', 'hide', 'seek', 'search', 'find', 'lose',
            # Common adjectives/adverbs
            'good', 'bad', 'great', 'small', 'large', 'big', 'little', 'huge', 'tiny',
            'long', 'short', 'tall', 'wide', 'narrow', 'thick', 'thin', 'heavy', 'light',
            'fast', 'slow', 'quick', 'quickly', 'slowly', 'soon', 'late', 'early',
            'old', 'new', 'young', 'ancient', 'modern', 'ancient', 'recent', 'old',
            'hot', 'cold', 'warm', 'cool', 'freezing', 'burning', 'warm', 'cool',
            'bright', 'dark', 'light', 'heavy', 'strong', 'weak', 'powerful', 'helpless',
            'happy', 'sad', 'angry', 'calm', 'excited', 'bored', 'tired', 'energetic',
            'beautiful', 'ugly', 'pretty', 'handsome', 'attractive', 'repulsive',
            'smart', 'stupid', 'clever', 'foolish', 'wise', 'foolish', 'brilliant', 'dumb',
            'rich', 'poor', 'wealthy', 'poverty', 'expensive', 'cheap', 'free', 'costly',
            'easy', 'hard', 'difficult', 'simple', 'complex', 'complicated', 'easy',
            'safe', 'dangerous', 'risky', 'secure', 'unsafe', 'protected', 'vulnerable',
            'true', 'false', 'real', 'fake', 'genuine', 'artificial', 'natural', 'man-made',
            'right', 'wrong', 'correct', 'incorrect', 'accurate', 'inaccurate', 'precise',
            'full', 'empty', 'complete', 'incomplete', 'finished', 'unfinished', 'done',
            'ready', 'prepared', 'unprepared', 'ready', 'set', 'go',
            # Common nouns (often capitalized)
            'time', 'day', 'night', 'morning', 'afternoon', 'evening', 'dawn', 'dusk',
            'year', 'month', 'week', 'hour', 'minute', 'second', 'moment', 'instant',
            'place', 'location', 'position', 'spot', 'area', 'region', 'zone', 'territory',
            'way', 'path', 'road', 'street', 'avenue', 'lane', 'route', 'direction',
            'thing', 'object', 'item', 'piece', 'part', 'section', 'portion', 'fragment',
            'person', 'people', 'human', 'man', 'woman', 'child', 'adult', 'elder',
            'group', 'team', 'crew', 'gang', 'band', 'party', 'crowd', 'mob',
            'house', 'home', 'building', 'structure', 'tower', 'castle', 'palace', 'mansion',
            'room', 'chamber', 'hall', 'corridor', 'passage', 'door', 'window', 'wall',
            'weapon', 'sword', 'knife', 'axe', 'bow', 'arrow', 'spear', 'shield',
            'armor', 'helmet', 'boot', 'glove', 'gauntlet', 'plate', 'mail', 'leather',
            # Common interjections/exclamations
            'ah', 'oh', 'ooh', 'aah', 'wow', 'whoa', 'hey', 'hi', 'hello', 'goodbye',
            'yes', 'no', 'maybe', 'perhaps', 'sure', 'okay', 'ok', 'alright', 'right',
            'well', 'hmm', 'um', 'uh', 'er', 'huh', 'what', 'why', 'how', 'when',
            # Common conjunctions/prepositions
            'and', 'or', 'but', 'nor', 'for', 'so', 'yet', 'because', 'since', 'although',
            'though', 'while', 'whereas', 'if', 'unless', 'until', 'till', 'before', 'after',
            'during', 'through', 'throughout', 'across', 'over', 'under', 'above', 'below',
            'beside', 'besides', 'between', 'among', 'amongst', 'within', 'without',
            'inside', 'outside', 'into', 'onto', 'upon', 'toward', 'towards', 'against',
            'with', 'without', 'by', 'from', 'to', 'of', 'in', 'on', 'at', 'for',
            'about', 'around', 'round', 'near', 'far', 'away', 'off', 'out', 'up', 'down',
            # Time-related
            'today', 'tomorrow', 'yesterday', 'now', 'then', 'soon', 'later', 'earlier',
            'recently', 'lately', 'always', 'never', 'often', 'sometimes', 'usually',
            'frequently', 'rarely', 'seldom', 'occasionally', 'constantly', 'continuously',
        }
        if len(words) == 1 and words[0].lower() in single_word_rejects:
            return False
        
        # Reject if it starts with common sentence starters
        common_starters = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'when', 'where', 'why', 'how',
            'what', 'who', 'which', 'that', 'this', 'these', 'those', 'some', 'any',
            'all', 'each', 'every', 'no', 'not', 'yes', 'so', 'then', 'now', 'here',
            'there', 'once', 'twice', 'first', 'last', 'next', 'previous', 'other',
            'another', 'many', 'much', 'more', 'most', 'less', 'few', 'little',
            'both', 'either', 'neither', 'such', 'same', 'different', 'various',
            'several', 'certain', 'particular', 'general', 'specific',
        }
        
        if words[0].lower() in common_starters:
            return False
        
        # Reject if it contains common determiners/articles in the middle
        # (e.g., "He resisted the" should not be a name)
        common_middle = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'when', 'where',
                        'why', 'how', 'what', 'who', 'which', 'that', 'this', 'these',
                        'those', 'some', 'any', 'all', 'each', 'every', 'no', 'not',
                        'yes', 'so', 'then', 'now', 'here', 'there', 'once', 'twice',
                        'first', 'last', 'next', 'previous', 'other', 'another', 'many',
                        'much', 'more', 'most', 'less', 'few', 'little', 'both',
                        'either', 'neither', 'such', 'same', 'different', 'various',
                        'several', 'certain', 'particular', 'general', 'specific',
                        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                        'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'done',
                        'will', 'would', 'could', 'should', 'may', 'might', 'must',
                        'can', 'cannot', 'resisted', 'resists', 'resist', 'resisting',
                        'let', 'lets', 'letting', 'had', 'has', 'have', 'having',
                        'heads', 'head', 'heading', 'headed', 'was', 'were', 'is',
                        'are', 'you', 'your', 'yours',
        }
        
        # Check if any middle word is a common word (not the first or last)
        if len(words) > 2:
            for word in words[1:-1]:
                if word.lower() in common_middle:
                    return False
        
        # Reject if it ends with common verbs/auxiliaries (e.g., "Growmash was", "Do you")
        common_endings = {
            'was', 'were', 'is', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'cannot', 'you', 'your', 'yours', 'he', 'she', 'it', 'we',
            'they', 'them', 'him', 'her', 'us', 'me', 'i',
        }
        if len(words) > 1 and words[-1].lower() in common_endings:
            return False
        
        # Reject titles when used alone (without a name)
        titles_alone = {
            'imperator', 'emperor', 'king', 'queen', 'prince', 'princess', 'duke',
            'duchess', 'lord', 'lady', 'sir', 'madam', 'miss', 'mister', 'mr',
            'mrs', 'ms', 'dr', 'doctor', 'professor', 'captain', 'general', 'colonel',
            'major', 'sergeant', 'lieutenant', 'warchief', 'chieftain', 'highlord',
            'highlady', 'councilor', 'councillor',
        }
        if len(words) == 1 and words[0].lower() in titles_alone:
            return False
        
        # Names should typically be 1-4 words, and each word should be capitalized
        if len(words) > 4:
            return False
        
        # All words should start with capital (except for particles like "de", "van", etc.)
        # For now, require first word to be capitalized, others can be lowercase
        # (to handle names like "de la Cruz" or "van der Berg")
        if not words[0][0].isupper():
            return False
        
        # For single-word names, be extra strict - reject if it's a common English word
        # Single-word names are less common in fiction, so we should be cautious
        if len(words) == 1:
            word_lower = words[0].lower()
            # Reject if it's a very short word (likely not a name)
            if len(word_lower) <= 2:
                return False
            # Reject if it's a common verb form (past tense, present tense, etc.)
            common_verb_forms = {
                'called', 'calls', 'calling', 'crawled', 'crawls', 'crawling',
                'escaped', 'escapes', 'escaping', 'joined', 'joins', 'joining',
                'picked', 'picks', 'picking', 'spoke', 'speaks', 'speaking',
                'taught', 'teaches', 'teaching', 'told', 'tells', 'telling',
                'waited', 'waits', 'waiting', 'trusted', 'trusts', 'trusting',
                'satisfied', 'satisfies', 'satisfying', 'prided', 'prides', 'priding',
            }
            if word_lower in common_verb_forms:
                return False
            # Reject if it's a common adjective/adverb
            common_adjectives = {
                'even', 'very', 'sour', 'terribly', 'perhaps', 'while', 'bile',
                'pride', 'satisfied', 'idiot', 'escape', 'crawl', 'call', 'join',
                'pick', 'speak', 'teach', 'tell', 'trust', 'wait', 'let', 'by',
            }
            if word_lower in common_adjectives:
                return False
            # Reject if it looks like a common word pattern (ends in -ly, -ed, -ing, etc.)
            if word_lower.endswith(('ly', 'ed', 'ing', 'er', 'est', 'tion', 'sion')):
                # But allow if it's a known name pattern (like "Kelly", "Ed", "King")
                known_name_endings = {'ly', 'ed', 'ing', 'er', 'est'}  # These can be names
                if word_lower.endswith(('tion', 'sion')):  # These are almost never names
                    return False
        
        return True

class SpeakerAssigner:
    """Assigns TTS speaker voices to detected characters."""
    
    def __init__(self, available_speakers: List[str], narrator_speaker: Optional[str] = None, character_voice_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize speaker assigner.
        
        Args:
            available_speakers: List of available TTS speaker IDs
            narrator_speaker: Optional narrator speaker ID (defaults to p225 if available)
            character_voice_mapping: Optional dictionary mapping character names (pre-pronunciation) to speaker IDs
        """
        self.available_speakers = available_speakers
        # Default narrator to p225 if available, otherwise first speaker
        if narrator_speaker:
            self.narrator_speaker = narrator_speaker
        elif 'p225' in available_speakers:
            self.narrator_speaker = 'p225'
        else:
            self.narrator_speaker = (available_speakers[0] if available_speakers else None)
        
        # Character name -> TTS speaker mapping
        self.character_map: Dict[str, str] = {}
        
        # Character-to-voice mapping from file (uses original/pre-pronunciation names)
        self.character_voice_mapping = character_voice_mapping or {}
        
        # Track which speakers are used (excluding narrator)
        self.used_speakers = set()
        if self.narrator_speaker:
            self.used_speakers.add(self.narrator_speaker)
        
        # Get pool of available speakers (excluding narrator)
        self.speaker_pool = [s for s in available_speakers if s != self.narrator_speaker]
        if not self.speaker_pool:
            # If only one speaker, use it for everything
            self.speaker_pool = available_speakers
        
        self.speaker_index = 0
        self.warnings = []
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize character names to merge variants (e.g., "Growmash Hellscream" = "Growmash" = "Hellscream").
        Also handles titles like "High Councilor Vareg" = "Vareg".
        
        Args:
            name: Character name to normalize
            
        Returns:
            Normalized name (canonical form)
        """
        name_lower = name.lower()
        
        # Known name variants - map to canonical form
        name_variants = {
            'growmash': 'Growmash',
            'growmash hellscream': 'Growmash',
            'hellscream': 'Growmash',
            'warchief hellscream': 'Growmash',
        }
        
        # Check if this name is a variant of a known name
        for variant, canonical in name_variants.items():
            if name_lower == variant or name_lower.endswith(' ' + variant) or name_lower.startswith(variant + ' '):
                return canonical
        
        # Check if this name contains another known name (e.g., "Growmash Hellscream" contains "Growmash")
        for variant, canonical in name_variants.items():
            if variant in name_lower:
                return canonical
        
        # Extract the last word as the base name (handles "High Councilor Vareg" -> "Vareg")
        # This helps merge names with titles
        words = name.split()
        if len(words) > 1:
            last_word = words[-1]
            # Check if the last word matches any known single-word name
            for variant, canonical in name_variants.items():
                variant_words = variant.split()
                if len(variant_words) == 1 and variant_words[0] == last_word.lower():
                    return canonical
        
        # If no variant found, return original (capitalized properly)
        return name
    
    def assign_speakers(self, segments: List[Segment]) -> Dict[str, str]:
        """
        Assign TTS speakers to all unique characters in segments.
        
        Args:
            segments: List of text segments
            
        Returns:
            Dictionary mapping character names to TTS speaker IDs
        """
        # Collect unique character names (excluding NARRATOR and UNKNOWN)
        # Use original_speaker (pre-pronunciation) if available, otherwise use speaker
        # Normalize names to merge variants
        character_names = set()
        name_mapping = {}  # Map post-pronunciation names to normalized names
        original_name_mapping = {}  # Map normalized names to original (pre-pronunciation) names
        
        for segment in segments:
            if segment.type == "dialogue" and segment.speaker not in ("NARRATOR", "UNKNOWN"):
                # Use original_speaker if available (pre-pronunciation), otherwise use speaker
                original_name = segment.original_speaker if segment.original_speaker else segment.speaker
                normalized = self._normalize_name(segment.speaker)
                character_names.add(normalized)
                name_mapping[segment.speaker] = normalized
                # Store original name for this normalized name
                if normalized not in original_name_mapping:
                    original_name_mapping[normalized] = original_name
        
        # Additional merging: if one name ends with another name, merge them
        # (e.g., "High Councilor Vareg" ends with "Vareg")
        # Group names by their last word (base name)
        base_name_groups = {}
        for name in character_names:
            words = name.split()
            if words:
                base = words[-1].lower()  # Last word is typically the surname/base name
                if base not in base_name_groups:
                    base_name_groups[base] = []
                base_name_groups[base].append(name)
        
        # For each group, choose the shortest name as canonical (prefer single-word names)
        additional_merges = {}
        for base, names in base_name_groups.items():
            if len(names) > 1:
                # Sort by length, then alphabetically
                names_sorted = sorted(names, key=lambda x: (len(x), x.lower()))
                canonical = names_sorted[0]  # Shortest name
                for name in names_sorted[1:]:
                    additional_merges[name] = canonical
        
        # Apply additional merges
        for name, canonical in additional_merges.items():
            if name in character_names:
                character_names.remove(name)
                character_names.add(canonical)
                # Update name_mapping for all names that mapped to 'name'
                for orig_name, norm_name in list(name_mapping.items()):
                    if norm_name == name:
                        name_mapping[orig_name] = canonical
        
        # Update segments with normalized names
        for segment in segments:
            if segment.type == "dialogue" and segment.speaker in name_mapping:
                segment.speaker = name_mapping[segment.speaker]
        
        # Sort character names for deterministic assignment
        sorted_characters = sorted(character_names, key=str.lower)
        
        # Assign speakers - check character_voice_mapping first (uses original/pre-pronunciation names)
        for char_name in sorted_characters:
            if char_name not in self.character_map:
                # Get original name for this character (pre-pronunciation)
                original_name = original_name_mapping.get(char_name, char_name)
                
                # Check if there's a manual mapping for this character (use original name)
                mapped_speaker = None
                # Try exact match first
                if original_name in self.character_voice_mapping:
                    mapped_speaker = self.character_voice_mapping[original_name]
                else:
                    # Try normalized version
                    normalized_original = self._normalize_name(original_name)
                    if normalized_original in self.character_voice_mapping:
                        mapped_speaker = self.character_voice_mapping[normalized_original]
                    else:
                        # Try case-insensitive match
                        for mapping_name, mapping_speaker_id in self.character_voice_mapping.items():
                            if mapping_name.lower() == original_name.lower() or mapping_name.lower() == normalized_original.lower():
                                mapped_speaker = mapping_speaker_id
                                break
                
                if mapped_speaker:
                    # Use the mapped speaker if it's available
                    if mapped_speaker in self.available_speakers:
                        self.character_map[char_name] = mapped_speaker
                        self.used_speakers.add(mapped_speaker)
                    else:
                        self.warnings.append(
                            f"Character '{char_name}' (original: '{original_name}') mapped to speaker '{mapped_speaker}' "
                            f"which is not available. Using auto-assignment instead."
                        )
                        # Fall through to auto-assignment
                        mapped_speaker = None
                
                # Auto-assign if no mapping found or mapping was invalid
                if not mapped_speaker:
                    if self.speaker_index < len(self.speaker_pool):
                        speaker = self.speaker_pool[self.speaker_index]
                        self.character_map[char_name] = speaker
                        self.used_speakers.add(speaker)
                        self.speaker_index += 1
                    else:
                        # Reuse speakers in round-robin fashion
                        speaker = self.speaker_pool[self.speaker_index % len(self.speaker_pool)]
                        self.character_map[char_name] = speaker
                        if len(character_names) > len(self.speaker_pool):
                            self.warnings.append(
                                f"More characters ({len(character_names)}) than available speakers "
                                f"({len(self.speaker_pool)}). Reusing speakers."
                            )
        
        # Handle UNKNOWN - assign to narrator speaker (p225 by default)
        if any(s.speaker == "UNKNOWN" for s in segments if s.type == "dialogue"):
            self.character_map["UNKNOWN"] = self.narrator_speaker
        
        return self.character_map
    
    def get_speaker_for_segment(self, segment: Segment) -> str:
        """
        Get the TTS speaker ID for a given segment.
        
        Args:
            segment: Text segment
            
        Returns:
            TTS speaker ID
        """
        if segment.type == "narration":
            return self.narrator_speaker or self.available_speakers[0]
        else:
            # Dialogue
            return self.character_map.get(segment.speaker, self.narrator_speaker or self.available_speakers[0])

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

def get_chunk_verbose_info(chunk, full_text, chunk_index=None, total_chunks=None):
    """
    Get verbose debugging information about a chunk.
    
    Args:
        chunk: The chunk text
        full_text: The full original text
        chunk_index: Optional chunk index (1-based)
        total_chunks: Optional total number of chunks
        
    Returns:
        Dictionary with verbose information
    """
    chunk_words = [w for w in chunk.split() if w.strip()]
    full_words = [w for w in full_text.split() if w.strip()]
    
    # Find word positions in full text
    chunk_start_word_idx = None
    chunk_end_word_idx = None
    
    # Find character positions
    chunk_start_char = full_text.find(chunk[:min(50, len(chunk))])
    if chunk_start_char == -1:
        # Try to find a shorter match
        chunk_start_char = full_text.find(chunk[:min(20, len(chunk))])
    chunk_end_char = chunk_start_char + len(chunk) if chunk_start_char >= 0 else None
    
    # Find word indices by matching words
    if chunk_words and full_words:
        # Try to find first word of chunk in full text
        first_chunk_word = chunk_words[0] if chunk_words else ""
        last_chunk_word = chunk_words[-1] if chunk_words else ""
        
        for i, word in enumerate(full_words):
            if word == first_chunk_word and chunk_start_word_idx is None:
                # Check if subsequent words match
                match = True
                for j in range(min(len(chunk_words), len(full_words) - i)):
                    if i + j >= len(full_words) or full_words[i + j] != chunk_words[j]:
                        match = False
                        break
                if match:
                    chunk_start_word_idx = i + 1  # 1-based
                    chunk_end_word_idx = i + len(chunk_words)  # 1-based
                    break
    
    # Get first and last 4 words
    first_words = " ".join(chunk_words[:4]) if len(chunk_words) >= 4 else " ".join(chunk_words)
    last_words = " ".join(chunk_words[-4:]) if len(chunk_words) >= 4 else " ".join(chunk_words)
    
    info = {
        'first_words': first_words,
        'last_words': last_words,
        'word_count': len(chunk_words),
        'char_count': len(chunk),
        'char_start': chunk_start_char + 1 if chunk_start_char is not None and chunk_start_char >= 0 else None,
        'char_end': chunk_end_char if chunk_end_char is not None else None,
        'word_start': chunk_start_word_idx,
        'word_end': chunk_end_word_idx,
        'chunk_index': chunk_index,
        'total_chunks': total_chunks
    }
    
    return info

def synthesize_text_chunk(tts, text, chunk_path, speaker=None, speaker_wav=None):
    """
    Synthesize a single chunk of text.
    
    Args:
        tts: Initialized TTS model
        text: Text chunk to synthesize
        chunk_path: Path to save the chunk audio file
        speaker: Optional speaker name for multi-speaker models
        speaker_wav: Optional path to reference audio file for multi-speaker models
        
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
        
        # Build kwargs for tts_to_file
        kwargs = {"text": text, "file_path": chunk_path}
        if speaker:
            kwargs["speaker"] = speaker
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        
        tts.tts_to_file(**kwargs)
        return True
    except Exception as e:
        error_msg = str(e).lower()
        print(f"  Error synthesizing chunk: {e}")
        
        # Multi-speaker model errors
        if "multi-speaker" in error_msg or "speaker" in error_msg and ("need" in error_msg or "pass" in error_msg or "require" in error_msg):
            print(f"  Note: This appears to be a multi-speaker model error.")
            print(f"  Use --speaker to specify a speaker name, or --speaker-wav for reference audio.")
        
        # Kernel size errors indicate insufficient tokenizable content
        if "kernel size" in error_msg:
            words = [w for w in text.split() if w.strip()]
            word_count = len(words)
            print(f"  Note: Chunk has insufficient content for TTS model ({word_count} words, {len(text)} chars)")
            print(f"  This chunk will be merged with adjacent chunks and retried.")
        
        return False

def _synthesize_with_auto_speaker(tts, segments: List[Segment], assigner: SpeakerAssigner, output_path: str, verbose: bool = False) -> bool:
    """
    Synthesize text with automatic speaker routing for dialogue segments.
    
    Args:
        tts: Initialized TTS model
        segments: List of text segments (narration/dialogue)
        assigner: SpeakerAssigner instance with character mappings
        output_path: Path to save the output audio file
        verbose: If True, output detailed debugging information
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_dir = Path(output_path).parent
        output_stem = Path(output_path).stem
        
        # Create temporary directory for segment files
        temp_segment_dir = output_dir / f".temp_segments_{output_stem}"
        temp_segment_dir.mkdir(parents=True, exist_ok=True)
        
        segment_files = []
        
        print(f"Synthesizing {len(segments)} segments with auto speaker routing...\n")
        
        for i, segment in enumerate(segments, 1):
            segment_speaker = assigner.get_speaker_for_segment(segment)
            segment_path = temp_segment_dir / f"segment_{i:04d}.wav"
            
            segment_type_label = "NARRATION" if segment.type == "narration" else f"DIALOGUE ({segment.speaker})"
            
            if verbose:
                preview = segment.text[:50] + "..." if len(segment.text) > 50 else segment.text
                print(f"  Segment {i}/{len(segments)} [{segment_type_label}] -> {segment_speaker}")
                print(f"    Preview: {preview}")
            else:
                print(f"  Segment {i}/{len(segments)} [{segment_type_label}] -> {segment_speaker}")
            
            # Synthesize segment
            try:
                kwargs = {"text": segment.text, "file_path": str(segment_path)}
                kwargs["speaker"] = segment_speaker
                tts.tts_to_file(**kwargs)
                segment_files.append(segment_path)
            except Exception as e:
                print(f"    [ERROR] Failed to synthesize segment {i}: {e}")
                # Continue with other segments
                continue
        
        if not segment_files:
            print("[ERROR] All segments failed to synthesize.")
            try:
                shutil.rmtree(temp_segment_dir)
            except Exception:
                pass
            return False
        
        # Combine all segment audio files
        if PYDUB_AVAILABLE and len(segment_files) > 1:
            print(f"\nCombining {len(segment_files)} audio segments...")
            combined = AudioSegment.empty()
            
            for segment_file in segment_files:
                audio = AudioSegment.from_wav(str(segment_file))
                combined += audio
                # Add small pause between segments (shorter than between chunks)
                combined += AudioSegment.silent(duration=200)  # 200ms pause
            
            combined.export(output_path, format="wav")
            print(f"[OK] Combined audio saved to: {output_path}")
            
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_segment_dir)
                print(f"  Cleaned up temporary directory: {temp_segment_dir.name}")
            except Exception as e:
                print(f"  Warning: Could not clean up temporary directory: {e}")
        elif len(segment_files) == 1:
            # Only one segment, just rename it
            segment_files[0].rename(output_path)
            print(f"[OK] Audio saved to: {output_path}")
            
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_segment_dir)
                print(f"  Cleaned up temporary directory: {temp_segment_dir.name}")
            except Exception as e:
                print(f"  Warning: Could not clean up temporary directory: {e}")
        else:
            print(f"[OK] Processed {len(segment_files)} segments (not combined - install pydub to combine)")
            print(f"  Segment files saved in: {temp_segment_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error during auto speaker synthesis: {e}")
        return False

def synthesize_text(text, output_path, model_name=None, chunk_size=5000, is_short_input=False, pronunciations=None, speaker=None, speaker_wav=None, verbose=False, auto_speaker=False, character_voice_mapping=None):
    """
    Synthesize text to speech, handling long texts by chunking.
    
    Args:
        text: Text to synthesize
        output_path: Path to save the output audio file
        model_name: Optional model name. If None, uses a recommended default model.
        chunk_size: Maximum characters per chunk (default: 5000)
        is_short_input: If True, indicates this is a short direct text input
        pronunciations: Optional dictionary of pronunciation mappings to apply
        speaker: Optional speaker name for multi-speaker models
        speaker_wav: Optional path to reference audio file for multi-speaker models
        verbose: If True, output detailed debugging information for each chunk
        auto_speaker: If True, enable automatic speaker detection and dialogue routing
        character_voice_mapping: Optional dictionary mapping character names (pre-pronunciation) to speaker IDs
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
        
        # Check if model is multi-speaker and handle speaker selection
        # Try to detect speakers from various possible attributes
        available_speakers = None
        try:
            if hasattr(tts, 'speakers') and tts.speakers is not None:
                available_speakers = list(tts.speakers) if tts.speakers else None
            elif hasattr(tts, 'speaker_manager') and tts.speaker_manager is not None:
                if hasattr(tts.speaker_manager, 'speaker_names'):
                    available_speakers = list(tts.speaker_manager.speaker_names)
                elif hasattr(tts.speaker_manager, 'speakers'):
                    available_speakers = list(tts.speaker_manager.speakers)
        except Exception:
            pass  # If we can't detect speakers, we'll handle it via error messages
        
        is_multi_speaker = available_speakers is not None and len(available_speakers) > 0
        
        # Handle auto speaker mode
        segmenter = None
        assigner = None
        segments = None
        
        if auto_speaker:
            if not is_multi_speaker:
                print("Error: Auto speaker mode requires a multi-speaker model.")
                return False
            
            print("Auto speaker mode enabled - detecting dialogue and assigning voices...")
            
            # Segment ORIGINAL text first (before pronunciations) to get original character names
            segmenter_original = DialogueSegmenter(verbose=False)  # Don't duplicate verbose output
            original_segments = segmenter_original.segment_text(original_text)
            
            # Extract original character names
            original_character_names = {}
            for seg in original_segments:
                if seg.type == "dialogue" and seg.speaker not in ("NARRATOR", "UNKNOWN"):
                    # Store original name
                    original_character_names[seg.speaker] = seg.speaker
            
            # Now segment the preprocessed text (with pronunciations) for actual synthesis
            segmenter = DialogueSegmenter(verbose=verbose)
            segments = segmenter.segment_text(text)
            
            # Map original character names to post-pronunciation segments
            # Match segments by position to find corresponding original names
            original_seg_by_pos = {(seg.start_pos, seg.end_pos): seg for seg in original_segments}
            for seg in segments:
                if seg.type == "dialogue" and seg.speaker not in ("NARRATOR", "UNKNOWN"):
                    # Try to find matching original segment by position
                    # Allow some tolerance for position changes due to pronunciation length differences
                    best_match = None
                    best_distance = float('inf')
                    for orig_seg in original_segments:
                        if orig_seg.type == "dialogue" and orig_seg.speaker not in ("NARRATOR", "UNKNOWN"):
                            # Calculate position distance
                            pos_distance = abs(seg.start_pos - orig_seg.start_pos) + abs(seg.end_pos - orig_seg.end_pos)
                            if pos_distance < best_distance:
                                best_distance = pos_distance
                                best_match = orig_seg
                    
                    if best_match and best_distance < 100:  # Reasonable tolerance
                        seg.original_speaker = best_match.speaker
                    else:
                        # Fallback: use post-pronunciation name
                        seg.original_speaker = seg.speaker
                else:
                    seg.original_speaker = seg.speaker
            
            if verbose:
                print(f"\n[VERBOSE] Dialogue Segmentation:")
                print(f"  Total segments: {len(segments)}")
                dialogue_count = sum(1 for s in segments if s.type == "dialogue")
                narration_count = sum(1 for s in segments if s.type == "narration")
                print(f"  Dialogue segments: {dialogue_count}")
                print(f"  Narration segments: {narration_count}")
                print()
            
            # Assign speakers
            assigner = SpeakerAssigner(available_speakers, character_voice_mapping=character_voice_mapping)
            character_map = assigner.assign_speakers(segments)
            
            # Build mapping of normalized names to original names for display
            original_name_map = {}
            for seg in segments:
                if seg.type == "dialogue" and seg.speaker not in ("NARRATOR", "UNKNOWN"):
                    if seg.speaker not in original_name_map:
                        original_name_map[seg.speaker] = seg.original_speaker if seg.original_speaker else seg.speaker
            
            # Print speaker assignment summary
            print(f"\nSpeaker Assignment:")
            print(f"  Narrator voice: {assigner.narrator_speaker}")
            print(f"  Characters detected: {len([k for k in character_map.keys() if k != 'UNKNOWN'])}")
            if character_map:
                print(f"\n  Character -> Voice Mapping:")
                for char_name, speaker_id in sorted(character_map.items()):
                    if char_name != 'UNKNOWN':
                        original_name = original_name_map.get(char_name, char_name)
                        if original_name != char_name:
                            print(f"    {original_name} (as \"{char_name}\") -> {speaker_id}")
                        else:
                            print(f"    {char_name} -> {speaker_id}")
                if 'UNKNOWN' in character_map:
                    print(f"    UNKNOWN -> {character_map['UNKNOWN']}")
            if segmenter.unknown_count > 0:
                print(f"\n  Warning: {segmenter.unknown_count} dialogue segments with unknown speaker")
            if assigner.warnings:
                for warning in assigner.warnings:
                    print(f"  Warning: {warning}")
            print()
        
        elif is_multi_speaker:
            if not speaker and not speaker_wav:
                # Try to use first available speaker as default
                speaker = available_speakers[0]
                print(f"Multi-speaker model detected. Using default speaker: {speaker}")
                print(f"  Available speakers: {', '.join(available_speakers[:10])}{'...' if len(available_speakers) > 10 else ''}")
                print(f"  (Use --speaker to specify a different speaker)")
            elif speaker:
                # Validate speaker exists
                if speaker not in available_speakers:
                    print(f"Warning: Speaker '{speaker}' not found in model.")
                    print(f"  Available speakers: {', '.join(available_speakers[:10])}{'...' if len(available_speakers) > 10 else ''}")
                    print(f"  Using first available speaker: {available_speakers[0]}")
                    speaker = available_speakers[0]
                else:
                    print(f"Using speaker: {speaker}")
            elif speaker_wav:
                if not os.path.exists(speaker_wav):
                    print(f"Error: Reference audio file '{speaker_wav}' not found.")
                    return False
                print(f"Using reference audio: {speaker_wav}")
        
        # Show text info
        print(f"Text length: {len(text)} characters")
        
        # If auto speaker mode, synthesize by segments; otherwise use chunking
        if auto_speaker and segments:
            return _synthesize_with_auto_speaker(tts, segments, assigner, output_path, verbose)
        
        # Split text into chunks if necessary
        chunks = split_text_into_chunks(text, max_chunk_size=chunk_size)
        
        if len(chunks) == 1:
            # Single chunk - process directly
            print(f"Synthesizing text...")
            
            # Verbose output for single chunk
            if verbose:
                chunk_info = get_chunk_verbose_info(text, text, chunk_index=1, total_chunks=1)
                total_words = len([w for w in text.split() if w.strip()])
                word_range_str = f"Words {chunk_info['word_start']}-{chunk_info['word_end']} ({total_words} total)" if chunk_info['word_start'] and chunk_info['word_end'] else ""
                char_range_str = f"Characters: 1-{len(text)} ({len(text)} total)"
                range_str = ", ".join(filter(None, [word_range_str, char_range_str]))
                print(f"  [VERBOSE] Single chunk details ({chunk_info['word_count']} words, {chunk_info['char_count']} characters):")
                print(f"    \"{chunk_info['first_words']}\" ... \"{chunk_info['last_words']}\"")
                if range_str:
                    print(f"    {range_str}")
            
            kwargs = {"text": text, "file_path": output_path}
            if speaker:
                kwargs["speaker"] = speaker
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
            tts.tts_to_file(**kwargs)
            
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
            
            # Create a hidden temporary directory for chunk files
            temp_chunk_dir = output_dir / f".temp_chunks_{output_stem}"
            temp_chunk_dir.mkdir(parents=True, exist_ok=True)
            
            chunk_files_dict = {}  # Dict of index -> path to maintain order
            failed_indices = set()  # Track which chunk indices failed
            
            # First pass: try to process all chunks
            for i, chunk in enumerate(merged_chunks, 1):
                chunk_path = temp_chunk_dir / f"{output_stem}_chunk_{i:04d}.wav"
                words = [w for w in chunk.split() if w.strip()]
                word_count = len(words)
                print(f"  Processing chunk {i}/{len(merged_chunks)} ({word_count} words, {len(chunk)} chars)...")
                
                # Verbose output
                if verbose:
                    chunk_info = get_chunk_verbose_info(chunk, text, chunk_index=i, total_chunks=len(merged_chunks))
                    total_words = len([w for w in text.split() if w.strip()])
                    word_range_str = f"Words {chunk_info['word_start']}-{chunk_info['word_end']} ({total_words} total)" if chunk_info['word_start'] and chunk_info['word_end'] else ""
                    char_range_str = f"Characters: {chunk_info['char_start']}-{chunk_info['char_end']} ({len(text)} total)" if chunk_info['char_start'] and chunk_info['char_end'] else ""
                    range_str = ", ".join(filter(None, [word_range_str, char_range_str]))
                    print(f"    [VERBOSE] Chunk {i}/{len(merged_chunks)} details ({chunk_info['word_count']} words, {chunk_info['char_count']} characters):")
                    print(f"      \"{chunk_info['first_words']}\" ... \"{chunk_info['last_words']}\"")
                    if range_str:
                        print(f"      {range_str}")
                
                # Chunks are already preprocessed, so use them directly
                if synthesize_text_chunk(tts, chunk, str(chunk_path), speaker=speaker, speaker_wav=speaker_wav):
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
                    retry_path = temp_chunk_dir / f"{output_stem}_retry_{failed_idx:04d}.wav"
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
                            
                            # Verbose output for retry
                            if verbose:
                                chunk_info = get_chunk_verbose_info(merged_retry, text, chunk_index=failed_idx, total_chunks=len(merged_chunks))
                                total_words = len([w for w in text.split() if w.strip()])
                                word_range_str = f"Words {chunk_info['word_start']}-{chunk_info['word_end']} ({total_words} total)" if chunk_info['word_start'] and chunk_info['word_end'] else ""
                                char_range_str = f"Characters: {chunk_info['char_start']}-{chunk_info['char_end']} ({len(text)} total)" if chunk_info['char_start'] and chunk_info['char_end'] else ""
                                range_str = ", ".join(filter(None, [word_range_str, char_range_str]))
                                print(f"      [VERBOSE] Retry chunk {failed_idx} details ({chunk_info['word_count']} words, {chunk_info['char_count']} characters):")
                                print(f"        \"{chunk_info['first_words']}\" ... \"{chunk_info['last_words']}\"")
                                if range_str:
                                    print(f"        {range_str}")
                                print(f"        Merged chunks: {sorted(merged_indices)}")
                            
                            if synthesize_text_chunk(tts, merged_retry, str(retry_path), speaker=speaker, speaker_wav=speaker_wav):
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
                            
                            # Verbose output for backward merge retry
                            if verbose:
                                chunk_info = get_chunk_verbose_info(merged_retry, text, chunk_index=failed_idx, total_chunks=len(merged_chunks))
                                total_words = len([w for w in text.split() if w.strip()])
                                word_range_str = f"Words {chunk_info['word_start']}-{chunk_info['word_end']} ({total_words} total)" if chunk_info['word_start'] and chunk_info['word_end'] else ""
                                char_range_str = f"Characters: {chunk_info['char_start']}-{chunk_info['char_end']} ({len(text)} total)" if chunk_info['char_start'] and chunk_info['char_end'] else ""
                                range_str = ", ".join(filter(None, [word_range_str, char_range_str]))
                                print(f"      [VERBOSE] Retry chunk {failed_idx} details (backward merge) ({chunk_info['word_count']} words, {chunk_info['char_count']} characters):")
                                print(f"        \"{chunk_info['first_words']}\" ... \"{chunk_info['last_words']}\"")
                                if range_str:
                                    print(f"        {range_str}")
                                print(f"        Merged chunks: {sorted(merged_indices)}")
                            
                            if synthesize_text_chunk(tts, merged_retry, str(retry_path), speaker=speaker, speaker_wav=speaker_wav):
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
                # Clean up temporary directory even on failure
                try:
                    shutil.rmtree(temp_chunk_dir)
                    print(f"  Cleaned up temporary directory: {temp_chunk_dir.name}")
                except Exception as cleanup_error:
                    print(f"  Warning: Could not clean up temporary directory: {cleanup_error}")
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
                
                # Clean up temporary directory and all chunk files
                try:
                    shutil.rmtree(temp_chunk_dir)
                    print(f"  Cleaned up temporary directory: {temp_chunk_dir.name}")
                except Exception as e:
                    print(f"  Warning: Could not clean up temporary directory: {e}")
            elif len(chunk_files) == 1:
                # Only one chunk succeeded, just rename it
                chunk_files[0].rename(output_path)
                print(f"[OK] Audio saved to: {output_path}")
                
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_chunk_dir)
                    print(f"  Cleaned up temporary directory: {temp_chunk_dir.name}")
                except Exception as e:
                    print(f"  Warning: Could not clean up temporary directory: {e}")
            else:
                print(f"[OK] Processed {len(chunk_files)} chunks (not combined - install pydub to combine)")
                print(f"  Chunk files saved in: {temp_chunk_dir}")
            
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

def identify_text_structure(text, model_name=None, chunk_size=5000, pronunciations=None, auto_speaker=False, verbose=False, character_voice_mapping=None):
    """
    Identify text structure (chunks, speakers, dialogue) without synthesizing.
    
    Args:
        text: Text to analyze
        model_name: Optional model name (needed to get available speakers)
        chunk_size: Maximum characters per chunk
        pronunciations: Optional pronunciation mappings
        auto_speaker: If True, perform speaker detection and assignment
        verbose: If True, output detailed debugging information
        character_voice_mapping: Optional dictionary mapping character names (pre-pronunciation) to speaker IDs
        
    Returns:
        Dictionary with identification results
    """
    results = {
        'chunks': [],
        'total_chunks': 0,
        'total_characters': len(text),
        'total_words': len([w for w in text.split() if w.strip()]),
        'speakers': {},
        'segments': [],
        'unknown_dialogue_count': 0
    }
    
    # Preprocess text
    original_text = text
    text = preprocess_text(text, is_short_input=False, pronunciations=pronunciations)
    
    # Split into chunks
    chunks = split_text_into_chunks(text, max_chunk_size=chunk_size)
    results['total_chunks'] = len(chunks)
    results['chunks'] = [{'index': i+1, 'text': chunk[:100] + '...' if len(chunk) > 100 else chunk, 
                          'length': len(chunk), 'word_count': len([w for w in chunk.split() if w.strip()])} 
                         for i, chunk in enumerate(chunks)]
    
    # If auto_speaker is enabled, perform dialogue segmentation and speaker assignment
    if auto_speaker:
        # Initialize TTS model to get available speakers
        try:
            if model_name:
                tts = TTS(model_name=model_name, progress_bar=False)
            else:
                default_model = "tts_models/en/ljspeech/tacotron2-DDC"
                tts = TTS(model_name=default_model, progress_bar=False)
            
            # Get available speakers
            available_speakers = None
            try:
                if hasattr(tts, 'speakers') and tts.speakers is not None:
                    available_speakers = list(tts.speakers) if tts.speakers else None
                elif hasattr(tts, 'speaker_manager') and tts.speaker_manager is not None:
                    if hasattr(tts.speaker_manager, 'speaker_names'):
                        available_speakers = list(tts.speaker_manager.speaker_names)
                    elif hasattr(tts.speaker_manager, 'speakers'):
                        available_speakers = list(tts.speaker_manager.speakers)
            except Exception:
                pass
            
            if not available_speakers or len(available_speakers) == 0:
                print("Warning: No speakers available in model. Auto speaker mode requires a multi-speaker model.")
                results['error'] = "No speakers available"
                return results
            
            # Segment ORIGINAL text first (before pronunciations) to get original character names
            if verbose:
                print("\n[VERBOSE] Starting dialogue segmentation...")
            segmenter_original = DialogueSegmenter(verbose=False)
            original_segments = segmenter_original.segment_text(original_text)
            
            # Now segment the preprocessed text (with pronunciations) for display
            segmenter = DialogueSegmenter(verbose=verbose)
            segments = segmenter.segment_text(text)
            
            # Map original character names to post-pronunciation segments
            original_seg_by_pos = {(seg.start_pos, seg.end_pos): seg for seg in original_segments}
            for seg in segments:
                if seg.type == "dialogue" and seg.speaker not in ("NARRATOR", "UNKNOWN"):
                    # Try to find matching original segment by position
                    best_match = None
                    best_distance = float('inf')
                    for orig_seg in original_segments:
                        if orig_seg.type == "dialogue" and orig_seg.speaker not in ("NARRATOR", "UNKNOWN"):
                            pos_distance = abs(seg.start_pos - orig_seg.start_pos) + abs(seg.end_pos - orig_seg.end_pos)
                            if pos_distance < best_distance:
                                best_distance = pos_distance
                                best_match = orig_seg
                    
                    if best_match and best_distance < 100:
                        seg.original_speaker = best_match.speaker
                    else:
                        seg.original_speaker = seg.speaker
                else:
                    seg.original_speaker = seg.speaker
            
            if verbose:
                print(f"\n[VERBOSE] Segmentation complete:")
                print(f"  Total segments: {len(segments)}")
                dialogue_count = sum(1 for s in segments if s.type == "dialogue")
                narration_count = sum(1 for s in segments if s.type == "narration")
                print(f"  Dialogue segments: {dialogue_count}")
                print(f"  Narration segments: {narration_count}")
                print()
            results['segments'] = [{'type': s.type, 'speaker': s.speaker, 'original_speaker': s.original_speaker if s.original_speaker else s.speaker, 'text': s.text[:50] + '...' if len(s.text) > 50 else s.text} 
                                   for s in segments]
            results['unknown_dialogue_count'] = segmenter.unknown_count
            
            # Assign speakers
            assigner = SpeakerAssigner(available_speakers, character_voice_mapping=character_voice_mapping)
            character_map = assigner.assign_speakers(segments)
            
            # Build mapping of normalized names to original names for display
            original_name_map = {}
            for seg in segments:
                if seg.type == "dialogue" and seg.speaker not in ("NARRATOR", "UNKNOWN"):
                    if seg.speaker not in original_name_map:
                        original_name_map[seg.speaker] = seg.original_speaker if seg.original_speaker else seg.speaker
            
            # Build speaker mapping (include narrator)
            results['speakers'] = {
                'NARRATOR': assigner.narrator_speaker,
                **character_map
            }
            results['original_name_map'] = original_name_map  # Store original names for display
            results['warnings'] = assigner.warnings
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Error during identification: {e}")
    
    return results

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
  
  # Use a multi-speaker model with speaker selection:
  python book-to-audiobook_coquitts.py input.txt -m tts_models/en/vctk/vits --speaker p225
  
  # Use a multi-speaker model with reference audio:
  python book-to-audiobook_coquitts.py input.txt -m tts_models/en/vctk/vits --speaker-wav reference.wav
  
  # Use automatic speaker detection and dialogue routing:
  python book-to-audiobook_coquitts.py input.txt -m tts_models/en/vctk/vits --auto-speaker
  
  # Or use --speaker auto:
  python book-to-audiobook_coquitts.py input.txt -m tts_models/en/vctk/vits --speaker auto
  
  # Identify text structure without synthesizing:
  python book-to-audiobook_coquitts.py input.txt -m tts_models/en/vctk/vits --identification-only --auto-speaker
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
    parser.add_argument(
        '--character-voice-mapping', '-cvm',
        type=str,
        help='Path to JSON file with character-to-voice mappings (keys: character name pre-pronunciation, values: speaker ID)'
    )
    parser.add_argument(
        '--speaker', '-s',
        type=str,
        help='Speaker name for multi-speaker models (e.g., "p225" for VCTK models, or "auto" for automatic speaker detection)'
    )
    parser.add_argument(
        '--auto-speaker', '-a',
        action='store_true',
        help='Enable automatic speaker detection and dialogue routing (equivalent to --speaker auto)'
    )
    parser.add_argument(
        '--speaker-wav',
        type=str,
        help='Path to reference audio file for multi-speaker models (alternative to --speaker)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed debugging information for each chunk'
    )
    parser.add_argument(
        '--identification-only', '-i',
        action='store_true',
        help='Only identify chunks and speakers without synthesizing audio (useful for planning)'
    )
    
    args = parser.parse_args()
    
    # Determine auto speaker mode
    # Precedence: --auto-speaker wins over --speaker unless --speaker is "auto"
    auto_speaker_mode = False
    manual_speaker = None
    
    if args.auto_speaker:
        auto_speaker_mode = True
        if args.speaker and args.speaker.lower() != "auto":
            print(f"Info: --auto-speaker enabled, overriding --speaker {args.speaker}")
    elif args.speaker and args.speaker.lower() == "auto":
        auto_speaker_mode = True
    elif args.speaker:
        manual_speaker = args.speaker
    
    print("Book to Audiobook Converter (CoquiTTS)\n" + "="*50)
    
    # Show NLTK status
    if NLTK_AVAILABLE:
        print("NLTK: Available (using advanced sentence tokenization)")
    elif NLTK_SETUP_MESSAGE:
        print(f"NLTK: {NLTK_SETUP_MESSAGE}")
    print()  # Blank line for readability
    
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
    
    # Determine speaker identifier for filename
    # If speaker_wav is provided, use a short identifier based on the filename
    speaker_for_filename = manual_speaker if not auto_speaker_mode else "auto"
    if args.speaker_wav and not speaker_for_filename:
        # Extract a short identifier from the speaker_wav filename
        speaker_path = Path(args.speaker_wav)
        speaker_for_filename = f"ref_{speaker_path.stem[:20]}"  # Limit length
    
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
        output_file = generate_unique_filename(safe_name, model=args.model, speaker=speaker_for_filename)
    else:
        output_file = get_output_filename(input_source, model=args.model, speaker=speaker_for_filename)
    
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
    
    # Load character-to-voice mapping if provided
    character_voice_mapping = None
    if args.character_voice_mapping:
        character_voice_mapping = load_character_voice_mapping(args.character_voice_mapping)
        if character_voice_mapping is None:
            print("Warning: Could not load character voice mapping file. Continuing without custom voice assignments.")
        print()  # Add blank line for readability
    
    # Handle identification-only mode
    if args.identification_only:
        print("="*50)
        print("IDENTIFICATION MODE - Analyzing text structure only")
        print("="*50 + "\n")
        
        results = identify_text_structure(
            text,
            model_name=args.model,
            chunk_size=args.chunk_size,
            pronunciations=pronunciations,
            auto_speaker=auto_speaker_mode,
            verbose=args.verbose,
            character_voice_mapping=character_voice_mapping
        )
        
        print(f"Text Analysis Results:")
        print(f"  Total characters: {results['total_characters']}")
        print(f"  Total words: {results['total_words']}")
        print(f"  Total chunks: {results['total_chunks']}\n")
        
        if results['total_chunks'] > 0:
            print("Chunk Summary:")
            for chunk_info in results['chunks']:
                print(f"  Chunk {chunk_info['index']}: {chunk_info['word_count']} words, {chunk_info['length']} characters")
                if args.verbose:
                    print(f"    Preview: {chunk_info['text']}")
            print()
        
        if auto_speaker_mode:
            if 'error' in results:
                print(f"Error during speaker identification: {results['error']}")
            else:
                print("Speaker Detection Results:")
                if results['speakers']:
                    print(f"  Narrator voice: {results['speakers'].get('NARRATOR', 'N/A')}")
                    print(f"  Characters detected: {len([k for k in results['speakers'].keys() if k != 'NARRATOR'])}")
                    print("\n  Character -> Voice Mapping:")
                    original_name_map = results.get('original_name_map', {})
                    for char_name, speaker_id in sorted(results['speakers'].items()):
                        if char_name != 'NARRATOR':
                            original_name = original_name_map.get(char_name, char_name)
                            if original_name != char_name:
                                print(f"    {original_name} (as \"{char_name}\") -> {speaker_id}")
                            else:
                                print(f"    {char_name} -> {speaker_id}")
                    if results.get('unknown_dialogue_count', 0) > 0:
                        print(f"\n  Warning: {results['unknown_dialogue_count']} dialogue segments with unknown speaker")
                    if results.get('warnings'):
                        for warning in results['warnings']:
                            print(f"  Warning: {warning}")
                else:
                    print("  No speakers detected (may need a multi-speaker model)")
        else:
            print("Auto speaker mode not enabled. Use --auto-speaker or --speaker auto to enable.")
        
        print("\n" + "="*50)
        print("Identification complete. No audio was synthesized.")
        sys.exit(0)
    
    # Synthesize text to speech
    # Mark as short input if text was provided directly (likely a single word/phrase)
    is_short = args.text is not None
    success = synthesize_text(
        text, 
        output_file, 
        model_name=args.model,
        chunk_size=args.chunk_size,
        is_short_input=is_short,
        pronunciations=pronunciations,
        speaker=manual_speaker if not auto_speaker_mode else None,
        speaker_wav=args.speaker_wav,
        verbose=args.verbose,
        auto_speaker=auto_speaker_mode,
        character_voice_mapping=character_voice_mapping
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
