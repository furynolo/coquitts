"""
Text processing and file I/O utilities.
"""
import re
import unicodedata
import sys
from typing import List, Dict, Optional

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
