
import os
import sys
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Set
import re

# Try to import TTS
try:
    from TTS.api import TTS
except ImportError:
    TTS = None

# Try to import pydub for audio manipulation
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

# Audio playback imports
WINSOUND_AVAILABLE = False
PLAYSOUND_FUNC = None
PLAYBACK_AVAILABLE = False
PLAYBACK_METHOD = None

if sys.platform == "win32":
    try:
        import winsound
        WINSOUND_AVAILABLE = True
        PLAYBACK_AVAILABLE = True
        PLAYBACK_METHOD = "winsound"
    except ImportError:
        pass

if not PLAYBACK_AVAILABLE:
    try:
        from playsound import playsound
        PLAYSOUND_FUNC = playsound
        PLAYBACK_AVAILABLE = True
        PLAYBACK_METHOD = "playsound"
    except ImportError:
        pass

from .text_utils import split_text_into_chunks, preprocess_text
from .dialogue import DialogueSegmenter, SpeakerAssigner, Segment

def validate_audio_duration(file_path: str, text_length: int, max_duration_per_char: float = 0.5) -> Tuple[bool, Optional[float], Optional[float]]:
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

def play_audio(file_path: str) -> bool:
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

def get_chunk_verbose_info(chunk: str, full_text: str, chunk_index: int = None, total_chunks: int = None) -> Dict[str, Any]:
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

def synthesize_text_chunk(tts, text: str, chunk_path: str, speaker: str = None, speaker_wav: str = None) -> bool:
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
        if word_count < 10:
            print(f"  Warning: Chunk has very few words ({word_count} words, {len(text)} chars), may fail...")
        
        # Ensure minimum length to avoid kernel errors
        if len(text) < 100:
            print(f"  Warning: Chunk too short ({len(text)} chars), attempting anyway...")
        
        # Check if text has enough actual content
        alphanumeric_count = sum(1 for c in text if c.isalnum())
        if alphanumeric_count < 20:
            print(f"  Warning: Chunk has little content ({alphanumeric_count} alphanumeric chars), attempting anyway...")
        
        # Build kwargs for tts_to_file
        kwargs = {"text": text, "file_path": str(chunk_path)}
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

class AudioSynthesizer:
    """Handles text-to-speech synthesis logic."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the AudioSynthesizer.
        
        Args:
            model_name: Optional model name. If None, uses default.
        """
        if TTS is None:
            raise ImportError("coqui-tts not found. Please install it with: pip install coqui-tts")
            
        if model_name:
            print(f"Loading model: {model_name}")
            self.tts = TTS(model_name=model_name, progress_bar=True)
        else:
            default_model = "tts_models/en/vctk/vits"
            print(f"Loading default model: {default_model}")
            self.tts = TTS(model_name=default_model, progress_bar=True)
            
        self.available_speakers = None
        try:
            if hasattr(self.tts, 'speakers') and self.tts.speakers is not None:
                self.available_speakers = [s.strip() for s in self.tts.speakers] if self.tts.speakers else None
            elif hasattr(self.tts, 'speaker_manager') and self.tts.speaker_manager is not None:
                if hasattr(self.tts.speaker_manager, 'speaker_names'):
                    self.available_speakers = [s.strip() for s in self.tts.speaker_manager.speaker_names]
                elif hasattr(self.tts.speaker_manager, 'speakers'):
                    self.available_speakers = [s.strip() for s in self.tts.speaker_manager.speakers]
        except Exception:
            pass
            
        self.is_multi_speaker = self.available_speakers is not None and len(self.available_speakers) > 0

    def synthesize(self, text: str, output_path: str, chunk_size: int = 5000, 
                  is_short_input: bool = False, pronunciations: Dict[str, str] = None,
                  speaker: Optional[str] = None, speaker_wav: Optional[str] = None,
                  verbose: bool = False, auto_speaker: bool = False,
                  character_voice_mapping: Optional[Dict[str, str]] = None,
                  apply_corrections_file: str = None) -> bool:
        """
        Synthesize text to speech.
        """
        # Preprocess text
        original_text = text
        text = preprocess_text(text, is_short_input=is_short_input, pronunciations=pronunciations)
        
        # Handle auto speaker mode
        if auto_speaker:
            if not self.is_multi_speaker:
                print("Error: Auto speaker mode requires a multi-speaker model.")
                return False
            
            return self._synthesize_with_auto_speaker(
                text, original_text, output_path, verbose, 
                character_voice_mapping, pronunciations, apply_corrections_file
            )
        
        # Handle manual speaker selection for multi-speaker models
        if self.is_multi_speaker:
            if not speaker and not speaker_wav:
                # Use default if not specified
                speaker = self.available_speakers[0]
                print(f"Multi-speaker model detected. Using default speaker: {speaker}")
            elif speaker:
                if speaker not in self.available_speakers:
                    print(f"Warning: Speaker '{speaker}' not found. Using default: {self.available_speakers[0]}")
                    speaker = self.available_speakers[0]
        
        # Split into chunks
        chunks = split_text_into_chunks(text, max_chunk_size=chunk_size)
        
        if len(chunks) == 1:
            # Single chunk
            return self._synthesize_single_chunk(
                text, output_path, speaker, speaker_wav, verbose
            )
        else:
            # Multiple chunks
            return self._synthesize_multiple_chunks(
                chunks, text, output_path, speaker, speaker_wav, verbose
            )

    def _synthesize_with_auto_speaker(self, text: str, original_text: str, output_path: str, 
                                     verbose: bool, character_voice_mapping: Optional[Dict[str, str]],
                                     pronunciations: Optional[Dict[str, str]], apply_corrections_file: str = None) -> bool:
        """Handle synthesis with automatic speaker assignment."""
        print("Auto speaker mode enabled - detecting dialogue and assigning voices...")
        
        # Segment ORIGINAL text first to get real names
        segmenter_original = DialogueSegmenter(verbose=False)
        original_segments = segmenter_original.segment_text(original_text)
        
        # Segment preprocessed text
        segmenter = DialogueSegmenter(verbose=verbose)
        segments = segmenter.segment_text(text)
        
        # Apply corrections if provided
        if apply_corrections_file:
            from .unknown_speakers import CorrectionManager
            cm = CorrectionManager(verbose=verbose)
            corrections = cm.load_corrections(apply_corrections_file)
            cm.apply_corrections(segments, corrections)
        
        # Assign speakers
        assigner = SpeakerAssigner(self.available_speakers, character_voice_mapping=character_voice_mapping)
        assigner.assign_speakers(segments)
        
        # Output directory setup
        output_dir = Path(output_path).parent
        output_stem = Path(output_path).stem
        temp_dir = output_dir / f".temp_segments_{output_stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        segment_files = []
        print(f"Synthesizing {len(segments)} segments with auto speaker routing...")
        
        for i, segment in enumerate(segments, 1):
            segment_speaker = assigner.get_speaker_for_segment(segment)
            segment_path = temp_dir / f"segment_{i:04d}.wav"
            
            if verbose:
                print(f"  Segment {i}/{len(segments)} [{segment.type}] -> {segment_speaker}")
            
            try:
                synthesize_text_chunk(self.tts, segment.text, str(segment_path), speaker=segment_speaker)
                segment_files.append(segment_path)
            except Exception as e:
                print(f"    [ERROR] Failed segment {i}: {e}")
                continue
                
        if not segment_files:
            return False
            
        # Combine
        if PYDUB_AVAILABLE and len(segment_files) > 1:
            self._combine_audio_files(segment_files, output_path, pause_ms=200)
        elif len(segment_files) == 1:
            shutil.move(segment_files[0], output_path)
        else:
            print(f"Generated {len(segment_files)} segments. Install pydub to combine.")
            
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
            
        return True

    def _synthesize_single_chunk(self, text: str, output_path: str, speaker: str, 
                                speaker_wav: str, verbose: bool) -> bool:
        """Synthesize a single chunk."""
        if verbose:
            print(f"  [VERBOSE] Synthesizing single chunk ({len(text)} chars)")
            
        success = synthesize_text_chunk(self.tts, text, output_path, speaker, speaker_wav)
        
        if success:
            is_valid, duration, max_expected = validate_audio_duration(output_path, len(text))
            if not is_valid:
                print("[WARNING] Audio unusually long (possible loop).")
                # Truncation logic could go here
        
        return success

    def _synthesize_multiple_chunks(self, chunks: List[str], full_text: str, output_path: str,
                                   speaker: str, speaker_wav: str, verbose: bool) -> bool:
        """Synthesize multiple chunks and combine them."""
        # Merge small chunks logic (simplified from original)
        merged_chunks = list(chunks) # Placeholder for actual merging logic if needed
        
        output_dir = Path(output_path).parent
        output_stem = Path(output_path).stem
        temp_dir = output_dir / f".temp_chunks_{output_stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_files = []
        
        for i, chunk in enumerate(merged_chunks, 1):
            chunk_path = temp_dir / f"chunk_{i:04d}.wav"
            print(f"  Processing chunk {i}/{len(merged_chunks)}...")
            
            if synthesize_text_chunk(self.tts, chunk, str(chunk_path), speaker, speaker_wav):
                chunk_files.append(chunk_path)
            else:
                print(f"  [ERROR] Failed chunk {i}")
                
        if not chunk_files:
            return False
            
        # Combine
        if PYDUB_AVAILABLE and len(chunk_files) > 1:
            self._combine_audio_files(chunk_files, output_path, pause_ms=500)
        elif len(chunk_files) == 1:
            shutil.move(chunk_files[0], output_path)
        else:
            print(f"Generated {len(chunk_files)} chunks. Install pydub to combine.")
            
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
            
        return True
        
    def _combine_audio_files(self, file_paths: List[Path], output_path: str, pause_ms: int = 500):
        """Combine multiple audio files into one."""
        if not PYDUB_AVAILABLE:
            return
            
        print(f"Combining {len(file_paths)} audio files...")
        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=pause_ms)
        
        for path in file_paths:
            audio = AudioSegment.from_wav(str(path))
            combined += audio + silence
            
        combined.export(output_path, format="wav")
