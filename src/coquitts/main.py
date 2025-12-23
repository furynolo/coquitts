
import os
import sys
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from .config import load_pronunciations, load_character_voice_mapping
from .text_utils import read_text_file, preprocess_text, split_text_into_chunks
from .dialogue import DialogueSegmenter, SpeakerAssigner
from .audio import AudioSynthesizer, play_audio, get_chunk_verbose_info

def parse_args():
    parser = argparse.ArgumentParser(description="Convert text file to audiobook using Coqui TTS")
    
    # Input/Output arguments
    parser.add_argument("input_file", nargs="?", help="Path to input text file")
    parser.add_argument("--text", "-t", help="Direct text input (overrides input file)")
    parser.add_argument("--output", "-o", help="Path to output audio file")
    
    # TTS Configuration
    parser.add_argument("--model", "-m", help="Name of the TTS model to use")
    parser.add_argument("--speaker", "-s", help="Speaker ID for multi-speaker models (use 'auto' for automatic dialogue routing)")
    parser.add_argument("--speaker-wav", "-w", help="Path to reference audio for voice cloning")
    parser.add_argument("--auto-speaker", "-a", action="store_true", help="Enable automatic speaker detection and dialogue routing")
    parser.add_argument("--character-voice-mapping", "-c", help="JSON file mapping character names to speaker IDs")
    parser.add_argument("--generate-corrections", "-g", action="store_true", help="Generate a JSON file for manual unknown speaker corrections")
    parser.add_argument("--apply-corrections", type=str, help="Path to JSON file with speaker corrections to apply")
    
    # Processing Configuration
    parser.add_argument("--chunk-size", type=int, default=5000, help="Maximum characters per chunk")
    parser.add_argument("--pronunciations", "-p", nargs="+", help="JSON file(s) containing pronunciation replacements")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Playback/Action
    parser.add_argument("--play", action="store_true", help="Play the audio after generation")
    parser.add_argument("--identification-only", "--identify", "-i", action="store_true", help="Analyze text structure and speakers without generating audio")
    
    return parser.parse_args()

def identify_text_structure(text: str, model_name: str = None, chunk_size: int = 5000, 
                          pronunciations: dict = None, auto_speaker: bool = False, 
                          verbose: bool = False, character_voice_mapping: dict = None,
                          generate_corrections: bool = False, apply_corrections_file: str = None,
                          input_file_path: str = None) -> Dict[str, Any]:
    """
    Identify text structure (chunks, speakers, dialogue) without synthesizing.
    """
    results = {
        'total_characters': len(text),
        'total_words': len(text.split()),
        'chunks': [],
        'speakers': None,
        'warnings': [],
        'unknown_dialogue_count': 0,
        'unknown_segments': [],
        'suspicious_speakers': []
    }
    
    # Preprocess text
    original_text = text
    if pronunciations:
        text = preprocess_text(text, pronunciations=pronunciations)
    
    # Split chunks
    chunks = split_text_into_chunks(text, max_chunk_size=chunk_size)
    results['total_chunks'] = len(chunks)
    
    for i, chunk in enumerate(chunks, 1):
        chunk_len = len(chunk)
        chunk_words = len(chunk.split())
        results['chunks'].append({
            'index': i,
            'length': chunk_len,
            'word_count': chunk_words,
            'text': chunk[:50] + "..." if len(chunk) > 50 else chunk
        })
        
    # Auto speaker analysis
    if auto_speaker:
        try:
            # Need to initialize synthesizer to get available speakers
            print("Loading model for speaker analysis...")
            synth = AudioSynthesizer(model_name)
            available_speakers = synth.available_speakers
            
            if not available_speakers:
                results['warnings'].append("Model does not appear to support multi-speaker output.")
                return results
                
            print("Detecting dialogue and assigning voices...")
            
            # Segment ORIGINAL text to get original names
            segmenter_orig = DialogueSegmenter(verbose=False)
            orig_segments = segmenter_orig.segment_text(original_text)
            
            # Segment processed text
            # Segment processed text
            segmenter = DialogueSegmenter(verbose=verbose)
            segments = segmenter.segment_text(text)
            
            # Apply corrections if provided
            if apply_corrections_file:
                from .unknown_speakers import CorrectionManager
                cm = CorrectionManager(verbose=verbose)
                corrections = cm.load_corrections(apply_corrections_file)
                cm.apply_corrections(segments, corrections)
            
            # Generate corrections file if requested
            if generate_corrections:
                from .unknown_speakers import CorrectionManager
                cm = CorrectionManager(verbose=verbose)
                
                # Determine output path
                if input_file_path:
                    p = Path(input_file_path)
                    filename_stem = p.stem
                else:
                    filename_stem = "output_speech"
                
                # Create corrections directory if it doesn't exist
                corrections_dir = Path("output/corrections")
                corrections_dir.mkdir(parents=True, exist_ok=True)
                
                json_output = str(corrections_dir / f"{filename_stem}_corrections.json")
                    
                cm.export_corrections(segments, text, json_output)
                results['correction_file'] = json_output
            
            # Build original name map
            original_name_map = {}
            # (Simplified mapping logic for display)
            for seg in segments:
                if seg.type == "dialogue" and seg.speaker not in ("NARRATOR", "UNKNOWN"):
                    original_name_map[seg.speaker] = seg.speaker # Default
            
            results['unknown_dialogue_count'] = segmenter.unknown_count
            results['unknown_segments'] = segmenter.unknown_segments
            results['original_name_map'] = original_name_map
            
            # Assign speakers
            assigner = SpeakerAssigner(available_speakers, character_voice_mapping=character_voice_mapping)
            character_map = assigner.assign_speakers(segments)
            
            results['speakers'] = character_map
            results['warnings'].extend(assigner.warnings)
            
        except Exception as e:
            results['error'] = str(e)
            
    return results

def main():
    args = parse_args()
    
    # Input validation
    if not args.input_file and not args.text:
        print("Error: detailed input required. Provide input file or text string.")
        return
        
    if args.text:
        input_source = f"Direct Text Input ({len(args.text)} chars)"
    else:
        input_source = args.input_file
    
    # Define output path
    # Determine model name for filename and synthesis
    # Matches default in audio.py
    model_name = args.model if args.model else "tts_models/en/vctk/vits" 
    # Sanitize model name for filename
    model_name_safe = model_name.replace("/", "_").replace("\\", "_")
    
    # Determine speaker suffix
    speaker_suffix = ""
    if args.auto_speaker or (args.speaker and args.speaker.lower() == 'auto'):
        speaker_suffix = "_autospeaker"
    elif args.speaker:
        speaker_suffix = f"_{args.speaker}"
    elif args.speaker_wav:
        speaker_suffix = "_cloned"
    
    if args.output:
        output_file = Path(args.output)
    else:
        # Default output directory is 'output/audio' in the current working directory
        output_dir = Path("output/audio")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.input_file:
            input_path = Path(args.input_file)
            input_stem = input_path.stem
            output_filename = f"{input_stem}_{model_name_safe}{speaker_suffix}_{timestamp}.wav"
        else:
            # Use snippets of text for filename if available
            raw_snippet = args.text[:30] if args.text else "output_speech"
            # Sanitize: keep only alphanumerics, replace others with underscore
            sanitized_snippet = re.sub(r'[^\w]+', '_', raw_snippet).strip('_').lower()
            
            if not sanitized_snippet:
                sanitized_snippet = "output_speech"
                
            output_filename = f"{sanitized_snippet}_{model_name_safe}{speaker_suffix}_{timestamp}.wav"
            
        output_file = output_dir / output_filename
            
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Read text
    if args.text:
        text = args.text
    else:
        text = read_text_file(args.input_file)
        if text is None:
            return
            
    # Load configuration
    pronunciations = {}
    if args.pronunciations:
        for p_file in args.pronunciations:
            loaded = load_pronunciations(p_file)
            if loaded:
                pronunciations.update(loaded)
                
    character_voice_mapping = None
    if args.character_voice_mapping:
        character_voice_mapping = load_character_voice_mapping(args.character_voice_mapping)
        
    # Handle auto-speaker flag
    auto_speaker_mode = args.auto_speaker or (args.speaker and args.speaker.lower() == 'auto')
    validated_speaker = args.speaker if (args.speaker and args.speaker.lower() != 'auto') else None

    # Processing/Identification
    if args.identification_only:
        print("="*50)
        print("IDENTIFICATION MODE")
        print("="*50 + "\n")
        
        results = identify_text_structure(
            text,
            model_name=args.model,
            chunk_size=args.chunk_size,
            pronunciations=pronunciations,
            auto_speaker=auto_speaker_mode,
            verbose=args.verbose,
            character_voice_mapping=character_voice_mapping,
            generate_corrections=args.generate_corrections,
            apply_corrections_file=args.apply_corrections,
            input_file_path=args.input_file
        )
        
        # Print results (simplified for brevity, similar to original)
        print(f"Analysis complete. Total chunks: {results.get('total_chunks', 0)}")
        if results.get('speakers'):
            print("Speakers identified:")
            for char, voice in results['speakers'].items():
                print(f"  {char}: {voice}")
        return

    # Synthesis
    print(f"Starting conversion for: {input_source}")
    print(f"Output will be saved to: {output_file}")
    
    synthesizer = AudioSynthesizer(model_name=args.model)
    
    success = synthesizer.synthesize(
        text,
        str(output_file),
        chunk_size=args.chunk_size,
        is_short_input=(args.text is not None),
        pronunciations=pronunciations,
        speaker=validated_speaker,
        speaker_wav=args.speaker_wav,
        verbose=args.verbose,
        auto_speaker=auto_speaker_mode,
        character_voice_mapping=character_voice_mapping,
        apply_corrections_file=args.apply_corrections
    )
    
    if success:
        print(f"\n[OK] Conversion complete: {output_file}")
        if args.play:
            play_audio(str(output_file))
    else:
        print("\n[ERROR] Conversion failed.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
