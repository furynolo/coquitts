import torch
import spacy
import torch
import spacy
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor # Removed
import librosa # Still potentially used nicely by pipeline internally or we can remove explicit import if pipeline handles it. 
# Actually pipeline uses ffmpeg/soundfile. Remove explicit librosa import if not used elsewhere, but let's keep it clean.
import numpy as np
import argparse
import difflib
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from transformers import pipeline

class AudioJudge:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        logging.info(f"Loading Wav2Vec2 pipeline: {model_name}...")
        # device=0 if GPU is available, else -1 for CPU. 
        # For safety on unknown user hardware without verifying CUDA, keeping to CPU or auto.
        # But 'pipeline' handles large files better with chunking.
        self.asr_pipeline = pipeline("automatic-speech-recognition", model=model_name)
        
        logging.info("Loading Spacy model: en_core_web_sm...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
            raise

    def transcribe(self, audio_path):
        """Transcribes the audio file using chunked processing with soundfile."""
        logging.info(f"Transcribing audio (manual chunking): {audio_path}")
        
        import soundfile as sf
        import librosa
        import numpy as np

        # Process in 5-minute chunks to manage memory and avoid cutting valid context too often
        CHUNK_DURATION_S = 300 
        
        full_transcription = []
        
        info = sf.info(audio_path)
        sr = info.samplerate
        total_samples = info.frames
        chunk_samples = int(CHUNK_DURATION_S * sr)
        
        for start in range(0, total_samples, chunk_samples):
            stop = min(start + chunk_samples, total_samples)
            
            # Read chunk
            audio, _ = sf.read(audio_path, start=start, stop=stop)
            
            # Convert to mono if necessary
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
                
            # Resample if necessary (Wav2Vec2 requires 16000 Hz)
            if sr != 16000:
                # librosa.resample requires float input, ensuring it is:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
            
            # Run inference on this chunk. 
            # Note: We still use chunk_length_s=30 internally for the pipeline to handle the 5-min block efficiently.
            prediction = self.asr_pipeline(audio, chunk_length_s=30, stride_length_s=5)
            text = prediction["text"]
            
            full_transcription.append(text)
            logging.info(f"Processed chunk {start//chunk_samples + 1}/{(total_samples+chunk_samples-1)//chunk_samples}: {len(text)} chars")
            
        final_text = " ".join(full_transcription).lower()
        return final_text

    def get_proper_nouns(self, text):
        """Identifies proper nouns in the text using Spacy."""
        doc = self.nlp(text)
        proper_nouns = set()
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "ORG", "NORP", "FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]:
                 # Add individual words from the entity to the set (normalized to lowercase)
                 for word in ent.text.split():
                     proper_nouns.add(word.lower())
        
        logging.info(f"Identified proper nouns: {proper_nouns}")
        return proper_nouns

    def evaluate(self, reference_text, audio_path, output_path=None, ignore_set=None):
        """
        Compares reference text with audio transcription, filtering out proper nouns.
        Outputs results to a file if output_path is provided.
        """
        transcription = self.transcribe(audio_path)
        normalized_ref = reference_text.lower()
        
        # Identify proper nouns from original text
        proper_nouns = self.get_proper_nouns(reference_text)
        
        # Add custom ignore words to proper nouns set (effectively treating them same way)
        if ignore_set:
            proper_nouns.update(ignore_set)

        # Clean punctuation for alignment
        import string
        translator = str.maketrans('', '', string.punctuation)
        clean_ref = normalized_ref.translate(translator)
        
        ref_words = clean_ref.split()
        trans_words = transcription.split()
        
        # Use SequenceMatcher to find the best alignment
        matcher = difflib.SequenceMatcher(None, ref_words, trans_words)
        
        issues = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            
            # Check for replacements or deletions (mismatches)
            if tag in ['replace', 'delete', 'insert']:
                missing_words = ref_words[i1:i2]
                perceived_words_list = trans_words[j1:j2]
                perceived_text = " ".join(perceived_words_list)
                
                # Check 1: If it's a "replace", acts as fuzzy match check
                if tag == 'replace':
                    expected_str = "".join(missing_words)
                    perceived_str = "".join(perceived_words_list)
                    
                    # similarity ratio (0.0 to 1.0)
                    sim_ratio = difflib.SequenceMatcher(None, expected_str, perceived_str).ratio()
                    
                    # Threshold for "close enough" (e.g., shamen vs shaman is likely > 0.8)
                    if sim_ratio > 0.8:
                        logging.info(f"Ignoring fuzzy match: '{expected_str}' ~= '{perceived_str}' ({sim_ratio:.2f})")
                        continue

                for word in missing_words:
                    # Clean word of punctuation for checking (redundant but safe)
                    clean_word = word.strip(".,!?;:\"'")
                    
                    if clean_word in proper_nouns:
                        logging.info(f"Skipping proper noun mismatch: '{clean_word}'")
                        continue
                        
                    # If it's not a proper noun, it's a potential issue
                    issue = {
                        "expected": word,
                        "perceived": perceived_text if perceived_text else "<silence/missing>",
                        "type": tag,
                        "context": f"...{' '.join(ref_words[max(0, i1-3):i1])} [[{word}]] {' '.join(ref_words[i2:min(len(ref_words), i2+3)])}..."
                    }
                    issues.append(issue)
                    logging.warning(f"Potential mispronunciation: Expected '{word}', Perceived '{issue['perceived']}'")

        if output_path:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(issues, f, indent=2)
            logging.info(f"Report saved to: {output_path}")

        return issues

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio pronunciation against reference text.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file (wav).")
    parser.add_argument("--output", "-o", type=str, help="Path to save the validation report (JSON).")
    
    parser.add_argument("--ignore-list", type=str, help="Text file containing words to ignore (one per line).")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Expected reference text string.")
    group.add_argument("--text-file", type=str, help="Path to a file containing the expected reference text.")
    
    args = parser.parse_args()
    
    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                reference_text = f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            sys.exit(1)
    else:
        reference_text = args.text
        
    ignore_words = set()
    if args.ignore_list:
        try:
            with open(args.ignore_list, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        ignore_words.add(word)
            logging.info(f"Loaded {len(ignore_words)} words to ignore.")
        except Exception as e:
            logging.warning(f"Could not load ignore list: {e}")
    
    judge = AudioJudge()
    
    # Default output if not specified
    out_path = args.output if args.output else "evaluation_report.json"
    
    found_issues = judge.evaluate(reference_text, args.audio, out_path, ignore_set=ignore_words)
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Found {len(found_issues)} potential issues.")
    print(f"Report saved to: {out_path}")
