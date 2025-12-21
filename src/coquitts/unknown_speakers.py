"""
Module for handling unknown speaker corrections via JSON export/import.
"""
import json
import hashlib
from typing import List, Dict, Optional, Any
from pathlib import Path
from .dialogue import Segment

class CorrectionManager:
    """Manages export and import of speaker corrections."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def _calculate_segment_hash(self, text: str, start_pos: int) -> str:
        """Create a stable hash for a segment to ensure we match the right one."""
        # Use text and position to be unique
        unique_str = f"{start_pos}:{text}"
        return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

    def export_corrections(self, segments: List[Segment], full_text: str, output_path: str, context_size: int = 500):
        """
        Export UNKNOWN segments to a JSON file for manual correction.
        
        Args:
            segments: List of all segments
            full_text: The complete input text (for extracting context)
            output_path: Path to save the JSON file
            context_size: Number of characters of context to include
        """
        export_data = []
        
        unknown_segments = [s for s in segments if s.type == "dialogue" and s.speaker == "UNKNOWN"]
        
        if not unknown_segments:
            print("No UNKNOWN speakers found. Nothing to export.")
            return

        print(f"Found {len(unknown_segments)} UNKNOWN segments. Generating correction file...")
        
        for i, segment in enumerate(unknown_segments):
            # Extract context
            start = max(0, segment.start_pos - context_size)
            end = min(len(full_text), segment.end_pos + context_size)
            
            before_context = full_text[start:segment.start_pos]
            after_context = full_text[segment.end_pos:end]
            
            # Simple heuristic for suggestion (very basic: look for capitalized words nearby)
            # This is improved in DialogueSegmenter, but we can add a placeholder here
            # or rely on what DialogueSegmenter might have attached if we passed it.
            # For now, we leave suggestion empty or generic.
            
            item = {
                "id": self._calculate_segment_hash(segment.text, segment.start_pos),
                "index": i, # Human readable order
                "text": segment.text,
                "speaker": "UNKNOWN", # Current value
                "correction": "", # User fills this in
                "context_before": before_context,
                "context_after": after_context,
                "metadata": {
                    "start_pos": segment.start_pos,
                    "end_pos": segment.end_pos
                }
            }
            export_data.append(item)
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"Successfully exported {len(export_data)} segments to '{output_path}'.")
            print(f"Instructions: Open this file, find segments with 'speaker': 'UNKNOWN', fill in the 'correction' field with the correct name, save, and run again with --apply-corrections.")
        except Exception as e:
            print(f"Error writing correction file: {e}")

    def load_corrections(self, input_path: str) -> Dict[str, str]:
        """
        Load corrections from a JSON file.
        
        Args:
            input_path: Path to the JSON correction file
            
        Returns:
            Dictionary mapping segment hash to corrected speaker name
        """
        corrections = {}
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            count = 0
            for item in data:
                # Check if a correction was provided
                correction = item.get("correction", "").strip()
                # Also accept if they edited the "speaker" field directly away from UNKNOWN
                speaker = item.get("speaker", "").strip()
                
                final_name = ""
                if correction:
                    final_name = correction
                elif speaker and speaker != "UNKNOWN":
                    final_name = speaker
                
                if final_name:
                    segment_id = item.get("id")
                    if segment_id:
                        corrections[segment_id] = final_name
                        count += 1
            
            print(f"Loaded {count} corrections from '{input_path}'.")
            return corrections
            
        except Exception as e:
            print(f"Error loading corrections: {e}")
            return {}

    def apply_corrections(self, segments: List[Segment], corrections: Dict[str, str]) -> int:
        """
        Apply loaded corrections to segments.
        
        Args:
            segments: List of segments to update
            corrections: Dictionary of hash -> new speaker
            
        Returns:
            Number of segments updated
        """
        applied_count = 0
        if not corrections:
            return 0
            
        for segment in segments:
            if segment.type == "dialogue":
                # We calculate hash same way as export
                seg_id = self._calculate_segment_hash(segment.text, segment.start_pos)
                if seg_id in corrections:
                    old_speaker = segment.speaker
                    new_speaker = corrections[seg_id]
                    segment.speaker = new_speaker
                    # Also update original_speaker if it was UNKNOWN
                    if segment.original_speaker == "UNKNOWN" or segment.original_speaker is None:
                        segment.original_speaker = new_speaker
                    
                    if self.verbose:
                        print(f"Applied correction: '{segment.text[:20]}...' {old_speaker} -> {new_speaker}")
                    applied_count += 1
                    
        print(f"Applied {applied_count} speaker corrections.")
        return applied_count
