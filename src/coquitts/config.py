"""
Configuration and resource loading utilities.
"""
import json
import re
import os
from typing import Dict, Optional

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
                cleaned_pronunciations[key] = cleaned_value.strip()
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
