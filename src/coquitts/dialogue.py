"""
Dialogue segmentation and speaker assignment.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple, Any
import re
import sys
import io

from .text_utils import NLTK_AVAILABLE

@dataclass
class Segment:
    """Represents a text segment (narration or dialogue)."""
    type: str  # "narration" or "dialogue"
    speaker: str  # Speaker name (e.g., "NARRATOR", "Alleria", "UNKNOWN") - post-pronunciation
    text: str  # The actual text content
    start_pos: int = 0  # Character position in original text
    end_pos: int = 0  # Character position in original text
    original_speaker: str = None  # Original speaker name before pronunciation mappings

def _is_suspicious_speaker_nltk(speaker_name: str) -> bool:
    """
    Use NLTK to determine if a detected speaker name is likely a false positive.
    
    Returns True if the name is suspicious (likely not a real person name).
    Uses NLTK POS tagging and stopwords to identify common words that shouldn't be names.
    """
    if not speaker_name or speaker_name in ("NARRATOR", "UNKNOWN"):
        return False
    
    words = speaker_name.split()
    if not words:
        return False
    
    # Very short names (1-2 chars) are suspicious unless they're clearly proper nouns
    if len(words) == 1 and len(speaker_name) <= 2:
        return True
    
    # Check if NLTK is available
    if not NLTK_AVAILABLE:
        # Fallback: basic heuristics if NLTK not available
        # Check for common articles/conjunctions at start
        first_word_lower = words[0].lower()
        if first_word_lower in ['the', 'a', 'an', 'and', 'or', 'but', 'if', 'when', 'where', 'why', 'how']:
            return True
        return False
    
    try:
        from nltk import word_tokenize
        from nltk.tag import pos_tag
        from nltk.corpus import stopwords
        
        # Try to load stopwords (download if needed)
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            try:
                import nltk
                nltk.download('stopwords', quiet=True)
                stop_words = set(stopwords.words('english'))
            except Exception:
                stop_words = set()  # Fallback to empty set
        
        # Tokenize and POS tag the speaker name
        # Create a simple sentence context to help NLTK tag properly
        # (e.g., "Are" might be tagged as verb, "John" as proper noun)
        test_sentence = f"{speaker_name} said something."
        tokens = word_tokenize(test_sentence)
        pos_tags = pos_tag(tokens)
        
        # Find the tags for words in the speaker name
        speaker_tokens = word_tokenize(speaker_name)
        speaker_pos_map = {}
        token_idx = 0
        for word, pos in pos_tags:
            if token_idx < len(speaker_tokens) and word.lower() == speaker_tokens[token_idx].lower():
                speaker_pos_map[word] = pos
                token_idx += 1
        
        # If we couldn't match tokens, try a simpler approach
        if not speaker_pos_map:
            # Just tag the speaker name directly
            speaker_tokens = word_tokenize(speaker_name)
            speaker_pos_tags = pos_tag(speaker_tokens)
            speaker_pos_map = {word: pos for word, pos in speaker_pos_tags}
        
        # Check each word in the speaker name
        all_proper_nouns = True
        has_common_word = False
        has_stopword = False
        words_with_pos = []
        
        for word in words:
            word_lower = word.lower()
            
            # Check if it's a stopword
            if word_lower in stop_words:
                has_stopword = True
                has_common_word = True
                all_proper_nouns = False
            
            # Get POS tag for this word
            word_pos = None
            for token, pos in speaker_pos_map.items():
                if token.lower() == word_lower:
                    word_pos = pos
                    break
            
            # If we couldn't find it in the map, tag it directly
            if word_pos is None:
                try:
                    word_tokens = word_tokenize(word)
                    word_pos_tags = pos_tag(word_tokens)
                    if word_pos_tags:
                        word_pos = word_pos_tags[0][1]
                except Exception:
                    pass
            
            words_with_pos.append((word, word_pos))
            
            # Check POS tag
            if word_pos:
                # Proper nouns (NNP, NNPS) are good - these are likely real names
                if not word_pos.startswith('NNP'):
                    # Not a proper noun - suspicious
                    all_proper_nouns = False
                    # Common parts of speech that shouldn't be names:
                    if word_pos.startswith(('VB', 'JJ', 'RB', 'DT', 'IN', 'CC', 'PRP')):
                        # Verb, Adjective, Adverb, Determiner, Preposition, Conjunction, Pronoun
                        has_common_word = True
        
        # Decision logic:
        # 1. If all words are proper nouns (NNP) and no stopwords, it's likely a real name
        if all_proper_nouns and not has_stopword and words_with_pos:
            # Double-check: make sure we actually got POS tags for all words
            if all(pos is not None for _, pos in words_with_pos):
                return False
        
        # 2. If any word is a stopword, it's suspicious
        if has_stopword:
            return True
        
        # 3. If any word is a common part of speech (verb, adjective, etc.), it's suspicious
        if has_common_word:
            return True
        
        # 4. Check for common sentence starters
        first_word_lower = words[0].lower()
        if first_word_lower in ['the', 'a', 'an', 'and', 'or', 'but', 'if', 'when', 'where', 'why', 'how', 'what', 'who']:
            return True
        
        # 5. If it's a multi-word phrase starting with common words, it's suspicious
        if len(words) >= 2:
            if first_word_lower in ['let', 'our', 'the', 'a', 'an']:
                return True
        
        # Default: if we can't determine, don't mark as suspicious
        return False
        
    except Exception as e:
        # If NLTK fails, fall back to basic heuristics
        first_word_lower = words[0].lower() if words else ''
        if first_word_lower in ['the', 'a', 'an', 'and', 'or', 'but', 'if', 'when']:
            return True
        return False

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
            # Try to resolve UNKNOWN speakers using conversation flow heuristics
            self._suggest_speaker_flow(segments, name_mapping)
            
            # Re-check for UNKNOWN
            if any(s.speaker == "UNKNOWN" for s in segments if s.type == "dialogue"):
                self.character_map["UNKNOWN"] = self.narrator_speaker
        
        return self.character_map
    
    def _suggest_speaker_flow(self, segments: List[Segment], name_mapping: Dict[str, str]) -> None:
        """
        Analyze conversation flow to suggest speakers for UNKNOWN segments.
        Assumes A-B-A-B patterns in close proximity.
        """
        print("  Running conversation flow analysis...")
        resolved_count = 0
        
        # Iterate through segments
        for i in range(len(segments)):
            segment = segments[i]
            
            # Skip if not UNKNOWN dialogue
            if segment.type != "dialogue" or segment.speaker != "UNKNOWN":
                continue
                
            # Look backwards for the last known speaker in dialogue
            prev_speaker = None
            prev_index = -1
            distance_back = 0
            
            for j in range(i - 1, -1, -1):
                other = segments[j]
                if other.type == "dialogue":
                    if other.speaker != "UNKNOWN" and other.speaker != "NARRATOR":
                        prev_speaker = other.speaker
                        prev_index = j
                        break
                    distance_back += 1
                    if distance_back > 3: # Don't look too far back (in dialogue terms)
                        break
            
            # Look forwards for the next known speaker
            next_speaker = None
            next_index = -1
            distance_fwd = 0
            
            for j in range(i + 1, len(segments)):
                other = segments[j]
                if other.type == "dialogue":
                    if other.speaker != "UNKNOWN" and other.speaker != "NARRATOR":
                        next_speaker = other.speaker
                        next_index = j
                        break
                    distance_fwd += 1
                    if distance_fwd > 3:
                        break
            
            # Heuristic 1: Sandwiched (A -> ? -> A) => Likely B? No, usually implies 3 parties or narration break.
            # But (A -> ? -> A) could be (A -> B -> A) implies ? is B. 
            # We don't know B yet.
            
            # Heuristic 2: Alternating (A -> ? -> B) => Likely ? is B? Or ? is A?
            # Standard pattern: A says something. B replies. A replies.
            # If we have A -> ? -> A, it implies the middle one is NOT A.
            
            proposed_speaker = None
            reason = ""
            
            # Scenario: A -> ? -> A
            # The middle one is likely the OTHER person in the conversation.
            # Do we know who the other person is?
            # We can look further back: B -> A -> ? -> A
            if prev_speaker and next_speaker and prev_speaker == next_speaker:
                # We are between two lines from the same person.
                # Likely we are the other interactant.
                # Check 2 steps back
                speaker_2_back = None
                for j in range(prev_index - 1, -1, -1):
                    other = segments[j]
                    if other.type == "dialogue":
                        if other.speaker != "UNKNOWN" and other.speaker != "NARRATOR":
                            speaker_2_back = other.speaker
                            break
                        # Stop if too far
                        if (prev_index - j) > 5: 
                            break
                            
                if speaker_2_back and speaker_2_back != prev_speaker:
                    # Pattern: B -> A -> [UNKNOWN] -> A
                    # Inference: UNKNOWN is B
                    proposed_speaker = speaker_2_back
                    reason = f"Alternating pattern ({speaker_2_back} -> {prev_speaker} -> ? -> {prev_speaker})"
            
            # Scenario: A -> B -> ? 
            # Likely A (Alternating)
            elif prev_speaker:
                # Check who spoke before prev_speaker
                speaker_2_back = None
                for j in range(prev_index - 1, -1, -1):
                    other = segments[j]
                    if other.type == "dialogue":
                        if other.speaker != "UNKNOWN" and other.speaker != "NARRATOR":
                            speaker_2_back = other.speaker
                            break
                        if (prev_index - j) > 5:
                            break
                
                if speaker_2_back and speaker_2_back != prev_speaker:
                    # Pattern: A -> B -> ?
                    # Inference: ? is A
                    proposed_speaker = speaker_2_back
                    reason = f"Alternating pattern ({speaker_2_back} -> {prev_speaker} -> ?)"
            
            # Scenario: ? -> A -> B
            # Likely B (Alternating backwards?) -> Harder to be sure.
            
            if proposed_speaker:
                # Apply change
                segment.speaker = proposed_speaker
                segment.original_speaker = proposed_speaker # Update original too so it sticks
                resolved_count += 1
                # print(f"  [FLOW] Resolved UNKNOWN at index {i}-ish to '{proposed_speaker}'. Reason: {reason}")
        
        if resolved_count > 0:
            print(f"  [FLOW] Resolved {resolved_count} UNKNOWN segments using conversation flow.")

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
            speaker_id = self.character_map.get(segment.speaker)
            if not speaker_id:
                # If mapped to None or not found, fallback to narrator
                return self.narrator_speaker or self.available_speakers[0]
            return speaker_id

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
        # Track UNKNOWN dialogue segments for identification mode
        self.unknown_segments = []  # List of dicts with text, before_context, after_context
        
        # Try to use NLTK for POS tagging if available
        if NLTK_AVAILABLE:
            try:
                from nltk import word_tokenize
                from nltk.tag import pos_tag
                from nltk.data import find as nltk_find
                
                # Try to download required data
                # NLTK recently split the tagger into language specific packages
                # We need to try both legacy and new names
                tagger_available = False
                tagger_resource = 'averaged_perceptron_tagger'
                
                # Check for new English-specific tagger first (for NLTK 3.9+)
                try:
                    nltk_find('taggers/averaged_perceptron_tagger_eng')
                    tagger_available = True
                    tagger_resource = 'averaged_perceptron_tagger_eng'
                except LookupError:
                    try:
                        nltk_find('taggers/averaged_perceptron_tagger')
                        tagger_available = True
                        tagger_resource = 'averaged_perceptron_tagger'
                    except LookupError:
                        pass
                
                if not tagger_available:
                    # Try to download both to be safe
                    try:
                        if self.verbose:
                            print("    [NLTK] Downloading NLTK taggers...")
                        import nltk
                        try:
                            nltk.download('averaged_perceptron_tagger_eng', quiet=not self.verbose)
                            tagger_resource = 'averaged_perceptron_tagger_eng'
                        except Exception:
                            nltk.download('averaged_perceptron_tagger', quiet=not self.verbose)
                            tagger_resource = 'averaged_perceptron_tagger'
                        
                        # Verify download
                        try:
                            try:
                                nltk_find(f'taggers/{tagger_resource}')
                            except LookupError:
                                # Fallback check
                                nltk_find('taggers/averaged_perceptron_tagger')
                                tagger_resource = 'averaged_perceptron_tagger'
                            
                            tagger_available = True
                            if self.verbose:
                                print("    [NLTK] Tagger downloaded successfully")
                        except LookupError:
                            if self.verbose:
                                print("    [NLTK] Warning: Tagger download may have failed")
                    except Exception as e:
                        if self.verbose:
                            print(f"    [NLTK] Error downloading taggers: {e}")
                
                # Verify punkt is available (and punkt_tab for newer NLTK)
                punkt_available = False
                try:
                    nltk_find('tokenizers/punkt')
                    punkt_available = True
                except LookupError:
                    try:
                        import nltk
                        nltk.download('punkt', quiet=True)
                        nltk_find('tokenizers/punkt')
                        punkt_available = True
                    except Exception:
                        pass

                # Newer NLTK versions require punkt_tab for some functions
                try:
                    nltk_find('tokenizers/punkt_tab')
                except LookupError:
                    try:
                        import nltk
                        nltk.download('punkt_tab', quiet=True)
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
                        # If we get a LookupError, it means a resource is missing
                        if self.verbose:
                            print(f"    [NLTK SETUP] Resource missing during test: {lookup_error}")
                            print(f"    [NLTK SETUP] Attempting to fix...")
                        
                        try:
                            import nltk
                            # Download everything we might need
                            nltk.download('averaged_perceptron_tagger_eng', quiet=False)
                            nltk.download('averaged_perceptron_tagger', quiet=False)
                            nltk.download('punkt', quiet=False)
                            nltk.download('punkt_tab', quiet=False)
                            
                            # Try the test again
                            test_tokens = word_tokenize("Test sentence.")
                            test_tags = pos_tag(test_tokens)
                            self._nltk_pos_available = True
                            self._pos_tag = pos_tag
                            self._word_tokenize = word_tokenize
                            if self.verbose:
                                print(f"    [NLTK SETUP] NLTK POS tagging fixed and initialized")
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
        
        # No speaker found - store information for identification mode
        import re
        # Store UNKNOWN segment info (always, not just in verbose mode)
        unknown_info = {
            'dialogue_text': dialogue_text,
            'before_context': before_context[-200:] if len(before_context) > 200 else before_context,
            'after_context': after_context[:200] if len(after_context) > 200 else after_context,
            'full_before_context': before_context_for_pronouns[-500:] if len(before_context_for_pronouns) > 500 else before_context_for_pronouns,
            'full_after_context': after_context_extended[:500] if len(after_context_extended) > 500 else after_context_extended,
        }
        # Check for potential names in context that were rejected
        potential_names = re.findall(r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,2})\b', before_context[-100:] + after_context[:100])
        if potential_names:
            unknown_info['potential_names'] = list(set(potential_names[:10]))
        self.unknown_segments.append(unknown_info)
        
        # Provide detailed debug output if verbose
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
        # CASE SENSITIVE name matching to avoid matching common words at start of sentence
        # Verbs are matched case-insensitively via regex flag (?i) for the verb group only
        verbs_pattern = r'(?i:' + '|'.join(self.SPEECH_VERBS) + r')'
        name_pattern = r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3})(?:[,:]?\s+)' + verbs_pattern + r'(?:[,:;]|\s+|$)'
        match = re.search(name_pattern, text)
        if match:
            name = match.group(1).strip()
            if self._validate_name(name):
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found: {name} {match.group(2)}")
                return name
        
        # Pattern 2: Name said [additional text] (e.g., "Mar'gok said, waving him off")
        # This allows text after the speech verb
        name_pattern2 = r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3})(?:[,:]?\s+)' + verbs_pattern + r'(?:[,:;]|\s+[^.!?]*)'
        match = re.search(name_pattern2, text)
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
        verbs_pattern = r'(?i:' + '|'.join(self.SPEECH_VERBS) + r')'
        speech_verb_pattern = r'\b(' + verbs_pattern + r')(?:[,:]?\s+)([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3})(?:[,:;]|\s+|\.|$)'
        match = re.search(speech_verb_pattern, text)
        if match:
            name = match.group(2).strip()
            if self._validate_name(name):
                if self.verbose:
                    print(f"      [PATTERN MATCH] Found: {match.group(1)} {name}")
                return name
        
        # Pattern 2: said Name [additional text] (e.g., "said Name, pointing at...")
        speech_verb_pattern2 = r'\b(' + verbs_pattern + r')(?:[,:]?\s+)([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3})(?:[,:;]|\s+[^.!?]*)'
        match = re.search(speech_verb_pattern2, text)
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
        # Pattern 4: Possessive patterns (e.g., "A weighty grunt was Growmash's only response")
        # Look for patterns like "X's response", "X's voice", "X's only", etc.
        # Strict capitalization for Name, case-insensitive for the noun part
        possessive_pattern = r'\b([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,2})\'s\s+(?i:(only|response|voice|reply|answer|words|statement|comment|remark|question|demand|request|order|command|threat|warning|promise|offer|suggestion|proposal|claim|admission|confession|denial|refusal|acceptance|agreement|disagreement|nod|shake|laugh|smile|frown|sigh|gasp|hiss|snap|bark|grunt|growl|snarl|roar|yell|scream|mutter|whisper|shout|reply|ask|speak|continue|interrupt|add|conclude|finish|agree|disagree))'
        match = re.search(possessive_pattern, dialogue_text)
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
            'days', 'weeks', 'months', 'years', 'hours', 'minutes', 'seconds', 'moments',
            # Contractions (start of sentence)
            "don't", "can't", "won't", "wouldn't", "couldn't", "shouldn't", "didn't",
            "doesn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
            "it's", "he's", "she's", "they're", "we're", "you're", "i'm", "there's", "here's",
            "let's", "that's", "what's", "how's", "who's", "where's", "when's", "why's",
            # Additional observed false positives
            'are', 'don', 'do', 'did', 'does', 'was', 'were', 'is', 'has', 'have', 'had',
            'going', 'looking', 'walking', 'standing', 'sitting', 'turning', 'moving',
            'soar', 'fly', 'flee', 'run', 'hide', 'fight', 'attack', 'defend',
            'left', 'right', 'just', 'nonsense', 'water', 'whatever', 'surely', 'survivors',
            'everything', 'anything', 'something', 'nothing', 'everyone', 'anyone', 'someone',
            'nobody', 'everybody', 'anybody', 'somebody', 'forest', 'ground', 'sky', 'sun',
            'moon', 'stars', 'world', 'land', 'sea', 'ocean', 'river', 'lake', 'mountain',
            'warlock', 'mage', 'priest', 'warrior', 'rogue', 'hunter', 'druid', 'shaman',
            'paladin', 'monk', 'demon', 'hunter', 'death', 'knight',
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
        
        # All words should start with capital (except for particles like "de", "van", "of", "the" in specific contexts)
        # We enforce that the FIRST word must be capitalized.
        # Subsequent words must be capitalized unless they are specific allowed particles.
        allowed_lowercase_particles = {'de', 'da', 'di', 'van', 'von', 'del', 'della', 'of', 'the', 'mc', 'mac'}
        
        if not words[0][0].isupper():
            return False
            
        for i, word in enumerate(words[1:], 1):
            if not word[0].isupper() and word.lower() not in allowed_lowercase_particles:
                return False
        
        return True

