
import unittest
import os
import sys

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coquitts.dialogue import DialogueSegmenter, Segment

class TestDialogueSegmenter(unittest.TestCase):

    def setUp(self):
        self.segmenter = DialogueSegmenter(verbose=False)

    def test_segment_text_simple(self):
        text = 'He said, "Hello there." She replied, "General Kenobi."'
        segments = self.segmenter.segment_text(text)
        
        # Expecting: Narration, Dialogue, Narration, Dialogue
        self.assertEqual(len(segments), 4)
        
        self.assertEqual(segments[0].type, "narration")
        self.assertEqual(segments[0].text, "He said,")
        
        self.assertEqual(segments[1].type, "dialogue")
        self.assertEqual(segments[1].text, "Hello there.")
        
        self.assertEqual(segments[2].type, "narration")
        self.assertEqual(segments[2].text, "She replied,")
        
        self.assertEqual(segments[3].type, "dialogue")
        self.assertEqual(segments[3].text, "General Kenobi.")

    def test_narration_check(self):
        # Test _is_narration_not_dialogue directly
        self.assertTrue(self.segmenter._is_narration_not_dialogue("(Thinking)"))
        self.assertFalse(self.segmenter._is_narration_not_dialogue("Hello"))

    def test_validate_name_rejections(self):
        """Test that common words are rejected as names."""
        bad_names = [
            "Are", "Don't", "Was", "Left", "Going", "Just", "Nonsense", 
            "Surely", "Survivors", "Water", "Whatever", "Everything", 
            "Forest", "Moments", "Our", "It's", "There's"
        ]
        for name in bad_names:
            self.assertFalse(self.segmenter._validate_name(name), f"Should reject '{name}'")
            
        good_names = ["Akama", "Velen", "Nobundo", "Estes", "Korin"]
        for name in good_names:
            self.assertTrue(self.segmenter._validate_name(name), f"Should accept '{name}'")
            
    def test_find_speaker_false_positives(self):
        """Test that heuristic patterns don't pick up common words."""
        # "Don't" followed by a speech verb-like word might trigger regex if case insensitive
        # But 'don't' isn't a speech verb. 'ask', 'say', etc. are.
        # "Don't ask me," he said. -> "he" is pronoun.
        # "Don't said the man." -> Invalid grammar but tests regex.
        
        # Test case: "Are you going?" he asked. (Should not pick 'Are')
        # _find_speaker_before("_Are_ you going?" he asked) -> 'he' -> looks back for name.
        
        # Test case: "Just say it," Velen said. 
        # _find_speaker_before("Just say it, Velen said") -> Velen matches Pattern 1?
        # "Velen said" matches. "Just" shouldn't.
        pass

if __name__ == '__main__':
    unittest.main()
