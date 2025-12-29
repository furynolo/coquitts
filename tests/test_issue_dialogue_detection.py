
import unittest
import sys
import os

# Add src to path to import coquitts
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from coquitts.dialogue import DialogueSegmenter

class TestDialogueDetection(unittest.TestCase):
    def setUp(self):
        self.segmenter = DialogueSegmenter(verbose=False)

    def test_when_you_escaped_quote(self):
        """Test that the specific 'When you escaped' quote is classified as dialogue."""
        text = 'Korin asked Nobundo, “When you escaped, did any others make it with you? Were there more survivors? We heard the orcs in the lower levels, but we did not want to risk discovery, so we fled.”'
        
        segments = self.segmenter.segment_text(text)
        
        found_dialogue = False
        for seg in segments:
            if seg.type == "dialogue" and "When you escaped" in seg.text:
                found_dialogue = True
                self.assertEqual(seg.speaker, "Korin", "Speaker should be identified as Korin")
                break
        
        self.assertTrue(found_dialogue, "Should classify 'When you escaped...' as dialogue, not narration")

    def test_smart_quotes(self):
        """Test that smart quotes are correctly handled."""
        text = 'He said, “This is a test.”'
        segments = self.segmenter.segment_text(text)
        
        dialogue_found = False
        for seg in segments:
            if seg.type == "dialogue":
                dialogue_found = True
                self.assertEqual(seg.text, "This is a test.")
                break
                
        self.assertTrue(dialogue_found, "Should extract dialogue with smart quotes")

    def test_narration_vs_dialogue_heuristics(self):
        """Test specific heuristics for narration vs dialogue."""
        # Narration starting with 'When'
        narration_text = '"When he arrived at the door, he stopped." he said.' 
        # Note: The above is a bit ambiguous in structure, let's use a clear narration block in quotes if that's what we test.
        # But _is_narration_not_dialogue is detecting if a QUOTE is narration.
        # "When he arrived at the door, he stopped." -> Should be narration if it matches the pattern.
        
        is_narration = self.segmenter._is_narration_not_dialogue("When he arrived at the door, he stopped.")
        self.assertTrue(is_narration, "Should be classified as narration because of 'When he...'")

        # Dialogue starting with 'When' (referring to 'you')
        is_narration = self.segmenter._is_narration_not_dialogue("When you escaped, did any others make it with you?")
        self.assertFalse(is_narration, "Should be classified as dialogue because of 'When you...'")

if __name__ == '__main__':
    unittest.main()
