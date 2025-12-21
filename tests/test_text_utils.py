
import unittest
import os
import sys

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coquitts.text_utils import split_text_into_chunks, normalize_for_tts, apply_pronunciations, preprocess_text

class TestTextUtils(unittest.TestCase):

    def test_normalize_for_tts(self):
        text = "Hello… world! It’s me."
        expected = "Hello... world! It's me."
        self.assertEqual(normalize_for_tts(text), expected)
        
    def test_preprocess_text(self):
        text_with_newlines = "Line 1.\nLine 2."
        # preprocess_text normalizes spaces and newlines
        processed = preprocess_text(text_with_newlines)
        # It replaces newlines with spaces if they are single newlines?
        # Let's check logic: re.sub(r' +', ' ', text)
        # It doesn't explicitly replace single newlines with spaces unless they are normalized out.
        # Wait, preprocess_text: 
        # Normalize multiple spaces. normalize newlines (keep max 2).
        # It doesn't replace single newlines with spaces!
        # So "Line 1.\nLine 2." stays "Line 1.\nLine 2."
        
        # Correction: Text usually gets striped.
        # Let's check what I want it to do.
        # If I want to test newline handling:
        text_multi_lines = "Line 1.\n\n\nLine 2."
        expected_multi = "Line 1.\n\nLine 2."
        self.assertEqual(preprocess_text(text_multi_lines), expected_multi)

    def test_apply_pronunciations(self):
        text = "The word is text."
        pronunciations = {"text": "test"}
        # Case sensitive by default in the implementation? Let's check.
        # Original script's apply_pronunciations used re.IGNORECASE for keys?
        # Inspecting previous view_file of text_utils:
        # It used re.compile(re.escape(k), re.IGNORECASE) usually.
        # Let's assume it works.
        self.assertEqual(apply_pronunciations(text, pronunciations), "The word is test.")
        
        # Test word boundary
        text2 = "context"
        self.assertEqual(apply_pronunciations(text2, pronunciations), "context")

    def test_split_text_into_chunks(self):
        text = "Sentence one. Sentence two. Sentence three."
        # Force small chunks
        chunks = split_text_into_chunks(text, max_chunk_size=20)
        # Should split appropriately
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(c) <= 20 for c in chunks))

if __name__ == '__main__':
    unittest.main()
