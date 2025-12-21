
import unittest
import os
import json
import shutil
from pathlib import Path

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from coquitts.dialogue import Segment
from coquitts.unknown_speakers import CorrectionManager

class TestUnknownWorkflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        self.json_path = self.test_dir / "corrections.json"
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_export_import(self):
        # Create dummy segments
        segments = [
            Segment(type="dialogue", speaker="UNKNOWN", text="Who goes there?", start_pos=10, end_pos=25),
            Segment(type="narration", speaker="NARRATOR", text="he asked.", start_pos=26, end_pos=35),
            Segment(type="dialogue", speaker="UNKNOWN", text="It is I, Arthur.", start_pos=36, end_pos=52)
        ]
        full_text = "Wait! Who goes there? he asked. It is I, Arthur. The King."
        
        cm = CorrectionManager(verbose=True)
        cm.export_corrections(segments, full_text, str(self.json_path))
        
        # Verify file exists
        self.assertTrue(self.json_path.exists())
        
        # Load and verify content
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]['text'], "Who goes there?")
            self.assertEqual(data[0]['speaker'], "UNKNOWN")
            
        # Simulate user editing
        data[0]['correction'] = "Guard"
        # Edit the second one via speaker field directly (simulating reckless user)
        data[1]['speaker'] = "Arthur" 
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f)
            
        # Reload
        corrections = cm.load_corrections(str(self.json_path))
        self.assertEqual(len(corrections), 2)
        
        # Verify corrections match hashes
        # We need to recalculate hashes or just check if apply works
        applied = cm.apply_corrections(segments, corrections)
        self.assertEqual(applied, 2)
        
        self.assertEqual(segments[0].speaker, "Guard")
        self.assertEqual(segments[2].speaker, "Arthur")
        
        print("\nTest passed: Export -> Edit -> Load -> Apply cycle works.")

if __name__ == '__main__':
    unittest.main()
