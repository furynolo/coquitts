
import unittest
import os
import sys

# Add src to path
# We are in tests/ so we need to go up one level to root, then into src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from coquitts.dialogue import Segment, SpeakerAssigner

class TestFlowHeuristics(unittest.TestCase):
    def test_alternating_pattern_backward(self):
        """Test resolving UNKNOWN in A -> B -> [?] pattern (where ? should be A)"""
        segments = [
            Segment(type="dialogue", speaker="Alice", text="Hello."),
            Segment(type="dialogue", speaker="Bob", text="Hi there."),
            Segment(type="dialogue", speaker="UNKNOWN", text="How are you?"), # Should be Alice
        ]
        
        assigner = SpeakerAssigner(["p1", "p2"])
        assigner._suggest_speaker_flow(segments, {})
        
        self.assertEqual(segments[2].speaker, "Alice")
        print("\nTest passed: A -> B -> ? resolved to A")

    def test_alternating_pattern_sandwich(self):
        """Test resolving UNKNOWN in B -> A -> [?] -> A pattern (where ? should be B)"""
        segments = [
            Segment(type="dialogue", speaker="Bob", text="Ready?"),
            Segment(type="dialogue", speaker="Alice", text="Yes."),
            Segment(type="dialogue", speaker="UNKNOWN", text="Let's go."), # Should be Bob
            Segment(type="dialogue", speaker="Alice", text="Okay."),
        ]
        
        assigner = SpeakerAssigner(["p1", "p2"])
        assigner._suggest_speaker_flow(segments, {})
        
        self.assertEqual(segments[2].speaker, "Bob")
        print("\nTest passed: B -> A -> ? -> A resolved to B")

    def test_no_resolution_if_too_far(self):
        """Test that heuristic respects distance limits"""
        segments = [
            Segment(type="dialogue", speaker="Alice", text="Start."),
            Segment(type="narration", speaker="NARRATOR", text="He waited for a very long time... " * 20), # Long gap
            Segment(type="dialogue", speaker="Bob", text="End."),
            Segment(type="dialogue", speaker="UNKNOWN", text="Hello?"), # Too far from Alice
        ]
        
        assigner = SpeakerAssigner(["p1", "p2"])
        assigner._suggest_speaker_flow(segments, {})
        
        self.assertEqual(segments[3].speaker, "UNKNOWN")
        print("\nTest passed: Long gap prevents unsafe resolution")

if __name__ == '__main__':
    unittest.main()
