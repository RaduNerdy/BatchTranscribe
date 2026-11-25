import unittest
from types import SimpleNamespace
from diarization import align_speakers


class AlignmentTests(unittest.TestCase):
    def test_repeated_speakers_and_unknown(self):
        segments = [SimpleNamespace(start=a, end=b, text="hello") for a, b in [(0, 1), (1, 3), (3, 4), (5, 6)]]
        turns = [(0, 1, "a"), (1, 3, "b"), (3, 4, "a")]
        self.assertEqual([s.speaker for s in align_speakers(segments, turns)], ["Speaker 1", "Speaker 2", "Speaker 1", "Unknown speaker"])

    def test_largest_overlap(self):
        segments = [SimpleNamespace(start=0, end=3, text="hello")]
        self.assertEqual(align_speakers(segments, [(0, 1, "a"), (1, 3, "b")])[0].speaker, "Speaker 2")


if __name__ == "__main__":
    unittest.main()

