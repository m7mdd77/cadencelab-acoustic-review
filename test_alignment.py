import unittest
from types import SimpleNamespace
from alignment import normalize_transcript, word_segments


class AlignmentTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_transcript("We choose, to go!"), "WE|CHOOSE|TO|GO")

    def test_numbers_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            normalize_transcript("1962")

    def test_empty(self):
        with self.assertRaises(ValueError):
            normalize_transcript("...")

    def test_non_english(self):
        with self.assertRaises(ValueError):
            normalize_transcript("\u4f60\u597d")

    def test_word_clock(self):
        labels = ["-", "A", "|", "B"]
        spans = [SimpleNamespace(token=t, start=s, end=e, score=.8) for t,s,e in [(1,1,3),(2,3,4),(3,6,8)]]
        self.assertEqual(word_segments(spans, labels, [1,2,3], 10, 10),
                         [{"word":"A","start":1,"end":3,"meanTokenPosterior":.8},
                          {"word":"B","start":6,"end":8,"meanTokenPosterior":.8}])

    def test_incomplete_alignment(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            word_segments([], ["-","A"], [1], 10, 10)

    def test_bad_clock(self):
        with self.assertRaises(ValueError):
            word_segments([], ["-"], [], 10, 0)

    def test_overlapping_spans(self):
        spans = [SimpleNamespace(token=1,start=1,end=5,score=.8),SimpleNamespace(token=1,start=4,end=6,score=.8)]
        with self.assertRaises(ValueError):
            word_segments(spans, ["-","A"], [1,1], 10, 10)


if __name__ == "__main__":
    unittest.main()
