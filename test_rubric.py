import unittest
from audio_engine import compare, signal_fixture
from rubric import acoustic_rubric


class RubricTests(unittest.TestCase):
    def test_reference(self):
        a, _ = signal_fixture()
        self.assertEqual(compare(a, a)['rubric']['score'], 100)

    def test_intervention(self):
        a, b = signal_fixture()
        score = compare(a, b)['rubric']
        self.assertLess(score['score'], 100)
        self.assertEqual(sum(score['weights'].values()), 1)
        self.assertGreater(score['penalties']['additionalSilence'], 0)

    def test_empty(self):
        with self.assertRaises(ValueError):
            acoustic_rubric([], [], 6)

    def test_monotonic_energy(self):
        base = [{'active': True, 'relativeDb': 0, 'clippingFraction': 0}] * 10
        scores = [acoustic_rubric(base, [{'active': True, 'relativeDb': -delta, 'clippingFraction': 0}] * 10, 6)['score'] for delta in (3, 9, 18)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(scores[0], 100)
