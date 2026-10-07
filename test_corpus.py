import unittest

from build_corpus import SPLITS, validate_splits


class CorpusTests(unittest.TestCase):
    def test_fixed_splits(self):
        validate_splits(SPLITS)

    def test_overlap(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_splits([('a', 0, 20), ('b', 10, 30)])

    def test_duplicate(self):
        with self.assertRaises(ValueError):
            validate_splits([('a', 0, 20), ('a', 30, 50)])

    def test_long(self):
        with self.assertRaises(ValueError):
            validate_splits([('a', 0, 31)])
