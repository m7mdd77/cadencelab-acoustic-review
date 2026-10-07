import unittest
from pathlib import Path

import numpy as np

from audio_engine import signal_fixture
from build_dataset import encode
from speech_features import compare_mfcc, mfcc


class SpeechFeatureTests(unittest.TestCase):
    def test_identical(self):
        raw, _ = signal_fixture()
        result = compare_mfcc(raw, raw)
        self.assertEqual(result['meanDistance'], 0)
        self.assertEqual(len(result['frames']), 80)
        self.assertEqual(len(result['frames'][0]['baseline']), 13)

    def test_real_speech(self):
        root = Path(__file__).parent / 'dataset'
        result = compare_mfcc((root / 'jfk-rice-120-140.wav').read_bytes(),
                              (root / 'jfk-rice-120-140-variants/variant-03.wav').read_bytes())
        self.assertGreater(result['meanDistance'], 0)
        self.assertEqual(len(result['frames']), 399)

    def test_reject_wrong_rate(self):
        with self.assertRaisesRegex(ValueError, '16 kHz'):
            mfcc(encode(np.ones(8000) * .1, 8000))

    def test_reject_long_audio(self):
        with self.assertRaisesRegex(ValueError, '30 seconds'):
            mfcc(encode(np.ones(31 * 16000) * .1, 16000))
