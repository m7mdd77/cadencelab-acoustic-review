import unittest
import json
import tempfile
from pathlib import Path

from evaluate_dataset import overlap, evaluate
from build_dataset import build
from audio_engine import signal_fixture


class EvaluationTests(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(overlap({'start': 1, 'end': 2}, {'start': 1, 'end': 2}), 1)

    def test_disjoint(self):
        self.assertEqual(overlap({'start': 1, 'end': 2}, {'start': 3, 'end': 4}), 0)

    def test_partial(self):
        self.assertAlmostEqual(overlap({'start': 1, 'end': 3}, {'start': 2, 'end': 4}), 1 / 3)

    def setUp(self):
        cache = Path(__file__).parent / '.cache'
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        self.baseline = self.root / 'baseline.wav'
        self.baseline.write_bytes(signal_fixture()[0])
        build(self.baseline, self.root / 'variants')
        self.manifest = self.root / 'variants/manifest.json'

    def tearDown(self):
        self.temp.cleanup()

    def test_baseline_checksum(self):
        self.baseline.write_bytes(signal_fixture()[1])
        with self.assertRaisesRegex(ValueError, 'Baseline checksum'):
            evaluate(self.baseline, self.manifest)

    def test_variant_checksum(self):
        (self.manifest.parent / 'variant-01.wav').write_bytes(signal_fixture()[1])
        with self.assertRaisesRegex(ValueError, 'Variant checksum'):
            evaluate(self.baseline, self.manifest)

    def test_path_escape(self):
        manifest = json.loads(self.manifest.read_text())
        manifest['variants'][0]['file'] = '../baseline.wav'
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'escapes dataset'):
            evaluate(self.baseline, self.manifest)
