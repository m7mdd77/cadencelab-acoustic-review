import tempfile
import unittest
from pathlib import Path
from audio_engine import signal_fixture
from build_dataset import build


class DatasetTests(unittest.TestCase):
    def test_manifest_and_control(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / ".cache") as temporary:
            root = Path(temporary)
            base = root / "baseline.wav"
            base.write_bytes(signal_fixture()[0])
            result = build(base, root / "variants")
            self.assertEqual(len(result["variants"]), 7)
            self.assertEqual(result["variants"][-1]["observedEvents"], [])
            self.assertEqual(result["variants"][-1]["labels"], [])
            self.assertEqual(result["variants"][0]["labels"][0]["kind"], "additional-silence")
            for item in result["variants"]:
                self.assertTrue((root / "variants" / item["file"]).exists())

    def test_repeatable(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / ".cache") as temporary:
            root = Path(temporary)
            base = root / "baseline.wav"
            base.write_bytes(signal_fixture()[0])
            self.assertEqual(build(base, root / "first"), build(base, root / "second"))


if __name__ == "__main__":
    unittest.main()
