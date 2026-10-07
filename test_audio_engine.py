import io
import unittest
import wave
import numpy as np
from audio_engine import compare, features, read_wav, signal_fixture


def wav(samples, rate=16000, channels=1, width=2):
    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(width)
        file.setframerate(rate)
        file.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return out.getvalue()


class AudioTests(unittest.TestCase):
    def test_known_regions(self):
        a, b = signal_fixture()
        events = compare(a, b)["events"]
        self.assertEqual([(e["kind"], e["start"], e["end"]) for e in events],
                         [("additional-silence", 1, 1.5), ("relative-energy-change", 2, 2.5)])

    def test_identical_audio(self):
        a, _ = signal_fixture()
        self.assertEqual(compare(a, a)["events"], [])

    def test_deterministic(self):
        a, b = signal_fixture()
        self.assertEqual(compare(a, b), compare(a, b))

    def test_global_gain_is_normalized(self):
        a, _ = signal_fixture()
        samples, rate, _ = read_wav(a)
        b = wav(samples * 32768 * 2, rate)
        self.assertEqual(compare(a, b)["events"], [])

    def test_mismatched_duration(self):
        a, _ = signal_fixture()
        with self.assertRaisesRegex(ValueError, "Timelines differ"):
            compare(a, wav(np.ones(16000, dtype="int16") * 1000))

    def test_silence_rejected(self):
        with self.assertRaisesRegex(ValueError, "no usable signal"):
            features(wav(np.zeros(16000)))

    def test_invalid_wav(self):
        with self.assertRaises(ValueError):
            read_wav(b"not a wave" * 10)

    def test_wrong_width(self):
        with self.assertRaisesRegex(ValueError, "16-bit"):
            read_wav(wav(np.zeros(16000), width=1))

    def test_truncated(self):
        a, _ = signal_fixture()
        with self.assertRaisesRegex(ValueError, "Truncated"):
            read_wav(a[:-10])

    def test_short_audio(self):
        with self.assertRaises(ValueError):
            features(wav(np.ones(1000) * 2000))

    def test_invalid_threshold(self):
        a, b = signal_fixture()
        for value in (0, 21, float("nan"), float("inf"), "6"):
            with self.assertRaises(ValueError):
                compare(a, b, value)

    def test_no_fake_quality_score(self):
        a, b = signal_fixture()
        self.assertIsNone(compare(a, b)["qualityScore"])

    def test_clipping_detection(self):
        a, _ = signal_fixture()
        samples, rate, _ = read_wav(a)
        samples[rate:rate * 2] = 32767 / 32768
        events = compare(a, wav(samples * 32768, rate))["events"]
        self.assertTrue(any(e["kind"] == "signal-clipping" for e in events))


if __name__ == "__main__":
    unittest.main()
