"""Produce labeled, deterministic acoustic interventions on real speech.

These labels identify injected signal changes, not human rhetorical judgments.
"""
import hashlib
import io
import json
import wave
from pathlib import Path

import numpy as np

from audio_engine import FRAME_SECONDS, compare, read_wav


def encode(samples, rate):
    output = io.BytesIO()
    with wave.open(output, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(np.clip(np.rint(samples * 32768), -32768, 32767).astype("<i2").tobytes())
    return output.getvalue()


def select_active_region(samples, rate, seconds=1):
    size = int(FRAME_SECONDS * rate)
    count = len(samples) // size
    blocks = samples[:count * size].reshape(count, size)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    active = rms > rms.max() * 10 ** (-35 / 20)
    width = round(seconds / FRAME_SECONDS)
    for index in np.argsort(-np.convolve(rms, np.ones(width), mode="valid"), kind="stable"):
        if index >= width and index + 2 * width < count and active[index:index + width].all():
            return index * size, (index + width) * size
    raise ValueError("No sufficiently active interior region for an intervention")


def build(baseline, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw = Path(baseline).read_bytes()
    samples, rate, _ = read_wav(raw)
    start, end = select_active_region(samples, rate)
    manifest = {"schemaVersion": 1, "baseline": Path(baseline).name,
                "baselineSha256": hashlib.sha256(raw).hexdigest(),
                "seed": None, "labelOrigin": "deterministic signal intervention",
                "groundTruthScope": "injected acoustic region only, not human speech quality",
                "generalization": "One speaker and one excerpt; no independent speaker evaluation",
                "variants": []}
    for kind, duration, multiplier in [
        ("additional-silence", .2, 0), ("additional-silence", .5, 0), ("additional-silence", 1, 0),
        ("relative-energy-change", 1, 10 ** (-3 / 20)),
        ("relative-energy-change", 1, 10 ** (-9 / 20)),
        ("relative-energy-change", 1, 10 ** (-18 / 20)),
        ("global-gain-control", len(samples) / rate, .5),
    ]:
        changed = samples.copy()
        a, b = (0, len(samples)) if kind == "global-gain-control" else (start, start + round(duration * rate))
        changed[a:b] *= multiplier
        data = encode(changed, rate)
        name = f"variant-{len(manifest['variants']) + 1:02d}.wav"
        (root / name).write_bytes(data)
        analysis = compare(raw, data)
        labels = [] if kind == "global-gain-control" else [{"kind": kind, "start": a / rate, "end": b / rate}]
        manifest["variants"].append({"file": name, "sha256": hashlib.sha256(data).hexdigest(),
                                     "multiplier": multiplier, "labels": labels,
                                     "transcript": "Same supplied transcript as baseline; muted speech is intentionally missing acoustically",
                                     "observedEvents": analysis["events"]})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = build(args.baseline, args.output_dir)
    print(f"Generated {len(result['variants'])} acoustic interventions")
