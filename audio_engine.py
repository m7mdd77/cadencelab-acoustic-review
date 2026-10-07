"""Deterministic acoustic comparison. Not a rhetorical-quality or medical model."""
import argparse
import io
import json
import wave
from pathlib import Path

import numpy as np

MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 180
FRAME_SECONDS = 0.05


def read_wav(data):
    if not isinstance(data, bytes) or not 44 <= len(data) <= MAX_BYTES:
        raise ValueError("WAV must be between 44 bytes and 20 MiB")
    try:
        with wave.open(io.BytesIO(data), "rb") as recording:
            channels, width, rate, frames, compression, _ = recording.getparams()
            if width != 2 or channels not in (1, 2) or compression != "NONE":
                raise ValueError("Use uncompressed 16-bit mono or stereo PCM WAV")
            if not 8000 <= rate <= 192000 or not 0 < frames / rate <= MAX_SECONDS:
                raise ValueError("Use 8-192 kHz audio of at most 180 seconds")
            pcm = recording.readframes(frames)
            if len(pcm) != frames * channels * width:
                raise ValueError("Truncated WAV payload")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid WAV container") from exc
    raw = np.frombuffer(pcm, dtype="<i2").astype(np.float64).reshape(-1, channels) / 32768
    return raw.mean(axis=1), rate, np.any(np.abs(raw) >= 0.999, axis=1)


def features(data):
    samples, rate, clipped = read_wav(data)
    size = int(rate * FRAME_SECONDS)
    count = len(samples) // size
    if count < 4:
        raise ValueError("Audio must contain at least 0.2 seconds")
    blocks = samples[:count * size].reshape(count, size)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    peak_rms = float(rms.max())
    if peak_rms < 1e-5:
        raise ValueError("Recording contains no usable signal")
    active = rms > peak_rms * 10 ** (-35 / 20)
    reference = float(np.median(rms[active]))
    relative_db = 20 * np.log10(np.maximum(rms, 1e-10) / reference)
    spectrum = np.abs(np.fft.rfft(blocks * np.hanning(size), axis=1))
    frequencies = np.fft.rfftfreq(size, 1 / rate)
    centroid = (spectrum * frequencies).sum(axis=1) / np.maximum(spectrum.sum(axis=1), 1e-10)
    crossings = np.mean(np.signbit(blocks[:, 1:]) != np.signbit(blocks[:, :-1]), axis=1)
    clipping = clipped[:count * size].reshape(count, size).mean(axis=1)
    return {
        "durationSeconds": len(samples) / rate,
        "sampleRate": rate,
        "frameSeconds": FRAME_SECONDS,
        "frames": [
            {"time": round(i * FRAME_SECONDS, 4), "active": bool(active[i]),
             "relativeDb": round(float(relative_db[i]), 3),
             "centroidHz": round(float(centroid[i]), 2),
             "zeroCrossingRate": round(float(crossings[i]), 5),
             "clippingFraction": round(float(clipping[i]), 5)}
            for i in range(count)
        ],
    }


def regions(mask, kind, explanations, minimum_frames=3):
    events = []
    start = None
    for index, matched in enumerate([*mask, False]):
        if matched and start is None:
            start = index
        elif not matched and start is not None:
            if index - start >= minimum_frames:
                events.append({"kind": kind, "start": round(start * FRAME_SECONDS, 4),
                               "end": round(index * FRAME_SECONDS, 4),
                               "evidence": explanations[start:index]})
            start = None
    return events


def compare(baseline, practice, energy_delta_db=6):
    if not isinstance(energy_delta_db, (int, float)) or not np.isfinite(energy_delta_db) or not 3 <= energy_delta_db <= 20:
        raise ValueError("Energy threshold must be between 3 and 20 dB")
    base, attempt = features(baseline), features(practice)
    if abs(base["durationSeconds"] - attempt["durationSeconds"]) > FRAME_SECONDS / 2:
        raise ValueError("Timelines differ. Acoustic comparison requires pre-aligned audio of equal duration; word alignment does not apply a time warp")
    if len(base["frames"]) != len(attempt["frames"]):
        raise ValueError("Frame counts differ; supply pre-aligned recordings")
    pause, energy, clipping = [], [], []
    pause_evidence, energy_evidence, clip_evidence = [], [], []
    for a, b in zip(base["frames"], attempt["frames"]):
        delta = b["relativeDb"] - a["relativeDb"]
        pause.append(a["active"] and not b["active"])
        energy.append(a["active"] and b["active"] and abs(delta) >= energy_delta_db)
        clipping.append(b["clippingFraction"] >= 0.02)
        pause_evidence.append({"time": a["time"], "baselineActive": a["active"], "practiceActive": b["active"]})
        energy_evidence.append({"time": a["time"], "normalizedDeltaDb": round(delta, 3)})
        clip_evidence.append({"time": b["time"], "clippedSampleFraction": b["clippingFraction"]})
    events = (regions(pause, "additional-silence", pause_evidence)
              + regions(energy, "relative-energy-change", energy_evidence)
              + regions(clipping, "signal-clipping", clip_evidence))
    events.sort(key=lambda event: (event["start"], event["kind"]))
    from rubric import acoustic_rubric
    return {"schemaVersion": 1, "method": "pre-aligned-acoustic-comparison-v1",
            "thresholdDb": energy_delta_db, "baseline": base, "practice": attempt,
            "events": events, "qualityScore": None,
            "rubric": acoustic_rubric(base['frames'], attempt['frames'], energy_delta_db),
            "limitations": ["Acoustic timeline comparison assumes pre-aligned audio; optional word alignment does not warp it",
                            "Differences are acoustic evidence, not proof of poor delivery",
                            "No calibrated rhetorical score or pitch tracking; optional MFCC distances are uncalibrated",
                            "50 ms frame resolution; events require at least 150 ms" ]}


def signal_fixture():
    rate = 16000
    time = np.arange(rate * 4) / rate
    original = 0.2 * np.sin(2 * np.pi * 220 * time)
    changed = original.copy()
    changed[rate:rate * 3 // 2] = 0
    changed[rate * 2:rate * 5 // 2] *= 4
    def encode(samples):
        stream = io.BytesIO()
        with wave.open(stream, "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(rate)
            recording.writeframes((samples * 32767).astype("<i2").tobytes())
        return stream.getvalue()
    return encode(original), encode(changed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("practice", type=Path)
    parser.add_argument("--threshold-db", type=float, default=6)
    args = parser.parse_args()
    for path in (args.baseline, args.practice):
        if path.stat().st_size > MAX_BYTES:
            parser.error("Input exceeds 20 MiB")
    print(json.dumps(compare(args.baseline.read_bytes(), args.practice.read_bytes(), args.threshold_db)))
