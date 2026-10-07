"""Local Wav2Vec2 emissions and TorchAudio CTC forced alignment.

Model token posteriors are not calibrated word-accuracy probabilities.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from audio_engine import MAX_BYTES, read_wav


def normalize_transcript(text):
    if not isinstance(text, str) or not 0 < len(text) <= 4000:
        raise ValueError("Provide an English transcript of 1-4000 characters")
    text = text.replace("\u2019", "'").upper()
    if re.search(r"[^A-Z'\s.,!?;:\-()]", text):
        raise ValueError("Use English words and spell out numbers; unsupported characters are not silently removed")
    words = re.sub(r"[.,!?;:\-()]", " ", text).split()
    if not words or any(not re.search("[A-Z]", word) for word in words):
        raise ValueError("Transcript must contain English words")
    return "|".join(words)


def word_segments(spans, labels, tokens, duration, frame_count):
    if frame_count <= 0 or duration <= 0:
        raise ValueError("Invalid frame clock")
    if [int(span.token) for span in spans] != list(tokens):
        raise ValueError("Alignment did not cover the supplied transcript exactly")
    ratio = duration / frame_count
    result, group = [], []
    def finish():
        if not group:
            return
        length = sum(span.end - span.start for span in group)
        if length <= 0:
            raise ValueError("Invalid token duration")
        result.append({"word": "".join(labels[int(span.token)] for span in group),
                       "start": round(group[0].start * ratio, 4),
                       "end": round(group[-1].end * ratio, 4),
                       "meanTokenPosterior": round(sum(float(span.score) * (span.end - span.start) for span in group) / length, 6)})
    previous = 0
    for span in spans:
        if not previous <= span.start < span.end <= frame_count:
            raise ValueError("Non-monotonic or out-of-bounds token spans")
        previous = span.end
        if labels[int(span.token)] == "|":
            finish()
            group = []
        else:
            group.append(span)
    finish()
    return result


def run(audio, transcript=None, cache_dir=None):
    import torch
    import torchaudio
    from torchaudio import functional as F

    if not torchaudio.__version__.startswith("2.8."):
        raise RuntimeError("This pipeline requires TorchAudio 2.8; do not silently replace its removed alignment API")
    samples, rate, _ = read_wav(audio)
    if len(samples) / rate > 30:
        raise ValueError("Align clips of at most 30 seconds; split longer speeches first")
    cache = Path(cache_dir or Path(__file__).parent / ".cache" / "models")
    cache.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache))
    torch.set_num_threads(4)
    torch.manual_seed(0)
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval()
    labels = bundle.get_labels()
    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    if rate != bundle.sample_rate:
        waveform = F.resample(waveform, rate, bundle.sample_rate)
    with torch.inference_mode():
        emissions, _ = model(waveform)
        emissions = torch.log_softmax(emissions, dim=-1)
    indices = torch.unique_consecutive(emissions[0].argmax(dim=-1)).tolist()
    recognized = "".join(labels[index] for index in indices if index != 0).replace("|", " ").strip()
    result = {"schemaVersion": 1, "model": "WAV2VEC2_ASR_BASE_960H",
              "torchVersion": torch.__version__, "torchaudioVersion": torchaudio.__version__,
              "audioSha256": hashlib.sha256(audio).hexdigest(),
              "durationSeconds": len(samples) / rate, "machineTranscript": recognized,
              "transcriptOrigin": "user-supplied" if transcript is not None else "model-generated-unverified",
              "words": [], "warnings": ["Token posteriors are not calibrated accuracy probabilities",
                                          "Acoustic alignment must be reviewed before serving as ground-truth labels"]}
    if transcript is not None:
        normalized = normalize_transcript(transcript)
        mapping = {character: index for index, character in enumerate(labels)}
        tokens = [mapping[character] for character in normalized]
        if len(tokens) >= emissions.shape[1]:
            raise ValueError("Transcript is too long for this recording")
        targets = torch.tensor([tokens], dtype=torch.int32)
        with torch.inference_mode():
            aligned, scores = F.forced_align(emissions, targets, blank=0)
            spans = F.merge_tokens(aligned[0], scores[0].exp(), blank=0)
        result["normalizedTranscript"] = normalized.replace("|", " ")
        result["words"] = word_segments(spans, labels, tokens, len(samples) / rate, emissions.shape[1])
        if result["normalizedTranscript"] != recognized:
            result["warnings"].append("Supplied transcript differs from greedy ASR; review discrepancies")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.audio.stat().st_size > MAX_BYTES:
        parser.error("Audio exceeds size limit")
    text = args.transcript.read_text(encoding="utf-8") if args.transcript else None
    result = run(args.audio.read_bytes(), text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.output}; {len(result['words'])} aligned words")
