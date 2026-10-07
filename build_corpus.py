"""Fixed, disjoint excerpt splits; no threshold search or model training."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from build_dataset import build
from evaluate_dataset import evaluate

SOURCE_SHA256 = '7486fba805c20bde4d43750bd9c34aee86ab7acfd3854a1d0f3d64f62ae68b27'
SPLITS = [('development', 120, 140), ('validation', 240, 260), ('heldout', 360, 380)]


def validate_splits(splits):
    seen = set()
    for index, (name, start, end) in enumerate(splits):
        if name in seen or start < 0 or not start < end or end - start > 30:
            raise ValueError('Invalid excerpt split')
        seen.add(name)
        for _, a, b in splits[:index]:
            if max(a, start) < min(b, end):
                raise ValueError('Source intervals overlap across splits')


def build_corpus(original, ffmpeg, destination):
    validate_splits(SPLITS)
    original, ffmpeg, root = Path(original), Path(ffmpeg), Path(destination)
    if hashlib.sha256(original.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Original recording checksum mismatch')
    if not ffmpeg.is_file():
        raise ValueError('Provide the existing trusted FFmpeg executable')
    root.mkdir(parents=True, exist_ok=True)
    report = {'schemaVersion': 1, 'sourceSha256': SOURCE_SHA256,
              'thresholdDb': 6, 'thresholdSelection': 'Frozen existing default; no search on validation or heldout',
              'scope': 'Disjoint excerpts from ONE speaker and recording, not independent-speaker or human delivery validation',
              'labels': 'Injected acoustic regions only', 'splits': []}
    for name, start, end in SPLITS:
        directory = root / name
        directory.mkdir(exist_ok=True)
        baseline = directory / 'baseline.wav'
        subprocess.run([str(ffmpeg.resolve()), '-hide_banner', '-loglevel', 'error', '-i', str(original.resolve()),
                        '-ss', str(start), '-t', str(end - start), '-ar', '16000', '-ac', '1',
                        '-c:a', 'pcm_s16le', '-y', str(baseline.resolve())], check=True)
        variants = directory / 'variants'
        manifest = build(baseline, variants)
        metrics = evaluate(baseline, variants / 'manifest.json', threshold=6)
        (directory / 'evaluation.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
        report['splits'].append({'name': name, 'sourceStartSeconds': start, 'sourceEndSeconds': end,
                                 'baselineSha256': manifest['baselineSha256'], 'variants': len(manifest['variants']),
                                 'transcriptStatus': 'Not reviewed or aligned; no transcript accuracy claim',
                                 'metrics': {key: value for key, value in metrics.items() if key != 'variants'}})
    (root / 'corpus-report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_corpus(args.original, args.ffmpeg, args.output)
    for split in result['splits']:
        print(split['name'], json.dumps(split['metrics']))
