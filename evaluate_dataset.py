"""Recompute temporal detection metrics; never trust cached observed events."""
import argparse
import hashlib
import json
from pathlib import Path

from audio_engine import compare


def overlap(a, b):
    intersection = max(0, min(a['end'], b['end']) - max(a['start'], b['start']))
    union = max(a['end'], b['end']) - min(a['start'], b['start'])
    return intersection / union if union > 0 else 0


def evaluate(baseline, manifest_path, threshold=6):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    raw = Path(baseline).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['baselineSha256']:
        raise ValueError('Baseline checksum mismatch')
    rows = []
    tp = fp = fn = 0
    for variant in manifest['variants']:
        path = manifest_path.parent / variant['file']
        if path.resolve().parent != manifest_path.parent.resolve():
            raise ValueError('Variant path escapes dataset')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != variant['sha256']:
            raise ValueError('Variant checksum mismatch')
        events = compare(raw, data, threshold)['events']
        unmatched = set(range(len(events)))
        matches = []
        for label in variant['labels']:
            candidates = [(overlap(label, events[i]), i) for i in unmatched
                          if label['kind'] == events[i]['kind']]
            score, index = max(candidates, default=(0, -1))
            if score >= .5:
                unmatched.remove(index)
                matches.append({'kind': label['kind'], 'iou': score,
                                'startErrorSeconds': events[index]['start'] - label['start'],
                                'endErrorSeconds': events[index]['end'] - label['end']})
        tp += len(matches)
        fn += len(variant['labels']) - len(matches)
        fp += len(unmatched)
        rows.append({'file': variant['file'], 'labels': len(variant['labels']),
                     'detections': len(events), 'matches': matches,
                     'unmatchedEvents': [{key: events[i][key] for key in ('kind', 'start', 'end')}
                                         for i in sorted(unmatched)]})
    return {'schemaVersion': 1, 'scope': 'Controlled acoustic interventions on one speech excerpt; not held-out human quality evaluation',
            'energyThresholdDb': threshold, 'matchingIoU': .5,
            'truePositives': tp, 'falsePositives': fp, 'falseNegatives': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None, 'variants': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.baseline, args.manifest)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'variants'}))
