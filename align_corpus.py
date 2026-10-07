"""Execute genuine alignment for corpus pairs; retain uncertainty warnings."""
import argparse
import hashlib
import json
from pathlib import Path

from alignment import run


def align_corpus(root):
    root = Path(root)
    descriptor = json.loads((root / 'corpus-report.json').read_text())
    summary = {'schemaVersion': 1, 'transcriptSource': 'https://www.rice.edu/kennedy',
               'reviewStatus': 'Published text matched to ASR excerpt context, not independent listening or manual timestamp certification',
               'boundaryCaveat': 'Cuts can contain partial words; CTC may force supplied words into noise or muted spans',
               'alignments': []}
    for split in descriptor['splits']:
        directory = root / split['name']
        transcript = (directory / 'transcript.txt').read_text(encoding='utf-8')
        manifest = json.loads((directory / 'variants/manifest.json').read_text())
        files = [('baseline.wav', split['baselineSha256'])] + [('variants/' + variant['file'], variant['sha256']) for variant in manifest['variants']]
        for relative, expected_hash in files:
            path = directory / relative
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError('Corpus audio checksum mismatch')
            result = run(raw, transcript)
            result['transcriptOrigin'] = 'published-speech-excerpt-provisional'
            result['warnings'].append(summary['boundaryCaveat'])
            output = path.with_suffix('.alignment.json')
            output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            summary['alignments'].append({'file': str(output.relative_to(root)), 'wordCount': len(result['words']),
                                          'audioSha256': expected_hash, 'transcriptSha256': hashlib.sha256(transcript.encode()).hexdigest()})
            print(f"Aligned {split['name']}/{relative}: {len(result['words'])} words", flush=True)
            (root / 'alignment-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=Path)
    args = parser.parse_args()
    align_corpus(args.corpus)
