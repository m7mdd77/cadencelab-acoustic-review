"""Narrated actual-capture walkthrough, explicitly not continuous recording."""
import argparse
import json
import math
import subprocess
import wave
from pathlib import Path

SEGMENTS = [
    ('demo-inputs.png', 'Scope and inputs',
     'This is CadenceLab, an AI-assisted acoustic comparison prototype for Multimodal AI Track C. '
     'This walkthrough uses actual application screenshots and synthetic Windows narration; it is not a continuous screen recording. '
     'Two WAV uploads are compared on an already aligned timeline. The optional English transcript enables genuine local forced alignment. '
     'Audio stays on the loopback server. CadenceLab measures signal deviations; it does not certify rhetorical skill.'),
    ('technical-report-page-1.png', 'Licensed data and stress gradient',
     'The reference recording is Kennedy at Rice University in nineteen sixty two, sourced from Wikimedia with its federal public-domain provenance recorded. '
     'Three disjoint twenty-second excerpts create development, validation and held-out signal tests. Each gets seven deterministic variants: '
     'three injected pauses, three local energy reductions, and one global gain control. SHA two fifty six hashes identify the audio. '
     'These are construction labels, not independent human judgments, and all excerpts share one speaker.'),
    ('demo-analysis.png', 'Actual completed comparison',
     'Here the actual application has analyzed the development excerpt against a one-second muted variant. '
     'The reference duration is nineteen point nine nine seconds. The shared transcript was loaded from a text file, and MFCC evidence was selected. '
     'The acoustic-fidelity score is a declared engineering rubric, not a validated delivery grade. Its penalties cover additional silence, excess normalized energy deviation and clipping. '
     'The displayed score must always be interpreted with the region evidence and methodological limits.'),
    ('demo-features.png', 'Temporal evidence and richer features',
     'The normalized energy comparison localizes the injected silence between six point three five and seven point three five seconds. '
     'The spectral view comes from genuine TorchAudio Kaldi MFCC extraction, thirteen coefficients with fifty-millisecond windows. '
     'C zero is excluded from the distance because energy is separately represented. The mean C one through C twelve distance is one point four zero three in this example. '
     'Microphone differences, noise and phonetics can also affect spectral distance, so it remains uncalibrated.'),
    ('demo-alignment.png', 'Genuine model alignment and uncertainty',
     'The word table is produced by local Wav2Vec2 emissions and TorchAudio CTC forced alignment, not by invented timestamps. '
     'Twenty-four actual corpus recordings were aligned. This excerpt yields forty-eight word spans. '
     'Muting speech can move model boundaries, because CTC still tries to place the supplied transcript. '
     'The interface warns that boundaries need review and that alignment does not time-warp the acoustic comparison. Independently timed practice recordings are therefore not currently supported.'),
    ('technical-report-page-2.png', 'Reproducible evaluation and limitations',
     'Evaluation recomputes predictions from checksum-verified WAV bytes. Same-kind regions require temporal intersection over union of at least one half and one-to-one matching. '
     'At the frozen six-decibel threshold, development precision is one and recall is zero point eight three three. '
     'Validation and held-out precision and recall are both zero point eight three three. '
     'The three-decibel intervention is missed, and fragmented stronger attenuation produces a false-positive region. These limitations are retained rather than tuning them away.'),
    ('demo-limits.png', 'Export, tests and honest status',
     'The report export contains measured frames, events, optional MFCC matrices, word alignment and the explicit rubric formula. '
     'Fifty-five local tests passed, and the real-speech browser upload and downloaded JSON were verified separately. '
     'The score uses forty-five percent silence, forty percent excess energy and fifteen percent clipping penalties; these weights are declared choices, not learned human ratings. '
     'Transcript review, independent-speaker validation and differently timed speech remain future work. This local walkthrough is not a claim of competition submission, acceptance or earnings.'),
]


def build(ffmpeg, shell):
    root = Path(__file__).resolve().parent
    ffmpeg = str(Path(ffmpeg).resolve())
    output = root / '.cache/demo'
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 'Actual application still captures with synthetic Windows narration; not continuous recording', 'segments': []}
    for index, (image, title, text) in enumerate(SEGMENTS):
        image_path = root / image
        if not image_path.is_file():
            raise ValueError(f'Missing actual evidence capture: {image}')
        stem = output / f'segment-{index:02d}'
        txt, wav, video = stem.with_suffix('.txt'), stem.with_suffix('.wav'), stem.with_suffix('.mp4')
        txt.write_text(text, encoding='utf-8')
        subprocess.run([shell, '-NoProfile', '-File', str(root / 'narrate-demo.ps1'),
                        '-TextFile', str(txt), '-OutputFile', str(wav)], check=True)
        with wave.open(str(wav)) as audio:
            speech = audio.getnframes() / audio.getframerate()
        duration = max(30, math.ceil(speech) + 2)
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-loop', '1', '-framerate', '12',
                        '-i', str(image_path), '-i', str(wav), '-vf',
                        'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:white',
                        '-af', 'apad', '-t', str(duration), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ar', '22050', '-ac', '1', str(video)], check=True)
        manifest['segments'].append({'title': title, 'capture': image, 'seconds': duration,
                                     'narrationSeconds': speech, 'narration': text})
    listing = output / 'concat.txt'
    listing.write_text('\n'.join(f"file 'segment-{i:02d}.mp4'" for i in range(len(SEGMENTS))), encoding='ascii')
    target = root / 'CadenceLab-narrated-walkthrough.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
                    '-i', listing.name, '-c', 'copy', '-movflags', '+faststart', str(target)], check=True, cwd=output)
    manifest['durationSeconds'] = sum(s['seconds'] for s in manifest['segments'])
    if not 180 <= manifest['durationSeconds'] <= 600:
        raise ValueError('Demo falls outside required three-to-ten-minute duration')
    (root / 'demo-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'path': target.name, 'seconds': manifest['durationSeconds'], 'bytes': target.stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--shell', default='powershell.exe')
    args = parser.parse_args()
    build(args.ffmpeg, args.shell)
