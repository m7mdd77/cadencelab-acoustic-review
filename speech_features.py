"""Optional Kaldi-compatible MFCC evidence on the existing 50 ms clock."""
import numpy as np

from audio_engine import FRAME_SECONDS, read_wav


def mfcc(data):
    import torch
    import torchaudio

    samples, rate, _ = read_wav(data)
    if len(samples) / rate > 30:
        raise ValueError('MFCC analysis supports clips of at most 30 seconds')
    if rate != 16000:
        raise ValueError('MFCC comparison requires 16 kHz WAV recordings')
    torch.set_num_threads(4)
    signal = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        matrix = torchaudio.compliance.kaldi.mfcc(
            signal, sample_frequency=rate, frame_length=50, frame_shift=50,
            num_ceps=13, num_mel_bins=23, dither=0, snip_edges=True,
            use_energy=False, subtract_mean=False)
    result = matrix.numpy().astype(np.float64)
    if result.shape != (len(samples) // int(rate * FRAME_SECONDS), 13) or not np.isfinite(result).all():
        raise ValueError('MFCC output does not match the expected frame clock')
    return result


def compare_mfcc(baseline, practice):
    a, b = mfcc(baseline), mfcc(practice)
    if a.shape != b.shape:
        raise ValueError('MFCC frame clocks differ')
    # C0 captures overall log energy, already represented by the RMS timeline.
    distances = np.sqrt(np.mean((a[:, 1:] - b[:, 1:]) ** 2, axis=1))
    return {'method': 'torchaudio.compliance.kaldi.mfcc',
            'frameSeconds': FRAME_SECONDS, 'coefficients': 13,
            'distanceDefinition': 'RMS difference of coefficients C1-C12; C0 excluded',
            'meanDistance': round(float(distances.mean()), 6),
            'frames': [{'time': round(i * FRAME_SECONDS, 4),
                        'distance': round(float(distance), 6),
                        'baseline': [round(float(x), 6) for x in a[i]],
                        'practice': [round(float(x), 6) for x in b[i]]}
                       for i, distance in enumerate(distances)],
            'limitations': ['Uncalibrated spectral difference, not delivery quality',
                            'Noise, microphone and phonetic differences affect MFCCs',
                            'Pre-aligned timelines required; no pitch or semantic score']}
