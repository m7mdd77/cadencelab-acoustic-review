"""Versioned, declared acoustic-fidelity rubric; not an oratory grade."""
import numpy as np


def acoustic_rubric(base_frames, practice_frames, threshold_db):
    if not base_frames or len(base_frames) != len(practice_frames):
        raise ValueError('Rubric requires matching nonempty frame clocks')
    active = sum(frame['active'] for frame in base_frames)
    if not active:
        raise ValueError('Rubric requires active baseline frames')
    pauses = sum(a['active'] and not b['active'] for a, b in zip(base_frames, practice_frames)) / active
    energy = sum(min(1, max(0, abs(a['relativeDb'] - b['relativeDb']) - threshold_db) / 12)
                 for a, b in zip(base_frames, practice_frames) if a['active'] and b['active']) / active
    clipping = sum(b['clippingFraction'] >= .02 for b in practice_frames) / len(practice_frames)
    penalties = {'additionalSilence': pauses, 'energyDeviation': energy, 'clipping': clipping}
    weights = {'additionalSilence': .45, 'energyDeviation': .40, 'clipping': .15}
    score = 100 * (1 - sum(weights[key] * penalties[key] for key in weights))
    if not np.isfinite(score):
        raise ValueError('Nonfinite rubric evidence')
    return {'version': 'acoustic-fidelity-v1', 'score': round(max(0, min(100, score)), 3),
            'scale': [0, 100], 'weights': weights, 'penalties': penalties,
            'energyThresholdDb': threshold_db, 'energySaturationAboveThresholdDb': 12,
            'definition': '100*(1-0.45*additionalSilenceFraction-0.40*meanExcessEnergySeverity-0.15*clippedFrameFraction)',
            'interpretation': 'Declared acoustic-fidelity rubric, not a calibrated rhetorical-quality or intelligibility score',
            'limitations': ['Weights are engineering choices, not learned from judge ratings',
                            'Scores require pre-aligned recordings of the same text',
                            'Spectral distance and forced alignment confidence are not folded into this score']}
