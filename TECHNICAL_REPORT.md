# CadenceLab: Evidence-First Acoustic Review

Working technical report, October 7, 2026. Not a submission or a claim of validated rhetorical assessment. AI-assisted implementation by Codex.

## Problem and Boundary

Compare a reference speech with a controlled practice recording and locate measurable deviations. Acoustic differences alone cannot establish poor delivery, intent, intelligibility or speaker quality. The system emits timestamped evidence and a declared acoustic-fidelity rubric, not a validated human delivery grade. It is not suitable for employment, medical or educational selection decisions.

## Data and Rights

The included baseline is a 19.9883125-second excerpt of John F. Kennedy's September 12, 1962 Rice speech. Source, public-domain marking, original and excerpt hashes, conversion settings and rights caveat are recorded in `dataset/provenance.json`. The provisional transcript is matched to Rice University's published federal speech text. Archival noise, ASR disagreement and partial speech at the final cut remain limitations; word boundaries require review.

The corpus includes disjoint development (120-140 s), validation (240-260 s) and held-out (360-380 s) excerpts from this recording. Each has seven deterministic variants: three silences (0.2, 0.5, 1 second), three local energy reductions (-3, -9, -18 dB), and a global half-gain control. Labels are exact intervention regions, not human rhetorical judgments. Checksums ensure evaluation uses the specified audio. All 21 variants share one speaker/recording; none constitutes independent-speaker evidence.

## Feature Extraction

PCM input is decoded with Python's WAV parser and normalized to floating-point amplitude. Complete nonoverlapping 50 ms windows yield RMS, energy relative to median active-frame RMS, Hann FFT centroid, zero crossings and clipped-sample fractions. Activity is relative to each recording's peak RMS by 35 dB. At least 150 ms is required for an event. Additional silence, normalized energy deltas and clipping are reported separately.

Optional Kaldi-compatible MFCC extraction uses TorchAudio, not an improvised cepstral implementation: 13 cepstra, 23 mel filters, 50 ms windows and hops, dither zero, complete frames, 16 kHz PCM. Distance is the root mean square coefficient difference across C1-C12 per frame. C0 is excluded from distance. Microphone, noise and phonetic differences remain confounders; no spectral-distance quality threshold is asserted.

## Actual Forced Alignment

Wav2Vec2 ASR BASE 960H produces English CTC emissions locally. TorchAudio 2.8 `forced_align` computes a monotonic alignment to the supplied transcript; merged token spans form words on the emission clock. Actual inference produced 24 alignment files: 48 words per development recording, 31 per validation recording and 38 per held-out recording. Token coverage, ordered boundaries and frame-clock bounds are validated. Mean token posterior is explicitly not a calibrated word accuracy probability. The same transcript can be aligned to both uploads in the dashboard.

Critically, this is not a time-warping implementation: the acoustic comparator still requires pre-aligned, equal-duration inputs. Alignment can force words into noise or silence when a supplied transcript is wrong. ASR disagreement is displayed as a warning. Independently spoken practice recordings require a future reviewed mapping before acoustic region comparisons become valid.

## Evaluation and Evidence

`evaluate_dataset.py` recomputes predictions from checksum-verified WAV bytes rather than cached events. One-to-one matches require equal event kind and temporal intersection-over-union at least 0.5. At the fixed 6 dB energy threshold, development has 5 TP, 0 FP, 1 FN (precision 1.0, recall 0.8333); validation and held-out each have 5 TP, 1 FP, 1 FN (precision/recall 0.8333). The -3 dB intervention is missed. Fragmented strong attenuation generates an extra unmatched region on the latter two excerpts. Global gain produces no event. These are controlled signal results, not held-out human speech-quality performance; the threshold was frozen before evaluation.

55 local tests cover decoding, signal logic, temporal matching, deterministic construction, alignment span validation, actual MFCC extraction, rubric monotonicity, HTTP uploads, bounded report retrieval and request rejection. Actual model inference and real-speech browser uploads are separate verification evidence, not inferred from helper tests. Desktop and 390x844 views were inspected; the browser layout check found no horizontal overflow. Downloaded report JSON was verified; JavaScript syntax check passed.

## Reproducible Acoustic Rubric

Version acoustic-fidelity-v1 uses score = 100 * (1 - 0.45*S - 0.40*E - 0.15*C). S is additional-silence frames divided by active reference frames. E is summed min(1, max(0, absolute normalized energy difference - threshold) / 12) over mutually active frames, divided by active reference frames. C is practice frames with at least 2% clipped samples divided by all frames. The score is bounded to 0-100. Weights and the 12 dB saturation are declared engineering choices, not learned from human ratings. MFCC distance and alignment posterior are deliberately not folded into the score. The separate qualityScore field remains null. Reference identity scores 100 and stronger injected attenuation reduces the score in tests.

## Interface and Privacy

The loopback dashboard provides upload playback, shared transcript text/TXT upload, optional MFCC, normalized-energy timeline, detected-region evidence, acoustic rubric, spectral-distance timeline and word-boundary table. It sends audio only to the local server. The server checks loopback Host and same-origin POST, bounds payloads and has a static-file allowlist. Uploaded audio is not saved. Up to eight reports are retained in memory for tokenized export until eviction/restart. Model weights must be installed through the CLI first; browser analysis does not download them. This is a single-user local development server, not an authenticated production deployment.

## Remaining Submission Work

Reviewed transcript/boundary labels, independent-speaker validation, support for independently timed speech, public dataset/source under reviewed rights and the required demo video remain. The included held-out excerpt evaluates injected signal regions only. No registration or entry submission is claimed. No earnings or reward acceptance is implied.

## Primary References

- Recording provenance: https://commons.wikimedia.org/wiki/File:Jfk_rice_university_we_choose_to_go_to_the_moon.ogg
- Published speech: https://www.rice.edu/kennedy
- CTC alignment: https://docs.pytorch.org/audio/2.8/tutorials/forced_alignment_tutorial.html
- MFCC implementation: https://docs.pytorch.org/audio/2.8/generated/torchaudio.compliance.kaldi.mfcc.html
- Competition rules: https://multimodal-ai-hackathon-2026-7.devpost.com/rules
