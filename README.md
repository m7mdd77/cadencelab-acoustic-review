# CadenceLab

Local pre-aligned WAV comparison prototype. AI-assisted implementation by Codex, October 7, 2026. Source publication does not imply competition submission, acceptance or earnings.

## Run

Python 3.11+ and NumPy are required. Install `requirements.txt` in a project-local virtual environment, then run:

```sh
python -m unittest -v
python server.py
```

Open http://127.0.0.1:3261 . The server binds to loopback, stores no uploaded audio and makes no external calls. Do not expose it publicly: no authentication or hardened multi-user deployment is provided.

Upload two uncompressed 16-bit PCM mono/stereo WAV recordings, 8-192 kHz, at most 180 seconds and 20 MiB each. Recordings must already share an aligned timeline and duration. Independent recordings generally do not satisfy this prerequisite. Matching duration alone does NOT prove alignment.

CLI: `python audio_engine.py baseline.wav practice.wav --threshold-db 6` writes JSON to stdout.

## Method

50 ms non-overlapping frames; incomplete final frame omitted. RMS, relative energy normalized against each recording's median active-frame energy, Hann-window FFT spectral centroid, zero crossing rate and clipping fraction. Activity threshold: 35 dB below each recording's peak frame RMS. Runs of at least three frames are flagged for additional silence, normalized energy delta (default 6 dB), or at least 2% clipped samples. Full-frame mathematical evidence is exported. Global gain normalization reduces microphone-level differences but does not make this model universally speaker-agnostic.

These are measured acoustic deviations, not evidence of bad rhetoric, speaker intent, intelligibility, or semantic correctness. No calibrated quality score is emitted. The custom acoustic rubric below is explicitly uncalibrated. Noise and recording conditions can cause false positives. No medical, hiring or educational selection use.

## Submission Assets

`build_report.py` uses ReportLab and pypdf to render the technical report and enforce its six-page maximum. `package_submission.py` creates a checksum-manifested source/dataset ZIP from explicit file classes, excluding model weights, cache, virtual environment, credentials and logs. The assets are local; no publication or competition submission is implied.

The repository also distributes `CadenceLab-source-dataset.zip`. Extract its `cadencelab` folder to obtain the dataset alongside the runnable source. Model weights are intentionally excluded and obtained through the official TorchAudio CLI pipeline. MIT applies to original code; speech recording rights are separately described in dataset/provenance.json.

`CadenceLab-narrated-walkthrough.mp4` is a 4 minute 50 second walkthrough assembled from actual application captures with synthetic Windows narration. It is explicitly not a continuous screen recording. `demo-manifest.json` records the captions, narration and timings. It does not claim a human listening review or a submitted entry.

## Competition Fit / Remaining Work

Candidate: Multimodal AI Hackathon 2026, Track C. Rules permit AI assistants with disclosure and solo entries. Official rules: https://multimodal-ai-hackathon-2026-7.devpost.com/rules . Track brief: https://drive.google.com/file/d/1ewVNqeGBcg76GUUvoj0buPWueQrgDf_6/view .

Entry not yet submitted. Model-derived transcript boundaries remain provisional and independent-speaker validation is not demonstrated. The technical PDF is two pages; the local walkthrough is within the three-to-ten-minute requirement. MFCC, dashboard transcript alignment, disjoint-excerpt evaluation and a declared acoustic rubric are implemented. Synthetic tones remain unit fixtures, not speech data. User approval is required at applicable registration/terms and publication steps. No paid services needed.

## Real Speech Pipeline

`dataset/provenance.json` records the recording's public-domain marking and source URLs. A 16 kHz mono Kennedy excerpt and seven deterministic real-speech variants are included. Labels describe signal interventions, not rhetorical quality. One speaker and one excerpt cannot demonstrate generalization. Do not call these held-out human delivery samples.

The optional model pipeline pins TorchAudio 2.8 because its genuine CTC alignment API was removed in 2.9. Install CPU packages locally using the official PyTorch CPU index; the first CLI run downloads Wav2Vec2 weights into `.cache/models`. The dashboard can align both uploads when a shared English transcript is supplied, but requires those weights already present locally. It never initiates their download. Word alignment does not time-warp the acoustic comparison; independently timed recordings are not yet supported.

The optional MFCC checkbox uses TorchAudio's Kaldi-compatible implementation with 13 coefficients, 23 mel bins, 50 ms windows/hops, no dither, complete frames only, and 16 kHz inputs up to 30 seconds. Full baseline/practice coefficients and RMS C1-C12 differences are exported as evidence. C0 is excluded from distance because energy is already represented separately. Distances have no calibrated quality interpretation. No pitch tracker is claimed.

Use the local virtual environment containing both requirements files for the full test suite and server:

```powershell
.\.venv\Scripts\python.exe -m unittest -q
.\.venv\Scripts\python.exe server.py
```

Verified on October 7: 55 tests passed. Real speech uploaded through both browser file inputs, optional MFCC and actual CTC alignment rendered (48 rows), desktop/mobile checked. A downloaded report JSON was verified; export is no longer marked unverified. Twenty-four actual CTC alignments cover three excerpt baselines and their 21 modified recordings. Shared transcripts and word boundaries remain provisional.

`build_corpus.py` creates development, validation and held-out excerpts from disjoint sections of the same recording. At the frozen 6 dB threshold, development has precision 1.0 / recall 0.8333; validation and held-out each have precision 0.8333 / recall 0.8333. These are not independent speakers. `align_corpus.py` executes actual model alignment for every recording.

`rubric.py` publishes an acoustic-fidelity score with 45% additional-silence, 40% energy-deviation and 15% clipping penalties. Its weights are engineering choices, not human-calibrated delivery judgments. The separate `qualityScore` remains null. The UI accepts a shared transcript as text or a TXT upload.

```sh
python alignment.py dataset/jfk-rice-120-140.wav --transcript dataset/jfk-rice-120-140.txt --output dataset/jfk-rice-120-140-alignment.json
python build_dataset.py dataset/jfk-rice-120-140.wav dataset/jfk-rice-120-140-variants
python evaluate_dataset.py dataset/jfk-rice-120-140.wav dataset/jfk-rice-120-140-variants/manifest.json --output dataset/jfk-rice-120-140-variants/evaluation.json
```

The actual alignment produced 48 word spans; transcript and boundaries remain provisional. ASR mistakes and partial speech at the clip end are disclosed in provenance. Token posteriors are not calibrated accuracy probabilities.

Recomputed controlled results at the unchanged 6 dB threshold: 5 true positives, 0 false positives, 1 false negative, precision 1.0, recall 0.8333, with same-kind temporal IoU >= 0.5 and one-to-one matching. The -3 dB intervention is missed; the global-gain control yields no event. This is an acoustic construction check, not a trained-model quality benchmark.
