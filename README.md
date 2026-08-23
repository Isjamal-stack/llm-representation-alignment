# Audio Stitching — Phase A Starter Code

This is a runnable starting point for Phase A of the protocol: validating the
stitching pipeline on speech models (wav2vec2 / HuBERT / WavLM) before moving
to audio-language or physiological signals.

## 1. Where to run this

**On Unity HPC?** Skip straight to [`UNITY_SETUP.md`](UNITY_SETUP.md) — it
covers storage quotas, conda setup via modules, interactive debug sessions,
and the `sbatch` script (`run_phase_a_sanity.sbatch`) to submit the real run.

**Anywhere else** (Colab, a lab GPU box, a cloud rental): you need a machine
with a GPU and internet access. A single consumer GPU is enough for Phase A.
See `notebooks/phase_a_sanity_colab.ipynb` for a self-contained Colab version.

## 2. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Run the first real experiment: the self-stitch sanity check

```bash
python scripts/run_phase_a_sanity.py --model wav2vec2-base --num_examples 64 --stitch_kind linear
```

This will:
1. Download `facebook/wav2vec2-base` and a small LibriSpeech dev sample (first run only — cached after).
2. Extract hidden states at every encoder layer.
3. For every (source layer, target layer) pair, fit a stitch layer with
   Direct-Matching initialization + LFM refinement.
4. Check whether every layer is most similar to itself.
5. Print a pass/fail report.

### What "pass" looks like
```
=== Self-Stitch Sanity Check (wav2vec2-base, linear stitcher) ===
Accuracy: 100.00%
No failures. Pipeline passes the sanity check.
```

### What to do if it fails
- Increase `--lfm_epochs` (undertrained stitchers can look worse than they are).
- Check `--lfm_lr` (too high can destabilize the linear solve's warm start).
- Confirm the closed-form Direct Matching init is actually being applied
  (only wired up for `--stitch_kind linear` in this starter script).
- If it still fails after tuning, that's a real, reportable finding —
  it means audio layers may not have the same "self-similarity" structure
  vision layers do. Don't force it to pass; investigate why.

## 4. Once sanity checks pass: next scripts to write

1. **Cross-model correspondence check** — repeat the same loop but load two
   *independently initialized/fine-tuned* wav2vec2 checkpoints (or two seeds
   of the same training run, if you fine-tune your own) as source and target.
   Reuse `sanity_checks.cross_model_correspondence_check`.
2. **TLT and FFM training loops** — add supervised task loss (e.g. phoneme
   classification on a SUPERB task) and the two-stage FFM→TLT recipe, using
   `stitching.task_loss` and `stitching.ffm_loss` plus `audio_models.head_from_layer`.
3. **OOD diagnostic** — use `ood_detector.py` to check whether TLT-trained
   stitchers produce separable (OOD) activations the way vision TLT does.
4. **Self-stitch control** (Phase B) — once you move to cross-model
   (wav2vec2 ↔ HuBERT) stitching, always run the matching self-stitch
   baselines alongside it.

## 5. File guide

| File | Purpose |
|---|---|
| `src/audio_models.py` | Loads wav2vec2/HuBERT/WavLM via HuggingFace; exposes per-layer hidden states and a "remaining layers" head for FFM. |
| `src/stitching.py` | `StitchLayer`, `TemporalResample`, `StitchedModel`, and the LFM/FFM/TLT loss functions + Direct Matching closed-form init. |
| `src/sanity_checks.py` | Self-stitch and cross-model correspondence checks + report printing. |
| `src/ood_detector.py` | Energy-based OOD detector (Liu et al. 2020 style) for the OOD diagnostic. |
| `scripts/run_phase_a_sanity.py` | End-to-end runnable script for the first sanity check. |
| `notebooks/phase_a_sanity_colab.ipynb` | Self-contained Colab version of the same check. |
| `tests/test_offline_logic.py` | Numpy-only unit tests, no GPU/torch needed. |
| `CLAUDE.md` | Project context auto-loaded by Claude Code at the start of a session. |

## 6. Known gaps to fill in yourself

- **Dataset**: swap the tiny HuggingFace dummy dataset for a real SUPERB
  task subset once you're past the smoke test.
- **Temporal resampling** isn't exercised yet because same-model self-stitch
  has matching frame rates. You'll need it as soon as you stitch across
  models with different hop sizes.
- **`head_from_layer`** assumes a standard `Wav2Vec2Model`-style encoder.
  Verify this matches the actual module structure for each new checkpoint.
- **Direct Matching init** is currently only wired for linear stitch layers.
