# Audio Stitching — Project Context for Claude Code

## What this project is
Extending "model stitching" (connecting frozen layers of two neural networks
via a small trainable connector) from vision models to audio/speech models.
Two reference papers motivate the method:
1. A reliability warning: naive task-loss-trained stitching can produce
   misleadingly high similarity by creating out-of-distribution shortcut
   representations.
2. A feasibility/fusion result: heterogeneous models CAN be stitched well if
   the connector is trained with Final Feature Matching, then fine-tuned on
   the task — and the result can be genuine complementary knowledge fusion,
   not just added capacity.

Full protocol: see `README.md` and `UNITY_SETUP.md` in this repo.

## Current status
Phase A (pipeline validation on wav2vec2-base) has been smoke-tested
successfully in Google Colab (`notebooks/phase_a_sanity_colab.ipynb`) using
a self-stitch identity check. Compute environment is now Unity HPC
(`unity.rc.umass.edu`), a SLURM cluster. This is the FIRST real run there.

## Environment: Unity HPC specifics
- Account details for this project: username `ijamal_umass_edu`, PI group
  `pi_cics-ur_umass_edu`, project directory
  `/work/pi_cics-ur_umass_edu/ijamal_umass_edu-audio-stitch` (also available
  as `$PROJECT_DIR` once `~/.bashrc` is sourced — see `UNITY_SETUP.md`).
- SLURM scheduler. Partitions: `gpu` (stable, for real batch jobs),
  `gpu-preempt` (fast queue, can be killed after ~2h — use for quick
  interactive debugging only), `gpu-long` (for jobs needing more time).
- **Never run heavy Python/model-loading directly on the login node.**
  Always go through `srun` (interactive) or `sbatch` (batch).
- `$HOME` has a small quota. ALL project files, the conda env, and the
  HuggingFace cache must live under `$PROJECT_DIR`.
- Conda is loaded via `module load conda/latest`, not a bare venv.
- Full step-by-step setup is in `UNITY_SETUP.md` — it already has the real
  account values filled in, follow it as-is.

## Project structure
- `src/audio_models.py` — loads wav2vec2/HuBERT/WavLM via HuggingFace,
  exposes per-layer hidden states and a `head_from_layer()` for FFM targets.
  NOTE: `head_from_layer` branches on `config.do_stable_layer_norm` — this
  was a real bug fix (base models are post-norm, large models are pre-norm
  with a final layer-norm). Don't "simplify" this away.
- `src/stitching.py` — `StitchLayer`, `TemporalResample`, `StitchedModel`,
  the LFM/FFM/TLT loss functions, and the closed-form Direct Matching init.
- `src/sanity_checks.py` — self-stitch and cross-model correspondence
  checks. Pure numpy, no torch dependency (keep it that way).
- `src/ood_detector.py` — energy-based OOD detector for the reliability
  diagnostic (checks whether TLT-trained stitchers produce OOD activations).
- `scripts/run_phase_a_sanity.py` — the runnable entry point for the
  self-stitch check. This is what gets submitted via `sbatch`.
- `run_phase_a_sanity.sbatch` — SLURM batch script. Already has the real
  `PI_NAME`/`USERNAME` values filled in for this project — do not revert
  these to placeholders.
- `tests/test_offline_logic.py` — numpy-only unit tests, runnable anywhere,
  no GPU needed. Run these first if anything in `sanity_checks.py` or the
  Direct Matching math is touched.

## What "done" looks like for this session's task
1. SSH access to Unity confirmed working.
2. Conda env created under `/work/...`, `requirements.txt` installed.
3. Interactive debug run (`srun` + `--num_examples 16`) completes without
   error and prints a sanity-check report.
4. `run_phase_a_sanity.sbatch` edited with real PI/username, submitted via
   `sbatch`, and monitored via `squeue --me` / `tail -f slurm-<jobid>.out`
   to completion.
5. Report back the final accuracy % and whether any layers failed — do not
   just report "it ran," report the actual sanity-check numbers.

## Ground rules
- If Unity requires an interactive 2FA/Duo approval, stop and ask the user
  to approve it — do not attempt to script around it.
- If `python3 -m py_compile` or the offline tests fail after an edit, fix
  before running anything on a GPU node — don't debug on the cluster's time.
- Don't invent PI group names, usernames, or file paths — ask if unknown.
- Keep `src/sanity_checks.py` and `src/ood_detector.py`'s numpy-only pieces
  free of torch imports; they're meant to run without a GPU for fast local
  testing.
