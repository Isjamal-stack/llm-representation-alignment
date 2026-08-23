# Running the Phase A Starter Code on Unity

Unity-specific setup, replacing the generic "find a GPU machine" instructions
in the main README.

**Account details for this project (already filled in below):**
- Username: `ijamal_umass_edu`
- PI group: `pi_cics-ur_umass_edu`
- Project directory: `/work/pi_cics-ur_umass_edu/ijamal_umass_edu-audio-stitch`

## 1. Get on the cluster

```bash
ssh ijamal_umass_edu@unity.rc.umass.edu
```

Prefer a browser? Use **Unity OnDemand** (`ood.unity.rc.umass.edu`) →
Interactive Apps → JupyterLab, and pick a GPU partition there instead of
following the sbatch steps below.

## 2. Storage: do NOT use $HOME for the conda env or datasets

`$HOME` has a small quota. Use the group's `/work` directory instead. Set
this once so every later command can just use `$PROJECT_DIR`:

```bash
echo 'export PROJECT_DIR=/work/pi_cics-ur_umass_edu/ijamal_umass_edu-audio-stitch' >> ~/.bashrc
source ~/.bashrc

mkdir -p $PROJECT_DIR
cd $PROJECT_DIR
```

## 3. Set up the conda environment

```bash
module load conda/latest

conda create --prefix $PROJECT_DIR/envs/audio-stitch python=3.11 -y
conda activate $PROJECT_DIR/envs/audio-stitch

pip install -r requirements.txt
```

Add to `~/.bashrc` (in addition to the `PROJECT_DIR` line from Step 2):
```bash
export HF_HOME=$PROJECT_DIR/hf_cache
export CONDA_ENVS_PATH=$PROJECT_DIR/envs
export CONDA_PKGS_DIRS=$PROJECT_DIR/conda_pkgs
```
then `source ~/.bashrc`.

## 4. Get the code onto Unity

If you haven't already (e.g. via `scp` of the zip file), transfer or clone
the project into `$PROJECT_DIR` now — this file assumes it's already there.

## 5. Quick interactive debug run (do this before batch jobs)

```bash
srun --partition=gpu-preempt --gpus=1 --mem=32G --cpus-per-task=4 --time=01:00:00 --pty bash
```

Then, on the allocated node:
```bash
cd $PROJECT_DIR
module load conda/latest
conda activate $PROJECT_DIR/envs/audio-stitch
python scripts/run_phase_a_sanity.py --model wav2vec2-base --num_examples 16
```

## 6. Batch job for the real run

The `PI_NAME` and `USERNAME` values in `run_phase_a_sanity.sbatch` are
already filled in with the real account details above — no editing needed,
just submit:

```bash
sbatch run_phase_a_sanity.sbatch
```

Monitor with:
```bash
squeue --me
tail -f slurm-<jobid>.out
```

Check your group's overall usage/quota:
```bash
unity-slurm-account-usage
```

## 7. Scaling up to Phase B / C

- `gpu-long` if a job needs to run longer than the standard time limit.
- `gpu-preempt` is fine for short exploratory sweeps but assume it can be killed.
- For larger models (CLAP, Whisper) and from-scratch ECG/PPG encoder training,
  you'll likely want a GPU with more VRAM — check Unity's node feature list
  and specify via `--constraint` or `--gpus` in your sbatch script.
