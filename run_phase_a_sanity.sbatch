#!/bin/bash
#SBATCH --job-name=phaseA-sanity
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# ---- Unity account details ----
PI_NAME=pi_cics-ur_umass_edu
USERNAME=ijamal_umass_edu
PROJECT_DIR=/work/${PI_NAME}/${USERNAME}-audio-stitch
# --------------------------------

# NOTE: --partition=gpu-preempt is required here, not the plain "gpu"
# partition -- A16 GPUs (the hardware we've validated this pipeline
# against) only exist under gpu-preempt and umd-cscdr-gpu. The plain
# "gpu" partition draws from a much wider, older mix of hardware
# (M40, V100, etc.) that this torch/CUDA build does not support.

module load conda/latest
conda activate ${PROJECT_DIR}/envs/audio-stitch

export HF_HOME=${PROJECT_DIR}/hf_cache

cd ${PROJECT_DIR}

python scripts/run_phase_a_sanity.py \
    --model wav2vec2-base \
    --num_examples 200 \
    --stitch_kind linear \
    --lfm_epochs 300 \
    --lfm_lr 1e-3 \
    --lfm_patience 20 \
    --extract_batch_size 8

# Capture the REAL exit code immediately -- don't let a later command
# (even just echo) reset $? before we report it.
EXIT_CODE=$?
echo "Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
