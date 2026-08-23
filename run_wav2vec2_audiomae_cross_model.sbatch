#!/bin/bash
#SBATCH --job-name=phaseB-wav2vec2-audiomae-crossmodel
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out

PI_NAME=pi_cics-ur_umass_edu
USERNAME=ijamal_umass_edu
PROJECT_DIR=/work/${PI_NAME}/${USERNAME}-audio-stitch

module load conda/latest
conda activate ${PROJECT_DIR}/envs/audio-stitch
export HF_HOME=${PROJECT_DIR}/hf_cache
cd ${PROJECT_DIR}

# The check that originally surfaced the choices= bug -- now unblocked.
# Both models have exactly 12 layers; AudioMAE's T=513 (512 patches + CLS)
# vs wav2vec2's T~499 is handled automatically by the existing temporal
# resampling step.
python -u scripts/run_cross_model_check.py \
    --source_model wav2vec2-base \
    --target_model audiomae \
    --num_examples 200 \
    --stitch_kind linear \
    --lfm_epochs 300 \
    --lfm_lr 1e-3 \
    --lfm_patience 20 \
    --extract_batch_size 8

EXIT_CODE=$?
echo "Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
