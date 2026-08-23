#!/bin/bash
#SBATCH --job-name=phaseA-wavlm-crossmodel
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

PI_NAME=pi_cics-ur_umass_edu
USERNAME=ijamal_umass_edu
PROJECT_DIR=/work/${PI_NAME}/${USERNAME}-audio-stitch

module load conda/latest
conda activate ${PROJECT_DIR}/envs/audio-stitch
export HF_HOME=${PROJECT_DIR}/hf_cache
cd ${PROJECT_DIR}

# Finishing Phase A: wav2vec2-base vs. WavLM (the wav2vec2-vs-HuBERT pair
# was already tested; this fills in the third pairing in the family).
python -u scripts/run_cross_model_check.py \
    --source_model wav2vec2-base \
    --target_model wavlm-base-plus \
    --num_examples 200 \
    --stitch_kind linear \
    --lfm_epochs 300 \
    --lfm_lr 1e-3 \
    --lfm_patience 20 \
    --extract_batch_size 8

EXIT_CODE=$?
echo "Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
