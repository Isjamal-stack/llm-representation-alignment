#!/bin/bash
#SBATCH --job-name=phaseA-crossmodel-tlt-ffm
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

python -u scripts/run_tlt_ffm_comparison.py \
    --source_model wav2vec2-base \
    --target_model hubert-base \
    --layers 3 6 9 \
    --num_examples 73 \
    --label_column chapter_id \
    --probe_epochs 200 \
    --tlt_epochs 200 \
    --ffm_epochs 200 \
    --lfm_epochs 200 \
    --lr 1e-3 \
    --patience 20 \
    --eval_every 10

EXIT_CODE=$?
echo "Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
