#!/bin/bash
#SBATCH --job-name=phaseB-audiomae-l1vl2
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%j.out

PI_NAME=pi_cics-ur_umass_edu
USERNAME=ijamal_umass_edu
PROJECT_DIR=/work/${PI_NAME}/${USERNAME}-audio-stitch

module load conda/latest
conda activate ${PROJECT_DIR}/envs/audio-stitch
export HF_HOME=${PROJECT_DIR}/hf_cache
cd ${PROJECT_DIR}

python -u scripts/diagnose_audiomae_layer1_vs_layer2.py --num_examples 64

EXIT_CODE=$?
echo "Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
