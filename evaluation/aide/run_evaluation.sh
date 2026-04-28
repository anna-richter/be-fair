#!/bin/bash

#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1 
#SBATCH --job-name=%a_evaluation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=gpu
#SBATCH --account=sc-users
#SBATCH --output=%a_evaluation.o%j 
#SBATCH --error=%a_evaluation.e%j


source /opt/miniforge/etc/profile.d/conda.sh
conda activate be_fair_env_cuda

cd ${SLURM_ARRAY_TASK_ID}-*
export DIR_NAME=$(basename "$PWD")
srun python ${DIR_NAME}_best_solution_DDI.py
mv DDI_predictions.csv ${DIR_NAME}_DDI_predictions.csv
cd ..