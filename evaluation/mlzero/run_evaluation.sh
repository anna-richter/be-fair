#!/bin/bash

#SBATCH --time=47:59:59
#SBATCH --gres=gpu:1 
#SBATCH --job-name=%a_evaluation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=gpu
#SBATCH --account=sc-users
#SBATCH --output=%a_evaluation.o%j 
#SBATCH --error=%a_evaluation.e%j


cd ${SLURM_ARRAY_TASK_ID}-*
export DIR_NAME=$(basename "$PWD")
srun bash ${DIR_NAME}_execution_script_DDI.sh
cd ..
