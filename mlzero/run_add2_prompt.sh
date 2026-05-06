#!/bin/bash

#SBATCH --time=47:59:59
#SBATCH --gres=gpu:1 
#SBATCH --job-name=%a_addition_2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=gpu
#SBATCH --account=sc-users
#SBATCH --output=%a_addition_2.o%j 
#SBATCH --error=%a_addition_2.e%j


export PIP_CACHE_DIR="/sc-scratch/sc-scratch-ikim-guidlight/pip-cache"
export TMPDIR="/sc-scratch/sc-scratch-ikim-guidlight/tmp"
source /opt/miniforge/etc/profile.d/conda.sh
conda activate mlzero_env
export OPENAI_API_KEY=$OPENAI_API_KEY

srun mlzero \
	--input /sc-projects/sc-proj-ikim-guidlight/be-fair/mlzero/addition_2_data \
	--config /sc-projects/sc-proj-ikim-guidlight/be-fair/mlzero/conf.yaml \
	--max-iterations 20 \
	--output /sc-projects/sc-proj-ikim-guidlight/be-fair/mlzero/${SLURM_ARRAY_TASK_ID}-addition_2 \
	--continuous_improvement
