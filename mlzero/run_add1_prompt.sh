#!/bin/bash

#SBATCH --time=47:59:59
#SBATCH --gres=gpu:1 
#SBATCH --job-name=%a_add1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=gpu
#SBATCH --account=sc-users
#SBATCH --output=%a_add1.o%j 
#SBATCH --error=%a_add1.e%j


export PIP_CACHE_DIR="/sc-scratch/sc-scratch-ikim-guidlight/pip-cache"
export TMPDIR="/sc-scratch/sc-scratch-ikim-guidlight/tmp"
export UV_CACHE_DIR="/sc-scratch/sc-scratch-ikim-guidlight/uv-cache"
source /opt/miniforge/etc/profile.d/conda.sh
conda activate mlzero_env
export OPENAI_API_KEY=$OPENAI_API_KEY

srun mlzero \
	--input /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data \
	--config /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/conf.yaml \
	--max-iterations 20 \
	--output /sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/${SLURM_ARRAY_TASK_ID}-addition_1 \
	--continuous_improvement \
	--remove-iteration-folders
