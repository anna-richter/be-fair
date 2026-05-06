#!/bin/bash

#SBATCH --time=47:59:59
#SBATCH --gres=gpu:1 
#SBATCH --job-name=%a_addition_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=gpu
#SBATCH --account=sc-users
#SBATCH --output=%a_addition_1.o%j 
#SBATCH --error=%a_addition_1.e%j


source /opt/miniforge/etc/profile.d/conda.sh
conda activate mlzero_env
export OPENAI_API_KEY=$OPENAI_API_KEY

srun mlzero \
	--input /sc-projects/sc-proj-ikim-guidlight/be-fair/mlzero/addition_1_data \
	--config /sc-projects/sc-proj-ikim-guidlight/be-fair/mlzero/conf.yaml \
	--max-iterations 20 \
	--output /sc-projects/sc-proj-ikim-guidlight/be-fair/mlzero/${SLURM_ARRAY_TASK_ID}-addition_1 \
	--verbosity 4 \
	--continuous_improvement
