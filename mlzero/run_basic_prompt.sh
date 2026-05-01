#!/bin/bash

#SBATCH --time=47:59:59
#SBATCH --gres=gpu:1 
#SBATCH --job-name=%a_basic_prompt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=gpu
#SBATCH --account=sc-users
#SBATCH --output=%a_basic_prompt.o%j 
#SBATCH --error=%a_basic_prompt.e%j


source /opt/miniforge/etc/profile.d/conda.sh
conda activate mlzero_env
export OPENAI_API_KEY=$OPENAI_API_KEY

srun mlzero \
	--input /home/anri21/be-fair/mlzero/basic_prompt_data \
	--config /home/anri21/be-fair/mlzero/conf.yaml \
	--max-iterations 20 \
	--output /home/anri21/be-fair/mlzero/${SLURM_ARRAY_TASK_ID}-basic_prompt \
	--verbosity 4 \
	--continuous_improvement