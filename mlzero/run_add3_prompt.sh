#!/bin/bash

#SBATCH --time=47:59:59
#SBATCH --gres=gpu:1 
#SBATCH --job-name=%a_addition_3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=gpu
#SBATCH --account=sc-users
#SBATCH --output=%a_addition_3.o%j 
#SBATCH --error=%a_addition_3.e%j


source /opt/miniforge/etc/profile.d/conda.sh
conda activate mlzero_env
export OPENAI_API_KEY=$OPENAI_API_KEY

srun mlzero \
	--input /home/anri21/be-fair/mlzero/addition_3/data \
	--output /home/anri21/be-fair/mlzero/addition_3 \
	--provider openai \
 	--verbosity 4
