#!/bin/bash

#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1 
#SBATCH --job-name=%a_add2_prompt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=gpu
#SBATCH --account=sc-users
#SBATCH --output=%a_add2_prompt.o%j 
#SBATCH --error=%a_add2_prompt.e%j


source /opt/miniforge/etc/profile.d/conda.sh
conda activate be_fair_env_cuda
export OPENAI_API_KEY=$OPENAI_API_KEY

srun aide agent.expose_prediction=true \
	exp_name="addition_2" \
	data_dir="MyData" \
 	goal="I'm a dermatologist. My colleagues and I have curated a skin lesion dataset over the past several years. I need a model trained on this data to classify lesions as malignant or benign. The trained model should be saved, and I need a function that takes a folder path of new images and returns a malignancy probability (0–1) for each image. Be fair towards skin tone. The AUROC gap between light and dark skin should be minimal, without compromising overall model performance." \
 	eval="Area under the receiver operator curve (AUROC)"
