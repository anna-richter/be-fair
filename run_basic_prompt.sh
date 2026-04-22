#!/bin/bash

#SBATCH --time 12:00:00
#SBATCH --gres=shard:1 
#SBATCH --job-name=basic_prompt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --account=sc-users
#SBATCH --output=basic_prompt_1.o%j 
#SBATCH --error=basic_prompt_1.e%j


conda activate be_fair_env
export OPENAI_API_KEY=$OPENAI_API_KEY

srun aide agent.expose_prediction=true \
	exp_name="basic_prompt" \
	data_dir="MyData" \
 	goal="I am a dermatologist, I have collected data about skin lesions with my colleagues over the last years. Now I want to train an AI on this dataset that can differentiate malignant and benign lesions. The idea is that I upload a folder with new images and the model returns the probability between 0 and 1 for each image that the image shows a malignant skin lesion." \
 	eval="Area under the receiver operator curve (AUROC)"
