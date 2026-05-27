#!/bin/bash
set -e

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/55-addition_3/node_7/conda_env"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_ML="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/machine learning/requirements.txt"
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/55-addition_3/node_7/generated_code.py"

# 1. Create conda environment in the specified folder with Python 3.11
conda create -y -p "$ENV_DIR" python=3.11

# 2. Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv via pip
pip install uv

# 4. Install required packages using uv, only as specified in requirements, allow prerelease
uv pip install --prerelease=allow -r "$REQ_ML" -r "$REQ_COMMON"

# 5. Install any additional packages needed for the script
pip install pandas scikit-learn lightgbm torch torchvision pillow tqdm

# 6. Run the Python script
python "$PY_SCRIPT"