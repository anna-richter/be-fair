#!/bin/bash
set -e

ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/42-addition_2/node_6/conda_env"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_ML="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/machine learning/requirements.txt"
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/42-addition_2/node_6/generated_code.py"

# 1. Create conda env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=3.11
fi

# 2. Activate env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install uv

# 4. Install requirements with uv (prerelease allowed)
uv pip install --prerelease=allow -r "$REQ_ML" -r "$REQ_COMMON"

# 5. Install extra packages needed for the script
pip install pandas scikit-learn lightgbm torch torchvision pillow tqdm joblib

# 6. Run the Python script
python "$PY_SCRIPT"