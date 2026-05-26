#!/bin/bash
set -e

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/49-addition_3/node_7/conda_env"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_ML="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/machine learning/requirements.txt"
# start change
# PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/49-addition_3/node_7/generated_code.py"  # original
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/49-addition_3/49-addition_3_generated_code_DDI.py"
# end change

# 1. Create conda env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=3.11
fi

# 2. Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install uv

# 4. Install requirements with uv (prerelease allowed, both files at once)
uv pip install --prerelease=allow -r "$REQ_ML" -r "$REQ_COMMON"

# 5. Install any additional packages needed for the script
pip install pandas numpy scikit-learn lightgbm pillow tqdm torch torchvision timm joblib

# 6. Run the Python script
python "$PY_SCRIPT"