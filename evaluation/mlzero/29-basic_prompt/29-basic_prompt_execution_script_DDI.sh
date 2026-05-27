#!/bin/bash
set -e

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/29-basic_prompt/node_13/conda_env"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_ML="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/machine learning/requirements.txt"
# start change
# PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/29-basic_prompt/node_13/generated_code.py"  # original
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/29-basic_prompt/29-basic_prompt_generated_code_DDI.py"
# end change

# 1. Create conda environment with Python 3.11
if [ ! -d "$ENV_DIR" ]; then
    conda create --yes --prefix "$ENV_DIR" python=3.11
fi

# 2. Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install uv

# 4. Install requirements via uv
uv pip install --prerelease=allow -r "$REQ_ML" -r "$REQ_COMMON"

# 5. Install additional required packages for the script
pip install torch torchvision timm pandas scikit-learn tqdm pillow

# 6. Run the Python script
python "$PY_SCRIPT"