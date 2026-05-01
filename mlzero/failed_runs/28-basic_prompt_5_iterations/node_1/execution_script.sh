#!/bin/bash
set -e

ENV_DIR="/home/anri21/be-fair/mlzero/28-basic_prompt/node_1/conda_env"
PYTHON_VERSION="3.11"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_ML="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/machine learning/requirements.txt"
PY_SCRIPT="/home/anri21/be-fair/mlzero/28-basic_prompt/node_1/generated_code.py"

# 1. Create conda environment if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=$PYTHON_VERSION
fi

# 2. Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv and required packages
pip install uv

uv pip install --strict --prerelease=allow \
    -r "$REQ_ML" \
    -r "$REQ_COMMON"

# 4. Install any additional packages needed for the script
pip install pandas numpy scikit-learn torch torchvision pillow joblib

# 5. Run the Python script
python "$PY_SCRIPT"