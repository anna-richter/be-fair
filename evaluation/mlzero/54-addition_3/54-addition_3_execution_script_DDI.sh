#!/bin/bash
set -euo pipefail

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/54-addition_3/node_15/conda_env"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_ML="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/machine learning/requirements.txt"
# start change
# PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/54-addition_3/node_15/generated_code.py"  # original
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/54-addition_3/54-addition_3_generated_code_DDI.py"
# end change

# 1. Create conda env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create --yes --prefix "$ENV_DIR" python=3.11
fi

# 2. Activate conda env
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv if not already
if ! python -m uv --version >/dev/null 2>&1; then
    pip install uv
fi

# 4. Install requirements with uv
uv pip install --prerelease=allow -r "$REQ_ML" -r "$REQ_COMMON"

# 5. Install extra packages needed for the script
pip install torch torchvision timm pandas scikit-learn pillow

# 6. Run the Python script
python "$PY_SCRIPT"