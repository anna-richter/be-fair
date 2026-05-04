#!/bin/bash
set -e

ENV_DIR="/home/anri21/be-fair/mlzero/37-addition_1/node_10/conda_env"
PYTHON_VERSION="3.11"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_ML="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/machine learning/requirements.txt"
PY_SCRIPT="/home/anri21/be-fair/mlzero/37-addition_1/node_10/generated_code.py"

# 1. Create conda env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python="$PYTHON_VERSION"
fi

# 2. Activate conda env
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install --no-cache-dir uv

# 4. Install requirements with uv, allow prerelease
uv pip install --prerelease=allow -r "$REQ_ML" -r "$REQ_COMMON"

# 5. Install extra packages needed by the script
pip install pandas scikit-learn

# 6. Run the Python script
python "$PY_SCRIPT"