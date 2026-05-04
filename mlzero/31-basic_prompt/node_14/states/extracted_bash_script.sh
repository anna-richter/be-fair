#!/bin/bash
set -e

# Paths
ENV_DIR="/home/anri21/be-fair/mlzero/31-basic_prompt/node_14/conda_env"
PYTHON_VERSION="3.11"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_TABULAR="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"
PY_SCRIPT="/home/anri21/be-fair/mlzero/31-basic_prompt/node_14/generated_code.py"

# 1. Create conda environment if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=$PYTHON_VERSION
fi

# 2. Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install uv

# 4. Install requirements with uv (do not upgrade autogluon.tabular if already correct)
uv pip install --prerelease=allow -r "$REQ_TABULAR" -r "$REQ_COMMON"

# 5. Install additional packages needed by the script
pip install pandas scikit-learn

# 6. Run the Python script
python "$PY_SCRIPT"