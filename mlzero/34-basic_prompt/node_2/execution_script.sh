#!/bin/bash
set -e

ENV_DIR="/home/anri21/be-fair/mlzero/34-basic_prompt/node_2/conda_env"
PYTHON_VERSION="3.11"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_TABULAR="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"
PY_SCRIPT="/home/anri21/be-fair/mlzero/34-basic_prompt/node_2/generated_code.py"

# 1. Create conda env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create --prefix "$ENV_DIR" python=$PYTHON_VERSION -y
fi

# 2. Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install uv

# 4. Install required packages (only exact packages and dependencies)
uv pip install --prerelease=allow \
    -r "$REQ_TABULAR" \
    -r "$REQ_COMMON"

# 5. Run the Python script
python "$PY_SCRIPT"