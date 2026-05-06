#!/usr/bin/env bash
set -e

# Variables
ENV_DIR="/home/anri21/be-fair/mlzero/29-basic_prompt/node_3/conda_env"
PYTHON_VERSION="3.11"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
PY_SCRIPT="/home/anri21/be-fair/mlzero/29-basic_prompt/node_3/generated_code.py"

# 1. Create conda environment
conda create -y -p "$ENV_DIR" python=$PYTHON_VERSION

# 2. Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install uv

# 4. Install requirements with uv (prerelease allowed, exact packages only)
uv pip install --prerelease=allow -r "$REQ_MM" -r "$REQ_COMMON"

# 5. Run the Python script
python "$PY_SCRIPT"