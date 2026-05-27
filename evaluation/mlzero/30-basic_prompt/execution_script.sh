#!/bin/bash
set -euo pipefail

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/30-basic_prompt/node_0/conda_env"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/30-basic_prompt/node_0/generated_code.py"

# 1. Create conda environment in custom folder if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create --yes --prefix "$ENV_DIR" python=3.11
fi

# 2. Activate environment
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv (fast Python package installer)
pip install --upgrade pip
pip install uv

# 4. Install required packages from requirements files using uv
uv pip install -r "$REQ_MM" --prerelease=allow -r "$REQ_COMMON"

# 5. Run the Python script
python "$PY_SCRIPT"