#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/35-addition_1/node_4/conda_env"
PYTHON_VERSION="3.11"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/35-addition_1/node_4/generated_code.py"

# 1. Create conda env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create --prefix "$ENV_DIR" python=$PYTHON_VERSION -y
fi

# 2. Activate env
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install --upgrade pip
pip install uv

# 4. Install requirements with uv (prerelease allowed, both files)
uv pip install --prerelease=allow -r "$REQ_MM" -r "$REQ_COMMON"

# 5. Run the Python script
python "$SCRIPT"