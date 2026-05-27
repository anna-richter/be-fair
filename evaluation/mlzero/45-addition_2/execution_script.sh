#!/usr/bin/env bash
set -euo pipefail

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/45-addition_2/node_0/conda_env"
REQUIREMENTS_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/45-addition_2/node_0/generated_code.py"

# 1. Create conda env in folder if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=3.11
fi

# 2. Activate env
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv (pip, not conda)
python -m pip install --upgrade pip
python -m pip install uv

# 4. Install requirements with uv (prerelease allowed)
uv pip install --prerelease=allow -r "$REQUIREMENTS_MM" -r "$REQUIREMENTS_COMMON"

# 5. Run the Python script
python "$PY_SCRIPT"