#!/usr/bin/env bash
set -euo pipefail

# --- CONFIGURATION ---
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/43-addition_2/node_11/conda_env"
PYTHON_VERSION="3.11"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
# start change
# PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/43-addition_2/node_11/generated_code.py"  # original
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/43-addition_2/43-addition_2_generated_code_DDI.py"
# end change

# --- 1. Create conda environment if not exists ---
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python="$PYTHON_VERSION"
fi

# --- 2. Activate environment ---
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# --- 3. Install uv and requirements ---
pip install --upgrade pip
pip install uv

uv pip install --prerelease=allow -r "$REQ_MM" -r "$REQ_COMMON"

# --- 4. Run the Python script ---
python "$PY_SCRIPT"