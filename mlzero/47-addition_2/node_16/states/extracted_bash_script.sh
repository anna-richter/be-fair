#!/bin/bash
set -e

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/47-addition_2/node_16/conda_env"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/47-addition_2/node_16/generated_code.py"

# 1. Create conda env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=3.11
fi

# 2. Activate env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install uv

# 4. Install requirements with uv (prerelease allowed, both files)
uv pip install --prerelease=allow -r "$REQ_MM" -r "$REQ_COMMON"

# 5. Run the Python script
python "$PY_SCRIPT"