#!/bin/bash
set -e

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/28-basic_prompt/node_14/conda_env"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/28-basic_prompt/node_14/generated_code.py"

# 1. Create conda environment with Python 3.11
conda create -y -p "$ENV_DIR" python=3.11

# 2. Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install uv

# 4. Install requirements via uv (only exact packages and dependencies)
uv pip install -r "$REQ_MM" --prerelease=allow -r "$REQ_COMMON"

# 5. Run the Python script
python "$PY_SCRIPT"