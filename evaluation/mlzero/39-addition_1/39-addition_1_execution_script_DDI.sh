#!/bin/bash
set -e

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/39-addition_1/node_3/conda_env"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
# start change
# PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/39-addition_1/node_3/generated_code.py"  # original
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/39-addition_1/39-addition_1_generated_code_DDI.py"
# end change

# 1. Create conda environment with Python 3.11
conda create --yes --prefix "$ENV_DIR" python=3.11

# 2. Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install uv

# 4. Install required packages from requirements files (only those, with dependencies)
uv pip install --prerelease=allow -r "$REQ_MM" -r "$REQ_COMMON"

# 5. Run the Python script
python "$PY_SCRIPT"