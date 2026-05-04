#!/bin/bash
set -e

ENV_DIR="/home/anri21/be-fair/mlzero/35-addition_1/node_2/conda_env"
PYTHON_VERSION="3.11"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_TABULAR="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"
PY_SCRIPT="/home/anri21/be-fair/mlzero/35-addition_1/node_2/generated_code.py"

conda create --yes --prefix "$ENV_DIR" python=$PYTHON_VERSION
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

pip install uv

uv pip install --prerelease=allow -r "$REQ_TABULAR" -r "$REQ_COMMON"

python "$PY_SCRIPT"