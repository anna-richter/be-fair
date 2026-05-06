#!/usr/bin/env bash
set -e

# Paths
ENV_DIR="/home/anri21/be-fair/mlzero/33-basic_prompt/node_8/conda_env"
PYTHON_VERSION="3.11"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
PY_SCRIPT="/home/anri21/be-fair/mlzero/33-basic_prompt/node_8/generated_code.py"

# 1. Create conda env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=$PYTHON_VERSION
fi

# 2. Activate env
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"

# 3. Install uv if not present
if ! python -m pip show uv &>/dev/null; then
    python -m pip install uv
fi

# 4. Install requirements with uv (prerelease allowed)
uv pip install --prerelease=allow -r "$REQ_MM" -r "$REQ_COMMON"

# 5. Run the Python script
python "$PY_SCRIPT"