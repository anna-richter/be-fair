#!/bin/bash
set -e

# Paths
ENV_DIR="/home/anri21/be-fair/mlzero/34-basic_prompt/node_0/conda_env"
PYTHON_VERSION="3.11"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.12/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
PY_SCRIPT="/home/anri21/be-fair/mlzero/34-basic_prompt/node_0/generated_code.py"

# 1. Create conda env in target folder if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create --prefix "$ENV_DIR" python=$PYTHON_VERSION -y
fi

# 2. Activate the environment
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv (fast pip replacement)
pip install uv

# 4. Install required packages (exact, with dependencies, no upgrades)
uv pip install --strict --prerelease=allow \
    -r "$REQ_MM" \
    -r "$REQ_COMMON"

# 5. Run the Python script
python "$PY_SCRIPT"