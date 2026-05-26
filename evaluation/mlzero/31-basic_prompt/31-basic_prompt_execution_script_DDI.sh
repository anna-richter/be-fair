#!/bin/bash
set -e

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/31-basic_prompt/node_3/conda_env"
PYTHON_VERSION="3.11"
REQUIREMENTS_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_MM="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
# start change
# PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/31-basic_prompt/node_3/generated_code.py"  # original
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/31-basic_prompt/31-basic_prompt_generated_code_DDI.py"
# end change

# 1. Create conda env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=$PYTHON_VERSION
fi

# 2. Activate env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
pip install uv

# 4. Install requirements with uv (do not upgrade/reinstall autogluon.multimodal if present)
uv pip install --prerelease=allow -r "$REQUIREMENTS_MM" -r "$REQUIREMENTS_COMMON"

# 5. Run the Python script
python "$PY_SCRIPT"